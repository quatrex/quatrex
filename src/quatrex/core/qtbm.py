# Copyright (c) 2025-2026 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.kernels import inplace
from qttools.kernels.linalg.kron import kron_matmul
from qttools.utils.inplace_utils import (
    compute_update_indices_dense,
    compute_update_indices_sparse,
)
from qttools.utils.mpi_utils import get_local_slice
from qttools.wave_function_solver import (
    MUMPS,
    SuperLU,
    WFSolver,
    cuDSS,
    preferred_matrix_type,
)
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.constants import e, h
from quatrex.core.quatrex_config import QuatrexConfig, SolverConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.device import Device
from quatrex.grid import get_electron_energies, monkhorst_pack


def allocate_system_matrix(
    hamiltonians: dict, overlap_matrices: dict, contacts: list
) -> sparse.csr_matrix:
    """Allocates the system matrix.

    Parameters
    ----------
    hamiltonians : dict
        Dictionary of Hamiltonian matrices for each hopping direction.
    overlap_matrices : dict
        Dictionary of Overlap matrices for each hopping direction.
    contacts : list


    Returns
    -------
    system_matrix : sparse.csr_matrix
        The allocated system matrix.

    """

    hamiltonians_host = {}
    overlap_matrices_host = {}

    for r, h_r in hamiltonians.items():
        hamiltonians_host[r] = h_r.get() if hasattr(h_r, "get") else h_r
    for r, s_r in overlap_matrices.items():
        overlap_matrices_host[r] = s_r.get() if hasattr(s_r, "get") else s_r

    size = hamiltonians_host[(0, 0, 0)].shape[0]

    indices_host = []
    indptr_host = np.zeros(size + 1, dtype=np.int64)

    # Build column indices and indptr
    for row in range(size):

        # Compute the union of all column indices affecting this row
        row_union = np.array([], dtype=np.int64)

        # Add Hamiltonian contributions
        for r, h_r in hamiltonians_host.items():
            row_start = h_r.indptr[row]
            row_end = h_r.indptr[row + 1]
            row_union = np.union1d(row_union, h_r.indices[row_start:row_end])

        # Add overlap contributions
        for r, s_r in overlap_matrices_host.items():
            row_start = s_r.indptr[row]
            row_end = s_r.indptr[row + 1]
            row_union = np.union1d(row_union, s_r.indices[row_start:row_end])

        # Add boundary self-energy contributions. This assumes that
        # the contact self-energies are dense.
        for contact in contacts:
            if row in contact.orbital_indices:
                row_union = np.union1d(row_union, contact.orbital_indices)

        # Store the column indices for this row
        indices_host.append(row_union)
        indptr_host[row + 1] = indptr_host[row] + len(row_union)

    # Allocate data and indices arrays
    indices_device = xp.asarray(np.concatenate(indices_host))
    indptr_device = xp.asarray(indptr_host)
    data = xp.zeros_like(indices_device, dtype=xp.complex128)

    system_matrix = sparse.csr_matrix(
        (data, indices_device, indptr_device),
        shape=(size, size),
        dtype=xp.complex128,
    )

    # Check if SM has canonical format
    if not system_matrix.has_canonical_format:
        raise ValueError("System matrix is not in canonical format after allocation.")

    return system_matrix


@dataclass
class Observables:
    """Container for transport observables from QTBM calculations.

    Attributes
    ----------
    electron_ldos : NDArray, optional
        Orbital-resolved local density of states (LDOS) for electrons
        with shape (n_atoms, n_energies).
    contact_currents : dict, optional
        Contact current values for each contact pair.
    transmissions : dict, optional
        Transmission coefficients between contact pairs.

    """

    electron_ldos: NDArray = None
    contact_currents: dict = field(default_factory=dict)
    transmissions: dict = field(default_factory=dict)


class QTBM:
    """Quantum Transmitting Boundary Method solver.

    Parameters
    ----------
    device : Device
        The quantum device object containing Hamiltonian, atomic
        structure, and attached contacts.
    quatrex_config : QuatrexConfig
        Configuration object containing calculation parameters, energy
        grid, and numerical settings.
    compute_config : ComputeConfig, optional
        Computational configuration specifying solver options and
        parallelization parameters. If None, default settings are used.

    Attributes
    ----------
    device : Device
        Reference to the device object.
    num_contacts : int
        Number of contacts attached to the device.
    k_grid : tuple
        k-point for the calculation.
    observables : Observables
        Container for computed transport observables including
        transmission matrices, density of states, and current
        distributions.
    electron_energies : NDArray
        Full energy grid for the calculation.
    local_energies : NDArray
        Local portion of energy grid for MPI parallelization.

    """

    def __init__(
        self,
        device: Device,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes the QTBM solver."""

        self.device = device
        self.num_orbitals = device.hamiltonians[0, 0, 0].shape[0]
        self.num_contacts = len(device.contacts)

        self.quatrex_config = quatrex_config
        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        kpoint_grid = quatrex_config.device.kpoint_grid
        if self.device.gamma_only and kpoint_grid != (1, 1, 1):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but more than one k-point is configured."
            )

        # Generate the Monkhorst-Pack k-point grid.
        self.kpoints = monkhorst_pack(kpoint_grid)
        # Shift the k-points.
        self.kpoints += np.array(quatrex_config.device.kpoint_shift)
        self.num_kpoints = self.kpoints.shape[0]

        self.max_batch_size = self.quatrex_config.qtbm.max_batch_size

        self.observables = Observables()

        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc
        self.block_sections = quatrex_config.electron.obc.block_sections

        # Get the electron energies.
        self.electron_energies = get_electron_energies(quatrex_config)

        # Get the local slice of the electron energies
        self.local_energies = get_local_slice(self.electron_energies)

        # Look for all the combinations of contacts
        self.num_transmissions = int((self.num_contacts**2 - self.num_contacts))

        for contact_in in self.device.contacts:
            for contact_out in self.device.contacts:
                if contact_in == contact_out:
                    continue

                # Initialize the observables
                self.observables.transmissions[contact_in, contact_out] = xp.zeros(
                    (self.num_kpoints, self.local_energies.shape[0]),
                    dtype=xp.float64,
                )

        self.observables.electron_ldos = xp.zeros(
            (
                self.num_kpoints,
                self.num_contacts,
                self.num_orbitals,
                self.local_energies.shape[0],
            ),
            dtype=xp.float64,
        )

        self.solver = self._configure_solver(quatrex_config.electron.solver)
        self.matrix_type = preferred_matrix_type[
            quatrex_config.electron.solver.direct_solver
        ]


    def _configure_solver(self, solver_config: SolverConfig) -> WFSolver:
        """Configures the wavefunction solver based on the config.

        Parameters
        ----------
        solver_config : SolverConfig
            The solver configuration containing solver type and options.

        Returns
        -------
        WFSolver
            The configured wavefunction solver instance.

        """
        if solver_config.direct_solver == "mumps":
            return MUMPS()
        if solver_config.direct_solver == "superlu":
            return SuperLU()
        if solver_config.direct_solver == "cudss":
            return cuDSS()

        raise ValueError(f"Unknown solver: {solver_config.direct_solver}")

    def _compute_observables(
        self,
        phi: NDArray,
        injection_segments: dict,
        local_energy_index: int,
        global_energy_index: int,
        sigma_obc_per_contact: dict,
        phi_surface_per_contact: dict,
        bloch_per_contact: dict,
        system_matrix: sparse.spmatrix,
        overlap_matrices: dict,
        k_idx: int,
    ):
        """Computes transport observables.

        Calculates transmission coefficients, density of states, and
        current distributions from the QTBM wavefunctions. This method
        processes the solution at a single energy point and updates the
        observable arrays.

        Parameters
        ----------
        phi : NDArray
            Wavefunction solution matrix. Each column represents a
            wavefunction for a specific injection mode.
        injection_segments : dict
            Dictionary of slices for each
            contact where each slice corresponds to the contact's injection modes.
        local_energy_index : int
            Energy index in the local energy array.
        global_energy_index : int
            Energy index in the global energy array for storing results.
        sigma_obc_per_contact : dict
            Self-energy matrices for each contact, used for transmission
            calculations.
        phi_surface_per_contact : dict
           Surface wavefunctions for each contact.
        bloch_per_contact : dict
            Bloch transmission matrices for each contact.
        system_matrix : sparse.spmatrix
            The system matrix used in the QTBM calculation.
            $E*S - H + \Sigma_{obc}$
        overlap_matrices : dict
            Overlap matrices for each hopping direction.
        k_idx : int
            Index of the current k-point being processed.

        """

        if phi.size == 0:
            return

        # Input manipulation is done to be able to
        # process one energy at a time
        injection_segments = {
            key[0]: value
            for key, value in injection_segments.items()
            if local_energy_index == key[1]
        }
        sigma_obc_per_contact = {
            contact: {
                key: value[local_energy_index] for key, value in sigma_obcs.items()
            }
            for contact, sigma_obcs in sigma_obc_per_contact.items()
        }
        phi_surface_per_contact = {
            contact: value[local_energy_index]
            for contact, value in phi_surface_per_contact.items()
        }
        bloch_per_contact = {
            contact: {key: value[local_energy_index] for key, value in bloch_k.items()}
            for contact, bloch_k in bloch_per_contact.items()
        }

        contacts = self.device.contacts

        # Compute transmissions for all the possible contact couples
        # for nt in range(self.num_transmissions):
        for (
            contact_in,
            contact_out,
        ), transmission in self.observables.transmissions.items():
            # Get the all the wavefunctions injected from contact 1 and
            # extract the elements inside contact 2

            # Wavefunctions injected from contact_in and evaluated at contact_out
            phi_nt = phi[
                contact_out.orbital_indices,
                injection_segments[contact_in],
            ]

            # Compute the transmission
            if phi_nt.size != 0:

                S_P = xp.zeros_like(phi_nt)

                # This upscales the self-energy if the contact
                # has periodicity in the transverse directions
                ny, nz = contact_out.transverse_repetition_grid
                indices_y = -xp.arange(ny)[:, None] + xp.arange(ny)[None, :]
                indices_z = -xp.arange(nz)[:, None] + xp.arange(nz)[None, :]

                indices_y = xp.kron(indices_y, xp.ones((nz, nz)))
                indices_z = xp.tile(indices_z, (ny, ny))

                for (ky, kz), sigma_obc in sigma_obc_per_contact[contact_out].items():
                    S_P += kron_matmul(
                        xp.exp(-1j * ky * indices_y - 1j * kz * indices_z),
                        sigma_obc,
                        phi_nt,
                    )

                transmission[k_idx, global_energy_index] = xp.trace(
                    -2 * xp.imag(phi_nt.T.conj() @ S_P)
                )

        # Compute the DOS
        # diag(phi^H @ S @ phi)
        # S @ phi needs to consider that
        # the overlap matrices are infinite

        phi_ortho = xp.zeros_like(phi)
        for overlap in overlap_matrices.values():
            phi_ortho += overlap @ phi

        for contact in contacts:
            orbital_indices = contact.orbital_indices
            ny, nz = contact.transverse_repetition_grid

            phi_cont = xp.zeros(
                (orbital_indices.shape[0], phi.shape[1]), dtype=xp.complex128
            )
            phi_cont[:, injection_segments[contact]] = phi_surface_per_contact[contact]

            indices_y = -xp.arange(ny)[:, None] + xp.arange(ny)[None, :]
            indices_z = -xp.arange(nz)[:, None] + xp.arange(nz)[None, :]

            indices_y = xp.kron(indices_y, xp.ones((nz, nz)))
            indices_z = xp.tile(indices_z, (ny, ny))

            # This upscales the block matrix if the contact
            # has periodicity in the transverse directions
            for key, value in bloch_per_contact[contact].items():
                phi_cont += kron_matmul(
                    xp.exp(-1j * key[0] * indices_y - 1j * key[1] * indices_z),
                    value,
                    phi[orbital_indices, :],
                )

            # Add the spill over from the overlap
            for overlap in overlap_matrices.values():
                phi_ortho[orbital_indices, :] += (
                    contact.get_coupling_matrix(overlap) @ phi_cont
                )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = xp.linalg.norm(
                contact.get_coupling_matrix(system_matrix) @ phi_cont
                + system_matrix[orbital_indices, :] @ phi
            )
            if comm.rank == 0:
                print(f"    Spill over error for contact {contact.name[0]}: {error}")

        # Compute the DOS for every injected wavefunction
        for contact_idx, contact in enumerate(contacts):

            injection_segment = injection_segments[contact]

            # Get the wavefunctions of the contact
            phi_c = phi[:, injection_segment]

            # Get the "orthogonalized" wavefunction of the contact
            phi_c_ortho = phi_ortho[:, injection_segment]

            if phi_c.size != 0:
                self.observables.electron_ldos[
                    k_idx, contact_idx, :, global_energy_index
                ] = xp.real(xp.sum(phi_c.conj() * phi_c_ortho, axis=1) / (2 * xp.pi))

    def _compute_current(self):
        """Computes the electron current from the transmission data."""

        # Compute the current from all the k dependent transmissions
        # for nt in range(self.num_transmissions):
        for (
            contact_in,
            contact_out,
        ), transmission in self.observables.transmissions.items():
            Fermi_factor = fermi_dirac(
                self.electron_energies - contact_in.fermi_level,
                self.quatrex_config.electron.temperature,
            ) - fermi_dirac(
                self.electron_energies - contact_out.fermi_level,
                self.quatrex_config.electron.temperature,
            )

            self.observables.contact_currents[contact_in, contact_out] = -(
                xp.sum(
                    xp.trapz(
                        Fermi_factor * transmission,
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.num_kpoints
                * (2 * e / h)
            )

    def _write_outputs(self):
        if comm.rank == 0:

            output_dir = self.quatrex_config.output_dir
            if not os.path.exists(self.quatrex_config.output_dir):
                os.mkdir(self.quatrex_config.output_dir)

            for (
                contact_in,
                contact_out,
            ), transmission in self.observables.transmissions.items():
                label = f"{contact_in.name[0]}{contact_out.name[0]}"
                np.save(
                    f"{output_dir}/transmission_{label}.npy",
                    transmission,
                )

            for (
                contact_in,
                contact_out,
            ), contact_current in self.observables.contact_currents.items():
                label = f"{contact_in.name[0]}{contact_out.name[0]}"
                np.save(
                    f"{output_dir}/current_{label}.npy",
                    contact_current,
                )

            for n in range(self.num_contacts):
                np.save(
                    f"{output_dir}/dos_{self.device.contacts[n].name[0]}.npy",
                    self.observables.electron_ldos[:, n, :, :],
                )

    def run(self) -> None:
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()

        # Allocate indices to update the system matrix in-place
        system_matrix = allocate_system_matrix(
            self.device.hamiltonians, self.device.overlap_matrices, self.device.contacts
        )  # Initialize the system matrix

        hamiltonian_update_indices = {}
        for r, h_r in self.device.hamiltonians.items():
            hamiltonian_update_indices[r] = compute_update_indices_sparse(
                system_matrix, h_r
            )
        overlap_update_indices = {}
        for r, s_r in self.device.overlap_matrices.items():
            overlap_update_indices[r] = compute_update_indices_sparse(
                system_matrix, s_r
            )

        sigma_obc_update_indices = {}
        for contact in self.device.contacts:
            sigma_obc_update_indices[contact] = compute_update_indices_dense(
                system_matrix, contact.orbital_indices
            )

        for k_idx in range(self.num_kpoints):

            if comm.rank == 0:
                print(f"Processing k-point {k_idx+1} of {self.num_kpoints}", flush=True)
            k = self.kpoints[k_idx, :]

            times.append(time.perf_counter())

            # Apply the k-point phase factors to the Hamiltonian and Overlap
            for r, h_r in self.device.hamiltonians.items():
                if r == (0, 0, 0):
                    continue
                h_r.data *= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            for r, s_r in self.device.overlap_matrices.items():
                if r == (0, 0, 0):
                    continue
                s_r.data *= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            times.append(time.perf_counter())

            for batch_start in range(0, len(self.local_energies), self.max_batch_size):

                energy_batch = self.local_energies[
                    batch_start : batch_start + self.max_batch_size
                ]

                if comm.rank == 0:
                    print(
                        f"Processing energies {batch_start} to {batch_start + len(energy_batch) - 1}",
                        flush=True,
                    )

                # append for iteration time
                times.append(time.perf_counter())

                times.append(time.perf_counter())

                injection_per_contact = {}
                phi_surface_per_contact = {}
                bloch_per_contact = {}
                sigma_obc_per_contact = {}

                # Compute the boundary self-energy and the injection vector.
                for contact in self.device.contacts:
                    times.append(time.perf_counter())

                    injection, phi_surface, sigma_obc_K, block_k = (
                        contact.compute_boundary(k * 2 * np.pi, energy_batch)
                    )
                    injection_per_contact[contact] = injection
                    phi_surface_per_contact[contact] = phi_surface

                    sigma_obc_per_contact[contact] = sigma_obc_K
                    bloch_per_contact[contact] = block_k

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time for OBC in contact {contact.name[0]}: {t_solve:.2f} s",
                            flush=True,
                        )

                # Count the number of injected modes per energy
                # needed to know the offset in the lhs/rhs vector
                injection_segments = {}
                injection_count = np.zeros(len(energy_batch), dtype=np.int32)
                for contact in self.device.contacts:
                    modes_per_energy = np.array(
                        [arr.shape[1] for arr in injection_per_contact[contact]]
                    )

                    for i, num_modes in enumerate(modes_per_energy):
                        start = injection_count[i]
                        injection_segments[contact, i] = slice(start, start + num_modes)

                    injection_count += modes_per_energy

                t_solve = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for OBC: {t_solve:.2f} s", flush=True)

                for i, energy in enumerate(energy_batch):

                    times.append(time.perf_counter())

                    # Scale the overlap matrices by the energy
                    for overlap in self.device.overlap_matrices.values():
                        overlap.data *= energy

                    # Set up sytem matrix and rhs for electron solver.
                    injection_tot = xp.zeros(
                        (self.num_orbitals, injection_count[i]),
                        dtype=xp.complex128,
                        order="F",
                    )

                    # Add the injection vector in the contact elements
                    # of the rhs
                    for contact in self.device.contacts:
                        injection_tot[
                            contact.orbital_indices, injection_segments[contact, i]
                        ] = injection_per_contact[contact][i]

                    system_matrix.data[:] = 0

                    # Add the Hamiltonian and overlap contributions
                    for r, h_r in self.device.hamiltonians.items():
                        inplace.isub(
                            system_matrix.data,
                            h_r.data,
                            hamiltonian_update_indices[r],
                        )

                    for r, s_r in self.device.overlap_matrices.items():
                        inplace.iadd(
                            system_matrix.data, s_r.data, overlap_update_indices[r]
                        )

                    # Add the boundary self-energy contributions
                    for contact, sigma_obc in sigma_obc_per_contact.items():
                        for key, value in sigma_obc.items():
                            inplace.isub_obc(
                                system_matrix.data,
                                value[i, :, :],
                                sigma_obc_update_indices[contact],
                                key[0],
                                key[1],
                                contact.transverse_repetition_grid[0],
                                contact.transverse_repetition_grid[1],
                            )

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time to set up system of eq.: {t_solve:.2f} s", flush=True
                        )

                    times.append(time.perf_counter())

                    # Solve for the wavefunction
                    if injection_tot.size != 0:
                        phi = self.solver.solve(system_matrix, injection_tot)

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                    times.append(time.perf_counter())

                    # Get the bare system matrix back, needed for
                    # transmission calculation

                    # Subtract the open boundary conditions
                    for contact, sigma_obc in sigma_obc_per_contact.items():
                        for key, value in sigma_obc.items():
                            inplace.iadd_obc(
                                system_matrix.data,
                                value[i, :, :],
                                sigma_obc_update_indices[contact],
                                key[0],
                                key[1],
                                contact.transverse_repetition_grid[0],
                                contact.transverse_repetition_grid[1],
                            )

                    # Unscale the overlap matrices
                    # to be able to process multiple energies
                    for overlap in self.device.overlap_matrices.values():
                        overlap.data *= 1 / energy

                    if injection_tot.size != 0:
                        # Input
                        self._compute_observables(
                            phi,
                            injection_segments,
                            i,
                            batch_start + i,
                            sigma_obc_per_contact,
                            phi_surface_per_contact,
                            bloch_per_contact,
                            system_matrix,
                            self.device.overlap_matrices,
                            k_idx,
                        )

                    t_observables = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time for computing observables: {t_observables:.2f} s",
                            flush=True,
                        )

                t_iteration = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for iteration: {t_iteration:.2f} s", flush=True)

            # Remove the k-point phase factors from the Hamiltonian and Overlap
            for r, h_r in self.device.hamiltonians.items():
                if r == (0, 0, 0):
                    continue
                h_r.data /= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            for r, s_r in self.device.overlap_matrices.items():
                if r == (0, 0, 0):
                    continue
                s_r.data /= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

        t_iteration = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(f"Time for QTBM: {t_iteration:.2f} s", flush=True)

        # Gather the observables
        comm.Barrier()
        # for nt in range(self.num_transmissions):
        for key, transmission in self.observables.transmissions.items():
            self.observables.transmissions[key] = xp.concatenate(
                comm.allgather(transmission), axis=-1
            )
        self.observables.electron_ldos = xp.concatenate(
            comm.allgather(self.observables.electron_ldos), axis=-1
        )

        self._compute_current()

        self._write_outputs()
