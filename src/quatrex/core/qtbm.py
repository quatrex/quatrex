# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

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
    PARDISO,
    SuperLU,
    Thomas,
    WFSolver,
    cuDSS,
    preferred_matrix_type,
)
from quatrex.core.config import QuatrexConfig, SolverConfig
from quatrex.core.constants import e, h
from quatrex.core.statistics import fermi_dirac
from quatrex.device import Device
from quatrex.grid import get_electron_energies, monkhorst_pack


def get_cpu_memory_gb() -> float:
    """Get current CPU memory usage in GB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return int(line.split()[1]) / 1024 / 1024
    except FileNotFoundError:
        pass
    return 0.0


def allocate_system_matrix(
    hamiltonians: dict, overlap_matrices: dict, contacts: list = []
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


def get_sparse_RHS(
    vector_per_cont: dict,
    injection_segments: dict,
    contacts: list,
    i: int,
    injection_count: dict,
    num_orbitals: int,
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            xp.concatenate(
                list(vector_per_cont[contact][i].flatten() for contact in contacts)
            ),
            (
                xp.concatenate(
                    list(
                        xp.repeat(
                            xp.asarray(contact.orbital_indices),
                            vector_per_cont[contact][i].shape[1],
                        )
                        for contact in contacts
                    )
                ),
                xp.concatenate(
                    list(
                        xp.tile(
                            xp.arange(
                                injection_segments[contact, i].start,
                                injection_segments[contact, i].stop,
                            ),
                            vector_per_cont[contact][i].shape[0],
                        )
                        for contact in contacts
                    )
                ),
            ),
        ),
        shape=(num_orbitals, injection_count[i]),
        dtype=xp.complex128,
    )


def get_sparse_RHS_transpose(
    vector_per_cont: dict,
    injection_segments: dict,
    contacts: list,
    i: int,
    injection_count: dict,
    num_orbitals: int,
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            xp.concatenate(
                list(vector_per_cont[contact][i].flatten() for contact in contacts),
            ),
            (
                xp.concatenate(
                    list(
                        xp.repeat(
                            xp.arange(
                                injection_segments[contact, i].start,
                                injection_segments[contact, i].stop,
                            ),
                            vector_per_cont[contact][i].shape[1],
                        )
                        for contact in contacts
                    )
                ),
                xp.concatenate(
                    list(
                        xp.tile(
                            xp.asarray(contact.orbital_indices),
                            vector_per_cont[contact][i].shape[0],
                        )
                        for contact in contacts
                    )
                ),
            ),
        ),
        shape=(injection_count[i], num_orbitals),
        dtype=xp.complex128,
    )


@dataclass
class Observables:
    """Container for transport observables from QTBM calculations.

    Attributes
    ----------
    electron_ldos : dict, optional
        Orbital-resolved local density of states (LDOS) for each
        contact.
    contact_currents : dict, optional
        Contact current values for each contact pair.
    transmissions : dict, optional
        Transmission coefficients between contact pairs.

    """

    electron_ldos: dict = field(default_factory=dict)
    contact_currents: dict = field(default_factory=dict)
    transmissions: dict = field(default_factory=dict)


class QTBM:
    """Quantum Transmitting Boundary Method solver.

    Parameters
    ----------
    device : Device
        The quantum device object containing Hamiltonian, atomic
        structure, and attached contacts.
    config : QuatrexConfig
        Configuration object containing calculation parameters, energy
        grid, and numerical settings.

    Attributes
    ----------
    device : Device
        Reference to the device object.
    kpoints : tuple
        k-points for the calculation.
    observables : Observables
        Container for computed transport observables including
        transmission matrices, density of states, and current
        distributions.
    electron_energies : NDArray
        Full energy grid for the calculation.
    local_energies : NDArray
        Local portion of energy grid for MPI parallelization.

    """

    def __init__(self, device: Device, config: QuatrexConfig) -> None:
        """Initializes the QTBM solver."""

        self.device = device
        self.num_orbitals = device.hamiltonians[0, 0, 0].shape[0]

        self.config = config

        kpoint_grid = config.device.kpoint_grid
        if self.device.gamma_only and kpoint_grid != (1, 1, 1):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but more than one k-point is configured."
            )

        # Generate the Monkhorst-Pack k-point grid.
        self.kpoints = monkhorst_pack(kpoint_grid, config.device.kpoint_shift)
        self.num_kpoints = self.kpoints.shape[0]

        self.max_batch_size = self.config.qtbm.max_batch_size

        self.observables = Observables()

        # Get the electron energies.
        self.electron_energies = get_electron_energies(config)

        # Get the local slice of the electron energies
        self.local_energies = get_local_slice(self.electron_energies)

        # Look for all the combinations of contacts
        for contact_in in self.device.contacts:
            for contact_out in self.device.contacts:
                if contact_in == contact_out:
                    continue

                # Initialize the observables
                self.observables.transmissions[contact_in, contact_out] = xp.zeros(
                    (self.num_kpoints, self.local_energies.shape[0]),
                    dtype=xp.float64,
                )

        for contact in self.device.contacts:
            self.observables.electron_ldos[contact] = xp.zeros(
                (self.num_kpoints, self.num_orbitals, self.local_energies.shape[0]),
                dtype=xp.float64,
            )

        if self.quatrex_config.qtbm.method == "SplitSolve":
            self.solver = self._configure_solver(quatrex_config.electron.solver)
        else:
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
            if self.quatrex_config.qtbm.method == "SplitSolve":
                raise ValueError(
                    "SplitSolve method is not compatible with MUMPS solver."
                )
            return MUMPS()
        if solver_config.direct_solver == "superlu":
            if self.quatrex_config.qtbm.method == "SplitSolve":
                raise ValueError(
                    "SplitSolve method is not compatible with SuperLU solver."
                )
            return SuperLU()
        if solver_config.direct_solver == "cudss":
            if self.quatrex_config.qtbm.method == "SplitSolve":
                return cuDSS(matrix_type="complex_hermitian_indefinite")
            else:
                return cuDSS(matrix_type="complex_nonsymmetric")
        if solver_config.direct_solver == "pardiso":
            if self.quatrex_config.qtbm.method == "SplitSolve":
                return PARDISO(matrix_type="complex_hermitian_indefinite")
            else:
                return PARDISO(matrix_type="complex_structurally_symmetric")

        if solver_config.direct_solver == "thomas":
            if self.quatrex_config.qtbm.method == "SplitSolve":
                return Thomas(sym=True, view="up")
            else:
                return Thomas(sym=False, view="default")

        raise ValueError(f"Unknown solver: {solver_config.direct_solver}")

    def _compute_transmissions(
        self,
        phi: NDArray,
        injection_segments: dict,
        global_energy_index: int,
        sigma_obc_per_contact: dict,
        reflection_per_contact: dict,
        eig_ref_per_contact: dict,
        phi_inv_ref_per_contact: dict,
        k_idx: int,
    ):
        """Computes transmission coefficients.

        Parameters
        ----------
        phi : NDArray
            Wavefunction solution matrix. Each column represents a
            wavefunction for a specific injection mode.
        injection_segments : dict
            Dictionary of slices for each
            contact where each slice corresponds to the contact's injection modes.
        global_energy_index : int
            Energy index in the global energy array for storing results.
        sigma_obc_per_contact : dict
            Self-energy matrices for each contact, used for transmission
            calculations.
        reflection_per_contact : dict
            Reflection matrices for each contact, used in SplitSolve method.
        eig_ref_per_contact : dict
            Eigenvalues of the reflected wavefunctions for each contact, used in SplitSolve method.
        phi_inv_ref_per_contact : dict
            Inverse of the reflected wavefunctions for each contact, used in SplitSolve method.
        k_idx : int
            Index of the current k-point being processed.

        """
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

                if self.quatrex_config.qtbm.method == "SplitSolve":
                    S_P = (
                        reflection_per_contact[contact_out]
                        @ xp.diag(1 / eig_ref_per_contact[contact_out])
                        @ phi_inv_ref_per_contact[contact_out]
                        @ phi_nt
                    )

                else:
                    # This upscales the self-energy if the contact
                    # has periodicity in the transverse directions
                    ny, nz = contact_out.transverse_repetition_grid
                    indices_y = -xp.arange(ny)[:, None] + xp.arange(ny)[None, :]
                    indices_z = -xp.arange(nz)[:, None] + xp.arange(nz)[None, :]

                    indices_y = xp.kron(indices_y, xp.ones((nz, nz)))
                    indices_z = xp.tile(indices_z, (ny, ny))

                    for (ky, kz), sigma_obc in sigma_obc_per_contact[
                        contact_out
                    ].items():
                        S_P += kron_matmul(
                            xp.exp(-1j * ky * indices_y - 1j * kz * indices_z),
                            sigma_obc,
                            phi_nt,
                        )

                transmission[k_idx, global_energy_index] = xp.trace(
                    -2 * xp.imag(phi_nt.T.conj() @ S_P)
                )

    def _compute_ldos(
        self,
        phi: NDArray,
        injection_segments: dict,
        global_energy_index: int,
        phi_inj_per_contact: dict,
        bloch_per_contact: dict,
        phi_ref_per_contact: dict,
        eig_ref_per_contact: dict,
        phi_inv_ref_per_contact: dict,
        system_matrix: sparse.spmatrix,
        overlap_matrices: dict,
        k_idx: int,
    ):
        """Computes density of states.

        Parameters
        ----------
        phi : NDArray
            Wavefunction solution matrix. Each column represents a
            wavefunction for a specific injection mode.
        injection_segments : dict
            Dictionary of slices for each
            contact where each slice corresponds to the contact's injection modes.
        global_energy_index : int
            Energy index in the global energy array for storing results.
        phi_inj_per_contact : dict
           Surface wavefunctions for each contact.
        bloch_per_contact : dict
            Bloch transmission matrices for each contact.
        phi_ref_per_contact : dict
            Reflected wavefunctions for each contact.
        eig_ref_per_contact : dict
            Eigenvalues of the reflected wavefunctions for each contact.
        phi_inv_ref_per_contact : dict
            Inverse of the reflected wavefunctions for each contact.
        system_matrix : sparse.spmatrix
            The system matrix used in the QTBM calculation.
            $E*S - H + \Sigma_{obc}$
        overlap_matrices : dict
            Overlap matrices for each hopping direction.
        k_idx : int
            Index of the current k-point being processed.

        """
        # Compute the DOS
        # diag(phi^H @ S @ phi)
        # S @ phi needs to consider that
        # the overlap matrices are infinite

        phi_ortho = xp.zeros_like(phi)
        for overlap in overlap_matrices.values():
            phi_ortho += overlap @ phi
            if self.quatrex_config.qtbm.method == "SplitSolve":
                phi_ortho += overlap.T.conj() @ phi
                phi_ortho -= sparse.diags(overlap.diagonal()) @ phi

        for contact in self.device.contacts:
            orbital_indices = contact.orbital_indices

            phi_cont = xp.zeros(
                (orbital_indices.shape[0], phi.shape[1]), dtype=xp.complex128
            )
            phi_cont[:, injection_segments[contact]] = phi_inj_per_contact[contact]

            if self.quatrex_config.qtbm.method == "SplitSolve":
                phi_cont += (
                    phi_ref_per_contact[contact]
                    @ xp.diag(1 / eig_ref_per_contact[contact])
                    @ phi_inv_ref_per_contact[contact]
                    @ phi[orbital_indices, :]
                )

            else:
                ny, nz = contact.transverse_repetition_grid
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
                if self.quatrex_config.qtbm.method == "SplitSolve":
                    phi_ortho[orbital_indices, :] += (
                        contact.get_coupling_matrix(overlap, transpose=True) @ phi_cont
                    )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = (
                contact.get_coupling_matrix(system_matrix) @ phi_cont
                + system_matrix[orbital_indices, :] @ phi
            )
            if self.quatrex_config.qtbm.method == "SplitSolve":
                error += (
                    contact.get_coupling_matrix(system_matrix, transpose=True)
                    @ phi_cont
                    + (system_matrix.T.conj() - sparse.diags(system_matrix.diagonal()))[
                        orbital_indices, :
                    ]
                    @ phi
                )

            error = xp.linalg.norm(error)

            if comm.rank == 0:
                print(f"    Spill over error for contact {contact.name[0]}: {error}")

        # Compute the DOS for every injected wavefunction
        for contact in self.device.contacts:

            injection_segment = injection_segments[contact]

            # Get the wavefunctions of the contact
            phi_c = phi[:, injection_segment]

            # Get the "orthogonalized" wavefunction of the contact
            phi_c_ortho = phi_ortho[:, injection_segment]

            if phi_c.size != 0:
                self.observables.electron_ldos[contact][
                    k_idx, :, global_energy_index
                ] = xp.real(xp.sum(phi_c.conj() * phi_c_ortho, axis=1) / (2 * xp.pi))

    def _compute_observables(
        self,
        phi: NDArray,
        injection_segments: dict,
        local_energy_index: int,
        global_energy_index: int,
        sigma_obc_per_contact: dict,
        reflection_per_contact: dict,
        eig_ref_per_contact: dict,
        phi_inv_ref_per_contact: dict,
        phi_inj_per_contact: dict,
        bloch_per_contact: dict,
        phi_ref_per_contact: dict,
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
        phi_inj_per_contact : dict
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

        # Reshuffling data structures to isolate the current energy.
        # TODO: Perhaps there is a better way to do all of this in some
        # batched approach.
        injection_segments = {
            contact: value
            for (contact, energy_index), value in injection_segments.items()
            if local_energy_index == energy_index
        }
        sigma_obc_per_contact = {
            contact: {
                key: value[local_energy_index] for key, value in sigma_obcs.items()
            }
            for contact, sigma_obcs in sigma_obc_per_contact.items()
        }
        phi_inj_per_contact = {
            contact: value[local_energy_index]
            for contact, value in phi_inj_per_contact.items()
        }
        bloch_per_contact = {
            contact: {key: value[local_energy_index] for key, value in bloch_k.items()}
            for contact, bloch_k in bloch_per_contact.items()
        }

        reflection_per_contact = {
            contact: value[local_energy_index]
            for contact, value in reflection_per_contact.items()
        }

        phi_ref_per_contact = {
            contact: value[local_energy_index]
            for contact, value in phi_ref_per_contact.items()
        }

        eig_ref_per_contact = {
            contact: value[local_energy_index]
            for contact, value in eig_ref_per_contact.items()
        }

        phi_inv_ref_per_contact = {
            contact: value[local_energy_index]
            for contact, value in phi_inv_ref_per_contact.items()
        }

        # Compute transmissions for all the possible contact couples
        self._compute_transmissions(
            phi,
            injection_segments,
            global_energy_index,
            sigma_obc_per_contact,
            reflection_per_contact,
            eig_ref_per_contact,
            phi_inv_ref_per_contact,
            k_idx,
        )

        # Compute the DOS
        self._compute_ldos(
            phi,
            injection_segments,
            global_energy_index,
            phi_inj_per_contact,
            bloch_per_contact,
            phi_ref_per_contact,
            eig_ref_per_contact,
            phi_inv_ref_per_contact,
            system_matrix,
            overlap_matrices,
            k_idx,
        )

    def _compute_current(self):
        """Computes the electron current from the transmission data."""

        # Compute the current from all the k dependent transmissions
        for (
            contact_in,
            contact_out,
        ), transmission in self.observables.transmissions.items():
            prefactor = fermi_dirac(
                self.electron_energies - contact_in.fermi_level,
                self.config.electron.temperature,
            ) - fermi_dirac(
                self.electron_energies - contact_out.fermi_level,
                self.config.electron.temperature,
            )

            self.observables.contact_currents[contact_in, contact_out] = -(
                xp.sum(
                    xp.trapz(
                        prefactor * transmission,
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.num_kpoints
                * (2 * e / h)
            )

    def _write_outputs(self):
        if comm.rank == 0:

            output_dir = self.config.output_dir
            if not os.path.exists(self.config.output_dir):
                os.mkdir(self.config.output_dir)

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

            for contact, ldos in self.observables.electron_ldos.items():
                np.save(
                    f"{output_dir}/dos_{contact.name[0]}.npy",
                    ldos,
                )

    def run(self) -> None:
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()

        # Allocate indices to update the system matrix in-place
        if self.quatrex_config.qtbm.method == "SplitSolve":
            system_matrix = allocate_system_matrix(
                self.device.hamiltonians, self.device.overlap_matrices
            )  # Initialize the system matrix without boundary self energies
        else:
            system_matrix = allocate_system_matrix(
                self.device.hamiltonians,
                self.device.overlap_matrices,
                self.device.contacts,
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

        if self.quatrex_config.qtbm.method != "SplitSolve":
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
                h_r.data *= xp.exp(2j * np.pi * np.dot(k, r))

            for r, s_r in self.device.overlap_matrices.items():
                if r == (0, 0, 0):
                    continue
                s_r.data *= xp.exp(2j * np.pi * np.dot(k, r))

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
                if xp.__name__ == "cupy":
                    xp.cuda.Stream.null.synchronize()

                injection_per_contact = {}
                phi_inj_per_contact = {}
                bloch_per_contact = {}
                sigma_obc_per_contact = {}

                reflection_per_contact = {}
                phi_ref_per_contact = {}
                eig_ref_per_contact = {}
                phi_inv_ref_per_contact = {}

                # Compute the boundary self-energy and the injection vector.
                for contact in self.device.contacts:
                    times.append(time.perf_counter())

                    if self.quatrex_config.qtbm.method == "SplitSolve":
                        (
                            injection_per_contact[contact],
                            phi_inj_per_contact[contact],
                            reflection_per_contact[contact],
                            phi_ref_per_contact[contact],
                            eig_ref_per_contact[contact],
                            phi_inv_ref_per_contact[contact],
                        ) = contact.compute_boundary(
                            k * 2 * np.pi, energy_batch, return_modes_only=True
                        )
                    else:
                        (
                            injection_per_contact[contact],
                            phi_inj_per_contact[contact],
                            sigma_obc_per_contact[contact],
                            bloch_per_contact[contact],
                        ) = contact.compute_boundary(k * 2 * np.pi, energy_batch)

                    if xp.__name__ == "cupy":
                        xp.cuda.Stream.null.synchronize()
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

                if self.quatrex_config.qtbm.method == "SplitSolve":
                    reflection_segments = {}
                    reflection_count = np.zeros(len(energy_batch), dtype=np.int32)
                    for contact in self.device.contacts:
                        modes_per_energy = np.array(
                            [arr.shape[1] for arr in reflection_per_contact[contact]]
                        )
                        for i, num_modes in enumerate(modes_per_energy):
                            start = reflection_count[i]
                            reflection_segments[contact, i] = slice(
                                start, start + num_modes
                            )

                        reflection_count += modes_per_energy

                if xp.__name__ == "cupy":
                    xp.cuda.Stream.null.synchronize()
                t_solve = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for OBC: {t_solve:.2f} s", flush=True)

                for i, energy in enumerate(energy_batch):

                    times.append(time.perf_counter())

                    # Set up sytem matrix and rhs for electron solver.
                    # injection_tot = xp.zeros(
                    #    (self.num_orbitals, injection_count[i]),
                    #    dtype=xp.complex128,
                    #    order="F",
                    # )

                    injection_tot = get_sparse_RHS(
                        injection_per_contact,
                        injection_segments,
                        self.device.contacts,
                        i,
                        injection_count,
                        self.num_orbitals,
                    )

                    # Add the injection vector in the contact elements
                    # of the rhs
                    # for contact in self.device.contacts:
                    #    injection_tot[
                    #        contact.orbital_indices, injection_segments[contact, i]
                    #    ] += injection_per_contact[contact][i]

                    if self.quatrex_config.qtbm.method == "SplitSolve":
                        # reflection_tot = xp.zeros(
                        #    (self.num_orbitals, reflection_count[i]),
                        #    dtype=xp.complex128,
                        #    order="F",
                        # )
                        # phi_inv_tot = xp.zeros(
                        #    (reflection_count[i], self.num_orbitals),
                        #    dtype=xp.complex128,
                        #    order="F",
                        # )

                        # eig_tot = xp.zeros(reflection_count[i], dtype=xp.complex128)

                        # for contact in self.device.contacts:
                        #    reflection_tot[
                        #        contact.orbital_indices, reflection_segments[contact, i]
                        #    ] = reflection_per_contact[contact][i]

                        #    phi_inv_tot[
                        #        reflection_segments[contact, i], contact.orbital_indices
                        #    ] = phi_inv_ref_per_contact[contact][i]

                        #    eig_tot[reflection_segments[contact, i]] = (
                        #        eig_ref_per_contact[contact][i]
                        #    )

                        reflection_tot = get_sparse_RHS(
                            reflection_per_contact,
                            reflection_segments,
                            self.device.contacts,
                            i,
                            reflection_count,
                            self.num_orbitals,
                        )
                        phi_inv_tot = get_sparse_RHS_transpose(
                            phi_inv_ref_per_contact,
                            reflection_segments,
                            self.device.contacts,
                            i,
                            reflection_count,
                            self.num_orbitals,
                        )
                        eig_tot = xp.concatenate(
                            [
                                eig_ref_per_contact[contact][i]
                                for contact in self.device.contacts
                            ]
                        )

                    system_matrix.data[:] = 0

                    # Add the Hamiltonian and overlap contributions
                    for r, h_r in self.device.hamiltonians.items():
                        inplace.isub(
                            system_matrix.data,
                            h_r.data,
                            hamiltonian_update_indices[r],
                        )

                    # Scale the overlap matrices by the energy
                    for overlap in self.device.overlap_matrices.values():
                        overlap.data *= energy

                    for r, s_r in self.device.overlap_matrices.items():
                        inplace.iadd(
                            system_matrix.data, s_r.data, overlap_update_indices[r]
                        )

                    if self.quatrex_config.qtbm.method != "SplitSolve":
                        # Add the boundary self-energy contributions
                        for contact, sigma_obc in sigma_obc_per_contact.items():
                            for k_t, sigma_obc_k in sigma_obc.items():
                                inplace.isub_obc(
                                    system_matrix.data,
                                    sigma_obc_k[i, :, :],
                                    sigma_obc_update_indices[contact],
                                    k_t,
                                    contact.transverse_repetition_grid,
                                )

                    if xp.__name__ == "cupy":
                        xp.cuda.Stream.null.synchronize()
                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time to set up system of eq.: {t_solve:.2f} s", flush=True
                        )

                    times.append(time.perf_counter())
                    injected_mask = xp.arange(injection_tot.shape[1])

                    if self.quatrex_config.qtbm.method == "SplitSolve":
                        reflected_mask = (
                            xp.arange(reflection_tot.shape[1]) + injection_tot.shape[1]
                        )
                        if injection_tot.size != 0:
                            t1 = time.perf_counter()
                            phi = self.solver.solve(
                                system_matrix,
                                sparse.hstack([injection_tot, reflection_tot]).toarray(
                                    order="F"
                                ),
                                reuse_sym_fact=True,
                                reuse_fact=False,
                            )
                            if xp.__name__ == "cupy":
                                xp.cuda.Stream.null.synchronize()
                            t2 = time.perf_counter()
                            if comm.rank == 0:
                                print(
                                    f"Time for solve: {t2 - t1:.2f} s",
                                    flush=True,
                                )
                            if xp.__name__ == "cupy":
                                xp.cuda.Stream.null.synchronize()
                            t1 = time.perf_counter()
                            phi[:, injected_mask] += phi[
                                :, reflected_mask
                            ] @ xp.linalg.solve(
                                xp.diag(eig_tot) - phi_inv_tot @ phi[:, reflected_mask],
                                phi_inv_tot @ phi[:, injected_mask],
                            )
                            if xp.__name__ == "cupy":
                                xp.cuda.Stream.null.synchronize()
                            t2 = time.perf_counter()
                            if comm.rank == 0:
                                print(
                                    f"Time for correction: {t2 - t1:.2f} s", flush=True
                                )
                    else:
                        # Solve for the wavefunction
                        if injection_tot.size != 0:
                            phi = self.solver.solve(
                                system_matrix,
                                injection_tot.toarray(order="F"),
                                reuse_sym_fact=True,
                                reuse_fact=False,
                            )
                    if xp.__name__ == "cupy":
                        xp.cuda.Stream.null.synchronize()
                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                    times.append(time.perf_counter())

                    # Get the bare system matrix back, needed for
                    # transmission calculation

                    if self.quatrex_config.qtbm.method != "SplitSolve":
                        # Subtract the open boundary conditions
                        for contact, sigma_obc in sigma_obc_per_contact.items():
                            for k_t, sigma_obc_k in sigma_obc.items():
                                inplace.iadd_obc(
                                    system_matrix.data,
                                    sigma_obc_k[i, :, :],
                                    sigma_obc_update_indices[contact],
                                    k_t,
                                    contact.transverse_repetition_grid,
                                )

                    # Unscale the overlap matrices
                    # to be able to process multiple energies
                    for overlap in self.device.overlap_matrices.values():
                        overlap.data *= 1 / energy

                    if injection_tot.size != 0:
                        # Input
                        self._compute_observables(
                            phi[:, injected_mask],
                            injection_segments,
                            i,
                            batch_start + i,
                            sigma_obc_per_contact,
                            reflection_per_contact,
                            eig_ref_per_contact,
                            phi_inv_ref_per_contact,
                            phi_inj_per_contact,
                            bloch_per_contact,
                            phi_ref_per_contact,
                            system_matrix,
                            self.device.overlap_matrices,
                            k_idx,
                        )

                        del phi

                    if xp.__name__ == "cupy":
                        xp.cuda.Stream.null.synchronize()
                    t_observables = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time for computing observables: {t_observables:.2f} s",
                            flush=True,
                        )

                    # Print memory usage at end of energy iteration
                    if comm.rank == 0:
                        cpu_mem_gb = get_cpu_memory_gb()
                        if xp.__name__ == "cupy":
                            gpu_mem_free, gpu_mem_total = xp.cuda.Device().mem_info
                            gpu_mem_used_gb = (
                                (gpu_mem_total - gpu_mem_free) / 1024 / 1024 / 1024
                            )
                            gpu_mem_total_gb = gpu_mem_total / 1024 / 1024 / 1024
                            print(
                                f"Energy {batch_start + i}: CPU memory: {cpu_mem_gb:.2f} GB, GPU memory: {gpu_mem_used_gb:.2f}/{gpu_mem_total_gb:.2f} GB",
                                flush=True,
                            )
                        else:
                            print(
                                f"Energy {batch_start + i}: CPU memory: {cpu_mem_gb:.2f} GB",
                                flush=True,
                            )

                t_iteration = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for iteration: {t_iteration:.2f} s", flush=True)

            # Remove the k-point phase factors from the Hamiltonian and Overlap
            for r, h_r in self.device.hamiltonians.items():
                if r == (0, 0, 0):
                    continue
                h_r.data /= xp.exp(2j * np.pi * np.dot(k, r))

            for r, s_r in self.device.overlap_matrices.items():
                if r == (0, 0, 0):
                    continue
                s_r.data /= xp.exp(2j * np.pi * np.dot(k, r))

        t_iteration = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(f"Time for QTBM: {t_iteration:.2f} s", flush=True)

        # Gather the observables
        comm.Barrier()
        for key, transmission in self.observables.transmissions.items():
            self.observables.transmissions[key] = xp.concatenate(
                comm.allgather(transmission), axis=-1
            )
        for contact, ldos in self.observables.electron_ldos.items():
            self.observables.electron_ldos[contact] = xp.concatenate(
                comm.allgather(ldos), axis=-1
            )

        self._compute_current()

        self._write_outputs()
