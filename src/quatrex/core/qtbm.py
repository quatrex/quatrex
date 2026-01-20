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
from quatrex.device import Device
from quatrex.grid import get_electron_energies, monkhorst_pack
from quatrex.core.quatrex_config import QuatrexConfig, SolverConfig
from quatrex.core.statistics import fermi_dirac


def allocate_sys_mat(
    ham: dict, ovl: dict, boundary_SE_indexes: list[NDArray]
) -> sparse.csr_matrix:
    """Allocates the system matrix with the correct sparsity pattern.

    Parameters
    ----------
    ham : dict
        Hopping hamiltonian sparse matrices.
    ovl : dict
        Hopping hamiltonian sparse matrices.
    boundary_SE_indexes : list[NDArray]
        List of destination indices for each boundary self-energy.

    Returns
    -------
    system_matrix : sparse.csr_matrix
        The allocated system matrix with the correct sparsity pattern.
    """

    ham_cpu = {}
    ovl_cpu = {}

    for key, value in ham.items():
        ham_cpu[key] = value.get() if hasattr(value, "get") else value
    for key, value in ovl.items():
        ovl_cpu[key] = value.get() if hasattr(value, "get") else value

    # System matrix size
    mat_size = ham_cpu[(0, 0, 0)].shape[0]

    # Given a row i for the system matrix, check if it is affected by each boundary self-energy
    boundary_SE_mask = []
    for indexes in boundary_SE_indexes:
        boundary_SE_mask.append(np.zeros(mat_size, dtype=np.bool))
        boundary_SE_mask[-1][:] = False
        boundary_SE_mask[-1][indexes] = True

    # List containing all System Matrix col indices for each row
    SM_indices_list = []

    # CSR System-matrix indptr array
    SM_indptr = np.zeros(mat_size + 1, dtype=np.int64)

    for i_row in range(mat_size):

        # Compute the union of all column indices affecting this row
        row_union = np.array([], dtype=np.int64)

        # Add Hamiltonian contributions
        for r, h_r in ham_cpu.items():
            row_start = h_r.indptr[i_row]
            row_end = h_r.indptr[i_row + 1]
            row_union = np.union1d(row_union, h_r.indices[row_start:row_end])

        # Add Overlap contributions
        for r, s_r in ovl_cpu.items():
            row_start = s_r.indptr[i_row]
            row_end = s_r.indptr[i_row + 1]
            row_union = np.union1d(row_union, s_r.indices[row_start:row_end])

        # Add Boundary self-energy contributions
        for i, s_ind in enumerate(boundary_SE_mask):
            if s_ind[i_row]:
                row_union = np.union1d(row_union, boundary_SE_indexes[i])

        # Store the column indices for this row
        SM_indices_list.append(row_union)
        SM_indptr[i_row + 1] = SM_indptr[i_row] + len(row_union)

    total_SM_nnz = SM_indptr[-1]

    # Allocate data and indices arrays
    SM_data = np.zeros(total_SM_nnz, dtype=xp.complex128)
    SM_indices = np.concatenate(SM_indices_list)

    SM_data_gpu = xp.asarray(SM_data)
    SM_indices_gpu = xp.asarray(SM_indices)
    SM_indptr_gpu = xp.asarray(SM_indptr)

    system_matrix = sparse.csr_matrix(
        (SM_data_gpu, SM_indices_gpu, SM_indptr_gpu),
        shape=(mat_size, mat_size),
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
        Local density of states (LDOS) for electrons with shape
        (n_atoms, n_energies). Provides site-resolved DOS information.
    electron_density : NDArray, optional
        Electron density distribution with shape (n_atoms,).
    hole_density : NDArray, optional
        Hole density distribution with shape (n_atoms,).
    electron_current : dict, optional
        Dictionary containing current density information. Keys may
        include directional components and spatial distributions.
    spill_over_error : NDArray, optional
        Error metric for boundary condition accuracy with shape
        (n_energies,). Quantifies how well the open boundary conditions
        are satisfied.
    electron_transmission_contacts : NDArray, optional
        Contact-to-contact transmission coefficients with shape
        (n_contact_pairs, n_energies). Each element T_ij(E) gives the
        transmission probability from contact i to contact j at energy
        E.
    electron_transmission_contacts_labels : list[str]
        String labels for each contact pair in the format
        "source->drain" corresponding to the transmission matrix rows.
    electron_transmission_x_slabs : NDArray, optional
        Spatial transmission between adjacent slabs with shape
        (n_contacts, n_slabs-1, n_energies). Shows current flow as a
        function of position for each injection contact.
    electron_dos_x_slabs : NDArray, optional
        Position-resolved density of states with shape (n_contacts,
        n_slabs, n_energies). Provides spatial distribution of DOS for
        each injection contact.
    excess_charge_density : NDArray, optional
        Excess charge density distribution with shape (n_atoms,).

    """

    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)

    spill_over_error: NDArray = None

    electron_transmission_contacts: NDArray = None
    electron_transmission_contacts_labels = []
    electron_transmission_indices = []

    electron_dos_orb: NDArray = None

    electron_charge_orb: NDArray = None
    electron_charge_at: NDArray = None

    hole_charge_orb: NDArray = None
    hole_charge_at: NDArray = None


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
    neutrality_level : float
        Charge neutrality level for the device.
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

        for contact_idx_in in range(self.num_contacts):
            for contact_idx_out in range(self.num_contacts):
                if contact_idx_out != contact_idx_in:
                    self.observables.electron_transmission_contacts_labels.append(
                        f"{self.device.contacts[contact_idx_in].name[0]}{self.device.contacts[contact_idx_out].name[0]}"
                    )
                    self.observables.electron_transmission_indices.append(
                        (contact_idx_in, contact_idx_out)
                    )

        # Initialize the observables
        self.observables.electron_transmission_contacts = xp.zeros(
            (self.num_kpoints, self.num_transmissions, self.local_energies.shape[0]),
            dtype=xp.float64,
        )

        self.observables.electron_dos_orb = xp.zeros(
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

        self.observables.electron_current["contact_current"] = xp.zeros(
            self.num_transmissions
        )

        self.observables.electron_charge_orb = xp.zeros(
            (self.num_orbitals,), dtype=xp.float64
        )

        self.observables.hole_charge_orb = xp.zeros(
            (self.num_orbitals,), dtype=xp.float64
        )

        if (
            self.quatrex_config.electron.conduction_band_edge is None
            or self.quatrex_config.electron.valence_band_edge is None
        ):
            if comm.rank == 0:
                print(
                    "WARNING: No band edges provided, only electron charge will be computed."
                )
            self.neutrality_level = -np.inf
        else:
            self.neutrality_level = 0.5 * (
                self.quatrex_config.electron.conduction_band_edge
                + self.quatrex_config.electron.valence_band_edge
            )

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
        for nt in range(self.num_transmissions):
            # Get the all the wavefunctions injected from contact 1 and
            # extract the elements inside contact 2
            contact_idx_in, contact_idx_out = (
                self.observables.electron_transmission_indices[nt]
            )

            contact_in = contacts[contact_idx_in]
            contact_out = contacts[contact_idx_out]

            # Wavefunctions injected from contact_in and evaluated at contact_out
            phi_nt = phi[
                contact_out.orbitals_contact,
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

                self.observables.electron_transmission_contacts[
                    k_idx, nt, global_energy_index
                ] = xp.trace(-2 * xp.imag(phi_nt.T.conj() @ S_P))

        # Compute the DOS
        # diag(phi^H @ S @ phi)
        # S @ phi needs to consider that
        # the overlap matrices are infinite

        phi_ortho = xp.zeros_like(phi)
        for overlap in overlap_matrices.values():
            phi_ortho += overlap @ phi

        for contact in contacts:
            orbitals_contact = contact.orbitals_contact
            ny, nz = contact.transverse_repetition_grid

            phi_cont = xp.zeros(
                (orbitals_contact.shape[0], phi.shape[1]), dtype=xp.complex128
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
                    phi[orbitals_contact, :],
                )

            # Add the spill over from the overlap
            for overlap in overlap_matrices.values():
                phi_ortho[orbitals_contact, :] += (
                    contact.get_coupling_matrix(overlap) @ phi_cont
                )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = xp.linalg.norm(
                contact.get_coupling_matrix(system_matrix) @ phi_cont
                + system_matrix[orbitals_contact, :] @ phi
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
                self.observables.electron_dos_orb[
                    k_idx, contact_idx, :, global_energy_index
                ] = xp.real(xp.sum(phi_c.conj() * phi_c_ortho, axis=1) / (2 * xp.pi))

    def _compute_current(self):
        """Computes the electron current from the transmission data."""

        # Compute the current from all the k dependent transmissions
        for nt in range(self.num_transmissions):
            contact_idx_in, contact_idx_out = (
                self.observables.electron_transmission_indices[nt]
            )
            Fermi_factor = fermi_dirac(
                self.electron_energies
                - self.device.contacts[contact_idx_in].fermi_level,
                self.quatrex_config.electron.temperature,
            ) - fermi_dirac(
                self.electron_energies
                - self.device.contacts[contact_idx_out].fermi_level,
                self.quatrex_config.electron.temperature,
            )

            self.observables.electron_current["contact_current"][nt] = -(
                xp.sum(
                    xp.trapz(
                        Fermi_factor
                        * self.observables.electron_transmission_contacts[:, nt, :],
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.num_kpoints
                * (2 * e / h)
            )

    def _compute_electron_charge(self):
        """Computes the electron charge from the DOS data."""

        # Compute the orbital electron charge density per orbital
        for n in range(self.num_contacts):
            Fermi_factor = fermi_dirac(
                self.electron_energies - self.device.contacts[n].fermi_level,
                self.quatrex_config.electron.temperature,
            )
            Fermi_factor[self.electron_energies < self.neutrality_level] = 0.0
            self.observables.electron_charge_orb += (
                2
                * xp.sum(
                    xp.trapz(
                        self.observables.electron_dos_orb[:, n, :, :] * Fermi_factor,
                        self.electron_energies,
                        axis=2,
                    ),
                    axis=0,
                )
                / self.num_kpoints
            )

        # Compute atomic electron charge from orbital contributions
        self.observables.electron_charge_at = xp.zeros(
            self.device.orbital_offsets.shape[0] - 1
        )
        self.observables.electron_charge_at = xp.add.reduceat(
            self.observables.electron_charge_orb, self.device.orbital_offsets[:-1]
        )

    def _compute_hole_charge(self):
        """Computes the hole charge from the DOS data."""

        # Compute the orbital hole charge density per orbital
        for n in range(self.num_contacts):
            Fermi_factor = fermi_dirac(
                self.electron_energies - self.device.contacts[n].fermi_level,
                self.quatrex_config.electron.temperature,
            )
            Fermi_factor[self.electron_energies > self.neutrality_level] = 1
            self.observables.hole_charge_orb += (
                2
                * xp.sum(
                    xp.trapz(
                        self.observables.electron_dos_orb[:, n, :, :]
                        * (1 - Fermi_factor),
                        self.electron_energies,
                        axis=2,
                    ),
                    axis=0,
                )
                / self.num_kpoints
            )

        # Compute atomic hole charge from orbital contributions
        self.observables.hole_charge_at = xp.zeros(
            self.device.orbital_offsets.shape[0] - 1
        )
        self.observables.hole_charge_at = xp.add.reduceat(
            self.observables.hole_charge_orb, self.device.orbital_offsets[:-1]
        )

    def _write_outputs(self):
        if comm.rank == 0:

            output_dir = self.quatrex_config.output_dir
            if not os.path.exists(self.quatrex_config.output_dir):
                os.mkdir(self.quatrex_config.output_dir)

            for n in range(self.num_transmissions):
                np.save(
                    f"{output_dir}/transmission_{self.observables.electron_transmission_contacts_labels[n]}.npy",
                    self.observables.electron_transmission_contacts[:, n, :],
                )

                np.save(
                    f"{output_dir}/current_{self.observables.electron_transmission_contacts_labels[n]}.npy",
                    self.observables.electron_current["contact_current"][n],
                )

            for n in range(self.num_contacts):
                np.save(
                    f"{output_dir}/dos_{self.device.contacts[n].name[0]}.npy",
                    self.observables.electron_dos_orb[:, n, :, :],
                )

            np.save(
                f"{output_dir}/el_charge_orb.npy",
                self.observables.electron_charge_orb,
            )

            np.save(
                f"{output_dir}/el_charge_at.npy",
                self.observables.electron_charge_at,
            )

            np.save(f"{output_dir}/orb.npy", self.device.orbital_offsets)

            np.save(
                f"{output_dir}/ho_charge_orb.npy",
                self.observables.hole_charge_orb,
            )

            np.save(
                f"{output_dir}/ho_charge_at.npy",
                self.observables.hole_charge_at,
            )

    def run(self) -> None:
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()

        cont_ind_list = []
        for contact in self.device.contacts:
            cont_ind_list.append(contact.orbitals_contact)

        # Allocate indices to update the system matrix in-place
        system_matrix = allocate_sys_mat(
            self.device.hamiltonians, self.device.overlap_matrices, cont_ind_list
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
                system_matrix, contact.orbitals_contact
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
                            contact.orbitals_contact, injection_segments[contact, i]
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
        self.observables.electron_transmission_contacts = xp.concatenate(
            comm.allgather(self.observables.electron_transmission_contacts), axis=-1
        )
        self.observables.electron_dos_orb = xp.concatenate(
            comm.allgather(self.observables.electron_dos_orb), axis=-1
        )

        self._compute_current()
        self._compute_electron_charge()
        self._compute_hole_charge()

        self._write_outputs()
