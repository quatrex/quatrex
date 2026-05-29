# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.kernels import inplace
from qttools.kernels.linalg.kron import kron_matmul
from qttools.utils.gpu_utils import free_mempool, synchronize_device
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
    """Get current CPU memory usage in GB.

    Returns
    -------
    float
        Current CPU memory usage in GB, or 0.0 if it cannot be determined.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return int(line.split()[1]) / 1024 / 1024
    except FileNotFoundError:
        # If status file is not found (e.g., on non-Linux systems), return 0.0
        return 0.0
    return 0.0


def print_memory_usage(stage: str, energy_index: int | None = None) -> None:
    """Print CPU/GPU memory usage for rank 0.

    Parameters
    ----------
    stage : str
        Human-readable stage label.
    energy_index : int | None, optional
        Global energy index to include in the message.
    """
    if comm.rank != 0:
        return

    prefix = f"[Memory] {stage}"
    if energy_index is not None:
        prefix += f" (energy {energy_index})"

    cpu_mem_gb = get_cpu_memory_gb()
    if xp.__name__ == "cupy":
        synchronize_device()
        gpu_mem_free, gpu_mem_total = xp.cuda.Device().mem_info
        gpu_mem_used_gb = (gpu_mem_total - gpu_mem_free) / 1024 / 1024 / 1024
        gpu_mem_total_gb = gpu_mem_total / 1024 / 1024 / 1024
        print(
            f"{prefix}: CPU {cpu_mem_gb:.2f} GB, GPU {gpu_mem_used_gb:.2f}/{gpu_mem_total_gb:.2f} GB",
            flush=True,
        )
    else:
        print(f"{prefix}: CPU {cpu_mem_gb:.2f} GB", flush=True)


def construct_device_pseudo_inverse(
    vector_per_cont: dict,
    injection_segments: dict,
    contacts: list,
    i: int,
    injection_count: dict,
    num_orbitals: int,
) -> sparse.csr_matrix:
    """
    Construct the sparse device-size pseudo-inverse.

    Parameters
    ----------
    vector_per_cont : dict
        Dictionary mapping each contact to its corresponding pseudo-inverse vector for the current energy index `i`.
    injection_segments : dict
        Dictionary mapping each contact and energy index to its corresponding injection segment.
    contacts : list
        List of contacts in the system.
    i : int
        Current energy index.
    injection_count : dict
        Dictionary mapping each energy index to the number of injection sites.
    num_orbitals : int
        Number of orbitals in the system.

    Returns
    -------
    sparse.csr_matrix
        The sparse transposed vector.
    """
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

        if self.config.qtbm.OBC_rank_reduced:
            self.system_matrix_UP_view = True
        else:
            self.system_matrix_UP_view = False

        # Check if we can use real arithmetic for the system matrix and solvers (only possible for reduced method with real Hamiltonian and no k-point shift)
        if (
            self.config.device.kpoint_grid == (1, 1, 1)
            and self.config.device.kpoint_shift == (0, 0, 0)
            and not self.device.matrices_complex
            and self.config.qtbm.OBC_rank_reduced
        ):
            self.real_system_matrix = True
            print(
                "REAL SYSTEM MATRIX OPTIMIZATION ENABLED: Using real arithmetic for the system matrix and solvers."
            )
        else:
            self.real_system_matrix = False

        self.solver = self._configure_solver(self.config.electron.solver)

        # TODO Preferred_matrix_type is not used at the moment
        self.matrix_type = preferred_matrix_type[
            self.config.electron.solver.direct_solver
        ]

        self._allocate_system_matrix()
        free_mempool()

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
        if self.config.qtbm.OBC_rank_reduced:
            if self.real_system_matrix:
                matrix_type = "real_symmetric_indefinite"
            else:
                matrix_type = "complex_hermitian_indefinite"
            if self.system_matrix_UP_view:
                view = "up"
            else:
                raise ValueError("Symmetric matrix currently only supports upper view.")
        else:
            matrix_type = "complex_nonsymmetric"
            if self.system_matrix_UP_view:
                raise ValueError("Nonsymmetric matrix cannot have upper view.")
            else:
                view = "default"

        if solver_config.direct_solver == "mumps":
            return MUMPS(matrix_type=matrix_type, view=view)
        if solver_config.direct_solver == "superlu":
            return SuperLU(matrix_type=matrix_type, view=view)
        if solver_config.direct_solver == "cudss":
            return cuDSS(matrix_type=matrix_type, view=view)
        if solver_config.direct_solver == "pardiso":
            return PARDISO(matrix_type=matrix_type, view=view)
        if solver_config.direct_solver == "thomas":
            return Thomas(matrix_type=matrix_type, view=view)
        if solver_config.direct_solver == "auto":
            # Auto-select the solver based on the matrix type and view
            from qttools.wave_function_solver.cudss import cudss_available
            from qttools.wave_function_solver.mumps import mumps_available
            from qttools.wave_function_solver.pardiso import pardiso_available

            if xp.__name__ == "cupy":
                if matrix_type in [
                    "real_symmetric_indefinite",
                    "complex_hermitian_indefinite",
                ]:
                    if cudss_available():
                        print("Auto-selecting cuDSS solver.")
                        return cuDSS(matrix_type=matrix_type, view=view)
                    else:
                        raise ValueError(
                            "On GPU, cuDSS is the only general solver that supports symmetric matrices"
                        )
                else:
                    if cudss_available():
                        print("Auto-selecting cuDSS solver.")
                        return cuDSS(matrix_type=matrix_type, view=view)
                    else:
                        print("Auto-selecting SuperLU solver as fallback.")
                        return SuperLU(matrix_type=matrix_type, view=view)
            else:
                if matrix_type in [
                    "real_symmetric_indefinite",
                    "complex_hermitian_indefinite",
                ]:
                    if pardiso_available():
                        print("Auto-selecting PARDISO solver.")
                        return PARDISO(matrix_type=matrix_type, view=view)
                    else:
                        raise ValueError(
                            "On CPU, PARDISO is the only general solver that supports symmetric matrices"
                        )
                else:
                    if pardiso_available():
                        print("Auto-selecting PARDISO solver.")
                        return PARDISO(matrix_type=matrix_type, view=view)
                    elif mumps_available():
                        print("Auto-selecting MUMPS solver as fallback.")
                        return MUMPS(matrix_type=matrix_type, view=view)
                    else:
                        print("Auto-selecting SuperLU solver as fallback.")
                        return SuperLU(matrix_type=matrix_type, view=view)

        raise ValueError(f"Unknown solver: {solver_config.direct_solver}")

    def _allocate_system_matrix(self) -> sparse.csr_matrix:
        """Allocates the system matrix."""

        size = self.device.hamiltonians[0, 0, 0].shape[0]
        # Count the total number of non-zero
        nnz_H = []
        nnz_S = []
        nnz_cont = []

        total_nnz = 0
        for r, h_r in self.device.hamiltonians.items():
            nnz_H.append(h_r.nnz)
            total_nnz += h_r.nnz
            if not self.system_matrix_UP_view:
                total_nnz += h_r.nnz  # Account for the symmetric part if not upper view
        for r, s_r in self.device.overlap_matrices.items():
            nnz_S.append(s_r.nnz)
            total_nnz += s_r.nnz
            if not self.system_matrix_UP_view:
                total_nnz += s_r.nnz  # Account for the symmetric part if not upper view
        if not self.config.qtbm.OBC_rank_reduced:
            for contact in self.device.contacts:
                nnz_cont.append(len(contact.orbital_indices))
                total_nnz += len(contact.orbital_indices) ** 2

        # Concaate all indices from the hamiltonians, overlaps, and contacts into a single array to find unique indices for allocation
        concatenated_indices = xp.zeros((total_nnz, 2), dtype=xp.int64)

        start_idx = 0
        for r, h_r in self.device.hamiltonians.items():
            nnz = h_r.nnz
            concatenated_indices[start_idx : start_idx + nnz, 1] = h_r.indices
            concatenated_indices[start_idx : start_idx + nnz, 0] = xp.repeat(
                xp.arange(h_r.shape[0], dtype=xp.int32), xp.diff(h_r.indptr).tolist()
            )
            start_idx += nnz
            if not self.system_matrix_UP_view:
                concatenated_indices[start_idx : start_idx + nnz, 0] = h_r.indices
                concatenated_indices[start_idx : start_idx + nnz, 1] = xp.repeat(
                    xp.arange(h_r.shape[0], dtype=xp.int32),
                    xp.diff(h_r.indptr).tolist(),
                )
                start_idx += nnz

        for r, s_r in self.device.overlap_matrices.items():
            nnz = s_r.nnz
            concatenated_indices[start_idx : start_idx + nnz, 1] = s_r.indices
            concatenated_indices[start_idx : start_idx + nnz, 0] = xp.repeat(
                xp.arange(s_r.shape[0], dtype=xp.int32), xp.diff(s_r.indptr).tolist()
            )
            start_idx += nnz
            if not self.system_matrix_UP_view:
                concatenated_indices[start_idx : start_idx + nnz, 0] = s_r.indices
                concatenated_indices[start_idx : start_idx + nnz, 1] = xp.repeat(
                    xp.arange(s_r.shape[0], dtype=xp.int32),
                    xp.diff(s_r.indptr).tolist(),
                )
                start_idx += nnz

        if not self.config.qtbm.OBC_rank_reduced:
            for contact in self.device.contacts:
                n_orb = contact.orbital_indices.shape[0]
                nnz = n_orb**2

                orbs = xp.asarray(contact.orbital_indices)
                concatenated_indices[start_idx : start_idx + nnz, 0] = xp.repeat(
                    orbs, n_orb
                )
                concatenated_indices[start_idx : start_idx + nnz, 1] = xp.tile(
                    orbs, n_orb
                )
                start_idx += nnz

        # Compress the indices from 2d to 1d (1d-unique is faster)
        concatenated_indices_M = (
            concatenated_indices[:, 0] * size + concatenated_indices[:, 1]
        )
        # Find the unique indices and the inverse mapping to the original concatenated array
        concatenated_indices_M, inverse_indices = xp.unique(
            concatenated_indices_M, return_inverse=True
        )
        # Decompress the unique indices back to 2d
        concatenated_indices = xp.zeros(
            (concatenated_indices_M.shape[0], 2), dtype=xp.int64
        )
        concatenated_indices[:, 0] = concatenated_indices_M // size
        concatenated_indices[:, 1] = concatenated_indices_M % size

        # Allocate system matrix
        if self.real_system_matrix:
            data = xp.zeros_like(concatenated_indices[:, 0], dtype=xp.float64)
        else:
            data = xp.zeros_like(concatenated_indices[:, 0], dtype=xp.complex128)
        self.system_matrix = sparse.csr_matrix(
            (data, (concatenated_indices[:, 0], concatenated_indices[:, 1])),
            shape=(size, size),
            dtype=data.dtype,
        )

        # Store the indices to update in-place the system matrix for each hamiltonian, overlap, and contact self-energy
        start_idx = 0
        self.hamiltonian_update_indices = {}
        self.hamiltonian_update_indices_transpose = {}
        for r, h_r in self.device.hamiltonians.items():
            self.hamiltonian_update_indices[r] = inverse_indices[
                start_idx : start_idx + h_r.nnz
            ]
            start_idx += h_r.nnz
            if not self.system_matrix_UP_view:
                self.hamiltonian_update_indices_transpose[r] = inverse_indices[
                    start_idx : start_idx + h_r.nnz
                ]
                start_idx += h_r.nnz

        self.overlap_update_indices = {}
        self.overlap_update_indices_transpose = {}
        for r, s_r in self.device.overlap_matrices.items():
            self.overlap_update_indices[r] = inverse_indices[
                start_idx : start_idx + s_r.nnz
            ]
            start_idx += s_r.nnz
            if not self.system_matrix_UP_view:
                self.overlap_update_indices_transpose[r] = inverse_indices[
                    start_idx : start_idx + s_r.nnz
                ]
                start_idx += s_r.nnz

        self.sigma_obc_update_indices = {}
        if not self.config.qtbm.OBC_rank_reduced:
            for contact in self.device.contacts:
                self.sigma_obc_update_indices[contact] = inverse_indices[
                    start_idx : start_idx + len(contact.orbital_indices) ** 2
                ]
                start_idx += len(contact.orbital_indices) ** 2

        # Check if SM has canonical format
        if not self.system_matrix.has_canonical_format:
            raise ValueError(
                "System matrix is not in canonical format after allocation."
            )

    def _add_matrix_to_system_matrix(
        self, k: xp.complex128, factor: xp.float64, type: str = "hamiltonian"
    ) -> None:
        """
        Adds the contribution of a matrix to the system matrix for a given k-point and multiplication factor.

        Parameters
        ----------
        k : np.complex128
            The k-point for which the system matrix is being constructed.
        factor : np.float64
            A scaling factor for the matrix contribution.
        type : str, optional
            The type of matrix to add, either "hamiltonian" or "overlap". Default is "hamiltonian".
        """

        if type == "hamiltonian":
            matrices = self.device.hamiltonians
            update_indices = self.hamiltonian_update_indices
            update_indices_transpose = self.hamiltonian_update_indices_transpose
        elif type == "overlap":
            matrices = self.device.overlap_matrices
            update_indices = self.overlap_update_indices
            update_indices_transpose = self.overlap_update_indices_transpose
        else:
            raise ValueError("Invalid matrix type. Must be 'hamiltonian' or 'overlap'.")

        for r, m_r in matrices.items():
            k_phase = np.exp(2j * np.pi * np.dot(k, r)) * factor
            if hasattr(k_phase, "get"):
                k_phase = k_phase.get()
            if k_phase.imag == 0:
                k_phase = np.float64(k_phase.real)
            else:
                k_phase = np.complex128(k_phase)
            inplace.scatter_add_scaled(
                self.system_matrix.data,
                m_r.data,
                update_indices[r],
                k_phase,
                False,
            )
            if not self.system_matrix_UP_view:
                inplace.scatter_add_scaled(
                    self.system_matrix.data,
                    m_r.data,
                    update_indices_transpose[r],
                    k_phase.conj(),
                    True,
                )
                self.system_matrix.setdiag(
                    self.system_matrix.diagonal() - m_r.diagonal() * k_phase
                )

    def _add_sigma_obc_to_system_matrix(
        self, factor: np.float64, sigma_obc_per_contact: dict, i: int
    ) -> None:
        """
        Adds the contribution of a contact self-energy to the system matrix for a given contact.

        Parameters
        ----------
        factor : np.float64
            A scaling factor for the self-energy contribution.
        sigma_obc_per_contact : dict
            Dictionary of self-energy matrices for each contact
        i : int
            Index of the current local energy being processed.
        """

        for contact, sigma_obc in sigma_obc_per_contact.items():
            for k_t, sigma_obc_k in sigma_obc.items():
                inplace.scatter_add_scaled_obc(
                    self.system_matrix.data,
                    sigma_obc_k[i, :, :],
                    self.sigma_obc_update_indices[contact],
                    k_t,
                    contact.transverse_repetition_grid,
                    factor,
                )

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
            Reflection matrices for each contact, used in reduced method.
        eig_ref_per_contact : dict
            Eigenvalues of the reflected wavefunctions for each contact, used in reduced method.
        phi_inv_ref_per_contact : dict
            Inverse of the reflected wavefunctions for each contact, used in reduced method.
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

                if self.config.qtbm.OBC_rank_reduced:
                    S_P = reflection_per_contact[contact_out] @ (
                        xp.diag(1 / eig_ref_per_contact[contact_out])
                        @ (phi_inv_ref_per_contact[contact_out] @ phi_nt)
                    )

                else:
                    S_P = xp.zeros_like(phi_nt)
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
        k_loc: float,
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
        k_loc : float
            The local k-point value for the current calculation.
        k_idx : int
            Index of the current k-point being processed.

        """

        # Compute the DOS
        # diag(phi^H @ S @ phi)
        # S @ phi needs to consider that
        # the overlap matrices are infinite

        phi_ortho = xp.zeros_like(phi)

        # Accumulate the contribution from every overlap matrix
        for r, overlap in self.device.overlap_matrices.items():
            phase = xp.exp(2j * np.pi * np.dot(k_loc, r))
            if overlap.dtype == xp.complex128:
                temp = overlap @ phi
                temp *= phase
            elif overlap.dtype == xp.float64:
                temp = phi.copy()

                # Convert to real with twice the number of columns
                temp = xp.ascontiguousarray(temp)
                temp = temp.view(xp.float64)
                temp = xp.asfortranarray(temp)

                temp = overlap @ temp

                # Convert back to complex
                temp = xp.ascontiguousarray(temp)
                temp = temp.view(xp.complex128)
                temp = xp.asfortranarray(temp)

                temp *= phase

            phi_ortho += temp
            del temp

            # Add the contribution from the transpose of the overlap matrix
            if overlap.dtype == xp.complex128:
                xp.conjugate(overlap.data, out=overlap.data)
                temp = overlap.T @ phi
                temp *= phase.conjugate()
                xp.conjugate(overlap.data, out=overlap.data)
            elif overlap.dtype == xp.float64:
                temp = phi.copy()

                # Convert to real with twice the number of columns
                temp = xp.ascontiguousarray(temp)
                temp = temp.view(xp.float64)
                temp = xp.asfortranarray(temp)

                temp = overlap.T @ temp

                # Convert back to complex
                temp = xp.ascontiguousarray(temp)
                temp = temp.view(xp.complex128)
                temp = xp.asfortranarray(temp)

                temp *= phase.conjugate()

            phi_ortho += temp
            del temp

            # Remove the contribution from the diagonal of the overlap matrix
            temp = sparse.diags(overlap.diagonal()) @ phi
            temp *= phase

            phi_ortho -= temp

            del temp

        for contact in self.device.contacts:
            orbital_indices = contact.orbital_indices

            phi_cont = xp.zeros(
                (orbital_indices.shape[0], phi.shape[1]), dtype=xp.complex128
            )
            phi_cont[:, injection_segments[contact]] = phi_inj_per_contact[contact]

            if self.config.qtbm.OBC_rank_reduced:
                phi_cont += phi_ref_per_contact[contact] @ (
                    xp.diag(1 / eig_ref_per_contact[contact])
                    @ (phi_inv_ref_per_contact[contact] @ phi[orbital_indices, :])
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
            for r, overlap in self.device.overlap_matrices.items():
                phi_ortho[orbital_indices, :] += (
                    contact.get_coupling_matrix(overlap)
                    * xp.exp(2j * np.pi * np.dot(k_loc, r))
                ) @ phi_cont
                phi_ortho[orbital_indices, :] += (
                    contact.get_coupling_matrix(overlap, transpose=True)
                    * xp.exp(-2j * np.pi * np.dot(k_loc, r))
                ) @ phi_cont
            # CHECK SPILL OVER ERROR (DEBUG)
            error = contact.get_coupling_matrix(self.system_matrix) @ phi_cont
            if self.real_system_matrix:
                # For real system matrix, we need to convert phi to real before multiplying with the system matrix, and then convert back to complex
                temp = phi.copy()
                temp = xp.ascontiguousarray(temp)
                temp = temp.view(xp.float64)
                temp = xp.asfortranarray(temp)
                temp = self.system_matrix @ temp
                temp = xp.ascontiguousarray(temp)
                temp = temp.view(xp.complex128)
                error += temp[orbital_indices, :]
                del temp
            else:
                error += (self.system_matrix @ phi)[orbital_indices, :]
            if self.system_matrix_UP_view:
                # Need to add the contribution from the lower view of the system matrix as well
                error += (
                    contact.get_coupling_matrix(self.system_matrix, transpose=True)
                    @ phi_cont
                )
                if self.real_system_matrix:
                    # For real system matrix, we need to convert phi to real before multiplying with the system matrix, and then convert back to complex
                    temp = phi.copy()
                    temp = xp.ascontiguousarray(temp)
                    temp = temp.view(xp.float64)
                    temp = xp.asfortranarray(temp)
                    temp = self.system_matrix.T @ temp
                    temp = xp.ascontiguousarray(temp)
                    temp = temp.view(xp.complex128)
                    error += temp[orbital_indices, :]
                    del temp
                else:
                    xp.conjugate(self.system_matrix.data, out=self.system_matrix.data)
                    error += (self.system_matrix.T @ phi)[orbital_indices, :]
                    xp.conjugate(self.system_matrix.data, out=self.system_matrix.data)

                error -= (
                    sparse.diags(self.system_matrix.diagonal(), format="csr")[
                        orbital_indices, :
                    ]
                    @ phi
                )

            error = xp.linalg.norm(error)

            if comm.rank == 0:
                print(f"    Spill over error for contact {contact.name[0]}: {error}")

        # Conjugate of the orthongonalized wavefunction
        xp.conjugate(phi_ortho, out=phi_ortho)

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
                ] = xp.real(xp.sum(phi_c * phi_c_ortho, axis=1) / (2 * xp.pi))

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
        k_loc: float,
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
        k_loc : float
            The local k-point value for the current calculation.
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
            k_loc,
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

        for k_idx in range(self.num_kpoints):

            if comm.rank == 0:
                print(f"Processing k-point {k_idx+1} of {self.num_kpoints}", flush=True)
            k = self.kpoints[k_idx, :]

            times.append(time.perf_counter())

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

                synchronize_device()

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

                    if self.config.qtbm.OBC_rank_reduced:
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

                    synchronize_device()
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

                if self.config.qtbm.OBC_rank_reduced:
                    reflection_segments = {}  # Needed to stack the pseudo-inverse
                    reflection_segments_translated = (
                        {}
                    )  # Needed to place the reflected modes in the correct position in the RHS
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
                            reflection_segments_translated[contact, i] = slice(
                                start + injection_count[i],
                                start + injection_count[i] + num_modes,
                            )

                        reflection_count += modes_per_energy

                synchronize_device()
                t_solve = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for OBC: {t_solve:.2f} s", flush=True)

                for i, energy in enumerate(energy_batch):
                    times.append(time.perf_counter())

                    if not self.config.qtbm.OBC_rank_reduced:
                        injection_tot = xp.zeros(
                            (self.num_orbitals, injection_count[i]),
                            dtype=xp.complex128,
                            order="F",
                        )
                    else:
                        injection_tot = xp.zeros(
                            (
                                self.num_orbitals,
                                injection_count[i] + reflection_count[i],
                            ),
                            dtype=xp.complex128,
                            order="F",
                        )

                    # Add the injection vector in the contact elements
                    # of the rhs
                    for contact in self.device.contacts:
                        injection_tot[
                            contact.orbital_indices, injection_segments[contact, i]
                        ] = injection_per_contact[contact][i]
                        if self.config.qtbm.OBC_rank_reduced:
                            # Add the reflection vectors
                            injection_tot[
                                contact.orbital_indices,
                                reflection_segments_translated[contact, i],
                            ] = reflection_per_contact[contact][i]

                    injection_tot = xp.asfortranarray(injection_tot)

                    # Variables needed for the correction in the reduced OBC method
                    if self.config.qtbm.OBC_rank_reduced:
                        # Generate the device-sized pseudo-inverse
                        phi_inv_tot = construct_device_pseudo_inverse(
                            phi_inv_ref_per_contact,
                            reflection_segments,
                            self.device.contacts,
                            i,
                            reflection_count,
                            self.num_orbitals,
                        )
                        # Generate the eigenvalue matrix
                        eig_tot = xp.concatenate(
                            [
                                eig_ref_per_contact[contact][i]
                                for contact in self.device.contacts
                            ]
                        )

                    # If system matrix is real, convert the RHS to real with twice the number of columns
                    if self.real_system_matrix:
                        injection_tot = xp.ascontiguousarray(injection_tot)
                        injection_tot = injection_tot.view(np.float64)
                        injection_tot = xp.asfortranarray(injection_tot)

                    self.system_matrix.data[:] = 0

                    synchronize_device()
                    # Add the Hamiltonian and overlap to the system matrix
                    self._add_matrix_to_system_matrix(k, -1, type="hamiltonian")
                    self._add_matrix_to_system_matrix(k, energy, type="overlap")

                    if not self.config.qtbm.OBC_rank_reduced:
                        # Add the boundary self-energy contributions
                        self._add_sigma_obc_to_system_matrix(
                            -1, sigma_obc_per_contact, i
                        )

                    synchronize_device()
                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time to set up system of eq.: {t_solve:.2f} s", flush=True
                        )

                    times.append(time.perf_counter())

                    n_injected = injection_count[i]

                    if self.config.qtbm.dump_system_matrix:
                        if comm.rank == 0:
                            print("Dumping system matrix...", flush=True)
                        if xp.__name__ == "cupy":
                            system_matrix_cpu = self.system_matrix.get()
                        else:
                            system_matrix_cpu = self.system_matrix

                        from scipy import sparse as sp_sparse

                        sp_sparse.save_npz(
                            f"{self.config.output_dir}/system_matrix_k{k_idx}_e{batch_start + i}",
                            system_matrix_cpu,
                        )

                    # SOLVE THE QTBM PROBLEM

                    if self.config.qtbm.OBC_rank_reduced:
                        if injection_tot.size != 0:
                            t1 = time.perf_counter()
                            # Solve the system
                            phi = self.solver.solve(
                                self.system_matrix,
                                injection_tot,
                                reuse_sym_fact=True,
                                reuse_fact=False,
                            )
                            # Apply the correction to the injected modes according to the reduced method

                            if self.real_system_matrix:
                                phi = xp.ascontiguousarray(phi)
                                phi = phi.view(xp.complex128)
                                phi = xp.asfortranarray(phi)

                            synchronize_device()
                            t2 = time.perf_counter()
                            if comm.rank == 0:
                                print(
                                    f"Time for solve: {t2 - t1:.2f} s",
                                    flush=True,
                                )
                            synchronize_device()
                            t1 = time.perf_counter()
                            # Correction
                            phi[:, :n_injected] += phi[
                                :, n_injected:
                            ] @ xp.linalg.solve(
                                xp.diag(eig_tot) - phi_inv_tot @ phi[:, n_injected:],
                                phi_inv_tot @ phi[:, :n_injected],
                            )
                            synchronize_device()
                            t2 = time.perf_counter()
                            if comm.rank == 0:
                                print(
                                    f"Time for correction: {t2 - t1:.2f} s", flush=True
                                )
                    else:
                        # Solve for the wavefunction
                        if injection_tot.size != 0:
                            phi = self.solver.solve(
                                self.system_matrix,
                                injection_tot,
                                reuse_sym_fact=True,
                                reuse_fact=False,
                            )

                        # No need here to convert from real to complex, since the system matrix will never be real in the non-reduced method

                    synchronize_device()
                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                    times.append(time.perf_counter())

                    # Get the bare system matrix back, needed for
                    # transmission calculation
                    if not self.config.qtbm.OBC_rank_reduced:
                        # Add the boundary self-energy contributions
                        self._add_sigma_obc_to_system_matrix(
                            1, sigma_obc_per_contact, i
                        )

                    if injection_tot.size != 0:
                        # Input
                        self._compute_observables(
                            phi[:, :n_injected],
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
                            k,
                            k_idx,
                        )

                        del phi

                    del injection_tot
                    if self.config.qtbm.OBC_rank_reduced:
                        del phi_inv_tot
                        del eig_tot

                    synchronize_device()
                    t_observables = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time for computing observables: {t_observables:.2f} s",
                            flush=True,
                        )

                    # Keep an end-of-energy memory report for all methods.
                    print_memory_usage("End of energy iteration", batch_start + i)
                    free_mempool()

                del injection_per_contact
                del phi_inj_per_contact
                del bloch_per_contact
                del sigma_obc_per_contact

                del reflection_per_contact
                del phi_ref_per_contact
                del eig_ref_per_contact
                del phi_inv_ref_per_contact
                free_mempool()

                t_iteration = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for iteration: {t_iteration:.2f} s", flush=True)

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
