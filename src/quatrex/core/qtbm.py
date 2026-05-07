# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
from dataclasses import dataclass, field

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.kernels import inplace
from qttools.kernels.linalg.kron import kron_matmul
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import free_mempool
from qttools.utils.memory_utils import print_memory_usage
from qttools.utils.mpi_utils import get_local_slice
from qttools.wave_function_solver import (
    MUMPS,
    PARDISO,
    SuperLU,
    Thomas,
    WFSolver,
    auto_select_solver,
    cuDSS,
    preferred_sparse_format,
)
from quatrex.core.config import QuatrexConfig, SolverConfig
from quatrex.core.constants import e, h
from quatrex.core.statistics import fermi_dirac
from quatrex.device import Device
from quatrex.device.contact import Contact, OBCResult
from quatrex.grid import get_electron_energies, monkhorst_pack

profiler = Profiler()


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

        if self.config.qtbm.low_rank_obc:
            self.system_matrix_view = "upper"
            # Check if we can use real arithmetic for the system matrix
            # and solvers (only possible for reduced method with real
            # Hamiltonian and no k-point shift)
            if (
                not self.device.matrices_complex
                and self.config.device.kpoint_grid == (1, 1, 1)
                and self.config.device.kpoint_shift == (0, 0, 0)
            ):
                if comm.rank == 0:
                    print(
                        "REAL SYSTEM MATRIX OPTIMIZATION ENABLED: "
                        "Using real arithmetic for the system matrix and solvers."
                    )
                self.system_matrix_type = "real_symmetric_indefinite"

            else:
                self.system_matrix_type = "complex_hermitian_indefinite"

        else:
            self.system_matrix_view = "full"
            self.system_matrix_type = "complex_nonsymmetric"

        self.solver = self._configure_solver(
            self.config.electron.solver,
            matrix_type=self.system_matrix_type,
            matrix_view=self.system_matrix_view,
        )

        # TODO Preferred_matrix_type is not used at the moment
        self.matrix_type = preferred_sparse_format[
            self.config.electron.solver.direct_solver
        ]

        self._allocate_system_matrix()
        free_mempool()

    @staticmethod
    def _configure_solver(
        solver_config: SolverConfig,
        matrix_type: str,
        matrix_view: str,
    ) -> WFSolver:
        """Configures the wavefunction solver based on the config.

        Parameters
        ----------
        solver_config : SolverConfig
            The solver configuration containing solver type and options.
        matrix_type : str
            The type of the system matrix, describing properties like
            symmetry and definiteness.
        matrix_view : str
            The view of the system matrix sparsity, indicating which part
            of the matrix to use for symmetric matrices.

        Returns
        -------
        WFSolver
            The configured wavefunction solver instance.

        """
        if solver_config.direct_solver == "mumps":
            return MUMPS(matrix_type=matrix_type, matrix_view=matrix_view)
        if solver_config.direct_solver == "superlu":
            return SuperLU(matrix_type=matrix_type, matrix_view=matrix_view)
        if solver_config.direct_solver == "cudss":
            return cuDSS(matrix_type=matrix_type, matrix_view=matrix_view)
        if solver_config.direct_solver == "pardiso":
            return PARDISO(matrix_type=matrix_type, matrix_view=matrix_view)
        if solver_config.direct_solver == "thomas":
            return Thomas(matrix_type=matrix_type, matrix_view=matrix_view)
        if solver_config.direct_solver == "auto":
            return auto_select_solver(matrix_type=matrix_type, matrix_view=matrix_view)

        raise ValueError(f"Unknown solver: {solver_config.direct_solver}")

    # TODO: Investigate performance of the system matrix allocation
    @profiler.profile("QTBM: Allocate system matrix", level="debug")
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
            if self.system_matrix_view != "upper":
                # Account for the symmetric part if not upper view
                total_nnz += h_r.nnz
        for r, s_r in self.device.overlap_matrices.items():
            nnz_S.append(s_r.nnz)
            total_nnz += s_r.nnz
            if self.system_matrix_view != "upper":
                # Account for the symmetric part if not upper view
                total_nnz += s_r.nnz
        if not self.config.qtbm.low_rank_obc:
            for contact in self.device.contacts:
                nnz_cont.append(len(contact.orbital_indices))
                total_nnz += len(contact.orbital_indices) ** 2

        # TODO: Investigate using a SET instead
        # Concaate all indices from the hamiltonians, overlaps, and
        # contacts into a single array to find unique indices for
        # allocation
        concatenated_indices = xp.zeros((total_nnz, 2), dtype=xp.int64)

        start_idx = 0
        for r, h_r in self.device.hamiltonians.items():
            nnz = h_r.nnz
            concatenated_indices[start_idx : start_idx + nnz, 1] = h_r.indices
            concatenated_indices[start_idx : start_idx + nnz, 0] = xp.repeat(
                xp.arange(h_r.shape[0], dtype=xp.int32), xp.diff(h_r.indptr).tolist()
            )
            start_idx += nnz
            if self.system_matrix_view != "upper":
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
            if self.system_matrix_view != "upper":
                concatenated_indices[start_idx : start_idx + nnz, 0] = s_r.indices
                concatenated_indices[start_idx : start_idx + nnz, 1] = xp.repeat(
                    xp.arange(s_r.shape[0], dtype=xp.int32),
                    xp.diff(s_r.indptr).tolist(),
                )
                start_idx += nnz

        if not self.config.qtbm.low_rank_obc:
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
        # Find the unique indices and the inverse mapping to the
        # original concatenated array
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
        if "real" in self.system_matrix_type:
            data = xp.zeros_like(concatenated_indices[:, 0], dtype=xp.float64)
        else:
            data = xp.zeros_like(concatenated_indices[:, 0], dtype=xp.complex128)
        self.system_matrix = sparse.csr_matrix(
            (data, (concatenated_indices[:, 0], concatenated_indices[:, 1])),
            shape=(size, size),
            dtype=data.dtype,
        )

        # Store the indices to update in-place the system matrix for
        # each hamiltonian, overlap, and contact self-energy
        start_idx = 0
        self.hamiltonian_update_indices = {}
        self.hamiltonian_update_indices_transpose = {}
        for r, h_r in self.device.hamiltonians.items():
            self.hamiltonian_update_indices[r] = inverse_indices[
                start_idx : start_idx + h_r.nnz
            ]
            start_idx += h_r.nnz
            if self.system_matrix_view != "upper":
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
            if self.system_matrix_view != "upper":
                self.overlap_update_indices_transpose[r] = inverse_indices[
                    start_idx : start_idx + s_r.nnz
                ]
                start_idx += s_r.nnz

        self.sigma_obc_update_indices = {}
        if not self.config.qtbm.low_rank_obc:
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

    def _get_obc_result_info(
        self, obc_results: dict[Contact, OBCResult], energy_ind: int
    ):
        """Extracts the number of injected and reflected modes for each
        contact at a given energy index.

        Parameters
        ----------
        obc_results : dict[Contact, OBCResult]
            Dictionary mapping each contact to its corresponding OBC
            result containing injection and reflection data.
        energy_ind : int
            The energy index for which to extract the information.

        Returns
        -------
        num_injected : np.ndarray
            Array containing the number of injected modes for each
            contact at the given energy index.
        num_reflected : np.ndarray
            Array containing the number of reflected modes for each
            contact at the given energy index.

        """
        num_injected = np.zeros(len(obc_results), dtype=np.int32)
        num_reflected = np.zeros(len(obc_results), dtype=np.int32)
        for i, obc_result in enumerate(obc_results.values()):
            num_injected[i] = obc_result.injection[energy_ind].shape[1]
            if obc_result.reflection is not None:
                num_reflected[i] = obc_result.reflection[energy_ind].shape[1]

        return num_injected, num_reflected

    @profiler.profile("QTBM: Assemble RHS", level="default")
    def _assemble_rhs(
        self, obc_results: dict[Contact, OBCResult], energy_ind: int
    ) -> NDArray:
        """Assembles the right-hand side vector for the linear system.

        Parameters
        ----------
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact, containing
            injection and reflection data.
        energy_ind : int
            Index of the current energy being processed.

        Returns
        -------
        rhs : NDArray
            The assembled right-hand side vector for the linear system.

        """
        num_injected, num_reflected = self._get_obc_result_info(obc_results, energy_ind)

        total_num_injected = num_injected.sum()

        if total_num_injected == 0:
            # This means we will be skipping the energy point.
            return xp.zeros((self.num_orbitals, 0), dtype=xp.complex128, order="F")

        rhs = xp.zeros(
            (self.num_orbitals, total_num_injected + num_reflected.sum()),
            dtype=xp.complex128,
            order="F",
        )

        offsets_injected = np.hstack((0, np.cumsum(num_injected)))
        offsets_reflected = total_num_injected + np.hstack(
            (0, np.cumsum(num_reflected))
        )

        # Add the injection vector in the contact elements of the rhs
        for i, (contact, obc_result) in enumerate(obc_results.items()):
            rhs[
                contact.orbital_indices, offsets_injected[i] : offsets_injected[i + 1]
            ] = obc_result.injection[energy_ind]
            if self.config.qtbm.low_rank_obc:
                # Add the reflections.
                rhs[
                    contact.orbital_indices,
                    offsets_reflected[i] : offsets_reflected[i + 1],
                ] = obc_result.reflection[energy_ind]

        rhs = xp.asfortranarray(rhs)

        # If system matrix is real, convert the RHS to real with twice
        # the number of columns
        if "real" in self.system_matrix_type:
            rhs = xp.ascontiguousarray(rhs)
            rhs = rhs.view(np.float64)
            rhs = xp.asfortranarray(rhs)

        return rhs

    def _add_matrix_to_system_matrix(
        self, k: xp.complex128, factor: xp.float64, type: str = "hamiltonian"
    ) -> None:
        """Adds the contribution of a matrix to the system matrix for a
        given k-point and multiplication factor.

        Parameters
        ----------
        k : np.complex128
            The k-point for which the system matrix is being
            constructed.
        factor : np.float64
            A scaling factor for the matrix contribution.
        type : str, optional
            The type of matrix to add, either "hamiltonian" or
            "overlap". Default is "hamiltonian".

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
            if self.system_matrix_view != "upper":
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
        self, factor: np.float64, obc_results: dict[Contact, OBCResult], energy_ind: int
    ) -> None:
        """Adds the contribution of a contact self-energy to the system
        matrix for a given contact.

        Parameters
        ----------
        factor : np.float64
            A scaling factor for the self-energy contribution.
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact.
        energy_ind : int
            Index of the current energy being processed.

        """

        for contact, obc_result in obc_results.items():
            for k_t, sigma_obc in obc_result.sigma_obc_k.items():
                inplace.scatter_add_scaled_obc(
                    self.system_matrix.data,
                    sigma_obc[energy_ind, :, :],
                    self.sigma_obc_update_indices[contact],
                    k_t,
                    contact.transverse_repetition_grid,
                    factor,
                )

    @profiler.profile("QTBM: Assemble system matrix", level="default")
    def _assemble_system_matrix(
        self,
        kpoint: xp.complex128,
        energy: xp.float64,
        obc_results: dict[Contact, OBCResult],
        energy_ind: int,
    ) -> None:
        """Assembles the system matrix for a given k-point and energy index.

        Parameters
        ----------
        kpoint : np.complex128
            The k-point for which the system matrix is being constructed.
        energy : np.float64
            The energy value for which to construct the system matrix.
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact.
        energy_ind : int
            Index of the current energy being processed.

        """
        self.system_matrix.data[:] = 0

        # Add the Hamiltonian and overlap to the system matrix
        self._add_matrix_to_system_matrix(kpoint, -1, type="hamiltonian")
        self._add_matrix_to_system_matrix(kpoint, energy, type="overlap")

        if self.config.qtbm.low_rank_obc:
            # No need to add the OBC self-energy to the system matrix
            return

        # Add the boundary self-energy contributions.
        self._add_sigma_obc_to_system_matrix(-1, obc_results, energy_ind)

    def _assemble_pseudo_inverse(
        self,
        obc_results: dict[Contact, OBCResult],
        offsets_reflected: NDArray,
        energy_ind: int,
        shape: tuple,
    ) -> sparse.csr_matrix:
        """Constructs the sparse device-size pseudo-inverse.

        Parameters
        ----------
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact.
        offsets_reflected : NDArray
            Array of offsets for the reflected modes.
        energy_ind : int
            Index of the current energy being processed.
        shape : tuple
            Shape of the resulting sparse matrix.

        Returns
        -------
        sparse.csr_matrix
            The sparse transposed vector.

        """

        data = xp.concatenate(
            [
                obc_result.phi_inv_reflected[energy_ind].flatten()
                for obc_result in obc_results.values()
            ],
        )

        rows = xp.concatenate(
            [
                xp.repeat(
                    xp.arange(start, stop),
                    obc_result.phi_inv_reflected[energy_ind].shape[1],
                )
                for start, stop, obc_result in zip(
                    offsets_reflected[:-1],
                    offsets_reflected[1:],
                    obc_results.values(),
                )
            ]
        )

        cols = xp.concatenate(
            [
                xp.tile(
                    xp.asarray(contact.orbital_indices),
                    obc_result.phi_inv_reflected[energy_ind].shape[0],
                )
                for contact, obc_result in obc_results.items()
            ]
        )

        return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=xp.complex128)

    @profiler.profile("QTBM: Recover full-rank wavefunction", level="default")
    def _recover_full_rank_wavefunction(
        self,
        phi: NDArray,
        obc_results: dict[Contact, OBCResult],
        energy_ind: int,
    ) -> NDArray:
        """Recovers the full-rank wavefunction from the low-rank solution.

        Parameters
        ----------
        phi : NDArray
            The low-rank wavefunction solution obtained from solving the
            linear system with the reduced method.
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact, containing
            injection and reflection data.
        energy_ind : int
            Index of the current energy being processed.

        Returns
        -------
        NDArray
            The recovered full-rank-equivalent wavefunction solution.

        """

        num_injected, num_reflected = self._get_obc_result_info(obc_results, energy_ind)

        total_num_injected = num_injected.sum()
        offsets_reflected = np.hstack((0, np.cumsum(num_reflected)))

        # Apply the correction to the injected modes
        # according to the reduced method

        # Generate the device-sized pseudo-inverse
        phi_inv_tot = self._assemble_pseudo_inverse(
            obc_results,
            offsets_reflected,
            energy_ind,
            shape=(num_reflected.sum(), self.num_orbitals),
        )

        # Generate the eigenvalue matrix
        eig_tot = xp.concatenate(
            [
                obc_result.eig_reflected[energy_ind]
                for obc_result in obc_results.values()
            ]
        )

        if "real" in self.system_matrix_type:
            phi = xp.ascontiguousarray(phi)
            phi = phi.view(xp.complex128)
            phi = xp.asfortranarray(phi)

        # NOTE: xp.split returns views, so this does not copy the data.
        phi_injected, phi_reflected = xp.split(phi, [total_num_injected], axis=1)

        phi_injected += phi_reflected @ xp.linalg.solve(
            xp.diag(eig_tot) - phi_inv_tot @ phi_reflected,
            phi_inv_tot @ phi_injected,
        )

        return phi_injected

    def _compute_transmissions(
        self,
        phi: NDArray,
        injection_slices: dict,
        global_energy_ind: int,
        obc_results: dict[Contact, OBCResult],
        kpoint_ind: int,
    ):
        """Computes transmission coefficients.

        Parameters
        ----------
        phi : NDArray
            Wavefunction solution matrix. Each column represents a
            wavefunction for a specific injection mode.
        injection_slices : dict
            Dictionary of slices for each contact where each slice
            corresponds to the contact's injection modes.
        global_energy_ind : int
            Energy index in the global energy array for storing results.
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact, containing
            injection and reflection data.
        kpoint_ind : int
            Index of the current k-point being processed.

        """
        for (
            contact_in,
            contact_out,
        ), transmission in self.observables.transmissions.items():
            # Get the all the wavefunctions injected from contact 1 and
            # extract the elements inside contact 2

            # Wavefunctions injected from contact_in and evaluated at contact_out
            phi_nt = phi[contact_out.orbital_indices, injection_slices[contact_in]]

            # Compute the transmission
            if phi_nt.size == 0:
                continue

            obc_result = obc_results[contact_out]
            if self.config.qtbm.low_rank_obc:
                S_P = obc_result.reflection @ (
                    xp.diag(1 / obc_result.eig_reflected)
                    @ (obc_result.phi_inv_reflected @ phi_nt)
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

                for (ky, kz), sigma_obc in obc_result.sigma_obc_k.items():
                    S_P += kron_matmul(
                        xp.exp(-1j * ky * indices_y - 1j * kz * indices_z),
                        sigma_obc,
                        phi_nt,
                    )

            transmission[kpoint_ind, global_energy_ind] = xp.trace(
                -2 * xp.imag(phi_nt.T.conj() @ S_P)
            )

    def _compute_ldos(
        self,
        phi: NDArray,
        injection_slices: dict,
        global_energy_ind: int,
        obc_results: dict[Contact, OBCResult],
        kpoint: float,
        kpoint_ind: int,
    ):
        r"""Computes density of states.

        Parameters
        ----------
        phi : NDArray
            Wavefunction solution matrix. Each column represents a
            wavefunction for a specific injection mode.
        injection_slices : dict
            Dictionary of slices for each contact where each slice
            corresponds to the contact's injection modes.
        global_energy_ind : int
            Energy index in the global energy array for storing results.
        obc_results : dict[Contact, OBCResult]
            Dictionary of OBC results for each contact, containing
            injection and reflection data.
        kpoint : float
            The local k-point value for the current calculation.
        kpoint_ind : int
            Index of the current k-point being processed.

        """

        # Compute the DOS
        # diag(phi^H @ S @ phi)
        # S @ phi needs to consider that
        # the overlap matrices are infinite

        phi_ortho = xp.zeros_like(phi)

        # Accumulate the contribution from every overlap matrix
        for r, overlap in self.device.overlap_matrices.items():
            phase = xp.exp(2j * np.pi * np.dot(kpoint, r))
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

            # Remove the contribution from the diagonal of the overlap
            # matrix
            temp = sparse.diags(overlap.diagonal()) @ phi
            temp *= phase

            phi_ortho -= temp

            del temp

        for contact, obc_result in obc_results.items():
            orbital_indices = contact.orbital_indices

            phi_cont = xp.zeros(
                (orbital_indices.shape[0], phi.shape[1]), dtype=xp.complex128
            )
            phi_cont[:, injection_slices[contact]] = obc_result.b_injected

            if self.config.qtbm.low_rank_obc:
                phi_cont += obc_result.phi_reflected @ (
                    xp.diag(1 / obc_result.eig_reflected)
                    @ (obc_result.phi_inv_reflected @ phi[orbital_indices, :])
                )

            else:
                ny, nz = contact.transverse_repetition_grid
                indices_y = -xp.arange(ny)[:, None] + xp.arange(ny)[None, :]
                indices_z = -xp.arange(nz)[:, None] + xp.arange(nz)[None, :]

                indices_y = xp.kron(indices_y, xp.ones((nz, nz)))
                indices_z = xp.tile(indices_z, (ny, ny))

                # This upscales the block matrix if the contact
                # has periodicity in the transverse directions
                for key, value in obc_result.bloch_k.items():
                    phi_cont += kron_matmul(
                        xp.exp(-1j * key[0] * indices_y - 1j * key[1] * indices_z),
                        value,
                        phi[orbital_indices, :],
                    )

            # Add the spill over from the overlap
            for r, overlap in self.device.overlap_matrices.items():
                phi_ortho[orbital_indices, :] += (
                    contact.get_coupling_matrix(overlap)
                    * xp.exp(2j * np.pi * np.dot(kpoint, r))
                ) @ phi_cont
                phi_ortho[orbital_indices, :] += (
                    contact.get_coupling_matrix(overlap, transpose=True)
                    * xp.exp(-2j * np.pi * np.dot(kpoint, r))
                ) @ phi_cont
            # CHECK SPILL OVER ERROR (DEBUG)
            error = contact.get_coupling_matrix(self.system_matrix) @ phi_cont
            if "real" in self.system_matrix_type:
                # For real system matrix, we need to convert phi to real
                # before multiplying with the system matrix, and then
                # convert back to complex
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
            if self.system_matrix_view == "upper":
                # Need to add the contribution from the lower view of
                # the system matrix as well
                error += (
                    contact.get_coupling_matrix(self.system_matrix, transpose=True)
                    @ phi_cont
                )
                if "real" in self.system_matrix_type:
                    # For real system matrix, we need to convert phi to
                    # real before multiplying with the system matrix,
                    # and then convert back to complex
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

            injection_segment = injection_slices[contact]

            # Get the wavefunctions of the contact
            phi_c = phi[:, injection_segment]

            # Get the "orthogonalized" wavefunction of the contact
            phi_c_ortho = phi_ortho[:, injection_segment]

            if phi_c.size != 0:
                self.observables.electron_ldos[contact][
                    kpoint_ind, :, global_energy_ind
                ] = xp.real(xp.sum(phi_c * phi_c_ortho, axis=1) / (2 * xp.pi))

    @profiler.profile("QTBM: Compute observables", level="default")
    def _compute_observables(
        self,
        phi: NDArray,
        local_energy_ind: int,
        global_energy_ind: int,
        obc_results: dict[Contact, OBCResult],
        kpoint: float,
        kpoint_ind: int,
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
        local_energy_ind : int
            Energy index in the local energy array.
        global_energy_ind : int
            Energy index in the global energy array for storing results.
        obc_results : dict[Contact, OBCResult]
            Dictionary mapping each contact to its corresponding OBC
            result containing injection and reflection data.
        kpoint : float
            The local k-point value for the current calculation.
        kpoint_ind : int
            Index of the current k-point being processed.

        """

        num_injected, __ = self._get_obc_result_info(obc_results, local_energy_ind)
        offsets_injected = np.hstack((0, np.cumsum(num_injected)))

        injection_slices = {
            contact: slice(offsets_injected[i], offsets_injected[i + 1])
            for i, contact in enumerate(obc_results.keys())
        }

        energy_obc_results = {
            contact: obc_result[local_energy_ind]
            for contact, obc_result in obc_results.items()
        }

        # Compute transmissions for all the possible contact couples
        self._compute_transmissions(
            phi,
            injection_slices,
            global_energy_ind,
            energy_obc_results,
            kpoint_ind,
        )

        # Compute the DOS
        self._compute_ldos(
            phi,
            injection_slices,
            global_energy_ind,
            energy_obc_results,
            kpoint,
            kpoint_ind,
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
                    xp.trapezoid(
                        prefactor * transmission,
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.num_kpoints
                * (2 * e / h)
            )

    def _write_outputs(self):
        """Writes the computed observables to output files."""
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

    def _compute_excess_charge_densities(self):
        """Computes the charge density from the local density of states.

        Returns
        -------
        excess_electron_density : NDArray
            The excess electron density computed from the local density
            of states.
        excess_hole_density : NDArray
            The excess hole density computed from the local density of
            states.
        """

        # Compute the spectral electron and hole densities.
        electron_density = np.zeros((self.num_orbitals, self.electron_energies.size))
        hole_density = np.zeros((self.num_orbitals, self.electron_energies.size))
        for contact, ldos in self.observables.electron_ldos.items():
            occupancy = fermi_dirac(
                self.electron_energies - contact.fermi_level,
                self.config.electron.temperature,
            )

            electron_density += occupancy * ldos.sum(axis=0) * 2  # Spin
            hole_density += (1 - occupancy) * ldos.sum(axis=0) * 2  # Spin

        mid_gap_energy = (
            self.config.electron.conduction_band_edge
            + self.config.electron.valence_band_edge
        ) / 2
        mid_gap_energy = self.device.potential + mid_gap_energy

        mask = self.electron_energies > mid_gap_energy[:, None]
        electron_density[~mask] = 0
        hole_density[mask] = 0

        excess_electron_density = np.trapezoid(
            electron_density, self.electron_energies, axis=1
        )
        excess_hole_density = np.trapezoid(hole_density, self.electron_energies, axis=1)

        return excess_electron_density, excess_hole_density

    def set_potential(self, potential: NDArray):
        """Sets the potential for the QTBM calculation.

        This method can be used to update the potential in the system
        matrix for self-consistent calculations. It modifies the system
        matrix in-place to include the new potential.

        Parameters
        ----------
        potential : NDArray
            The new potential values to be set in the system matrix.

        """
        if potential.shape[0] == self.device.atom_coordinates.shape[0]:
            # Upscale the potential to the number of orbitals
            orbitals_per_atom = [
                self.config.device.num_orbitals_per_atom.get(species, 1)
                for species in self.device.atomic_species
            ]
            potential = xp.repeat(potential, orbitals_per_atom, axis=0)

        # HACK: Because the potential is baked into the Hamiltonian, we
        # need to update the Hamiltonian matrices.
        if self.device.potential is None:
            self.device.potential = potential
        else:
            delta_potential = potential - self.device.potential
            self.device.potential = delta_potential

        self.device.apply_potential()
        for contact in self.device.contacts:
            contact.unit_cell_hamiltonian = {}
            contact.unit_cell_overlap = {}

            contact._init_hamiltonian_overlap_matrices()

    def compute_charge_density(self) -> NDArray:
        """Computes the charge density from the QTBM calculation.

        This method runs the full QTBM calculation and then integrates
        the local density of states to obtain the charge density. This
        is typically used in self-consistent calculations where the
        charge density is needed to update the potential.

        Returns
        -------
        charge_density : NDArray
            The computed charge density for the device.

        """

        self.run()

        electron_density, hole_density = self._compute_excess_charge_densities()
        charge_density = electron_density - hole_density

        # From orbital to atom resolved charge density.
        charge_density = np.add.reduceat(
            charge_density, self.device.orbital_offsets[:-1]
        )

        return charge_density

    @profiler.profile(label="QTBM", level="default", comm=comm)
    def run(self) -> None:
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        comm.barrier()

        for kpoint_ind, kpoint in enumerate(self.kpoints):
            if comm.rank == 0:
                print(
                    f"Processing k-point {kpoint_ind+1} of {self.num_kpoints}",
                    flush=True,
                )

            for batch_start in range(0, len(self.local_energies), self.max_batch_size):
                with profiler.profile_range(
                    label="QTBM: Process energy batch", level="default"
                ):
                    energy_batch = self.local_energies[
                        batch_start : batch_start + self.max_batch_size
                    ]

                    if comm.rank == 0:
                        print(
                            f"Processing energies {batch_start} to {batch_start + len(energy_batch) - 1}",
                            flush=True,
                        )

                    # Compute the boundary self-energy and injection vector.
                    obc_results = {}
                    free_mempool()

                    with profiler.profile_range(
                        label="QTBM: Boundary conditions", level="default"
                    ):
                        for contact in self.device.contacts:
                            obc_results[contact] = contact.compute_boundary(
                                kpoint * 2 * np.pi,
                                energy_batch,
                                return_modes_only=self.config.qtbm.low_rank_obc,
                            )

                    for energy_ind, energy in enumerate(energy_batch):
                        rhs = self._assemble_rhs(obc_results, energy_ind)

                        if rhs.size == 0:
                            # No modes are injected at this energy, so we
                            # can skip the calculation.
                            continue

                        self._assemble_system_matrix(
                            kpoint, energy, obc_results, energy_ind
                        )

                        # Solve for the wavefunction
                        phi = self.solver.solve(
                            self.system_matrix,
                            rhs,
                            reuse_analysis=True,
                            reuse_factorization=False,
                        )

                        if self.config.qtbm.low_rank_obc:
                            phi = self._recover_full_rank_wavefunction(
                                phi, obc_results, energy_ind
                            )

                        if not self.config.qtbm.low_rank_obc:
                            # Get the bare system matrix back, needed for
                            # transmission calculation
                            self._add_sigma_obc_to_system_matrix(
                                1, obc_results, energy_ind
                            )

                        # Input
                        self._compute_observables(
                            phi,
                            energy_ind,
                            batch_start + energy_ind,
                            obc_results,
                            kpoint,
                            kpoint_ind,
                        )

                        del phi
                        del rhs

                        # Keep an end-of-energy memory report for all methods.
                        print_memory_usage()
                        free_mempool()

        # Gather the observables
        for key, transmission in self.observables.transmissions.items():
            self.observables.transmissions[key] = comm.stack.all_gather_v(
                transmission, axis=1
            )
        for contact, ldos in self.observables.electron_ldos.items():
            self.observables.electron_ldos[contact] = comm.stack.all_gather_v(
                ldos, axis=2
            )

        self._compute_current()

        self._write_outputs()

        if comm.rank == 0:
            print("QTBM calculation complete", flush=True)
