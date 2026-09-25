# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the core class for QTBM calculations."""

import os
from dataclasses import dataclass, field

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dcsx import DCSX
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
from quatrex.contact.qtbm import OBCResult, QTBMContact
from quatrex.core.config import QuatrexConfig, SolverConfig
from quatrex.core.constants import e, h
from quatrex.core.statistics import fermi_dirac
from quatrex.core.transport import TransportSolver
from quatrex.device import QTBMDevice
from quatrex.grid import get_electron_energies

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
        QTBMContact current values for each contact pair.
    transmissions : dict, optional
        Transmission coefficients between contact pairs.
    excess_electron_density : NDArray, optional
        Orbital-resolved excess electron density.
    excess_hole_density : NDArray, optional
        Orbital-resolved excess hole density.

    """

    electron_ldos: dict[QTBMContact, NDArray] = field(default_factory=dict)
    transmissions: dict[tuple[QTBMContact, QTBMContact], NDArray] = field(
        default_factory=dict
    )

    contact_currents: dict[tuple[QTBMContact, QTBMContact], NDArray] | None = None
    excess_electron_density: NDArray | None = None
    excess_hole_density: NDArray | None = None


class QTBM(TransportSolver):
    """Quantum Transmitting Boundary Method solver.

    Parameters
    ----------
    config : QuatrexConfig
        Configuration object containing calculation parameters, energy
        grid, and numerical settings.
    device : QTBMDevice
        The quantum device object containing Hamiltonian, atomic
        structure, and attached contacts.


    Attributes
    ----------
    device : QTBMDevice
        Reference to the device object.
    observables : Observables
        Container for computed transport observables including
        transmission matrices, density of states, and current
        distributions.
    electron_energies : NDArray
        Full energy grid for the calculation.
    local_energies : NDArray
        Local portion of energy grid for MPI parallelization.

    """

    def __init__(self, config: QuatrexConfig, device: QTBMDevice) -> None:
        """Initializes the QTBM solver."""

        self.device = device

        self.config = config
        self.low_rank_obc = config.qtbm.low_rank_obc

        kpoint_grid = config.device.kpoint_grid
        if self.device.gamma_only and kpoint_grid != (1, 1, 1):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but more than one k-point is configured."
            )

        self.max_batch_size = self.config.qtbm.max_batch_size

        self.observables = Observables()

        # Get the electron energies.
        self.electron_energies = get_electron_energies(config)

        # Get the local slice of the electron energies
        self.local_energies = get_local_slice(self.electron_energies, comm.stack)

        # Look for all the combinations of contacts
        for contact_in in self.device.contacts:
            for contact_out in self.device.contacts:
                if contact_in == contact_out:
                    continue

                # Initialize the observables
                self.observables.transmissions[contact_in, contact_out] = xp.zeros(
                    (self.device.num_kpoints, self.local_energies.shape[0]),
                    dtype=xp.float64,
                )

        for contact in self.device.contacts:
            self.observables.electron_ldos[contact] = xp.zeros(
                (
                    self.device.num_kpoints,
                    self.device.num_orbitals,
                    self.local_energies.shape[0],
                ),
                dtype=xp.float64,
            )

        if self.low_rank_obc:
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

        self._allocate_bare_system_matrix()
        if not self.low_rank_obc:
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

    @profiler.profile("QTBM: Allocate bare system matrix", level="debug")
    def _allocate_bare_system_matrix(self):
        """Allocates the bare system matrix."""

        # Concatenate all indices from the hamiltonians and overlaps
        # into a single array to find unique indices for allocation
        row_ind = []
        col_ind = []

        for operator in [self.device.hamiltonians, self.device.overlap_matrices]:
            for mat in operator.values():
                row_ind.append(mat.row_ind)
                col_ind.append(mat.col_ind)

        # TODO: Check if we need to default to int64 i.e. estimate from
        # the nnz. This can be an ugly bug if the nnz of the system
        # matrix is larger than 2^31-1 while the individual matrices are
        # smaller than 2^31-1. This can happen for large systems with
        # many contacts.
        row_ind = xp.concatenate(row_ind, axis=-1)
        col_ind = xp.concatenate(col_ind, axis=-1)

        rows = self.device.hamiltonians[0, 0, 0].rows
        cols = self.device.hamiltonians[0, 0, 0].cols

        # Compress the indices from 2d to 1d (1d-unique is faster)
        indices = row_ind * cols + col_ind
        indices = xp.unique(indices)

        # Decompress the unique indices back to 2d
        row_ind = indices // cols
        col_ind = indices % cols

        if self.device.matrices_complex:
            symmetry = "hermitian"
        else:
            symmetry = "symmetric"

        # TODO: Directly pass in row and col indices.
        sparray = sparse.coo_matrix(
            (xp.empty_like(row_ind, dtype=xp.bool_), (row_ind, col_ind)),
            shape=(rows, cols),
        )

        # Allocate system matrix
        if "real" in self.system_matrix_type:
            system_matrix_dtype = xp.float64
        else:
            system_matrix_dtype = xp.complex128
        self.bare_system_matrix = DCSX.from_sparray(
            sparray=sparray,
            symmetry=symmetry,
            allocate=False,
            dtype=system_matrix_dtype,
        )

    # TODO: Investigate performance of the system matrix allocation
    @profiler.profile("QTBM: Allocate system matrix", level="debug")
    def _allocate_system_matrix(self):
        """Allocates the system matrix."""

        if self.low_rank_obc:
            raise ValueError(
                "Full system matrix allocation is not needed for low-rank OBCs."
            )

        # Concatenate all indices from the hamiltonians, overlaps, and
        # contacts into a single array to find unique indices for
        # allocation
        row_ind = []
        col_ind = []

        for operator in [self.device.hamiltonians, self.device.overlap_matrices]:
            for mat in operator.values():
                mat = mat.expand_symmetry()
                row_ind.append(mat.row_ind)
                col_ind.append(mat.col_ind)

        contact_rows = {}
        contact_cols = {}
        for contact in self.device.contacts:
            # (0,1,2) 3x3 matrix ->
            # (0,0,0,1,1,1,2,2,2)
            row_orbs = xp.asarray(contact.local_orbital_indices)
            contact_rows[contact.name] = xp.repeat(
                row_orbs, len(contact.orbital_indices)
            )
            row_ind.append(contact_rows[contact.name])
            # (0,1,2,0,1,2,0,1,2)
            col_orbs = xp.asarray(contact.orbital_indices)
            contact_cols[contact.name] = xp.tile(
                col_orbs, len(contact.local_orbital_indices)
            )
            col_ind.append(contact_cols[contact.name])

        # TODO: Check if we need to default to int64 i.e. estimate from
        # the nnz. This can be an ugly bug if the nnz of the system
        # matrix is larger than 2^31-1 while the individual matrices are
        # smaller than 2^31-1. This can happen for large systems with
        # many contacts.
        row_ind = xp.concatenate(row_ind, axis=-1)
        col_ind = xp.concatenate(col_ind, axis=-1)

        rows = self.device.hamiltonians[0, 0, 0].rows
        cols = self.device.hamiltonians[0, 0, 0].cols

        # Compress the indices from 2d to 1d (1d-unique is faster)
        indices = row_ind * cols + col_ind
        indices = xp.unique(indices)

        # Decompress the unique indices back to 2d
        row_ind = indices // cols
        col_ind = indices % cols

        # TODO: Directly pass in row and col indices.
        sparray = sparse.coo_matrix(
            (xp.empty_like(row_ind, dtype=xp.bool_), (row_ind, col_ind)),
            shape=(rows, cols),
        )

        # Allocate system matrix
        if "real" in self.system_matrix_type:
            system_matrix_dtype = xp.float64
        else:
            system_matrix_dtype = xp.complex128
        self.system_matrix = DCSX.from_sparray(
            sparray=sparray,
            allocate=False,
            dtype=system_matrix_dtype,
        )

        self.sigma_update_indices = {}
        for contact in self.device.contacts:
            self.sigma_update_indices[contact] = self.system_matrix._get_update_indices(
                contact_rows[contact.name],
                contact_cols[contact.name],
            )

    def _get_obc_result_info(
        self, obc_results: dict[QTBMContact, OBCResult], energy_ind: int
    ):
        """Extracts the number of injected and reflected modes for each
        contact at a given energy index.

        Parameters
        ----------
        obc_results : dict[QTBMContact, OBCResult]
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
        self, obc_results: dict[QTBMContact, OBCResult], energy_ind: int
    ) -> NDArray:
        """Assembles the right-hand side vector for the linear system.

        Parameters
        ----------
        obc_results : dict[QTBMContact, OBCResult]
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
            return xp.zeros(
                (self.device.num_orbitals, 0), dtype=xp.complex128, order="F"
            )

        rhs = xp.zeros(
            (self.device.num_orbitals, total_num_injected + num_reflected.sum()),
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
            if self.low_rank_obc:
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

    def _add_sigma_to_system_matrix(
        self,
        system_matrix: DCSX,
        obc_results: dict[QTBMContact, OBCResult],
        energy_ind: int,
        sigma_update_indices: dict,
    ) -> None:
        """Adds the contribution of a contact self-energy to the system
        matrix for a given contact.

        Parameters
        ----------
        system_matrix : DCSX
            The system matrix to which the self-energy will be added.
        obc_results : dict[QTBMContact, OBCResult]
            Dictionary of OBC results for each contact.
        energy_ind : int
            Index of the current energy being processed.
        sigma_update_indices : dict
            Dictionary mapping each contact to its corresponding update
            indices in the system matrix.

        """

        for contact, obc_result in obc_results.items():
            for k_t, sigma in obc_result.sigma_obc_k.items():
                if len(sigma_update_indices[contact]) > 0:
                    inplace.scatter_add_scaled_obc(
                        system_matrix.data,
                        sigma[energy_ind, :, :],
                        sigma_update_indices[contact],
                        k_t,
                        contact.transverse_repetition_grid,
                        -1.0,
                    )

    def _add_quantity(
        self,
        bare_system_matrix: DCSX,
        matrices: dict[tuple[int, int, int], DCSX],
        kpoint: xp.complex128,
        prefactor=1.0,
    ) -> None:
        """Adds a quantity (Hamiltonian or overlap) to the bare system
        matrix for a given k-point.

        Parameters
        ----------
        bare_system_matrix : DCSX
            The bare system matrix to which the quantity will be added.
        matrices : dict[tuple[int, int, int], DCSX]
            Dictionary of matrices (Hamiltonian or overlap) indexed by
            lattice vector indices.
        kpoint : np.complex128
            The k-point for which the quantity is being added.
        prefactor : float, optional
            A prefactor to scale the quantity being added. Default is
            1.0.

        """
        for r, m_r in matrices.items():
            if bare_system_matrix.symmetry != m_r.symmetry:
                raise ValueError(
                    f"Symmetry mismatch between bare system matrix "
                    f"({bare_system_matrix.symmetry}) and "
                    f"matrix ({m_r.symmetry})."
                )
            bare_system_matrix.add_(
                m_r,
                prefactor=np.exp(2j * np.pi * np.dot(kpoint, r)) * prefactor,
            )

    @profiler.profile("QTBM: Assemble system matrix", level="default")
    def _assemble_bare_system_matrix(
        self,
        kpoint: xp.complex128,
        energy: xp.float64,
    ) -> None:
        """Assembles the bare system matrix for a given k-point and
        energy index.

        Parameters
        ----------
        kpoint : np.complex128
            The k-point for which the system matrix is being
            constructed.
        energy : np.float64
            The energy value for which to construct the system matrix.

        """
        self.bare_system_matrix.data[:] = 0

        # First add the potential to simplify the assembly since we
        # inplace assemble.
        # TODO: Simplify this when there is identity overlap matrix
        # E * S
        self._add_quantity(
            self.bare_system_matrix,
            self.device.overlap_matrices,
            kpoint,
        )
        # NOTE: This is done to not add the overlap matrix thrice.
        overlap_data = self.bare_system_matrix.data.copy()
        full_data = energy * overlap_data

        # -0.5 * V @ S
        local_potential = self.device.potential[
            self.device.offsets[comm.block.rank] : self.device.offsets[
                comm.block.rank + 1
            ]
        ]
        self.bare_system_matrix.multiply_(-0.5 * local_potential[:, xp.newaxis])
        full_data += self.bare_system_matrix.data
        self.bare_system_matrix.data = overlap_data

        # -0.5 * S @ V
        self.bare_system_matrix.multiply_(-0.5 * self.device.potential)
        self.bare_system_matrix.data += full_data

        # Add the Hamiltonian
        # -H
        self._add_quantity(
            self.bare_system_matrix,
            self.device.hamiltonians,
            kpoint,
            prefactor=-1.0,
        )

    def _assemble_pseudo_inverse(
        self,
        obc_results: dict[QTBMContact, OBCResult],
        offsets_reflected: NDArray,
        energy_ind: int,
        shape: tuple,
    ) -> sparse.csr_matrix:
        """Constructs the sparse device-size pseudo-inverse.

        Parameters
        ----------
        obc_results : dict[QTBMContact, OBCResult]
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
        obc_results: dict[QTBMContact, OBCResult],
        energy_ind: int,
    ) -> NDArray:
        """Recovers the full-rank wavefunction from the low-rank solution.

        Parameters
        ----------
        phi : NDArray
            The low-rank wavefunction solution obtained from solving the
            linear system with the reduced method.
        obc_results : dict[QTBMContact, OBCResult]
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
            shape=(num_reflected.sum(), self.device.num_orbitals),
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
        obc_results: dict[QTBMContact, OBCResult],
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
        obc_results : dict[QTBMContact, OBCResult]
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
            phi_nt = phi[
                contact_out.local_orbital_indices, injection_slices[contact_in]
            ]
            phi_nt = comm.block.all_gather_v(phi_nt, axis=0)

            # Compute the transmission
            if phi_nt.size == 0:
                continue

            obc_result = obc_results[contact_out]
            if self.low_rank_obc:
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

                for (ky, kz), sigma in obc_result.sigma_obc_k.items():
                    S_P += kron_matmul(
                        xp.exp(-1j * ky * indices_y - 1j * kz * indices_z),
                        sigma,
                        phi_nt,
                    )

            transmission[kpoint_ind, global_energy_ind] = xp.trace(
                -2 * xp.imag(phi_nt.T.conj() @ S_P)
            )

    # def _compute_spillover_error(
    #     self,
    # ):
    #     # CHECK SPILL OVER ERROR (DEBUG)
    #     error = contact.get_coupling_matrix(bare_system_matrix) @ phi_cont

    #     phi = comm.block.all_gather_v(phi, axis=0)
    #     if "real" in self.system_matrix_type:
    #         # For real system matrix, we need to convert phi to real
    #         # before multiplying with the system matrix, and then
    #         # convert back to complex
    #         tmp = phi.copy()
    #         tmp = xp.ascontiguousarray(tmp)
    #         tmp = tmp.view(xp.float64)
    #         tmp = xp.asfortranarray(tmp)
    #         tmp = bare_system_matrix @ tmp
    #         tmp = xp.ascontiguousarray(tmp)
    #         tmp = tmp.view(xp.complex128)
    #         error += tmp[orbital_indices, :]
    #         del tmp
    #     else:
    #         error += (bare_system_matrix @ phi)[orbital_indices, :]
    #     if self.system_matrix_view == "upper":
    #         # Need to add the contribution from the lower view of
    #         # the system matrix as well
    #         error += (
    #             contact.get_coupling_matrix(bare_system_matrix, transpose=True)
    #             @ phi_cont
    #         )
    #         if "real" in self.system_matrix_type:
    #             # For real system matrix, we need to convert phi to
    #             # real before multiplying with the system matrix,
    #             # and then convert back to complex
    #             tmp = phi.copy()
    #             tmp = xp.ascontiguousarray(tmp)
    #             tmp = tmp.view(xp.float64)
    #             tmp = xp.asfortranarray(tmp)
    #             tmp = bare_system_matrix.T @ tmp
    #             tmp = xp.ascontiguousarray(tmp)
    #             tmp = tmp.view(xp.complex128)
    #             error += tmp[orbital_indices, :]
    #             del tmp
    #         else:
    #             xp.conjugate(bare_system_matrix.data, out=bare_system_matrix.data)
    #             error += (bare_system_matrix.T @ phi)[orbital_indices, :]
    #             xp.conjugate(bare_system_matrix.data, out=bare_system_matrix.data)

    #         error -= (
    #             sparse.diags(bare_system_matrix.diagonal(), format="csr")[
    #                 orbital_indices, :
    #             ]
    #             @ phi
    #         )

    #     error = xp.sum(xp.abs(error)**2, keepdims=True)
    #     out = xp.zeros_like(error)
    #     comm.block.all_reduce(error, out)

    #     if comm.rank == 0:
    #         print(f"    Spill over error for contact {contact.name[0]}: {out}")

    def _compute_ldos(
        self,
        phi: NDArray,
        injection_slices: dict,
        global_energy_ind: int,
        obc_results: dict[QTBMContact, OBCResult],
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
        obc_results : dict[QTBMContact, OBCResult]
            Dictionary of OBC results for each contact, containing
            injection and reflection data.
        kpoint : float
            The local k-point value for the current calculation.
        kpoint_ind : int
            Index of the current k-point being processed.

        """

        # NOTE: We use here the bare system matrix as a proxy for the
        # overlap matrix since it constructed already in the correct
        # way.
        # NOTE: I believe that assembling the overlap for a k-point is
        # faster than doing them SPMM for every r-point as done
        # previously, but this is not benchmarked yet. Reasoning is that
        # we need to unsymmetrize.
        # Doing every r-point is currently hard to bring back since the
        # unsymmetrization logic is not trivial since k-points are
        # symmetric, but not r-point matrices.
        self.bare_system_matrix.allocate_data()
        self.bare_system_matrix.data = 0.0
        self._add_quantity(
            self.bare_system_matrix,
            self.device.overlap_matrices,
            kpoint,
        )

        overlap_matrix = self.bare_system_matrix.expand_symmetry()
        self.bare_system_matrix.free_data()

        # Compute the DOS
        # diag(phi^H @ S @ phi)
        # S @ phi needs to consider that
        # the overlap matrices are infinite

        # TODO: correctly do distributed SPMM
        # For now allgather it.
        phi = comm.block.all_gather_v(phi, axis=0)

        if overlap_matrix.dtype == xp.complex128:
            phi_ortho = overlap_matrix @ phi
        elif overlap_matrix.dtype == xp.float64:
            tmp = phi.copy()

            # Convert to real with twice the number of columns
            tmp = xp.ascontiguousarray(tmp)
            tmp = tmp.view(xp.float64)
            tmp = xp.asfortranarray(tmp)

            phi_ortho = overlap_matrix @ tmp

            # Convert back to complex
            phi_ortho = xp.ascontiguousarray(phi_ortho)
            phi_ortho = phi_ortho.view(xp.complex128)
            phi_ortho = xp.asfortranarray(phi_ortho)

        # Account for the spill over from the contacts
        # i.e. since it assumed that matrices are infinite.

        for contact, obc_result in obc_results.items():
            orbital_indices = contact.orbital_indices
            local_orbital_indices = contact.local_orbital_indices

            phi_cont = xp.zeros(
                (orbital_indices.shape[0], phi.shape[1]), dtype=xp.complex128
            )
            phi_cont[:, injection_slices[contact]] = obc_result.b_injected

            if self.low_rank_obc:
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
                        comm.block.all_gather_v(phi[local_orbital_indices, :], axis=0),
                    )

            # TODO: This is a bit hacky. Should be done inside the
            # contact.
            count = len(local_orbital_indices)
            counts = np.zeros(comm.block.size, dtype=xp.int64)
            comm.block.all_gather(
                np.array(count, dtype=xp.int64),
                counts,
                backend="device_mpi",
            )
            offsets = np.array([0] + list(np.cumsum(counts)), dtype=xp.int64)

            # Add the spill over from the overlap
            phi_ortho[local_orbital_indices, :] += (
                (contact.get_coupling_matrix(overlap_matrix)) @ phi_cont
            )[offsets[comm.block.rank] : offsets[comm.block.rank + 1], :]

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
                tmp = xp.real(xp.sum(phi_c * phi_c_ortho, axis=1) / (2 * xp.pi))

                tmp = comm.block.all_gather_v(tmp, axis=0)

                self.observables.electron_ldos[contact][
                    kpoint_ind, :, global_energy_ind
                ] = tmp

        # TODO: bring back the spill over error.

        self.bare_system_matrix.free_data()

    @profiler.profile("QTBM: Compute observables", level="default")
    def _compute_observables(
        self,
        phi: NDArray,
        local_energy_ind: int,
        global_energy_ind: int,
        obc_results: dict[QTBMContact, OBCResult],
        kpoint: float,
        kpoint_ind: int,
    ):
        r"""Computes transport observables.

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
        obc_results : dict[QTBMContact, OBCResult]
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

        contact_currents = {}

        # Compute the current from all the k dependent transmissions
        for (
            contact_in,
            contact_out,
        ), transmission in self.observables.transmissions.items():
            mu_in = contact_in.fermi_level - contact_in.voltage
            mu_out = contact_out.fermi_level - contact_out.voltage
            prefactor = fermi_dirac(
                self.electron_energies - mu_in,
                contact_in.temperature,
            ) - fermi_dirac(
                self.electron_energies - mu_out,
                contact_out.temperature,
            )

            contact_currents[contact_in, contact_out] = -(
                xp.sum(
                    xp.trapezoid(
                        prefactor * transmission,
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.device.num_kpoints
                * (2 * e / h)
            )

        return contact_currents

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
                    (
                        xp.add.reduceat(ldos, self.device.orbital_offsets[:-1], axis=1)
                        if self.config.qtbm.atom_resolved_outputs
                        else ldos
                    ),
                )

            if self.observables.excess_electron_density is not None:
                np.save(
                    f"{output_dir}/excess_electron_density.npy",
                    (
                        xp.add.reduceat(
                            self.observables.excess_electron_density,
                            self.device.orbital_offsets[:-1],
                        )
                        if self.config.qtbm.atom_resolved_outputs
                        else self.observables.excess_electron_density
                    ),
                )
            if self.observables.excess_hole_density is not None:
                np.save(
                    f"{output_dir}/excess_hole_density.npy",
                    (
                        xp.add.reduceat(
                            self.observables.excess_hole_density,
                            self.device.orbital_offsets[:-1],
                        )
                        if self.config.qtbm.atom_resolved_outputs
                        else self.observables.excess_hole_density
                    ),
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
        electron_density = xp.zeros(
            (self.device.num_orbitals, self.electron_energies.size)
        )
        hole_density = xp.zeros((self.device.num_orbitals, self.electron_energies.size))
        for contact, ldos in self.observables.electron_ldos.items():
            mu = contact.fermi_level - contact.voltage
            occupancy = fermi_dirac(
                self.electron_energies - mu,
                contact.temperature,
            )

            electron_density += occupancy * ldos.mean(axis=0) * 2  # Spin
            hole_density += (1 - occupancy) * ldos.mean(axis=0) * 2  # Spin

        # Find the reference contact mid-gap energy to separate
        # electrons and holes.
        for contact in self.device.contacts:
            if contact.voltage == 0:
                mid_gap_energy = contact.mid_gap_energy
                break
        else:  # Did not break, no reference contact found
            raise ValueError(
                "No reference contact with zero voltage found to determine mid-gap energy."
            )

        mid_gap_energy = self.device.potential + mid_gap_energy

        mask = self.electron_energies > mid_gap_energy[:, None]
        electron_density[~mask] = 0
        hole_density[mask] = 0

        excess_electron_density = xp.trapezoid(
            electron_density, self.electron_energies, axis=1
        )
        excess_hole_density = xp.trapezoid(hole_density, self.electron_energies, axis=1)

        return (
            excess_electron_density,
            excess_hole_density,
        )

    def set_potential(self, potential: NDArray):
        """Sets the potential for the QTBM calculation.

        This method can be used to update the potential for
        self-consistent calculations.

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

        self.device.potential = potential

    def get_charge_density(self) -> NDArray:
        """Gets the charge density from the QTBM calculation.

        This method integrates the local density of states to obtain the
        charge density. This is typically used in self-consistent
        calculations where the charge density is needed to update the
        potential.

        Returns
        -------
        charge_density : NDArray
            The computed charge density for the device.

        """
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

        for kpoint_ind, kpoint in enumerate(self.device.kpoints):
            if comm.rank == 0:
                print(
                    f"Processing k-point {kpoint_ind+1} of {self.device.num_kpoints}",
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

                    for energy_ind, energy in enumerate(energy_batch):

                        self.bare_system_matrix.allocate_data()
                        self._assemble_bare_system_matrix(kpoint, energy)

                        # Compute the boundary self-energy and injection vector.
                        obc_results = {}
                        free_mempool()

                        with profiler.profile_range(
                            label="QTBM: Boundary conditions", level="default"
                        ):
                            for contact in self.device.contacts:
                                obc_results[contact] = contact.compute_boundary(
                                    self.bare_system_matrix,
                                    list(kpoint * 2 * np.pi),
                                    return_modes_only=self.low_rank_obc,
                                )

                        # TODO: Only assemble the local part of the RHS
                        rhs = self._assemble_rhs(obc_results, energy_ind)
                        rhs = rhs[
                            self.device.offsets[comm.block.rank] : self.device.offsets[
                                comm.block.rank + 1
                            ],
                            :,
                        ]

                        if rhs.size == 0:
                            # No modes are injected at this energy, so we
                            # can skip the calculation.
                            continue

                        if not self.low_rank_obc:
                            # Assemble the full system matrix with the
                            # self-energy contributions from the
                            # contacts.
                            self.system_matrix.allocate_data()
                            self.system_matrix.data = 0.0
                            # The `add_` method implicitly unsymmetrizes
                            # the bare matrix. Up for discussion if this
                            # should be more explicit.
                            self.system_matrix.add_(self.bare_system_matrix)
                            self.bare_system_matrix.free_data()

                            self._add_sigma_to_system_matrix(
                                system_matrix=self.system_matrix,
                                obc_results=obc_results,
                                energy_ind=energy_ind,
                                sigma_update_indices=self.sigma_update_indices,
                            )

                        system_matrix = (
                            self.system_matrix
                            if not self.low_rank_obc
                            else self.bare_system_matrix
                        )

                        # Solve for the wavefunction
                        # NOTE: Initially just allgather for testing.
                        data = comm.block.all_gather_v(system_matrix.data, axis=0)
                        col_indices = comm.block.all_gather_v(
                            system_matrix.col_ind, axis=0
                        )
                        row_indices = comm.block.all_gather_v(
                            system_matrix.row_ind
                            + system_matrix.row_offsets[comm.block.rank],
                            axis=0,
                        )
                        system_matrix = sparse.coo_matrix(
                            (data, (row_indices, col_indices)),
                            shape=(system_matrix.shape[-1], system_matrix.shape[-1]),
                        )
                        system_matrix = system_matrix.tocsr()

                        rhs = comm.block.all_gather_v(rhs, axis=0)

                        phi = self.solver.solve(
                            system_matrix,
                            rhs,
                            reuse_analysis=True,
                            reuse_factorization=False,
                        )
                        phi = phi[
                            self.device.offsets[comm.block.rank] : self.device.offsets[
                                comm.block.rank + 1
                            ],
                            :,
                        ]

                        if self.low_rank_obc:
                            phi = self._recover_full_rank_wavefunction(
                                phi, obc_results, energy_ind
                            )

                        if not self.low_rank_obc:
                            self.system_matrix.free_data()

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

        self.observables.contact_currents = self._compute_current()
        (
            self.observables.excess_electron_density,
            self.observables.excess_hole_density,
        ) = self._compute_excess_charge_densities()

        self._write_outputs()

        if comm.rank == 0:
            print("QTBM calculation complete", flush=True)
