# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the SCBA contact class."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.boundary_conditions import lyapunov, obc
from qttools.comm import comm
from qttools.nevp import NEVP, Beyn, Full
from quatrex.bandstructure.contact import contact_band_structure
from quatrex.contact.base import BaseContact
from quatrex.core.config import (
    ContactConfig,
    LyapunovComputeConfig,
    LyapunovConfig,
    NEVPConfig,
    OBCConfig,
)


def order_vector(
    vector: NDArray,
    order: str | NDArray | None,
):
    """Reorders the elements of the given vector according to the
    specified order.

    Parameters
    ----------
    vector : NDArray
        The vector to reorder.
    order : str | NDArray | None
        The order in which to reorder the elements. The only supported
        string is "reverse", which reverses the order of the elements.

    Returns
    -------
    NDArray
        The reordered vector.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return vector
    if order == "reverse":
        return xp.flip(vector, axis=-1)
    return vector[..., order]


def order_block(
    block: NDArray,
    order: str | NDArray | None,
) -> NDArray:
    """Reorders the blocks of the given matrix according to the
    specified order.

    Parameters
    ----------
    block : NDArray
        The matrix block to reorder.
    order : str | NDArray | None
        The order in which to reorder the blocks. The only supported
        string is "reverse", which reverses the order of the blocks.

    Returns
    -------
    NDArray
        The reordered matrix block.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return block
    if order == "reverse":
        return xp.flip(block, axis=(-2, -1))
    return block[..., :, order][..., order, :]


def get_inverse_order(
    order: str | NDArray | None,
) -> str | NDArray | None:
    """Computes the inverse of the given order.

    Parameters
    ----------
    order : str | NDArray | None

    Returns
    -------
    str | NDArray | None
        The inverse order, or None if the input order is None.

    """
    # TODO: This should be only called once inside
    # the contact.

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return None
    if order == "reverse":
        return "reverse"
    return xp.argsort(order)


class SCBAContact(BaseContact):
    """Class representing a contact for SCBA calculations.

    Parameters
    ----------
    device : BaseDevice
        The device object to which this contact is attached. Contains
        the Hamiltonian, overlap matrices, and atomic structure
        information.
    contact_config : ContactConfig
        The configuration object containing the contact settings such as
        lattice vectors, origin, transport direction, and Fermi level
        information.
    sparsity_pattern : sparse.spmatrix
        The sparsity pattern of the device Hamiltonian, used to identify
        the contact orbitals and their connectivity.

    Attributes
    ----------
    device : BaseDevice
        The device object to which this contact is attached.
    name : str
        The contact identifier.
    transport_direction : int
        Transport direction index (0, 1, or 2).
    unit_cell_orbital_indices : dict
        Dict of orbital indices for each contact cell indexed by (i, j,
        k) tuples.
    origin_key : tuple[int, int, int]
        The key corresponding to the origin cell in the
        unit_cell_orbital_indices.
    transverse_repetition_grid: NDArray
        Number of periodic repetitions in the two transverse directions.
    transport_repetitions : int
        Number of repetitions needed in transport direction for
        convergence.
    orbital_indices : NDArray
        Flattened array of orbital indices for the contact, sorted first
        in transport direction, then in transverse directions.
    orbital_indices_per_layer : list[NDArray]
        List of orbital indices for each layer in the transport
        direction, sorted first in transverse directions, then in
        transport direction.
    transverse_to_transport_indices : NDArray
        Indices to reorder the coupling matrix from transverse-first to
        transport-first ordering.
    fermi_level : float
        Fermi level of the contact in eV.
    mid_gap_energy : float
        Mid-gap energy of the contact in eV.
    conduction_band_edge : float
        Energy of the conduction band edge in eV.
    voltage : float
        Voltage applied to the contact in V.
    temperature : float
        Temperature of the contact in K.
    diagonal_inds : tuple[int, int]
        Tuple of indices corresponding to the diagonal block of the
        contact in the Hamiltonian.
    upper_inds : tuple[int, int]
        Tuple of indices corresponding to the upper block of the contact
        in the Hamiltonian.
    order : str | None
        Order of the contact indices, either None or "reverse" for
        ascending or descending order, respectively.
    owning_rank : int
        The rank of the block comm process that owns the contact indices.
    obc_solvers : dict[str, obc.OBCSystem]
        Dictionary of OBC solvers for different subsystems (e.g.,
        electron, phonon, photon, etc.).
    lyapunov_solvers : dict[str, lyapunov.LyapunovSystem]
        Dictionary of Lyapunov solvers for different subsystems (e.g.,
        phonon, photon, etc.).

    """

    def __init__(
        self,
        device,
        contact_config: ContactConfig,
        sparsity_pattern: sparse.spmatrix,
    ):
        super().__init__(device, contact_config, sparsity_pattern)

        # Determine to which blocks the indices correspond to.
        self._analyze_contact_indices()

        self.obc_solvers = {}
        self.lyapunov_solvers = {}

        config = self.device.config

        # TODO: Make a compute nevp compute config per subsystem.
        # TODO: Make an obc config per contact.
        self.obc_solvers["electron"] = self._configure_obc(
            config.electron.obc,
            config.compute.nevp,
        )

        if config.scba.coulomb_screening:
            self.obc_solvers["coulomb_screening"] = self._configure_obc(
                config.coulomb_screening.obc,
                config.compute.nevp,
                block_sections=self.device.coulomb_num_connected_blocks
                * self.transport_repetitions,
            )
            self.lyapunov_solvers["coulomb_screening"] = self._configure_lyapunov(
                config.coulomb_screening.lyapunov,
                config.compute.lyapunov,
            )

        # TODO: Allocate OBC solver for the other systems (photons /
        # phonons) when needed.

    def _analyze_from_unit(self):
        """Map the contact indices to the corresponding blocks."""
        contact_name = self.name
        if contact_name == "left":
            self.diagonal_inds = (0, 0)
            self.upper_inds = (0, 1)
            self.order = None
            self.owning_rank = 0
        elif contact_name == "right":
            n = self.device.hamiltonians.num_local_blocks - 1
            m = n - 1
            self.diagonal_inds = (n, n)
            self.upper_inds = (n, m)
            self.order = "reverse"
            self.owning_rank = comm.block.size - 1

    def _analyze_real_space(self):
        """Map the contact indices to the corresponding blocks."""
        ny, nz = self.transverse_repetition_grid

        contact_size = sum(
            len(self.unit_cell_orbital_indices[i, j, k])
            for i, j, k in np.ndindex(self.transport_repetitions, ny, nz)
        )
        if np.min(self.unit_cell_orbital_indices[0, 0, 0]) == 0:
            if contact_size != self.device.block_sizes[0]:
                raise ValueError(
                    "The contact indices do not match the first block of the Hamiltonian.\n"
                    f"Contact size: {contact_size}, first block size: {self.device.block_sizes[0]}"
                )
        elif (
            np.max(self.unit_cell_orbital_indices[0, 0, 0])
            == self.device.hamiltonians.shape[-1] - 1
        ):
            if contact_size != self.device.block_sizes[-1]:
                raise ValueError(
                    "The contact indices do not match the last block of the Hamiltonian."
                    f"Contact size: {contact_size}, last block size: {self.device.block_sizes[-1]}"
                )
        else:
            raise ValueError(
                "The contact indices cannot be matched\n"
                "since they do not correspond to either the first"
                "or last block of the Hamiltonian."
            )

        indices = np.concatenate(
            [
                self.unit_cell_orbital_indices[i, j, k]
                for i, j, k in np.ndindex(self.transport_repetitions + 1, ny, nz)
            ]
        )

        # Check that the indices are contiguous i.e. no gaps
        sorted_indices = np.sort(indices)
        if not np.all(np.diff(sorted_indices) == 1):
            raise ValueError(
                "The contact indices are not contiguous.\n"
                "This is currently not supported for real-space contacts in SCBA."
            )

        # To correctly subslice, they also need to be contiguous in each layer.
        for i, j, k in np.ndindex(self.transport_repetitions + 1, ny, nz):
            sorted_indices = np.sort(self.unit_cell_orbital_indices[i, j, k])
            if not np.all(np.diff(sorted_indices) == 1):
                raise ValueError(
                    "The contact indices are not contiguous.\n"
                    "This is currently not supported for real-space contacts in SCBA."
                )

        # TODO: Currently we do not allow orders except None and "reverse"
        # i.e. with real space only left and right are supported
        # where the hamiltonian is already correctly sorted.

        # TODO: These checks are not robust and should be improved
        # to handle more general cases.
        if np.min(self.unit_cell_orbital_indices[0, 0, 0]) == 0:
            last_value = 0
            for i, j, k in np.ndindex(self.transport_repetitions + 1, ny, nz):
                if last_value > np.min(self.unit_cell_orbital_indices[i, j, k]):
                    raise ValueError(
                        "The contact indices are not sorted in ascending order.\n"
                        "This is currently not supported for real-space contacts in SCBA."
                    )
                last_value = np.max(self.unit_cell_orbital_indices[i, j, k])

            self.order = None
            self.diagonal_inds = (0, 0)
            self.upper_inds = (0, 1)
            self.owning_rank = 0

        elif (
            np.max(self.unit_cell_orbital_indices[0, 0, 0])
            == self.device.hamiltonians.shape[-1] - 1
        ):
            last_value = self.device.hamiltonians.shape[-1] - 1
            for i, j, k in np.ndindex(self.transport_repetitions + 1, ny, nz):
                if last_value < np.max(self.unit_cell_orbital_indices[i, j, k]):
                    raise ValueError(
                        "The contact indices are not sorted in descending order.\n"
                        "This is currently not supported for real-space contacts in SCBA."
                    )
                last_value = np.min(self.unit_cell_orbital_indices[i, j, k])

            n = self.device.hamiltonians.num_local_blocks - 1
            m = n - 1
            self.diagonal_inds = (n, n)
            self.upper_inds = (n, m)
            self.order = "reverse"
            self.owning_rank = comm.block.size - 1

        # TODO validate that contacts do not span multiple ranks

    def _analyze_contact_indices(self):
        """Map the contact indices to the corresponding blocks."""
        if self.contact_config._contact_finder_method == "from_unit":
            self._analyze_from_unit()
        elif self.contact_config._contact_finder_method == "real_space":
            self._analyze_real_space()

    def _configure_obc(
        self,
        obc_config: OBCConfig,
        nevp_config: NEVPConfig,
        block_sections: int | None = None,
    ) -> obc.OBCSystem:
        """Configures the OBC solver.

        Parameters
        ----------
        obc_config : OBCConfig
            Configuration object containing OBC algorithm settings
            including solver type, convergence parameters, and numerical
            options.
        nevp_config : NEVPConfig
            Configuration object containing NEVP solver settings
            including solver type and algorithm-specific parameters.
        block_sections : int | None, optional
            Number of block sections to use in the OBC solver. This is
            only needed for the Coulomb subsystem where the number of
            sections need to be multiplied by the bandwidth increase.

        Returns
        -------
        obc_solver: obc.OBCSystem
            Configured OBC system ready for boundary condition
            calculations.

        """
        if obc_config.algorithm == "sancho-rubio":
            obc_solver = obc.SanchoRubio(
                obc_config.max_iterations, obc_config.convergence_tol
            )

        elif obc_config.algorithm == "spectral":
            block_sections = (
                self.transport_repetitions if block_sections is None else block_sections
            )

            nevp = self._configure_nevp(obc_config, nevp_config, block_sections)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                min_propagation=obc_config.min_propagation,
                residual_tolerance=obc_config.residual_tolerance,
                residual_normalization=obc_config.residual_normalization,
                eta_decay=obc_config.eta_decay,
            )

        else:
            raise NotImplementedError(
                f"OBC algorithm '{obc_config.algorithm}' not implemented."
            )

        # NOTE: wrapper handles if the memoizer is off
        obc_solver = obc.OBCSystem(
            boundary_solver=obc_solver,
            num_ref_iterations=obc_config.memoizer.num_ref_iterations,
            relative_tol=obc_config.memoizer.relative_tol,
            absolute_tol=obc_config.memoizer.absolute_tol,
            warning_threshold=obc_config.memoizer.warning_threshold,
            memoization_mode=obc_config.memoizer.mode,
            agreement_threshold=obc_config.memoizer.agreement_threshold,
        )

        return obc_solver

    def _configure_nevp(
        self,
        obc_config: OBCConfig,
        nevp_config: NEVPConfig,
        block_sections: int,
    ) -> NEVP:
        """Configures the Nonlinear Eigenvalue Problem (NEVP) solver.

        Parameters
        ----------
        obc_config : OBCConfig
            Configuration object containing NEVP solver settings
            including solver type and algorithm-specific parameters.
        nevp_config : NEVPConfig
            Configuration object containing NEVP solver settings
            including solver type and algorithm-specific parameters.
        block_sections : int
            Number of block sections to use in the OBC solver.

        Returns
        -------
        NEVP
            Configured NEVP solver ready for eigenvalue calculations.

        """
        if obc_config.nevp_solver == "beyn":
            return Beyn(
                r_o=obc_config.r_o ** (1 / block_sections),
                r_i=obc_config.r_i ** (1 / block_sections),
                m_0=obc_config.m_0,
                num_quad_points=obc_config.num_quad_points,
                num_threads_contour=nevp_config.num_threads_contour,
                eig_compute_location=nevp_config.eig_compute_location,
                project_compute_location=nevp_config.project_compute_location,
                use_qr=nevp_config.use_qr,
                contour_batch_size=nevp_config.contour_batch_size,
                use_pinned_memory=nevp_config.use_pinned_memory,
            )
        if obc_config.nevp_solver == "full":
            return Full(
                eig_compute_location=nevp_config.eig_compute_location,
                use_pinned_memory=nevp_config.use_pinned_memory,
                reduce=nevp_config.reduce_sparsity,
            )

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def _configure_lyapunov(
        self,
        lyapunov_config: LyapunovConfig,
        lyapunov_compute_config: LyapunovComputeConfig,
    ) -> lyapunov.LyapunovSystem:
        """Configures the Lyapunov solver from the config.

        Parameters
        ----------
        lyapunov_config : LyapunovConfig
            The Lyapunov configuration.
        lyapunov_compute_config : LyapunovComputeConfig
            The Lyapunov compute configuration.

        Returns
        -------
        lyapunov.LyapunovSystem
            The configured Lyapunov solver.

        """
        if lyapunov_config.algorithm == "spectral":
            lyapunov_solver = lyapunov.Spectral(
                num_ref_iterations=lyapunov_config.num_ref_iterations,
                eig_compute_location=lyapunov_compute_config.eig_compute_location,
                use_pinned_memory=lyapunov_compute_config.use_pinned_memory,
            )
        elif lyapunov_config.algorithm == "doubling":
            lyapunov_solver = lyapunov.Doubling(
                max_iterations=lyapunov_config.max_iterations,
                convergence_rel_tol=lyapunov_config.relative_tol,
                convergence_abs_tol=lyapunov_config.absolute_tol,
            )
        else:
            raise NotImplementedError(
                f"Lyapunov algorithm '{lyapunov_config.algorithm}' not implemented."
            )

        lyapunov_system_reducer = lyapunov.LyapunovSystemReducer(
            reduce_sparsity=lyapunov_config.reduce_sparsity,
            assume_constant_sparsity=lyapunov_config.assume_constant_sparsity,
        )

        # NOTE: wrapper handles if the memoizer is off
        lyapunov_solver = lyapunov.LyapunovSystem(
            boundary_solver=lyapunov_solver,
            system_reducer=lyapunov_system_reducer,
            num_ref_iterations=lyapunov_config.memoizer.num_ref_iterations,
            relative_tol=lyapunov_config.memoizer.relative_tol,
            absolute_tol=lyapunov_config.memoizer.absolute_tol,
            warning_threshold=lyapunov_config.memoizer.warning_threshold,
            memoization_mode=lyapunov_config.memoizer.mode,
            agreement_threshold=lyapunov_config.memoizer.agreement_threshold,
        )

        return lyapunov_solver

    def compute_contact_bandstructure(
        self,
        kpoints_transport: NDArray,
    ) -> NDArray:
        """Computes the band structure for the contact along the
        transport direction.

        Parameters
        ----------
        kpoints_transport : NDArray
            The k-points along the transport direction.

        Returns
        -------
        e_k : NDArray
            The eigenvalues for the contact band structure.

        """
        h_xx = (
            self.device.hamiltonians.blocks[*self.upper_inds[::-1]],
            self.device.hamiltonians.blocks[*self.diagonal_inds],
            self.device.hamiltonians.blocks[*self.upper_inds],
        )

        if self.device.overlap_matrices is not None:
            s_xx = (
                self.device.overlap_matrices.blocks[*self.upper_inds[::-1]],
                self.device.overlap_matrices.blocks[*self.diagonal_inds],
                self.device.overlap_matrices.blocks[*self.upper_inds],
            )
        else:
            s_xx = None

        e_k = contact_band_structure(kpoints_transport, h_xx, s_xx)
        e_k = e_k.reshape(e_k.shape[0], -1, e_k.shape[-1])

        return e_k
