# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the Coulomb screening solver class."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse, DSDBCOO
from qttools.datastructures.dsdbsparse import _DStackView
from qttools.datastructures.routines import bd_matmul, bd_sandwich
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler
from qttools.toeplitz.toeplitz import (
    expand_periodic_superblocks,
    get_periodic_superblocks,
    homogenize,
)
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.solvers_utils import get_batches
from qttools.utils.sparse_utils import product_sparsity_pattern_dsdbsparse
from quatrex.core.config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import compute_num_connected_blocks
from quatrex.device.contact import get_inverse_order, order_block
from quatrex.device.inputs import assemble_matrix
from qttools.kernels.datastructure import dsdbcoo_kernels

profiler = Profiler()


def _spin_extended_sparsity_pattern(
    sparsity_pattern: sparse.coo_matrix
) -> sparse.coo_matrix:
    """Extends the sparsity pattern for the spin resolved case.
    
    Each non-zero element of sparsity_pattern is turned into a 2x2 spin block
    (i,j) -> (2i,2j), (2i+1,2j), (2i,2j+1), (2i+1,2j+1)

    Parameters
    ----------
    sparsity_pattern : sparse.coo_matrix
        The sparsity pattern of the non spin-resolved case

    Returns
    -------
    double_sparsity_pattern : sparse.coo_matrix
        The extended sparsity pattern 
    """

    col = sparsity_pattern.col
    row = sparsity_pattern.row
    size_extended = len(col) * 4

    col_extended = np.zeros(shape=(size_extended), dtype = np.int32)
    row_extended = np.zeros(shape=(size_extended), dtype = np.int32)

    idx = np.arange(0, size_extended, 4)
    col_extended[idx] = 2 * col
    row_extended[idx] = 2 * row

    idx = np.add(idx, 1)
    col_extended[idx] = np.add(2 * col, 1)
    row_extended[idx] = np.add(2 * row, 0)

    idx = np.add(idx, 1)
    col_extended[idx] = np.add(2 * col, 0)
    row_extended[idx] = np.add(2 * row, 1)

    idx = np.add(idx, 1)
    col_extended[idx] = np.add(2 * col, 1)
    row_extended[idx] = np.add(2 * row, 1)

    data_extended = np.ones(shape=(len(col) * 4), dtype = np.float32)

    spin_extended_sparsity_pattern = sparse.coo_matrix((data_extended, (row_extended, col_extended)), dtype=np.float32)

    return spin_extended_sparsity_pattern

def _cast_to_spin_matrix(
    A: DSDBSparse, B: DSDBSparse | None = None
) -> DSDBSparse:
    """Casts A and B to a matrix of size 2Nx2N the following way:

    |A_11 B_11|A_12 B_12| ...
    |B_11 A_11|B_12 A_12|
    |A_21 B_21|A_22 B_22|
    |B_21 A_21|B_22 A_22|
    ...

    Parameters
    ----------
    A : DSDBSparse
        The matrix to be cast to a bigger matrix

    B : DSDBSparse
        optional, if None all B_ij = 0

    Returns
    -------
    spin_matrix : DSBSSparse
        A matrix of size 2Nx2N

    """
    if B is not None and A.shape != B.shape:
        raise AssertionError("A and B must have the same shape")

    if B is not None and not np.array_equal(A.block_sizes, B.block_sizes):
        raise AssertionError("A and B must have the same block sizes")
    
    data_A = np.concatenate((A.data, A.data), axis = 1)
    rows_A = np.concatenate((2*A.rows, np.add(2*A.rows, 1)))
    cols_A = np.concatenate((2*A.cols, np.add(2*A.cols, 1)))

    if B is not None:
        data_B = np.concatenate((B.data, B.data), axis = 1)
        rows_B = np.concatenate((np.add(2*B.rows, 1), 2*B.rows))
        cols_B = np.concatenate((2*B.cols, np.add(2*B.cols, 1)))
    else:
        data_B = np.zeros(shape=data_A.shape)
        rows_B = np.concatenate((np.add(2*A.rows, 1), 2*A.rows))
        cols_B = np.concatenate((2*A.cols, np.add(2*A.cols, 1)))

    _data = np.concatenate((data_A, data_B), axis = 1)
    _rows = np.concatenate((rows_A, rows_B))
    _cols = np.concatenate((cols_A, cols_B))
    
    block_sizes = A.block_sizes * 2

    block_sort_index = dsdbcoo_kernels.compute_block_sort_index(
        _rows, _cols, block_sizes
    )
    data = _data[:,block_sort_index]
    rows = _rows[block_sort_index]
    cols = _cols[block_sort_index]

    spin_matrix = DSDBCOO(A.dtype, rows, cols, block_sizes, A.local_stack_shape, A.global_stack_shape, A.symmetry)
    spin_matrix.allocate_data()
    spin_matrix.data = data
    
    return spin_matrix

def _cast_from_spin_matrix(
    A: DSDBSparse, out: DSDBSparse | None = None
) -> DSDBSparse:
    """ Takes a spin resolved matrix A of size 2Nx2N and returns the elements 
    corresponding to (up, up) as a NxN matrix

    Parameters
    ----------
    A : DSDBSparse
        spin resolved matrix

    out : DSDBSparse
        The (up, up) components of A

    """

    _rows = A.rows
    _cols = A.cols

    mask = np.argwhere(np.logical_and(_rows % 2, _cols % 2)).flatten()

    data = A.data[:, mask]
    rows = np.right_shift(np.subtract(_rows[mask], 1), 1)
    cols = np.right_shift(np.subtract(_cols[mask], 1), 1)

    if out is not None:
        out.data = data
        out.rows = rows
        out.cols = cols
    else:
        block_sizes = A.block_sizes / 2
        out = DSDBCOO(A.dtype, rows, cols, block_sizes, A.local_stack_shape, A.global_stack_shape, A.symmetry)
        out.allocate_data()
        out.data = data

    return out

def _compute_sparsity_pattern(
    *matrices: DSDBSparse, dtype: xp.dtype = None
) -> sparse.coo_matrix:
    """Computes the sparsity pattern of the product of several DSDBSparse matrices."""
    num_blocks = matrices[0].num_blocks
    local_blocks, _ = get_section_sizes(num_blocks, comm.block.size)
    start_block = sum(local_blocks[: comm.block.rank])
    end_block = start_block + local_blocks[comm.block.rank]
    rows, cols = product_sparsity_pattern_dsdbsparse(
        *matrices, start_block=start_block, end_block=end_block, spillover=True
    )
    shape = matrices[0].shape[-2:]
    dtype = dtype or matrices[0].dtype
    return sparse.coo_matrix(
        (xp.ones_like(rows), (rows, cols)), shape=shape, dtype=dtype
    )


class CoulombScreeningSolver(SubsystemSolver):
    """Solves the dynamics of the screened Coulomb interaction.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    energies : NDArray
        The energies at which to solve.

    """

    system = "coulomb_screening"

    def __init__(
        self,
        config: QuatrexConfig,
        coulomb_matrix: DSDBSparse,
        energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config, energies)

        self.spin = config.coulomb_screening.spin

        # read exchange matrix v_ap
        if self.spin:
            # Load the Exchange matrix.
            exchange_matrix, __ = assemble_matrix(
                config=config,
                matrix_name="exchange_matrix",
                sparsity_pattern=sparsity_pattern,
                shift_kpoints=True,
            )

            # Make sure the exchange matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if exchange_matrix.symmetry is None:
                exchange_matrix.symmetrize("hermitian")
            exchange_matrix._data /= config.coulomb_screening.epsilon_r

        if self.spin:
            self.coulomb_matrix = _cast_to_spin_matrix(coulomb_matrix, exchange_matrix)
        else:
            self.coulomb_matrix = coulomb_matrix

        self.small_block_sizes = self.coulomb_matrix.block_sizes

        if self.spin:
            sparsity_pattern = _spin_extended_sparsity_pattern(sparsity_pattern)


        self.num_connected_blocks = config.coulomb_screening.num_connected_blocks
        if self.num_connected_blocks == "auto":
            self.num_connected_blocks = compute_num_connected_blocks(
                sparsity_pattern, self.small_block_sizes
            )

        if len(self.small_block_sizes) % self.num_connected_blocks != 0:
            # Not implemented yet.
            raise ValueError(
                f"Number of blocks must be divisible by {self.num_connected_blocks}."
            )

        self.block_sizes = (
            self.small_block_sizes[
                : len(self.small_block_sizes) // self.num_connected_blocks
            ]
            * self.num_connected_blocks
        )
        # Check that the provided block sizes match the coulomb matrix.
        if self.small_block_sizes.sum() != self.coulomb_matrix.shape[-2]:
            raise ValueError(
                "Block sizes do not match Coulomb matrix. "
                f"{self.small_block_sizes.sum()} != {self.coulomb_matrix.shape[-2]}"
            )

        # The dummy dsbsparse is only used to compute the sparsity
        # pattern of the system matrix and the l_lesser and l_greater
        # matrices.
        dummy_dsbsparse = config.compute.dsdbsparse_type.from_sparray(
            sparray=sparsity_pattern.astype(xp.float32),
            block_sizes=self.small_block_sizes,
            global_stack_shape=(comm.size,),
            symmetry="hermitian" if config.scba.symmetric else None,
        )
        v_times_p_sparsity_pattern = _compute_sparsity_pattern(
            dummy_dsbsparse, dummy_dsbsparse, dtype=xp.float32
        )

        # Allocate memory for the System matrix (1 - V @ P).
        kpoint_grid = config.device.kpoint_grid
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            sparray=v_times_p_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            allocate=False,
        )

        # Explicitely try to free the memory for the sparsity pattern.
        del v_times_p_sparsity_pattern

        l_sparsity_pattern = _compute_sparsity_pattern(
            dummy_dsbsparse,
            dummy_dsbsparse,
            dummy_dsbsparse,
            dtype=xp.float32,
        )
        del dummy_dsbsparse

        # Allocate memory for the L_lesser and L_greater matrices.
        self.l_lesser = config.compute.dsdbsparse_type.from_sparray(
            sparray=l_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry="skew-hermitian" if config.scba.symmetric else None,
            allocate=False,
        )
        self.l_greater = config.compute.dsdbsparse_type.empty_like(self.l_lesser)
        # Explicitely try to free the memory for the sparsity pattern.
        del l_sparsity_pattern

        # Allocate object for the retarded polarization.
        # This is only used as a temporary assembling the full retarded
        # polarization before multiplying with the Coulomb matrix.
        # This is simpler in case of domain distributed solver.
        # Otherwise, it would be more memory efficient to directly assemble
        # when doing the product with the Coulomb matrix, but the life time
        # is not during the peak (quadratic solve).
        self.p_retarded = config.compute.dsdbsparse_type.from_sparray(
            sparray=sparsity_pattern.astype(xp.complex128),
            block_sizes=self.small_block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            allocate=False,
        )

        self.dos_peak_limit = config.coulomb_screening.dos_peak_limit

        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)

        self.block_sections = config.coulomb_screening.obc.block_sections
        if (
            self.block_sections % self.num_connected_blocks != 0
            and self.block_sections != 1
        ):
            raise ValueError(
                f"Block sections must be divisible by {self.num_connected_blocks} or equal to 1."
            )
        if self.block_sections == 1:
            self.small_block_sections = 1
        else:
            self.small_block_sections = self.block_sections // self.num_connected_blocks

        self.flatband = config.electron.flatband
        self.solve_call_count = 0
        self.filtering_iteration_limit = (
            config.coulomb_screening.filtering_iteration_limit
        )

        self.max_batch_size = config.coulomb_screening.max_batch_size

    def _set_block_sizes(self, block_sizes: NDArray) -> None:
        """Sets the block sizes of all matrices.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        self.system_matrix.block_sizes = block_sizes
        self.l_lesser.block_sizes = block_sizes
        self.l_greater.block_sizes = block_sizes

    def _compute_contact_obc(
        self,
        contact: str,
        p_lesser: DSDBSparse | _DStackView,
        p_greater: DSDBSparse | _DStackView,
        p_retarded: DSDBSparse | _DStackView,
        diagonal_inds: tuple,
        upper_inds: tuple,
        order: str | NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Computes the OBC for a specific contact.

        Parameters
        ----------
        contact : str
            The contact for which to compute the OBC.
            Used for profiling and caching purposes.
        p_lesser : DSDBSparse | _DStackView
            The lesser polarization.
        p_greater : DSDBSparse | _DStackView
            The greater polarization.
        p_retarded : DSDBSparse | _DStackView
            The retarded polarization.
        diagonal_inds : tuple
            The indices of the diagonal blocks corresponding to the contact.
        upper_inds : tuple
            The indices of the upper off-diagonal blocks corresponding to the contact.
        order : str | NDArray | None, optional
            The permutation of the blocks to achieve the same order as the canonical left contact.
            If None, the left contact order is assumed.
            Instead of an explicit permutation, the string "reverse" can be passed
            to reverse the order of the blocks, which is equivalent to the right contact order.

        Returns
        -------
        obc_retarded : NDArray
            The retarded OBC for the contact.
        obc_lesser : NDArray
            The lesser OBC for the contact.
        obc_greater : NDArray
            The greater OBC for the contact.

        """

        inverse_order = get_inverse_order(order)

        with profiler.profile_range(
            label=f"CoulombScreeningSolver: Get OBCR blocks {contact}",
            level="default",
            comm=comm.stack,
        ):

            p_retarded_10, p_retarded_00, p_retarded_01 = expand_periodic_superblocks(
                a_ji=order_block(p_retarded.blocks[*upper_inds[::-1]], order),
                a_ii=order_block(p_retarded.blocks[*diagonal_inds], order),
                a_ij=order_block(p_retarded.blocks[*upper_inds], order),
                block_sections=self.small_block_sections,
                repetitions=self.num_connected_blocks,
            )
            v_10, v_00, v_01 = expand_periodic_superblocks(
                a_ji=order_block(self.coulomb_matrix.blocks[*upper_inds[::-1]], order),
                a_ii=order_block(self.coulomb_matrix.blocks[*diagonal_inds], order),
                a_ij=order_block(self.coulomb_matrix.blocks[*upper_inds], order),
                block_sections=self.small_block_sections,
                repetitions=self.num_connected_blocks,
            )

            # NOTE: The blocks need to be recomputed here
            # similarly to the spillover correction because
            # the periodic superblocks are used for the OBC computation,
            m_00 = (
                xp.eye(p_retarded_00.shape[-1])
                - v_10 @ p_retarded_01
                - v_00 @ p_retarded_00
                - v_01 @ p_retarded_10
            )
            m_01 = -v_00 @ p_retarded_01 - v_01 @ p_retarded_00
            m_10 = -v_10 @ p_retarded_00 - v_00 @ p_retarded_10

        with profiler.profile_range(
            label=f"CoulombScreeningSolver: OBCR {contact}",
            level="default",
            comm=comm.stack,
        ):

            x_00, *__ = self.obc((m_00, m_01, m_10), contact="W: " + contact)

            m_10_x_00 = m_10 @ x_00
            obc_retarded = m_10_x_00 @ m_01

        with profiler.profile_range(
            label=f"CoulombScreeningSolver: Get Lyapunov blocks {contact}",
            level="default",
            comm=comm.stack,
        ):

            def _get_l_superblocks(p_):
                p_10, p_00, p_01 = expand_periodic_superblocks(
                    a_ji=order_block(p_.blocks[*upper_inds[::-1]], order),
                    a_ii=order_block(p_.blocks[*diagonal_inds], order),
                    a_ij=order_block(p_.blocks[*upper_inds], order),
                    block_sections=self.small_block_sections,
                    repetitions=self.num_connected_blocks,
                )
                l_00 = (
                    v_10 @ p_00 @ v_01
                    + v_10 @ p_01 @ v_00
                    + v_00 @ p_10 @ v_01
                    + v_00 @ p_00 @ v_00
                    + v_00 @ p_01 @ v_10
                    + v_01 @ p_10 @ v_00
                    + v_01 @ p_00 @ v_10
                )
                l_01 = (
                    v_10 @ p_01 @ v_01
                    + v_00 @ p_00 @ v_01
                    + v_00 @ p_01 @ v_00
                    + v_01 @ p_10 @ v_01
                    + v_01 @ p_00 @ v_00
                )
                return l_00, l_01

            l_lesser_00, l_lesser_01 = _get_l_superblocks(p_lesser)
            l_greater_00, l_greater_01 = _get_l_superblocks(p_greater)

        with profiler.profile_range(
            label=f"CoulombScreeningSolver: Lyapunov {contact}",
            level="default",
            comm=comm.stack,
        ):
            # Compute and apply the left lesser/greater boundary self-energy.
            a_00_lesser = m_10_x_00 @ l_lesser_01
            a_00_greater = m_10_x_00 @ l_greater_01

            q_00_lesser = (
                x_00
                @ (l_lesser_00 - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2)))
                @ x_00.conj().swapaxes(-1, -2)
            )
            q_00_greater = (
                x_00
                @ (l_greater_00 - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2)))
                @ x_00.conj().swapaxes(-1, -2)
            )

            b_00 = x_00 @ m_10
            q_00 = xp.stack((q_00_lesser, q_00_greater))

            w_00, *__ = self.lyapunov((b_00, q_00), "W: " + contact)

            m_w_m = m_10 @ w_00 @ m_10.conj().swapaxes(-1, -2)

            m_w_m_lesser, m_w_m_greater = m_w_m

            obc_lesser = m_w_m_lesser - (
                a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2)
            )

            obc_greater = m_w_m_greater - (
                a_00_greater - a_00_greater.conj().swapaxes(-1, -2)
            )

        return (
            order_block(obc_retarded, inverse_order),
            order_block(obc_lesser, inverse_order),
            order_block(obc_greater, inverse_order),
        )

    @profiler.profile(label="CoulombScreeningSolver: OBC", level="default", comm=comm)
    def _compute_obc(
        self,
        p_lesser: DSDBSparse | _DStackView,
        p_greater: DSDBSparse | _DStackView,
        p_retarded: DSDBSparse | _DStackView,
        batch_slice: slice,
    ) -> None:
        """Computes open boundary conditions (OBC).

        Both the OBC for retarded and lesser/greater components are
        computed, as the former is needed for the latter. This done for
        all the contacts of the system, which are currently assumed to
        be only the left and right boundaries.

        The result of this method is that the `obc_blocks` attribute of
        the solver is filled.

        NOTE: The polarizations are passed as arguments and not the
        system matrix. This is because not the blocks of the system
        matrix are used, but fully periodic superblocks are assembled
        with the polarizations and the Coulomb matrix. In the case of no
        subdivision, the system matrix blocks could be used directly.

        Parameters
        ----------
        p_lesser : DSDBSparse | _DStackView
            The lesser polarization.
        p_greater : DSDBSparse | _DStackView
            The greater polarization.
        p_retarded : DSDBSparse | _DStackView
            The retarded polarization.
        batch_slice : slice
            The slice of the energy stack corresponding to the current batch.

        """
        if comm.block.rank == 0:
            obc_retarded, obc_lesser, obc_greater = self._compute_contact_obc(
                contact="left-" + str(batch_slice),
                p_lesser=p_lesser,
                p_greater=p_greater,
                p_retarded=p_retarded,
                diagonal_inds=(0, 0),
                upper_inds=(0, 1),
            )
            self.obc_blocks.retarded[0] = obc_retarded
            self.obc_blocks.lesser[0] = obc_lesser
            self.obc_blocks.greater[0] = obc_greater

        if comm.block.rank == comm.block.size - 1:

            n = p_retarded.num_local_blocks - 1
            m = n - 1
            obc_retarded, obc_lesser, obc_greater = self._compute_contact_obc(
                contact="right-" + str(batch_slice),
                p_lesser=p_lesser,
                p_greater=p_greater,
                p_retarded=p_retarded,
                diagonal_inds=(n, n),
                upper_inds=(n, m),
                order="reverse",
            )
            self.obc_blocks.retarded[-1] = obc_retarded
            self.obc_blocks.lesser[-1] = obc_lesser
            self.obc_blocks.greater[-1] = obc_greater

    @profiler.profile(
        label="CoulombScreeningSolver: Assemble Pr", level="default", comm=comm
    )
    def _assemble_retarded_polarization(
        self,
        p_lesser: DSDBSparse | _DStackView,
        p_greater: DSDBSparse | _DStackView,
        p_retarded_hermitian: DSDBSparse | _DStackView,
    ) -> None:
        r"""Assembles the full retarded polarization from the Hermitian part
        and the lesser and greater parts.

        $$\mathbf{P}^R = \mathbf{P}^R + \frac{1}{2} \left(\mathbf{P}^{>} - \mathbf{P}^{<} \right)$$

        This modifies retarded polarization in-place i.e. the result is stored in `self.p_retarded`.

        Parameters
        ----------
        p_lesser : DSDBSparse | _DStackView
            The lesser polarization.
        p_greater : DSDBSparse | _DStackView
            The greater polarization.
        p_retarded_hermitian : DSDBSparse | _DStackView
            The hermitian part of the retarded polarization.

        """
        p_retarded_ = self.p_retarded.stack[...]
        p_retarded_hermitian_ = p_retarded_hermitian.stack[...]
        p_lesser_ = p_lesser.stack[...]
        p_greater_ = p_greater.stack[...]
        for i in range(self.p_retarded.num_local_blocks):
            j = i + 1
            p_retarded_.blocks[i, i] = p_retarded_hermitian_.blocks[i, i] + 0.5 * (
                p_greater_.blocks[i, i] - p_lesser_.blocks[i, i]
            )

            if (
                j >= self.p_retarded.num_local_blocks
                and comm.block.rank == comm.block.size - 1
            ):
                # The last rank does not have these blocks.
                continue

            p_retarded_.blocks[i, j] = p_retarded_hermitian_.blocks[i, j] + 0.5 * (
                p_greater_.blocks[i, j] - p_lesser_.blocks[i, j]
            )
            p_retarded_.blocks[j, i] = p_retarded_hermitian_.blocks[j, i] + 0.5 * (
                p_greater_.blocks[j, i] - p_lesser_.blocks[j, i]
            )

    def _contact_spillover_matmul(
        self,
        diagonal_inds: tuple,
        upper_inds: tuple,
        order: str | NDArray | None = None,
    ):
        r"""Applies the spillover correction to

        $$\mathbf{V} \mathbf{P}^{R}$$

        for a specific contact.

        Parameters
        ----------
        diagonal_inds : tuple
            The indices of the diagonal blocks corresponding to the contact.
        upper_inds : tuple
            The indices of the upper off-diagonal blocks corresponding to the contact.
        order : str | NDArray | None, optional
            The permutation of the blocks to achieve the same order as the canonical left contact.
            If None, the left contact order is assumed.
            Instead of an explicit permutation, the string "reverse" can be passed
            to reverse the order of the blocks, which is equivalent to the right contact order.

        """

        inverse_order = get_inverse_order(order)

        v_10, __, __ = get_periodic_superblocks(
            a_ji=order_block(self.coulomb_matrix.blocks[*upper_inds[::-1]], order),
            a_ii=order_block(self.coulomb_matrix.blocks[*diagonal_inds], order),
            a_ij=order_block(self.coulomb_matrix.blocks[*upper_inds], order),
            block_sections=self.small_block_sections,
        )
        __, __, p_01 = get_periodic_superblocks(
            a_ji=order_block(self.p_retarded.blocks[*upper_inds[::-1]], order),
            a_ii=order_block(self.p_retarded.blocks[*diagonal_inds], order),
            a_ij=order_block(self.p_retarded.blocks[*upper_inds], order),
            block_sections=self.small_block_sections,
        )
        self.system_matrix.blocks[*diagonal_inds] += order_block(
            v_10 @ p_01, inverse_order
        )

    @profiler.profile(
        label="CoulombScreeningSolver: Assembly", level="default", comm=comm
    )
    def _assemble_system_matrix(
        self,
    ) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data = 0.0
        local_blocks, _ = get_section_sizes(
            len(self.system_matrix.block_sizes), comm.block.size
        )
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        bd_matmul(
            self.coulomb_matrix,
            self.p_retarded,
            out=self.system_matrix,
            start_block=start_block,
            end_block=end_block,
        )
        # apply spillover correction to the system matrix.
        # This is necessary because infinite leads are assumed.
        if comm.block.rank == 0:
            self._contact_spillover_matmul(
                diagonal_inds=(0, 0),
                upper_inds=(0, 1),
            )

        if comm.block.rank == comm.block.size - 1:
            n = self.system_matrix.num_local_blocks - 1
            m = n - 1
            self._contact_spillover_matmul(
                diagonal_inds=(n, n),
                upper_inds=(n, m),
                order="reverse",
            )

        xp.negative(self.system_matrix.data, out=self.system_matrix.data)
        self.system_matrix.fill_diagonal(self.system_matrix.diagonal() + 1.0)

    def _contact_spillover_sandwich(
        self,
        p_: DSDBSparse | _DStackView,
        l_: DSDBSparse | _DStackView,
        diagonal_inds: tuple,
        upper_inds: tuple,
        order: str | NDArray | None = None,
    ) -> None:
        r"""Applies the spillover correction to

        $$\mathbf{L}^{\lessgtr} = \mathbf{V} \mathbf{P}^{\lessgtr} \mathbf{V}^{\dagger}$$

        for either the lesser or the greater component at a specific contact.

        Parameters
        ----------
        p_ : DSDBSparse | _DStackView
            The polarization (either lesser or greater).
        l_ : DSDBSparse | _DStackView
            The matrix to which the spillover correction will be applied (either
            `l_lesser` or `l_greater`).
        diagonal_inds : tuple
            The indices of the diagonal blocks corresponding to the contact.
        upper_inds : tuple
            The indices of the upper off-diagonal blocks corresponding to the contact.
        order : str | NDArray | None, optional
            The permutation of the blocks to achieve the same order as the canonical left contact.
            If None, the left contact order is assumed.
            Instead of an explicit permutation, the string "reverse" can be passed
            to reverse the order of the blocks, which is equivalent to the right contact order.

        """

        inverse_order = get_inverse_order(order)

        v_10, v_00, v_01 = get_periodic_superblocks(
            a_ji=order_block(self.coulomb_matrix.blocks[*upper_inds[::-1]], order),
            a_ii=order_block(self.coulomb_matrix.blocks[*diagonal_inds], order),
            a_ij=order_block(self.coulomb_matrix.blocks[*upper_inds], order),
            block_sections=self.small_block_sections,
        )

        p_10, p_00, p_01 = get_periodic_superblocks(
            a_ji=order_block(p_.blocks[*upper_inds[::-1]], order),
            a_ii=order_block(p_.blocks[*diagonal_inds], order),
            a_ij=order_block(p_.blocks[*upper_inds], order),
            block_sections=self.small_block_sections,
        )

        l_.blocks[*diagonal_inds] += order_block(
            v_10 @ p_01 @ v_00 + v_00 @ p_10 @ v_01 + v_10 @ p_00 @ v_01, inverse_order
        )
        l_.blocks[*upper_inds] += order_block(v_10 @ p_01 @ v_01, inverse_order)
        if l_.symmetry is None:
            l_.blocks[*upper_inds[::-1]] += order_block(
                v_10 @ p_10 @ v_01, inverse_order
            )

    def _apply_spillover_sandwich(
        self,
        p_: DSDBSparse | _DStackView,
        l_: DSDBSparse | _DStackView,
    ) -> None:
        r"""Applies the spillover correction to

        $$\mathbf{L}^{\lessgtr} = \mathbf{V} \mathbf{P}^{\lessgtr} \mathbf{V}^{\dagger}$$

        for either the lesser or the greater component at all contacts.

        Parameters
        ----------
        p_ : DSDBSparse | _DStackView
            The polarization (either lesser or greater).
        l_ : DSDBSparse | _DStackView
            The matrix to which the spillover correction will be applied (either
            `l_lesser` or `l_greater`).

        """

        if comm.block.rank == 0:
            self._contact_spillover_sandwich(
                p_=p_,
                l_=l_,
                diagonal_inds=(0, 0),
                upper_inds=(0, 1),
            )

        if comm.block.rank == comm.block.size - 1:
            n = self.system_matrix.num_local_blocks - 1
            m = n - 1
            self._contact_spillover_sandwich(
                p_=p_,
                l_=l_,
                # NOTE: Order of inds is reversed for the right contact
                # i.e. `upper_inds` is not `(m, n)`
                diagonal_inds=(n, n),
                upper_inds=(n, m),
                order="reverse",
            )

    def _filter_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Filters out peaks in the Green's functions.

        Parameters
        ----------
        out : tuple[DSDBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """

        w_lesser, w_greater, *__ = out
        local_dos = []

        w_lesser_diag = w_lesser.diagonal()
        w_greater_diag = w_greater.diagonal()

        w_lesser_diag = comm.block.all_gather_v(
            w_lesser_diag,
            axis=-1,
        )
        w_greater_diag = comm.block.all_gather_v(
            w_greater_diag,
            axis=-1,
        )

        block_sizes = w_lesser.block_sizes
        block_offsets = w_lesser.block_offsets
        for i, (bzs, boff) in enumerate(zip(block_sizes, block_offsets)):
            w_lesser_density = w_lesser_diag[..., boff : boff + bzs].imag.mean(-1)
            w_greater_density = -w_greater_diag[..., boff : boff + bzs].imag.mean(-1)
            local_dos.append(0.5 * (w_greater_density - w_lesser_density))

        local_dos = xp.array(local_dos)
        dos = comm.stack.all_gather_v(
            local_dos,
            axis=1,
            mask=w_lesser._stack_padding_mask,
        )

        dos_gradient = xp.abs(xp.gradient(dos, self.energies, axis=1))
        mask = xp.max(dos_gradient, axis=0) > self.dos_peak_limit

        section_sizes, __ = get_section_sizes(self.energies.size, comm.stack.size)
        section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
        local_mask = mask[
            section_offsets[comm.stack.rank] : section_offsets[comm.stack.rank + 1]
        ]

        w_lesser.data[local_mask] = 0.0
        w_greater.data[local_mask] = 0.0

    @profiler.profile(label="CoulombScreeningSolver", level="default", comm=comm)
    def solve(
        self,
        p_lesser: DSDBSparse,
        p_greater: DSDBSparse,
        p_retarded_hermitian: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Solves for the screened Coulomb interaction.

        Parameters
        ----------
        p_lesser : DSDBSparse
            The lesser polarization.
        p_greater : DSDBSparse
            The greater polarization.
        p_retarded_hermitian : DSDBSparse
            The hermitian part of the retarded polarization.
            The anti-hermitian part is calculated from lesser and greater.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """

        if self.max_batch_size is None:
            max_batch_size = p_lesser.shape[0]
        else:
            max_batch_size = self.max_batch_size
        batch_sizes, batch_offsets = get_batches(p_lesser.shape[0], max_batch_size)

        if self.spin:
            p_lesser = _cast_to_spin_matrix(p_lesser)
            p_greater = _cast_to_spin_matrix(p_greater)
            p_retarded_hermitian = _cast_to_spin_matrix(p_retarded_hermitian)

        for i in range(len(batch_sizes)):

            batch_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            # Free data when the batch size changes
            if i > 0 and batch_sizes[i] != batch_sizes[i - 1]:
                self.system_matrix.free_data()
                self.l_lesser.free_data()
                self.l_greater.free_data()

            with profiler.profile_range(
                label="CoulombScreeningSolver: Set block sizes",
                level="default",
                comm=comm,
            ):
                self.p_retarded.allocate_data(stack_size=batch_sizes[i])
                self.system_matrix.allocate_data(stack_size=batch_sizes[i])
                self.l_lesser.allocate_data(stack_size=batch_sizes[i])
                self.l_greater.allocate_data(stack_size=batch_sizes[i])

                p_lesser_batch = p_lesser.stack[batch_slice]
                p_greater_batch = p_greater.stack[batch_slice]
                p_retarded_hermitian_batch = p_retarded_hermitian.stack[batch_slice]

                # Change the block sizes to match the Coulomb matrix.
                self._set_block_sizes(self.small_block_sizes)

            self._assemble_retarded_polarization(
                p_lesser_batch,
                p_greater_batch,
                p_retarded_hermitian_batch,
            )

            # Assemble the system matrix (Includes matrix multiplication).
            self._assemble_system_matrix()

            # Apply the OBC algorithm.
            self._compute_obc(
                p_lesser_batch,
                p_greater_batch,
                self.p_retarded,
                batch_slice,
            )
            self.p_retarded.free_data()

            with profiler.profile_range(
                label="CoulombScreeningSolver: Sandwich", level="default", comm=comm
            ):
                local_blocks, _ = get_section_sizes(
                    len(self.coulomb_matrix.block_sizes), comm.block.size
                )
                start_block = sum(local_blocks[: comm.block.rank])
                end_block = start_block + local_blocks[comm.block.rank]
                bd_sandwich(
                    self.coulomb_matrix,
                    p_lesser_batch,
                    out=self.l_lesser,
                    start_block=start_block,
                    end_block=end_block,
                )
                self._apply_spillover_sandwich(p_lesser_batch, self.l_lesser)

                bd_sandwich(
                    self.coulomb_matrix,
                    p_greater_batch,
                    out=self.l_greater,
                    start_block=start_block,
                    end_block=end_block,
                )
                self._apply_spillover_sandwich(p_greater_batch, self.l_greater)

            if self.flatband:
                with profiler.profile_range(
                    label="CoulombScreeningSolver: Homogenize",
                    level="default",
                    comm=comm,
                ):
                    homogenize(self.system_matrix)
                    homogenize(self.l_lesser)
                    homogenize(self.l_greater)

            with profiler.profile_range(
                label="CoulombScreeningSolver: Set block sizes back",
                level="default",
                comm=comm,
            ):
                # Go back to normal block sizes.
                self._set_block_sizes(self.block_sizes)

            with profiler.profile_range(
                label="CoulombScreeningSolver: Solve", level="default", comm=comm
            ):
                out_l, out_g = out
                if self.spin:
                    out_l = _cast_to_spin_matrix(out_l, out_l)
                    out_g = _cast_to_spin_matrix(out_g, out_g)

                out_slice = (
                    out_l.stack[batch_slice],
                    out_g.stack[batch_slice],
                )

                # Solve the system
                if comm.block.size > 1:
                    self.solver_dist.selected_solve(
                        a=self.system_matrix,
                        sigma_lesser=self.l_lesser,
                        sigma_greater=self.l_greater,
                        obc_blocks=self.obc_blocks,
                        out=out_slice,
                        return_retarded=False,
                    )

                else:
                    self.solver.selected_solve(
                        a=self.system_matrix,
                        sigma_lesser=self.l_lesser,
                        sigma_greater=self.l_greater,
                        obc_blocks=self.obc_blocks,
                        out=out_slice,
                        return_retarded=False,
                    )

        if self.spin:
            w_lesser, w_greater, *_ = out
            _cast_from_spin_matrix(out_l, w_lesser)
            _cast_from_spin_matrix(out_g, w_greater)


        with profiler.profile_range(
            label="CoulombScreeningSolver: Filter", level="default", comm=comm
        ):
            # Only filter the peaks for the first few iterations.
            if self.solve_call_count < self.filtering_iteration_limit:
                self._filter_peaks(out)

            self.system_matrix.free_data()
            self.l_lesser.free_data()
            self.l_greater.free_data()

            w_lesser, w_greater, *__ = out
            if comm.stack.rank == 0:
                w_greater.data[0, :] = 0.0
                w_lesser.data[0, :] = 0.0

        self.solve_call_count += 1
