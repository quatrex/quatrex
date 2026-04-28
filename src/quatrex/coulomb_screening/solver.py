# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.routines import bd_matmul_distr, bd_sandwich_distr
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import product_sparsity_pattern_dsdbsparse
from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import (
    compute_num_connected_blocks,
    get_periodic_superblocks,
    homogenize,
)

profiler = Profiler()


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

        self.coulomb_matrix = coulomb_matrix
        self.small_block_sizes = self.coulomb_matrix.block_sizes

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
            sparsity_pattern.astype(xp.float32),
            block_sizes=self.small_block_sizes,
            global_stack_shape=(comm.size,),
            symmetry=config.scba.symmetric,
            symmetry_op=xp.conj,
        )
        v_times_p_sparsity_pattern = _compute_sparsity_pattern(
            dummy_dsbsparse, dummy_dsbsparse, dtype=xp.float32
        )

        # Allocate memory for the System matrix (1 - V @ P).
        kpoint_grid = config.device.kpoint_grid
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            v_times_p_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
        )
        self.system_matrix.free_data()
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
            l_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry=config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )
        self.l_greater = config.compute.dsdbsparse_type.zeros_like(self.l_lesser)
        # Explicitely try to free the memory for the sparsity pattern.
        del l_sparsity_pattern
        self.l_lesser.free_data()
        self.l_greater.free_data()

        # Allocate object for the retarded polarization.
        # This is only used as a temporary assembling the full retarded
        # polarization before multiplying with the Coulomb matrix.
        # This is simpler in case of domain distributed solver.
        # Otherwise, it would be more memory efficient to directly assemble
        # when doing the product with the Coulomb matrix, but the life time
        # is not during the peak (quadratic solve).
        self.p_retarded = config.compute.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.small_block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
        )
        self.p_retarded.free_data()

        # Boundary conditions.
        self.left_occupancies = bose_einstein(
            self.local_energies,
            config.coulomb_screening.temperature,
        )
        self.right_occupancies = bose_einstein(
            self.local_energies,
            config.coulomb_screening.temperature,
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

    def _get_periodic_superblocks(
        self, m_ji: NDArray, m_ii: NDArray, m_ij: NDArray, lower=False
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Constructs the periodic superblocks for the given blocks."""

        new_shape = list(m_ii.shape)
        new_shape[-1] = new_shape[-1] * self.num_connected_blocks
        new_shape[-2] = new_shape[-2] * self.num_connected_blocks

        m_ji_out = xp.zeros_like(m_ji, shape=new_shape)
        m_ii_out = xp.zeros_like(m_ii, shape=new_shape)
        m_ij_out = xp.zeros_like(m_ij, shape=new_shape)

        m_ji_tmp, m_ii_tmp, m_ij_tmp = get_periodic_superblocks(
            a_ji=m_ji if not lower else xp.flip(m_ji, axis=(-2, -1)),
            a_ii=m_ii if not lower else xp.flip(m_ii, axis=(-2, -1)),
            a_ij=m_ij if not lower else xp.flip(m_ij, axis=(-2, -1)),
            block_sections=self.small_block_sections,
        )

        for i in range(self.num_connected_blocks):
            m_ii_out[
                ...,
                i * m_ii.shape[-1] : (i + 1) * m_ii.shape[-1],
                i * m_ii.shape[-1] : (i + 1) * m_ii.shape[-1],
            ] = m_ii_tmp

        for i in range(self.num_connected_blocks - 1):
            m_ii_out[
                ...,
                i * m_ii.shape[-1] : (i + 1) * m_ii.shape[-1],
                (i + 1) * m_ii.shape[-1] : (i + 2) * m_ii.shape[-1],
            ] = m_ij_tmp
            m_ii_out[
                ...,
                (i + 1) * m_ii.shape[-1] : (i + 2) * m_ii.shape[-1],
                i * m_ii.shape[-1] : (i + 1) * m_ii.shape[-1],
            ] = m_ji_tmp

        m_ij_out[..., -m_ij.shape[-1] :, : m_ij.shape[-1]] = m_ij_tmp
        m_ji_out[..., : m_ji.shape[-1], -m_ji.shape[-1] :] = m_ji_tmp

        m_ji_out = m_ji_out if not lower else xp.flip(m_ji_out, axis=(-2, -1))
        m_ii_out = m_ii_out if not lower else xp.flip(m_ii_out, axis=(-2, -1))
        m_ij_out = m_ij_out if not lower else xp.flip(m_ij_out, axis=(-2, -1))

        return m_ji_out, m_ii_out, m_ij_out

    @profiler.profile(label="CoulombScreeningSolver: OBC", level="default", comm=comm)
    def _compute_obc(self, p_lesser, p_greater, p_retarded) -> None:
        """Computes open boundary conditions."""
        if comm.block.rank == 0:

            with profiler.profile_range(
                label="CoulombScreeningSolver: Get blocks left",
                level="default",
                comm=comm.stack,
            ):

                p_retarded_10, p_retarded_00, p_retarded_01 = (
                    self._get_periodic_superblocks(
                        m_ji=p_retarded.blocks[1, 0],
                        m_ii=p_retarded.blocks[0, 0],
                        m_ij=p_retarded.blocks[0, 1],
                    )
                )
                v_10, v_00, v_01 = self._get_periodic_superblocks(
                    m_ji=self.coulomb_matrix.blocks[1, 0],
                    m_ii=self.coulomb_matrix.blocks[0, 0],
                    m_ij=self.coulomb_matrix.blocks[0, 1],
                )

                m_00 = (
                    xp.eye(p_retarded_00.shape[-1])
                    - v_10 @ p_retarded_01
                    - v_00 @ p_retarded_00
                    - v_01 @ p_retarded_10
                )
                m_01 = -v_00 @ p_retarded_01 - v_01 @ p_retarded_00
                m_10 = -v_10 @ p_retarded_00 - v_00 @ p_retarded_10

                p_lesser_10, p_lesser_00, p_lesser_01 = self._get_periodic_superblocks(
                    m_ji=p_lesser.blocks[1, 0],
                    m_ii=p_lesser.blocks[0, 0],
                    m_ij=p_lesser.blocks[0, 1],
                )
                p_greater_10, p_greater_00, p_greater_01 = (
                    self._get_periodic_superblocks(
                        m_ji=p_greater.blocks[1, 0],
                        m_ii=p_greater.blocks[0, 0],
                        m_ij=p_greater.blocks[0, 1],
                    )
                )
                l_lesser_00 = (
                    v_10 @ p_lesser_00 @ v_01
                    + v_10 @ p_lesser_01 @ v_00
                    + v_00 @ p_lesser_10 @ v_01
                    + v_00 @ p_lesser_00 @ v_00
                    + v_00 @ p_lesser_01 @ v_10
                    + v_01 @ p_lesser_10 @ v_00
                    + v_01 @ p_lesser_00 @ v_10
                )
                l_lesser_01 = (
                    v_10 @ p_lesser_01 @ v_01
                    + v_00 @ p_lesser_00 @ v_01
                    + v_00 @ p_lesser_01 @ v_00
                    + v_01 @ p_lesser_10 @ v_01
                    + v_01 @ p_lesser_00 @ v_00
                )
                l_greater_00 = (
                    v_10 @ p_greater_00 @ v_01
                    + v_10 @ p_greater_01 @ v_00
                    + v_00 @ p_greater_10 @ v_01
                    + v_00 @ p_greater_00 @ v_00
                    + v_00 @ p_greater_01 @ v_10
                    + v_01 @ p_greater_10 @ v_00
                    + v_01 @ p_greater_00 @ v_10
                )
                l_greater_01 = (
                    v_10 @ p_greater_01 @ v_01
                    + v_00 @ p_greater_00 @ v_01
                    + v_00 @ p_greater_01 @ v_00
                    + v_01 @ p_greater_10 @ v_01
                    + v_01 @ p_greater_00 @ v_00
                )

            with profiler.profile_range(
                label="CoulombScreeningSolver: OBCR left",
                level="default",
                comm=comm.stack,
            ):

                x_00, *__ = self.obc((m_00, m_01, m_10), contact="left")

                m_10_x_00 = m_10 @ x_00
                self.obc_blocks.retarded[0] = m_10_x_00 @ m_01

            with profiler.profile_range(
                label="CoulombScreeningSolver: Lyapunov left",
                level="default",
                comm=comm.stack,
            ):
                # Compute and apply the left lesser/greater boundary self-energy.
                a_00_lesser = m_10_x_00 @ l_lesser_01
                a_00_greater = m_10_x_00 @ l_greater_01

                q_00_lesser = (
                    x_00
                    @ (
                        l_lesser_00
                        - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))
                    )
                    @ x_00.conj().swapaxes(-1, -2)
                )
                q_00_greater = (
                    x_00
                    @ (
                        l_greater_00
                        - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))
                    )
                    @ x_00.conj().swapaxes(-1, -2)
                )

                b_00 = x_00 @ m_10
                q_00 = xp.stack((q_00_lesser, q_00_greater))

                w_00, *__ = self.lyapunov((b_00, q_00), "left")
                w_00_lesser, w_00_greater = w_00

                self.obc_blocks.lesser[0] = m_10 @ w_00_lesser @ m_10.conj().swapaxes(
                    -1, -2
                ) - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))

                self.obc_blocks.greater[0] = m_10 @ w_00_greater @ m_10.conj().swapaxes(
                    -1, -2
                ) - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))

        if comm.block.rank == comm.block.size - 1:

            with profiler.profile_range(
                label="CoulombScreeningSolver: Get blocks right",
                level="default",
                comm=comm.stack,
            ):

                n = p_retarded.num_local_blocks - 1
                m = n - 1

                p_retarded_mn, p_retarded_nn, p_retarded_nm = (
                    self._get_periodic_superblocks(
                        m_ji=p_retarded.blocks[m, n],
                        m_ii=p_retarded.blocks[n, n],
                        m_ij=p_retarded.blocks[n, m],
                        lower=True,
                    )
                )
                v_mn, v_nn, v_nm = self._get_periodic_superblocks(
                    m_ji=self.coulomb_matrix.blocks[m, n],
                    m_ii=self.coulomb_matrix.blocks[n, n],
                    m_ij=self.coulomb_matrix.blocks[n, m],
                    lower=True,
                )

                m_nn = (
                    xp.eye(p_retarded_nn.shape[-1])
                    - v_mn @ p_retarded_nm
                    - v_nn @ p_retarded_nn
                    - v_nm @ p_retarded_mn
                )
                m_nm = -v_nn @ p_retarded_nm - v_nm @ p_retarded_nn
                m_mn = -v_mn @ p_retarded_nn - v_nn @ p_retarded_mn

                p_lesser_mn, p_lesser_nn, p_lesser_nm = self._get_periodic_superblocks(
                    m_ji=p_lesser.blocks[m, n],
                    m_ii=p_lesser.blocks[n, n],
                    m_ij=p_lesser.blocks[n, m],
                    lower=True,
                )
                p_greater_mn, p_greater_nn, p_greater_nm = (
                    self._get_periodic_superblocks(
                        m_ji=p_greater.blocks[m, n],
                        m_ii=p_greater.blocks[n, n],
                        m_ij=p_greater.blocks[n, m],
                        lower=True,
                    )
                )
                l_lesser_nn = (
                    v_mn @ p_lesser_nn @ v_nm
                    + v_mn @ p_lesser_nm @ v_nn
                    + v_nn @ p_lesser_mn @ v_nm
                    + v_nn @ p_lesser_nn @ v_nn
                    + v_nn @ p_lesser_nm @ v_mn
                    + v_nm @ p_lesser_mn @ v_nn
                    + v_nm @ p_lesser_nn @ v_mn
                )
                l_lesser_nm = (
                    v_mn @ p_lesser_nm @ v_nm
                    + v_nn @ p_lesser_nn @ v_nm
                    + v_nn @ p_lesser_nm @ v_nn
                    + v_nm @ p_lesser_mn @ v_nm
                    + v_nm @ p_lesser_nn @ v_nn
                )
                l_greater_nn = (
                    v_mn @ p_greater_nn @ v_nm
                    + v_mn @ p_greater_nm @ v_nn
                    + v_nn @ p_greater_mn @ v_nm
                    + v_nn @ p_greater_nn @ v_nn
                    + v_nn @ p_greater_nm @ v_mn
                    + v_nm @ p_greater_mn @ v_nn
                    + v_nm @ p_greater_nn @ v_mn
                )
                l_greater_nm = (
                    v_mn @ p_greater_nm @ v_nm
                    + v_nn @ p_greater_nn @ v_nm
                    + v_nn @ p_greater_nm @ v_nn
                    + v_nm @ p_greater_mn @ v_nm
                    + v_nm @ p_greater_nn @ v_nn
                )

            with profiler.profile_range(
                label="CoulombScreeningSolver: OBCR right",
                level="default",
                comm=comm.stack,
            ):

                x_nn, *__ = self.obc(
                    # Twist it, flip it, ...
                    (
                        xp.flip(m_nn, axis=(-2, -1)),
                        xp.flip(m_nm, axis=(-2, -1)),
                        xp.flip(m_mn, axis=(-2, -1)),
                    ),
                    contact="right",
                )
                # ... bop it.
                x_nn = xp.flip(x_nn, axis=(-2, -1))

                m_mn_x_nn = m_mn @ x_nn

                self.obc_blocks.retarded[-1] = m_mn_x_nn @ m_nm

            with profiler.profile_range(
                label="CoulombScreeningSolver: Lyapunov right",
                level="default",
                comm=comm.stack,
            ):
                # Compute and apply the right lesser/greater boundary self-energy.
                a_nn_lesser = m_mn_x_nn @ l_lesser_nm
                a_nn_greater = m_mn_x_nn @ l_greater_nm

                q_nn_lesser = (
                    x_nn
                    @ (
                        l_lesser_nn
                        - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))
                    )
                    @ x_nn.conj().swapaxes(-1, -2)
                )
                q_nn_greater = (
                    x_nn
                    @ (
                        l_greater_nn
                        - (a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2))
                    )
                    @ x_nn.conj().swapaxes(-1, -2)
                )

                b_nn = x_nn @ m_mn

                q_nn = xp.stack((q_nn_lesser, q_nn_greater))

                w_nn, *__ = self.lyapunov((b_nn, q_nn), "right")
                w_nn_lesser, w_nn_greater = w_nn

                self.obc_blocks.lesser[-1] = m_mn @ w_nn_lesser @ m_mn.conj().swapaxes(
                    -1, -2
                ) - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))

                self.obc_blocks.greater[
                    -1
                ] = m_mn @ w_nn_greater @ m_mn.conj().swapaxes(-1, -2) - (
                    a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2)
                )

    def _assemble_retarded_polarization(
        self,
        p_lesser: DSDBSparse,
        p_greater: DSDBSparse,
        p_retarded_hermitian: DSDBSparse,
    ) -> None:
        r"""Assembles the full retarded polarization from the Hermitian part
        and the lesser and greater parts.

        $$\mathbf{P}^R = \mathbf{P}^R + \frac{1}{2} \left(\mathbf{P}^{>} - \mathbf{P}^{<} \right)$$

        This modifies retarded polarization in-place i.e. the result is stored in `self.p_retarded`.

        Parameters
        ----------
        p_lesser : DSDBSparse
            The lesser polarization.
        p_greater : DSDBSparse
            The greater polarization.
        p_retarded_hermitian : DSDBSparse
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

        def _order_block(block):
            if order is None:
                return block
            elif order == "reverse":
                return xp.flip(block, axis=(-2, -1))
            else:
                return block[..., :, order][..., order, :]

        v_10, __, __ = get_periodic_superblocks(
            a_ii=_order_block(self.coulomb_matrix.blocks[*diagonal_inds]),
            a_ji=_order_block(self.coulomb_matrix.blocks[*upper_inds[::-1]]),
            a_ij=_order_block(self.coulomb_matrix.blocks[*upper_inds]),
            block_sections=self.small_block_sections,
        )
        __, __, p_01 = get_periodic_superblocks(
            a_ii=_order_block(self.p_retarded.blocks[*diagonal_inds]),
            a_ji=_order_block(self.p_retarded.blocks[*upper_inds[::-1]]),
            a_ij=_order_block(self.p_retarded.blocks[*upper_inds]),
            block_sections=self.small_block_sections,
        )
        self.system_matrix.blocks[*diagonal_inds] += _order_block(v_10 @ p_01)

    @profiler.profile(
        label="CoulombScreeningSolver: Assembly", level="default", comm=comm
    )
    def _assemble_system_matrix(
        self,
        p_lesser: DSDBSparse,
        p_greater: DSDBSparse,
        p_retarded_hermitian: DSDBSparse,
    ) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        p_lesser : DSDBSparse
            The lesser polarization.
        p_greater : DSDBSparse
            The greater polarization.
        p_retarded_hermitian : DSDBSparse
            The hermitian part of the retarded polarization.
             The anti-hermitian part is calculated from lesser and greater.

        """
        self.system_matrix.data = 0.0
        local_blocks, _ = get_section_sizes(
            len(self.system_matrix.block_sizes), comm.block.size
        )
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        self.p_retarded.allocate_data()

        self._assemble_retarded_polarization(
            p_lesser,
            p_greater,
            p_retarded_hermitian,
        )
        bd_matmul_distr(
            self.coulomb_matrix,
            self.p_retarded,
            out=self.system_matrix,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=False,
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

        self.p_retarded.free_data()


        xp.negative(self.system_matrix.data, out=self.system_matrix.data)
        self.system_matrix += sparse.eye(self.system_matrix.shape[-1])

    def _contact_spillover_sandwich(
        self,
        p_: DSDBSparse,
        l_: DSDBSparse,
        diagonal_inds: tuple,
        upper_inds: tuple,
        order: str | NDArray | None = None,
    ) -> None:
        r"""Applies the spillover correction to

        $$\mathbf{L}^{\lessgtr} = \mathbf{V} \mathbf{P}^{\lessgtr} \mathbf{V}^{\dagger}$$

        for either the lesser or the greater component at a specific contact.

        Parameters
        ----------
        p_ : DSDBSparse
            The polarization (either lesser or greater).
        l_ : DSDBSparse
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

        if isinstance(order, str) and order not in ["reverse"]:
            raise ValueError(
                f"Invalid order string: {order}. Must be 'reverse' or None."
            )
        elif isinstance(order, xp.ndarray) and order.ndim != 1:
            raise ValueError(
                f"Order array must be 1-dimensional, got shape {order.shape}."
            )

        def _order_block(block):
            if order is None:
                return block
            elif order == "reverse":
                return xp.flip(block, axis=(-2, -1))
            else:
                return block[..., :, order][..., order, :]

        v_10, v_00, v_01 = get_periodic_superblocks(
            a_ii=_order_block(self.coulomb_matrix.blocks[*diagonal_inds]),
            a_ji=_order_block(self.coulomb_matrix.blocks[*upper_inds[::-1]]),
            a_ij=_order_block(self.coulomb_matrix.blocks[*upper_inds]),
            block_sections=self.small_block_sections,
        )

        p_10, p_00, p_01 = get_periodic_superblocks(
            a_ii=_order_block(p_.blocks[*diagonal_inds]),
            a_ji=_order_block(p_.blocks[*upper_inds[::-1]]),
            a_ij=_order_block(p_.blocks[*upper_inds]),
            block_sections=self.small_block_sections,
        )

        l_.blocks[*diagonal_inds] += _order_block(
            v_10 @ p_01 @ v_00 + v_00 @ p_10 @ v_01 + v_10 @ p_00 @ v_01
        )
        l_.blocks[*upper_inds] += _order_block(v_10 @ p_01 @ v_01)
        if not l_.symmetry:
            l_.blocks[*upper_inds[::-1]] += _order_block(v_10 @ p_10 @ v_01)

    def _apply_spillover_sandwich(
        self,
        p_: DSDBSparse,
        l_: DSDBSparse,
    ) -> None:
        r"""Applies the spillover correction to

        $$\mathbf{L}^{\lessgtr} = \mathbf{V} \mathbf{P}^{\lessgtr} \mathbf{V}^{\dagger}$$

        for either the lesser or the greater component at all contacts.

        Parameters
        ----------
        p_ : DSDBSparse
            The polarization (either lesser or greater).
        l_ : DSDBSparse
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
        with profiler.profile_range(
            label="CoulombScreeningSolver: Set block sizes", level="default", comm=comm
        ):
            self.system_matrix.allocate_data()
            self.l_lesser.allocate_data()
            self.l_greater.allocate_data()
            # Change the block sizes to match the Coulomb matrix.
            self._set_block_sizes(self.small_block_sizes)

        # Assemble the system matrix (Includes matrix multiplication).
        self._assemble_system_matrix(p_lesser, p_greater, p_retarded_hermitian)

        with profiler.profile_range(
            label="CoulombScreeningSolver: Sandwich", level="default", comm=comm
        ):
            local_blocks, _ = get_section_sizes(
                len(self.coulomb_matrix.block_sizes), comm.block.size
            )
            start_block = sum(local_blocks[: comm.block.rank])
            end_block = start_block + local_blocks[comm.block.rank]
            bd_sandwich_distr(
                self.coulomb_matrix,
                p_lesser,
                out=self.l_lesser,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            self._apply_spillover_sandwich(p_lesser, self.l_lesser)

            bd_sandwich_distr(
                self.coulomb_matrix,
                p_greater,
                out=self.l_greater,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            self._apply_spillover_sandwich(p_greater, self.l_greater)

        if self.flatband:
            with profiler.profile_range(
                label="CoulombScreeningSolver: Homogenize", level="default", comm=comm
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

        # Apply the OBC algorithm.
        self._compute_obc(
            p_lesser,
            p_greater,
            p_retarded,
        )

        with profiler.profile_range(
            label="CoulombScreeningSolver: Solve", level="default", comm=comm
        ):
            # Solve the system
            if comm.block.size > 1:
                self.solver_dist.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=self.l_lesser,
                    sigma_greater=self.l_greater,
                    obc_blocks=self.obc_blocks,
                    out=out,
                    return_retarded=False,
                )

            else:
                self.solver.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=self.l_lesser,
                    sigma_greater=self.l_greater,
                    obc_blocks=self.obc_blocks,
                    out=out,
                    return_retarded=False,
                )

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
