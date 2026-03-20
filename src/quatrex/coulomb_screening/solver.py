# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.routines import bd_matmul_distr, bd_sandwich_distr
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.kernels.mixed_precision import compress, decompress
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import create_stream, synchronize_stream
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.solvers_utils import get_batches
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
        rows,
        cols,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config, energies)

        self.coulomb_matrix = coulomb_matrix
        self.coulomb_matrix.to_host()

        self.small_block_sizes = self.coulomb_matrix.block_sizes

        self.num_connected_blocks = config.coulomb_screening.num_connected_blocks
        if self.num_connected_blocks == "auto":
            self.num_connected_blocks = compute_num_connected_blocks(
                rows, cols, self.small_block_sizes
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
            rows,
            cols,
            block_sizes=self.small_block_sizes,
            global_stack_shape=(comm.size,),
            symmetry=config.scba.symmetric,
            symmetry_op=xp.conj,
            bits=config.compute.num_bits,
        )
        dummy_dsbsparse.allocate_data()
        dummy_dsbsparse.data[:] = 1.0

        v_times_p_sparsity_pattern = _compute_sparsity_pattern(
            dummy_dsbsparse, dummy_dsbsparse, dtype=xp.float32
        )

        # Allocate memory for the System matrix (1 - V @ P).
        kpoint_grid = config.device.kpoint_grid
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            v_times_p_sparsity_pattern.row,
            v_times_p_sparsity_pattern.col,
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            bits=config.compute.num_bits,
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
            l_sparsity_pattern.row,
            l_sparsity_pattern.col,
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry=config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
            bits=config.compute.num_bits,
        )
        self.l_greater = config.compute.dsdbsparse_type.empty_like(self.l_lesser)
        # Explicitely try to free the memory for the sparsity pattern.
        del l_sparsity_pattern
        self.l_lesser.free_data()
        self.l_greater.free_data()

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

        self.flatband = config.electron.flatband
        self.solve_call_count = 0
        self.filtering_iteration_limit = (
            config.coulomb_screening.filtering_iteration_limit
        )

        self.max_batch_size = config.coulomb_screening.max_batch_size

        self._system_stream = create_stream()

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

    @profiler.profile(label="CoulombScreeningSolver: OBC", level="default", comm=comm)
    def _compute_obc(self, stack_slice: slice) -> None:
        """Computes open boundary conditions."""
        if comm.block.rank == 0:

            with profiler.profile_range(
                label="CoulombScreeningSolver: OBCR left",
                level="default",
                comm=comm.stack,
            ):

                m_10, m_00, m_01 = get_periodic_superblocks(
                    a_ii=self.system_matrix.blocks[0, 0],
                    a_ji=self.system_matrix.blocks[1, 0],
                    a_ij=self.system_matrix.blocks[0, 1],
                    block_sections=self.block_sections,
                )

                x_00, *__ = self.obc(
                    (m_00, m_01, m_10), contact="left-" + str(stack_slice)
                )

                m_10_x_00 = m_10 @ x_00
                self.obc_blocks.retarded[0] = m_10_x_00 @ m_01

            with profiler.profile_range(
                label="CoulombScreeningSolver: Lyapunov left",
                level="default",
                comm=comm.stack,
            ):
                # Compute and apply the left lesser/greater boundary self-energy.
                a_00_lesser = m_10_x_00 @ self.l_lesser.blocks[0, 1]
                a_00_greater = m_10_x_00 @ self.l_greater.blocks[0, 1]

                q_00_lesser = (
                    x_00
                    @ (
                        self.l_lesser.blocks[0, 0]
                        - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))
                    )
                    @ x_00.conj().swapaxes(-1, -2)
                )
                q_00_greater = (
                    x_00
                    @ (
                        self.l_greater.blocks[0, 0]
                        - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))
                    )
                    @ x_00.conj().swapaxes(-1, -2)
                )

                b_00 = x_00 @ m_10
                q_00 = xp.stack((q_00_lesser, q_00_greater))

                w_00, *__ = self.lyapunov((b_00, q_00), "left-" + str(stack_slice))
                w_00_lesser, w_00_greater = w_00

                self.obc_blocks.lesser[0] = m_10 @ w_00_lesser @ m_10.conj().swapaxes(
                    -1, -2
                ) - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))

                self.obc_blocks.greater[0] = m_10 @ w_00_greater @ m_10.conj().swapaxes(
                    -1, -2
                ) - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))

        if comm.block.rank == comm.block.size - 1:

            with profiler.profile_range(
                label="CoulombScreeningSolver: OBCR right",
                level="default",
                comm=comm.stack,
            ):

                n = self.system_matrix.num_local_blocks - 1
                m = n - 1

                m_mn, m_nn, m_nm = get_periodic_superblocks(
                    # Twist it, flip it, ...
                    a_ii=xp.flip(self.system_matrix.blocks[n, n], axis=(-2, -1)),
                    a_ji=xp.flip(self.system_matrix.blocks[m, n], axis=(-2, -1)),
                    a_ij=xp.flip(self.system_matrix.blocks[n, m], axis=(-2, -1)),
                    block_sections=self.block_sections,
                )
                # ... bop it.
                m_nn = xp.flip(m_nn, axis=(-2, -1))
                m_nm = xp.flip(m_nm, axis=(-2, -1))
                m_mn = xp.flip(m_mn, axis=(-2, -1))
                x_nn, *__ = self.obc(
                    # Twist it, flip it, ...
                    (
                        xp.flip(m_nn, axis=(-2, -1)),
                        xp.flip(m_nm, axis=(-2, -1)),
                        xp.flip(m_mn, axis=(-2, -1)),
                    ),
                    contact="right-" + str(stack_slice),
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
                a_nn_lesser = m_mn_x_nn @ self.l_lesser.blocks[n, m]
                a_nn_greater = m_mn_x_nn @ self.l_greater.blocks[n, m]

                q_nn_lesser = (
                    x_nn
                    @ (
                        self.l_lesser.blocks[n, n]
                        - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))
                    )
                    @ x_nn.conj().swapaxes(-1, -2)
                )
                q_nn_greater = (
                    x_nn
                    @ (
                        self.l_greater.blocks[n, n]
                        - (a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2))
                    )
                    @ x_nn.conj().swapaxes(-1, -2)
                )

                b_nn = x_nn @ m_mn

                q_nn = xp.stack((q_nn_lesser, q_nn_greater))

                w_nn, *__ = self.lyapunov((b_nn, q_nn), "right-" + str(stack_slice))
                w_nn_lesser, w_nn_greater = w_nn

                self.obc_blocks.lesser[-1] = m_mn @ w_nn_lesser @ m_mn.conj().swapaxes(
                    -1, -2
                ) - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))

                self.obc_blocks.greater[
                    -1
                ] = m_mn @ w_nn_greater @ m_mn.conj().swapaxes(-1, -2) - (
                    a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2)
                )

    @profiler.profile(
        label="CoulombScreeningSolver: Assembly", level="default", comm=comm
    )
    def _assemble_system_matrix(self, p_retarded: DSDBSparse) -> None:

        self.coulomb_matrix.to_device(
            delete_host=False, stream=self._system_stream, sync=False
        )

        """Assembles the system matrix."""
        self.system_matrix.data = 0.0
        local_blocks, _ = get_section_sizes(
            len(self.system_matrix.block_sizes), comm.block.size
        )
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        synchronize_stream(self._system_stream)
        bd_matmul_distr(
            self.coulomb_matrix,
            p_retarded,
            out=self.system_matrix,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
            ozaki=self.config.compute.w_assembly_ozaki,
            slices=self.config.compute.w_assembly_slices,
        )
        # TODO: inefficient
        if self.system_matrix.bits is None:
            xp.negative(self.system_matrix.data, out=self.system_matrix.data)
        else:
            tmp = xp.negative(
                decompress(self.system_matrix.data, self.system_matrix.bits)
            )
            compress(tmp, self.system_matrix.bits, out=self.system_matrix.data)

        self.system_matrix += sparse.eye(self.system_matrix.shape[-1])

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
        p_retarded: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Solves for the screened Coulomb interaction.

        Parameters
        ----------
        p_lesser : DSDBSparse
            The lesser polarization.
        p_greater : DSDBSparse
            The greater polarization.
        p_retarded : DSDBSparse
            The retarded polarization.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """

        if self.max_batch_size is None:
            max_batch_size = p_lesser.shape[0]
        else:
            max_batch_size = self.max_batch_size

        batch_sizes, batch_offsets = get_batches(p_lesser.shape[0], max_batch_size)

        for i in range(len(batch_sizes)):

            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))
            off_slice = slice(0, int(batch_offsets[i + 1] - batch_offsets[i]))

            reallocate = False
            if i > 0 and batch_sizes[i] != batch_sizes[i - 1]:
                reallocate = True

            with profiler.profile_range(
                label="CoulombScreeningSolver: Set block sizes",
                level="default",
                comm=comm,
            ):
                if reallocate:
                    self.system_matrix.free_data()
                    self.l_lesser.free_data()
                    self.l_greater.free_data()
                    # free_mempool()
                self.system_matrix.allocate_data(stack_size=batch_sizes[i])
                self.l_lesser.allocate_data(stack_size=batch_sizes[i])
                self.l_greater.allocate_data(stack_size=batch_sizes[i])

                p_lesser_tmp = p_lesser.stack[stack_slice]
                p_greater_tmp = p_greater.stack[stack_slice]
                p_retarded_tmp = p_retarded.stack[stack_slice]
                l_lesser_tmp = self.l_lesser.stack[off_slice]
                l_greater_tmp = self.l_greater.stack[off_slice]

                # Change the block sizes to match the Coulomb matrix.
                self._set_block_sizes(self.small_block_sizes)

            # Assemble the system matrix (Includes matrix multiplication).
            self._assemble_system_matrix(p_retarded_tmp)

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
                    p_lesser_tmp,
                    out=l_lesser_tmp,
                    start_block=start_block,
                    end_block=end_block,
                    spillover_correction=True,
                    ozaki=self.config.compute.w_assembly_ozaki,
                    slices=self.config.compute.w_assembly_slices,
                )
                bd_sandwich_distr(
                    self.coulomb_matrix,
                    p_greater_tmp,
                    out=l_greater_tmp,
                    start_block=start_block,
                    end_block=end_block,
                    spillover_correction=True,
                    ozaki=self.config.compute.w_assembly_ozaki,
                    slices=self.config.compute.w_assembly_slices,
                )

            self.coulomb_matrix.free_data()

            if self.flatband:
                with profiler.profile_range(
                    label="CoulombScreeningSolver: Homogenize",
                    level="default",
                    comm=comm,
                ):
                    homogenize(self.system_matrix)
                    homogenize(l_lesser_tmp)
                    homogenize(l_greater_tmp)

            with profiler.profile_range(
                label="CoulombScreeningSolver: Set block sizes back",
                level="default",
                comm=comm,
            ):
                # Go back to normal block sizes.
                self._set_block_sizes(self.block_sizes)

            # Apply the OBC algorithm.
            self._compute_obc(stack_slice)

            out_l, out_g = out
            out_slice = (
                out_l.stack[stack_slice],
                out_g.stack[stack_slice],
            )

            with profiler.profile_range(
                label="CoulombScreeningSolver: Solve", level="default", comm=comm
            ):
                # Solve the system
                if comm.block.size > 1:
                    self.solver_dist.selected_solve(
                        a=self.system_matrix,
                        sigma_lesser=l_lesser_tmp,
                        sigma_greater=l_greater_tmp,
                        obc_blocks=self.obc_blocks,
                        out=out_slice,
                        return_retarded=False,
                    )

                else:
                    self.solver.selected_solve(
                        a=self.system_matrix,
                        sigma_lesser=l_lesser_tmp,
                        sigma_greater=l_greater_tmp,
                        obc_blocks=self.obc_blocks,
                        out=out_slice,
                        return_retarded=False,
                        ozaki=self.config.compute.w_rgf_ozaki,
                        slices=self.config.compute.w_rgf_slices,
                    )

        with profiler.profile_range(
            label="CoulombScreeningSolver: Filter", level="default", comm=comm
        ):
            # Only filter the peaks for the first few iterations.
            # if self.solve_call_count < self.filtering_iteration_limit:
            #     self._filter_peaks(out)

            self.system_matrix.free_data()
            self.l_lesser.free_data()
            self.l_greater.free_data()

            w_lesser, w_greater, *__ = out
            if comm.stack.rank == 0:
                w_greater.data[0, :] = 0.0
                w_lesser.data[0, :] = 0.0

        self.solve_call_count += 1
