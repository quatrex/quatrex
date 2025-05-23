# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import time

import numpy as np

from qttools import NDArray
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver import _serinv
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import synchronize_device
from qttools.utils.solvers_utils import get_batches

profiler = Profiler()


@decorate_methods(profiler.profile(level="api"), exclude=["__init__"])
class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
    solve_lesser : bool, optional
        Whether to solve the quadratic system associated with the lesser right-hand-side,
        by default False.
    solve_greater : bool, optional
        Whether to solve the quadratic system associated with the greater right-hand-side,
        by default False.
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        100.

    """

    def __init__(self, max_batch_size: int = 100) -> None:
        """Initializes the selected inversion solver."""
        self.max_batch_size = max_batch_size

    def selected_inv(
        self,
        a: DSDBSparse,
        out: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
    ) -> None | DSDBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        out : DSDBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSDBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSDBSparse object.

        """

        # Initialize temporary buffers.
        reduced_system = _serinv.ReducedSystem()

        # Initialize dense temporary buffers for the diagonal blocks and
        # the upper and lower auxiliary buffer blocks.
        x_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_local_blocks)

        batch_sizes, batch_offsets = get_batches(a.shape[0], self.max_batch_size)

        for i in range(len(batch_sizes)):
            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            a_ = a.stack[stack_slice]
            out_ = out.stack[stack_slice]

            if comm.block.rank == 0:
                # Direction: downward Schur-complement
                _serinv.downward_schur(
                    a_,
                    x_diag_blocks,
                    obc_blocks,
                    stack_slice=stack_slice,
                    invert_last_block=False,
                )
            elif comm.block.rank == comm.block.size - 1:
                # Direction: upward Schur-complement
                _serinv.upward_schur(
                    a_,
                    x_diag_blocks,
                    obc_blocks,
                    stack_slice=stack_slice,
                    invert_last_block=False,
                )
            else:
                # Permuted Schur-complement
                _serinv.permuted_schur(
                    a_,
                    x_diag_blocks,
                    buffer_lower,
                    buffer_upper,
                    obc_blocks,
                    stack_slice=stack_slice,
                )

            # Construct the reduced system.
            if np.all(a.block_sizes == a.block_sizes[0]):
                gather_reduced_system = reduced_system.gather_constant_block_size
            else:
                # If the block sizes are not the same, we need to use pickle.
                gather_reduced_system = reduced_system.gather

            gather_reduced_system(a_, x_diag_blocks, buffer_upper, buffer_lower)
            # Perform selected-inversion on the reduced system.
            reduced_system.solve()
            # Scatter the result to the output matrix.
            reduced_system.scatter(x_diag_blocks, buffer_upper, buffer_lower, out_)

            if comm.block.rank == 0:
                # Direction: upward sell-inv
                _serinv.downward_selinv(a_, x_diag_blocks, out_)
            elif comm.block.rank == comm.block.size - 1:
                # Direction: downward sell-inv
                _serinv.upward_selinv(a_, x_diag_blocks, out_)
            else:
                # Permuted Sell-inv
                _serinv.permuted_selinv(
                    a_, x_diag_blocks, buffer_lower, buffer_upper, out_
                )

    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
    ):
        """Performs selected inversion of a block-tridiagonal matrix.

        Can optionally solve the quadratic system associated with the
        Bl and Bg matrices in the equation AXA^T = B.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        sigma_lesser : DSDBSparse
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSDBSparse, ...]
            Preallocated output matrices, by default None
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False

        """

        t_init_start = time.perf_counter()

        # Initialize temporary buffers.
        reduced_system = _serinv.ReducedSystem(selected_solve=True)

        xr_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        xr_buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        xr_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        xl_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        xl_buffer_lower = None
        xl_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        xg_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        xg_buffer_lower = None
        xg_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_local_blocks)

        xl_out, xg_out, *xr_out = out
        if return_retarded:
            if len(xr_out) != 1:
                raise ValueError("Invalid number of output matrices.")
            xr_out = xr_out[0]

        batch_sizes, batch_offsets = get_batches(a.shape[0], self.max_batch_size)

        for i in range(len(batch_sizes)):
            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xl_out_ = xl_out.stack[stack_slice]
            xg_out_ = xg_out.stack[stack_slice]
            xr_out_ = xr_out.stack[stack_slice] if return_retarded else None

        synchronize_device()
        t_init_end = time.perf_counter()
        comm.barrier()
        t_init_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"        Init: {t_init_end-t_init_start}", flush=True)
            print(f"        Init all: {t_init_end_all-t_init_start}", flush=True)

        t_schur_start = time.perf_counter()

        if comm.block.rank == 0:
            # Direction: downward Schur-complement
            _serinv.downward_schur(
                a=a_,
                xr_diag_blocks=xr_diag_blocks,
                # Lesser quantities.
                sigma_lesser=sigma_lesser_,
                xl_diag_blocks=xl_diag_blocks,
                # Greater quantities.
                sigma_greater=sigma_greater_,
                xg_diag_blocks=xg_diag_blocks,
                # OBC and settings.
                obc_blocks=obc_blocks,
                stack_slice=stack_slice,
                invert_last_block=False,
                selected_solve=True,
            )
        elif comm.block.rank == comm.block.size - 1:
            # Direction: upward Schur-complement
            _serinv.upward_schur(
                a=a_,
                xr_diag_blocks=xr_diag_blocks,
                # Lesser quantities.
                sigma_lesser=sigma_lesser_,
                xl_diag_blocks=xl_diag_blocks,
                # Greater quantities.
                sigma_greater=sigma_greater_,
                xg_diag_blocks=xg_diag_blocks,
                # OBC and settings.
                obc_blocks=obc_blocks,
                stack_slice=stack_slice,
                invert_last_block=False,
                selected_solve=True,
            )
        else:
            # Permuted Schur-complement
            _serinv.permuted_schur(
                a=a_,
                xr_diag_blocks=xr_diag_blocks,
                xr_buffer_lower=xr_buffer_lower,
                xr_buffer_upper=xr_buffer_upper,
                # Lesser quantities.
                sigma_lesser=sigma_lesser_,
                xl_diag_blocks=xl_diag_blocks,
                xl_buffer_lower=xl_buffer_lower,
                xl_buffer_upper=xl_buffer_upper,
                # Greater quantities.
                sigma_greater=sigma_greater_,
                xg_diag_blocks=xg_diag_blocks,
                xg_buffer_lower=xg_buffer_lower,
                xg_buffer_upper=xg_buffer_upper,
                # OBC and settings.
                obc_blocks=obc_blocks,
                stack_slice=stack_slice,
                selected_solve=True,
            )

        synchronize_device()
        t_schur_end = time.perf_counter()
        comm.barrier()
        t_schur_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"        Schur: {t_schur_end-t_schur_start}", flush=True)
            print(f"        Schur all: {t_schur_end_all-t_schur_start}", flush=True)

        t_reduce_gather_start = time.perf_counter()

        # Construct the reduced system.
        if np.all(a.block_sizes == a.block_sizes[0]):
            gather_reduced_system = reduced_system.gather_constant_block_size
        else:
            # If the block sizes are not the same, we need to use pickle.
            gather_reduced_system = reduced_system.gather

        synchronize_device()
        comm.barrier()
        gather_reduced_system(
            a=a_,
            xr_diag_blocks=xr_diag_blocks,
            xr_buffer_lower=xr_buffer_lower,
            xr_buffer_upper=xr_buffer_upper,
            # Lesser quantities.
            sigma_lesser=sigma_lesser_,
            xl_diag_blocks=xl_diag_blocks,
            xl_buffer_lower=xl_buffer_lower,
            xl_buffer_upper=xl_buffer_upper,
            # Greater quantities.
            sigma_greater=sigma_greater_,
            xg_diag_blocks=xg_diag_blocks,
            xg_buffer_lower=xg_buffer_lower,
            xg_buffer_upper=xg_buffer_upper,
        )
        synchronize_device()
        t_reduce_gather_end = time.perf_counter()
        comm.barrier()
        t_reduce_gather_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"        Reduced gather: {t_reduce_gather_end-t_reduce_gather_start}",
                flush=True,
            )
            print(
                f"        Reduced gather all: {t_reduce_gather_end_all-t_reduce_gather_start}",
                flush=True,
            )

        t_reduce_solve_start = time.perf_counter()

        # Perform selected-inversion on the reduced system.
        reduced_system.solve()

        synchronize_device()
        t_reduce_solve_end = time.perf_counter()
        comm.barrier()
        t_reduce_solve_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"        Reduced solve: {t_reduce_solve_end-t_reduce_solve_start}",
                flush=True,
            )
            print(
                f"        Reduced solve all: {t_reduce_solve_end_all-t_reduce_solve_start}",
                flush=True,
            )

        t_reduce_scatter_start = time.perf_counter()

        # Scatter the result to the output matrix.
        reduced_system.scatter(
            xr_diag_blocks=xr_diag_blocks,
            xr_buffer_lower=xr_buffer_lower,
            xr_buffer_upper=xr_buffer_upper,
            xr_out=xr_out_,
            return_retarded=return_retarded,
            # Lesser quantities.
            xl_diag_blocks=xl_diag_blocks,
            xl_buffer_lower=xl_buffer_lower,
            xl_buffer_upper=xl_buffer_upper,
            xl_out=xl_out_,
            # Greater quantities.
            xg_diag_blocks=xg_diag_blocks,
            xg_buffer_lower=xg_buffer_lower,
            xg_buffer_upper=xg_buffer_upper,
            xg_out=xg_out_,
        )

        synchronize_device()
        t_reduce_scatter_end = time.perf_counter()
        comm.barrier()
        t_reduce_scatter_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"        Reduced scatter: {t_reduce_scatter_end-t_reduce_scatter_start}",
                flush=True,
            )
            print(
                f"        Reduced scatter all: {t_reduce_scatter_end_all-t_reduce_scatter_start}",
                flush=True,
            )

        t_selinv_start = time.perf_counter()

        if comm.block.rank == 0:
            # Direction: upward sell-inv
            _serinv.downward_selinv(
                a=a_,
                xr_diag_blocks=xr_diag_blocks,
                xr_out=xr_out_,
                # Lesser quantities.
                sigma_lesser=sigma_lesser_,
                xl_diag_blocks=xl_diag_blocks,
                xl_out=xl_out_,
                # Greater quantities.
                sigma_greater=sigma_greater_,
                xg_diag_blocks=xg_diag_blocks,
                xg_out=xg_out_,
                selected_solve=True,
                return_retarded=return_retarded,
            )
        elif comm.block.rank == comm.block.size - 1:
            # Direction: downward sell-inv
            _serinv.upward_selinv(
                a=a_,
                xr_diag_blocks=xr_diag_blocks,
                xr_out=xr_out_,
                # Lesser quantities.
                sigma_lesser=sigma_lesser_,
                xl_diag_blocks=xl_diag_blocks,
                xl_out=xl_out_,
                # Greater quantities.
                sigma_greater=sigma_greater_,
                xg_diag_blocks=xg_diag_blocks,
                xg_out=xg_out_,
                selected_solve=True,
                return_retarded=return_retarded,
            )
        else:
            # Permuted Sell-inv
            _serinv.permuted_selinv(
                a=a_,
                xr_diag_blocks=xr_diag_blocks,
                xr_buffer_lower=xr_buffer_lower,
                xr_buffer_upper=xr_buffer_upper,
                xr_out=xr_out_,
                # Lesser quantities.
                sigma_lesser=sigma_lesser_,
                xl_diag_blocks=xl_diag_blocks,
                # xl_buffer_lower=xl_buffer_lower,
                xl_buffer_upper=xl_buffer_upper,
                xl_out=xl_out_,
                # Greater quantities.
                sigma_greater=sigma_greater_,
                xg_diag_blocks=xg_diag_blocks,
                # xg_buffer_lower=xg_buffer_lower,
                xg_buffer_upper=xg_buffer_upper,
                xg_out=xg_out_,
                selected_solve=True,
                return_retarded=return_retarded,
            )

        synchronize_device()
        t_selinv_end = time.perf_counter()
        comm.barrier()
        t_selinv_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"        Selinv: {t_selinv_end-t_selinv_start}", flush=True)
            print(f"        Selinv all: {t_selinv_end_all-t_selinv_start}", flush=True)
