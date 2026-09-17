# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the distributed selected inversion solver."""

from collections.abc import Callable

import numpy as np

from qttools import NDArray
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, _DStackView
from qttools.greens_function_solver import _serinv
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.profiling import Profiler
from qttools.utils.solvers_utils import get_batches

profiler = Profiler()


class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
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
    ) -> None:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        out : DSDBSparse
            Preallocated output matrix.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.

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
        a: DSDBSparse | _DStackView,
        sigma_lesser: DSDBSparse | _DStackView,
        sigma_greater: DSDBSparse | _DStackView,
        out: tuple[DSDBSparse, ...] | tuple[_DStackView, ...],
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
        callbacks: list[Callable] | None = None,
    ) -> None:
        r"""Performs selected inversion of a block-tridiagonal matrix.

        Can optionally solve the quadratic system associated with the
        lesser and greater right-hand-sides.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        sigma_lesser : DSDBSparse
            Lesser matrix. This matrix is expected to be skew-hermitian,
            i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSDBSparse, ...]
            Preallocated output matrices
        obc_blocks : dict[int, OBCBlocks], optional
            OBC blocks for lesser, greater and retarded Green's
            functions, by default None.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False.
        callbacks : list[Callable], optional
            List of callback functions to be called during the back
            substitution step. Each callback function should accept a
            single argument of type `BackwardSubstitutionContext`, by
            default None.

        """

        with profiler.profile_range(
            label="RGF dist: init", level="default", comm=comm.block
        ):

            if obc_blocks is None:
                obc_blocks = OBCBlocks(num_blocks=sigma_lesser.num_local_blocks)

            xl_out, xg_out, *xr_out = out
            if return_retarded:
                if len(xr_out) != 1:
                    raise ValueError("Invalid number of output matrices.")
                xr_out = xr_out[0]

            if xl_out.symmetry not in [None, "skew-hermitian"]:
                raise ValueError(
                    "Invalid symmetry for lesser Green's function. "
                    "Expected None or 'skew-hermitian'."
                )
            if xg_out.symmetry not in [None, "skew-hermitian"]:
                raise ValueError(
                    "Invalid symmetry for greater Green's function. "
                    "Expected None or 'skew-hermitian'."
                )

            batch_sizes, batch_offsets = get_batches(
                sigma_lesser.local_stack_shape[0], self.max_batch_size
            )

        for i in range(len(batch_sizes)):

            # Initialize temporary buffers.
            reduced_system = _serinv.ReducedSystem(selected_solve=True)

            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))

            serinv_context = {
                # Input quantities.
                "a": a.stack[stack_slice],
                "sigma_lesser": sigma_lesser.stack[stack_slice],
                "sigma_greater": sigma_greater.stack[stack_slice],
                # Retarded buffers.
                "xr_diag_blocks": [None] * sigma_lesser.num_local_blocks,
                "xr_buffer_lower": [None] * sigma_lesser.num_local_blocks,
                "xr_buffer_upper": [None] * sigma_lesser.num_local_blocks,
                # Lesser buffers.
                "xl_diag_blocks": [None] * sigma_lesser.num_local_blocks,
                "xl_buffer_lower": None,
                "xl_buffer_upper": [None] * sigma_lesser.num_local_blocks,
                # Greater buffers.
                "xg_diag_blocks": [None] * sigma_lesser.num_local_blocks,
                "xg_buffer_lower": None,
                "xg_buffer_upper": [None] * sigma_lesser.num_local_blocks,
                # Output quantities.
                "xr_out": xr_out.stack[stack_slice] if return_retarded else None,
                "xl_out": xl_out.stack[stack_slice],
                "xg_out": xg_out.stack[stack_slice],
                # OBC, settings and callbacks.
                "obc_blocks": obc_blocks,
                "stack_slice": stack_slice,
                "invert_last_block": False,
                "selected_solve": True,
                "return_retarded": return_retarded,
                "callbacks": callbacks,
            }

            with profiler.profile_range(
                label="RGF dist: Schur", level="default", comm=comm.block
            ):
                if comm.block.rank == 0:
                    _serinv.downward_schur(**serinv_context)
                elif comm.block.rank == comm.block.size - 1:
                    _serinv.upward_schur(**serinv_context)
                else:
                    _serinv.permuted_schur(**serinv_context)

            with profiler.profile_range(
                label="RGF dist: Reduce gather", level="default", comm=comm.block
            ):
                # Construct the reduced system.
                if np.all(a.block_sizes == a.block_sizes[0]):
                    reduced_system.gather_constant_block_size(**serinv_context)
                else:
                    # If the block sizes are not the same, we need to use pickle.
                    reduced_system.gather(**serinv_context)

            # Perform selected-inversion on the reduced system.
            with profiler.profile_range(
                label="RGF dist: Reduce solve", level="default", comm=comm.block
            ):
                reduced_system.solve()

            with profiler.profile_range(
                label="RGF dist: Reduce scatter", level="default", comm=comm.block
            ):
                # Scatter the result to the output matrix.
                reduced_system.scatter(**serinv_context)

            with profiler.profile_range(
                label="RGF dist: Selinv", level="default", comm=comm.block
            ):

                if comm.block.rank == 0:
                    _serinv.downward_selinv(**serinv_context)
                elif comm.block.rank == comm.block.size - 1:
                    _serinv.upward_selinv(**serinv_context)
                else:
                    _serinv.permuted_selinv(**serinv_context)
