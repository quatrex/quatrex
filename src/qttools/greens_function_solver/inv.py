# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the selected inversion solver based on dense inversion."""

from collections.abc import Callable

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.kernels import linalg
from qttools.utils.solvers_utils import get_batches


class Inv(GFSolver):
    """Selected inversion solver based on dense matrix inversion.

    Warning
    -------
    This solver will densify the matrix to invert it. It is intended as
    a reference implementation and should not be used in production
    code.

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

        This method will densify the matrix, invert it, and then select
        the elements to keep by matching the sparse structure of the
        input matrix.

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
        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        # Allocate batching buffer
        inv_a = xp.zeros((max(batches_sizes), *a.shape[1:]), dtype=a.dtype)

        rows, cols = out.spy()

        # Perform the inversion in batches
        for i in range(len(batches_sizes)):
            stack_slice = slice(batches_slices[i], batches_slices[i + 1], 1)
            a_dense = a.to_dense()[stack_slice]

            # Assemble the OBC blocks.
            for j, block in enumerate(obc_blocks.retarded):
                if block is None:
                    continue
                b_ = slice(a.block_offsets[j], a.block_offsets[j + 1], 1)
                a_dense[:, b_, b_] -= block[stack_slice]

            inv_a[: batches_sizes[i]] = linalg.inv(a_dense)

            out.data[stack_slice] = inv_a[: batches_sizes[i], ..., rows, cols]

    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
        callbacks: list[Callable] | None = None,
    ) -> None | NDArray:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

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
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        a_hat : DSDBSparse, optional
            The bare system matrix without self-energy contributions.
            This is used to compute the device current.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        callbacks : list[Callable], optional
            List of callback functions. Not supported in this solver.

        """
        if callbacks is not None:
            raise NotImplementedError("The use of callbacks is not implemented.")

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        # Allocate batching buffer
        x_r = xp.zeros((max(batches_sizes), *a.shape[1:]), dtype=a.dtype)
        x_l = xp.zeros((max(batches_sizes), *a.shape[1:]), dtype=a.dtype)
        x_g = xp.zeros((max(batches_sizes), *a.shape[1:]), dtype=a.dtype)

        # Get output datastructures
        sel_x_l, sel_x_g, *sel_x_r = out
        if return_retarded:
            if len(sel_x_r) == 0:
                raise ValueError("Missing output for the retarded Green's function.")
            sel_x_r = sel_x_r[0]
        rows_l, cols_l = sel_x_l.spy()
        rows_g, cols_g = sel_x_g.spy()
        if return_retarded:
            rows_r, cols_r = sel_x_r.spy()

        # Perform the inversion in batches
        for i in range(len(batches_sizes)):
            stack_slice = slice(batches_slices[i], batches_slices[i + 1], 1)
            a_dense = a.to_dense()[stack_slice]
            sigma_lesser_dense = sigma_lesser.to_dense()[stack_slice]
            sigma_greater_dense = sigma_greater.to_dense()[stack_slice]

            # Assemble the OBC blocks.
            for j, (block_r, block_l, block_g) in enumerate(
                zip(obc_blocks.retarded, obc_blocks.lesser, obc_blocks.greater)
            ):
                b_ = slice(a.block_offsets[j], a.block_offsets[j + 1], 1)
                if block_r is not None:
                    a_dense[:, b_, b_] -= block_r[stack_slice]
                if block_l is not None:
                    sigma_lesser_dense[:, b_, b_] -= block_l[stack_slice]
                if block_g is not None:
                    sigma_greater_dense[:, b_, b_] -= block_g[stack_slice]

            x_r[: batches_sizes[i]] = linalg.inv(a_dense)
            x_l[: batches_sizes[i]] = (
                x_r[: batches_sizes[i]]
                @ sigma_lesser_dense
                @ x_r[: batches_sizes[i]].conj().swapaxes(-2, -1)
            )
            x_g[: batches_sizes[i]] = (
                x_r[: batches_sizes[i]]
                @ sigma_greater_dense
                @ x_r[: batches_sizes[i]].conj().swapaxes(-2, -1)
            )

            # Store the dense batches in the DSDBSparse datastructures
            sel_x_l.data[stack_slice,] = x_l[: batches_sizes[i], ..., rows_l, cols_l]
            sel_x_g.data[stack_slice,] = x_g[: batches_sizes[i], ..., rows_g, cols_g]
            if return_retarded:
                sel_x_r.data[stack_slice,] = x_r[
                    : batches_sizes[i], ..., rows_r, cols_r
                ]
