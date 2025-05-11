# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import os

import cupy as cp
import numpy as np

from qttools import QTX_USE_CUPY_JIT, NDArray, strtobool
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK
from qttools.profiling import Profiler

if QTX_USE_CUPY_JIT:
    from qttools.kernels.datastructure.cupy import _cupy_jit as cupy_backend
else:
    from qttools.kernels.datastructure.cupy import _cupy_rawkernel as cupy_backend


# NOTE: CUDA kernels are not profiled, as the jit-compiled kernels
# cannot find the correct name of the function to profile.
profiler = Profiler()

QTX_USE_DENSIFY_BLOCK = strtobool(os.getenv("QTX_USE_DENSIFY_BLOCK", "False"), False)


@profiler.profile(level="api")
def find_inds(
    self_rows: NDArray, self_cols: NDArray, rows: NDArray, cols: NDArray
) -> tuple[NDArray, NDArray, int]:
    """Finds the corresponding indices of the given rows and columns.

    Parameters
    ----------
    self_rows : NDArray
        The rows of this matrix.
    self_cols : NDArray
        The columns of this matrix.
    rows : NDArray
        The rows to find the indices for.
    cols : NDArray
        The columns to find the indices for.

    Returns
    -------
    inds : NDArray
        The indices of the given rows and columns.
    value_inds : NDArray
        The matching indices of this matrix.
    max_counts : int
        The maximum number of matches found.

    """
    rows = rows.astype(cp.int32)
    cols = cols.astype(cp.int32)
    full_inds = cp.zeros(self_rows.shape[0], dtype=cp.int32)
    counts = cp.zeros(self_rows.shape[0], dtype=cp.int16)
    THREADS_PER_BLOCK
    blocks_per_grid = (self_rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cupy_backend._find_inds(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            self_rows,
            self_cols,
            rows,
            cols,
            full_inds,
            counts,
            self_rows.shape[0],
            rows.shape[0],
        ),
    )

    # Find the valid indices.
    inds = cp.nonzero(counts)[0]
    value_inds = full_inds[inds]

    return inds, value_inds, int(cp.max(counts))


@profiler.profile(level="api")
def compute_block_slice(
    rows: NDArray, cols: NDArray, block_offsets: NDArray, row: int, col: int
) -> slice:
    """Computes the slice of the block in the data.

    Parameters
    ----------
    rows : NDArray
        The row indices of the matrix.
    cols : NDArray
        The column indices of the matrix.
    block_offsets : NDArray
        The offsets of the blocks.
    row : int
        The block row to compute the slice for.
    col : int
        The block column to compute the slice for.

    Returns
    -------
    start : int
        The start index of the block.
    stop : int
        The stop index of the block.

    """
    mask = cp.zeros(rows.shape[0], dtype=cp.int32)
    row_start, row_stop = np.int32(block_offsets[row]), np.int32(block_offsets[row + 1])
    col_start, col_stop = np.int32(block_offsets[col]), np.int32(block_offsets[col + 1])

    rows = rows.astype(cp.int32)
    cols = cols.astype(cp.int32)

    blocks_per_grid = (rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cupy_backend._compute_coo_block_mask(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            rows,
            cols,
            row_start,
            row_stop,
            col_start,
            col_stop,
            mask,
            np.int32(rows.shape[0]),
        ),
    )
    if cp.sum(mask) == 0:
        # No data in this block, return an empty slice.
        return None, None

    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.
    inds = cp.nonzero(mask)[0]

    # NOTE: this copies back to the host
    return int(inds[0]), int(inds[-1] + 1)


@profiler.profile(level="api")
def densify_block(
    block: NDArray,
    rows: NDArray,
    cols: NDArray,
    data: NDArray,
    block_slice: slice,
    row_offset: int,
    col_offset: int,
    use_kernel: bool = QTX_USE_DENSIFY_BLOCK,
):
    """Fills the dense block with the given data.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to fill the block with.
    block : NDArray
        Preallocated dense block. Should be filled with zeros.
    block_slice : slice
        The slice of the block to fill.
    row_offset : int
        The row offset of the block.
    col_offset : int
        The column offset of the block

    """

    # TODO: Needs profilig to see if this is faster than the raw kernel.
    if not use_kernel:
        block[..., rows[block_slice] - row_offset, cols[block_slice] - col_offset] = (
            data[..., block_slice]
        )

    else:
        THREADS_PER_BLOCK
        stack_size = data.size // data.shape[-1]
        stack_stride = data.shape[-1]
        block_start = block_slice.start or 0
        nnz_per_block = block_slice.stop - block_start
        num_blocks = (
            stack_size * nnz_per_block + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK
        cupy_backend._densify_block(
            (num_blocks,),
            (THREADS_PER_BLOCK,),
            (
                block.reshape(-1),
                rows,
                cols,
                data.reshape(-1),
                stack_size,
                stack_stride,
                nnz_per_block,
                block.shape[-2],
                block.shape[-1],
                block_start,
                row_offset,
                col_offset,
            ),
        )


@profiler.profile(level="api")
def sparsify_block(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray):
    """Fills the data with the given dense block.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    block : NDArray
        The dense block to sparsify.
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to be filled with the block.

    """
    # TODO: Test whether a custom kernel could be faster here.
    data[:] = block[..., rows, cols]


@profiler.profile(level="api")
def compute_block_sort_index(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> NDArray:
    """Computes the block-sorting index for a sparse matrix.

    Note
    ----
    Due to the Python for loop around the kernel, this method will
    perform best for larger block sizes (>500).

    Parameters
    ----------
    coo_rows : NDArray
        The row indices of the matrix in coordinate format.
    coo_cols : NDArray
        The column indices of the matrix in coordinate format.
    block_sizes : NDArray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : NDArray
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = block_sizes.shape[0]
    block_offsets = np.hstack((np.array([0]), np.cumsum(block_sizes)), dtype=np.int32)

    sort_index = cp.zeros(len(coo_cols), dtype=cp.int32)
    mask = cp.zeros(len(coo_cols), dtype=cp.int32)
    coo_rows = coo_rows.astype(cp.int32)
    coo_cols = coo_cols.astype(cp.int32)

    blocks_per_grid = (len(coo_cols) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    offset = 0

    for i, j in cp.ndindex(num_blocks, num_blocks):
        cupy_backend._compute_coo_block_mask(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                coo_rows,
                coo_cols,
                np.int32(block_offsets[i]),
                np.int32(block_offsets[i + 1]),
                np.int32(block_offsets[j]),
                np.int32(block_offsets[j + 1]),
                mask,
                np.int32(len(coo_cols)),
            ),
        )

        # NOTE: Fix for AMD cupy where cub was not used
        if QTX_USE_CUPY_JIT:
            bnnz = cp.sum(mask)
        else:
            bnnz = cupy_backend.reduction(mask)

        if bnnz != 0:
            # Sort the data by block-row and -column.
            sort_index[offset : offset + bnnz] = cp.nonzero(mask)[0]

            offset += bnnz

    return sort_index
