# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(parallel=True, cache=True, no_rewrites=True)
def find_inds(
    self_rows: NDArray, self_cols: NDArray, rows: NDArray, cols: NDArray
) -> tuple[NDArray, NDArray]:
    """Finds the corresponding indices of the given rows and columns.

    This also counts the number of matches found, which is used to check
    if the indices contain duplicates.

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
    full_inds = np.zeros(self_rows.shape[0], dtype=np.int32)
    counts = np.zeros(self_rows.shape[0], dtype=np.int16)
    for i in nb.prange(self_rows.shape[0]):
        for j in range(rows.shape[0]):
            cond = int((self_rows[i] == rows[j]) & (self_cols[i] == cols[j]))
            full_inds[i] = full_inds[i] * (1 - cond) + j * cond
            counts[i] += cond

    # Find the valid indices.
    inds = np.nonzero(counts)[0]
    value_inds = full_inds[inds]

    return inds, value_inds, np.max(counts)


@nb.njit(parallel=True, cache=True)
def compute_block_slice(
    rows: NDArray, cols: NDArray, block_offsets: NDArray, row: int, col: int
):
    """Computes the slice in the data for the given block.

    Parameters
    ----------
    rows : NDArray
        The rows in the COO matrix.
    cols : NDArray
        The columns in the COO matrix.
    block_offsets : NDArray
        The block offsets.
    row : int
        THe block row.
    col : int
        The block column.

    Returns
    -------
    start : int
        The start index of the block.
    stop : int
        The stop index of the block.

    """
    mask = np.zeros(rows.shape[0], dtype=np.bool_)
    row_start, row_stop = block_offsets[row], block_offsets[row + 1]
    col_start, col_stop = block_offsets[col], block_offsets[col + 1]
    for i in nb.prange(rows.shape[0]):
        mask[i] = (
            (rows[i] >= row_start)
            & (rows[i] < row_stop)
            & (cols[i] >= col_start)
            & (cols[i] < col_stop)
        )

    if np.sum(mask) == 0:
        # No data in this block, return an empty slice.
        return None, None

    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.
    inds = np.nonzero(mask)[0]
    return inds[0], inds[-1] + 1


@nb.njit(parallel=True, cache=True)
def densify_block(
    block: NDArray,
    rows: NDArray,
    cols: NDArray,
    data: NDArray,
):
    """Fills the dense block with the given data.

    Note
    ----
    If the blocks to be densified get very small, the overhead of
    starting the CPU threads can lead to worse performance in the jitted
    version than in the bare API implementation.

    Parameters
    ----------
    block : NDArray
        Preallocated dense block. Should be filled with zeros.
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to fill the block with.

    """
    for i in nb.prange(rows.shape[0]):
        block[..., rows[i], cols[i]] = data[..., i]


@nb.njit(parallel=True, cache=True)
def sparsify_block(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray):
    """Fills the data with the given dense block.

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
    for i in nb.prange(rows.shape[0]):
        data[..., i] = block[..., rows[i], cols[i]]


@nb.njit(parallel=True, cache=True)
def compute_block_sort_index(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> NDArray:
    """Computes the block-sorting index for a sparse matrix.

    Note
    ----
    This method incurs a bit of memory overhead compared to a naive
    implementation. No assumptions on the sparsity pattern of the matrix
    are made here. See the source code for more details.

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
    block_offsets = np.hstack((np.array([0]), np.cumsum(block_sizes)))

    sort_index = np.zeros(len(coo_cols), dtype=np.int32)

    # NOTE: This is a very generous estimate of the number of
    # nonzeros in each row of blocks. No assumption on the sparsity
    # pattern of the matrix is made here.
    nnz_estimate = min(len(coo_cols), max(block_sizes) ** 2)
    inds = np.zeros((num_blocks, nnz_estimate), dtype=np.int32)

    block_nnz = np.zeros(num_blocks, dtype=np.int32)
    nnz_offset = 0
    for i in range(num_blocks):
        # Precompute the row mask.
        row_mask = (block_offsets[i] <= coo_rows) & (coo_rows < block_offsets[i + 1])

        # Process in parallel.
        for j in nb.prange(num_blocks):
            mask = (
                row_mask
                & (block_offsets[j] <= coo_cols)
                & (coo_cols < block_offsets[j + 1])
            )
            nnz = np.sum(mask)
            block_nnz[j] = nnz
            if nnz > 0:
                inds[j, :nnz] = np.nonzero(mask)[0]

        # Reduce the indices sequentially.
        for j in range(num_blocks):
            nnz = block_nnz[j]
            if nnz > 0:
                # Sort the data by block-row and -column.
                sort_index[nnz_offset : nnz_offset + nnz] = inds[j, :nnz]
                nnz_offset += nnz

    return sort_index
