# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(parallel=True, cache=True)
def _find_bcoords(block_offsets: NDArray, rows: NDArray, cols: NDArray) -> NDArray:
    """Finds the block coordinates of the given rows and columns.

    Parameters
    ----------
    block_offsets : NDArray
        The offsets of the blocks.
    rows : NDArray
        The row indices.
    cols : NDArray
        The column indices.

    Returns
    -------
    brows : NDArray
        The block row indices.
    bcols : NDArray
        The block column indices.

    """
    brows = np.zeros(rows.shape[0], dtype=np.int16)
    bcols = np.zeros(cols.shape[0], dtype=np.int16)
    for i in nb.prange(rows.shape[0]):
        for j in range(block_offsets.shape[0]):
            cond_rows = int(block_offsets[j] <= rows[i])
            brows[i] = brows[i] * (1 - cond_rows) + j * cond_rows
            cond_cols = int(block_offsets[j] <= cols[i])
            bcols[i] = bcols[i] * (1 - cond_cols) + j * cond_cols

    return brows, bcols


@nb.njit(parallel=True, cache=True)
def _find_block_inds(
    brows: NDArray,
    bcols: NDArray,
    brow: int,
    bcol: int,
    rows: NDArray,
    cols: NDArray,
    block_offsets: NDArray,
    self_cols: NDArray,
    rowptr: NDArray,
):
    """Finds the indices of the given block.

    Parameters
    ----------
    brows : NDArray
        The block row indices.
    bcols : NDArray
        The block column indices.
    brow : int
        The block row.
    bcol : int
        The block column.
    rows : NDArray
        The requested row indices.
    cols : NDArray
        The requested column indices.
    block_offsets : NDArray
        The block offsets.
    self_cols : NDArray
        The column indices of this matrix.
    rowptr : NDArray
        The row pointer of this matrix block.

    Returns
    -------
    inds : NDArray
        The indices of the given block.
    value_inds : NDArray
        The matching indices of this matrix.

    """
    mask = np.zeros(brows.shape[0], dtype=np.bool_)
    for i in nb.prange(rows.shape[0]):
        mask[i] = (brows[i] == brow) & (bcols[i] == bcol)

    mask_inds = np.nonzero(mask)[0]

    # Renormalize the row indices for this block.
    rr = rows[mask] - block_offsets[brow]
    cc = cols[mask]

    inds = np.zeros(rr.shape[0], dtype=np.int32) - 1
    for i in nb.prange(rr.shape[0]):
        r = rr[i]
        ind = np.nonzero(self_cols[rowptr[r] : rowptr[r + 1]] == cc[i])[0]
        if len(ind) == 0:
            continue
        inds[i] = rowptr[r] + ind[0]

    valid = inds != -1
    return inds[valid], mask_inds[valid]


def find_inds(
    rowptr_map: dict[tuple, NDArray],
    block_offsets: NDArray,
    self_cols: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> tuple[NDArray, NDArray]:
    """Finds the corresponding indices of the given rows and columns.

    Parameters
    ----------
    rowptr_map : dict
        The row pointer map.
    block_offsets : NDArray
        The block offsets.
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

    """
    brows, bcols = _find_bcoords(block_offsets, rows, cols)

    # Get an ordered list of unique blocks.
    unique_blocks = dict.fromkeys(zip(map(int, brows), map(int, bcols))).keys()

    inds, value_inds = [], []
    for brow, bcol in unique_blocks:
        rowptr = rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            continue

        block_inds, block_value_inds = _find_block_inds(
            brows=brows,
            bcols=bcols,
            brow=brow,
            bcol=bcol,
            rows=rows,
            cols=cols,
            block_offsets=block_offsets,
            self_cols=self_cols,
            rowptr=rowptr,
        )

        inds.extend(block_inds)
        value_inds.extend(block_value_inds)

    return np.array(inds, dtype=int), np.array(value_inds, dtype=int)


@nb.njit(parallel=True, cache=True)
def densify_block(
    block: NDArray,
    block_offset: NDArray,
    self_cols: NDArray,
    rowptr: NDArray,
    data: NDArray,
):
    """Fills the dense block with the given data.

    Parameters
    ----------
    block : NDArray
        Preallocated dense block. Should be filled with zeros.
    block_offset : NDArray
        The block offset.
    self_cols : NDArray
        The column indices of this matrix.
    rowptr : NDArray
        The row pointer of this matrix block.
    data : NDArray
        The data to fill the block with.

    """
    for i in nb.prange(rowptr.shape[0] - 1):
        cols = self_cols[rowptr[i] : rowptr[i + 1]] - block_offset
        block[..., i, cols] = data[..., rowptr[i] : rowptr[i + 1]]


@nb.njit(parallel=True, cache=True)
def sparsify_block(
    block: NDArray,
    block_offset: NDArray,
    self_cols: NDArray,
    rowptr: NDArray,
    data: NDArray,
):
    """Fills the data with the given dense block.

    Parameters
    ----------
    block : NDArray
        The dense block to sparsify.
    block_offset : NDArray
        The block offset.
    self_cols : NDArray
        The column indices of this matrix.
    rowptr : NDArray
        The row pointer of this matrix block.
    data : NDArray
        The data to be filled with the block.

    """
    for i in nb.prange(rowptr.shape[0] - 1):
        cols = self_cols[rowptr[i] : rowptr[i + 1]] - block_offset
        data[..., rowptr[i] : rowptr[i + 1]] = block[..., i, cols]


# @nb.njit(parallel=True, cache=True)
def _compute_rowptr_map_kernel(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> nb.typed.Dict:
    """Computes the block-sorting index and the rowptr map.

    Note
    ----
    This is a combination of the bare block-sorting index computation
    and the rowptr map computation. This returns a Numba typed
    dictionary (not a pure Python dictionary) so it has to be typecast.

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
        The block-sorting index for the sparse matrix.
    rowptr_map : nb.typed.Dict
        The row pointer map, describing the block-sparse matrix in
        blockwise column-sparse-row format.

    """
    num_blocks = block_sizes.shape[0]
    block_offsets = np.hstack((np.array([0]), np.cumsum(block_sizes)))

    nnz_offset = 0
    sort_index = np.zeros(len(coo_cols), dtype=np.int32)
    rowptr_map = {}

    block_nnz = np.zeros(num_blocks, dtype=np.int32)

    # NOTE: This is a very generous estimate of the number of
    # nonzeros in each row of blocks. No assumption on the sparsity
    # pattern of the matrix is made here.
    nnz_estimate = min(len(coo_cols), max(block_sizes) ** 2)
    inds = np.zeros((num_blocks, nnz_estimate), dtype=np.int32)

    for i in range(num_blocks):
        # Precompute the row mask.
        row_mask = (block_offsets[i] <= coo_rows) & (coo_rows < block_offsets[i + 1])
        hists = np.zeros((num_blocks, block_sizes[i]), dtype=np.int32)
        bins = np.arange(block_sizes[i] + 1)
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
                # Compute the block-sorting index.
                inds[j, :nnz] = np.nonzero(mask)[0]

                # Compute the rowptr map.
                hists[j, :] = np.histogram(
                    coo_rows[mask] - block_offsets[i], bins=bins
                )[0]

        # Reduce the indices sequentially.
        for j in range(num_blocks):
            nnz = block_nnz[j]
            if nnz > 0:
                sort_index[nnz_offset : nnz_offset + nnz] = inds[j, :nnz]

                rowptr = np.hstack((np.array([0]), np.cumsum(hists[j]))) + nnz_offset
                rowptr_map[(i, j)] = rowptr

                nnz_offset += nnz

    return sort_index, rowptr_map


def compute_rowptr_map(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> dict:
    """Computes the rowptr map for a sparse matrix.

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
        The block-sorting index for the sparse matrix.
    rowptr_map : dict
        The row pointer map, describing the block-sparse matrix in
        blockwise column-sparse-row format.

    """
    sort_index, rowptr_map = _compute_rowptr_map_kernel(coo_rows, coo_cols, block_sizes)
    return sort_index, dict(rowptr_map)
