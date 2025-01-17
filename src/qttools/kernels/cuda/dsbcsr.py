# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import cupy as cp
from cupyx import jit

from qttools import NDArray
from qttools.kernels.cuda import THREADS_PER_BLOCK
from qttools.kernels.cuda.dsbcoo import _compute_coo_block_mask_kernel


@jit.rawkernel()
def _find_bcoords_kernel(
    block_offsets: NDArray,
    rows: NDArray,
    cols: NDArray,
    brows: NDArray,
    bcols: NDArray,
):
    """Finds the block coordinates of the given rows and columns.

    Parameters
    ----------
    block_offsets : NDArray
        The offsets of the blocks.
    rows : NDArray
        The row indices.
    cols : NDArray
        The column indices.
    brows : NDArray
        The block row indices.
    bcols : NDArray
        The block column indices.

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < rows.shape[0]:
        for j in range(block_offsets.shape[0]):
            cond_rows = int(block_offsets[j] <= rows[i])
            brows[i] = brows[i] * (1 - cond_rows) + j * cond_rows
            cond_cols = int(block_offsets[j] <= cols[i])
            bcols[i] = bcols[i] * (1 - cond_cols) + j * cond_cols


@jit.rawkernel()
def _compute_block_mask_kernel(
    brows: NDArray,
    bcols: NDArray,
    brow: int,
    bcol: int,
    mask: NDArray,
):
    """Computes the mask for the given block coordinates.

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
    mask : NDArray
        The mask for the given block coordinates.

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < brows.shape[0]:
        mask[i] = (brows[i] == brow) & (bcols[i] == bcol)


@jit.rawkernel()
def _compute_block_inds_kernel(
    rr: NDArray,
    cc: NDArray,
    self_cols: NDArray,
    rowptr: NDArray,
    block_inds: NDArray,
):
    """Finds the indices of the given block.

    Parameters
    ----------
    rr : NDArray
        The row indices.
    cc : NDArray
        The column indices.
    self_cols : NDArray
        The columns of this matrix.
    rowptr : NDArray
        The row pointer.
    block_inds : NDArray
        The indices of the given block

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < rr.shape[0]:
        r = rr[i]
        ind = -1
        for j in range(rowptr[r], rowptr[r + 1]):
            cond = int(self_cols[j] == cc[i])
            ind = ind * (1 - cond) + j * cond

        block_inds[i] = ind


def find_inds(
    rowptr_map: dict,
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
    brows = cp.zeros(rows.shape[0], dtype=cp.int16)
    bcols = cp.zeros(cols.shape[0], dtype=cp.int16)

    bcoords_blocks_per_grid = (
        rows.shape[0] + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    _find_bcoords_kernel(
        (bcoords_blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (block_offsets, rows, cols, brows, bcols),
    )
    # Get an ordered list of unique blocks.
    unique_blocks = dict.fromkeys(zip(map(int, brows), map(int, bcols))).keys()

    block_mask_blocks_per_grid = (
        brows.shape[0] + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    inds, value_inds = [], []
    for brow, bcol in unique_blocks:
        rowptr = rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            continue
        mask = cp.zeros(brows.shape[0], dtype=cp.bool_)
        _compute_block_mask_kernel(
            (block_mask_blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (brows, bcols, brow, bcol, mask),
        )
        mask_inds = cp.nonzero(mask)[0]

        # Renormalize the row indices for this block.
        rr = rows[mask] - block_offsets[brow]
        cc = cols[mask]

        block_inds = cp.zeros(rr.shape[0], dtype=cp.int32)
        blocks_per_grid = (rr.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _compute_block_inds_kernel(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (rr, cc, self_cols, rowptr, block_inds),
        )

        valid = block_inds != -1

        inds.extend(block_inds[valid])
        value_inds.extend(mask_inds[valid])

    return cp.array(inds, dtype=int), cp.array(value_inds, dtype=int)


@jit.rawkernel()
def _expand_rows_kernel(rows: NDArray, rowptr: NDArray):
    """Expands the rowptr into actual rows.

    Parameters
    ----------
    rows : NDArray
        The rows to fill.
    rowptr : NDArray
        The row pointer.

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < rowptr.shape[0] - 1:
        for j in range(rowptr[i], rowptr[i + 1]):
            rows[j] = i


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
    cols = self_cols[rowptr[0] : rowptr[-1]] - block_offset
    rows = cp.zeros(cols.shape[0], dtype=cp.int32)
    blocks_per_grid = (rowptr.shape[0] + THREADS_PER_BLOCK - 2) // THREADS_PER_BLOCK
    _expand_rows_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (rows, rowptr - rowptr[0]),
    )
    block[..., rows, cols] = data[..., rowptr[0] : rowptr[-1]]


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
    cols = self_cols[rowptr[0] : rowptr[-1]] - block_offset
    rows = cp.zeros(cols.shape[0], dtype=cp.int32)
    blocks_per_grid = (rowptr.shape[0] + THREADS_PER_BLOCK) // THREADS_PER_BLOCK
    _expand_rows_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (rows, rowptr - rowptr[0]),
    )
    data[..., rowptr[0] : rowptr[-1]] = block[..., rows, cols]


def compute_rowptr_map(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> dict:
    """Computes the block-sorting index and the rowptr map.

    Note
    ----
    This is a combination of the bare block-sorting index computation
    and the rowptr map computation.

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
    num_blocks = block_sizes.shape[0]
    block_offsets = cp.hstack((cp.array([0]), cp.cumsum(block_sizes)))

    sort_index = cp.zeros(len(coo_cols), dtype=cp.int32)
    rowptr_map = {}
    mask = cp.zeros(len(coo_cols), dtype=cp.bool_)

    blocks_per_grid = (len(coo_cols) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    offset = 0
    for i, j in cp.ndindex(num_blocks, num_blocks):
        _compute_coo_block_mask_kernel(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                coo_rows,
                coo_cols,
                int(block_offsets[i]),
                int(block_offsets[i + 1]),
                int(block_offsets[j]),
                int(block_offsets[j + 1]),
                mask,
            ),
        )

        bnnz = cp.sum(mask)

        if bnnz != 0:
            # Sort the data by block-row and -column.
            sort_index[offset : offset + bnnz] = cp.nonzero(mask)[0]

            # Compute the rowptr map.
            hist, __ = cp.histogram(
                coo_rows[mask] - block_offsets[i],
                bins=cp.arange(block_sizes[i] + 1),
            )
            rowptr = cp.hstack((cp.array([0]), cp.cumsum(hist))) + offset
            rowptr_map[(i, j)] = rowptr

            offset += bnnz

    return sort_index, rowptr_map
