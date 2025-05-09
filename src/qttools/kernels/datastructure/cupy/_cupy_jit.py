# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.
import cupy as cp
from cupyx import jit

from qttools import NDArray
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK


@jit.rawkernel()
def _find_inds(
    self_rows: NDArray,
    self_cols: NDArray,
    rows: NDArray,
    cols: NDArray,
    full_inds: NDArray,
    counts: NDArray,
    num_self_rows: int,
    num_rows: int,
):
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
    full_inds : NDArray
        The indices of the given rows and columns.
    counts : NDArray
        The number of matches found.


    """
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < num_self_rows:
        for j in range(num_rows):
            cond = int((self_rows[i] == rows[j]) & (self_cols[i] == cols[j]))
            full_inds[i] = full_inds[i] * (1 - cond) + j * cond
            counts[i] += cond


@jit.rawkernel()
def _find_inds_new(
    self_rows: NDArray,
    self_cols: NDArray,
    rows: NDArray,
    cols: NDArray,
    full_inds: NDArray,
    counts: NDArray,
    num_self_rows: int,
    num_rows: int,
):
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
    full_inds : NDArray
        The indices of the given rows and columns.
    counts : NDArray
        The number of matches found.


    """
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    tid = int(jit.threadIdx.x)
    cache_rows = jit.shared_memory(cp.int32, THREADS_PER_BLOCK)
    cache_cols = jit.shared_memory(cp.int32, THREADS_PER_BLOCK)

    if i < num_self_rows:
        my_row = self_rows[i]
        my_col = self_cols[i]
    else:
        my_row = -1
        my_col = -1

    my_full_ind = 0
    my_count = 0

    for j in range(0, num_rows, THREADS_PER_BLOCK):
        if j + tid < num_rows:
            cache_rows[tid] = rows[j + tid]
            cache_cols[tid] = cols[j + tid]
        jit.syncthreads()

        for idx in range(j, min(j + THREADS_PER_BLOCK, num_rows)):
            cond = int(
                (my_row == cache_rows[idx - j]) & (my_col == cache_cols[idx - j])
            )
            my_full_ind = my_full_ind * (1 - cond) + idx * cond
            my_count += cond
        jit.syncthreads()

    if i < self_rows.shape[0]:
        full_inds[i] = my_full_ind
        counts[i] = my_count


@jit.rawkernel()
def _compute_coo_block_mask(
    rows: NDArray,
    cols: NDArray,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    mask: NDArray,
    rows_len: int,
):
    """Computes the mask for the block in the coordinates.

    Parameters
    ----------
    rows : NDArray
        The row indices of the matrix.
    cols : NDArray
        The column indices of the matrix.
    row_start : int
        The start row index of the block.
    row_stop : int
        The stop row index of the block.
    col_start : int
        The start column index of the block.
    col_stop : int
        The stop column index of the block.
    mask : NDArray
        The mask to store the result.

    """
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < rows_len:
        mask[i] = (
            (rows[i] >= row_start)
            & (rows[i] < row_stop)
            & (cols[i] >= col_start)
            & (cols[i] < col_stop)
        )


@jit.rawkernel()
def _densify_block(
    block: NDArray,
    rows: NDArray,
    cols: NDArray,
    data: NDArray,
    stack_size: int,
    stack_stride: int,
    nnz_per_block: int,
    num_rows: int,
    num_cols: int,
    block_start: int,
    row_offset: int,
    col_offset: int,
):
    """Fills the dense block with the given data.

    Parameters
    ----------
    block : NDArray
        The dense block to fill.
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to fill the block with.

    """

    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    nnz_total = stack_size * nnz_per_block

    if i < nnz_total:
        stack_idx = i // nnz_per_block
        stack_start = stack_idx * stack_stride
        nnz_idx = i % nnz_per_block + block_start
        block_size = num_rows * num_cols

        row = rows[nnz_idx]
        col = cols[nnz_idx]
        block[
            stack_idx * block_size + (row - row_offset) * num_cols + (col - col_offset)
        ] = data[stack_start + nnz_idx]


@jit.rawkernel()
def _find_bcoords(
    block_offsets: NDArray,
    rows: NDArray,
    cols: NDArray,
    brows: NDArray,
    bcols: NDArray,
    rows_len: int,
    block_offsets_len: int,
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
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < rows_len:
        for j in range(block_offsets_len):
            cond_rows = int(block_offsets[j] <= rows[i])
            brows[i] = brows[i] * (1 - cond_rows) + j * cond_rows
            cond_cols = int(block_offsets[j] <= cols[i])
            bcols[i] = bcols[i] * (1 - cond_cols) + j * cond_cols


@jit.rawkernel()
def _compute_block_mask(
    brows: NDArray,
    bcols: NDArray,
    brow: int,
    bcol: int,
    mask: NDArray,
    brows_len: int,
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
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < brows_len:
        mask[i] = (brows[i] == brow) & (bcols[i] == bcol)


@jit.rawkernel()
def _compute_block_inds(
    rr: NDArray,
    cc: NDArray,
    self_cols: NDArray,
    rowptr: NDArray,
    block_inds: NDArray,
    rr_len: int,
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
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < rr_len:
        r = rr[i]
        ind = -1
        for j in range(rowptr[r], rowptr[r + 1]):
            cond = int(self_cols[j] == cc[i])
            ind = ind * (1 - cond) + j * cond

        block_inds[i] = ind


@jit.rawkernel()
def _expand_rows(rows: NDArray, rowptr: NDArray, rowptr_len: int):
    """Expands the rowptr into actual rows.

    Parameters
    ----------
    rows : NDArray
        The rows to fill.
    rowptr : NDArray
        The row pointer.

    """
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < rowptr_len - 1:
        for j in range(rowptr[i], rowptr[i + 1]):
            rows[j] = i


@jit.rawkernel()
def _find_ranks(
    nnz_section_offsets: NDArray,
    inds: NDArray,
    ranks: NDArray,
    nnz_section_offsets_len: int,
    inds_len: int,
):
    """Finds the ranks of the indices in the offsets.

    Parameters
    ----------
    nnz_section_offsets : NDArray
        The offsets of the non-zero sections.
    inds : NDArray
        The indices to find the ranks for.
    ranks : NDArray
        The ranks of the indices in the offsets.

    """
    i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i < inds_len:
        for j in range(nnz_section_offsets_len):
            cond = int(nnz_section_offsets[j] <= inds[i])
            ranks[i] = ranks[i] * (1 - cond) + j * cond
