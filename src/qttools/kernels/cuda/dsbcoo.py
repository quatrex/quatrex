import cupy as cp
from cupy.typing import ArrayLike
from cupyx import jit

from qttools.kernels.cuda import THREADS_PER_BLOCK


@jit.rawkernel()
def _find_inds_kernel(
    self_rows: ArrayLike,
    self_cols: ArrayLike,
    rows: ArrayLike,
    cols: ArrayLike,
    full_inds: ArrayLike,
):
    """Finds the corresponding indices of the given rows and columns.

    Parameters
    ----------
    self_rows : array_like
        The rows of this matrix.
    self_cols : array_like
        The columns of this matrix.
    rows : array_like
        The rows to find the indices for.
    cols : array_like
        The columns to find the indices for.
    full_inds : array_like
        The indices of the given rows and columns.

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < self_rows.shape[0]:
        for j in range(rows.shape[0]):
            cond = int((self_rows[i] == rows[j]) & (self_cols[i] == cols[j]))
            full_inds[i] = full_inds[i] * (1 - cond) + j * cond


def find_inds(
    self_rows: ArrayLike, self_cols: ArrayLike, rows: ArrayLike, cols: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Finds the corresponding indices of the given rows and columns.

    Parameters
    ----------
    self_rows : ArrayLike
        The rows of this matrix.
    self_cols : ArrayLike
        The columns of this matrix.
    rows : ArrayLike
        The rows to find the indices for.
    cols : ArrayLike
        The columns to find the indices for.

    Returns
    -------
    inds : ArrayLike
        The indices of the given rows and columns.
    value_inds : ArrayLike
        The matching indices of this matrix.

    """
    full_inds = cp.zeros(self_rows.shape[0], dtype=cp.int32) - 1
    THREADS_PER_BLOCK
    blocks_per_grid = (self_rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _find_inds_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (self_rows, self_cols, rows, cols, full_inds),
    )

    # Find the valid indices.
    inds = cp.nonzero(full_inds + 1)[0]
    value_inds = full_inds[inds]

    return inds, value_inds


@jit.rawkernel()
def _compute_block_mask_kernel(
    rows: ArrayLike,
    cols: ArrayLike,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    mask: ArrayLike,
):
    """Computes the mask for the block.

    Parameters
    ----------
    rows : array_like
        The row indices of the matrix.
    cols : array_like
        The column indices of the matrix.
    row_start : int
        The start row index of the block.
    row_stop : int
        The stop row index of the block.
    col_start : int
        The start column index of the block.
    col_stop : int
        The stop column index of the block.
    mask : array_like
        The mask to store the result.

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < rows.shape[0]:
        mask[i] = (
            (rows[i] >= row_start)
            & (rows[i] < row_stop)
            & (cols[i] >= col_start)
            & (cols[i] < col_stop)
        )


def compute_block_slice(
    rows: ArrayLike, cols: ArrayLike, block_offsets: ArrayLike, row: int, col: int
) -> slice:
    """Computes the slice of the block in the data.

    Parameters
    ----------
    rows : ArrayLike
        The row indices of the matrix.
    cols : ArrayLike
        The column indices of the matrix.
    block_offsets : ArrayLike
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
    mask = cp.zeros(rows.shape[0], dtype=cp.bool_)
    row_start, row_stop = int(block_offsets[row]), int(block_offsets[row + 1])
    col_start, col_stop = int(block_offsets[col]), int(block_offsets[col + 1])

    blocks_per_grid = (rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _compute_block_mask_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (rows, cols, row_start, row_stop, col_start, col_stop, mask),
    )
    if cp.sum(mask) == 0:
        # No data in this block, return an empty slice.
        return None, None

    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.
    inds = cp.nonzero(mask)[0]
    return inds[0], inds[-1] + 1


def fill_block(block: ArrayLike, rows: ArrayLike, cols: ArrayLike, data: ArrayLike):
    """Fills the dense block with the given data.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    rows : ArrayLike
        The rows at which to fill the block.
    cols : ArrayLike
        The columns at which to fill the block.
    data : ArrayLike
        The data to fill the block with.
    block : ArrayLike
        Preallocated dense block. Should be filled with zeros.

    """
    # TODO: The bare API implementation on the GPU is faster than the
    # very simple, non-general kernel i came up with. Thus, for now i
    # will just use the CuPy API directly. Since for very large blocks
    # (10'000x10'000) this starts to break even, this needs to be
    # revisited!
    block[..., rows, cols] = data[:]
