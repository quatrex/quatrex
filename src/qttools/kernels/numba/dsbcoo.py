import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(parallel=True, cache=True, no_rewrites=True)
def find_inds(
    self_rows: NDArray, self_cols: NDArray, rows: NDArray, cols: NDArray
) -> tuple[NDArray, NDArray]:
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

    """
    full_inds = np.zeros(self_rows.shape[0], dtype=np.int32) - 1
    for i in nb.prange(self_rows.shape[0]):
        for j in range(rows.shape[0]):
            cond = int((self_rows[i] == rows[j]) & (self_cols[i] == cols[j]))
            full_inds[i] = full_inds[i] * (1 - cond) + j * cond

    # Find the valid indices.
    inds = np.nonzero(full_inds + 1)[0]
    value_inds = full_inds[inds]

    return inds, value_inds


def compute_block_slice(
    rows: NDArray, cols: NDArray, block_offsets: NDArray, row: int, col: int
) -> slice:
    """Computes the slice of the block in the data.

    Note
    ----
    This is not a jitted function. This vectorized version seems to be
    faster than the jitted version.

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
    slice
        The slice of the block in the data.

    """
    mask = (
        (rows >= block_offsets[row])
        & (rows < block_offsets[row + 1])
        & (cols >= block_offsets[col])
        & (cols < block_offsets[col + 1])
    )
    inds = mask.nonzero()[0]
    if len(inds) == 0:
        # No data in this block, cache an empty slice.
        return slice(None)

    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.
    return slice(inds[0], inds[-1] + 1)


@nb.njit(parallel=True, cache=True)
def _fill_block(
    rows: NDArray,
    cols: NDArray,
    data: NDArray,
    block: NDArray,
):
    """Fills the dense block with the given data.

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

    """
    for i in nb.prange(rows.shape[0]):
        block[..., rows[i], cols[i]] = data[..., i]


def fill_block(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray):
    """Fills the dense block with the given data.

    Note
    ----
    This function follows two different paths depending on the number of
    rows in the block. If the number of rows is greater than 256, the
    block is densified using a parallelized numba function, which, in
    that case seems to be faster than a numpy-vectorized version.

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

    """
    if block.shape[-1] > 256:
        return _fill_block(rows, cols, data, block)

    block[..., rows, cols] = data[:]
