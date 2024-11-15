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
def fill_block(
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
