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

    inds = np.zeros(rr.shape[0], dtype=np.int32)
    for i in nb.prange(rr.shape[0]):
        r = rr[i]
        ind = np.nonzero(self_cols[rowptr[r] : rowptr[r + 1]] == cc[i])[0]
        if len(ind) == 0:
            continue
        inds[i] = rowptr[r] + ind[0]

    valid = inds != 0
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

        inds.append(block_inds)
        value_inds.append(block_value_inds)

        mask = (brows == brow) & (bcols == bcol)
        mask_inds = np.nonzero(mask)[0]

        # Renormalize the row indices for this block.
        rr = rows[mask] - block_offsets[brow]
        cc = cols[mask]

        # TODO: This could perhaps be done in an efficient way.
        for i, (r, c) in enumerate(zip(rr, cc)):
            ind = np.nonzero(self_cols[rowptr[r] : rowptr[r + 1]] == c)[0]

            if len(ind) == 0:
                continue

            value_inds.append(mask_inds[i])
            inds.append(rowptr[r] + ind[0])

    return np.array(inds, dtype=int), np.array(value_inds, dtype=int)


@nb.njit(parallel=True, cache=True)
def fill_block(
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
    for i in nb.prange(int(block.shape[-1])):
        cols = self_cols[rowptr[i] : rowptr[i + 1]]
        block[..., i, cols - block_offset] = data[..., rowptr[i] : rowptr[i + 1]]
