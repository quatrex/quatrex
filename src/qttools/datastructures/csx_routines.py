# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes routines for handling CSX matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qttools import NDArray, xp
from qttools.comm.comm import _SubCommunicator, pad_buffer

if TYPE_CHECKING:
    # Adjust this import path to where CSX is actually defined
    from qttools.datastructures.csx import CSX


def remove_duplicate_entries(
    row_ind: NDArray,
    col_ind: NDArray,
    cols: int,
):
    """Removes duplicate entries from the given COO format indices.

    Note
    ----
    Also, makes the indices canonical by sorting them in lexicographical
    order.

    Parameters
    ----------
    row_ind : NDArray
        The row indices of the COO format.
    col_ind : NDArray
        The column indices of the COO format.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    tuple[NDArray, NDArray]
        The row and column indices of the COO format with duplicates
        removed.

    """
    sort_idx = xp.lexsort(xp.stack((col_ind, row_ind)))

    sorted_rows = row_ind[sort_idx]
    sorted_cols = col_ind[sort_idx]

    is_diff = (sorted_rows[1:] != sorted_rows[:-1]) | (
        sorted_cols[1:] != sorted_cols[:-1]
    )
    keep = xp.concatenate((xp.array([True], dtype=bool), is_diff))

    return sorted_rows[keep], sorted_cols[keep]


def make_canonical_coo(
    row_ind: NDArray,
    col_ind: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Returns the canonical COO format of the given indices.

    Parameters
    ----------
    row_ind : NDArray
        The row indices of the COO format.
    col_ind : NDArray
        The column indices of the COO format.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The row indices, column indices, and the sorting indices of the
        canonical COO format of the given indices.

    """
    # Sort the indices and data to get the canonical format
    sort_idx = xp.lexsort(xp.stack((col_ind, row_ind)))

    row_ind = row_ind[sort_idx]
    col_ind = col_ind[sort_idx]
    return row_ind, col_ind, sort_idx


def allgather_csx(
    csx: CSX,
    comm: _SubCommunicator,
    axis: int = 0,
) -> CSX:
    """Allgathers a CSX matrix along the specified axis.

    Parameters
    ----------
    csx : CSX
        The CSX matrix to be allgathered.
    comm : _SubCommunicator
        The communicator over which to perform the allgather operation.
    axis : int, optional
        The axis along which to allgather the CSX matrix. Must be 0 or
        1. Default is 0.

    Returns
    -------
    CSX
        The allgathered CSX matrix.

    """
    from qttools.datastructures.csx import CSX

    # make axis positive
    if axis < 0:
        axis += 2

    # Check that the axis is valid meaning that it is one of the real
    # space dimensions
    if axis not in [0, 1]:
        raise ValueError(
            f"Invalid axis {axis}. Must be 0 or 1 for real space dimensions."
        )

    # TODO: Assumes int64 for now, but should be more general.
    if axis == 0:
        count = csx.rows
    else:
        count = csx.cols
    counts = np.zeros(comm.size, dtype=np.int64)
    comm.all_gather(
        np.array(count, dtype=np.int64),
        counts,
        backend="device_mpi",
    )
    offsets = np.array([0] + list(np.cumsum(counts)), dtype=np.int64)

    if axis == 0:
        rows = offsets[-1]
        cols = csx.cols
    else:
        rows = csx.rows
        cols = offsets[-1]

    # TODO: Assumes int64 for now, but should be more general.
    counts = np.zeros(comm.size, dtype=np.int64)
    comm.all_gather(
        np.array(len(csx.row_ind), dtype=np.int64),
        counts,
        backend="device_mpi",
    )
    global_size = np.max(counts) * comm.size
    mask = xp.zeros(global_size, dtype=bool)
    for i in range(comm.size):
        mask[np.max(counts) * i : np.max(counts) * i + counts[i]] = True

    # Communicate the row and column indices
    row_ind = csx.row_ind
    col_ind = csx.col_ind

    if axis == 0:
        row_ind = row_ind + offsets[comm.rank]
    else:
        col_ind = col_ind + offsets[comm.rank]

    # Communicate the row indices
    sendbuf = pad_buffer(row_ind, global_size, comm.size, -1)
    recvbuf = xp.empty((global_size), dtype=sendbuf.dtype)
    comm.all_gather(sendbuf, recvbuf)
    row_ind = recvbuf[mask]

    # Communicate the column indices
    sendbuf = pad_buffer(col_ind, global_size, comm.size, -1)
    recvbuf = xp.empty((global_size), dtype=sendbuf.dtype)
    comm.all_gather(sendbuf, recvbuf)
    col_ind = recvbuf[mask]

    # Communicate the data
    sendbuf = pad_buffer(csx.data, global_size, comm.size, -1)

    sendbuf = xp.ascontiguousarray(xp.moveaxis(sendbuf, -1, 0))
    recvbuf = xp.empty((global_size, *sendbuf.shape[1:]), dtype=sendbuf.dtype)
    comm.all_gather(sendbuf, recvbuf)
    recvbuf = xp.moveaxis(recvbuf, 0, -1)

    indices = xp.where(mask)[0]
    data = xp.take(recvbuf, indices, axis=-1)

    # Need to make canonical
    # NOTE: When allgathering along rows, the result should be already
    # canonical.
    if axis == 1:
        row_ind, col_ind, sort_idx = make_canonical_coo(
            row_ind=row_ind,
            col_ind=col_ind,
        )
        data = data[..., sort_idx]

    csx = CSX(
        dtype=data.dtype,
        rows=int(rows),
        cols=int(cols),
        local_stack_shape=csx.local_stack_shape,
        row_ind=row_ind,
        col_ind=col_ind,
        symmetry=csx.symmetry,
    )
    csx.allocate_data()
    csx.data = data

    return csx
