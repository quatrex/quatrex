# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp
import numpy as np

from qttools import QTX_USE_CUPY_JIT, NDArray
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK

if QTX_USE_CUPY_JIT:
    from qttools.kernels.datastructure.cupy import _cupy_jit as cupy_backend
else:
    from qttools.kernels.datastructure.cupy import _cupy_rawkernel as cupy_backend


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

    dtype = rows.dtype.type
    if (
        self_cols.dtype.type != dtype
        or cols.dtype.type != dtype
        or block_offsets.dtype.type != dtype
    ):
        raise TypeError(
            f"All input arrays must have the same dtype, but got {self_cols.dtype}, {rows.dtype}, {cols.dtype}, {block_offsets.dtype}."
        )

    brows = cp.zeros_like(rows)
    bcols = cp.zeros_like(cols)

    bcoords_blocks_per_grid = (
        rows.shape[0] + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    cupy_backend._find_bcoords(
        (bcoords_blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            block_offsets,
            rows,
            cols,
            brows,
            bcols,
            dtype(rows.shape[0]),
            dtype(block_offsets.shape[0]),
        ),
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
        mask = cp.zeros_like(brows)

        if (
            rowptr.dtype != dtype
            or brows.dtype != dtype
            or bcols.dtype != dtype
            or mask.dtype != dtype
        ):
            raise TypeError(
                f"All input arrays must have the same dtype, but got {rowptr.dtype}, {brows.dtype}, {bcols.dtype}, {mask.dtype}."
            )

        cupy_backend._compute_block_mask(
            (block_mask_blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                brows,
                bcols,
                brow,
                bcol,
                mask,
                dtype(brows.shape[0]),
            ),
        )
        mask = mask.astype(cp.bool_)
        mask_inds = cp.nonzero(mask)[0]

        # Renormalize the row indices for this block.
        rr = rows[mask] - block_offsets[brow]
        cc = cols[mask]
        block_inds = cp.zeros_like(rr)

        if (
            rr.dtype != dtype
            or cc.dtype != dtype
            or self_cols.dtype != dtype
            or rowptr.dtype != dtype
            or block_inds.dtype != dtype
        ):
            raise TypeError(
                f"All input arrays must have the same dtype, but got {rr.dtype}, {cc.dtype}, {self_cols.dtype}, {rowptr.dtype}, {block_inds.dtype}."
            )

        blocks_per_grid = (rr.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        cupy_backend._compute_block_inds(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                rr,
                cc,
                self_cols,
                rowptr,
                block_inds,
                dtype(rr.shape[0]),
            ),
        )

        valid = block_inds != -1

        inds.extend(block_inds[valid])
        value_inds.extend(mask_inds[valid])

    return cp.array(inds, dtype=dtype), cp.array(value_inds, dtype=dtype)


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
    dtype = self_cols.dtype.type
    if self_cols.dtype.type != dtype or rowptr.dtype.type != dtype:
        raise TypeError(
            f"All input arrays must have the same dtype, but got {self_cols.dtype}, {rowptr.dtype}."
        )

    cols = self_cols[rowptr[0] : rowptr[-1]] - block_offset
    rows = cp.zeros_like(cols)
    blocks_per_grid = (rowptr.shape[0] + THREADS_PER_BLOCK - 2) // THREADS_PER_BLOCK

    cupy_backend._expand_rows(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            rows,
            rowptr - rowptr[0],
            dtype(rowptr.shape[0]),
        ),
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
    dtype = self_cols.dtype.type
    if self_cols.dtype.type != dtype or rowptr.dtype.type != dtype:
        raise TypeError(
            f"All input arrays must have the same dtype, but got {self_cols.dtype}, {rowptr.dtype}."
        )

    cols = self_cols[rowptr[0] : rowptr[-1]] - block_offset
    rows = cp.zeros_like(cols)
    blocks_per_grid = (rowptr.shape[0] + THREADS_PER_BLOCK) // THREADS_PER_BLOCK

    cupy_backend._expand_rows(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            rows,
            rowptr - rowptr[0],
            dtype(rowptr.shape[0]),
        ),
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

    dtype = coo_rows.dtype.type
    if coo_cols.dtype.type != dtype:
        raise TypeError(
            f"All input arrays must have the same dtype, but got {coo_rows.dtype}, {coo_cols.dtype}."
        )

    num_blocks = block_sizes.shape[0]
    block_offsets = np.hstack((np.array([0]), np.cumsum(block_sizes)), dtype=dtype)

    sort_index = cp.zeros_like(coo_cols)
    rowptr_map = {}
    mask = cp.zeros_like(coo_cols)

    blocks_per_grid = (len(coo_cols) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    offset = 0
    for i, j in cp.ndindex(num_blocks, num_blocks):
        cupy_backend._compute_coo_block_mask(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                coo_rows,
                coo_cols,
                dtype(block_offsets[i]),
                dtype(block_offsets[i + 1]),
                dtype(block_offsets[j]),
                dtype(block_offsets[j + 1]),
                mask,
                dtype(len(coo_rows)),
            ),
        )

        # NOTE: Fix for AMD cupy where cub was not used
        if cp.cuda.runtime.is_hip:
            if QTX_USE_CUPY_JIT:
                # TODO: investigate this again
                # this was a previous fix for AMD on Frontier
                # remove the custom reduction if not needed anymore
                # CUPY_ACCELERATORS still seems to be "" on AMD GPUs
                raise RuntimeError(
                    "AMD cupy does not support cub, custom reduction had to be used."
                )

            bnnz = cupy_backend.reduction(mask)
        else:
            bnnz = cp.sum(mask)

        if bnnz != 0:
            # Sort the data by block-row and -column.
            sort_index[offset : offset + bnnz] = cp.nonzero(mask)[0]

            # Compute the rowptr map.
            hist, __ = cp.histogram(
                coo_rows[mask.astype(cp.bool_)] - block_offsets[i],
                bins=cp.arange(block_sizes[i] + 1),
            )
            rowptr = cp.hstack((cp.array([0]), cp.cumsum(hist))) + offset
            rowptr_map[(i, j)] = rowptr.astype(dtype)

            offset += bnnz

    return sort_index, rowptr_map
