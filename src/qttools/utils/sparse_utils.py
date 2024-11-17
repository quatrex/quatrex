# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import functools

from qttools import sparse, xp
from qttools.utils.gpu_utils import ArrayLike


def densify_selected_blocks(
    coo: sparse.coo_matrix,
    block_sizes: ArrayLike,
    blocks: list[tuple[int, int]],
) -> sparse.coo_matrix:
    """Densifies the selected blocks of a sparse coo matrix.

    This adds indices to the selected blocks to make them dense.

    Parameters
    ----------
    coo : sparse.coo_matrix
        The sparse matrix in coordinate format.
    block_sizes : array_like
        The block sizes of the block-sparse matrix we want to construct.
    blocks : list[tuple[int, int]]
        A list of blocks to densify.

    Returns
    -------
    coo : sparse.coo_matrix
        The selectively densified sparse matrix in coordinate format.

    """
    num_blocks = len(block_sizes)
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    added_nnz = int(xp.sum(xp.prod(block_sizes[blocks], axis=1)))
    added_rows = xp.empty(added_nnz, dtype=xp.int32)
    added_cols = xp.empty(added_nnz, dtype=xp.int32)

    offset = 0
    for i, j in blocks:
        # Unsign the block indices.
        i = num_blocks + i if i < 0 else i
        j = num_blocks + j if j < 0 else j

        row_size = int(block_sizes[i])
        col_size = int(block_sizes[j])
        nnz = row_size * col_size
        added_rows[offset : offset + nnz] = (
            xp.repeat(xp.arange(row_size), col_size) + block_offsets[i]
        )
        added_cols[offset : offset + nnz] = (
            xp.tile(xp.arange(col_size), row_size) + block_offsets[j]
        )
        offset += nnz

    coo.row = xp.append(coo.row, added_rows)
    coo.col = xp.append(coo.col, added_cols)
    coo.data = xp.append(coo.data, xp.zeros(added_nnz, dtype=coo.data.dtype))

    coo.sum_duplicates()

    return coo


def compute_block_sort_index(
    coo_rows: ArrayLike, coo_cols: ArrayLike, block_sizes: ArrayLike
) -> ArrayLike:
    """Computes the block-sorting index for a sparse matrix.

    Parameters
    ----------
    coo_rows : array_like
        The row indices of the matrix in coordinate format.
    coo_cols : array_like
        The column indices of the matrix in coordinate format.
    block_sizes : array_like
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : array_like
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = len(block_sizes)
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    sort_index = xp.zeros(len(coo_cols), dtype=int)
    offset = 0
    for i, j in xp.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not xp.any(mask):
            # Skip empty blocks.
            continue

        bnnz = xp.sum(mask)

        # Sort the data by block-row and -column.
        sort_index[offset : offset + bnnz] = xp.argwhere(mask).squeeze()

        offset += bnnz

    return sort_index


def compute_ptr_map(
    coo_rows: ArrayLike, coo_cols: ArrayLike, block_sizes: ArrayLike
) -> dict:
    """Computes the rowptr map for a sparse matrix.

    Parameters
    ----------
    coo_rows : array_like
        The row indices of the matrix in coordinate format.
    coo_cols : array_like
        The column indices of the matrix in coordinate format.
    block_sizes : array_like
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    rowptr_map : dict
        The row pointer map, describing the block-sparse matrix in
        blockwise column-sparse-row format.

    """
    num_blocks = len(block_sizes)
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    # NOTE: This is a naive implementation and can be parallelized.
    rowptr_map = {}
    offset = 0
    for i, j in xp.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not xp.any(mask):
            # Skip empty blocks.
            continue

        # Compute the rowptr map.
        rowptr, __ = xp.histogram(
            coo_rows[mask] - block_offsets[i],
            bins=xp.arange(block_sizes[i] + 1),
        )
        rowptr = xp.hstack(([0], xp.cumsum(rowptr))) + offset
        rowptr_map[(i, j)] = rowptr

        bnnz = xp.sum(mask)
        offset += bnnz

    return rowptr_map


def product_sparsity_pattern(
    *matrices: sparse.spmatrix,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Computes the sparsity pattern of the product of a sequence of matrices.

    Parameters
    ----------
    matrices : sparse.spmatrix
        A sequence of sparse matrices.

    Returns
    -------
    rows : xp.ndarray
        The row indices of the sparsity pattern.
    cols : xp.ndarray
        The column indices of the sparsity pattern.

    """
    # NOTE: cupyx.scipy.sparse does not support bool dtype in matmul.
    csrs = [matrix.tocsr().astype(xp.float32) for matrix in matrices]
    product = functools.reduce(lambda x, y: x @ y, csrs)
    product = product.tocoo()
    # Canonicalize
    product.sum_duplicates()

    return product.row, product.col
