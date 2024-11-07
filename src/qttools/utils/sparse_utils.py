# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from qttools import xp
from qttools.utils.gpu_utils import ArrayLike
from qttools.datastructures import DSBSparse
from scipy import sparse


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


def sparsity_pattern_of_product(
    matrices: tuple[sparse.spmatrix, ...] | tuple[DSBSparse, ...]
) -> tuple[xp.ndarray, xp.ndarray]:
    """Computes the sparsity pattern of the product of a sequence of matrices."""
    product = None
    for matrix in matrices:
        if isinstance(matrix, DSBSparse):
            mat_ones = sparse.coo_matrix(
                (xp.ones(matrix.nnz, dtype=xp.float32), (matrix.rows, matrix.cols)),
            )
        elif sparse.issparse(matrix):
            mat_ones = sparse.coo_matrix(
                (xp.ones(matrix.nnz, dtype=xp.float32), (matrix.row, matrix.col)),
            )
        else: 
            raise ValueError("matrices must be either DSBSparse or sparse.spmatrix")
        if product is None:
            product = mat_ones
        else:
            product = product @ mat_ones
    product = product.tocoo()
    # Canonicalize
    product.sum_duplicates()
    return (product.row, product.col)
