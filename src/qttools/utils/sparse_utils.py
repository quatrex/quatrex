# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools

from qttools import NDArray, sparse, xp


def densify_selected_blocks(
    coo: sparse.coo_matrix,
    block_sizes: NDArray,
    blocks: list[tuple[int, int]],
) -> sparse.coo_matrix:
    """Densifies the selected blocks of a sparse coo matrix.

    This adds indices to the selected blocks to make them dense.

    Parameters
    ----------
    coo : sparse.coo_matrix
        The sparse matrix in coordinate format.
    block_sizes : NDArray
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


def product_sparsity_pattern(
    *matrices: sparse.spmatrix,
) -> tuple[NDArray, NDArray]:
    """Computes the sparsity pattern of the product of a sequence of matrices.

    Parameters
    ----------
    matrices : sparse.spmatrix
        A sequence of sparse matrices.

    Returns
    -------
    rows : NDArray
        The row indices of the sparsity pattern.
    cols : NDArray
        The column indices of the sparsity pattern.

    """
    # NOTE: cupyx.scipy.sparse does not support bool dtype in matmul.
    csrs = [matrix.tocsr().astype(xp.float32) for matrix in matrices]
    product = functools.reduce(lambda x, y: x @ y, csrs)
    product = product.tocoo()
    # Canonicalize
    product.sum_duplicates()

    return product.row, product.col
