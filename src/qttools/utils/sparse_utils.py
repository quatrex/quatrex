# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.datastructures.routines import BlockMatrix, bd_matmul_distr
from qttools.profiling import Profiler

profiler = Profiler()


@profiler.profile(level="debug")
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
    csrs = [matrix.tocsr() for matrix in matrices]
    for i, csr in enumerate(csrs):
        if xp.iscomplexobj(csr.data):
            csr = csr.copy()
            csr.data = csr.data.real
        csrs[i] = csr.astype(xp.float32)
    product = functools.reduce(lambda x, y: x @ y, csrs)
    product = product.tocoo()
    # Canonicalize
    product.sum_duplicates()

    return product.row, product.col


def tocsr_dict(matrix: DSDBSparse) -> dict[tuple[int, int], sparse.csr_matrix]:
    """Converts a DSDBSparse matrix to a dictionary of CSR blocks."""

    blocks = {}

    for i in range(matrix.num_blocks):
        for j in range(matrix.num_blocks):
            sparse_data = matrix.sparse_blocks[i, j]
            data = xp.ones_like(sparse_data[0][-1], dtype=xp.float32)
            sparse_data = (data, *sparse_data[1:])
            blocks[i, j] = sparse.csr_matrix(
                sparse_data, shape=(matrix.block_sizes[i], matrix.block_sizes[j])
            )

    return blocks


def product_sparsity_pattern_dsdbsparse(
    *matrices: DSDBSparse,
    in_num_diag: int = 3,
    out_num_diag: int = None,
    start_block: int = 0,
    end_block: int = None,
    spillover: bool = False,
) -> tuple[NDArray, NDArray]:
    """Computes the sparsity pattern of the product of a sequence of DSDBSparse matrices.

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

    assert len(matrices) > 1

    a = matrices[0]

    # Assuming that all matrices have the same number of blocks, same block sizes, and same block diagonals.
    num_blocks = a.num_blocks
    end_block = end_block or num_blocks
    block_offsets = a.block_offsets
    a_num_diag = in_num_diag

    local_keys = set()
    for i in range(start_block, end_block):
        for j in range(
            max(start_block, i - in_num_diag // 2),
            min(num_blocks, i + in_num_diag // 2 + 1),
        ):
            local_keys.add((i, j))
    for j in range(start_block, end_block):
        for i in range(
            max(end_block, j - in_num_diag // 2),
            min(num_blocks, j + in_num_diag // 2 + 1),
        ):
            local_keys.add((i, j))
    a_ = BlockMatrix(a, local_keys, (start_block, start_block))

    out_num_diag = out_num_diag or in_num_diag * len(matrices) - len(matrices) + 1

    for n, b in enumerate(matrices[1:]):

        b_ = BlockMatrix(b, local_keys, (start_block, start_block))
        b_num_diag = in_num_diag
        tmp_num_diag = a_num_diag + b_num_diag - 1
        if n == len(matrices) - 2:
            tmp_num_diag = out_num_diag

        c_ = bd_matmul_distr(
            a=a_,
            b=b,
            out=None,
            a_num_diag=a_num_diag,
            b_num_diag=b_num_diag,
            out_num_diag=tmp_num_diag,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=False,
        )

        if spillover:
            if start_block == 0:
                # Left spillover
                c_[0, 0] = c_[0, 0] + a_[1, 0] @ b_[0, 1]
            if end_block == num_blocks:
                # Right spillover
                c_[num_blocks - 1, num_blocks - 1] = (
                    c_[num_blocks - 1, num_blocks - 1]
                    + a_[num_blocks - 2, num_blocks - 1]
                    @ b_[num_blocks - 1, num_blocks - 2]
                )

        a_ = c_
        a_num_diag = tmp_num_diag

    c_rows = xp.empty(0, dtype=xp.int32)
    c_cols = xp.empty(0, dtype=xp.int32)

    local_keys = set()
    for i in range(start_block, end_block):
        for j in range(
            max(start_block, i - out_num_diag // 2),
            min(num_blocks, i + out_num_diag // 2 + 1),
        ):
            local_keys.add((i, j))
    for j in range(start_block, end_block):
        for i in range(
            max(end_block, j - out_num_diag // 2),
            min(num_blocks, j + out_num_diag // 2 + 1),
        ):
            local_keys.add((i, j))

    for i in range(num_blocks):
        for j in range(num_blocks):
            if (i, j) not in local_keys:
                continue
            c_block = c_[i, j]
            m, n = c_block.shape[-2:]
            c_block = sparse.coo_matrix(c_block.flat[: m * n].reshape(m, n))
            c_block.sum_duplicates()
            c_block.eliminate_zeros()
            c_rows = xp.append(c_rows, c_block.row + block_offsets[i])
            c_cols = xp.append(c_cols, c_block.col + block_offsets[j])

    return c_rows, c_cols
