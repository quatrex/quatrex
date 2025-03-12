# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools

import pytest
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.datastructures.dbsparse import DBSparse
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import (
    product_sparsity_pattern,
    product_sparsity_pattern_dbsparse,
    product_sparsity_pattern_dsbsparse,
)


def _create_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)

    def _rvs(size=None, rng=rng):
        return xp.ones(size)

    coo = sparse.random(size, size, density=density, data_rvs=_rvs, format="coo")
    return coo


def _create_btd_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(sizes)))

    arr = xp.zeros((size, size), dtype=xp.float32)
    for i in range(len(sizes)):
        # Diagonal block.
        block_shape = (int(sizes[i]), int(sizes[i]))
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = xp.random.rand(
            *block_shape
        )  # + 1j * xp.random.rand(*block_shape)
        # Superdiagonal block.
        if i < len(sizes) - 1:
            block_shape = (int(sizes[i]), int(sizes[i + 1]))
            arr[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = (
                xp.random.rand(*block_shape)  # + 1j * xp.random.rand(*block_shape)
            )
            arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = (
                xp.random.rand(*block_shape).T  # + 1j * xp.random.rand(*block_shape).T
            )
    rng = xp.random.default_rng()
    cutoff = rng.uniform(low=0.1, high=0.4)
    arr[xp.abs(arr) < cutoff] = 0
    coo = sparse.coo_matrix(arr)
    coo.data[:] = 1
    return coo


def test_product_sparsity(
    num_matrices: int,
    block_sizes: NDArray,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    matrices = [_create_coo(block_sizes) for _ in range(num_matrices)]

    product = functools.reduce(lambda x, y: x @ y, matrices)
    product.data[:] = 1
    ref = product.toarray()

    rows, cols = product_sparsity_pattern(*matrices)
    val = sparse.coo_matrix(
        (xp.ones(len(rows)), (rows, cols)), shape=product.shape
    ).toarray()

    assert xp.allclose(ref, val)


def test_product_sparsity_dsbsparse(
    dsbsparse_type: DSBSparse,
    num_matrices: int,
    block_sizes: NDArray,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    matrices = [_create_coo(block_sizes) for _ in range(num_matrices)]
    dsbsparse_matrices = [
        dsbsparse_type.from_sparray(matrix, block_sizes, (1,)) for matrix in matrices
    ]

    product = functools.reduce(lambda x, y: x @ y, matrices)
    product.data[:] = 1
    ref = product.toarray()

    rows, cols = product_sparsity_pattern_dsbsparse(*dsbsparse_matrices)
    val = sparse.coo_matrix(
        (xp.ones(len(rows)), (rows, cols)), shape=product.shape
    ).toarray()

    assert xp.allclose(ref, val)


def _expand_matrix(
    matrix: sparse.spmatrix, block_sizes: NDArray, NBC: int = 1
) -> sparse.spmatrix:

    shape = list(matrix.shape)
    left_obc = int(sum(block_sizes[0:NBC]))
    right_obc = int(sum(block_sizes[-NBC:]))
    shape[-2] += left_obc + right_obc
    shape[-1] += left_obc + right_obc

    csr = matrix.tocsr()

    expanded = sparse.csr_matrix(tuple(shape), dtype=matrix.dtype)

    # expanded[left_obc : -right_obc, left_obc : -right_obc] = csr
    # expanded[:left_obc, left_obc:2*left_obc] = csr[:left_obc, left_obc:2*left_obc]
    # expanded[left_obc:2*left_obc, :left_obc] = csr[left_obc:2*left_obc, :left_obc]
    # expanded[-right_obc:, -2*right_obc:-right_obc] = csr[-2*right_obc:-right_obc, -right_obc:]
    # expanded[-2*right_obc:-right_obc, -right_obc:] = csr[-right_obc:, -2*right_obc:-right_obc]

    # simply repeat the boundaries slices
    expanded[
        left_obc : left_obc + int(sum(block_sizes)),
        left_obc : left_obc + int(sum(block_sizes)),
    ] = csr
    expanded[:left_obc, :-left_obc] = expanded[left_obc : 2 * left_obc, left_obc:]
    expanded[:-left_obc, :left_obc] = expanded[left_obc:, left_obc : 2 * left_obc]
    expanded[-right_obc:, right_obc:] = expanded[
        -2 * right_obc : -right_obc, :-right_obc
    ]
    expanded[right_obc:, -right_obc:] = expanded[
        :-right_obc, -2 * right_obc : -right_obc
    ]

    return expanded


def _spillover_matmul(
    a: sparse.spmatrix, b: sparse.spmatrix, block_sizes
) -> sparse.spmatrix:
    """Multiplies two sparse matrices with spillover correction."""
    c = (a @ b).tocsr()

    a = a.tocsr()
    b = b.tocsr()

    # Left spillover
    i_ = slice(None, int(block_sizes[0]))
    j_ = slice(int(block_sizes[0]), int(sum(block_sizes[:2])))
    c[i_, i_] += a[j_, i_] @ b[i_, j_]

    # Right spillover
    i_ = slice(int(-block_sizes[-1]), None)
    j_ = slice(int(-sum(block_sizes[-2:])), int(-block_sizes[-1]))
    c[i_, i_] += a[j_, i_] @ b[i_, j_]

    return c


def test_product_sparsity_dsbsparse_spillover(
    dsbsparse_type: DSBSparse,
    num_matrices: int,
    block_sizes: NDArray,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    matrices = [_create_btd_coo(block_sizes) for _ in range(num_matrices)]
    dsbsparse_matrices = [
        dsbsparse_type.from_sparray(matrix, block_sizes, (1,)) for matrix in matrices
    ]

    shape = matrices[0].shape
    expanded_matrices = [_expand_matrix(matrix, block_sizes, 1) for matrix in matrices]
    product = functools.reduce(lambda x, y: x @ y, expanded_matrices)
    product.data[:] = 1
    ref = product.toarray()[
        block_sizes[0] : block_sizes[0] + int(sum(block_sizes)),
        block_sizes[0] : block_sizes[0] + int(sum(block_sizes)),
    ]
    # product = functools.reduce(lambda x, y: _spillover_matmul(x, y, block_sizes), matrices)
    # product.data[:] = 1
    # ref = product.toarray()

    rows, cols = product_sparsity_pattern_dsbsparse(*dsbsparse_matrices, spillover=True)
    val = sparse.coo_matrix((xp.ones(len(rows)), (rows, cols)), shape=shape).toarray()

    print(xp.nonzero(ref - val))

    assert xp.allclose(ref, val)


def test_product_sparsity_dbsparse(
    dbsparse_type: DBSparse,
    num_matrices: int,
    block_sizes: NDArray,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    last_block_sizes = block_sizes[-3:]
    if num_matrices > 3:
        block_sizes = xp.hstack(
            (block_sizes, *[last_block_sizes for _ in range(num_matrices - 3)])
        )
    matrices = [_create_btd_coo(block_sizes) for _ in range(num_matrices)]
    matrices = [comm.bcast(matrix, root=0) for matrix in matrices]
    dsbsparse_matrices = [
        dbsparse_type.from_sparray(matrix, block_sizes) for matrix in matrices
    ]
    dense_matrices = [matrix.to_dense() for matrix in dsbsparse_matrices]
    for i in range(num_matrices):
        assert xp.allclose(dense_matrices[i], matrices[i].toarray())

    product = functools.reduce(lambda x, y: x @ y, matrices)
    product.data[:] = 1
    ref = product.toarray()

    local_blocks, _ = get_section_sizes(len(block_sizes), comm.size)
    start_block = sum(local_blocks[: comm.rank])
    end_block = start_block + local_blocks[comm.rank]

    rows, cols = product_sparsity_pattern_dbsparse(
        *dsbsparse_matrices,
        in_num_diag=3,
        start_block=start_block,
        end_block=end_block,
        comm=comm,
    )
    val = sparse.coo_matrix(
        (xp.ones(len(rows)), (rows, cols)), shape=product.shape
    ).toarray()

    if comm.rank == 0:
        print(xp.nonzero(ref - val))

    assert xp.allclose(ref, val)


def test_product_sparsity_dbsparse_spillover(
    dbsparse_type: DBSparse,
    num_matrices: int,
    block_sizes: NDArray,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    last_block_sizes = block_sizes[-3:]
    if num_matrices > 3:
        block_sizes = xp.hstack(
            (block_sizes, *[last_block_sizes for _ in range(num_matrices - 3)])
        )
    matrices = [_create_btd_coo(block_sizes) for _ in range(num_matrices)]
    matrices = [comm.bcast(matrix, root=0) for matrix in matrices]
    dsbsparse_matrices = [
        dbsparse_type.from_sparray(matrix, block_sizes) for matrix in matrices
    ]
    dense_matrices = [matrix.to_dense() for matrix in dsbsparse_matrices]
    for i in range(num_matrices):
        assert xp.allclose(dense_matrices[i], matrices[i].toarray())

    shape = matrices[0].shape
    expanded_matrices = [_expand_matrix(matrix, block_sizes, 1) for matrix in matrices]
    product = functools.reduce(lambda x, y: x @ y, expanded_matrices)
    product.data[:] = 1
    ref = product.toarray()[
        block_sizes[0] : block_sizes[0] + int(sum(block_sizes)),
        block_sizes[0] : block_sizes[0] + int(sum(block_sizes)),
    ]
    # product = functools.reduce(lambda x, y: _spillover_matmul(x, y, block_sizes), matrices)
    # product.data[:] = 1
    # ref = product.toarray()

    local_blocks, _ = get_section_sizes(len(block_sizes), comm.size)
    start_block = sum(local_blocks[: comm.rank])
    end_block = start_block + local_blocks[comm.rank]

    rows, cols = product_sparsity_pattern_dbsparse(
        *dsbsparse_matrices,
        in_num_diag=3,
        start_block=start_block,
        end_block=end_block,
        comm=comm,
        spillover=True,
    )
    val = sparse.coo_matrix((xp.ones(len(rows)), (rows, cols)), shape=shape).toarray()

    print(f"{comm.rank=}, {start_block=}, {end_block=}", flush=True)
    if comm.rank == 0:
        print(xp.nonzero(ref - val))

    assert xp.allclose(ref, val)


if __name__ == "__main__":
    pytest.main([__file__])
