# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools

import pytest

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.utils.sparse_utils import (
    product_sparsity_pattern,
    product_sparsity_pattern_dsdbsparse,
)

GLOBAL_STACK_SHAPES = [
    pytest.param((4,), id="1D-stack"),
    pytest.param((5, 2), id="2D-stack"),
]


def setup_module():
    """setup any state specific to the execution of the given module."""
    if xp.__name__ == "cupy":
        _default_config = {
            "all_to_all": "host_mpi",
            "all_gather": "host_mpi",
            "all_reduce": "host_mpi",
            "bcast": "host_mpi",
        }
    elif xp.__name__ == "numpy":
        _default_config = {
            "all_to_all": "device_mpi",
            "all_gather": "device_mpi",
            "all_reduce": "device_mpi",
            "bcast": "device_mpi",
        }
    # Configure the comm singleton.
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
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
    offsets = xp.hstack(([0], xp.cumsum(xp.asarray(sizes))))

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


@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_product_sparsity_dsdbsparse(
    dsdbsparse_type: DSDBSparse,
    num_matrices: int,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    matrices = [_create_btd_coo(block_sizes) for _ in range(num_matrices)]
    dsdbsparse_matrices = [
        dsdbsparse_type.from_sparray(matrix, block_sizes, global_stack_shape)
        for matrix in matrices
    ]
    dense_matrices = [matrix.to_dense() for matrix in dsdbsparse_matrices]
    for i in range(num_matrices):
        assert xp.allclose(dense_matrices[i], matrices[i].toarray())

    product = functools.reduce(lambda x, y: x @ y, matrices)
    product.data[:] = 1
    ref = product.toarray()

    rows, cols = product_sparsity_pattern_dsdbsparse(
        *dsdbsparse_matrices,
        in_num_diag=3,
    )
    val = sparse.coo_matrix(
        (xp.ones(len(rows)), (rows, cols)), shape=product.shape
    ).toarray()

    if comm.rank == 0:
        print(xp.nonzero(ref - val))

    assert xp.allclose(ref, val)


@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_product_sparsity_dsdbsparse_spillover(
    dsdbsparse_type: DSDBSparse,
    num_matrices: int,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the computation of the matrix product's sparsity pattern."""
    matrices = [_create_btd_coo(block_sizes) for _ in range(num_matrices)]
    dsdbsparse_matrices = [
        dsdbsparse_type.from_sparray(matrix, block_sizes, global_stack_shape)
        for matrix in matrices
    ]
    dense_matrices = [matrix.to_dense() for matrix in dsdbsparse_matrices]
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

    rows, cols = product_sparsity_pattern_dsdbsparse(
        *dsdbsparse_matrices,
        in_num_diag=3,
        spillover=True,
    )
    val = sparse.coo_matrix((xp.ones(len(rows)), (rows, cols)), shape=shape).toarray()

    if comm.rank == 0:
        print(xp.nonzero(ref - val))

    assert xp.allclose(ref, val)


if __name__ == "__main__":
    pytest.main([__file__])
