# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools

import pytest

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.sparse_utils import (
    product_sparsity_pattern,
    product_sparsity_pattern_dsbsparse,
)


def _create_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo")
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


if __name__ == "__main__":
    pytest.main([__file__])
