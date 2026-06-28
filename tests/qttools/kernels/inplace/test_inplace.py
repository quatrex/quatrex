# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import sparse, xp
from qttools.kernels import inplace


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("shape", [(50, 50)])
@pytest.mark.parametrize("int64", [False, True])
@pytest.mark.parametrize("double", [False, True])
def test_scatter_add_scaled(
    shape: tuple[int, int],
    conjugate: bool,
    int64: bool,
    double: bool,
):

    assert shape[0] == shape[1], "shape must be square for this test."

    rng = xp.random.default_rng(42)

    a = sparse.random(*shape, density=0.2, format="csr") + 1j * sparse.random(
        *shape, density=0.2, format="csr"
    )
    b = sparse.random(*shape, density=0.2, format="csr") + 1j * sparse.random(
        *shape, density=0.2, format="csr"
    )

    # make a superset of the indices of b in a
    a = (a + b).tocoo()
    b = b.tocoo()

    flat_a = a.row * shape[1] + a.col
    flat_b = b.row * shape[1] + b.col

    sort_a = xp.argsort(flat_a)
    sort_b = xp.argsort(flat_b)

    sorted_flat_a = flat_a[sort_a]
    sorted_flat_b = flat_b[sort_b]

    sorted_match_positions = xp.searchsorted(sorted_flat_a, sorted_flat_b)

    inds = xp.ascontiguousarray(sort_a[sorted_match_positions])

    b_data_sorted = xp.ascontiguousarray(b.data[sort_b])

    if double:
        alpha = xp.float64(rng.uniform(0.1, 1.0))
    else:
        alpha = xp.complex128(rng.uniform(0.1, 1.0) + 1j * rng.uniform(0.1, 1.0))

    a_copy = a.copy()

    if int64:
        inds = inds.astype(xp.int64)
    else:
        inds = inds.astype(xp.int32)

    inplace.scatter_add_scaled(
        a.data, b_data_sorted, inds, alpha=alpha, conjugate=conjugate
    )

    ref = a_copy + alpha * (b.conj() if conjugate else b)

    assert xp.allclose(a.toarray(), ref.toarray())
