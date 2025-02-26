# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import xp
from qttools.kernels.operator import operator_inverse


@pytest.mark.usefixtures(
    "batchsize",
    "n",
    "num_quatrature_points",
    "num_blocks",
)
def test_operator_inverse(
    batchsize,
    n,
    num_quatrature_points,
    num_blocks,
):

    rng = xp.random.default_rng(0)

    # mirror shape change inside beyn
    a_xx = tuple(
        rng.random((batchsize, 1, n, n), dtype=xp.float64)
        + 1j * rng.random((batchsize, 1, n, n), dtype=xp.float64)
        for _ in range(2 * num_blocks + 1)
    )

    z = rng.random(
        (1, num_quatrature_points, 1, 1), dtype=xp.float64
    ) + 1j * rng.random((1, num_quatrature_points, 1, 1), dtype=xp.float64)

    operator_inv = operator_inverse(a_xx, z, z.dtype, z.dtype)

    operator_inv_ref = xp.linalg.inv(
        sum(z**n * a_xn for a_xn, n in zip(a_xx, range(-num_blocks, num_blocks + 1)))
    )

    assert xp.allclose(operator_inv, operator_inv_ref)
