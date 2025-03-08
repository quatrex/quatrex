# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse, DBCOO, DBSparse

DSBSPARSE_TYPES = [DSBCSR, DSBCOO]
DBSPARSE_TYPES = [DBCOO]

BLOCK_SIZES = [
    pytest.param(xp.array([2] * 10), id="constant-block-size-2"),
    pytest.param(xp.array([5] * 10), id="constant-block-size-5"),
    pytest.param(xp.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size-2"),
    pytest.param(xp.array([5] * 3 + [10] * 2 + [5] * 3), id="mixed-block-size-5"),
]

NUM_MATRICES = [2, 3, 4, 5]


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def block_sizes(request: pytest.FixtureRequest) -> NDArray:
    return request.param


@pytest.fixture(params=DSBSPARSE_TYPES)
def dsbsparse_type(request: pytest.FixtureRequest) -> DSBSparse:
    return request.param


@pytest.fixture(params=DBSPARSE_TYPES)
def dbsparse_type(request: pytest.FixtureRequest) -> DBSparse:
    return request.param


@pytest.fixture(params=NUM_MATRICES, autouse=True)
def num_matrices(request: pytest.FixtureRequest) -> int:
    return request.param
