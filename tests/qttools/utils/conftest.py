# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray
from qttools.datastructures import DSDBCOO, DSDBCSR, DSDBSparse

DSDBSPARSE_TYPES = [DSDBCSR, DSDBCOO]

BLOCK_SIZES = [
    pytest.param(np.array([2] * 10), id="constant-block-size-2"),
    pytest.param(np.array([5] * 10), id="constant-block-size-5"),
    pytest.param(np.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size-2"),
    pytest.param(np.array([5] * 3 + [10] * 2 + [5] * 3), id="mixed-block-size-5"),
]

NUM_MATRICES = [2, 3, 4, 5]

SHAPES = [
    pytest.param((10,), id="shape-10"),
    pytest.param((10, 10), id="shape-10x10"),
]

DTYPES = [
    pytest.param(complex, id="dtype-complex"),
]

ORDERS = [
    pytest.param("C", id="order-C"),
    pytest.param("F", id="order-F"),
]


OUTPUT_MODULE = [
    pytest.param("numpy", id="numpy"),
    pytest.param("cupy", id="cupy"),
]

INPUT_MODULE = [
    pytest.param("numpy", id="numpy"),
    pytest.param("cupy", id="cupy"),
]

USE_PINNED_MEMORY = [
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
]


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def block_sizes(request: pytest.FixtureRequest) -> NDArray:
    return request.param


@pytest.fixture(params=DSDBSPARSE_TYPES)
def dsdbsparse_type(request: pytest.FixtureRequest) -> DSDBSparse:
    return request.param


@pytest.fixture(params=NUM_MATRICES, autouse=True)
def num_matrices(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=SHAPES, autouse=True)
def shape(request: pytest.FixtureRequest) -> int | tuple[int, ...]:
    return request.param


@pytest.fixture(params=DTYPES, autouse=True)
def dtype(request: pytest.FixtureRequest) -> type | str:
    return request.param


@pytest.fixture(params=ORDERS, autouse=True)
def order(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=OUTPUT_MODULE)
def output_module(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=INPUT_MODULE)
def input_module(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=USE_PINNED_MEMORY)
def use_pinned_memory(request: pytest.FixtureRequest):
    return request.param
