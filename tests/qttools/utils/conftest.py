# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import numpy as np
import pytest

from qttools import NDArray, xp
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

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"
MOS2_EXAMPLE = EXAMPLES_DIR / "w90" / "mos2" / "inputs"


@pytest.fixture(autouse=True, scope="session")
def unit_cells(request: pytest.FixtureRequest) -> NDArray:
    """Returns the wannier tight binding matrix of the mos2 example"""

    input_path = MOS2_EXAMPLE / "hamiltonian_unit_cells.npy"
    unit_cells = xp.load(input_path)

    return unit_cells


@pytest.fixture(params=BLOCK_SIZES)
def block_sizes(request: pytest.FixtureRequest) -> NDArray:
    return request.param


@pytest.fixture(params=DSDBSPARSE_TYPES)
def dsdbsparse_type(request: pytest.FixtureRequest) -> DSDBSparse:
    return request.param


@pytest.fixture(params=NUM_MATRICES)
def num_matrices(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=SHAPES)
def shape(request: pytest.FixtureRequest) -> int | tuple[int, ...]:
    return request.param


@pytest.fixture(params=DTYPES)
def dtype(request: pytest.FixtureRequest) -> type | str:
    return request.param


@pytest.fixture(params=ORDERS)
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
