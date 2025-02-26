# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import pytest

nb.set_num_threads(1)

NUM_INDS = [
    pytest.param(10, id="10"),
    pytest.param(42, id="42"),
]

NNZ = [
    pytest.param(200, id="200"),
    pytest.param(700, id="700"),
]

NUM_BLOCKS = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

COMM_SIZE = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

SHAPE = [
    pytest.param((50, 50), id="50x50"),
    pytest.param((97, 97), id="97x97"),
]

BLOCK_COORDS = [
    pytest.param((0, 2), id="10x10"),
    pytest.param((3, 4), id="20x20"),
]

BATCHSIZE = [
    pytest.param(1, id="1"),
    pytest.param(3, id="3"),
]

N = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

M = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

NUM_QUATRATURE_POINTS = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

BATCH_SHAPE = [
    pytest.param((1,), id="1"),
    pytest.param((3,), id="1"),
    pytest.param((5, 1), id="5x1"),
    pytest.param((5, 2), id="5x2"),
]

COMPUTE_MODULE = [
    pytest.param("numpy", id="numpy"),
    pytest.param("cupy", id="cupy"),
]

OUTPUT_MODULE = [
    pytest.param("numpy", id="numpy"),
    pytest.param("cupy", id="cupy"),
]

INPUT_MODULE = [
    pytest.param("numpy", id="numpy"),
    pytest.param("cupy", id="cupy"),
]

FULL_MATRICES = [
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
]


@pytest.fixture(params=NUM_INDS)
def num_inds(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=NNZ)
def nnz(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=COMM_SIZE)
def comm_size(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=SHAPE)
def shape(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=NUM_BLOCKS)
def num_blocks(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=BLOCK_COORDS)
def block_coords(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=BATCHSIZE)
def batchsize(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=NUM_QUATRATURE_POINTS)
def num_quatrature_points(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=N)
def n(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=M)
def m(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=BATCH_SHAPE)
def batch_shape(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=FULL_MATRICES)
def full_matrices(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=COMPUTE_MODULE)
def compute_module(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=OUTPUT_MODULE)
def output_module(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=INPUT_MODULE)
def input_module(request: pytest.FixtureRequest):
    return request.param
