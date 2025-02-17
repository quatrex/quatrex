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

BLOCKSIZE = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
]

NUM_QUATRATURE_POINTS = [
    pytest.param(5, id="5"),
    pytest.param(10, id="10"),
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


@pytest.fixture(params=BLOCKSIZE)
def blocksize(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=NUM_QUATRATURE_POINTS)
def num_quatrature_points(request: pytest.FixtureRequest):
    return request.param
