# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse

DSBSPARSE_TYPES = [DSBCSR, DSBCOO]

BLOCK_SIZES = [
    pytest.param(xp.array([2] * 10), id="constant-block-size"),
    pytest.param(xp.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size"),
]

DENSIFY_BLOCKS = [
    pytest.param(None, id="no-densify"),
    pytest.param([(0, 0), (-1, -1)], id="densify-boundary"),
    pytest.param([(2, 4)], id="densify-random"),
]

ACCESSED_BLOCKS = [
    pytest.param((0, 0), id="first-block"),
    pytest.param((-1, -1), id="last-block"),
    pytest.param((2, 4), id="random-block"),
    pytest.param((-9, 3), id="out-of-bounds"),
]

ACCESSED_ELEMENTS = [
    pytest.param((0, 0), id="first-element"),
    pytest.param((-1, -1), id="last-element"),
    pytest.param((2, -7), id="random-element"),
]

GLOBAL_STACK_SHAPES = [
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
]

NUM_INDS = [
    pytest.param(5, id="5-inds"),
    pytest.param(10, id="10-inds"),
    pytest.param(20, id="20-inds"),
]

STACK_INDICES = [
    pytest.param((5,), id="single"),
    pytest.param((slice(1, 4),), id="slice"),
    pytest.param((Ellipsis,), id="ellipsis"),
]

BLOCK_CHANGE_FACTORS = [
    pytest.param(1.0, id="no-change"),
    pytest.param(0.5, id="half-change"),
    pytest.param(2.0, id="double-change"),
]


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def block_sizes(request: pytest.FixtureRequest) -> NDArray:
    return request.param


@pytest.fixture(params=DSBSPARSE_TYPES, autouse=True)
def dsbsparse_type(request: pytest.FixtureRequest) -> DSBSparse:
    return request.param


@pytest.fixture(params=DENSIFY_BLOCKS)
def densify_blocks(request: pytest.FixtureRequest) -> list[tuple]:
    return request.param


@pytest.fixture(params=ACCESSED_BLOCKS)
def accessed_block(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=ACCESSED_ELEMENTS)
def accessed_element(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=NUM_INDS)
def num_inds(request):
    return request.param


@pytest.fixture(params=GLOBAL_STACK_SHAPES, autouse=True)
def global_stack_shape(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=STACK_INDICES)
def stack_index(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=BLOCK_CHANGE_FACTORS)
def block_change_factor(request):
    return request.param
