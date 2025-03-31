# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, host_xp, xp
from qttools.datastructures import DSBCOO, DSBCSR, DSDBCOO, DSBSparse, DSDBSparse

DSBSPARSE_TYPES = [DSBCSR, DSBCOO]
DSDBSPARSE_TYPES = [DSDBCOO]

BLOCK_SIZES = [
    pytest.param(host_xp.array([2] * 10), id="constant-block-size-2"),
    pytest.param(host_xp.array([5] * 10), id="constant-block-size-5"),
    pytest.param(host_xp.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size-2"),
    pytest.param(host_xp.array([5] * 3 + [10] * 2 + [5] * 3), id="mixed-block-size-5"),
]

DENSIFY_BLOCKS = [
    pytest.param(None, id="no-densify"),
    pytest.param([(0, 0), (-1, -1)], id="densify-boundary"),
    pytest.param([(2, 4)], id="densify-random"),
]

ACCESSED_BLOCKS = [
    pytest.param((0, 0), id="first-block"),
    pytest.param((-1, -1), id="last-block"),
    pytest.param((4, 2), id="random-lower-block"),
    pytest.param((2, 4), id="random-upper-block"),
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
    pytest.param((1,), id="single"),
    pytest.param((slice(0, 2),), id="slice"),
    pytest.param((Ellipsis,), id="ellipsis"),
]

BLOCK_CHANGE_FACTORS = [
    pytest.param(1.0, id="no-change"),
    pytest.param(0.5, id="half-change"),
    pytest.param(2.0, id="double-change"),
]

OPS = [
    pytest.param(xp.add, id="add"),
    pytest.param(xp.subtract, id="subtract"),
]

SYMMETRY_TYPE = [
    pytest.param((False, lambda x: x), id="non-symmetric"),
    pytest.param((True, lambda x: x), id="symmetric"),
    pytest.param((True, lambda x: -x), id="skew-symmetric"),
    pytest.param((True, xp.conj), id="hermitian"),
    pytest.param((True, lambda x: -xp.conj(x)), id="skew-hermitian"),
]


@pytest.fixture(params=BLOCK_SIZES)
def block_sizes(request: pytest.FixtureRequest) -> NDArray:
    return request.param


@pytest.fixture(params=DSBSPARSE_TYPES)
def dsbsparse_type(request: pytest.FixtureRequest) -> DSBSparse:
    return request.param


@pytest.fixture(params=DSDBSPARSE_TYPES)
def dsdbsparse_type(request: pytest.FixtureRequest) -> DSDBSparse:
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


@pytest.fixture(params=GLOBAL_STACK_SHAPES)
def global_stack_shape(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=STACK_INDICES)
def stack_index(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=BLOCK_CHANGE_FACTORS)
def block_change_factor(request):
    return request.param


@pytest.fixture(params=OPS)
def op(request):
    return request.param


@pytest.fixture(params=SYMMETRY_TYPE)
def symmetry_type(request: pytest.FixtureRequest) -> bool:
    return request.param
