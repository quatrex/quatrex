# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from scipy import sparse

from qttools.datastructures import DSBCOO, DSBCSR

ARRAY_SHAPE = (20, 20)
DSBSPARSE_TYPES = [DSBCSR, DSBCOO]


BLOCK_SIZES = [
    pytest.param(np.array([2] * 10), id="constant-block-size"),
    pytest.param(np.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size"),
]


GLOBAL_STACK_SHAPES = [
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
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

STACK_INDICES = [
    pytest.param((5,), id="single"),
    pytest.param((slice(1, 4),), id="slice"),
    pytest.param((Ellipsis,), id="ellipsis"),
]


@pytest.fixture(autouse=True)
def coo() -> sparse.coo_array:
    """Returns a random complex sparse array."""
    return sparse.random(*ARRAY_SHAPE, density=0.3, format="coo", dtype=complex)


@pytest.fixture(params=DSBSPARSE_TYPES, autouse=True)
def dsbsparse_type(request):
    return request.param


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def block_sizes(request):
    return request.param


@pytest.fixture(params=GLOBAL_STACK_SHAPES, autouse=True)
def global_stack_shape(request):
    return request.param


@pytest.fixture(params=DENSIFY_BLOCKS)
def densify_blocks(request):
    return request.param


@pytest.fixture(params=ACCESSED_BLOCKS)
def accessed_block(request):
    return request.param


@pytest.fixture(params=STACK_INDICES)
def stack_index(request):
    return request.param
