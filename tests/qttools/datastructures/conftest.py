# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest

from qttools.datastructures import DSBCOO, DSBCSR

DSBSPARSE_TYPES = [DSBCSR, DSBCOO]

BLOCK_SIZES = [
    pytest.param(np.array([2] * 10), id="constant-block-size"),
    pytest.param(np.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size"),
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

STACK_INDICES = [
    pytest.param((5,), id="single"),
    pytest.param((slice(1, 4),), id="slice"),
    pytest.param((Ellipsis,), id="ellipsis"),
]


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def block_sizes(request):
    return request.param


@pytest.fixture(params=DSBSPARSE_TYPES, autouse=True)
def dsbsparse_type(request):
    return request.param


@pytest.fixture(params=DENSIFY_BLOCKS)
def densify_blocks(request):
    return request.param


@pytest.fixture(params=ACCESSED_BLOCKS)
def accessed_block(request):
    return request.param


@pytest.fixture(params=ACCESSED_ELEMENTS)
def accessed_element(request):
    return request.param


@pytest.fixture(params=STACK_INDICES)
def stack_index(request):
    return request.param
