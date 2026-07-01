# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

BLOCK_SIZE = [
    pytest.param(21, id="21x21"),
    pytest.param(18, id="18x18"),
    pytest.param(32, id="21x21"),
]

BLOCK_SECTIONS = [
    pytest.param(1, id="no-subblocks"),
    pytest.param(2, id="no-subblocks"),
    pytest.param(3, id="three-subblocks"),
    pytest.param(4, id="three-subblocks"),
]

BATCH_SIZE = [
    pytest.param(1, id="single-batch"),
    pytest.param(3, id="three-batches"),
]


@pytest.fixture(params=BLOCK_SIZE)
def block_size(request: pytest.FixtureRequest) -> int:
    """Returns the block size."""
    return request.param


@pytest.fixture(params=BLOCK_SECTIONS)
def block_sections(request: pytest.FixtureRequest) -> int:
    """Returns the number of block sections."""
    return request.param


@pytest.fixture(params=BLOCK_SECTIONS)
def block_sections_x(request: pytest.FixtureRequest) -> int:
    """Returns the number of block sections."""
    return request.param


@pytest.fixture(params=BLOCK_SECTIONS)
def block_sections_y(request: pytest.FixtureRequest) -> int:
    """Returns the number of block sections."""
    return request.param


@pytest.fixture(params=BATCH_SIZE)
def batch_size(request: pytest.FixtureRequest) -> int:
    """Returns the block size."""
    return request.param
