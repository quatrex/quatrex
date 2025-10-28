# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

X_II_FORMULAS = ["self-energy", "direct"]

BLOCK_SIZE = [
    pytest.param(21, id="21x21"),
    pytest.param(18, id="18x18"),
]

BLOCK_SECTIONS = [
    pytest.param(1, id="no-subblocks"),
    pytest.param(3, id="three-subblocks"),
]

BATCH_SIZE = [
    pytest.param(1, id="single-batch"),
    pytest.param(3, id="three-batches"),
]

CONTACTS = ["left", "right"]


@pytest.fixture(params=X_II_FORMULAS)
def x_ii_formula(request: pytest.FixtureRequest) -> str:
    """Returns a NEVP solver."""
    return request.param


@pytest.fixture(params=BLOCK_SIZE)
def block_size(request: pytest.FixtureRequest) -> int:
    """Returns the block size."""
    return request.param


@pytest.fixture(params=BLOCK_SECTIONS)
def block_sections(request: pytest.FixtureRequest) -> int:
    """Returns the number of block sections."""
    return request.param


@pytest.fixture(params=BATCH_SIZE)
def batch_size(request: pytest.FixtureRequest) -> int:
    """Returns the block size."""
    return request.param


@pytest.fixture(params=CONTACTS, autouse=True)
def contact(request: pytest.FixtureRequest) -> str:
    """Returns a contact."""
    return request.param
