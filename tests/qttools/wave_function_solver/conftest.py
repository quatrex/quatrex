# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import pytest

NUM_ROWS = [
    pytest.param(8, id="8-rows"),
    pytest.param(13, id="13-rows"),
]
NUM_RHS = [
    pytest.param(1, id="1-rhs"),
    pytest.param(5, id="5-rhs"),
]


@pytest.fixture(params=NUM_ROWS, autouse=True)
def n(request: pytest.FixtureRequest) -> int:
    """Fixture to provide the number of rows for the sparse matrix."""
    return request.param


@pytest.fixture(params=NUM_RHS, autouse=True)
def m(request: pytest.FixtureRequest) -> int:
    """Fixture to provide the number of right-hand sides for the solver."""
    return request.param
