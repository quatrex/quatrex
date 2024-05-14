import pytest
from scipy import sparse

ARRAY_SHAPE = (100, 100)


@pytest.fixture(scope="function", autouse=True)
def coo() -> sparse.coo_array:
    """Returns a random complex sparse array."""
    return sparse.random(*ARRAY_SHAPE, density=0.1, format="coo", dtype=complex)
