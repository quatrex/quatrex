import numpy as np
import pytest

ARRAY_SHAPE = (2, 2)


@pytest.fixture(scope="package", autouse=True)
def a_ii() -> np.ndarray:
    """Returns a random complex array."""
    return 1e2 * (np.random.rand(*ARRAY_SHAPE) + 1j * np.random.rand(*ARRAY_SHAPE))


@pytest.fixture(scope="package", autouse=True)
def a_ij() -> np.ndarray:
    """Returns a random complex array."""
    return np.random.rand(*ARRAY_SHAPE) + 1j * np.random.rand(*ARRAY_SHAPE)
