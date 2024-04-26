# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors. All rights reserved.
import numpy as np
import pytest

ARRAY_SHAPE = (2, 2)


@pytest.fixture(scope="package", autouse=True)
def a_ii() -> np.ndarray:
    """Returns a random complex array."""
    return np.random.rand(*ARRAY_SHAPE) + 1j * np.random.rand(*ARRAY_SHAPE)


@pytest.fixture(scope="package", autouse=True)
def a_ij() -> np.ndarray:
    """Returns a random complex array."""
    return np.random.rand(*ARRAY_SHAPE) + 1j * np.random.rand(*ARRAY_SHAPE)
