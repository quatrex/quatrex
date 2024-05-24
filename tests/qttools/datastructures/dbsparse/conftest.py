# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import pytest
from scipy import sparse
import numpy as np

ARRAY_SHAPE = (100, 100)


@pytest.fixture(scope="function", autouse=True)
def coo() -> sparse.coo_array:
    """Returns a random complex sparse array."""
    return sparse.random(*ARRAY_SHAPE, density=0.1, format="coo", dtype=complex)
