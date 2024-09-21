# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.
import numpy as np
import pytest

from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP, Beyn

NEVP_SOLVERS = [
    pytest.param(Beyn(r_o=10, r_i=0.9, c_hat=5, num_quad_points=10), id="Beyn-small"),
    pytest.param(Beyn(r_o=10, r_i=0.9, c_hat=15, num_quad_points=50), id="Beyn-large"),
]

BLOCK_SIZE = [
    pytest.param(20, id="20x20"),
    pytest.param(17, id="17x17"),
]


@pytest.fixture(params=NEVP_SOLVERS, autouse=True)
def nevp(request) -> NEVP:
    """Returns a NEVP solver."""
    return request.param


@pytest.fixture(params=BLOCK_SIZE, autouse=True)
def a_xx(request) -> np.ndarray:
    """Returns some random complex boundary blocks."""
    size = request.param * 2
    # Generate a decaying random complex array.
    arr = np.triu(np.arange(size, 0, -1) + np.arange(size)[:, np.newaxis])
    arr = arr.astype(np.complex128)
    arr += arr.T
    arr **= 2
    # Add some noise.
    arr += np.random.rand(size, size) * arr + 1j * np.random.rand(size, size) * arr
    # Normalize.
    arr /= size**2
    # Make it diagonally dominant.
    np.fill_diagonal(arr, 2 * np.abs(arr.sum(-1).max() + arr.diagonal()))
    # Filter out small values to make it sparse.
    arr[arr < 1] = 0

    blocks = _block_view(_block_view(arr, -1, 2), -2, 2)
    return blocks[1, 0], blocks[0, 0], blocks[0, 1]
