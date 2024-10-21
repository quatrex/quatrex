# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.
import numpy as np
import pytest

from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP, Beyn, Full
from qttools.utils.gpu_utils import get_device

# NOTE: The matrices we generate here generally have eigenvalues with an
# absolute value around 130. We set the outer radius to 150 and the
# inner radius to 0.9. The subspace dimension is chosen sufficiently
# large to capture all the eigenvalues in that annulus. The number of
# quadrature points is set such that non-spurious eigenvalues get
# approximated accurately enough.
NEVP_SOLVERS = [
    pytest.param(Beyn(r_o=200, r_i=0.9, c_hat=23, num_quad_points=13), id="Beyn"),
    pytest.param(Full(), id="Full"),
]

BLOCK_SIZE = [
    pytest.param(20, id="20x20"),
    pytest.param(17, id="17x17"),
]

CONTACTS = ["left", "right"]


@pytest.fixture(params=NEVP_SOLVERS)
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
    np.fill_diagonal(arr, np.abs(arr.sum(-1) + arr.diagonal()))

    arr = get_device(arr)

    blocks = _block_view(_block_view(arr, -1, 2), -2, 2)
    return blocks[1, 0], blocks[0, 0], blocks[0, 1]


@pytest.fixture(params=CONTACTS, autouse=True)
def contact(request) -> str:
    """Returns a contact."""
    return request.param
