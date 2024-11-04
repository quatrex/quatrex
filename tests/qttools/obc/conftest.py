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

X_II_FORMULAS = ["self-energy", "direct"]

BLOCK_SIZE = [
    pytest.param(21, id="21x21"),
    pytest.param(18, id="18x18"),
]

BLOCK_SECTIONS = [
    pytest.param(1, id="no-subblocks"),
    pytest.param(3, id="three-subblocks"),
]

CONTACTS = ["left", "right"]


@pytest.fixture(params=X_II_FORMULAS)
def x_ii_formula(request) -> str:
    """Returns a NEVP solver."""
    return request.param


@pytest.fixture(params=NEVP_SOLVERS)
def nevp(request) -> NEVP:
    """Returns a NEVP solver."""
    return request.param


@pytest.fixture(params=BLOCK_SECTIONS)
def block_sections(request) -> int:
    """Returns the number of block sections."""
    return request.param


@pytest.fixture(params=BLOCK_SIZE, autouse=True)
def a_xx(request) -> tuple[np.ndarray, ...]:
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
