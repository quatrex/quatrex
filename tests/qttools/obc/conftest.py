# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP, Beyn, Full

# NOTE: The matrices we generate here generally have eigenvalues with an
# absolute value around 130. We set the outer radius to 150 and the
# inner radius to 0.9. The subspace dimension is chosen sufficiently
# large to capture all the eigenvalues in that annulus. The number of
# quadrature points is set such that non-spurious eigenvalues get
# approximated accurately enough.
NEVP_SOLVERS = [
    pytest.param(Beyn(r_o=200, r_i=0.9, m_0=23, num_quad_points=13), id="Beyn"),
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

BATCH_SIZE = [
    pytest.param(1, id="single-batch"),
    pytest.param(3, id="three-batches"),
]

CONTACTS = ["left", "right"]

TWO_SIDED = [True, False]

TREAT_PAIRWISE = [True, False]

RESIDUAL_NORMALIZATION_FORMULAS = ["operator", "eigenvalue", None]


@pytest.fixture(params=X_II_FORMULAS)
def x_ii_formula(request: pytest.FixtureRequest) -> str:
    """Returns a NEVP solver."""
    return request.param


@pytest.fixture(params=NEVP_SOLVERS)
def nevp(request: pytest.FixtureRequest) -> NEVP:
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


@pytest.fixture(params=BLOCK_SIZE, autouse=True)
def a_xx(request: pytest.FixtureRequest) -> tuple[NDArray, NDArray, NDArray]:
    """Returns some random complex boundary blocks."""
    size = request.param * 2
    # Generate a decaying random complex array.
    arr = xp.triu(xp.arange(size, 0, -1) + xp.arange(size)[:, xp.newaxis])
    arr = arr.astype(xp.complex128)
    arr += arr.T
    arr **= 2
    # Add some noise.
    arr += xp.random.rand(size, size) * arr + 1j * xp.random.rand(size, size) * arr
    # Normalize.
    arr /= size**2
    # Make it diagonally dominant.
    xp.fill_diagonal(arr, xp.abs(arr.sum(-1) + arr.diagonal()))

    blocks = _block_view(_block_view(arr, -1, 2), -2, 2)
    return blocks[1, 0], blocks[0, 0], blocks[0, 1]


@pytest.fixture(params=CONTACTS, autouse=True)
def contact(request: pytest.FixtureRequest) -> str:
    """Returns a contact."""
    return request.param


@pytest.fixture(params=TWO_SIDED)
def two_sided(request: pytest.FixtureRequest) -> bool:
    """Whether only right eigenvectors or both are used."""
    return request.param


@pytest.fixture(params=TREAT_PAIRWISE)
def treat_pairwise(request: pytest.FixtureRequest) -> bool:
    """Whether the eigenvalues are pairwise filtered."""
    return request.param


@pytest.fixture(params=RESIDUAL_NORMALIZATION_FORMULAS)
def residual_normalization(request: pytest.FixtureRequest) -> str | None:
    """Returns a residual normalization formula."""
    return request.param
