import numpy as np
import pytest

from qttools.nevp import NEVP, Beyn

BLOCK_SIZE = [
    pytest.param(51, id="51x51"),
    pytest.param(27, id="27x27"),
]

# NOTE: The matrices we generate generally have their eigenvalues close
# to the unit circle. We set the outer radius to 1.2 and the inner
# radius to 0.9. The subspace dimension is chosen sufficiently large to
# capture all the eigenvalues. The number of quadrature points is set to
# a very large number to ensure that the non-spurious eigenvalues get
# approximated very accurately.
SUBSPACE_NEVP_SOLVERS = [
    pytest.param(Beyn(r_o=1.2, r_i=0.9, c_hat=60, num_quad_points=200), id="Beyn")
]


# TODO: It's a good idea to generalize the tests with input data
# constructed from a hand-picked eigenvalues and eigenvectors. This
# allows us to choose when the subspace solvers should find the chosen
# eigenvalues and eigenvectors.
@pytest.fixture(params=BLOCK_SIZE, autouse=True)
def a_xx(request) -> np.ndarray:
    """Returns some random complex boundary blocks."""
    size = request.param
    a_xx = tuple(
        np.random.rand(size, size) + 1j * np.random.rand(size, size) for _ in range(3)
    )
    return a_xx


@pytest.fixture(params=SUBSPACE_NEVP_SOLVERS)
def subspace_nevp(request) -> NEVP:
    """Returns a NEVP solver."""
    return request.param
