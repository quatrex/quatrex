import numpy as np
import pytest

from qttools.lyapunov import Doubling, Spectral, Vectorize

BLOCK_SIZE = [
    pytest.param(11, id="11x11"),
    pytest.param(27, id="27x27"),
]

LYAPUNOV_SOLVERS = [
    pytest.param(Spectral(), id="Spectral"),
    pytest.param(Doubling(), id="Doubling"),
    pytest.param(Vectorize(), id="Vectorize"),
]


@pytest.fixture(params=BLOCK_SIZE, autouse=True)
def inputs(request) -> tuple[np.ndarray, np.ndarray]:
    """Returns some random complex matrices."""
    size = request.param
    a = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    a /= 10 * size  # Ensure that the spectral radius is less than 1.
    np.fill_diagonal(a, np.sum(np.abs(a), axis=1))

    q = np.random.rand(size, size) + 1j * np.random.rand(size, size)

    return a, q


@pytest.fixture(params=LYAPUNOV_SOLVERS, autouse=True)
def lyapunov_solver(request) -> Spectral | Doubling | Vectorize:
    """Returns a Lyapunov solver."""
    return request.param
