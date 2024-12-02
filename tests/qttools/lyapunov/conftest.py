# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.lyapunov import Doubling, LyapunovSolver, Spectral, Vectorize

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
def inputs(request: pytest.FixtureRequest) -> tuple[NDArray, NDArray]:
    """Returns some random complex matrices."""
    size = request.param
    a = xp.random.rand(size, size) + 1j * xp.random.rand(size, size)
    a /= 10 * size  # Ensure that the spectral radius is less than 1.
    xp.fill_diagonal(a, xp.sum(xp.abs(a), axis=1))

    q = xp.random.rand(size, size) + 1j * xp.random.rand(size, size)

    return a, q


@pytest.fixture(params=LYAPUNOV_SOLVERS, autouse=True)
def lyapunov_solver(request: pytest.FixtureRequest) -> LyapunovSolver:
    """Returns a Lyapunov solver."""
    return request.param
