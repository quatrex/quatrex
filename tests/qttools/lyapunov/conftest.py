# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from itertools import product

import pytest

from qttools import NDArray, xp
from qttools.lyapunov import Doubling, LyapunovSolver, Spectral

BLOCK_SIZE = [
    pytest.param(11, id="11x11"),
    pytest.param(27, id="27x27"),
]
SHAPES = [
    pytest.param(((11, 11), (11, 11)), id="11x11"),
    pytest.param(((3, 11, 11), (3, 11, 11)), id="3x11x11"),
    pytest.param(((3, 11, 11), (2, 2, 3, 11, 11)), id="2x2x3x11x11"),
]

ROW_REDUCTION = [
    pytest.param(0, id="0"),
    pytest.param(2, id="2"),
]

COL_REDUCTION = [
    pytest.param(0, id="0"),
    pytest.param(2, id="2"),
]

LYAPUNOV_SOLVERS = [
    pytest.param(Spectral(), id="Spectral"),
    pytest.param(Doubling(), id="Doubling"),
]


# Create a list of all combinations of the parameters
input_params = list(product(SHAPES, ROW_REDUCTION, COL_REDUCTION))


@pytest.fixture(params=input_params)
def inputs(request: pytest.FixtureRequest) -> tuple[NDArray, NDArray]:
    """Returns some random complex matrices."""
    shape, row_reduction, col_reduction = request.param
    a_shape, q_shape = shape[0][0][0], shape[0][0][1]
    size = a_shape[-1]
    a = xp.random.rand(*a_shape) + 1j * xp.random.rand(*a_shape)
    a /= 10 * size  # Ensure that the spectral radius is less than 1.
    for higher_dim_indices in xp.ndindex(a_shape[:-2]):
        xp.fill_diagonal(
            a[higher_dim_indices + (slice(None), slice(None))],
            xp.sum(xp.abs(a[higher_dim_indices + (slice(None), slice(None))]), axis=1),
        )

    q = xp.random.rand(*q_shape) + 1j * xp.random.rand(*q_shape)

    row_reduction = row_reduction[0][0]
    col_reduction = col_reduction[0][0]

    row_slice = slice(a.shape[-1] - row_reduction, row_reduction, -1)
    col_slice = slice(a.shape[-1] - col_reduction, col_reduction, -1)

    a[..., row_slice, col_slice] = 0
    q[..., row_slice, col_slice] = 0

    return a, q, row_slice, col_slice


@pytest.fixture(params=LYAPUNOV_SOLVERS, autouse=True)
def lyapunov_solver(request: pytest.FixtureRequest) -> LyapunovSolver:
    """Returns a Lyapunov solver."""
    return request.param
