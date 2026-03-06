# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.boundary_conditions.lyapunov import (
    LyapunovSolver,
    LyapunovSystem,
    LyapunovSystemReducer,
)


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_correctness(
    inputs: tuple[NDArray, NDArray, slice, slice],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):

    lyapunov_system_reducer = LyapunovSystemReducer(reduce_sparsity=reduce_sparsity)

    lyapunov_system = LyapunovSystem(
        lyapunov_solver,
        system_reducer=lyapunov_system_reducer,
        mode="off",
    )

    a, q, _, _ = inputs

    x, _, _ = lyapunov_system((a, q), "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_correctness_zeros(
    inputs: tuple[NDArray, NDArray, slice, slice],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):
    lyapunov_system_reducer = LyapunovSystemReducer(reduce_sparsity=reduce_sparsity)

    lyapunov_system = LyapunovSystem(
        lyapunov_solver,
        system_reducer=lyapunov_system_reducer,
        mode="off",
    )

    a, q, _, _ = inputs

    a[:] = 0

    x, _, _ = lyapunov_system((a, q), "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_memoizer(
    inputs: tuple[NDArray, NDArray, slice, slice],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):
    """Tests that the Lyapunov memoizer works."""
    a, q, row_slice, col_slice = inputs

    lyapunov_system_reducer = LyapunovSystemReducer(reduce_sparsity=reduce_sparsity)

    lyapunov_system = LyapunovSystem(
        lyapunov_solver,
        system_reducer=lyapunov_system_reducer,
        mode="force-after-first",
    )
    x, _, _ = lyapunov_system((a, q), contact="contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)

    # Add a little noise to the input matrices.
    a = a + 1e-3 * a
    q = q + 1e-3 * q

    a[..., row_slice, col_slice] = 0
    a[..., row_slice, col_slice] = 0

    x, _, _ = lyapunov_system((a, q), contact="contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)
