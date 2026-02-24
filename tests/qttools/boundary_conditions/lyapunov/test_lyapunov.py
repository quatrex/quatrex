# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.boundary_conditions.lyapunov import LyapunovSolver, LyapunovSystem
from quatrex.core.config import MemoizerConfig


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_correctness(
    inputs: tuple[NDArray, NDArray],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):

    config = MemoizerConfig()
    config.mode = "off"

    lyapunov_system = LyapunovSystem(
        lyapunov_solver,
        reduce_sparsity=reduce_sparsity,
        config=config,
    )

    a, q, _, _ = inputs

    _, _, x = lyapunov_system((a, q), "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_correctness_zeros(
    inputs: tuple[NDArray, NDArray],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):

    config = MemoizerConfig()
    config.mode = "off"

    lyapunov_system = LyapunovSystem(
        lyapunov_solver,
        reduce_sparsity=reduce_sparsity,
        config=config,
    )

    a, q, _, _ = inputs

    a[:] = 0

    _, _, x = lyapunov_system((a, q), "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_memoizer(
    inputs: tuple[NDArray, NDArray],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):
    """Tests that the Lyapunov memoizer works."""
    a, q, row_slice, col_slice = inputs

    config = MemoizerConfig()
    config.mode = "force-after-first"

    lyapunov_system = LyapunovSystem(
        lyapunov_solver,
        reduce_sparsity=reduce_sparsity,
        config=config,
    )
    _, _, x = lyapunov_system((a, q), contact="contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)

    # Add a little noise to the input matrices.
    a = a + 1e-3 * a
    q = q + 1e-3 * q

    a[..., row_slice, col_slice] = 0
    a[..., row_slice, col_slice] = 0

    _, _, x = lyapunov_system((a, q), contact="contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)
