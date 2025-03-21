# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.lyapunov import LyapunovMemoizer, LyapunovSolver


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_correctness(
    inputs: tuple[NDArray, NDArray],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):

    lyapunov_solver.reduce_sparsity = reduce_sparsity

    a, q, _, _ = inputs

    print(a.shape)
    print(q.shape)

    x = lyapunov_solver(a, q, "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


@pytest.mark.parametrize("reduce_sparsity", [True, False])
def test_memoizer(
    inputs: tuple[NDArray, NDArray],
    lyapunov_solver: LyapunovSolver,
    reduce_sparsity: bool,
):
    """Tests that the Lyapunov memoizer works."""
    a, q, row_slice, col_slice = inputs

    lyapunov_solver.reduce_sparsity = reduce_sparsity

    lyapunov_solver = LyapunovMemoizer(lyapunov_solver, reduce_sparsity=reduce_sparsity)
    x = lyapunov_solver(a, q, "contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)

    # Add a little noise to the input matrices.
    a += 1e-3 * xp.random.randn(*a.shape)
    q += 1e-3 * xp.random.randn(*q.shape)

    a[..., row_slice, col_slice] = 0
    a[..., row_slice, col_slice] = 0

    # do not allow fallback
    lyapunov_solver.force_memoizing = True
    x = lyapunov_solver(a, q, "contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)
