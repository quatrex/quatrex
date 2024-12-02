# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.lyapunov import LyapunovMemoizer, LyapunovSolver


def test_correctness(inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver):
    """Tests that the Lyapunov solver returns the correct result."""
    a, q = inputs
    x = lyapunov_solver(a, q, "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_correctness_batch(
    inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver
):
    """Tests that the batched Lyapunov solver returns the correct result."""
    a, q = inputs
    a = xp.stack([a for __ in range(10)])
    q = xp.stack([q for __ in range(10)])

    x = lyapunov_solver(a, q, "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_memoizer(inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver):
    """Tests that the Lyapunov memoizer works."""
    a, q = inputs
    lyapunov_solver = LyapunovMemoizer(lyapunov_solver)
    x = lyapunov_solver(a, q, "contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)

    # Add a little noise to the input matrices.
    a, q = inputs
    a += 1e-3 * xp.random.randn(*a.shape)
    q += 1e-3 * xp.random.randn(*q.shape)

    x = lyapunov_solver(a, q, "contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)
