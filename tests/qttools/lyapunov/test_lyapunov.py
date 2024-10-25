import numpy as np

from qttools.lyapunov import Lyapunov


def test_correctness(inputs: tuple[np.ndarray, np.ndarray], lyapunov_solver: Lyapunov):
    """Tests that the Lyapunov solver returns the correct result."""
    a, q = inputs
    x = lyapunov_solver(a, q, "contact")

    assert np.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_correctness_batch(
    inputs: tuple[np.ndarray, np.ndarray], lyapunov_solver: Lyapunov
):
    """Tests that the batched Lyapunov solver returns the correct result."""
    a, q = inputs
    a = np.stack([a for __ in range(10)])
    q = np.stack([q for __ in range(10)])

    x = lyapunov_solver(a, q, "contact")

    assert np.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)
