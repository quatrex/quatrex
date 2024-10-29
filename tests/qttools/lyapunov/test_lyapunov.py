import numpy as np

from qttools.lyapunov import LyapunovMemoizer, LyapunovSolver
from qttools.utils.gpu_utils import get_device, get_host


def test_correctness(
    inputs: tuple[np.ndarray, np.ndarray], lyapunov_solver: LyapunovSolver
):
    """Tests that the Lyapunov solver returns the correct result."""
    a, q = inputs
    x = get_host(lyapunov_solver(get_device(a), get_device(q), "contact"))

    assert np.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_correctness_batch(
    inputs: tuple[np.ndarray, np.ndarray], lyapunov_solver: LyapunovSolver
):
    """Tests that the batched Lyapunov solver returns the correct result."""
    a, q = inputs
    a = np.stack([a for __ in range(10)])
    q = np.stack([q for __ in range(10)])

    x = get_host(lyapunov_solver(get_device(a), get_device(q), "contact"))

    assert np.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_memoizer(
    inputs: tuple[np.ndarray, np.ndarray], lyapunov_solver: LyapunovSolver
):
    """Tests that the Lyapunov memoizer works."""
    a, q = inputs
    lyapunov_solver = LyapunovMemoizer(lyapunov_solver)
    x = get_host(lyapunov_solver(get_device(a), get_device(q), "contact"))
    assert np.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)

    # Add a little noise to the input matrices.
    a, q = inputs
    a += 1e-3 * np.random.randn(*a.shape)
    q += 1e-3 * np.random.randn(*q.shape)

    x = get_host(lyapunov_solver(get_device(a), get_device(q), "contact"))
    assert np.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)
