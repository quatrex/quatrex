import numpy as np
import numpy.linalg as npla

from qttools.obc.full import Full


def test_correctness(a_ii: np.ndarray, a_ij: np.ndarray):
    """Tests that the OBC return the correct result."""
    full = Full()
    x_ii = full(a_ii=a_ii, a_ij=a_ij, a_ji=a_ij.conj().T, contact=None)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ij.conj().T @ x_ii @ a_ij))


def test_correctness_batch(a_ii: np.ndarray, a_ij: np.ndarray):
    """Tests that the OBC return the correct result."""
    full = Full()
    a_ii = np.stack(10 * [a_ii])
    a_ji = np.stack(10 * [a_ij.conj().T])
    a_ij = np.stack(10 * [a_ij])
    x_ii = full(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=None)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ji @ x_ii @ a_ij))
