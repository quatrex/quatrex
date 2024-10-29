import numpy as np
import pytest

from qttools.nevp import NEVP
from qttools.obc import OBCMemoizer, Spectral
from qttools.utils.gpu_utils import xp


@pytest.mark.usefixtures("nevp", "x_ii_formula")
def test_correctness(
    a_xx: tuple[np.ndarray, ...], nevp: NEVP, x_ii_formula: str, contact: str
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(nevp=nevp, x_ii_formula=x_ii_formula)
    a_ji, a_ii, a_ij = a_xx
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)


@pytest.mark.usefixtures("nevp", "x_ii_formula")
def test_correctness_batch(
    a_xx: tuple[np.ndarray, ...], nevp: NEVP, x_ii_formula: str, contact: str
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(nevp=nevp, x_ii_formula=x_ii_formula)
    a_ji, a_ii, a_ij = a_xx
    a_ji = xp.stack([a_ji for __ in range(10)])
    a_ii = xp.stack([a_ii for __ in range(10)])
    a_ij = xp.stack([a_ij for __ in range(10)])
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)


@pytest.mark.usefixtures("nevp", "x_ii_formula")
def test_memoizer(
    a_xx: tuple[np.ndarray, ...], nevp: NEVP, x_ii_formula: str, contact: str
):
    """Tests that the Memoization works."""
    spectral = Spectral(nevp=nevp, x_ii_formula=x_ii_formula)
    spectral = OBCMemoizer(spectral)
    a_ji, a_ii, a_ij = a_xx
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert np.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)
    # Add a little noise to the input matrices.
    a_ji += xp.random.randn(*a_ji.shape)
    a_ii += xp.random.randn(*a_ii.shape)
    a_ij += xp.random.randn(*a_ij.shape)

    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert np.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)
