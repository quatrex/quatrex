import numpy as np
import numpy.linalg as npla
import pytest

from qttools.nevp import NEVP
from qttools.obc import Spectral


@pytest.mark.usefixtures("nevp")
def test_correctness(a_xx: tuple[np.ndarray], nevp: NEVP, contact: str):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(nevp=nevp)
    a_ji, a_ii, a_ij = a_xx
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)


@pytest.mark.usefixtures("nevp")
def test_correctness_batch(a_xx: tuple[np.ndarray], nevp: NEVP, contact: str):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(nevp=nevp)
    a_ji, a_ii, a_ij = a_xx
    factors = np.random.rand(10)
    a_ji = np.stack([f * a_ji for f in factors])
    a_ii = np.stack([f * a_ii for f in factors])
    a_ij = np.stack([f * a_ij for f in factors])
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)
