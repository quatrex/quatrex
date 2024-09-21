import numpy as np
import numpy.linalg as npla

from qttools.nevp import NEVP
from qttools.obc import Spectral


def test_correctness(a_xx: tuple[np.ndarray], nevp: NEVP):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(nevp=nevp)
    a_ji, a_ii, a_ij = a_xx
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="left")
    assert np.allclose(x_ii, npla.inv(a_ii - a_ji @ x_ii @ a_ij))
