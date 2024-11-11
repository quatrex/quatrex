import numpy as np
import pytest

from qttools import NDArray, xp
from qttools.nevp import Beyn, Full


def test_full(a_xx: tuple[NDArray, ...]):
    """Tests that the Full NEVP solver returns the correct result."""
    full_nevp = Full()
    ws, vs = full_nevp(a_xx)
    a_ji, a_ii, a_ij = a_xx

    for i in range(ws.shape[1]):
        w = ws[0, i]
        v = vs[0, :, i] / xp.linalg.norm(vs[0, :, i])

        assert xp.allclose((a_ji / w + a_ii + a_ij * w) @ v, 0)


@pytest.mark.usefixtures("subspace_nevp")
def test_subspace(a_xx: tuple[NDArray, ...], subspace_nevp: Beyn):
    """Tests that the subspace NEVP solver returns the correct result."""
    ws, vs = subspace_nevp(a_xx)

    a_ji, a_ii, a_ij = a_xx
    residuals = []
    for k in range(ws.shape[1]):
        w = ws[0, k]
        v = vs[0, :, k] / xp.linalg.norm(vs[0, :, k])
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals.append(xp.linalg.norm((a_ji / w + a_ii + a_ij * w) @ v))

    residuals = xp.nan_to_num(xp.array(residuals))

    # Filter outlier eigenmodes (robust Z-score method).
    median = xp.median(residuals)
    median_abs_deviation = xp.median(xp.abs(residuals - median))
    z_scores = 0.6745 * (residuals - median) / median_abs_deviation
    spurious_mask = xp.abs(z_scores) > 30  # Very generous threshold.

    assert residuals[~spurious_mask].max() < 1e-10
