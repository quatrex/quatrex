# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, xp
from qttools.nevp import NEVP


@pytest.mark.usefixtures("nevp")
def test_nevp(a_xx: tuple[NDArray, ...], nevp: NEVP):
    """Tests that the subspace NEVP solver returns the correct result."""
    wrs, vrs = nevp(a_xx)

    a_ji, a_ii, a_ij = a_xx
    residuals = []
    for e in range(wrs.shape[0]):
        for k in range(wrs.shape[1]):
            w = wrs[e, k]
            v = vrs[e, :, k] / xp.linalg.norm(vrs[0, :, k])
            with np.errstate(divide="ignore", invalid="ignore"):
                residuals.append(
                    xp.linalg.norm((a_ji[e] / w + a_ii[e] + a_ij[e] * w) @ v)
                    / xp.linalg.norm(w)
                )

    residuals = xp.nan_to_num(xp.array(residuals))

    # Filter outlier eigenmodes (robust Z-score method).
    median = xp.median(residuals)
    median_abs_deviation = xp.median(xp.abs(residuals - median))
    z_scores = 0.6745 * (residuals - median) / median_abs_deviation
    spurious_mask = xp.abs(z_scores) > 30  # Very generous threshold.

    # assert some eigenvalues were found
    assert not xp.all(spurious_mask)

    assert residuals[~spurious_mask].max() < 1e-5
