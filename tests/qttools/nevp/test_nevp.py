# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, xp
from qttools.nevp import Beyn, Full


@pytest.mark.usefixtures("left")
def test_full(a_xx: tuple[NDArray, ...], left: bool):
    """Tests that the Full NEVP solver returns the correct result."""
    full_nevp = Full()
    if left:
        wrs, vrs, wls, vls = full_nevp(a_xx, left=left)
    else:
        wrs, vrs = full_nevp(a_xx, left=left)

    a_ji, a_ii, a_ij = a_xx

    for i in range(wrs.shape[1]):
        w = wrs[0, i]
        v = vrs[0, :, i] / xp.linalg.norm(vrs[0, :, i])

        assert xp.allclose((a_ji / w + a_ii + a_ij * w) @ v, 0)

    if left:
        for i in range(wrs.shape[1]):
            w = wls[0, i]
            v = vls[0, :, i] / xp.linalg.norm(vls[0, :, i])

            assert xp.allclose(v.conj().T @ (a_ji / w + a_ii + a_ij * w), 0)


@pytest.mark.usefixtures("subspace_nevp", "left")
def test_subspace(a_xx: tuple[NDArray, ...], subspace_nevp: Beyn, left: bool):
    """Tests that the subspace NEVP solver returns the correct result."""
    if left:
        wrs, vrs, wls, vls = subspace_nevp(a_xx, left=left)
    else:
        wrs, vrs = subspace_nevp(a_xx, left=left)

    a_ji, a_ii, a_ij = a_xx
    residuals = []
    for k in range(wrs.shape[1]):
        w = wrs[0, k]
        v = vrs[0, :, k] / xp.linalg.norm(vrs[0, :, k])
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals.append(xp.linalg.norm((a_ji / w + a_ii + a_ij * w) @ v))
    if left:
        for k in range(wrs.shape[1]):
            w = wls[0, k]
            v = vls[0, :, k] / xp.linalg.norm(vls[0, :, k])
            with np.errstate(divide="ignore", invalid="ignore"):
                residuals.append(
                    xp.linalg.norm(v.conj().T @ (a_ji / w + a_ii + a_ij * w))
                )

    residuals = xp.nan_to_num(xp.array(residuals))

    # Filter outlier eigenmodes (robust Z-score method).
    median = xp.median(residuals)
    median_abs_deviation = xp.median(xp.abs(residuals - median))
    z_scores = 0.6745 * (residuals - median) / median_abs_deviation
    spurious_mask = xp.abs(z_scores) > 30  # Very generous threshold.

    assert residuals[~spurious_mask].max() < 1e-10
