# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, sparse, xp
from qttools.nevp import NEVP, Full


@pytest.mark.usefixtures("nevp")
def test_nevp(a_xx: tuple[NDArray, ...], nevp: NEVP):
    """Tests that the subspace NEVP solver returns the correct result."""
    ws, vs = nevp(a_xx)

    a_ji, a_ii, a_ij = a_xx
    residuals = []
    for e in range(ws.shape[0]):
        for k in range(ws.shape[1]):
            w = ws[e, k]
            v = vs[e, :, k] / xp.linalg.norm(vs[0, :, k])
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


@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("provide_sparsity", [False, True])
def test_full(a_xx: tuple[NDArray, ...], reduce: bool, provide_sparsity: bool):
    """Tests that the full NEVP solver returns the correct result."""

    a_xx = tuple(a_x.copy() for a_x in a_xx)

    if reduce:
        size = a_xx[0].shape[-1]
        # Introduce some zero columns in a_ji and a_ij
        a_xx[0][..., : size // 2] = 0
        a_xx[2][..., size // 2 :] = 0

    a_xx_sparsity = None
    if provide_sparsity:
        if len(a_xx[0].shape) > 2:
            a_xx_sparsity = tuple(sparse.csc_matrix(a_x[0]) for a_x in a_xx)
        else:
            a_xx_sparsity = tuple(sparse.csc_matrix(a) for a in a_xx)

    full_nevp = Full(a_xx_sparsity=a_xx_sparsity, reduce=reduce)
    ws, vs = full_nevp(a_xx)

    a_ji, a_ii, a_ij = a_xx
    residuals = []
    for e in range(ws.shape[0]):
        for k in range(ws.shape[1]):
            w = ws[e, k]
            v = vs[e, :, k] / xp.linalg.norm(vs[0, :, k])
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
