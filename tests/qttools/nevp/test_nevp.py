# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, xp
from qttools.nevp import Beyn, Full


@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("reduce", [False, True])
def test_full(a_xx: tuple[NDArray, ...], left: bool, reduce: bool):
    """Tests that the Full NEVP solver returns the correct result."""

    a_sparsity = tuple((a != 0).astype(xp.float32) for a in a_xx)

    batch_size = 1 if len(a_xx[0].shape) == 2 else a_xx[0].shape[0]

    if len(a_xx[0].shape) > 2:
        a_sparsity = tuple(xp.sum(a, axis=0) for a in a_sparsity)

    full_nevp = Full(a_sparsity=a_sparsity, reduce=reduce)
    if left:
        wrs, vrs, wls, vls = full_nevp(a_xx, left=left)
    else:
        wrs, vrs = full_nevp(a_xx, left=left)

    if a_xx[0].ndim == 2:
        a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

    a_ji, a_ii, a_ij = a_xx
    for b in range(batch_size):

        for i in range(wrs.shape[1]):
            w = wrs[b, i]
            v = vrs[b, :, i] / xp.linalg.norm(vrs[b, :, i])

            if xp.abs(w) < 1e-12:
                continue
            elif xp.abs(w) > 1e12:
                continue

            assert xp.allclose((a_ji[b] / w + a_ii[b] + a_ij[b] * w) @ v, 0)

        if left:
            for i in range(wrs.shape[1]):
                w = wls[b, i]
                v = vls[b, :, i] / xp.linalg.norm(vls[b, :, i])

                if xp.abs(w) < 1e-12:
                    continue
                elif xp.abs(w) > 1e12:
                    continue

                assert xp.allclose(
                    v.conj().T @ (a_ji[b] / w + a_ii[b] + a_ij[b] * w), 0
                )


@pytest.mark.usefixtures("subspace_nevp")
@pytest.mark.parametrize("left", [False, True])
def test_subspace(a_xx: tuple[NDArray, ...], subspace_nevp: Beyn, left: bool):
    """Tests that the subspace NEVP solver returns the correct result."""

    batch_size = 1 if len(a_xx[0].shape) == 2 else a_xx[0].shape[0]

    if left:
        wrs, vrs, wls, vls = subspace_nevp(a_xx, left=left)
    else:
        wrs, vrs = subspace_nevp(a_xx, left=left)

    if a_xx[0].ndim == 2:
        a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

    a_ji, a_ii, a_ij = a_xx

    residuals = []
    for b in range(batch_size):
        for k in range(wrs.shape[1]):
            w = wrs[b, k]
            v = vrs[b, :, k] / xp.linalg.norm(vrs[b, :, k])

            if xp.abs(w) < subspace_nevp.r_i:
                continue
            elif xp.abs(w) > subspace_nevp.r_o:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                residuals.append(
                    xp.linalg.norm((a_ji[b] / w + a_ii[b] + a_ij[b] * w) @ v)
                )
        if left:
            for k in range(wrs.shape[1]):
                w = wls[b, k]
                v = vls[b, :, k] / xp.linalg.norm(vls[b, :, k])

                if xp.abs(w) < subspace_nevp.r_i:
                    continue
                elif xp.abs(w) > subspace_nevp.r_o:
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    residuals.append(
                        xp.linalg.norm(
                            v.conj().T @ (a_ji[b] / w + a_ii[b] + a_ij[b] * w)
                        )
                    )

    residuals = xp.nan_to_num(xp.array(residuals))

    # Filter outlier eigenmodes (robust Z-score method).
    median = xp.median(residuals)
    median_abs_deviation = xp.median(xp.abs(residuals - median))
    z_scores = 0.6745 * (residuals - median) / median_abs_deviation
    spurious_mask = xp.abs(z_scores) > 30  # Very generous threshold.

    # assert some eigenvalues were found
    assert not xp.all(spurious_mask)

    if subspace_nevp.use_qr:
        # Single shot beyn with QR is less numerically stable.
        assert residuals[~spurious_mask].max() < 1e-3
    else:
        assert residuals[~spurious_mask].max() < 1e-4
