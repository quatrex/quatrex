import numpy as np
import pytest

from qttools.nevp import Beyn, Full


def test_full(a_xx: tuple[np.ndarray]):
    """Tests that the Full NEVP solver returns the correct result."""
    full_nevp = Full()
    ws, vs = full_nevp(a_xx)
    a_ji, a_ii, a_ij = a_xx

    for i in range(ws.shape[1]):
        w = ws[0, i]
        v = vs[0, :, i] / np.linalg.norm(vs[0, :, i])

        assert np.allclose((a_ji / w + a_ii + a_ij * w) @ v, 0)


@pytest.mark.usefixtures("subspace_nevp")
def test_subspace(a_xx: tuple[np.ndarray], subspace_nevp: Beyn):
    """Tests that the subspace NEVP solver returns the correct result."""
    ws, vs = subspace_nevp(a_xx)

    a_ji, a_ii, a_ij = a_xx
    residuals = []
    for k in range(ws.shape[1]):
        w = ws[0, k]
        v = vs[0, :, k] / np.linalg.norm(vs[0, :, k])
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals.append(np.linalg.norm((a_ji / w + a_ii + a_ij * w) @ v))

    residuals = np.nan_to_num(np.array(residuals))
    # Filter spurious eigenmodes.
    spurious_mask = (np.mean(residuals) < residuals) | (np.isnan(residuals))

    assert residuals[~spurious_mask].mean() < 1e-4
