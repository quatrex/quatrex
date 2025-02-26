import numpy as np
import pytest

from qttools import xp
from qttools.kernels.svd import svd
from qttools.utils.gpu_utils import get_host

if xp.__name__ == "cupy":
    import cupy as cp


@pytest.mark.usefixtures(
    "m", "n", "full_matrices", "compute_module", "input_module", "output_module"
)
def test_svd(
    m: int,
    n: int,
    full_matrices: bool,
    compute_module: str,
    input_module: str,
    output_module: str,
):
    """Tests the svd function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((m, n)) + 1j * rng.random((m, n))

    u, s, vh = svd(
        A,
        compute_module=compute_module,
        output_module=output_module,
        full_matrices=full_matrices,
    )

    # check residual on the host
    u = get_host(u)
    s = get_host(s)
    vh = get_host(vh)
    A = get_host(A)

    if full_matrices:
        k = min(m, n)
        u = u[:, :k]
        s = s[:k]
        vh = vh[:k, :]

    assert xp.allclose(A, u @ np.diag(s) @ vh)
