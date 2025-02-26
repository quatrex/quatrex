import numpy as np
import pytest

from qttools import xp
from qttools.kernels.eig import eig
from qttools.utils.gpu_utils import get_host

if xp.__name__ == "cupy":
    import cupy as cp


@pytest.mark.usefixtures(
    "matrix_size", "compute_module", "input_module", "output_module"
)
def test_eig(
    matrix_size: int, compute_module: str, input_module: str, output_module: str
):
    """Tests the eig function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return
    if xp.__name__ == "cupy" and (hasattr(xp.linalg, "eig") is False):
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((matrix_size, matrix_size))

    w, v = eig(A, compute_module=compute_module, output_module=output_module)

    # check residual on the host
    w = get_host(w)
    v = get_host(v)
    A = get_host(A)

    for i in range(matrix_size):
        assert xp.allclose(A @ v[:, i], w[i] * v[:, i])
