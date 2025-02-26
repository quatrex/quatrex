import numpy as np
import pytest

from qttools import xp
from qttools.kernels.eig import eig
from qttools.utils.gpu_utils import get_host

if xp.__name__ == "cupy":
    import cupy as cp


@pytest.mark.usefixtures("n", "compute_module", "input_module", "output_module")
def test_eig(n: int, compute_module: str, input_module: str, output_module: str):
    """Tests the eig function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return
    if compute_module == "cupy" and (hasattr(xp.linalg, "eig") is False):
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((n, n)) + 1j * rng.random((n, n))

    w, v = eig(A, compute_module=compute_module, output_module=output_module)

    # check residual on the host
    w = get_host(w)
    v = get_host(v)
    A = get_host(A)

    for i in range(n):
        assert xp.allclose(A @ v[:, i], w[i] * v[:, i])


@pytest.mark.usefixtures(
    "n", "batch_shape", "compute_module", "input_module", "output_module"
)
def test_eig_batched(
    n: int,
    batch_shape: tuple[int, ...],
    compute_module: str,
    input_module: str,
    output_module: str,
):
    """Tests the eig function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return
    if compute_module == "cupy" and (hasattr(xp.linalg, "eig") is False):
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((*batch_shape, n, n)) + 1j * rng.random((*batch_shape, n, n))

    w, v = eig(A, compute_module=compute_module, output_module=output_module)

    assert w.shape[:-1] == batch_shape
    assert v.shape[:-2] == batch_shape

    assert w.shape[-1] == n
    assert v.shape[-2] == n
    assert v.shape[-1] == n

    # check residual on the host
    w = get_host(w)
    v = get_host(v)
    A = get_host(A)

    batch_size = np.prod(batch_shape)
    w = w.reshape(batch_size, n)
    v = v.reshape(batch_size, n, n)
    A = A.reshape(batch_size, n, n)

    for j in range(batch_size):
        for i in range(n):
            assert xp.allclose(A[j] @ v[j, :, i], w[j, i] * v[j, :, i])
