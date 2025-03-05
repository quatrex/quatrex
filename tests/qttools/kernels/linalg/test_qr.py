import numpy as np
import pytest

from qttools import xp
from qttools.kernels import linalg
from qttools.kernels.linalg.qr import _qr_numba
from qttools.utils.gpu_utils import get_host

if xp.__name__ == "cupy":
    import cupy as cp


@pytest.mark.usefixtures("m", "n", "batch_shape")
def test_qr_numba_ndarray(m: int, n: int, batch_shape: tuple[int, ...]):
    """Tests the _qr_numba_ndarray function."""

    rng = np.random.default_rng()

    batch_size = np.prod(batch_shape)

    A = rng.random((batch_size, m, n)) + 1j * rng.random((batch_size, m, n))

    q, r = _qr_numba(
        A,
    )

    assert xp.allclose(A, q @ r)


@pytest.mark.usefixtures("m", "n", "compute_module", "input_module", "output_module")
def test_qr(
    m: int,
    n: int,
    compute_module: str,
    input_module: str,
    output_module: str,
):
    """Tests the qr function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((m, n)) + 1j * rng.random((m, n))

    q, r = linalg.qr(
        A,
        compute_module=compute_module,
        output_module=output_module,
    )

    # check residual on the host
    q = get_host(q)
    r = get_host(r)
    A = get_host(A)

    assert xp.allclose(A, q @ r)


@pytest.mark.usefixtures(
    "m",
    "n",
    "batch_shape",
    "full_matrices",
    "compute_module",
    "input_module",
    "output_module",
)
def test_qr_batched(
    m: int,
    n: int,
    batch_shape: tuple[int, ...],
    compute_module: str,
    input_module: str,
    output_module: str,
):
    """Tests the qr function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((*batch_shape, m, n)) + 1j * rng.random((*batch_shape, m, n))

    q, r = linalg.qr(
        A,
        compute_module=compute_module,
        output_module=output_module,
    )

    q = get_host(q)
    r = get_host(r)
    A = get_host(A)

    assert xp.allclose(A, q @ r)
