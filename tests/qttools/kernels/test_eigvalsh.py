import numpy as np
import pytest
import scipy

from qttools import xp
from qttools.kernels.eigvalsh import eigvalsh
from qttools.utils.gpu_utils import get_device, get_host


@pytest.mark.usefixtures("n", "compute_module", "input_module", "output_module")
def test_eigvalsh(n: int, compute_module: str, input_module: str, output_module: str):
    """Tests the eig function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return

    rng = np.random.default_rng()

    A = rng.random((n, n)) + 1j * rng.random((n, n))

    A = (A + A.conj().swapaxes(-1, -2)) / 2

    if compute_module == "cupy":
        A = get_device(A)

    w = eigvalsh(A, compute_module=compute_module, output_module=output_module)

    # check residual on the host
    w = get_host(w)
    A = get_host(A)

    w_ref = scipy.linalg.eigvalsh(A)

    w = np.sort(w)
    w_ref = np.sort(w_ref)

    assert xp.allclose(w, w_ref)


@pytest.mark.usefixtures(
    "n", "batch_shape", "compute_module", "input_module", "output_module"
)
def test_eigvalsh_batched(
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

    rng = np.random.default_rng()

    A = rng.random((*batch_shape, n, n)) + 1j * rng.random((*batch_shape, n, n))

    A = (A + A.conj().swapaxes(-1, -2)) / 2

    if compute_module == "cupy":
        A = get_device(A)

    w = eigvalsh(A, compute_module=compute_module, output_module=output_module)

    assert w.shape[:-1] == batch_shape

    assert w.shape[-1] == n

    # check residual on the host
    w = get_host(w)
    A = get_host(A)

    w_ref = np.linalg.eigvalsh(A)

    w = np.sort(w)
    w_ref = np.sort(w_ref)

    assert xp.allclose(w, w_ref)


@pytest.mark.usefixtures("n", "compute_module", "input_module", "output_module")
def test_eigvalsh_generalized(
    n: int, compute_module: str, input_module: str, output_module: str
):
    """Tests the eig function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return

    rng = np.random.default_rng()

    A = rng.random((n, n)) + 1j * rng.random((n, n))
    B = rng.random((n, n)) + 1j * rng.random((n, n))

    A = (A + A.conj().swapaxes(-1, -2)) / 2

    B = (B + B.conj().swapaxes(-1, -2)) / 2
    B += n * np.eye(n)

    if compute_module == "cupy":
        B = get_device(B)
        A = get_device(A)

    w = eigvalsh(A, B=B, compute_module=compute_module, output_module=output_module)

    # check residual on the host
    w = get_host(w)
    A = get_host(A)
    B = get_host(B)

    w_ref = scipy.linalg.eigvalsh(A, b=B)

    w = np.sort(w)
    w_ref = np.sort(w_ref)

    assert xp.allclose(w, w_ref)


@pytest.mark.usefixtures(
    "n", "batch_shape", "compute_module", "input_module", "output_module"
)
def test_eigvalsh_generalized_batched(
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

    rng = np.random.default_rng()

    A = rng.random((*batch_shape, n, n)) + 1j * rng.random((*batch_shape, n, n))
    B = rng.random((*batch_shape, n, n)) + 1j * rng.random((*batch_shape, n, n))

    A = (A + A.conj().swapaxes(-1, -2)) / 2

    B = (B + B.conj().swapaxes(-1, -2)) / 2
    # need to be positive definite
    # loop over batch dimensions

    B = B.reshape((np.prod(batch_shape), n, n))
    for i in range(np.prod(batch_shape)):
        B[i] += n * np.eye(n)

    B = B.reshape((*batch_shape, n, n))

    if compute_module == "cupy":
        B = get_device(B)
        A = get_device(A)

    w = eigvalsh(A, B=B, compute_module=compute_module, output_module=output_module)

    assert w.shape[:-1] == batch_shape

    assert w.shape[-1] == n

    # check residual on the host
    w = get_host(w)
    A = get_host(A)
    B = get_host(B)

    w_ref = np.zeros_like(w)
    w_ref = w_ref.reshape((np.prod(batch_shape), n))

    A = A.reshape((np.prod(batch_shape), n, n))
    B = B.reshape((np.prod(batch_shape), n, n))

    for i in range(np.prod(batch_shape)):
        w_ref[i] = scipy.linalg.eigvalsh(A[i], b=B[i])

    w_ref = w_ref.reshape((*batch_shape, n))

    w = np.sort(w, axis=-1)
    w_ref = np.sort(w_ref, axis=-1)

    assert xp.allclose(w, w_ref)
