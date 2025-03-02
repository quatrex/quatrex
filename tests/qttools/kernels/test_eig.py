import numpy as np
import pytest
from numba.typed import List

from qttools import xp
from qttools.kernels.eig import _eig_numba_list, _eig_numba_ndarray, eig
from qttools.utils.gpu_utils import get_host

if xp.__name__ == "cupy":
    import cupy as cp


@pytest.mark.usefixtures("n", "batch_shape")
def test_eig_numba_ndarray(n: int, batch_shape: tuple[int, ...]):
    """Tests the _eig_numba_ndarray function."""

    rng = np.random.default_rng()

    batch_size = np.prod(batch_shape)

    A = rng.random((batch_size, n, n)) + 1j * rng.random((batch_size, n, n))

    w, v = _eig_numba_ndarray(A)

    for b in range(batch_size):
        for i in range(n):
            assert xp.allclose(A[b] @ v[b][:, i], w[b][i] * v[b][:, i])


@pytest.mark.usefixtures("n", "batch_shape")
def test_eig_numba_list(n: int, batch_shape: tuple[int, ...]):
    """Tests the _eig_numba_list function."""

    rng = np.random.default_rng()

    batch_size = np.prod(batch_shape)

    A = []
    for _ in range(batch_size):
        A.append(rng.random((n, n)) + 1j * rng.random((n, n)))
    for _ in range(batch_size):
        A.append(rng.random((n // 2, n // 2)) + 1j * rng.random((n // 2, n // 2)))
    for _ in range(batch_size // 2):
        A.append(rng.random((n // 3, n // 3)) + 1j * rng.random((n // 3, n // 3)))

    batch_size = len(A)

    A = List(A)
    w = List([np.empty((a.shape[-1]), dtype=a.dtype) for a in A])
    v = List([np.empty((a.shape[-1], a.shape[-1]), dtype=a.dtype) for a in A])

    _eig_numba_list(A, w, v)

    for b in range(batch_size):
        n_ = A[b].shape[-1]
        for i in range(n_):
            assert xp.allclose(A[b] @ v[b][:, i], w[b][i] * v[b][:, i])


@pytest.mark.usefixtures(
    "n", "compute_module", "input_module", "output_module", "if_list"
)
def test_eig(
    n: int,
    compute_module: str,
    input_module: str,
    output_module: str,
    if_list: bool,
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

    A = rng.random((n, n)) + 1j * rng.random((n, n))

    if if_list:
        A = [A]

    w, v = eig(A, compute_module=compute_module, output_module=output_module)

    if if_list:
        w = w[0]
        v = v[0]
        A = A[0]

    # check residual on the host
    w = get_host(w)
    v = get_host(v)
    A = get_host(A)

    for i in range(n):
        assert xp.allclose(A @ v[:, i], w[i] * v[:, i])


@pytest.mark.usefixtures(
    "n", "batch_shape", "compute_module", "input_module", "output_module", "if_list"
)
def test_eig_batched(
    n: int,
    batch_shape: tuple[int, ...],
    compute_module: str,
    input_module: str,
    output_module: str,
    if_list: bool,
):
    """Tests the eig function."""

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        return
    if compute_module == "cupy" and (hasattr(xp.linalg, "eig") is False):
        return

    if if_list and len(batch_shape) > 1:
        return

    if input_module == "cupy":
        rng = cp.random.default_rng()
    elif input_module == "numpy":
        rng = np.random.default_rng()

    A = rng.random((*batch_shape, n, n)) + 1j * rng.random((*batch_shape, n, n))

    if if_list:
        A = [a for a in A]

    w, v = eig(A, compute_module=compute_module, output_module=output_module)

    if if_list:
        A = xp.array([a for a in A])
        w = xp.array([w_ for w_ in w])
        v = xp.array([v_ for v_ in v])

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
