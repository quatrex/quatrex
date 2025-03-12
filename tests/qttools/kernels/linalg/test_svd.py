import numpy as np
import pytest

from qttools import xp
from qttools.kernels import linalg
from qttools.kernels.linalg.svd import _svd_numba
from qttools.utils.gpu_utils import get_host

if xp.__name__ == "cupy":
    import cupy as cp


@pytest.mark.usefixtures("m", "n", "full_matrices", "batch_shape")
def test_svd_numba_ndarray(
    m: int, n: int, full_matrices: bool, batch_shape: tuple[int, ...]
):
    """Tests the _svd_numba_ndarray function."""

    rng = np.random.default_rng()

    batch_size = np.prod(batch_shape)

    A = rng.random((batch_size, m, n)) + 1j * rng.random((batch_size, m, n))

    u, s, vh = _svd_numba(
        A,
        full_matrices=full_matrices,
    )

    k = min(m, n)
    if full_matrices:
        u = u[:, :, :k]
        s = s[:, :k]
        vh = vh[:, :k, :]

    s_dense = np.zeros((batch_size, k, k), dtype=A.dtype)
    for i in range(batch_size):
        s_dense[i] = np.diag(s[i])

    assert xp.allclose(A, u @ s_dense @ vh)


@pytest.mark.usefixtures(
    "m",
    "n",
    "full_matrices",
    "compute_module",
    "input_module",
    "output_module",
    "use_pinned_memory",
)
def test_svd(
    m: int,
    n: int,
    full_matrices: bool,
    compute_module: str,
    input_module: str,
    output_module: str,
    use_pinned_memory: bool,
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

    u, s, vh = linalg.svd(
        A,
        compute_module=compute_module,
        output_module=output_module,
        full_matrices=full_matrices,
        use_pinned_memory=use_pinned_memory,
    )

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


@pytest.mark.usefixtures(
    "m",
    "n",
    "batch_shape",
    "full_matrices",
    "compute_module",
    "input_module",
    "output_module",
    "use_pinned_memory",
)
def test_svd_batched(
    m: int,
    n: int,
    batch_shape: tuple[int, ...],
    full_matrices: bool,
    compute_module: str,
    input_module: str,
    output_module: str,
    use_pinned_memory: bool,
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

    A = rng.random((*batch_shape, m, n)) + 1j * rng.random((*batch_shape, m, n))

    u, s, vh = linalg.svd(
        A,
        compute_module=compute_module,
        output_module=output_module,
        full_matrices=full_matrices,
        use_pinned_memory=use_pinned_memory,
    )

    u = get_host(u)
    s = get_host(s)
    vh = get_host(vh)
    A = get_host(A)

    k = min(m, n)
    if full_matrices:
        u = u.reshape((-1, m, m))
        vh = vh.reshape((-1, n, n))
    else:
        u = u.reshape((-1, m, k))
        vh = vh.reshape((-1, k, n))
    s = s.reshape((-1, k))
    A = A.reshape((-1, m, n))

    if full_matrices:
        u = u[:, :, :k]
        s = s[:, :k]
        vh = vh[:, :k, :]

    batch_size = np.prod(batch_shape)
    s_dense = np.zeros((batch_size, k, k), dtype=A.dtype)
    for i in range(batch_size):
        s_dense[i] = np.diag(s[i])

    assert xp.allclose(A, u @ s_dense @ vh)
