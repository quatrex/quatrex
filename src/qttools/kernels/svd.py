# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_array_module_name, get_device, get_host


@nb.njit(parallel=True, cache=True, no_rewrites=True)
def _svd_numba_ndarray(
    A: NDArray,
    full_matrices: bool = True,
) -> tuple[NDArray, NDArray, NDArray]:
    """Computes the singular value decomposition of a batch of matrices.

    Parallelized over the batch dimension with numba. ndim of A must be 3.

    Parameters
    ----------
    A : NDArray
        The matrices.
    full_matrices : bool, optional
        Whether to compute the full matrices u and vh (see numpy.linalg.svd).

    Returns
    -------
    NDArray
        The left singular vectors.
    NDArray
        The singular values.
    NDArray
        The right singular vectors.

    """

    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)
    batch_size = A.shape[0]

    s = np.empty((batch_size, k), dtype=A.dtype)
    if full_matrices:
        u = np.empty((batch_size, m, m), dtype=A.dtype)
        vh = np.empty((batch_size, n, n), dtype=A.dtype)
    else:
        u = np.empty((batch_size, m, k), dtype=A.dtype)
        vh = np.empty((batch_size, k, n), dtype=A.dtype)

    for i in nb.prange(batch_size):
        u_, s_, vh_ = np.linalg.svd(A[i], full_matrices=full_matrices)
        s[i] = s_
        u[i] = u_
        vh[i] = vh_

    return u, s, vh


def svd(
    A: NDArray,
    full_matrices: bool = True,
    compute_module: str = "numpy",
    output_module: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Computes the singular value decomposition of a matrix on a given location.

    The kwargs compute_uv and hermitian are not supported.
    They are implicitly set to True and False, respectively.

    Parameters
    ----------
    A : NDArray
        The matrix.
    full_matrices : bool, optional
        Whether to compute the full matrices u and vh (see numpy.linalg.svd).
    compute_module : str, optional
        The location where to compute the singular value decomposition.
        Can be either "numpy" or "cupy".
    output_module : str, optional
        The location where to store the singular value decomposition.
        Can be either "numpy"
        or "cupy". If None, the output location is the same as the input location

    Returns
    -------
    NDArray
        The left singular vectors.
    NDArray
        The singular values.
    NDArray
        The right singular vectors.

    """
    input_module = get_array_module_name(A)

    if output_module is None:
        output_module = input_module

    if output_module not in ["numpy", "cupy"]:
        raise ValueError(f"Invalid output location: {output_module}")
    if compute_module not in ["numpy", "cupy"]:
        raise ValueError(f"Invalid compute location: {compute_module}")
    if input_module not in ["numpy", "cupy"]:
        raise ValueError(f"Invalid input location: {input_module}")

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    # memcopy to correct location
    if compute_module == "numpy" and input_module == "cupy":
        A = get_host(A)
    elif compute_module == "cupy" and input_module == "numpy":
        A = get_device(A)

    if compute_module == "cupy":
        u, s, vh = xp.linalg.svd(A, full_matrices=full_matrices)
    elif compute_module == "numpy":
        batch_shape = A.shape[:-2]
        m = A.shape[-2]
        n = A.shape[-1]
        A = A.reshape((-1, m, n))

        u, s, vh = _svd_numba_ndarray(A, full_matrices)

        k = min(m, n)
        if full_matrices:
            u = u.reshape((*batch_shape, m, m))
            vh = vh.reshape((*batch_shape, n, n))
        else:
            u = u.reshape((*batch_shape, m, k))
            vh = vh.reshape((*batch_shape, k, n))
        s = s.reshape((*batch_shape, k))

    if output_module == "numpy" and compute_module == "cupy":
        u, s, vh = get_host(u), get_host(s), get_host(vh)
    elif output_module == "cupy" and compute_module == "numpy":
        u, s, vh = get_device(u), get_device(s), get_device(vh)

    return u, s, vh
