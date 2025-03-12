# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np

from qttools import NDArray, xp
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import (
    get_any_location,
    get_any_location_pinned,
    get_array_module_name,
)

profiler = Profiler()


@profiler.profile(level="debug")
@nb.njit(parallel=True, cache=True, no_rewrites=True)
def _svd_numba(
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


@profiler.profile(level="api")
def svd(
    A: NDArray,
    full_matrices: bool = True,
    compute_module: str = "numpy",
    output_module: str | None = None,
    use_pinned_memory: bool = True,
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
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

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

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    # memcopy to correct location
    if use_pinned_memory:
        A = get_any_location_pinned(A, compute_module)
    else:
        A = get_any_location(A, compute_module)

    if compute_module == "cupy":
        u, s, vh = xp.linalg.svd(A, full_matrices=full_matrices)
    elif compute_module == "numpy":
        batch_shape = A.shape[:-2]
        m = A.shape[-2]
        n = A.shape[-1]
        A = A.reshape((-1, m, n))

        u, s, vh = _svd_numba(A, full_matrices)

        k = min(m, n)
        if full_matrices:
            u = u.reshape((*batch_shape, m, m))
            vh = vh.reshape((*batch_shape, n, n))
        else:
            u = u.reshape((*batch_shape, m, k))
            vh = vh.reshape((*batch_shape, k, n))
        s = s.reshape((*batch_shape, k))

    if use_pinned_memory:
        u, s, vh = (
            get_any_location_pinned(u, output_module),
            get_any_location_pinned(s, output_module),
            get_any_location_pinned(vh, output_module),
        )
    else:
        u, s, vh = (
            get_any_location(u, output_module),
            get_any_location(s, output_module),
            get_any_location(vh, output_module),
        )

    return u, s, vh
