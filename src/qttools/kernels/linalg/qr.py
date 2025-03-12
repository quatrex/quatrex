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
def _qr_numba(
    A: NDArray,
) -> tuple[NDArray, NDArray]:
    """Computes the QR decomposition of a batch of matrices.

    Parallelized with numba.

    Parameters
    ----------
    A : NDArray
        The matrices.

    Returns
    -------
    NDArray
        Unitary matrix Q in the QR decomposition.
    NDArray
        Upper triangular matrix R in the QR decomposition.

    """

    m = A.shape[-2]
    n = A.shape[-1]
    batch_size = A.shape[0]

    k = min(m, n)

    q = np.empty((batch_size, m, k), dtype=A.dtype)
    r = np.empty((batch_size, k, n), dtype=A.dtype)

    for i in nb.prange(batch_size):
        q_, r_ = np.linalg.qr(A[i])
        q[i] = q_
        r[i] = r_

    return q, r


@profiler.profile(level="api")
def qr(
    A: NDArray,
    compute_module: str = "numpy",
    output_module: str | None = None,
    use_pinned_memory: bool = True,
) -> tuple[NDArray, NDArray]:
    """Computes the QR decomposition of a batch of matrices.

    If compute_module is "numpy", the computation is done with numpy and parallelized with numba.
    Only mode 'reduced' is supported due to numba limitations.

    Parameters
    ----------
    A : NDArray
        The matrices.
    compute_module : str, optional
        The location where to compute the QR decomposition.
        Can be either "numpy" or "cupy".
    output_module : str, optional
        The location where to store the QR decomposition.
        Can be either "numpy"
        or "cupy". If None, the output location is the same as the input location
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

    Returns
    -------
    NDArray
        Unitary matrix Q in the QR decomposition.
    NDArray
        Upper triangular matrix R in the QR decomposition.

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
        q, r = xp.linalg.qr(A)
    elif compute_module == "numpy":
        batch_shape = A.shape[:-2]
        m = A.shape[-2]
        n = A.shape[-1]
        A = A.reshape((-1, m, n))

        q, r = _qr_numba(A)

        k = min(m, n)
        q = q.reshape((*batch_shape, m, k))
        r = r.reshape((*batch_shape, k, n))

    if use_pinned_memory:
        q, r = get_any_location_pinned(q, output_module), get_any_location_pinned(
            r, output_module
        )
    else:
        q, r = get_any_location(q, output_module), get_any_location(r, output_module)

    return q, r
