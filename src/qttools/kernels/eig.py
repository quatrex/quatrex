# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_array_module_name, get_device, get_host


@nb.njit(parallel=True, cache=True, no_rewrites=True)
def _eig_numba(A: NDArray) -> tuple[NDArray, NDArray]:
    """Computes the eigenvalues and eigenvectors of a matrix on a given location.

    Parallelized over the batch dimension with numba.

    Parameters
    ----------
    A : NDArray
        The matrix.

    Returns
    -------
    NDArray
        The eigenvalues.
    NDArray
        The eigenvectors.
    """

    n = A.shape[-1]
    batch_size = A.shape[0]
    ws = np.empty((batch_size, n), dtype=A.dtype)
    vs = np.empty((batch_size, n, n), dtype=A.dtype)

    for i in nb.prange(batch_size):
        w, v = np.linalg.eig(A[i])
        ws[i] = w
        vs[i] = v

    return ws, vs


def eig(
    A: NDArray,
    compute_module: str = "numpy",
    output_module: str | None = None,
) -> tuple[NDArray, NDArray]:
    """Computes the eigenvalues and eigenvectors of a matrix on a given location.

    To compute the eigenvalues and eigenvectors on the device with cupy
    is only possible if the cupy.linalg.eig function is available.

    Parameters
    ----------
    A : NDArray
        The matrix.
    compute_module : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "numpy" or "cupy".
    output_module : str, optional
        The location where to store the eigenvalues and eigenvectors.
        Can be either "numpy"
        or "cupy". If None, the output location is the same as the input location

    Returns
    -------
    NDArray
        The eigenvalues.
    NDArray
        The eigenvectors.

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

    if compute_module == "cupy" and hasattr(xp.linalg, "eig") is False:
        raise ValueError("Eig is not available in cupy.")

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
        w, v = xp.linalg.eig(A)
    elif compute_module == "numpy":
        batch_shape = A.shape[:-2]
        if A.shape[-1] != A.shape[-2]:
            raise ValueError("Matrix must be square.")
        # NOTE: more error handling with zero size could be done
        n = A.shape[-1]
        A = A.reshape((-1, n, n))
        w, v = _eig_numba(A)
        w = w.reshape(*batch_shape, n)
        v = v.reshape(*batch_shape, n, n)

    if output_module == "numpy" and compute_module == "cupy":
        w, v = get_host(w), get_host(v)
    elif output_module == "cupy" and compute_module == "numpy":
        w, v = get_device(w), get_device(v)

    return w, v
