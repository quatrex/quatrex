# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
from numba.typed import List

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_array_module_name, get_device, get_host


@nb.njit(parallel=True, cache=True, no_rewrites=True)
def _eig_numba_ndarray(A: NDArray) -> tuple[NDArray, NDArray]:
    """Computes the eigenvalues and eigenvectors of a batch of matrices.

    Parallelized over the batch dimension with numba.

    Parameters
    ----------
    A : NDArray
        The matrices.

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


@nb.njit(parallel=True, cache=True, no_rewrites=True)
def _eig_numba_list(A: list[NDArray], ws: list[NDArray], vs: list[NDArray]):
    """Computes the eigenvalues and eigenvectors of a list of matrices.

    Parallelized over the list dimension with numba.

    Parameters
    ----------
    A : list[NDArray]
        The matrices.
    ws : list[NDArray]
        The eigenvalues.
    vs : list[NDArray]
        The eigenvectors.
    """

    batch_size = len(A)

    for i in nb.prange(batch_size):
        w, v = np.linalg.eig(A[i])
        ws[i][:] = w
        vs[i][:] = v


def eig(
    A: NDArray | list[NDArray],
    compute_module: str = "numpy",
    output_module: str | None = None,
) -> tuple[NDArray, NDArray]:
    """Computes the eigenvalues and eigenvectors of matrices on a given location.

    To compute the eigenvalues and eigenvectors on the device with cupy
    is only possible if the cupy.linalg.eig function is available.

    A list of matrices is beneficial if not all the matrices have the same shape.
    Then the host numba implementation will still parallelize, but not the cupy implementation.
    Only over the list will be parallelized, further extra dimensions are not allowed.

    Assumes that all the input matrices are at the same location.

    Parameters
    ----------
    A : NDArray | list[NDArray]
        The matrices.
    compute_module : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "numpy" or "cupy".
    output_module : str, optional
        The location where to store the eigenvalues and eigenvectors.
        Can be either "numpy"
        or "cupy". If None, the output location is the same as the input location

    Returns
    -------
    NDArray | list[NDArray]
        The eigenvalues.
    NDArray | list[NDArray]
        The eigenvectors.

    """
    if isinstance(A, list):
        input_module = get_array_module_name(A[0])
        if not all(get_array_module_name(a) == input_module for a in A):
            raise ValueError("All matrices must be at the same location.")
        if not all(a.ndim == 2 for a in A):
            raise ValueError("Only 2D matrices are allowed with a list input.")
    else:
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
        # NOTE: ininstace checks are not very portable
        # TODO: check as well if NDArray
        if isinstance(A, list):
            A = [get_host(a) for a in A]
        else:
            A = get_host(A)
    elif compute_module == "cupy" and input_module == "numpy":
        if isinstance(A, list):
            A = [get_device(a) for a in A]
        else:
            A = get_device(A)

    if compute_module == "cupy":
        if isinstance(A, list):
            w = []
            v = []
            for a in A:
                w_, v_ = xp.linalg.eig(a)
                w.append(w_)
                v.append(v_)
        else:
            w, v = xp.linalg.eig(A)
    elif compute_module == "numpy":

        if isinstance(A, list):
            A = List(A)
            w = List([np.empty((a.shape[-1]), dtype=a.dtype) for a in A])
            v = List([np.empty((a.shape[-1], a.shape[-1]), dtype=a.dtype) for a in A])

            _eig_numba_list(A, w, v)
        else:
            batch_shape = A.shape[:-2]
            if A.shape[-1] != A.shape[-2]:
                raise ValueError("Matrix must be square.")
            # NOTE: more error handling with zero size could be done
            n = A.shape[-1]
            A = A.reshape((-1, n, n))

            w, v = _eig_numba_ndarray(A)
            w = w.reshape(*batch_shape, n)
            v = v.reshape(*batch_shape, n, n)

    if output_module == "numpy" and compute_module == "cupy":
        if isinstance(w, List):
            w = [get_host(w) for w in w]
            v = [get_host(v) for v in v]
        else:
            w, v = get_host(w), get_host(v)
    elif output_module == "cupy" and compute_module == "numpy":
        if isinstance(w, List):
            w = [get_device(w) for w in w]
            v = [get_device(v) for v in v]
        else:
            print(w.shape, v.shape)
            w, v = get_device(w), get_device(v)

    return w, v
