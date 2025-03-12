# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
from numba.typed import List

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
def _eig_numba(
    A: NDArray | List[NDArray],
    ws: NDArray | List[NDArray],
    vs: NDArray | List[NDArray],
    batch_size: int,
) -> None:
    """Computes the eigenvalues and eigenvectors of multiple matrices.

    Parallelized with numba.

    Parameters
    ----------
    A : NDArray | List[NDArray]
        The matrices.
    ws : NDArray | List[NDArray]
        The eigenvalues.
    vs : NDArray | List[NDArray]
        The eigenvectors.
    batch_size : int
        The number of matrices.

    """
    for i in nb.prange(batch_size):
        i = np.int64(i)
        w, v = np.linalg.eig(A[i])
        ws[i][:] = w
        vs[i][:] = v


@profiler.profile(level="debug")
def _eig_numpy(
    A: NDArray | List[NDArray],
) -> tuple[NDArray, NDArray] | tuple[List[NDArray], List[NDArray]]:
    """Computes the eigenvalues and eigenvectors of multiple matrices.

    Parameters
    ----------
    A : NDArray | List[NDArray]
        The matrices.

    Returns
    -------
    NDArray | List[NDArray]
        The eigenvalues.
    NDArray | List[NDArray]
        The eigenvectors.

    """

    if isinstance(A, list):
        A = List(A)
        w = List([np.empty((a.shape[-1]), dtype=a.dtype) for a in A])
        v = List([np.empty((a.shape[-1], a.shape[-1]), dtype=a.dtype) for a in A])
        batch_size = len(A)

        _eig_numba(A, w, v, batch_size)
    else:
        batch_shape = A.shape[:-2]
        if A.shape[-1] != A.shape[-2]:
            raise ValueError("Matrix must be square.")
        # NOTE: more error handling with zero size could be done
        n = A.shape[-1]
        A = A.reshape((-1, n, n))

        w = np.empty((A.shape[0], n), dtype=A.dtype)
        v = np.empty((A.shape[0], n, n), dtype=A.dtype)

        batch_size = A.shape[0]

        _eig_numba(A, w, v, batch_size)
        w = w.reshape(*batch_shape, n)
        v = v.reshape(*batch_shape, n, n)

    return w, v


@profiler.profile(level="debug")
def _eig_cupy(
    A: NDArray | List[NDArray],
) -> tuple[NDArray, NDArray] | tuple[List[NDArray], List[NDArray]]:
    """Computes the eigenvalues and eigenvectors of multiple matrices.

    Parameters
    ----------
    A : NDArray | List[NDArray]
        The matrices.

    Returns
    -------
    NDArray | List[NDArray]
        The eigenvalues.
    NDArray | List[NDArray]
        The eigenvectors.

    """
    if isinstance(A, list):
        w = []
        v = []
        for a in A:
            w_, v_ = xp.linalg.eig(a)
            w.append(w_)
            v.append(v_)
    else:
        w, v = xp.linalg.eig(A)


@profiler.profile(level="api")
def eig(
    A: NDArray | list[NDArray],
    compute_module: str = "numpy",
    output_module: str | None = None,
    use_pinned_memory: bool = True,
) -> tuple[NDArray, NDArray] | tuple[list[NDArray], list[NDArray]]:
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
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

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

    if compute_module == "cupy" and hasattr(xp.linalg, "eig") is False:
        raise ValueError("Eig is not available in cupy.")

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    if isinstance(A, (List, list)):
        if use_pinned_memory:
            A = [get_any_location_pinned(a, compute_module) for a in A]
        else:
            A = [get_any_location(a, compute_module) for a in A]
    else:
        if use_pinned_memory:
            A = get_any_location_pinned(A, compute_module)
        else:
            A = get_any_location(A, compute_module)

    if compute_module == "cupy":
        w, v = _eig_cupy(A)
    elif compute_module == "numpy":
        w, v = _eig_numpy(A)

    if isinstance(w, (List, list)):
        if use_pinned_memory:
            w = [get_any_location_pinned(w, output_module) for w in w]
            v = [get_any_location_pinned(v, output_module) for v in v]
        else:
            w = [get_any_location(w, output_module) for w in w]
            v = [get_any_location(v, output_module) for v in v]
    else:
        if use_pinned_memory:
            w, v = get_any_location_pinned(w, output_module), get_any_location_pinned(
                v, output_module
            )
        else:
            w, v = get_any_location(w, output_module), get_any_location(
                v, output_module
            )

    return w, v
