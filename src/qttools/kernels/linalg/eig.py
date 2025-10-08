# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
import nvmath
from numba.typed import List
from nvmath.bindings import cusolverDn

from qttools import NDArray, xp
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_any_location, get_array_module_name

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
    if isinstance(
        A, list
    ):  # comment: soon there will be batched geev, so pls replace for loop then.
        w = []
        v = []
        for a in A:
            w_, v_ = xp.linalg.eig(a)
            w.append(w_)
            v.append(v_)
    else:
        w, v = xp.linalg.eig(A)

    return w, v


@profiler.profile(level="debug")
def _eig_nvmath(
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
            w_, v_ = _eig_nvmath_kernel(a)
            w.append(w_)
            v.append(v_)
    else:
        # Single matrix case
        w, v = _eig_nvmath_kernel(A[0, :, :])

    return w[xp.newaxis, :], v[xp.newaxis, :, :]


@profiler.profile(level="debug")
def _eig_nvmath_kernel(
    A: NDArray,
) -> tuple[NDArray, NDArray]:
    # Initialize cuSolver handle
    handle = cusolverDn.create()
    params = cusolverDn.create_params()

    n = A.shape[0]
    n_64 = np.int64(n)
    lda = n_64
    ldvl = n_64
    ldvr = n_64

    # Prepare input matrix (copy since xgeev modifies it) We need column major format here!
    # Comment: I assume xp is cupy here, please check how you want to handle this
    A_work = xp.asfortranarray(xp.copy(A))

    # Allocate output arrays
    w_complex = xp.zeros(n, dtype=xp.complex128)
    vl = xp.zeros(
        (n, n), dtype=xp.complex128, order="F"
    )  # Left eigenvectors (not computed)
    vr = xp.zeros((n, n), dtype=xp.complex128, order="F")  # Right eigenvectors
    info = xp.zeros(1, dtype=xp.int64)

    yes_vector = nvmath.bindings.cusolver.EigMode.VECTOR
    no_vector = nvmath.bindings.cusolver.EigMode.NOVECTOR

    # Query workspace size
    lwork_device, lwork_host = cusolverDn.xgeev_buffer_size(
        handle,
        params,
        no_vector,  # jobvl: don't compute left eigenvectors
        yes_vector,  # jobvr: compute right eigenvectors
        n_64,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_a
        A_work.data.ptr,
        lda,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_w
        w_complex.data.ptr,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_vl
        vl.data.ptr,
        ldvl,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_vr
        vr.data.ptr,
        ldvr,
        nvmath.CudaDataType.CUDA_C_64F,  # compute_type
    )

    workspace_device = xp.cuda.alloc(lwork_device)
    workspace_host = np.empty(lwork_host, dtype=xp.int8)
    # Compute eigenvalues and eigenvectors
    cusolverDn.xgeev(
        handle,
        params,
        no_vector,  # jobvl: don't compute left eigenvectors
        yes_vector,  # jobvr: compute right eigenvectors
        n_64,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_a
        A_work.data.ptr,
        lda,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_w
        w_complex.data.ptr,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_vl
        vl.data.ptr,
        ldvl,
        nvmath.CudaDataType.CUDA_C_64F,  # data_type_vr
        vr.data.ptr,
        ldvr,
        nvmath.CudaDataType.CUDA_C_64F,  # compute_type
        workspace_device.ptr,
        lwork_device,
        workspace_host.ctypes.data,
        lwork_host,
        info.data.ptr,
    )

    return w_complex, xp.ascontiguousarray(vr)


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
        Can be either "numpy" or "cupy". The default is "numpy".
        Can be either "numpy" or "cupy". The default is "numpy".
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

    # see comment below (of course nvmath doesn't call it "eig")
    # if compute_module == "cupy" and hasattr(xp.linalg, "eig") is False:
    #     raise ValueError("Eig is not available in cupy.")

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    if isinstance(A, (List, list)):
        A = [
            get_any_location(a, compute_module, use_pinned_memory=use_pinned_memory)
            for a in A
        ]
    else:
        A = get_any_location(A, compute_module, use_pinned_memory=use_pinned_memory)

    if compute_module == "cupy":
        """comment: You seem to use "cupy" to mean "in GPU memory" when calling get_any_location,
        and to use the cupy module when running eig so
          i let you decide how you want to handle this situation.
        """
        # w, v = _eig_cupy(A)
        w, v = _eig_nvmath(A)
    elif compute_module == "numpy":
        w, v = _eig_numpy(A)

    if isinstance(w, (List, list)):
        return (
            [
                get_any_location(wi, output_module, use_pinned_memory=use_pinned_memory)
                for wi in w
            ],
            [
                get_any_location(vi, output_module, use_pinned_memory=use_pinned_memory)
                for vi in v
            ],
        )
    else:
        return get_any_location(
            w, output_module, use_pinned_memory=use_pinned_memory
        ), get_any_location(v, output_module, use_pinned_memory=use_pinned_memory)
