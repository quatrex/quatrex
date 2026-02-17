# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
from numba.typed import List

try:
    import nvmath
    from nvmath.bindings import cusolverDn

    nvmath_available = True
except ImportError:
    nvmath_available = False

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_any_location, get_array_module_name

library_to_location = {
    "numpy": "numpy",
    "cupy": "cupy",
    "nvmath": "cupy",
}


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
        in_dtype = A[0].dtype
    else:
        in_dtype = A.dtype

    # real matrices have complex eigenvalues/vectors in general
    out_dtype = xp.complex128

    if in_dtype != xp.complex128:
        raise ValueError("nvmath implementation only supports complex128 matrices.")

    if isinstance(A, list):
        w = []
        v = []
        for a in A:
            w_, v_ = _eig_nvmath_kernel(a)
            w.append(w_)
            v.append(v_)
    elif A.ndim == 2:
        w, v = _eig_nvmath_kernel(A)
    else:
        batch_size = A.shape[0]
        w = xp.zeros((batch_size, A.shape[1]), dtype=out_dtype)
        v = xp.zeros((batch_size, A.shape[1], A.shape[1]), dtype=out_dtype)
        for i in range(batch_size):
            w[i, :], v[i, :, :] = _eig_nvmath_kernel(A[i, :, :])

    return w, v


def _eig_nvmath_kernel(
    A: NDArray,
) -> tuple[NDArray, NDArray]:

    # TODO: this will not work for real A
    # and the cupy binding should be followed

    # TODO: not recommended to create/destroy handle every time
    handle = cusolverDn.create()
    params = cusolverDn.create_params()

    n = A.shape[0]
    n_64 = np.int64(n)
    lda = n_64
    ldvl = n_64
    ldvr = n_64

    # Prepare input matrix (copy since xgeev modifies it) We need column major format here!
    A_work = xp.asfortranarray(xp.copy(A))

    # Allocate output arrays
    w_complex = xp.zeros(n, dtype=xp.complex128)
    vl = xp.zeros((n, n), dtype=xp.complex128, order="F")
    # Left eigenvectors (not computed)
    vr = xp.zeros((n, n), dtype=xp.complex128, order="F")
    info = xp.zeros(1, dtype=xp.int64)

    yes_vector = nvmath.bindings.cusolver.EigMode.VECTOR
    no_vector = nvmath.bindings.cusolver.EigMode.NOVECTOR

    try:
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
    finally:
        cusolverDn.destroy_params(params)
        cusolverDn.destroy(handle)

    return w_complex, xp.ascontiguousarray(vr)


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
        Can be either "numpy" or "cupy" or "nvmath". The default is "numpy".
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

    if compute_module == "nvmath" and nvmath_available is False:
        raise ValueError("nvmath is not available.")

    if xp.__name__ == "numpy" and (
        compute_module in ["cupy", "nvmath"]
        or output_module in ["cupy", "nvmath"]
        or input_module in ["cupy", "nvmath"]
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    if isinstance(A, (List, list)):
        A = [
            get_any_location(
                a,
                library_to_location[compute_module],
                use_pinned_memory=use_pinned_memory,
            )
            for a in A
        ]
    else:
        A = get_any_location(
            A, library_to_location[compute_module], use_pinned_memory=use_pinned_memory
        )

    if compute_module == "cupy":
        w, v = _eig_cupy(A)
    elif compute_module == "nvmath":
        w, v = _eig_nvmath(A)
    elif compute_module == "numpy":
        w, v = _eig_numpy(A)
    else:
        raise ValueError(
            'compute_module must be either "numpy", "cupy" or "nvmath".',
        )

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
