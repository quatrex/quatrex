# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import inspect

from mpi4py import MPI

from qttools import NDArray, xp
from qttools.profiling import Profiler

profiler = Profiler()

if xp.__name__ == "cupy":
    import cupyx


@profiler.profile(level="debug")
def get_array_module_name(arr: NDArray) -> str:
    """Given an array, returns the array's module name.

    This works for `numpy` even when `cupy` is not available.

    Parameters
    ----------
    arr : NDArray
        The array to check.

    Returns
    -------
    str
        The array module name used by the array.

    """
    submodule = inspect.getmodule(type(arr))
    return submodule.__name__.split(".")[0]


@profiler.profile(level="debug")
def get_host(arr: NDArray, out: None | NDArray = None) -> NDArray:
    """Returns the host array of the given array.

    Note: special behaviour if numpy is used:
    If out is not set, then the returned array is the same as the input
    and the pointers alias.

    Parameters
    ----------
    arr : NDArray
        The array to convert.
    out : NDArray, optional
        The output array.

    Returns
    -------
    np.ndarray
        The equivalent numpy array.

    """
    if get_array_module_name(arr) == "numpy":
        if out is None:
            return arr
        out[:] = arr
        return out
    return xp.asnumpy(arr, out=out)


@profiler.profile(level="debug")
def get_device(arr: NDArray, out: None | NDArray = None) -> NDArray:
    """Returns the device array of the given array.

    Note: special behaviour if cupy is used:
    If out is not set, then the returned array is the same as the input
    and the pointers alias.

    Parameters
    ----------
    arr : NDArray
        The array to convert.
    out : NDArray, optional
        The output array.

    Returns
    -------
    NDArray
        The equivalent cupy array.

    """
    if get_array_module_name(arr) == "cupy" or xp.__name__ == "numpy":
        if out is None:
            return arr
        out[:] = arr
        return out
    if out is None:
        out = xp.empty_like(arr)
        out.set(arr)
        return out
    out.set(arr)
    return out


@profiler.profile(level="debug")
def get_any_location(
    arr: NDArray,
    output_module: str,
    use_pinned_memory: bool = False,
):
    """Returns the array in the desired location.

    Parameters
    ----------
    arr : NDArray
        The array to convert.
    output_module : str
        The desired location.
        The location can be either "numpy" or "cupy".
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

    Returns
    -------
    NDArray
        The equivalent array in the desired location

    """

    input_module = get_array_module_name(arr)

    arr_in = arr
    if (
        use_pinned_memory
        and input_module == "numpy"
        and output_module == "cupy"
        and xp.__name__ == "cupy"
    ):
        # detect if host memory is not pinned
        if (
            xp.cuda.runtime.pointerGetAttributes(arr.ctypes.data).type
            != xp.cuda.runtime.memoryTypeHost
        ):
            arr_in = empty_like_pinned(arr)
            arr_in[:] = arr

    arr_out = None
    if (
        use_pinned_memory
        and input_module == "cupy"
        and output_module == "numpy"
        and xp.__name__ == "cupy"
    ):
        # Fix issue that for get/asnumpy, both arrays need to be contiguous
        arr_in = xp.ascontiguousarray(arr)
        arr_out = empty_like_pinned(arr_in)

    if output_module == "numpy":
        return get_host(arr_in, arr_out)
    elif output_module == "cupy":
        arr_out = get_device(arr_in, arr_out)
        # IF pinnend memory is used,
        # then the h2d copy is asynchronous and we need to synchronize
        synchronize_current_stream()
        return arr_out
    else:
        raise ValueError(f"Invalid output location: {output_module}")


@profiler.profile(level="debug")
def empty_pinned(
    shape: int | tuple[int, ...],
    dtype: xp.dtype = float,
    order: str = "C",
):
    """Returns a new, uninitialized NumPy array with the given shape
    and dtype. The array is allocated in pinned memory if using cupy.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the empty array.
    dtype : data-type, optional
        Desired data-type for the array. Default is `float`.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in memory.
        Default is 'C'.

    Returns
    -------
    NDArray
        The empty array.

    .. seealso:: :func:`numpy.empty` :func:`cupy.empty` :func:`cupyx.empty_pinned`

    """

    if xp.__name__ == "cupy":
        return cupyx.empty_pinned(shape, dtype=dtype, order=order)
    else:
        return xp.empty(shape, dtype=dtype, order=order)


@profiler.profile(level="debug")
def zeros_pinned(
    shape: int | tuple[int, ...],
    dtype: xp.dtype = float,
    order: str = "C",
):
    """Returns a new array of given shape and type, filled with zeros.
    The array is allocated in pinned memory if using cupy.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array.
    dtype : data-type, optional
        The desired data-type for the array. Default is `float`.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in memory.
        Default is 'C'.

    Returns
    -------
    NDArray
        The array of zeros.

    .. seealso:: :func:`numpy.zeros` :func:`cupy.zeros` :func:`cupyx.zeros_pinned`

    """

    if xp.__name__ == "cupy":
        return cupyx.zeros_pinned(shape, dtype=dtype, order=order)
    else:
        return xp.zeros(shape, dtype=dtype, order=order)


@profiler.profile(level="debug")
def empty_like_pinned(
    a: NDArray,
    dtype: xp.dtype = None,
    order: str = "K",
    shape: int | tuple[int, ...] = None,
):
    """Returns a new array with the same shape and type as a given array.
    The array is allocated in pinned memory if using cupy.

    Parameters
    ----------
    a : NDArray
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', 'K'}, optional
        Overrides the memory layout of the
        result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means
        ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.
        ``'K'`` means match the layout of ``a`` as closely as possible.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.

    Returns
    -------
    NDArray
        The empty array.

    .. seealso:: :func:`numpy.empty_like` :func:`cupy.empty_like` :func:`cupyx.empty_like_pinned`

    """

    if xp.__name__ == "cupy":
        return cupyx.empty_like_pinned(a, dtype=dtype, order=order, shape=shape)
    else:
        return xp.empty_like(a, dtype=dtype, order=order, shape=shape)


@profiler.profile(level="debug")
def zeros_like_pinned(
    a: NDArray,
    dtype: xp.dtype = None,
    order: str = "K",
    shape: int | tuple[int, ...] = None,
):
    """Returns an array of zeros with the same shape and type as a given array.
    The array is allocated in pinned memory if using cupy.

    Parameters
    ----------
    a : NDArray
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', 'K'}, optional
        Overrides the memory layout of the
        result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means
        ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.
        ``'K'`` means match the layout of ``a`` as closely as possible.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.

    Returns
    -------
    NDArray
        The array of zeros.

    .. seealso:: :func:`numpy.zeros_like` :func:`cupy.zeros_like` :func:`cupyx.zeros_like_pinned`

    """

    if xp.__name__ == "cupy":
        return cupyx.zeros_like_pinned(a, dtype=dtype, order=order, shape=shape)
    else:
        return xp.zeros_like(a, dtype=dtype, order=order, shape=shape)


@profiler.profile(level="debug")
def synchronize_current_stream():
    """Synchronizes the current stream if using cupy.

    Does nothing if using numpy.

    """
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()


@profiler.profile(level="debug")
def synchronize_device():
    """Synchronizes the device if using cupy.

    Does nothing if using numpy.

    """
    if xp.__name__ == "cupy":
        xp.cuda.runtime.deviceSynchronize()


@profiler.profile(level="debug")
def get_nccl_communicator(mpi_comm: MPI.Comm = MPI.COMM_WORLD):
    """Returns the NCCL communicator if using cupy.

    Does nothing if using numpy.

    Parameters
    ----------
    mpi_comm : MPI.Comm
        The MPI communicator to use.


    Returns
    -------
    cupyx.distributed.nccl.NCCLBackend
        The NCCL communicator

    """
    if not xp.__name__ == "cupy":
        return None

    from cupy.cuda import nccl

    if not nccl.available:
        return None

    from cupyx import distributed

    # TODO: This will probably not work with communicators other than
    # MPI.COMM_WORLD. We need to fix this if we want to use other
    # communicators.
    return distributed.NCCLBackend(mpi_comm.size, mpi_comm.rank, use_mpi=True)
