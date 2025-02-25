# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import inspect

from qttools import NDArray, xp


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


def get_host(arr: NDArray) -> NDArray:
    """Returns the host array of the given array.

    Parameters
    ----------
    arr : NDArray
        The array to convert.

    Returns
    -------
    np.ndarray
        The equivalent numpy array.

    """
    if get_array_module_name(arr) == "numpy":
        return arr
    return arr.get()


def get_device(arr: NDArray) -> NDArray:
    """Returns the device array of the given array.

    Parameters
    ----------
    arr : NDArray
        The array to convert.

    Returns
    -------
    NDArray
        The equivalent cupy array.

    """
    if get_array_module_name(arr) == "cupy":
        return arr
    return xp.asarray(arr)


def synchronize_current_stream():
    """Synchronizes the current stream if using cupy.

    Does nothing if using numpy.

    """
    if xp.__name__ == "cupy":
        xp.cuda.get_current_stream().synchronize()


def get_cuda_devices(return_names: bool = False):
    """Returns the list of available CUDA devices."""
    if xp.__name__ != "cupy":
        return []
    num_devices = xp.cuda.runtime.getDeviceCount()
    if return_names:
        devices = [f"cuda:{i}" for i in range(num_devices)]
    else:
        devices = list(range(num_devices))
    return devices
