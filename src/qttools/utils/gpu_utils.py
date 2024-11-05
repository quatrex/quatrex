# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import inspect

import numpy as np
from numpy.typing import ArrayLike

from qttools import xp


def get_array_module_name(arr: ArrayLike) -> str:
    """Given an array, returns the array's module name.

    This works for numpy even when cupy is not available.

    Parameters
    ----------
    arr : ArrayLike
        The array to check.

    Returns
    -------
    str
        The array module name used by the array.

    """
    submodule = inspect.getmodule(type(arr))
    return submodule.__name__.split(".")[0]


def get_host(arr: ArrayLike) -> np.ndarray:
    """Returns the host array of the given array.

    Parameters
    ----------
    arr : ArrayLike
        The array to convert.

    Returns
    -------
    np.ndarray
        The equivalent numpy array.

    """
    if get_array_module_name(arr) == "numpy":
        return arr
    return arr.get()


def get_device(arr: ArrayLike) -> xp.ndarray:
    """Returns the device array of the given array.

    Parameters
    ----------
    arr : ArrayLike
        The array to convert.

    Returns
    -------
    ArrayLike
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
