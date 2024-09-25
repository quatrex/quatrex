# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import inspect
import os
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

# Allows user to specify the array module via an environment variable.
ARRAY_MODULE = os.environ.get("ARRAY_MODULE")
if ARRAY_MODULE is not None:
    if ARRAY_MODULE == "numpy":
        xp = np
    elif ARRAY_MODULE == "cupy":
        try:
            import cupy as xp
        except ImportError as e:
            warn(f"'cupy' is unavailable, defaulting to 'numpy'. ({e})")
            xp = np
    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{ARRAY_MODULE}'")
else:
    # If the user does not specify the array module, prioritize cupy but
    # default to numpy if cupy is not available or not working.
    try:
        import cupy as xp

        try:
            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)
        except Exception as e:
            warn(f"'cupy' is unavailable, defaulting to 'numpy'. ({e})")
            xp = np

    except ImportError as e:
        warn(f"'cupy' is unavailable, defaulting to 'numpy'. ({e})")
        xp = np


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


def get_device(arr: ArrayLike):
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
