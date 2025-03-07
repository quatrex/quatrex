# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from typing import Callable

from qttools import xp


def decorate_methods(
    decorator: Callable,
    exclude: list[str] | None = None,
) -> Callable:
    """Apply a decorator to multiple methods of a class.

    Parameters
    ----------
    decorator : Callable
        The decorator to apply to the methods.
    exclude : list[str] | None, optional
        A list of method names to exclude from decoration. By default
        all methods are decorated.

    Returns
    -------
    Callable
        A class decorator that applies the decorator to all methods of the class.

    """
    if exclude is None:
        exclude = []

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr not in exclude:
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def get_cuda_devices(return_names: bool = False):
    """Returns the list of available CUDA devices.

    Parameters
    ----------
    return_names
        If the device names should be written out.

    Returns
    ----------
    list
        List of available devices
    """
    if xp.__name__ != "cupy":
        return []
    num_devices = xp.cuda.runtime.getDeviceCount()
    if return_names:
        return [f"cuda:{i}" for i in range(num_devices)]

    return list(range(num_devices))
