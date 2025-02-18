# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_device, get_host


def eig(A: NDArray, compute_location: str = "host") -> tuple[NDArray, NDArray]:
    """Computes the eigenvalues and eigenvectors of a matrix on a given location.

    The variable compute_location has no effect if numpy is used.
    To compute the eigenvalues and eigenvectors on the device with cupy
    is only possible if the cupy.linalg.eig function is available.

    Parameters
    ----------
    A : NDArray
        The matrix.
    compute_location : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "device" or "host".

    Returns
    -------
    NDArray
        The eigenvalues.
    NDArray
        The eigenvectors.

    """

    if (
        compute_location == "device"
        and xp.__name__ == "cupy"
        and hasattr(xp.linalg, "eig")
    ):
        w, v = xp.linalg.eig(A)
    elif compute_location == "host":
        A = get_host(A)
        w, v = np.linalg.eig(A)
        w, v = get_device(w), get_device(v)
    else:
        raise ValueError(f"Invalid compute location: {compute_location}")

    return w, v
