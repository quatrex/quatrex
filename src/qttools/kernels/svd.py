# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_array_module_name, get_device, get_host


def svd(
    A: NDArray,
    full_matrices: bool = True,
    compute_uv: bool = True,
    compute_module: str = "numpy",
    output_module: str | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Computes the singular value decomposition of a matrix on a given location.

    Parameters
    ----------
    A : NDArray
        The matrix.
    compute_module : str, optional
        The location where to compute the singular value decomposition.
        Can be either "numpy" or "cupy".
    output_module : str, optional
        The location where to store the singular value decomposition.
        Can be either "numpy"
        or "cupy". If None, the output location is the same as the input location

    Returns
    -------
    NDArray
        The left singular vectors.
    NDArray
        The singular values.
    NDArray
        The right singular vectors.

    """
    input_module = get_array_module_name(A)

    if output_module is None:
        output_module = input_module

    if output_module not in ["numpy", "cupy"]:
        raise ValueError(f"Invalid output location: {output_module}")
    if compute_module not in ["numpy", "cupy"]:
        raise ValueError(f"Invalid compute location: {compute_module}")
    if input_module not in ["numpy", "cupy"]:
        raise ValueError(f"Invalid input location: {input_module}")

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    # memcopy to correct location
    if compute_module == "numpy" and input_module == "cupy":
        A = get_host(A)
    elif compute_module == "cupy" and input_module == "numpy":
        A = get_device(A)

    if compute_module == "cupy":
        u, s, vh = xp.linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
    elif compute_module == "numpy":
        u, s, vh = np.linalg.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)

    if output_module == "numpy" and compute_module == "cupy":
        u, s, vh = get_host(u), get_host(s), get_host(vh)
    elif output_module == "cupy" and compute_module == "numpy":
        u, s, vh = get_device(u), get_device(s), get_device(vh)

    return u, s, vh
