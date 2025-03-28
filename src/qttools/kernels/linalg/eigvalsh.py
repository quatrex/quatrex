import numpy as np

from qttools import NDArray, xp
from qttools.kernels import linalg
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_any_location, get_array_module_name

profiler = Profiler()


@profiler.profile(level="debug")
def _eigvalsh(
    A: NDArray,
    B: NDArray,
    xp_eigvalsh: xp,
) -> NDArray:
    """Compute eigenvalues of a hermitain generalized eigenvalue problem.

    Parameters
    ----------
    A : NDArray
        A complex or real symmetric or hermitian matrix.
    B : NDArray
        A complex or real symmetric or hermitian positive definite matrix.
    xp_eigvalsh : module
        The location where to compute the eigenvalues.

    Returns
    -------
    NDArray
        The eigenvalues of the generalized eigenvalue problem.

    """

    R = xp_eigvalsh.linalg.cholesky(B)

    # NOTE: would be more efficient to use cholesky_solve
    # if it would be supported by cupy
    if xp_eigvalsh.__name__ == "cupy":
        R_inv = linalg.inv(R)
    else:
        R_inv = xp_eigvalsh.linalg.inv(R)
    A_hat = R_inv @ A @ R_inv.swapaxes(-2, -1).conj()
    w = xp_eigvalsh.linalg.eigvalsh(A_hat)

    return w


@profiler.profile(level="api")
def eigvalsh(
    A: NDArray,
    B: NDArray | None = None,
    compute_module: str = "cupy",
    output_module: str | None = None,
    use_pinned_memory: bool = True,
) -> NDArray:
    """Compute eigenvalues of a hermitain generalized eigenvalue problem.

    TODO: only type 1 generalized problems are supported,
    Only a subset of keywords of scipy.linalg.eigvalsh are supported.

    Parameters
    ----------
    A : NDArray
        A complex or real symmetric or hermitian matrix.
    B : NDArray, optional
        A complex or real symmetric or hermitian positive definite matrix.
        If omitted, identity matrix is assumed.
    compute_module : str, optional
        The location where to compute the eigenvalues.
    output_module : str, optional
        The location where to store the resulting eigenvalues.
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

    Returns
    -------
    NDArray
        The eigenvalues of the generalized eigenvalue problem.

    """
    input_module = get_array_module_name(A)

    if B is not None and input_module != get_array_module_name(B):
        raise ValueError("Input arrays must be at the same location.")

    if output_module is None:
        output_module = input_module

    if xp.__name__ == "numpy" and (
        compute_module == "cupy" or output_module == "cupy" or input_module == "cupy"
    ):
        raise ValueError("Cannot do gpu computation with numpy as xp.")

    # memcopy to correct location
    A = get_any_location(A, compute_module, use_pinned_memory=use_pinned_memory)

    if B is None:
        if compute_module == "numpy":
            w = np.linalg.eigvalsh(A)
        elif compute_module == "cupy":
            w = xp.linalg.eigvalsh(A)
    else:

        B = get_any_location(B, compute_module, use_pinned_memory=use_pinned_memory)

        if compute_module == "numpy":
            w = _eigvalsh(A, B, np)
        elif compute_module == "cupy":
            w = _eigvalsh(A, B, xp)

    return get_any_location(w, output_module, use_pinned_memory=use_pinned_memory)
