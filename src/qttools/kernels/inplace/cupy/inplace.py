# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp

from qttools import NDArray
from qttools.kernels.inplace.cupy import _rawkernel

THREADS_PER_BLOCK = 1024


def scatter_add_scaled(
    a: NDArray, b: NDArray, inds: NDArray, alpha: float = 1.0, conjugate: bool = False
) -> None:
    """Adds array `b` to array `a` at indices `inds` in-place.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added to `a`.
    inds : NDArray
        The indices at which to add `b` to `a`.
    alpha : complex, optional
        The scalar multiplier for `b` before adding it to `a`.
    conjugate : bool, optional
        Whether to take the complex conjugate of `b` before adding it to
        `a`.

    """
    num_inds = inds.shape[0]
    blocks_per_grid = (num_inds + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    ker = _rawkernel.ker_dict[(a.dtype.type, b.dtype.type, alpha.dtype.type)]
    ker(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (a, b, inds, num_inds, alpha, conjugate),
    )


def scatter_add_scaled_obc(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    k: tuple[float, float],
    transverse_repetition_grid: tuple[int, int],
    alpha: float = 1.0,
):
    """Adds array `b` to array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added to `a`.
    inds : NDArray
        The indices at which to add `b` to `a`.
    k : tuple[float, float]
        The transverse wavevector components.
    transverse_repetition_grid : tuple[int, int]
        The transverse repetition grid of the contact.
    alpha : float
        The scalar multiplier for `b` before adding it to `a`.

    """

    num_inds = inds.shape[0]

    ky, kz = k
    ny, nz = transverse_repetition_grid

    # Launch kernel
    blocks_per_grid = (num_inds + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

    alpha = cp.float64(alpha)

    _rawkernel._scatter_add_scaled_obc(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            a,
            b.flatten(),
            ky,
            kz,
            b.shape[1] * ny * nz,
            b.shape[1],
            nz,
            inds,
            num_inds,
            alpha,
        ),
    )
