# Copyright (c) 2026 ETH Zurich and the authors of the qttools package.
import cupy as cp

from qttools import NDArray
from qttools.kernels.inplace.cupy import _rawkernel

THREADS_PER_BLOCK = 1024


def iadd(a: NDArray, b: NDArray, inds: NDArray) -> None:
    """Adds array `b` to array `a` at indices `inds` in-place.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added to `a`.
    inds : NDArray
        The indices at which to add `b` to `a`.

    """

    num_inds = inds.shape[0]
    blocks_per_grid = (num_inds + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    if a.dtype != cp.complex128:
        raise ValueError("In-place addition kernel requires a to be complex128.")
    if b.dtype == cp.complex128:
        _rawkernel._iadd_comp(
            (blocks_per_grid,), (THREADS_PER_BLOCK,), (a, b, inds, num_inds)
        )
    elif b.dtype == cp.float64:
        _rawkernel._iadd_real(
            (blocks_per_grid,), (THREADS_PER_BLOCK,), (a, b, inds, num_inds)
        )
    else:
        raise ValueError("Unsupported dtype for b in in-place addition.")


def iadd_obc(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    k: tuple[float, float],
    transverse_repetition_grid: tuple[int, int],
):
    """Adds array `b` to array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be subtracted from `a`.
    inds : NDArray
        The indices at which to subtract `b` from `a`.
    k : tuple[float, float]
        The transverse wavevector components.
    transverse_repetition_grid : tuple[int, int]
        The transverse repetition grid of the contact.

    """

    num_inds = inds.shape[0]

    ky, kz = k
    ny, nz = transverse_repetition_grid

    # Launch kernel
    blocks_per_grid = (num_inds + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

    _rawkernel._iadd_obc(
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
        ),
    )


def isub(a: NDArray, b: NDArray, inds: NDArray) -> None:
    """Subtracts array `b` from array `a` at indices `inds` in-place.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be subtracted from `a`.
    inds : NDArray
        The indices at which to subtract `b` from `a`.

    """
    num_inds = inds.shape[0]
    blocks_per_grid = (num_inds + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    if a.dtype != cp.complex128:
        raise ValueError("In-place subtraction kernel requires `a` to be complex128.")
    if b.dtype == cp.complex128:
        _rawkernel._isub_comp(
            (blocks_per_grid,), (THREADS_PER_BLOCK,), (a, b, inds, num_inds)
        )
    elif b.dtype == cp.float64:
        _rawkernel._isub_real(
            (blocks_per_grid,), (THREADS_PER_BLOCK,), (a, b, inds, num_inds)
        )
    else:
        raise ValueError("Unsupported dtype for `b` in in-place subtraction.")


def isub_obc(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    k: tuple[float, float],
    transverse_repetition_grid: tuple[int, int],
):
    """Subtracts array `b` from array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be subtracted from `a`.
    inds : NDArray
        The indices at which to subtract `b` from `a`.
    k : tuple[float, float]
        The transverse wavevector components.
    transverse_repetition_grid : tuple[int, int]
        The transverse repetition grid of the contact.

    """

    num_inds = inds.shape[0]

    ky, kz = k
    ny, nz = transverse_repetition_grid

    # Launch kernel
    blocks_per_grid = (num_inds + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

    _rawkernel._isub_obc(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            a,
            b.flatten(),
            ky,
            kz,
            b.shape[0] * ny * nz,
            b.shape[0],
            nz,
            inds,
            num_inds,
        ),
    )
