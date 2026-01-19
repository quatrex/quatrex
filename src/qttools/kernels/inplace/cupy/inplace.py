# Copyright (c) 2026 ETH Zurich and the authors of the qttools package.
import cupy as cp
from pygments.unistring import Nd

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
    key1: float,
    key2: float,
    nrep1: int,
    nrep2: int,
):
    # TODO: figure out names
    """Adds array `b` to array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added to `a`.
    inds : NDArray
        The indices at which to add `b` to `a`.
    key1 : float
        The first OBC key.
    key2 : float
        The second OBC key.
    nrep1 : int
        The number of repetitions in the first direction.
    nrep2 : int
        The number of repetitions in the second direction.

    """

    num_inds = inds.shape[0]

    # Launch kernel
    blocks_per_grid = (num_inds + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

    _rawkernel._iadd_obc(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            a,
            b.flatten(),
            key1,
            key2,
            b.shape[1] * nrep1 * nrep2,
            b.shape[1],
            nrep2,
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


def isub_obc(a, b, inds, key1, key2, nrep1, nrep2):
    """Subtracts array `b` from array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be subtracted from `a`.
    inds : NDArray
        The indices at which to subtract `b` from `a`.
    key1 : float
        The first OBC key.
    key2 : float
        The second OBC key.
    nrep1 : int
        The number of repetitions in the first direction.
    nrep2 : int
        The number of repetitions in the second direction.

    """

    num_inds = inds.shape[0]

    # Launch kernel
    blocks_per_grid = (num_inds + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

    _rawkernel._isub_obc(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            a,
            b.flatten(),
            key1,
            key2,
            b.shape[0] * nrep1 * nrep2,
            b.shape[0],
            nrep2,
            inds,
            num_inds,
        ),
    )
