# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import xp

if xp.__name__ == "cupy":
    from qttools.kernels.inplace.cupy import THREADS_PER_BLOCK
    from qttools.kernels.inplace.cupy.inplace_add import (
        add_kernel_comp,
        add_kernel_real,
        add_OBC_inplace,
    )
    from qttools.kernels.inplace.cupy.inplace_sub import (
        sub_kernel_comp,
        sub_kernel_real,
        sub_OBC_inplace,
    )

elif xp.__name__ == "numpy":
    from qttools.kernels.inplace.numba.inplace_add import add_OBC_inplace_CPU
    from qttools.kernels.inplace.numba.inplace_sub import sub_OBC_inplace_CPU


def add_inplace(M, U, ind):
    """In-place addition of U to M at indices ind.

    Parameters
    ----------
    M : xp.ndarray
        The array to be updated in-place.
    U : xp.ndarray
        The array to be added to M.
    ind : xp.ndarray
        The indices at which to add U to M.
    """

    if xp.__name__ == "cupy":
        N = U.shape[0]
        blocks_per_grid = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        if M.dtype != xp.complex128:
            raise ValueError("In-place addition kernel requires M to be complex128.")
        if U.dtype == xp.complex128:
            add_kernel_comp((blocks_per_grid,), (THREADS_PER_BLOCK,), (M, U, ind, N))
        elif U.dtype == xp.float64:
            add_kernel_real((blocks_per_grid,), (THREADS_PER_BLOCK,), (M, U, ind, N))
        else:
            raise ValueError("Unsupported dtype for U in in-place addition.")

    else:
        M[ind] += U


def add_inplace_OBC(M, U, ind, key1, key2, nrep1, nrep2):

    if xp.__name__ == "cupy":
        N_S = U.shape[1]
        N_S_big = N_S * nrep1 * nrep2

        N = ind.shape[0]

        # Launch kernel
        blocks_per_grid = (N + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

        add_OBC_inplace(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                M,
                U.flatten(),
                key1,
                key2,
                N_S_big,
                N_S,
                nrep2,
                ind,
                N,
            ),
        )

    elif xp.__name__ == "numpy":
        N_S = U.shape[1]
        N_S_big = N_S * nrep1 * nrep2
        N = ind.shape[0]
        add_OBC_inplace_CPU(
            M,
            U.flatten(),
            key1,
            key2,
            N_S_big,
            N_S,
            nrep2,
            ind,
            N,
        )


def sub_inplace(M, U, ind):
    """In-place subtraction of U from M at indices ind.

    Parameters
    ----------
    M : xp.ndarray
        The array to be updated in-place.
    U : xp.ndarray
        The array to be subtracted from M.
    ind : xp.ndarray
        The indices at which to subtract U from M.
    """

    if xp.__name__ == "cupy":
        N = U.shape[0]
        blocks_per_grid = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        if M.dtype != xp.complex128:
            raise ValueError("In-place subtraction kernel requires M to be complex128.")
        if U.dtype == xp.complex128:
            sub_kernel_comp((blocks_per_grid,), (THREADS_PER_BLOCK,), (M, U, ind, N))
        elif U.dtype == xp.float64:
            sub_kernel_real((blocks_per_grid,), (THREADS_PER_BLOCK,), (M, U, ind, N))
        else:
            raise ValueError("Unsupported dtype for U in in-place subtraction.")

    else:
        M[ind] -= U


def sub_inplace_OBC(M, U, ind, key1, key2, nrep1, nrep2):

    if xp.__name__ == "cupy":

        N_S = U.shape[0]
        N_S_big = N_S * nrep1 * nrep2

        N = ind.shape[0]

        # Launch kernel
        blocks_per_grid = (N + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

        sub_OBC_inplace(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                M,
                U.flatten(),
                key1,
                key2,
                N_S_big,
                N_S,
                nrep2,
                ind,
                N,
            ),
        )

    else:
        N_S = U.shape[1]
        N_S_big = N_S * nrep1 * nrep2
        N = ind.shape[0]
        sub_OBC_inplace_CPU(
            M,
            U.flatten(),
            key1,
            key2,
            N_S_big,
            N_S,
            nrep2,
            ind,
            N,
        )
