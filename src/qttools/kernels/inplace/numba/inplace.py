# Copyright (c) 2026 ETH Zurich and the authors of the qttools package.
import math

import numba as nb
import numpy as np

from qttools import NDArray


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
    a[inds] += b


@nb.njit(parallel=True)
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

    N_S = b.shape[1]
    N_S_big = N_S * nrep1 * nrep2
    num_inds = inds.shape[0]

    b = b.reshape(-1)

    for i in nb.prange(num_inds):
        i_S_big = i // N_S_big
        j_S_big = i % N_S_big

        i_S = i_S_big % N_S
        j_S = j_S_big % N_S

        i_cell = i_S_big // N_S
        j_cell = j_S_big // N_S

        cell_rep_1_i = i_cell // nrep2
        cell_rep_2_i = i_cell % nrep2

        cell_rep_1_j = j_cell // nrep2
        cell_rep_2_j = j_cell % nrep2

        phase_1 = -key1 * (cell_rep_1_j - cell_rep_1_i)
        phase_2 = -key2 * (cell_rep_2_j - cell_rep_2_i)
        total_phase = phase_1 + phase_2

        c = math.cos(total_phase)
        s = math.sin(total_phase)
        s_idx = i_S * N_S + j_S

        # Potential race if ind has duplicates.
        a[inds[i]] += b[s_idx] * (c + 1j * s)


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
    a[inds] -= b


@nb.njit(parallel=True)
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
    N_S = b.shape[1]
    N_S_big = N_S * nrep1 * nrep2
    num_inds = inds.shape[0]
    b = b.reshape(-1)

    for i in nb.prange(num_inds):
        i_S_big = i // N_S_big
        j_S_big = i % N_S_big

        i_S = i_S_big % N_S
        j_S = j_S_big % N_S

        i_cell = i_S_big // N_S
        j_cell = j_S_big // N_S

        cell_rep_1_i = i_cell // nrep2
        cell_rep_2_i = i_cell % nrep2

        cell_rep_1_j = j_cell // nrep2
        cell_rep_2_j = j_cell % nrep2

        phase_1 = -key1 * (cell_rep_1_j - cell_rep_1_i)
        phase_2 = -key2 * (cell_rep_2_j - cell_rep_2_i)
        total_phase = phase_1 + phase_2

        c = math.cos(total_phase)
        s = math.sin(total_phase)
        s_idx = i_S * N_S + j_S

        # Potential race if ind has duplicates.
        a[inds[i]] -= b[s_idx] * (c + 1j * s)
