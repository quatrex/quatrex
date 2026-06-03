# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import math

import numba as nb
import numpy as np

from qttools import NDArray


@nb.njit(parallel=True)
def scatter_add_scaled(
    a: NDArray, b: NDArray, inds: NDArray, alpha: np.complex128, conjugate: bool
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

    """
    for i in nb.prange(inds.shape[0]):
        if conjugate:
            a[inds[i]] += alpha * np.conj(b[i])
        else:
            a[inds[i]] += alpha * b[i]


@nb.njit(parallel=True)
def scatter_add_scaled_obc(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    k: tuple[float, float],
    transverse_repetition_grid: tuple[int, int],
    alpha: np.complex128,
):
    """Adds array `b` to array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added from `a`.
    inds : NDArray
        The indices at which to subtract `b` from `a`. The indices are
        assumed unique. If there are duplicates, the behavior is
        undefined due to potential race conditions.
    k : tuple[float, float]
        The transverse wavevector components.
    transverse_repetition_grid : tuple[int, int]
        The transverse repetition grid of the contact.
    alpha : np.complex128
        The scaling factor.

    """
    ky, kz = k
    ny, nz = transverse_repetition_grid

    N_S = b.shape[1]
    N_S_big = N_S * ny * nz
    num_inds = inds.shape[0]

    b = b.reshape(-1)

    for i in nb.prange(num_inds):
        i_S_big = i // N_S_big
        j_S_big = i % N_S_big

        i_S = i_S_big % N_S
        j_S = j_S_big % N_S

        i_cell = i_S_big // N_S
        j_cell = j_S_big // N_S

        cell_rep_1_i = i_cell // nz
        cell_rep_2_i = i_cell % nz

        cell_rep_1_j = j_cell // nz
        cell_rep_2_j = j_cell % nz

        phase_1 = -ky * (cell_rep_1_j - cell_rep_1_i)
        phase_2 = -kz * (cell_rep_2_j - cell_rep_2_i)
        total_phase = phase_1 + phase_2

        c = math.cos(total_phase)
        s = math.sin(total_phase)
        s_idx = i_S * N_S + j_S

        # Potential race if ind has duplicates.
        a[inds[i]] += b[s_idx] * (c + 1j * s) * alpha
