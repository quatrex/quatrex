# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import math

from numba import njit, prange


@njit(parallel=True)
def iadd_OBC_CPU(M, S, K1, K2, N_S_big, N_S, n_rep_2, ind, N):
    for i in prange(N):
        i_S_big = i // N_S_big
        j_S_big = i % N_S_big

        i_S = i_S_big % N_S
        j_S = j_S_big % N_S

        i_cell = i_S_big // N_S
        j_cell = j_S_big // N_S

        cell_rep_1_i = i_cell // n_rep_2
        cell_rep_2_i = i_cell % n_rep_2

        cell_rep_1_j = j_cell // n_rep_2
        cell_rep_2_j = j_cell % n_rep_2

        phase_1 = -K1 * (cell_rep_1_j - cell_rep_1_i)
        phase_2 = -K2 * (cell_rep_2_j - cell_rep_2_i)
        total_phase = phase_1 + phase_2

        c = math.cos(total_phase)
        s = math.sin(total_phase)
        s_idx = i_S * N_S + j_S

        # Potential race if ind has duplicates.
        M[ind[i]] += S[s_idx] * (c + 1j * s)
