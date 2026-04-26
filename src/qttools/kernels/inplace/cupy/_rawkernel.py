# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp

_iadd_comp = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void _iadd_comp(complex<double>* M, const complex<double>* U,
                const long long* ind, const int N, const complex<double> alpha) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N) {
            M[ind[i]] += alpha * U[i];
        }
    }
    """,
    "_iadd_comp",
)

_iadd_real = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void _iadd_real(complex<double>* M, const double* U,
                const long long* ind, const int N, const complex<double> alpha) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N) {
        M[ind[i]] += alpha * U[i];
        }
    }
    """,
    "_iadd_real",
)

_iadd_obc = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>

    extern "C" __global__
    void _iadd_obc(complex<double>* M, const complex<double>* S, const double K1, const double K2, const int N_S_big, const int N_S, const int n_rep_2,
                const long long* ind, const int N) {

        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int i_S_big, j_S_big, i_cell, j_cell;
        int i_S, j_S, cell_rep_1_i, cell_rep_1_j, cell_rep_2_i, cell_rep_2_j;

        complex<double> im = complex<double>(0.0, 1.0);

        if (i < N) {

            i_S_big = i / N_S_big;
            j_S_big = i % N_S_big;

            i_S = i_S_big % N_S;
            j_S = j_S_big % N_S;

            i_cell = i_S_big / N_S;
            j_cell = j_S_big / N_S;

            cell_rep_1_i = i_cell / n_rep_2;
            cell_rep_2_i = i_cell % n_rep_2;

            cell_rep_1_j = j_cell / n_rep_2;
            cell_rep_2_j = j_cell % n_rep_2;

            // Compute phase factors
            double phase_1 = -K1 * (cell_rep_1_j - cell_rep_1_i);
            double phase_2 = -K2 * (cell_rep_2_j - cell_rep_2_i);

            double total_phase = phase_1 + phase_2;

            complex<double> phase_factor(cos(total_phase), sin(total_phase));

            // Get value from base self-energy matrix
            long long s_idx = (long long)i_S * N_S + j_S;

            // Apply phase and add to system matrix
            M[ind[i]] += S[s_idx] * phase_factor;
        }
    }
    """,
    "_iadd_obc",
)

_isub_comp = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void _isub_comp(complex<double>* M, const complex<double>* U,
                const long long* ind, const int N, const complex<double> alpha) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N) {
        M[ind[i]] -= alpha * U[i];
        }
    }
    """,
    "_isub_comp",
)

_isub_real = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void _isub_real(complex<double>* M, const double* U,
                const long long* ind, const int N, const complex<double> alpha) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N) {
        M[ind[i]] -= alpha * U[i];
        }
    }
    """,
    "_isub_real",
)

_isub_obc = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>

    extern "C" __global__
    void _isub_obc(complex<double>* M, const complex<double>* S, const double K1, const double K2, const int N_S_big, const int N_S, const int n_rep_2,
                const long long* ind, const int N) {

        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int i_S_big, j_S_big, i_cell, j_cell;
        int i_S, j_S, cell_rep_1_i, cell_rep_1_j, cell_rep_2_i, cell_rep_2_j;

        complex<double> im = complex<double>(0.0, 1.0);

        if (i < N) {

            i_S_big = i / N_S_big;
            j_S_big = i % N_S_big;

            i_S = i_S_big % N_S;
            j_S = j_S_big % N_S;

            i_cell = i_S_big / N_S;
            j_cell = j_S_big / N_S;

            cell_rep_1_i = i_cell / n_rep_2;
            cell_rep_2_i = i_cell % n_rep_2;

            cell_rep_1_j = j_cell / n_rep_2;
            cell_rep_2_j = j_cell % n_rep_2;

            // Compute phase factors
            double phase_1 = -K1 * (cell_rep_1_j - cell_rep_1_i);
            double phase_2 = -K2 * (cell_rep_2_j - cell_rep_2_i);

            double total_phase = phase_1 + phase_2;

            complex<double> phase_factor(cos(total_phase), sin(total_phase));

            // Get value from base self-energy matrix
            long long s_idx = (long long)i_S * N_S + j_S;

            // Apply phase and add to system matrix
            M[ind[i]] -= S[s_idx] * phase_factor;
        }
    }
    """,
    "_isub_obc",
)
