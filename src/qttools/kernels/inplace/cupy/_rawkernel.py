# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp
import numpy as np

_iadd_cuda_code = r"""
#include <cupy/complex.cuh>

template <typename T>
__device__ inline T maybe_conj(T x) { return x; }

template <>
__device__ inline complex<double> maybe_conj(complex<double> x) {
    return conj(x);
}

template<typename T1, typename T2, typename T3> 
__global__ void _iadd(T1* M, const T2* U,
            const long* ind, const int N, const T3 alpha, bool conjugate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        T2 u = U[i];
        if (conjugate) {
            u = maybe_conj<T2>(u);
        } else {
            u=U[i];
        }
        M[ind[i]] += u * (alpha);
    }
    }
"""

_iadd_kers = (
    "_iadd<double, double, double>",
    "_iadd<complex<double>, double, double>",
    "_iadd<complex<double>,complex<double>, double>",
    "_iadd<complex<double>, double, complex<double>>",
    "_iadd<complex<double>, complex<double>, complex<double>>",
)

_iadd = cp.RawModule(code=_iadd_cuda_code, name_expressions=_iadd_kers)

ker_dict = {
    (cp.float64, cp.float64, np.float64): _iadd.get_function(
        "_iadd<double, double, double>"
    ),
    (cp.complex128, cp.float64, np.float64): _iadd.get_function(
        "_iadd<complex<double>, double, double>"
    ),
    (cp.complex128, cp.complex128, np.float64): _iadd.get_function(
        "_iadd<complex<double>,complex<double>, double>"
    ),
    (cp.complex128, cp.float64, np.complex128): _iadd.get_function(
        "_iadd<complex<double>, double, complex<double>>"
    ),
    (cp.complex128, cp.complex128, np.complex128): _iadd.get_function(
        "_iadd<complex<double>, complex<double>, complex<double>>"
    ),
}

_iadd_obc = cp.RawKernel(
    r"""
#include <cupy/complex.cuh>
extern "C" __global__
void _iadd_obc(complex<double>* M, const complex<double>* S, const double K1, const double K2, const int N_S_big, const int N_S, const int n_rep_2,
            const long* ind, const int N, const complex<double> alpha) {

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
        long s_idx = (long)i_S * N_S + j_S;

        // Apply phase and add to system matrix
        M[ind[i]] += S[s_idx] * phase_factor * alpha;
    }
}
""",
    "_iadd_obc",
)
