# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from itertools import product

import cupy as cp

value_types = [
    (cp.float64, "double"),
    (cp.complex128, "complex<double>"),
]

index_types = [
    (cp.int32, "int"),
    (cp.uintp, "size_t"),
    (cp.int64, "long long"),
]

_scatter_add_scaled_cuda_code = r"""
#include <cupy/complex.cuh>

template <typename T> __device__ inline T make_conj(T x) { return x; }
template <> __device__ inline complex<double> make_conj(complex<double> x) { return conj(x); }
template <> __device__ inline complex<float> make_conj(complex<float> x) { return conj(x); }

template<typename T1, typename T2, typename T3, typename IndexType> 
__global__ void _scatter_add_scaled(
    T1* M, const T2* U, const IndexType* ind, const IndexType N, const T3 alpha, bool conjugate
) {
    // This upcasts before overflowing
    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    if (i < N) {
        T2 u = U[i];
        if (conjugate) {
            u = make_conj(u);
        }
        // ind has no duplicates, so we can safely add
        M[ind[i]] += u * alpha;
    }
}
"""

signatures = []
for t1, t2, t3, idx in product(value_types, value_types, value_types, index_types):

    # Skip when T1 is real and T2/T3 is complex, since this would result in a type mismatch.
    if (t1[0] in [cp.float64, cp.float32]) and (
        t2[0] in [cp.complex128, cp.complex64] or t3[0] in [cp.complex128, cp.complex64]
    ):
        continue

    signatures.append((t1, t2, t3, idx))

_scatter_add_scaled_functions = tuple(
    f"_scatter_add_scaled<{t1[1]},{t2[1]},{t3[1]},{idx[1]}>"
    for t1, t2, t3, idx in signatures
)

_scatter_add_scaled = cp.RawModule(
    code=_scatter_add_scaled_cuda_code,
    name_expressions=_scatter_add_scaled_functions,
    options=("-std=c++17",),
)

scatter_add_scaled_kernels = {
    (t1[0], t2[0], t3[0], idx[0]): _scatter_add_scaled.get_function(expr)
    for (t1, t2, t3, idx), expr in zip(signatures, _scatter_add_scaled_functions)
}

_scatter_add_scaled_obc_cuda_code = r"""
#include <cupy/complex.cuh>

template<typename IndexType> 
__global__ void _scatter_add_scaled_obc(
    complex<double>* M,
    const complex<double>* S,
    const double K1,
    const double K2,
    const IndexType N_S_big,
    const IndexType N_S,
    const IndexType n_rep_2,
    const IndexType* ind,
    const IndexType N,
    const double alpha
){


    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    IndexType i_S_big, j_S_big, i_cell, j_cell;
    IndexType i_S, j_S, cell_rep_1_i, cell_rep_1_j, cell_rep_2_i, cell_rep_2_j;

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
        IndexType s_idx = (IndexType)i_S * N_S + j_S;

        // Apply phase and add to system matrix
        M[ind[i]] += S[s_idx] * phase_factor * alpha;
    }
}
"""

_scatter_add_scaled_obc_functions = tuple(
    f"_scatter_add_scaled_obc<{idx[1]}>" for idx in index_types
)

_scatter_add_scaled_obc = cp.RawModule(
    code=_scatter_add_scaled_obc_cuda_code,
    name_expressions=_scatter_add_scaled_obc_functions,
    options=("-std=c++17",),
)

_scatter_add_scaled_obc_kernels = {
    idx[0]: _scatter_add_scaled_obc.get_function(expr)
    for idx, expr in zip(index_types, _scatter_add_scaled_obc_functions)
}


__all__ = ["scatter_add_scaled_kernels", "_scatter_add_scaled_obc_kernels"]
