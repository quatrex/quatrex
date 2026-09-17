// Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
#include <cupy/complex.cuh>

template <typename T> __device__ inline T make_conj(T x) { return x; }
template <> __device__ inline complex<double> make_conj(complex<double> x) {
  return conj(x);
}
template <> __device__ inline complex<float> make_conj(complex<float> x) {
  return conj(x);
}

template <typename T1, typename T2, typename T3, typename IndexType>
__global__ void _scatter_add_scaled(T1 *M, const T2 *U, const IndexType *ind,
                                    const IndexType N, const T3 alpha,
                                    bool conjugate) {
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

template <typename IndexType>
__global__ void
_scatter_add_scaled_obc(complex<double> *M, const complex<double> *S,
                        const double K1, const double K2,
                        const IndexType N_S_big, const IndexType N_S,
                        const IndexType n_rep_2, const IndexType *ind,
                        const IndexType N, const double alpha) {

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
