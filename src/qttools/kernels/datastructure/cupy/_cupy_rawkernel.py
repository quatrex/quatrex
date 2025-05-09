# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import cupy as cp
import numpy as np

from qttools import NDArray
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK

_reduction = cp.RawKernel(
    r"""
    extern "C" __global__
    void _reduction(
        int *a,
        int *out,
        int n
    ){
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        int tmp = 0;
        for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
            tmp += a[i];
        }
        if(tid < blockDim.x * gridDim.x){
            out[tid] = tmp;
        }

    }
""",
    "_reduction",
)


def reduction(
    a: NDArray,
):

    n_blocks = 4

    out = cp.zeros((n_blocks * THREADS_PER_BLOCK), dtype=cp.int32)

    n = a.size
    _reduction(
        (n_blocks,),
        (THREADS_PER_BLOCK,),
        (
            a,
            out,
            np.int32(n),
        ),
    )

    out = cp.sum(out)

    return out


_find_inds = cp.RawKernel(
    f"""
        extern "C" __global__
        void find_inds(
            int* self_rows,
            int* self_cols,
            int* rows,
            int* cols,
            int* full_inds,
            short* counts,
            int num_self_rows,
            int num_rows
        ) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;
            __shared__ int cache_rows[{THREADS_PER_BLOCK}];
            __shared__ int cache_cols[{THREADS_PER_BLOCK}];
                                         

            int my_row = (i < num_self_rows) ? self_rows[i] : -1;
            int my_col = (i < num_self_rows) ? self_cols[i] : -1;
                                
            int my_full_ind = 0;
            int my_count = 0;
                                        
            for (int j = 0; j < num_rows; j += {THREADS_PER_BLOCK}) {{
                if (j + tid < num_rows) {{
                    cache_rows[tid] = rows[j + tid];
                    cache_cols[tid] = cols[j + tid];
                }}
                __syncthreads();
                                         
                for (int idx = j; idx < min(j + {THREADS_PER_BLOCK}, num_rows); idx++) {{
                    int cond = (my_row == cache_rows[idx - j]) & (my_col == cache_cols[idx - j]);
                    my_full_ind = my_full_ind * (1 - cond) + idx * cond;
                    my_count += cond;
                }}
                __syncthreads();
            }}
                                         
            if (i < num_self_rows) {{
                full_inds[i] = my_full_ind;
                counts[i] = my_count;
            }}
        }}
    """,
    "find_inds",
)


_compute_coo_block_mask = cp.RawKernel(
    r"""
        extern "C" __global__
        void _compute_coo_block_mask_kernel(
            int *rows,
            int *cols,
            int row_start,
            int row_stop,
            int col_start,
            int col_stop,
            int *mask,
            int rows_len
        ){
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < rows_len) {
                mask[tid] = (
                    (rows[tid] >= row_start)
                    && (rows[tid] < row_stop)
                    && (cols[tid] >= col_start)
                    && (cols[tid] < col_stop)
                );
            }
        }
    """,
    "_compute_coo_block_mask_kernel",
)


_densify_block = cp.RawKernel(
    r"""
        #include <cupy/complex.cuh>
        extern "C" __global__
        void densify_block(
            complex<double>* block,
            int* rows,
            int* cols,
            complex<double>* data,
            int stack_size,
            int stack_stride,
            int nnz_per_block,
            int num_rows,
            int num_cols,
            int block_start,
            int row_offset,
            int col_offset
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int nnz_total = stack_size * nnz_per_block;

            if (i < nnz_total) {
                int stack_idx = i / nnz_per_block;
                int stack_start = stack_idx * stack_stride;
                int nnz_idx = i % nnz_per_block + block_start;
                int block_size = num_rows * num_cols;

                int row = rows[nnz_idx];
                int col = cols[nnz_idx];

                // printf("row: %d, col: %d, nnz_idx: %d, stack_idx: %d, stack_start: %d, block_start: %d, row_offset: %d, col_offset: %d\n", row, col, nnz_idx, stack_idx, stack_start, block_start, row_offset, col_offset);

                block[stack_idx * block_size + (row - row_offset) * num_cols + (col - col_offset)] = data[stack_start + nnz_idx];
            } 
        }
    """,
    "densify_block",
)


_find_bcoords = cp.RawKernel(
    r"""
        extern "C" __global__
        void _find_bcoords_kernel(
            int *block_offsets,
            int *rows,
            int *cols,
            int *brows,
            int *bcols,
            int rows_len,
            int block_offsets_len
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < rows_len) {
                for (int j = 0; j < block_offsets_len; j++) {
                    int cond_rows = block_offsets[j] <= rows[tid];
                    brows[tid] = brows[tid] * (1 - cond_rows) + j * cond_rows;
                    int cond_cols = block_offsets[j] <= cols[tid];
                    bcols[tid] = bcols[tid] * (1 - cond_cols) + j * cond_cols;
                }
            }
        }
    """,
    "_find_bcoords_kernel",
)

_compute_block_mask = cp.RawKernel(
    r"""
        extern "C" __global__
        void _compute_block_mask_kernel(
            int *brows,
            int *bcols,
            int brow,
            int bcol,
            bool *mask,
            int brows_len
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < brows_len) {
                mask[tid] = (brows[tid] == brow) & (bcols[tid] == bcol);
            }
        }
    """,
    "_compute_block_mask_kernel",
)


_compute_block_inds = cp.RawKernel(
    r"""
        extern "C" __global__
        void _compute_block_inds_kernel(
            int *rr,
            int *cc,
            int *self_cols,
            int *rowptr,
            int *block_inds,
            int rr_len
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < rr_len) {
                int r = rr[tid];
                int ind = -1;
                for (int j = rowptr[r]; j < rowptr[r + 1]; j++) {
                    int cond = self_cols[j] == cc[tid];
                    ind = ind * (1 - cond) + j * cond;
                }
                block_inds[tid] = ind;
            }
        }
    """,
    "_compute_block_inds_kernel",
)

_expand_rows = cp.RawKernel(
    r"""
    extern "C" __global__
    void _expand_rows_kernel(
        int *rows,
        int *rowptr,
        int rowptr_len
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < rowptr_len - 1) {
            for (int j = rowptr[tid]; j < rowptr[tid + 1]; j++) {
                rows[j] = tid;
            }
        }
    }
    """,
    "_expand_rows_kernel",
)


_find_ranks = cp.RawKernel(
    r"""
        extern "C" __global__
        void _find_ranks_kernel(
            int *nnz_section_offsets,
            int *inds,
            short *ranks,
            int nnz_section_offsets_len,
            int inds_len
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < inds_len) {
                for (int j = 0; j < nnz_section_offsets_len; j++) {
                    int cond = nnz_section_offsets[j] <= inds[tid];
                    ranks[tid] = ranks[tid] * (1 - cond) + j * cond;
                }
            }
        }
    """,
    "_find_ranks_kernel",
)
