// Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
#include <cupy/complex.cuh>

#define THREADS_PER_BLOCK TEMPLATE_THREADS_PER_BLOCK

template <typename IndexType1, typename IndexType2>
__global__ void _reduction(IndexType1 *a, IndexType2 *out, IndexType2 n) {
  IndexType2 global_block_offset = (IndexType2)blockDim.x * blockIdx.x;
  IndexType2 tid = global_block_offset + threadIdx.x;

  IndexType2 tmp = 0;
  for (IndexType2 i = tid; i < n; i += (IndexType2)blockDim.x * gridDim.x) {
    tmp += a[i];
  }
  if (tid < (IndexType2)blockDim.x * gridDim.x) {
    out[tid] = tmp;
  }
}

template <typename IndexType>
__global__ void _find_inds(IndexType *self_rows, IndexType *self_cols,
                           IndexType *rows, IndexType *cols,
                           IndexType *full_inds, IndexType *counts,
                           IndexType num_self_rows, IndexType num_rows) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  IndexType tid = threadIdx.x;
  __shared__ IndexType cache_rows[THREADS_PER_BLOCK];
  __shared__ IndexType cache_cols[THREADS_PER_BLOCK];

  IndexType my_row = (i < num_self_rows) ? self_rows[i] : -1;
  IndexType my_col = (i < num_self_rows) ? self_cols[i] : -1;

  IndexType my_full_ind = 0;
  IndexType my_count = 0;

  for (IndexType j = 0; j < num_rows; j += THREADS_PER_BLOCK) {
    if (j + tid < num_rows) {
      cache_rows[tid] = rows[j + tid];
      cache_cols[tid] = cols[j + tid];
    }
    __syncthreads();

    for (IndexType idx = j; idx < min(j + THREADS_PER_BLOCK, num_rows); idx++) {
      IndexType cond =
          (my_row == cache_rows[idx - j]) & (my_col == cache_cols[idx - j]);
      my_full_ind = my_full_ind * (1 - cond) + idx * cond;
      my_count += cond;
    }
    __syncthreads();
  }

  if (i < num_self_rows) {
    full_inds[i] = my_full_ind;
    counts[i] = my_count;
  }
}

template <typename IndexType>
__global__ void _compute_coo_block_mask(IndexType *rows, IndexType *cols,
                                        IndexType row_start, IndexType row_stop,
                                        IndexType col_start, IndexType col_stop,
                                        bool *mask, IndexType rows_len) {

  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  if (i < rows_len) {
    mask[i] = ((rows[i] >= row_start) && (rows[i] < row_stop) &&
               (cols[i] >= col_start) && (cols[i] < col_stop));
  }
}

template <typename IndexType, typename ValueType>
__global__ void _densify_block(ValueType *block, IndexType *rows,
                               IndexType *cols, ValueType *data,
                               IndexType stack_size, IndexType stack_stride,
                               IndexType nnz_per_block, IndexType num_rows,
                               IndexType num_cols, IndexType block_start,
                               IndexType row_offset, IndexType col_offset) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  IndexType nnz_total = stack_size * nnz_per_block;

  if (i < nnz_total) {
    IndexType stack_idx = i / nnz_per_block;
    IndexType stack_start = stack_idx * stack_stride;
    IndexType nnz_idx = i % nnz_per_block + block_start;
    IndexType block_size = num_rows * num_cols;

    IndexType row = rows[nnz_idx];
    IndexType col = cols[nnz_idx];

    block[stack_idx * block_size + (row - row_offset) * num_cols +
          (col - col_offset)] = data[stack_start + nnz_idx];
  }
}

template <typename IndexType>
__global__ void _find_bcoords(IndexType *block_offsets, IndexType *rows,
                              IndexType *cols, IndexType *brows,
                              IndexType *bcols, IndexType rows_len,
                              IndexType block_offsets_len) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  if (i < rows_len) {
    for (IndexType j = 0; j < block_offsets_len; j++) {
      IndexType cond_rows = block_offsets[j] <= rows[i];
      brows[i] = brows[i] * (1 - cond_rows) + j * cond_rows;
      IndexType cond_cols = block_offsets[j] <= cols[i];
      bcols[i] = bcols[i] * (1 - cond_cols) + j * cond_cols;
    }
  }
}

template <typename IndexType>
__global__ void _compute_block_mask(IndexType *brows, IndexType *bcols,
                                    IndexType brow, IndexType bcol, bool *mask,
                                    IndexType brows_len) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  if (i < brows_len) {
    mask[i] = (brows[i] == brow) & (bcols[i] == bcol);
  }
}

template <typename IndexType>
__global__ void _compute_block_inds(IndexType *rr, IndexType *cc,
                                    IndexType *self_cols, IndexType *rowptr,
                                    IndexType *block_inds, IndexType rr_len) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  if (i < rr_len) {
    IndexType r = rr[i];
    IndexType ind = -1;
    for (IndexType j = rowptr[r]; j < rowptr[r + 1]; j++) {
      IndexType cond = self_cols[j] == cc[i];
      ind = ind * (1 - cond) + j * cond;
    }
    block_inds[i] = ind;
  }
}

template <typename IndexType>
__global__ void _expand_rows(IndexType *rows, IndexType *rowptr,
                             IndexType rowptr_len) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  if (i < rowptr_len - 1) {
    for (IndexType j = rowptr[i]; j < rowptr[i + 1]; j++) {
      rows[j] = i;
    }
  }
}

template <typename IndexType>
__global__ void _find_ranks(IndexType *nnz_section_offsets, IndexType *inds,
                            IndexType *ranks, IndexType nnz_section_offsets_len,
                            IndexType inds_len) {
  IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
  IndexType i = global_block_offset + threadIdx.x;
  if (i < inds_len) {
    for (IndexType j = 0; j < nnz_section_offsets_len; j++) {
      IndexType cond = nnz_section_offsets[j] <= inds[i];
      ranks[i] = ranks[i] * (1 - cond) + j * cond;
    }
  }
}
