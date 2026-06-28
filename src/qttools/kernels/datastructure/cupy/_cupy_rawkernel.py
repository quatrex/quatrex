# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp

from qttools import NDArray
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK

index_types = {
    cp.int32: "int",
    cp.int64: "long long",
}

reduction_template = r"""
template<typename IndexType>
__global__ void _reduction(
        IndexType *a,
        IndexType *out,
        IndexType n
){
    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType tid = global_block_offset + threadIdx.x;

    IndexType tmp = 0;
    for(IndexType i = tid; i < n; i += (IndexType)blockDim.x * gridDim.x) {
        tmp += a[i];
    }
    if(tid < (IndexType)blockDim.x * gridDim.x){
        out[tid] = tmp;
    }

}
"""

name_expressions = [f"_reduction<{idx}>" for idx in index_types.values()]

reduction_module = cp.RawModule(
    code=reduction_template,
    name_expressions=name_expressions,
    options=("-std=c++17",),
)


def reduction(
    a: NDArray,
) -> NDArray:
    """Performs a reduction operation on the input array.

    NOTE: This is a naive implementation for SC25
    This was needed since cupy didnt perform well on MI250X.
    TODO: Further investigate on newer cupy versions.

    Parameters
    ----------
    a : NDArray
        Input array to be reduced.

    Returns
    -------
    NDArray
        Reduced output array.

    """
    dtype = a.dtype.type

    n_blocks = 4
    out = cp.zeros((n_blocks * THREADS_PER_BLOCK), dtype=dtype)

    n = a.size

    _reduction = reduction_module.get_function(f"_reduction<{index_types[dtype]}>")

    _reduction(
        (n_blocks,),
        (THREADS_PER_BLOCK,),
        (
            a,
            out,
            dtype(n),
        ),
    )

    out = cp.sum(out)

    return out


kernels_template = r"""
#include <cupy/complex.cuh>

template<typename IndexType>
__global__ void _find_inds(
    IndexType* self_rows,
    IndexType* self_cols,
    IndexType* rows,
    IndexType* cols,
    IndexType* full_inds,
    IndexType* counts,
    IndexType num_self_rows,
    IndexType num_rows
) {
    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    IndexType tid = threadIdx.x;
    __shared__ IndexType cache_rows[%THREADS_PER_BLOCK%];
    __shared__ IndexType cache_cols[%THREADS_PER_BLOCK%];
                                    

    IndexType my_row = (i < num_self_rows) ? self_rows[i] : -1;
    IndexType my_col = (i < num_self_rows) ? self_cols[i] : -1;
                        
    IndexType my_full_ind = 0;
    IndexType my_count = 0;
                                
    for (IndexType j = 0; j < num_rows; j += %THREADS_PER_BLOCK%) {
        if (j + tid < num_rows) {
            cache_rows[tid] = rows[j + tid];
            cache_cols[tid] = cols[j + tid];
        }
        __syncthreads();
                                    
        for (IndexType idx = j; idx < min(j + %THREADS_PER_BLOCK%, num_rows); idx++) {
            IndexType cond = (my_row == cache_rows[idx - j]) & (my_col == cache_cols[idx - j]);
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

template<typename IndexType>
__global__ void _compute_coo_block_mask(
    IndexType *rows,
    IndexType *cols,
    IndexType row_start,
    IndexType row_stop,
    IndexType col_start,
    IndexType col_stop,
    IndexType *mask,
    IndexType rows_len
){

    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    if (i < rows_len) {
        mask[i] = (
            (rows[i] >= row_start)
            && (rows[i] < row_stop)
            && (cols[i] >= col_start)
            && (cols[i] < col_stop)
        );
    }
}

template<typename IndexType, typename ValueType>
__global__ void _densify_block(
    ValueType* block,
    IndexType* rows,
    IndexType* cols,
    ValueType* data,
    IndexType stack_size,
    IndexType stack_stride,
    IndexType nnz_per_block,
    IndexType num_rows,
    IndexType num_cols,
    IndexType block_start,
    IndexType row_offset,
    IndexType col_offset
) {
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

        block[stack_idx * block_size + (row - row_offset) * num_cols + (col - col_offset)] = data[stack_start + nnz_idx];
    } 
}

template<typename IndexType>
__global__ void _find_bcoords(
    IndexType *block_offsets,
    IndexType *rows,
    IndexType *cols,
    IndexType *brows,
    IndexType *bcols,
    IndexType rows_len,
    IndexType block_offsets_len
) {
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

template<typename IndexType>
__global__ void _compute_block_mask(
    IndexType *brows,
    IndexType *bcols,
    IndexType brow,
    IndexType bcol,
    IndexType *mask,
    IndexType brows_len
) {
    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    if (i < brows_len) {
        mask[i] = (brows[i] == brow) & (bcols[i] == bcol);
    }
}

template<typename IndexType>
__global__ void _compute_block_inds(
    IndexType *rr,
    IndexType *cc,
    IndexType *self_cols,
    IndexType *rowptr,
    IndexType *block_inds,
    IndexType rr_len
) {
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

template<typename IndexType>
__global__ void _expand_rows(
    IndexType *rows,
    IndexType *rowptr,
    IndexType rowptr_len
) {
    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    if (i < rowptr_len - 1) {
        for (IndexType j = rowptr[i]; j < rowptr[i + 1]; j++) {
            rows[j] = i;
        }
    }
}

template<typename IndexType>
__global__ void _find_ranks(
    IndexType *nnz_section_offsets,
    IndexType *inds,
    IndexType *ranks,
    IndexType nnz_section_offsets_len,
    IndexType inds_len
) {
    IndexType global_block_offset = (IndexType)blockDim.x * blockIdx.x;
    IndexType i = global_block_offset + threadIdx.x;
    if (i < inds_len) {
        for (IndexType j = 0; j < nnz_section_offsets_len; j++) {
            IndexType cond = nnz_section_offsets[j] <= inds[i];
            ranks[i] = ranks[i] * (1 - cond) + j * cond;
        }
    }
}

"""


value_types = {
    cp.float32: "float",
    cp.float64: "double",
    cp.complex64: "complex<float>",
    cp.complex128: "complex<double>",
}

kernels_template = kernels_template.replace(
    "%THREADS_PER_BLOCK%", str(THREADS_PER_BLOCK)
)

index_kernel_names = [
    "_find_inds",
    "_compute_coo_block_mask",
    "_find_bcoords",
    "_compute_block_mask",
    "_compute_block_inds",
    "_expand_rows",
    "_find_ranks",
]

name_expressions = [
    f"{name}<{idx}>" for idx in index_types.values() for name in index_kernel_names
]
kernel_names = [
    "_densify_block",
]
for name in kernel_names:
    for idx in index_types.values():
        for val in value_types.values():
            name_expressions.append(f"{name}<{idx}, {val}>")

module = cp.RawModule(
    code=kernels_template,
    name_expressions=name_expressions,
    options=("-std=c++17",),
)


def _find_inds(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = module.get_function(f"_find_inds<{index_types[args[0].dtype.type]}>")
    kernel(grid, block, args)


def _compute_coo_block_mask(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = module.get_function(
        f"_compute_coo_block_mask<{index_types[args[0].dtype.type]}>"
    )
    kernel(grid, block, args)


def _densify_block(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = module.get_function(
        f"_densify_block<{index_types[args[1].dtype.type]}, {value_types[args[0].dtype.type]}>"
    )
    kernel(grid, block, args)


def _find_bcoords(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = module.get_function(f"_find_bcoords<{index_types[args[0].dtype.type]}>")
    kernel(grid, block, args)


def _compute_block_mask(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = module.get_function(
        f"_compute_block_mask<{index_types[args[0].dtype.type]}>"
    )
    kernel(grid, block, args)


def _compute_block_inds(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = module.get_function(
        f"_compute_block_inds<{index_types[args[0].dtype.type]}>"
    )
    kernel(grid, block, args)


def _expand_rows(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = module.get_function(f"_expand_rows<{index_types[args[0].dtype.type]}>")
    kernel(grid, block, args)


def _find_ranks(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = module.get_function(f"_find_ranks<{index_types[args[0].dtype.type]}>")
    kernel(grid, block, args)


__all__ = [
    "_find_inds",
    "_compute_coo_block_mask",
    "_densify_block",
    "_find_bcoords",
    "_compute_block_mask",
    "_compute_block_inds",
    "_expand_rows",
    "_find_ranks",
]
