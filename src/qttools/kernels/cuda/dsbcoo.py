# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import os
import warnings

import cupy as cp
from cupyx import jit

from qttools import USE_CUPY_JIT, NDArray, host_xp
from qttools.kernels.cuda import THREADS_PER_BLOCK
from qttools.profiling import Profiler

# NOTE: CUDA kernels are not profiled, as the jit-compiled kernels
# cannot find the correct name of the function to profile.
profiler = Profiler()

USE_FIND_INDS = os.environ.get("USE_FIND_INDS", "true").lower()
if USE_FIND_INDS in ("y", "yes", "t", "true", "on", "1"):
    USE_FIND_INDS = True
elif USE_FIND_INDS in ("n", "no", "f", "false", "off", "0"):
    USE_FIND_INDS = False
else:
    warnings.warn(f"Invalid truth value {USE_FIND_INDS=}. Defaulting to 'true'.")
    USE_FIND_INDS = True

USE_DENSIFY_BLOCKS = os.environ.get("USE_DENSIFY_BLOCK", "false").lower()
if USE_DENSIFY_BLOCKS in ("y", "yes", "t", "true", "on", "1"):
    USE_DENSIFY_BLOCKS = True
elif USE_DENSIFY_BLOCKS in ("n", "no", "f", "false", "off", "0"):
    USE_DENSIFY_BLOCKS = False
else:
    warnings.warn(f"Invalid truth value {USE_DENSIFY_BLOCKS=}. Defaulting to 'false'.")
    USE_DENSIFY_BLOCKS = False

if USE_CUPY_JIT:

    @jit.rawkernel()
    def _find_inds_kernel(
        self_rows: NDArray,
        self_cols: NDArray,
        rows: NDArray,
        cols: NDArray,
        full_inds: NDArray,
        counts: NDArray,
        num_self_rows: int,
        num_rows: int,
    ):
        """Finds the corresponding indices of the given rows and columns.

        This also counts the number of matches found, which is used to check
        if the indices contain duplicates.

        Parameters
        ----------
        self_rows : NDArray
            The rows of this matrix.
        self_cols : NDArray
            The columns of this matrix.
        rows : NDArray
            The rows to find the indices for.
        cols : NDArray
            The columns to find the indices for.
        full_inds : NDArray
            The indices of the given rows and columns.
        counts : NDArray
            The number of matches found.


        """
        i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
        if i < num_self_rows:
            for j in range(num_rows):
                cond = int((self_rows[i] == rows[j]) & (self_cols[i] == cols[j]))
                full_inds[i] = full_inds[i] * (1 - cond) + j * cond
                counts[i] += cond

else:
    _find_inds_kernel = cp.RawKernel(
        r"""
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
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num_self_rows) {
                for (int j = 0; j < num_rows; j++) {
                    int cond = (self_rows[i] == rows[j]) & (self_cols[i] == cols[j]);
                    full_inds[i] = full_inds[i] * (1 - cond) + j * cond;
                    counts[i] += cond;
                }
            }
        }
    """,
        "find_inds",
    )

if USE_CUPY_JIT:

    @jit.rawkernel()
    def _find_inds_kernel_new(
        self_rows: NDArray,
        self_cols: NDArray,
        rows: NDArray,
        cols: NDArray,
        full_inds: NDArray,
        counts: NDArray,
        num_self_rows: int,
        num_rows: int,
    ):
        """Finds the corresponding indices of the given rows and columns.

        This also counts the number of matches found, which is used to check
        if the indices contain duplicates.

        Parameters
        ----------
        self_rows : NDArray
            The rows of this matrix.
        self_cols : NDArray
            The columns of this matrix.
        rows : NDArray
            The rows to find the indices for.
        cols : NDArray
            The columns to find the indices for.
        full_inds : NDArray
            The indices of the given rows and columns.
        counts : NDArray
            The number of matches found.


        """
        i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
        tid = int(jit.threadIdx.x)
        cache_rows = jit.shared_memory(cp.int32, THREADS_PER_BLOCK)
        cache_cols = jit.shared_memory(cp.int32, THREADS_PER_BLOCK)

        if i < num_self_rows:
            my_row = self_rows[i]
            my_col = self_cols[i]
        else:
            my_row = -1
            my_col = -1

        my_full_ind = 0
        my_count = 0

        for j in range(0, num_rows, THREADS_PER_BLOCK):
            if j + tid < num_rows:
                cache_rows[tid] = rows[j + tid]
                cache_cols[tid] = cols[j + tid]
            jit.syncthreads()

            for idx in range(j, min(j + THREADS_PER_BLOCK, num_rows)):
                cond = int(
                    (my_row == cache_rows[idx - j]) & (my_col == cache_cols[idx - j])
                )
                my_full_ind = my_full_ind * (1 - cond) + idx * cond
                my_count += cond
            jit.syncthreads()

        if i < self_rows.shape[0]:
            full_inds[i] = my_full_ind
            counts[i] = my_count

else:
    _find_inds_kernel_new = cp.RawKernel(
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


@profiler.profile(level="api")
def find_inds(
    self_rows: NDArray, self_cols: NDArray, rows: NDArray, cols: NDArray
) -> tuple[NDArray, NDArray, int]:
    """Finds the corresponding indices of the given rows and columns.

    Parameters
    ----------
    self_rows : NDArray
        The rows of this matrix.
    self_cols : NDArray
        The columns of this matrix.
    rows : NDArray
        The rows to find the indices for.
    cols : NDArray
        The columns to find the indices for.

    Returns
    -------
    inds : NDArray
        The indices of the given rows and columns.
    value_inds : NDArray
        The matching indices of this matrix.
    max_counts : int
        The maximum number of matches found.

    """
    rows = rows.astype(cp.int32)
    cols = cols.astype(cp.int32)
    full_inds = cp.zeros(self_rows.shape[0], dtype=cp.int32)
    counts = cp.zeros(self_rows.shape[0], dtype=cp.int16)
    THREADS_PER_BLOCK
    blocks_per_grid = (self_rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    if USE_FIND_INDS:
        _find_inds_kernel_new(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                self_rows,
                self_cols,
                rows,
                cols,
                full_inds,
                counts,
                self_rows.shape[0],
                rows.shape[0],
            ),
        )
    else:
        _find_inds_kernel(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                self_rows,
                self_cols,
                rows,
                cols,
                full_inds,
                counts,
                self_rows.shape[0],
                rows.shape[0],
            ),
        )

    # Find the valid indices.
    inds = cp.nonzero(counts)[0]
    value_inds = full_inds[inds]

    return inds, value_inds, int(cp.max(counts))


if USE_CUPY_JIT:

    @jit.rawkernel()
    def _compute_coo_block_mask_kernel(
        rows: NDArray,
        cols: NDArray,
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int,
        mask: NDArray,
        rows_len: int,
    ):
        """Computes the mask for the block in the coordinates.

        Parameters
        ----------
        rows : NDArray
            The row indices of the matrix.
        cols : NDArray
            The column indices of the matrix.
        row_start : int
            The start row index of the block.
        row_stop : int
            The stop row index of the block.
        col_start : int
            The start column index of the block.
        col_stop : int
            The stop column index of the block.
        mask : NDArray
            The mask to store the result.

        """
        i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
        if i < rows_len:
            mask[i] = (
                (rows[i] >= row_start)
                & (rows[i] < row_stop)
                & (cols[i] >= col_start)
                & (cols[i] < col_stop)
            )

else:
    _compute_coo_block_mask_kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void _compute_coo_block_mask_kernel(
            int *rows,
            int *cols,
            int row_start,
            int row_stop,
            int col_start,
            int col_stop,
            bool *mask,
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


@profiler.profile(level="api")
def compute_block_slice(
    rows: NDArray, cols: NDArray, block_offsets: NDArray, row: int, col: int
) -> slice:
    """Computes the slice of the block in the data.

    Parameters
    ----------
    rows : NDArray
        The row indices of the matrix.
    cols : NDArray
        The column indices of the matrix.
    block_offsets : NDArray
        The offsets of the blocks.
    row : int
        The block row to compute the slice for.
    col : int
        The block column to compute the slice for.

    Returns
    -------
    start : int
        The start index of the block.
    stop : int
        The stop index of the block.

    """
    mask = cp.zeros(rows.shape[0], dtype=cp.bool_)
    row_start, row_stop = host_xp.int32(block_offsets[row]), host_xp.int32(
        block_offsets[row + 1]
    )
    col_start, col_stop = host_xp.int32(block_offsets[col]), host_xp.int32(
        block_offsets[col + 1]
    )

    rows = rows.astype(cp.int32)
    cols = cols.astype(cp.int32)

    blocks_per_grid = (rows.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _compute_coo_block_mask_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            rows,
            cols,
            row_start,
            row_stop,
            col_start,
            col_stop,
            mask,
            host_xp.int32(rows.shape[0]),
        ),
    )
    if cp.sum(mask) == 0:
        # No data in this block, return an empty slice.
        return None, None

    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.
    inds = cp.nonzero(mask)[0]

    # NOTE: this copies back to the host
    return int(inds[0]), int(inds[-1] + 1)


@profiler.profile(level="api")
# def densify_block(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray):
def densify_block(
    block: NDArray,
    rows: NDArray,
    cols: NDArray,
    data: NDArray,
    block_slice: slice,
    row_offset: int,
    col_offset: int,
):
    """Fills the dense block with the given data.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to fill the block with.
    block : NDArray
        Preallocated dense block. Should be filled with zeros.

    """
    # TODO: The bare API implementation on the GPU is faster than the
    # very simple, non-general kernel i came up with. Thus, for now i
    # will just use the CuPy API directly. Since for very large blocks
    # (10'000x10'000) this starts to break even, this needs to be
    # revisited!
    if not USE_DENSIFY_BLOCKS:
        block[..., rows[block_slice] - row_offset, cols[block_slice] - col_offset] = (
            data[..., block_slice]
        )

    else:
        THREADS_PER_BLOCK
        stack_size = data.size // data.shape[-1]
        stack_stride = data.shape[-1]
        block_start = block_slice.start or 0
        nnz_per_block = block_slice.stop - block_start
        num_blocks = (
            stack_size * nnz_per_block + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK
        _densify_block_kernel(
            (num_blocks,),
            (THREADS_PER_BLOCK,),
            (
                block.reshape(-1),
                rows,
                cols,
                data.reshape(-1),
                stack_size,
                stack_stride,
                nnz_per_block,
                block.shape[-2],
                block.shape[-1],
                block_start,
                row_offset,
                col_offset,
            ),
        )


if USE_CUPY_JIT:

    @jit.rawkernel()
    def _densify_block_kernel(
        block: NDArray,
        rows: NDArray,
        cols: NDArray,
        data: NDArray,
        stack_size: int,
        stack_stride: int,
        nnz_per_block: int,
        num_rows: int,
        num_cols: int,
        block_start: int,
        row_offset: int,
        col_offset: int,
    ):
        """Fills the dense block with the given data.

        Parameters
        ----------
        block : NDArray
            The dense block to fill.
        rows : NDArray
            The rows at which to fill the block.
        cols : NDArray
            The columns at which to fill the block.
        data : NDArray
            The data to fill the block with.

        """

        i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
        nnz_total = stack_size * nnz_per_block

        if i < nnz_total:
            stack_idx = i // nnz_per_block
            stack_start = stack_idx * stack_stride
            nnz_idx = i % nnz_per_block + block_start
            block_size = num_rows * num_cols

            row = rows[nnz_idx]
            col = cols[nnz_idx]
            block[
                stack_idx * block_size
                + (row - row_offset) * num_cols
                + (col - col_offset)
            ] = data[stack_start + nnz_idx]

else:
    _densify_block_kernel = cp.RawKernel(
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


@profiler.profile(level="api")
def sparsify_block(block: NDArray, rows: NDArray, cols: NDArray, data: NDArray):
    """Fills the data with the given dense block.

    Note
    ----
    This is not a raw kernel, as there seems to be no performance gain
    for this operation on the GPU.

    Parameters
    ----------
    block : NDArray
        The dense block to sparsify.
    rows : NDArray
        The rows at which to fill the block.
    cols : NDArray
        The columns at which to fill the block.
    data : NDArray
        The data to be filled with the block.

    """
    # TODO: Test whether a custom kernel could be faster here.
    data[:] = block[..., rows, cols]


@profiler.profile(level="api")
def compute_block_sort_index(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> NDArray:
    """Computes the block-sorting index for a sparse matrix.

    Note
    ----
    Due to the Python for loop around the kernel, this method will
    perform best for larger block sizes (>500).

    Parameters
    ----------
    coo_rows : NDArray
        The row indices of the matrix in coordinate format.
    coo_cols : NDArray
        The column indices of the matrix in coordinate format.
    block_sizes : NDArray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : NDArray
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = block_sizes.shape[0]
    block_offsets = host_xp.hstack(
        (host_xp.array([0]), host_xp.cumsum(block_sizes)), dtype=host_xp.int32
    )

    sort_index = cp.zeros(len(coo_cols), dtype=cp.int32)
    mask = cp.zeros(len(coo_cols), dtype=cp.bool_)
    coo_rows = coo_rows.astype(cp.int32)
    coo_cols = coo_cols.astype(cp.int32)

    blocks_per_grid = (len(coo_cols) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    offset = 0
    for i, j in cp.ndindex(num_blocks, num_blocks):
        _compute_coo_block_mask_kernel(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                coo_rows,
                coo_cols,
                host_xp.int32(block_offsets[i]),
                host_xp.int32(block_offsets[i + 1]),
                host_xp.int32(block_offsets[j]),
                host_xp.int32(block_offsets[j + 1]),
                mask,
                host_xp.int32(len(coo_cols)),
            ),
        )

        bnnz = cp.sum(mask)

        if bnnz != 0:
            # Sort the data by block-row and -column.
            sort_index[offset : offset + bnnz] = cp.nonzero(mask)[0]

            offset += bnnz

    return sort_index
