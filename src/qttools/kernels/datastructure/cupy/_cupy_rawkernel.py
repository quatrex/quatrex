# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import importlib.resources

import cupy as cp
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray
from qttools.kernels import index_types, value_types
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK

kernels_template = None

if comm.rank == 0:
    cu_file = importlib.resources.files(__package__) / "_cupy_rawkernel.cu"
    kernels_template = cu_file.read_text(encoding="utf-8")

kernels_template = comm.bcast(kernels_template, root=0)

kernels_template = kernels_template.replace(
    "TEMPLATE_THREADS_PER_BLOCK", str(THREADS_PER_BLOCK)
)

kernel_names = [
    "_find_inds",
    "_compute_coo_block_mask",
    "_find_bcoords",
    "_compute_block_mask",
    "_compute_block_inds",
    "_expand_rows",
    "_find_ranks",
]

name_expressions = {
    (index_numpy_type, name): f"{name}<{index_c_type}>"
    for (index_numpy_type, index_c_type) in index_types.items()
    for name in kernel_names
}

for index1_numpy_type, index1_c_type in index_types.items():
    for index2_numpy_type, index2_c_type in index_types.items():
        name_expressions[(index1_numpy_type, index2_numpy_type, "_reduction")] = (
            f"_reduction<{index1_c_type}, {index2_c_type}>"
        )

for index_numpy_type, index_c_type in index_types.items():
    name_expressions[cp.bool_, index_numpy_type, "_reduction"] = (
        f"_reduction<bool, {index_c_type}>"
    )

for index_numpy_type, index_c_type in index_types.items():
    for val_numpy_type, val_c_type in value_types.items():
        name_expressions[(index_numpy_type, val_numpy_type, "_densify_block")] = (
            f"_densify_block<{index_c_type}, {val_c_type}>"
        )

module = cp.RawModule(
    code=kernels_template,
    name_expressions=name_expressions.values(),
    options=("-std=c++17",),
)

kernels = {key: module.get_function(value) for key, value in name_expressions.items()}


def _find_inds(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = kernels[(args[0].dtype.type, "_find_inds")]
    kernel(grid, block, args)


def _compute_coo_block_mask(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = kernels[(args[0].dtype.type, "_compute_coo_block_mask")]
    kernel(grid, block, args)


def _densify_block(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = kernels[(args[1].dtype.type, args[0].dtype.type, "_densify_block")]
    kernel(grid, block, args)


def _find_bcoords(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = kernels[(args[0].dtype.type, "_find_bcoords")]
    kernel(grid, block, args)


def _compute_block_mask(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = kernels[(args[0].dtype.type, "_compute_block_mask")]
    kernel(grid, block, args)


def _compute_block_inds(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = kernels[(args[0].dtype.type, "_compute_block_inds")]
    kernel(grid, block, args)


def _expand_rows(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = kernels[(args[0].dtype.type, "_expand_rows")]
    kernel(grid, block, args)


def _find_ranks(grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple):
    kernel = kernels[(args[0].dtype.type, "_find_ranks")]
    kernel(grid, block, args)


def reduction(
    a: NDArray,
) -> NDArray:
    """Performs a reduction operation on the input array.

    Notes
    -----
    This is a naive implementation for SC25 This was needed since cupy
    didnt perform well on MI250X. TODO: Further investigate on newer cupy
    versions.
    This function is not optimized for performance and may be slow.
    Furthermore, it can not handle all input types and shapes.

    Parameters
    ----------
    a : NDArray
        Input array to be reduced.

    Returns
    -------
    NDArray
        Reduced output array.

    """
    if a.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")

    dtype = a.dtype.type

    # NOTE: Harcode output dtype to int64
    # to prevent issues with overflow
    out_dtype = cp.int64

    n_blocks = 4
    out = cp.zeros((n_blocks * THREADS_PER_BLOCK), dtype=out_dtype)

    n = a.size

    _reduction = kernels[(dtype, out_dtype, "_reduction")]

    _reduction(
        (n_blocks,),
        (THREADS_PER_BLOCK,),
        (
            a,
            out,
            out_dtype(n),
        ),
    )

    out = cp.sum(out)

    return out
