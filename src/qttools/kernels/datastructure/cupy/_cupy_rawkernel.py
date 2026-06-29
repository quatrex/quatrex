# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray
from qttools.kernels import index_types, value_types
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK

kernels_template = None

if comm.rank == 0:
    with open(__file__.replace(".py", ".cu"), "r") as f:
        kernels_template = f.read()

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
    (idx[0], name): f"{name}<{idx[1]}>"
    for idx in index_types.items()
    for name in kernel_names
}

for idx1 in index_types.items():
    for idx2 in index_types.items():
        name_expressions[(idx1[0], idx2[0], "_reduction")] = (
            f"_reduction<{idx1[1]}, {idx2[1]}>"
        )

for idx in index_types.items():
    name_expressions[cp.bool_, idx[0], "_reduction"] = f"_reduction<bool, {idx[1]}>"

for idx in index_types.items():
    for val in value_types.items():
        name_expressions[(idx[0], val[0], "_densify_block")] = (
            f"_densify_block<{idx[1]}, {val[1]}>"
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

    _reduction = kernels[(index_types[dtype], "_reduction")]

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
