# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import importlib.resources
from itertools import product

import cupy as cp
from mpi4py.MPI import COMM_WORLD as comm

from qttools.kernels import index_types, value_types

kernels_template = None

if comm.rank == 0:
    cu_file = importlib.resources.files(__package__) / "_cupy_rawkernel.cu"
    kernels_template = cu_file.read_text(encoding="utf-8")

kernels_template = comm.bcast(kernels_template, root=0)

name_expressions = {}
for (
    (value1_numpy_type, value1_c_type),
    (value2_numpy_type, value2_c_type),
    (value3_numpy_type, value3_c_type),
    (index_numpy_type, index_c_type),
) in product(
    value_types.items(), value_types.items(), value_types.items(), index_types.items()
):

    # Only compile double versions of the kernels, since float versions are not used in practice.
    if (
        value1_numpy_type in [cp.float32, cp.complex64]
        or value2_numpy_type in [cp.float32, cp.complex64]
        or value3_numpy_type in [cp.float32, cp.complex64]
    ):
        continue

    # Skip when T1 is real and T2/T3 is complex, since this would result in a type mismatch.
    if (value1_numpy_type in [cp.float64]) and (
        value2_numpy_type in [cp.complex128] or value3_numpy_type in [cp.complex128]
    ):
        continue

    name = "_scatter_add_scaled"
    name_expressions[
        (
            value1_numpy_type,
            value2_numpy_type,
            value3_numpy_type,
            index_numpy_type,
            name,
        )
    ] = f"{name}<{value1_c_type},{value2_c_type},{value3_c_type},{index_c_type}>"


for index_numpy_type, index_c_type in index_types.items():
    name = "_scatter_add_scaled_obc"
    name_expressions[(index_numpy_type, name)] = f"{name}<{index_c_type}>"

module = cp.RawModule(
    code=kernels_template,
    name_expressions=name_expressions.values(),
    options=("-std=c++17",),
)

kernels = {key: module.get_function(value) for key, value in name_expressions.items()}


def _scatter_add_scaled(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = kernels[
        (
            args[0].dtype.type,
            args[1].dtype.type,
            args[4].dtype.type,
            args[2].dtype.type,
            "_scatter_add_scaled",
        )
    ]
    kernel(grid, block, args)


def _scatter_add_scaled_obc(
    grid: tuple[int, int, int], block: tuple[int, int, int], args: tuple
):
    kernel = kernels[(args[7].dtype.type, "_scatter_add_scaled_obc")]
    kernel(grid, block, args)
