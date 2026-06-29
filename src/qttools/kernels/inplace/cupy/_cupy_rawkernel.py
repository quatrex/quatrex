# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from itertools import product

import cupy as cp
from mpi4py.MPI import COMM_WORLD as comm

from qttools.kernels import index_types, value_types

kernels_template = None

if comm.rank == 0:
    with open(__file__.replace(".py", ".cu"), "r") as f:
        kernels_template = f.read()

kernels_template = comm.bcast(kernels_template, root=0)

name_expressions = {}
for t1, t2, t3, idx in product(
    value_types.items(), value_types.items(), value_types.items(), index_types.items()
):

    # Only compile double versions of the kernels, since float versions are not used in practice.
    if (
        t1[0] in [cp.float32, cp.complex64]
        or t2[0] in [cp.float32, cp.complex64]
        or t3[0] in [cp.float32, cp.complex64]
    ):
        continue

    # Skip when T1 is real and T2/T3 is complex, since this would result in a type mismatch.
    if (t1[0] in [cp.float64]) and (
        t2[0] in [cp.complex128] or t3[0] in [cp.complex128]
    ):
        continue

    name = "_scatter_add_scaled"
    name_expressions[(t1[0], t2[0], t3[0], idx[0], name)] = (
        f"{name}<{t1[1]},{t2[1]},{t3[1]},{idx[1]}>"
    )


for idx in index_types.items():
    name = "_scatter_add_scaled_obc"
    name_expressions[(idx[0], name)] = f"{name}<{idx[1]}>"

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
