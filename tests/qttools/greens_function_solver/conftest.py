# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import pytest
import scipy.sparse as sparse
from numpy.typing import ArrayLike

from qttools.datastructures import DSBCOO
from qttools.greens_function_solver import RGF, Inv
from qttools.utils.gpu_utils import xp

GFSOLVERS_TYPE = [Inv, RGF]

DSBSPARSE_TYPES = [DSBCOO]

BLOCK_SIZES = [
    pytest.param(xp.array([2] * 10), id="constant-block-size"),
    pytest.param(xp.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size"),
]


BATCHING_TYPE = [
    pytest.param("no-batching"),
    pytest.param("2-batching"),
    pytest.param("all-batching"),
]

OUT = [
    pytest.param(True, id="out_true"),
    pytest.param(False, id="out_false"),
]

RETURN_RETARDED = [
    pytest.param(True, id="return_retarded"),
    pytest.param(False, id="not_return_retarded"),
]

GLOBAL_STACK_SHAPES = [
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
]


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def block_sizes(request):
    return request.param


@pytest.fixture(params=GFSOLVERS_TYPE, autouse=True)
def gfsolver_type(request):
    return request.param


@pytest.fixture(params=DSBSPARSE_TYPES, autouse=True)
def dsbsparse_type(request):
    return request.param


@pytest.fixture(scope="function", autouse=False)
def bt_dense(
    block_sizes: ArrayLike,
):
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
    num_blocks = len(block_sizes)
    size = int(xp.sum(block_sizes))

    arr = xp.zeros((size, size), dtype=xp.complex128)

    # Fill the block-tridiagonal blocks
    for i in range(num_blocks):
        block = xp.asarray(
            sparse.random(
                block_sizes[i],
                block_sizes[i],
                density=0.5,
                format="coo",
                dtype=xp.complex128,
            ).toarray()
        ) + xp.identity(int(block_sizes[i]), dtype=xp.complex128)

        arr[
            block_offsets[i] : block_offsets[i + 1],
            block_offsets[i] : block_offsets[i + 1],
        ] = block

        if i > 0:
            block = xp.asarray(
                sparse.random(
                    block_sizes[i],
                    block_sizes[i - 1],
                    density=0.5,
                    format="coo",
                    dtype=xp.complex128,
                ).toarray()
            )

            arr[
                block_offsets[i] : block_offsets[i + 1],
                block_offsets[i - 1] : block_offsets[i],
            ] = block

            block = xp.asarray(
                sparse.random(
                    block_sizes[i - 1],
                    block_sizes[i],
                    density=0.5,
                    format="coo",
                    dtype=xp.complex128,
                ).toarray()
            )

            arr[
                block_offsets[i - 1] : block_offsets[i],
                block_offsets[i] : block_offsets[i + 1],
            ] = block

    # Make the matrix diagonally dominant
    for i in range(arr.shape[0]):
        arr[i, i] = (1 + 1j) + complex(xp.sum(arr[i, :]))

    return arr


@pytest.fixture(params=BATCHING_TYPE, autouse=True)
def max_batch_size(request) -> int:
    batching_type = request.param

    if batching_type == "no-batching":
        return 1
    elif batching_type == "2-batching":
        return 2
    elif batching_type == "all-batching":
        return 100


@pytest.fixture(params=OUT, autouse=True)
def out(request) -> bool:
    return request.param


@pytest.fixture(params=RETURN_RETARDED, autouse=True)
def return_retarded(request) -> bool:
    return request.param


@pytest.fixture(params=GLOBAL_STACK_SHAPES, autouse=True)
def global_stack_shape(request):
    return request.param
