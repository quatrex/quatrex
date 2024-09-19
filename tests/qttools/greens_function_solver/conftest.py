# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest

from qttools.greens_function_solver import RGF, Inv

SEED = 63
np.random.seed(SEED)

GFSOLVERS_TYPE = [Inv, RGF]

BLOCK_SIZES = [
    pytest.param(np.array([2] * 10), id="constant-block-size"),
    pytest.param(np.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size"),
]


@pytest.fixture(params=GFSOLVERS_TYPE, autouse=True)
def gfsolver_type(request):
    return request.param


@pytest.fixture(params=BLOCK_SIZES, autouse=True)
def bt_dense() -> np.ndarray:
    """Returns a random block-tridiagonal matrix."""
    block_sizes = BLOCK_SIZES

    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
    num_blocks = len(block_sizes)
    size = np.sum(block_sizes)

    arr = np.zeros((size, size), dtype=np.complex128)

    # Fill the block-tridiagonal blocks
    for i in range(num_blocks):
        arr[
            block_offsets[i] : block_offsets[i + 1],
            block_offsets[i] : block_offsets[i + 1],
        ] = np.random.rand(block_sizes[i], block_sizes[i]) + 1j * np.random.rand(
            block_sizes[i], block_sizes[i]
        )

        if i > 0:
            arr[
                block_offsets[i] : block_offsets[i + 1],
                block_offsets[i - 1] : block_offsets[i],
            ] = np.random.rand(
                block_sizes[i], block_sizes[i - 1]
            ) + 1j * np.random.rand(
                block_sizes[i], block_sizes[i - 1]
            )
            arr[
                block_offsets[i - 1] : block_offsets[i],
                block_offsets[i] : block_offsets[i + 1],
            ] = np.random.rand(
                block_sizes[i - 1], block_sizes[i]
            ) + 1j * np.random.rand(
                block_sizes[i - 1], block_sizes[i]
            )

    # Make the matrix diagonally dominant
    for i in range(arr.shape[0]):
        arr[i, i] = 1 + np.sum(arr[i, :])

    return arr
