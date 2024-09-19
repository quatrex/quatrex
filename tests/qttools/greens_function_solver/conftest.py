# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest

from qttools.greens_function_solver import Inv, RGF

SEED = 63
np.random.seed(SEED)

GFSOLVERS_TYPE = [Inv, RGF]

N_DIAGONAL_BLOCKS = [
    pytest.param(1, id="1-blocks"),
    pytest.param(2, id="2-blocks"),
    pytest.param(3, id="3-blocks"),
    pytest.param(10, id="10-blocks"),
]

BLOCK_SIZES = [
    pytest.param(np.array([2] * 10), id="constant-block-size"),
    pytest.param(np.array([2] * 3 + [4] * 2 + [2] * 3), id="mixed-block-size"),
]


@pytest.fixture(params=GFSOLVERS_TYPE, autouse=True)
def gfsolver_type(request):
    return request.param


@pytest.fixture(autouse=True)
def bt_dense() -> np.ndarray:
    """Returns a random block-tridiagonal matrix."""
    bt = np.zeros(
        (N_DIAGONAL_BLOCKS * BLOCK_SIZES, N_DIAGONAL_BLOCKS * BLOCK_SIZES),
        dtype=complex,
    )

    # Fill the block-tridiagonal blocks
    for i in range(N_DIAGONAL_BLOCKS):
        bt[
            i * BLOCK_SIZES : (i + 1) * BLOCK_SIZES,
            i * BLOCK_SIZES : (i + 1) * BLOCK_SIZES,
        ] = np.random.rand(BLOCK_SIZES, BLOCK_SIZES) + 1j * np.random.rand(
            BLOCK_SIZES, BLOCK_SIZES
        )
        if i > 0:
            bt[
                i * BLOCK_SIZES : (i + 1) * BLOCK_SIZES,
                (i - 1) * BLOCK_SIZES : i * BLOCK_SIZES,
            ] = np.random.rand(BLOCK_SIZES, BLOCK_SIZES) + 1j * np.random.rand(
                BLOCK_SIZES, BLOCK_SIZES
            )
            bt[
                (i - 1) * BLOCK_SIZES : i * BLOCK_SIZES,
                i * BLOCK_SIZES : (i + 1) * BLOCK_SIZES,
            ] = np.random.rand(BLOCK_SIZES, BLOCK_SIZES) + 1j * np.random.rand(
                BLOCK_SIZES, BLOCK_SIZES
            )

    # Make the matrix diagonally dominant
    for i in range(bt.shape[0]):
        bt[i, i] = 1 + np.sum(bt[i, :])

    return bt


@pytest.fixture(scope="function", autouse=True)
def BT_array(blocksize: int, n_blocks: int) -> np.ndarray:
    """Returns a dense random complex block-tridiagonal matrix."""
    array_shape = (blocksize * n_blocks, blocksize * n_blocks)
    BT = np.zeros(array_shape, dtype=complex)
    for i in range(n_blocks):
        BT[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            np.random.rand(blocksize, blocksize)
            + 1j * np.random.rand(blocksize, blocksize)
        )
        if i > 0:
            BT[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ] = np.random.rand(blocksize, blocksize) + 1j * np.random.rand(
                blocksize, blocksize
            )
            BT[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ] = np.random.rand(blocksize, blocksize) + 1j * np.random.rand(
                blocksize, blocksize
            )

    return BT


@pytest.fixture(scope="function", autouse=True)
def BT_block_sizes(blocksize: int, n_blocks: int) -> np.ndarray:
    """Returns block sizes based on the array_shape."""
    return np.repeat(blocksize, n_blocks)


@pytest.fixture(scope="function", autouse=True)
def cut_dense_to_BT():
    def _cut_dense_to_BT(
        BT_array: np.ndarray, blocksize: int, n_blocks: int
    ) -> np.ndarray:
        """Returns a dense block-tridiagonal matrix."""
        array_shape = (blocksize * n_blocks, blocksize * n_blocks)
        BT = np.zeros(array_shape, dtype=complex)
        for i in range(n_blocks):
            BT[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ] = BT_array[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ]
            if i > 0:
                BT[
                    i * blocksize : (i + 1) * blocksize,
                    (i - 1) * blocksize : i * blocksize,
                ] = BT_array[
                    i * blocksize : (i + 1) * blocksize,
                    (i - 1) * blocksize : i * blocksize,
                ]
                BT[
                    (i - 1) * blocksize : i * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ] = BT_array[
                    (i - 1) * blocksize : i * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]

        return BT

    return _cut_dense_to_BT
