# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

SEED = 63

import pytest
from scipy import sparse
import numpy as np

np.random.seed(SEED)


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
