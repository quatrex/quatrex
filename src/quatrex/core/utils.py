# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.
from qttools.datastructures import DSBSparse
from qttools import xp


def homogenize(matrix: DSBSparse) -> None:
    """Homogenizes a matrix in stack distribution.

    Parameters
    ----------
    matrix : DSBSparse
        The matrix to homogenize.
    """

    assert xp.all(matrix.block_sizes == matrix.block_sizes[0])
    if matrix.distribution_state != "stack":
        raise ValueError("Matrix must be in stack distribution")

    for i in range(len(matrix.block_sizes) - 2):
        matrix.blocks[i + 1, i + 1] = matrix.blocks[0, 0]
        matrix.blocks[i + 1, i + 2] = matrix.blocks[0, 1]
        matrix.blocks[i + 2, i + 1] = matrix.blocks[1, 0]

    matrix.blocks[-1, -1] = matrix.blocks[0, 0]
    matrix.blocks[-1, -2] = matrix.blocks[1, 0]
