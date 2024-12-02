# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
from qttools.datastructures import DSBSparse


def assemble_boundary_blocks(
    diag_left: NDArray,
    right: NDArray,
    below: NDArray,
    diag_right: NDArray,
    above: NDArray,
    left: NDArray,
    mat: DSBSparse,
) -> None:
    """Assembles boundary blocks from the small system matrix blocks.

    Parameters
    ----------
    diag_left : NDArray
        The diagonal left block.
    right : NDArray
        The right block.
    below : NDArray
        The below block.
    diag_right : NDArray
        The diagonal right block.
    above : NDArray
        The above block.
    left : NDArrays
        The left block.
    mat : DSBSparse
        The small system matrix.

    """
    diag_left[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[0, 0],
                    mat.blocks[0, 1],
                    mat.blocks[0, 2],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[1, 0],
                    mat.blocks[0, 0],
                    mat.blocks[0, 1],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[2, 0],
                    mat.blocks[1, 0],
                    mat.blocks[0, 0],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    right[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[0, 3],
                    xp.zeros_like(mat.blocks[0, 3]),
                    xp.zeros_like(mat.blocks[0, 3]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[0, 2],
                    mat.blocks[0, 3],
                    xp.zeros_like(mat.blocks[0, 2]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[0, 1],
                    mat.blocks[0, 2],
                    mat.blocks[0, 3],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    below[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[3, 0],
                    xp.zeros_like(mat.blocks[3, 0]),
                    xp.zeros_like(mat.blocks[3, 0]),
                ),
                axis=1,
            ),
            xp.concatenate(
                (
                    mat.blocks[2, 0],
                    mat.blocks[3, 0],
                    xp.zeros_like(mat.blocks[2, 0]),
                ),
                axis=1,
            ),
            xp.concatenate(
                (
                    mat.blocks[1, 0],
                    mat.blocks[2, 0],
                    mat.blocks[3, 0],
                ),
                axis=1,
            ),
        ),
        axis=2,
    )
    diag_right[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[-1, -1],
                    mat.blocks[-2, -1],
                    mat.blocks[-3, -1],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-1, -2],
                    mat.blocks[-1, -1],
                    mat.blocks[-2, -1],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-1, -3],
                    mat.blocks[-1, -2],
                    mat.blocks[-1, -1],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    above[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[-4, -1],
                    xp.zeros_like(mat.blocks[-4, -1]),
                    xp.zeros_like(mat.blocks[-4, -1]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-3, -1],
                    mat.blocks[-4, -1],
                    xp.zeros_like(mat.blocks[-3, -1]),
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    mat.blocks[-2, -1],
                    mat.blocks[-3, -1],
                    mat.blocks[-4, -1],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
    left[:] = xp.concatenate(
        (
            xp.concatenate(
                (
                    mat.blocks[-1, -4],
                    mat.blocks[-1, -3],
                    mat.blocks[-1, -2],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    xp.zeros_like(mat.blocks[-1, -3]),
                    mat.blocks[-1, -4],
                    mat.blocks[-1, -3],
                ),
                axis=2,
            ),
            xp.concatenate(
                (
                    xp.zeros_like(mat.blocks[-1, -4]),
                    xp.zeros_like(mat.blocks[-1, -4]),
                    mat.blocks[-1, -4],
                ),
                axis=2,
            ),
        ),
        axis=1,
    )
