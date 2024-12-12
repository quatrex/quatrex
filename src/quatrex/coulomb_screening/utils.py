# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray
from qttools.datastructures import DSBSparse


def assemble_boundary_blocks(
    diag_left: NDArray,
    right: NDArray,
    below: NDArray,
    diag_right: NDArray,
    above: NDArray,
    left: NDArray,
    mat: DSBSparse,
    nbc: int,
) -> None:
    """Assembles the boundary blocks from the smaller blocks of input Hamiltonian.

    Parameters
    ----------
    diag_left : NDArray
        The diagonal left block.
    right : NDArray
        The right block of the diagonal left block.
    below : NDArray
        The below block of the diagonal left block.
    diag_right : NDArray
        The diagonal right block.
    above : NDArray
        The above block of the diagonal right block.
    left : NDArray
        The left block of the diagonal right block.
    mat : DSBSparse
        The boundary system matrix.
    nbc : int
        The number of small blocks used for the boundary blocks.
    """
    # TODO: Below is not fault-tolerant. Add checks.
    bs = diag_left.shape[1] // nbc
    for i in range(nbc):
        for j in range(nbc):
            k = i - j
            diag_left[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                max(0, k), -min(0, k)
            ]
            right[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                0, -k + nbc
            ]
            below[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                k + nbc, 0
            ]
            diag_right[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                min(0, k) - 1, -max(0, k) - 1
            ]
            above[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                k - nbc - 1, -1
            ]
            left[..., i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = mat.blocks[
                -1, -k - nbc - 1
            ]
