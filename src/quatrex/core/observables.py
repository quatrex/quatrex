import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse

from quatrex.electron import ElectronSolver


def get_block(
    coo: sparse.coo_matrix,
    block_sizes: xp.ndarray,
    block_offsets: xp.ndarray,
    index: tuple,
) -> xp.ndarray:
    """Gets a block from a COO matrix."""
    row, col = index

    mask = (
        (block_offsets[row] <= coo.row)
        & (coo.row < block_offsets[row + 1])
        & (block_offsets[col] <= coo.col)
        & (coo.col < block_offsets[col + 1])
    )
    block = xp.zeros((int(block_sizes[row]), int(block_sizes[col])), dtype=coo.dtype)
    block[
        coo.row[mask] - block_offsets[row],
        coo.col[mask] - block_offsets[col],
    ] = coo.data[mask]

    return block


def density(x: DSBSparse, overlap: sparse.spmatrix | None = None) -> np.ndarray:
    """Computes the density from the Green's function."""
    if overlap is None:
        local_density = x.diagonal().imag
        return np.vstack(comm.allgather(local_density))

    local_density = []
    overlap = overlap.tocoo()
    for i in range(x.num_blocks):
        overlap_diag = get_block(overlap, x.block_sizes, x.block_offsets, (i, i))
        local_density_slice = np.diagonal(
            x.blocks[i, i] @ overlap_diag, axis1=-2, axis2=-1
        ).copy()
        if i < x.num_blocks - 1:
            overlap_upper = get_block(
                overlap, x.block_sizes, x.block_offsets, (i + 1, i)
            )
            local_density_slice += np.diagonal(
                x.blocks[i, i + 1] @ overlap_upper, axis1=-2, axis2=-1
            )
        if i > 0:
            overlap_lower = get_block(
                overlap, x.block_sizes, x.block_offsets, (i - 1, i)
            )
            local_density_slice += np.diagonal(
                x.blocks[i, i - 1] @ overlap_lower, axis1=-2, axis2=-1
            )

        local_density.append(local_density_slice.imag)

    return np.vstack(comm.allgather(np.hstack(local_density)))


def contact_currents(solver: ElectronSolver) -> np.ndarray:
    """Computes the contact currents."""
    i_left = np.hstack(comm.allgather(solver.i_left))
    i_right = np.hstack(comm.allgather(solver.i_right))
    return i_left, i_right
