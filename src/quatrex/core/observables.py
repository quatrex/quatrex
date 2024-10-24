import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures.dsbsparse import DSBSparse
from scipy import sparse

from quatrex.electron import ElectronSolver


def density(x: DSBSparse, overlap: sparse.sparray | None = None) -> np.ndarray:
    """Computes the density from the Green's function."""
    if overlap is None:
        local_density = x.diagonal().imag
        return np.vstack(comm.allgather(local_density))

    local_density = []
    overlap = overlap.tolil()
    for i in range(x.num_blocks):
        overlap_diag = overlap[
            x.block_offsets[i] : x.block_offsets[i + 1],
            x.block_offsets[i] : x.block_offsets[i + 1],
        ].toarray()
        local_density_slice = np.diagonal(
            x.blocks[i, i] @ overlap_diag, axis1=-2, axis2=-1
        ).copy()
        if i < x.num_blocks - 1:
            overlap_upper = overlap[
                x.block_offsets[i + 1] : x.block_offsets[i + 2],
                x.block_offsets[i] : x.block_offsets[i + 1],
            ].toarray()
            local_density_slice += np.diagonal(
                x.blocks[i, i + 1] @ overlap_upper, axis1=-2, axis2=-1
            )
        if i > 0:
            overlap_lower = overlap[
                x.block_offsets[i - 1] : x.block_offsets[i],
                x.block_offsets[i] : x.block_offsets[i + 1],
            ].toarray()
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
