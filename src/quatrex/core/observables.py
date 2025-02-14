# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from functools import partial

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import OBCBlocks


def get_block(
    coo: sparse.coo_matrix,
    block_sizes: NDArray,
    block_offsets: NDArray,
    index: tuple,
) -> NDArray:
    """Gets a block from a COO matrix.

    Parameters
    ----------
    coo : sparse.coo_matrix
        The COO matrix.
    block_sizes : NDArray
        The block sizes.
    block_offsets : NDArray
        The block offsets.
    index : tuple
        The index of the block to extract.

    Returns
    -------
    block : NDArray
        The requested, dense block.

    """
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


def density(x: DSBSparse, overlap: sparse.spmatrix | None = None) -> NDArray:
    """Computes the density from Green's function and overlap matrix.

    Parameters
    ----------
    x : DSBSparse
        The Green's function.
    overlap : sparse.spmatrix, optional
        The overlap matrix, by default None.

    Returns
    -------
    NDArray
        The density, i.e. the imaginary part of the diagonal of the
        Green's function.

    """
    if overlap is None:
        local_density = x.diagonal().imag
        return xp.vstack(comm.allgather(local_density))

    local_density = []
    overlap = overlap.tocoo()
    _overlap_block = partial(get_block, overlap, x.block_sizes, x.block_offsets)
    for i in range(x.num_blocks):
        local_density_slice = xp.diagonal(
            x.blocks[i, i] @ _overlap_block((i, i)),
            axis1=-2,
            axis2=-1,
        ).copy()
        if i < x.num_blocks - 1:
            local_density_slice += xp.diagonal(
                x.blocks[i, i + 1] @ _overlap_block((i + 1, i)),
                axis1=-2,
                axis2=-1,
            )
        if i > 0:
            local_density_slice += xp.diagonal(
                x.blocks[i, i - 1] @ _overlap_block((i - 1, i)),
                axis1=-2,
                axis2=-1,
            )

        local_density.append(local_density_slice.imag)

    return xp.vstack(comm.allgather(xp.hstack(local_density)))


def contact_currents(
    x_lesser: DSBSparse, x_greater: DSBSparse, sigma_obc_blocks: OBCBlocks
) -> tuple[NDArray, NDArray]:
    """Computes the contact currents.

    Parameters
    ----------
    x_lesser : DSBSparse
        The lesser Green's function.
    x_greater : DSBSparse
        The greater Green's function.
    sigma_obc_blocks : OBCBlocks
        The OBC self-energy blocks.


    Returns
    -------
    NDArray
        The contact currents, gathered across all participating ranks.

    """
    i_left = xp.trace(
        sigma_obc_blocks.greater[0] @ x_lesser.blocks[0, 0]
        - x_greater.blocks[0, 0] @ sigma_obc_blocks.lesser[0],
        axis1=-2,
        axis2=-1,
    )
    i_right = xp.trace(
        sigma_obc_blocks.greater[-1] @ x_lesser.blocks[-1, -1]
        - x_greater.blocks[-1, -1] @ sigma_obc_blocks.lesser[-1],
        axis1=-2,
        axis2=-1,
    )

    i_left = xp.hstack(comm.allgather(i_left))
    i_right = xp.hstack(comm.allgather(i_right))
    return i_left, i_right


def device_current(x_lesser: DSBSparse, operator: sparse.spmatrix) -> NDArray:
    """Computes the current from the lesser Green's function.

    Parameters
    ----------
    x_lesser : DSBSparse
        The lesser Green's function.
    operator : sparse.spmatrix
        The operator that governs the system dynamics.

    Returns
    -------
    NDArray
        The current, gathered across all participating ranks.

    """
    operator = operator.tocoo()
    _operator_block = partial(
        get_block, operator, x_lesser.block_sizes, x_lesser.block_offsets
    )
    local_current = []
    for i in range(x_lesser.num_blocks - 1):
        j = i + 1
        layer_current = (
            _operator_block((i, j)) * x_lesser.blocks[j, i].swapaxes(-2, -1)
            - x_lesser.blocks[i, j] * _operator_block((j, i)).swapaxes(-2, -1)
        ).sum(axis=(-1, -2))
        local_current.append(layer_current)

    return xp.vstack(comm.allgather(xp.vstack(local_current).T))
