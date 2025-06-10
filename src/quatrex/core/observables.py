# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from functools import partial

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks


def get_block(
    coo: sparse.coo_matrix | DSDBSparse,
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

    if isinstance(coo, DSDBSparse):
        start_block = coo.block_section_offsets[comm.block.rank]
        return coo.blocks[row - start_block, col - start_block]

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


def density(x: DSDBSparse, overlap: sparse.spmatrix | None = None) -> NDArray:
    """Computes the density from Green's function and overlap matrix.

    Parameters
    ----------
    x : DSDBSparse
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
        return comm.stack.all_gather_v(
            local_density,
            axis=0,
            mask=x._stack_padding_mask,
        )

    if comm.block.size > 1:
        raise NotImplementedError(
            "Overlap density calculation is not implemented for distributed systems."
        )

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

    local_density = xp.concatenate(local_density, axis=-1)

    return comm.stack.all_gather_v(
        local_density,
        axis=0,
        mask=x._stack_padding_mask,
    )


def contact_currents(
    x_lesser: DSDBSparse, x_greater: DSDBSparse, sigma_obc_blocks: OBCBlocks
) -> tuple[NDArray, NDArray]:
    """Computes the contact currents.

    Parameters
    ----------
    x_lesser : DSDBSparse
        The lesser Green's function.
    x_greater : DSDBSparse
        The greater Green's function.
    sigma_obc_blocks : OBCBlocks
        The OBC self-energy blocks.


    Returns
    -------
    NDArray
        The contact currents, gathered across all participating ranks.

    """
    if comm.block.rank == 0:
        i_left = xp.trace(
            sigma_obc_blocks.greater[0] @ x_lesser.blocks[0, 0]
            - x_greater.blocks[0, 0] @ sigma_obc_blocks.lesser[0],
            axis1=-2,
            axis2=-1,
        )
    else:
        i_left = xp.empty(x_lesser.stack_shape, dtype=x_lesser.dtype)

    if comm.block.rank == comm.block.size - 1:
        n = x_lesser.num_local_blocks - 1
        i_right = xp.trace(
            sigma_obc_blocks.greater[-1] @ x_lesser.blocks[n, n]
            - x_greater.blocks[n, n] @ sigma_obc_blocks.lesser[-1],
            axis1=-2,
            axis2=-1,
        )
    else:
        i_right = xp.empty(x_lesser.stack_shape, dtype=x_lesser.dtype)

    comm.block.bcast(i_left, root=0)
    comm.block.bcast(i_right, root=comm.block.size - 1)

    full_i_left = comm.stack.all_gather_v(
        i_left,
        axis=0,
        mask=x_lesser._stack_padding_mask,
    )
    full_i_right = comm.stack.all_gather_v(
        i_right,
        axis=0,
        mask=x_lesser._stack_padding_mask,
    )

    return (
        full_i_left,
        full_i_right,
    )


def device_current(
    x_lesser: DSDBSparse, operator: sparse.spmatrix | DSDBSparse
) -> NDArray:
    """Computes the current from the lesser Green's function.

    Parameters
    ----------
    x_lesser : DSDBSparse
        The lesser Green's function.
    operator : sparse.spmatrix
        The operator that governs the system dynamics.

    Returns
    -------
    NDArray
        The current, gathered across all participating ranks.

    """
    if isinstance(operator, sparse.spmatrix):
        operator = operator.tocoo()
    _operator_block = partial(
        get_block, operator, x_lesser.block_sizes, x_lesser.block_offsets
    )

    local_current = []
    start_block = x_lesser.block_section_offsets[comm.block.rank]
    num_offdiags = x_lesser.num_local_blocks

    if comm.block.rank == comm.block.size - 1:
        num_offdiags -= 1

    for i in range(num_offdiags):
        j = i + 1
        layer_current = (
            _operator_block((i + start_block, j + start_block))
            * x_lesser.blocks[j, i].swapaxes(-2, -1)
            - x_lesser.blocks[i, j]
            * _operator_block((j + start_block, i + start_block)).swapaxes(-2, -1)
        ).sum(axis=(-1, -2))
        local_current.append(layer_current)

    local_current = xp.array(local_current)
    block_local_current = comm.block.all_gather_v(local_current, axis=0)
    block_local_current = xp.ascontiguousarray(block_local_current)
    block_local_current = xp.moveaxis(block_local_current, 0, -1)

    return comm.stack.all_gather_v(
        block_local_current,
        axis=0,
        mask=x_lesser._stack_padding_mask,
    )
