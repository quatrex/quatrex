# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from functools import partial

from qttools import (
    NCCL_AVAILABLE,
    NDArray,
    block_comm,
    nccl_stack_comm,
    sparse,
    stack_comm,
    xp,
)
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.utils.gpu_utils import synchronize_device


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
        start_block = coo.block_section_offsets[block_comm.rank]
        return coo.local_blocks[row - start_block, col - start_block]

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
        if not NCCL_AVAILABLE:
            return xp.vstack(stack_comm.allgather(local_density))

        # NOTE: NCCL does not expose all_gather_v. This is a hack.
        pad_width = x.total_stack_size // stack_comm.size - local_density.shape[0]
        local_density = xp.pad(local_density, ((0, pad_width), (0, 0)))
        density = xp.empty(
            (x.total_stack_size, local_density.shape[-1]), dtype=local_density.dtype
        )
        synchronize_device()
        nccl_stack_comm.all_gather(local_density, density, local_density.size)
        synchronize_device()
        return density[x._stack_padding_mask, ...]

    raise NotImplementedError(
        "Overlap density calculation is not implemented for distributed systems."
    )

    # local_density = []
    # overlap = overlap.tocoo()
    # _overlap_block = partial(get_block, overlap, x.block_sizes, x.block_offsets)
    # for i in range(x.num_blocks):
    #     local_density_slice = xp.diagonal(
    #         x.blocks[i, i] @ _overlap_block((i, i)),
    #         axis1=-2,
    #         axis2=-1,
    #     ).copy()
    #     if i < x.num_blocks - 1:
    #         local_density_slice += xp.diagonal(
    #             x.blocks[i, i + 1] @ _overlap_block((i + 1, i)),
    #             axis1=-2,
    #             axis2=-1,
    #         )
    #     if i > 0:
    #         local_density_slice += xp.diagonal(
    #             x.blocks[i, i - 1] @ _overlap_block((i - 1, i)),
    #             axis1=-2,
    #             axis2=-1,
    #         )

    #     local_density.append(local_density_slice.imag)

    # local_density = xp.hstack(local_density)

    # if not NCCL_AVAILABLE:
    #     return xp.vstack(stack_comm.allgather(local_density))

    # # NOTE: NCCL does not expose all_gather_v. This is a hack.
    # local_density = xp.vstack(local_density)
    # pad_width = x.total_stack_size // stack_comm.size - local_density.shape[0]
    # local_density = xp.pad(local_density, ((0, pad_width), (0, 0)))
    # density = xp.empty(
    #     (x.total_stack_size, local_density.shape[-1]), dtype=local_density.dtype
    # )
    # synchronize_device()
    # nccl_stack_comm.all_gather(local_density, density, local_density.size)
    # synchronize_device()
    # return density[x._stack_padding_mask, ...]


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
    i_left = None
    i_right = None
    if block_comm.rank == 0:
        i_left = xp.trace(
            sigma_obc_blocks.greater[0] @ x_lesser.local_blocks[0, 0]
            - x_greater.local_blocks[0, 0] @ sigma_obc_blocks.lesser[0],
            axis1=-2,
            axis2=-1,
        )
    if block_comm.rank == block_comm.size - 1:
        n = x_lesser.num_local_blocks - 1
        i_right = xp.trace(
            sigma_obc_blocks.greater[-1] @ x_lesser.local_blocks[n, n]
            - x_greater.local_blocks[n, n] @ sigma_obc_blocks.lesser[-1],
            axis1=-2,
            axis2=-1,
        )

    i_left = block_comm.bcast(i_left, root=0)
    i_right = block_comm.bcast(i_right, root=block_comm.size - 1)

    if not NCCL_AVAILABLE:
        i_left = xp.hstack(stack_comm.allgather(i_left))
        i_right = xp.hstack(stack_comm.allgather(i_right))
        return i_left, i_right

    # NOTE: NCCL does not expose all_gather_v. This is a hack.
    pad_width = x_lesser.total_stack_size // stack_comm.size - i_left.shape[0]
    i_left = xp.pad(i_left, (0, pad_width))
    i_right = xp.pad(i_right, (0, pad_width))
    full_i_left = xp.empty((x_lesser.total_stack_size,), dtype=i_left.dtype)
    full_i_right = xp.empty((x_lesser.total_stack_size,), dtype=i_right.dtype)
    synchronize_device()
    nccl_stack_comm.all_gather(i_left, full_i_left, i_left.size)
    synchronize_device()
    nccl_stack_comm.all_gather(i_right, full_i_right, i_right.size)
    synchronize_device()
    return (
        full_i_left[x_lesser._stack_padding_mask],
        full_i_right[x_lesser._stack_padding_mask],
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
    # local_current = []
    # x_lesser_upper_blocks = x_lesser.block_diagonal(offset=1)
    # x_lesser_lower_blocks = x_lesser.block_diagonal(offset=-1)

    # for i in range(x_lesser.num_blocks - 1):
    #     j = i + 1
    #     layer_current = (
    #         _operator_block((i, j)) * (x_lesser_lower_blocks[i].swapaxes(-2, -1))
    #         - x_lesser_upper_blocks[i] * _operator_block((j, i)).swapaxes(-2, -1)
    #     ).sum(axis=(-1, -2))
    #     local_current.append(layer_current)
    local_current = []
    start_block = x_lesser.block_section_offsets[block_comm.rank]
    num_offdiags = x_lesser.num_local_blocks

    if block_comm.rank == block_comm.size - 1:
        num_offdiags -= 1

    for i in range(num_offdiags):
        j = i + 1
        layer_current = (
            _operator_block((i + start_block, j + start_block))
            * x_lesser.local_blocks[j, i].swapaxes(-2, -1)
            - x_lesser.local_blocks[i, j]
            * _operator_block((j + start_block, i + start_block)).swapaxes(-2, -1)
        ).sum(axis=(-1, -2))
        local_current.append(layer_current)
    local_current = xp.vstack(block_comm.allgather(local_current))

    local_current = xp.ascontiguousarray(xp.vstack(local_current).T)

    if not NCCL_AVAILABLE:
        return xp.vstack(stack_comm.allgather(local_current))

    # NOTE: NCCL does not expose all_gather_v. This is a hack.
    pad_width = x_lesser.total_stack_size // stack_comm.size - local_current.shape[0]
    local_current = xp.pad(local_current, ((0, pad_width), (0, 0)))

    device_current = xp.empty(
        (x_lesser.total_stack_size, local_current.shape[-1]), dtype=local_current.dtype
    )
    synchronize_device()
    nccl_stack_comm.all_gather(local_current, device_current, local_current.size)
    synchronize_device()
    return device_current[x_lesser._stack_padding_mask, ...]
