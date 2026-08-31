# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the methods to calculate density and current from Green's functions."""

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse


def density(x: DSDBSparse, overlap: DSDBSparse | None = None) -> NDArray:
    """Computes the density from Green's function and overlap matrix.

    Parameters
    ----------
    x : DSDBSparse
        The Green's function.
    overlap : DSDBSparse, optional
        The overlap matrix, by default None.

    Returns
    -------
    NDArray
        The density, i.e. the imaginary part of the diagonal of the
        Green's function.

    """
    if overlap is None:
        local_density = x.diagonal().imag
        local_density = comm.block.all_gather_v(
            local_density,
            axis=-1,
        )

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
    for i in range(x.num_blocks):
        local_density_slice = xp.diagonal(
            x.blocks[i, i] @ overlap.blocks[i, i],
            axis1=-2,
            axis2=-1,
        ).copy()
        if i < x.num_blocks - 1:
            local_density_slice += xp.diagonal(
                x.blocks[i, i + 1] @ overlap.blocks[i + 1, i],
                axis1=-2,
                axis2=-1,
            )
        if i > 0:
            local_density_slice += xp.diagonal(
                x.blocks[i, i - 1] @ overlap.blocks[i - 1, i],
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


def current_conservation(
    x_lesser: DSDBSparse,
    x_greater: DSDBSparse,
    se_int_lesser: DSDBSparse,
    se_int_greater: DSDBSparse,
) -> NDArray:
    r"""Checks current conservation.
    See eq. (12.34) in H. Haug and A.-P. Jauho,
    "Quantum Kinetics in Transport and Optics of Semiconductors"

    $$
    \int dE dk sum_{ij} sigma_{ij}^< * G_{ji}^> - sigma_{ij}^> * G_{ji}^< = 0
    $$

    We can use the skew-symmetric property of the Green's functions $G_{ji}^< = -[G_{ij}^<]^*$
    such that we don't have to communicate the greater Green's function.

    Parameters
    ----------
    x_lesser : DSDBSparse
        The lesser Green's function.
    x_greater : DSDBSparse
        The greater Green's function.
    se_int_lesser : DSDBSparse
        The lesser interaction self-energy.
    se_int_greater : DSDBSparse
        The greater interaction self-energy.
    Returns
    -------
    tuple[NDArray, NDArray]
        The absolute value of the current conservation and the
        relative value of the current conservation.
    """
    term1 = (se_int_lesser.data * (-x_greater.data.conj())).sum()
    term2 = (se_int_greater.data * (-x_lesser.data.conj())).sum()

    sendbuff = xp.array([term1, term2], dtype=x_lesser.dtype)
    recvbuff_block = xp.empty_like(sendbuff)
    comm.block.all_reduce(sendbuff, recvbuff_block)
    recvbuff_stack = xp.empty_like(sendbuff)
    comm.stack.all_reduce(recvbuff_block, recvbuff_stack)

    term1, term2 = recvbuff_stack

    current_conservation_absolute = term1 - term2
    current_conservation_relative = (
        current_conservation_absolute / (term1 + term2) if (term1 + term2) != 0 else 0
    )

    return xp.abs(current_conservation_absolute), xp.abs(current_conservation_relative)
