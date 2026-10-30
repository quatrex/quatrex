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


def scattering_rate(
    x_lesser: DSDBSparse,
    x_greater: DSDBSparse,
    sse_lesser: DSDBSparse,
    sse_greater: DSDBSparse,
) -> tuple[NDArray, NDArray]:
    """Scattering rates from Green's functions and self-energies.

    The in-scattering rate is given by the diagonal of the lesser
    self-energy times the greater Green's function, while the
    out-scattering rate is given by the diagonal of the greater
    self-energy times the lesser Green's function.

    The sign convention is such that in-scattering is positive and
    out-scattering is negative.

    Note that this implementation only works if the Green's functions
    and self-energies have identical sparsity patterns.

    Parameters
    ----------
    x_lesser : DSDBSparse
        Lesser Green's function.
    x_greater : DSDBSparse
        Greater Green's function.
    sse_lesser : DSDBSparse
        Lesser self-energy.
    sse_greater : DSDBSparse
        Greater self-energy.

    Returns
    -------
    scattering : NDArray
        The scattering rate. Positive values correspond to
        in-scattering, while negative values mean out-scattering.

    """
    # The Hadamard product of two skew-hermitian matrices is Hermitian.
    # To get the diagonal elements of the matrix product, we need to sum
    # over rows afterwards, exploiting the Hermitian symmetry. After
    # multiplying the two matrices, take the real part, because the
    # imaginary part will sum to zero anyway.
    product = xp.real(
        sse_lesser.data * -x_greater.data.conj()  # in-scattering
        - sse_greater.data * -x_lesser.data.conj()  # out-scattering
    )
    scattering = xp.zeros(x_lesser.shape[:-1], dtype=product.dtype)
    xp.add.at(scattering, (..., x_lesser.rows), product)

    if x_lesser.symmetry:
        off = x_lesser.rows != x_lesser.cols
        xp.add.at(scattering, (..., x_lesser.cols[off]), product[..., off])

    scattering = comm.block.all_gather_v(scattering, axis=-1)
    scattering = comm.stack.all_gather_v(
        scattering, axis=0, mask=x_lesser._stack_padding_mask
    )

    return scattering


def current_conservation(
    x_lesser: DSDBSparse,
    x_greater: DSDBSparse,
    sse_lesser: DSDBSparse,
    sse_greater: DSDBSparse,
) -> tuple[NDArray, NDArray]:
    r"""Checks current conservation.

    See eq. (12.34) in H. Haug and A.-P. Jauho,
    "Quantum Kinetics in Transport and Optics of Semiconductors"

    $$
    \int dE dk sum_{ij} sigma_{ij}^< * G_{ji}^> - sigma_{ij}^> * G_{ji}^< = 0
    $$

    This function computes a relative value of the current conservation
    condition. This is the absolute value of the condition divided by
    the mean of the in-scattering and out-scattering terms.

    We can use the fact that traces of matrix products can be computed
    using the Frobenius inner product, i.e. $Tr(A B) = sum_{ij} A_{ij}
    B_{ji}$, and the skew-symmetric property of the Green's functions
    $G_{ji}^< = -[G_{ij}^<]^*$ such that we don't have to communicate
    stuff.

    Note that this implementation only works if the Green's functions
    and self-energies have identical sparsity patterns.

    Parameters
    ----------
    x_lesser : DSDBSparse
        The lesser Green's function.
    x_greater : DSDBSparse
        The greater Green's function.
    sse_lesser : DSDBSparse
        The lesser interaction self-energy.
    sse_greater : DSDBSparse
        The greater interaction self-energy.

    Returns
    -------
    current_conservation : NDArray
        Relative difference between in-scattering and out-scattering.

    """
    # Taking the real part here, because when symmetry is enabled, we
    # only store the upper triangular part of the Green's functions and
    # self-energies. If symmetry is off and we sum the whole tensor, the
    # imaginary part of this sum is around machine epsilon as expected.
    in_scattering = xp.sum(xp.real(sse_lesser.data * -x_greater.data.conj()))
    out_scattering = xp.sum(xp.real(sse_greater.data * -x_lesser.data.conj()))

    sendbuf = xp.array([in_scattering, out_scattering], dtype=xp.float64)
    recvbuf_block = xp.empty_like(sendbuf)
    comm.block.all_reduce(sendbuf, recvbuf_block)
    recvbuf_stack = xp.empty_like(sendbuf)
    comm.stack.all_reduce(recvbuf_block, recvbuf_stack)

    in_scattering, out_scattering = recvbuf_stack

    mean = 0.5 * (xp.abs(in_scattering) + xp.abs(out_scattering))
    abs_diff = xp.abs(in_scattering - out_scattering)
    rel_diff = abs_diff / mean if mean != 0 else 0
    return rel_diff
