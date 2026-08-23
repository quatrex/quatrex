# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes our CUDA inplace kernels."""

import cupy as cp
from cupyx.scipy import sparse

from qttools import NDArray
from qttools.kernels.inplace.cupy import _cupy_rawkernel

THREADS_PER_BLOCK = 1024


def scatter_add_scaled(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    alpha: complex | float = 1.0,
    conjugate: bool = False,
) -> None:
    """Adds array `b` to array `a` at indices `inds` in-place.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added to `a`.
    inds : NDArray
        The indices at which to add `b` to `a`.
    alpha : complex | float, optional
        The scalar multiplier for `b` before adding it to `a`.
    conjugate : bool, optional
        Whether to take the complex conjugate of `b` before adding it to
        `a`.

    """
    num_inds = inds.shape[0]
    blocks_per_grid = (num_inds + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    if isinstance(alpha, complex):
        alpha = cp.complex128(alpha)
    elif isinstance(alpha, float):
        alpha = cp.float64(alpha)
    else:
        raise TypeError(
            f"Unsupported type for alpha: {type(alpha)}. Must be float or complex."
        )

    index_type = inds.dtype.type

    _cupy_rawkernel._scatter_add_scaled(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (a, b, inds, index_type(num_inds), alpha, conjugate),
    )


def scatter_add_scaled_obc(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    k: tuple[float, float],
    transverse_repetition_grid: tuple[int, int],
    alpha: float = 1.0,
):
    """Adds array `b` to array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added to `a`.
    inds : NDArray
        The indices at which to add `b` to `a`.
    k : tuple[float, float]
        The transverse wavevector components.
    transverse_repetition_grid : tuple[int, int]
        The transverse repetition grid of the contact.
    alpha : float
        The scalar multiplier for `b` before adding it to `a`.

    """

    num_inds = inds.shape[0]

    ky, kz = k
    ny, nz = transverse_repetition_grid

    # Launch kernel
    blocks_per_grid = (num_inds + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK

    if a.dtype.type != cp.complex128 or b.dtype.type != cp.complex128:
        raise TypeError(
            "Only complex128 arrays are supported for scatter_add_scaled_obc."
        )

    if not isinstance(alpha, float):
        # NOTE: cupy will match float with double
        raise TypeError(
            "Only float alpha is supported for scatter_add_scaled_obc.\n"
            f"Got {type(alpha)} instead."
        )

    index_type = inds.dtype.type

    _cupy_rawkernel._scatter_add_scaled_obc(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            a,
            b.flatten(),
            ky,
            kz,
            index_type(b.shape[1] * ny * nz),
            index_type(b.shape[1]),
            index_type(nz),
            inds,
            index_type(num_inds),
            alpha,
        ),
    )


def add_bond_resolved_current(
    current: NDArray,
    system_matrix: sparse.csr_matrix | sparse.coo_matrix,
    phi: NDArray,
    alpha: float | complex = 1.0,
) -> None:
    r"""Add contribution for the bond resolved current.

    Parameters
    ----------
    current : NDArray
        The array to be updated with the bond resolved current.
    system_matrix : sparse.csr_matrix | sparse.coo_matrix
        The system matrix in either CSR or COO format.
    phi : NDArray
        The wavefunction array.
    alpha : float | complex, optional
        The scalar multiplier for the contribution before adding it to `current`.

    """

    if not (
        isinstance(system_matrix, sparse.csr_matrix)
        or isinstance(system_matrix, sparse.coo_matrix)
    ):
        raise TypeError("system_matrix must be a cupyx sparse matrix (csr or coo).")

    if len(current) != system_matrix.nnz:
        raise ValueError(
            "current and system_matrix must have the same number of non-zero elements."
        )

    if phi.shape[0] != system_matrix.shape[0]:
        raise ValueError(
            "phi must have the same number of rows as the system_matrix shape."
        )

    if isinstance(alpha, complex):
        alpha = cp.complex128(alpha)
    elif isinstance(alpha, float):
        alpha = cp.float64(alpha)
    else:
        raise TypeError(
            f"Unsupported type for alpha: {type(alpha)}. Must be float or complex."
        )

    if phi.dtype != cp.complex128:
        raise TypeError("phi must be of type complex128.")

    if current.dtype != cp.float64:
        raise TypeError("current must be of type float64.")

    if isinstance(system_matrix, sparse.csr_matrix):
        index_type = system_matrix.indices.dtype.type
        blocks_per_grid = (
            system_matrix.shape[0] + (THREADS_PER_BLOCK - 1)
        ) // THREADS_PER_BLOCK

        _cupy_rawkernel._add_bond_resolved_current_csr(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                system_matrix.indptr,
                system_matrix.indices,
                current,
                system_matrix.data,
                index_type(system_matrix.nnz),
                index_type(system_matrix.shape[0]),
                phi,
                index_type(phi.shape[1]),
                index_type(phi.strides[0] // phi.itemsize),
                index_type(phi.strides[1] // phi.itemsize),
                alpha,
            ),
        )

    elif isinstance(system_matrix, sparse.coo_matrix):
        index_type = system_matrix.row.dtype.type
        blocks_per_grid = (
            system_matrix.nnz + (THREADS_PER_BLOCK - 1)
        ) // THREADS_PER_BLOCK
        _cupy_rawkernel._add_bond_resolved_current_coo(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                system_matrix.row,
                system_matrix.col,
                current,
                system_matrix.data,
                index_type(system_matrix.nnz),
                index_type(system_matrix.shape[0]),
                phi,
                index_type(phi.shape[1]),
                index_type(phi.strides[0] // phi.itemsize),
                index_type(phi.strides[1] // phi.itemsize),
                alpha,
            ),
        )
