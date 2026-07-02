# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes our Numba inplace kernels."""

import math

import numba as nb
import numpy as np
from scipy import sparse

from qttools import NDArray


@nb.njit(parallel=True)
def scatter_add_scaled(
    a: NDArray, b: NDArray, inds: NDArray, alpha: np.complex128, conjugate: bool
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

    """
    for i in nb.prange(inds.shape[0]):
        if conjugate:
            a[inds[i]] += alpha * np.conj(b[i])
        else:
            a[inds[i]] += alpha * b[i]


@nb.njit(parallel=True)
def scatter_add_scaled_obc(
    a: NDArray,
    b: NDArray,
    inds: NDArray,
    k: tuple[float, float],
    transverse_repetition_grid: tuple[int, int],
    alpha: np.complex128,
):
    """Adds array `b` to array `a` at indices `ind` in-place with OBC repetitions.

    Parameters
    ----------
    a : NDArray
        The array to be updated.
    b : NDArray
        The array to be added from `a`.
    inds : NDArray
        The indices at which to subtract `b` from `a`. The indices are
        assumed unique. If there are duplicates, the behavior is
        undefined due to potential race conditions.
    k : tuple[float, float]
        The transverse wavevector components.
    transverse_repetition_grid : tuple[int, int]
        The transverse repetition grid of the contact.
    alpha : np.complex128
        The scaling factor.

    """
    ky, kz = k
    ny, nz = transverse_repetition_grid

    N_S = b.shape[1]
    N_S_big = N_S * ny * nz
    num_inds = inds.shape[0]

    b = b.reshape(-1)

    for i in nb.prange(num_inds):
        i_S_big = i // N_S_big
        j_S_big = i % N_S_big

        i_S = i_S_big % N_S
        j_S = j_S_big % N_S

        i_cell = i_S_big // N_S
        j_cell = j_S_big // N_S

        cell_rep_1_i = i_cell // nz
        cell_rep_2_i = i_cell % nz

        cell_rep_1_j = j_cell // nz
        cell_rep_2_j = j_cell % nz

        phase_1 = -ky * (cell_rep_1_j - cell_rep_1_i)
        phase_2 = -kz * (cell_rep_2_j - cell_rep_2_i)
        total_phase = phase_1 + phase_2

        c = math.cos(total_phase)
        s = math.sin(total_phase)
        s_idx = i_S * N_S + j_S

        # Potential race if ind has duplicates.
        a[inds[i]] += b[s_idx] * (c + 1j * s) * alpha


@nb.njit(parallel=True)
def _add_bond_resolved_current_csr(indptr, indices, current, system_matrix, phi, alpha):
    N = indptr.shape[0] - 1
    m = phi.shape[1]

    for i in nb.prange(N):
        for j in range(indptr[i], indptr[i + 1]):
            row = i
            col = indices[j]
            system_matrix_ab = system_matrix[j]
            current_ab = current[j]

            out = 0.0 + 0.0j

            for k in range(m):
                phi_a = phi[row, k]
                phi_b = phi[col, k]

                out += phi_a.conjugate() * phi_b * system_matrix_ab

            current[j] = current_ab + (alpha * out).imag


@nb.njit(parallel=True)
def _add_bond_resolved_current_coo(rows, cols, current, system_matrix, phi, alpha):
    nnz = rows.shape[0]
    m = phi.shape[1]

    for i in nb.prange(nnz):
        row = rows[i]
        col = cols[i]
        system_matrix_ab = system_matrix[i]
        current_ab = current[i]

        out = 0.0 + 0.0j

        for j in range(m):
            phi_a = phi[row, j]
            phi_b = phi[col, j]

            out += phi_a.conjugate() * phi_b * system_matrix_ab

        current[i] = current_ab + (alpha * out).imag


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

    if isinstance(system_matrix, sparse.csr_matrix):

        _add_bond_resolved_current_csr(
            system_matrix.indptr,
            system_matrix.indices,
            current,
            system_matrix.data,
            phi,
            alpha,
        )

    elif isinstance(system_matrix, sparse.coo_matrix):
        _add_bond_resolved_current_coo(
            system_matrix.row,
            system_matrix.col,
            current,
            system_matrix.data,
            phi,
            alpha,
        )
