# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from typing import List

from qttools import NDArray
from qttools.utils.gpu_utils import xp

from quatrex.exciton.response.comm import fetch_overlaping_data, find_overlaping_data


def calc_four_point_correlation_distributed(
    GG_local: NDArray,
    GL_local: NDArray,
    G_energies: NDArray,
    G_nnz_section_offsets: List[int],
    G_rows: List[int],
    G_cols: List[int],
    G_bandwidth: int,
    L_nen: int,
    L_step_E: int,
    rank: int,
):
    nnz_to_fetch, nnz_rank = find_overlaping_data(
        G_nnz_section_offsets,
        G_bandwidth,
        G_rows,
        G_cols,
    )
    gg_recv = fetch_overlaping_data(
        nnz_to_fetch,
        nnz_rank,
        GG_local,
        G_nnz_section_offsets,
        tag=0,
    )
    gl_recv = fetch_overlaping_data(
        nnz_to_fetch,
        nnz_rank,
        GL_local,
        G_nnz_section_offsets,
        tag=1,
    )
    extended_local_GG = xp.concatenate([GG_local, gg_recv], axis=-1)
    extended_local_GL = xp.concatenate([GL_local, gl_recv], axis=-1)
    extended_local_G_rows = xp.concatenate(
        [
            G_rows[G_nnz_section_offsets[rank] : G_nnz_section_offsets[rank + 1]],
            G_rows[nnz_to_fetch[rank]],
        ],
        axis=-1,
    )
    extended_local_G_cols = xp.concatenate(
        [
            G_cols[G_nnz_section_offsets[rank] : G_nnz_section_offsets[rank + 1]],
            G_cols[nnz_to_fetch[rank]],
        ],
        axis=-1,
    )
    L_nnz = estimate_L_nnz(extended_local_G_rows, extended_local_G_cols)
    prefactor = -1j / xp.pi * (G_energies[1] - G_energies[0])  # equispaced energies
    return four_point_correlation(
        extended_local_GG,
        extended_local_GL,
        extended_local_G_rows,
        extended_local_G_cols,
        L_nnz,
        L_nen,
        L_step_E,
        prefactor,
    )


def estimate_L_nnz(
    G_rows: List[int],
    G_cols: List[int],
):
    """Estimate the number of non-zero elements in the four-point correlation function."""
    G_nnz = len(G_rows)
    G_bandwidth = xp.max(xp.abs(xp.array(G_rows) - xp.array(G_cols))) + 1
    L_nnz = G_nnz * G_bandwidth * G_bandwidth * 4
    return L_nnz


def four_point_correlation(
    GG: NDArray,
    GL: NDArray,
    G_rows: List[int],
    G_cols: List[int],
    L_nnz: int,
    L_nen: int,
    L_step_E: int,
    prefactor,
):
    """Computes the four-point correlation function.
    This function computes the four-point correlation function
    using the element-wise product of the two-point correlation
    functions GG and GL. The correlation is computed using
    the FFT convolution method. The flipping of the
    second function in convolution is done in the Fourier space, by
    taking its conjugate.
    Parameters
    ----------
    GG : NDArray
        Two-point Green's function, last dimension is energy, first dimension is space.
    GL : NDArray
        Two-point Green's function, last dimension is energy, first dimension is space.
    Returns
    -------
    NDArray
        Four-point correlation function, last dimension is energy, first dimension is space.
    """
    G_nen = GG.shape[-1]
    n = G_nen + G_nen - 1
    G_nnz = len(G_rows)
    assert G_nnz == len(G_cols)
    assert GG.shape[0] == GL.shape[0]

    LG = xp.zeros((L_nnz, L_nen), dtype=GG.dtype)
    LL = xp.zeros((L_nnz, L_nen), dtype=GG.dtype)
    L_rows = xp.zeros(L_nnz, dtype=int)
    L_cols = xp.zeros(L_nnz, dtype=int)

    GG_fft = xp.fft.fftn(GG, (n,), axes=(-1,))

    L_inz = 0
    for inz in range(G_nnz):
        i = G_rows[inz]
        j = G_cols[inz]

        for jnz in range(G_nnz):
            k = G_rows[jnz]
            L = G_cols[jnz]

            ind1 = find_index(G_rows, G_cols, L, j)
            ind2 = find_index(G_rows, G_cols, i, k)

            if ind1 == -1 or ind2 == -1:
                continue

            GL_fft = xp.fft.fftn(GL[ind1], (n,), axes=(-1,))
            L_fft = prefactor * xp.multiply(GG_fft[ind2], GL_fft.conj())
            L_t = xp.fft.ifftn(L_fft)
            LG[L_inz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

            GL_fft = xp.fft.fftn(GL[ind2], (n,), axes=(-1,))
            L_fft = prefactor * xp.multiply(GL_fft, GG_fft[ind1].conj())
            L_t = xp.fft.ifftn(L_fft)
            LL[L_inz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

            L_rows[L_inz] = inz
            L_cols[L_inz] = jnz
            L_inz += 1

    return LG[:L_inz], LL[:L_inz], L_rows[:L_inz], L_cols[:L_inz]


def find_index(
    rows: NDArray,
    cols: NDArray,
    row: int,
    col: int,
):
    """Finds the index of a given row and column in the rows and columns arrays."""
    cond1 = xp.where(rows == row)[0]
    cond2 = xp.where(cols == col)[0]
    cond = xp.all(cond1, cond2)
    if cond.size == 0:
        return -1
    if cond.size == 1:
        return cond[0]
    if cond.size > 1:
        raise ValueError("Multiple indices found for the given row and column.")
