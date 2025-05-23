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
    """Calculates the four-point correlation function in a distributed manner.

    Arguments
    ---------
    GG_local : NDArray
        Local two-point Green's function. first dimension is energy, last dimension is space.
    GL_local : NDArray
        Local two-point Green's function. first dimension is energy, last dimension is space.
    G_energies : NDArray
        Energies of the two-point Green's function.
    G_nnz_section_offsets : List[int]
        Offsets of the sections in the global data array.
    G_rows : List[int]
        Rows of the two-point Green's function (in the global matrix).
    G_cols : List[int]
        Columns of the two-point Green's function (in the global matrix).
    G_bandwidth : int
        Bandwidth of the two-point Green's function.
    L_nen : int
        Number of energies in the four-point correlation function.
    L_step_E : int
        Step size in the energies of the four-point correlation function.
    rank : int
        Rank of the current process.

    Returns
    -------
    NDArray
        Four-point correlation function. The first dimension is space, the last dimension is energy.
    """
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
    extended_local_G_indices = xp.concatenate(
        [
            xp.arange(G_nnz_section_offsets[rank], G_nnz_section_offsets[rank + 1]),
            nnz_to_fetch[rank],
        ],
    )
    L_nnz = estimate_L_nnz(G_rows, G_cols, extended_local_G_indices)
    prefactor = -1j / xp.pi * (G_energies[1] - G_energies[0])  # equispaced energies
    # swapping axes to have the energy dimension last. Not sure if it's faster in FFT.
    return four_point_correlation(
        extended_local_GG.swapaxes(0, -1),
        extended_local_GL.swapaxes(0, -1),
        G_rows,
        G_cols,
        extended_local_G_indices,
        L_nnz,
        L_nen,
        L_step_E,
        prefactor,
    )


def estimate_L_nnz(
    G_rows: List[int],
    G_cols: List[int],
    extended_local_G_indices: List[int],
):
    """Estimate the number of non-zero elements in the four-point correlation function that should be computed on this rank."""
    local_G_nnz = len(extended_local_G_indices)
    G_bandwidth = xp.max(xp.abs(xp.array(G_rows) - xp.array(G_cols))) + 1
    L_nnz = local_G_nnz * G_bandwidth * G_bandwidth * 4
    return L_nnz


def four_point_correlation(
    GG: NDArray,
    GL: NDArray,
    G_rows: List[int],
    G_cols: List[int],
    G_indices: List[int],
    L_rows: List[int],
    L_cols: List[int],
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
    G_rows : List[int]
        Rows of the two-point Green's function (in the global matrix).
    G_cols : List[int]
        Columns of the two-point Green's function (in the global matrix).
    G_indices : List[int]
        Indices of the two-point Green's function available on this rank.
    L_nnz : int
        Estimated number of non-zero elements in the four-point correlation function.
    L_nen : int
        Number of energies in the four-point correlation function.
    L_step_E : int
        Step size in the energies of the four-point correlation function.
    prefactor : float
        Prefactor for the four-point correlation function.

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
    L_nnz = len(L_rows)
    assert L_nnz == len(L_cols)

    LG = xp.zeros((L_nnz, L_nen), dtype=GG.dtype)
    LL = xp.zeros((L_nnz, L_nen), dtype=GG.dtype)

    GG_fft = xp.fft.fftn(GG, (n,), axes=(-1,))
    GL_fft = xp.fft.fftn(GL, (n,), axes=(-1,))

    for L_inz, (ii, jj) in enumerate(zip(L_rows, L_cols)):
        G_ind1 = xp.where(G_indices == ii)[0]
        G_ind2 = xp.where(G_indices == jj)[0]
        i = G_rows[G_ind1]
        j = G_cols[G_ind1]
        k = G_rows[G_ind2]
        L = G_cols[G_ind2]

        ind1 = find_index(G_rows, G_cols, L, j)
        ind2 = find_index(G_rows, G_cols, i, k)

        L_fft = prefactor * xp.multiply(GG_fft[ind2], GL_fft[ind1].conj())
        L_t = xp.fft.ifftn(L_fft)
        LG[L_inz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

        L_fft = prefactor * xp.multiply(GL_fft[ind2], GG_fft[ind1].conj())
        L_t = xp.fft.ifftn(L_fft)
        LL[L_inz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

    return (LG, LL, L_rows, L_cols)


def find_index(
    rows: NDArray,
    cols: NDArray,
    row: int,
    col: int,
):
    """Finds the index of a given row and column in the rows and columns arrays."""
    cond = xp.where((rows == row) & (cols == col))[0]
    if cond.size == 0:
        return -1
    if cond.size == 1:
        return cond[0]
    if cond.size > 1:
        raise ValueError("Multiple indices found for the given row and column.")
