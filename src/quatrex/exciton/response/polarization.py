# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray
from qttools.kernels.datastructure.cupy.dsdbsparse import find_ranks
from qttools.utils.gpu_utils import xp

from quatrex.exciton.response.comm import fetch_overlaping_data


def calc_four_point_correlation_distributed(
    GG_local: NDArray,
    GL_local: NDArray,
    G_energies: NDArray,
    G_nnz_section_offsets: np.ndarray,
    G_rows: np.ndarray,
    G_cols: np.ndarray,
    L_rows: np.ndarray,
    L_cols: np.ndarray,
    L_nen: int,
    L_step_E: int,
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
        Global rows of the two-point Green's function.
    G_cols : List[int]
        Global columns of the two-point Green's function.
    L_rows : List[int]
        Rows of the four-point correlation function to compute on this rank (in the global matrix).
    L_cols : List[int]
        Columns of the four-point correlation function to compute on this rank (in the global matrix).
    L_nen : int
        Number of energies in the four-point correlation function.
    L_step_E : int
        Step size in the energies of the four-point correlation function.

    Returns
    -------
    NDArray
        Four-point correlation function. The first dimension is space, the last dimension is energy.
    """
    G_indices = xp.arange(
        G_nnz_section_offsets[comm.rank], G_nnz_section_offsets[comm.rank + 1]
    )
    nnz_to_fetch, nnz_rank = find_overlaping_data_for_L(
        G_rows,
        G_cols,
        G_nnz_section_offsets,
        L_rows,
        L_cols,
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
            G_indices,
            nnz_to_fetch[comm.rank],
        ],
    )

    prefactor = -1j / xp.pi * (G_energies[1] - G_energies[0])  # equispaced energies
    # swapping axes to have the energy dimension last. Not sure if it's needed for faster FFT.
    return four_point_correlation(
        extended_local_GG.swapaxes(0, -1),
        extended_local_GL.swapaxes(0, -1),
        G_rows,
        G_cols,
        extended_local_G_indices,
        L_rows,
        L_cols,
        L_nen,
        L_step_E,
        prefactor,
    )


def four_point_correlation(
    GG: NDArray,
    GL: NDArray,
    G_rows: np.ndarray,
    G_cols: np.ndarray,
    G_indices: np.ndarray,
    L_rows: np.ndarray,
    L_cols: np.ndarray,
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
        Global rows of the two-point Green's function (in the global matrix).
    G_cols : List[int]
        Global columns of the two-point Green's function (in the global matrix).
    G_indices: List[int]
        Indices of local G data
    L_rows : List[int]
        Rows of the four-point correlation function (in the global matrix).
    L_cols : List[int]
        Columns of the four-point correlation function (in the global matrix).
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
        i = G_rows[ii]
        j = G_cols[ii]
        k = G_rows[jj]
        L = G_cols[jj]

        G_inz = find_index(G_rows, G_cols, int(L), int(j))
        ind1 = xp.where(G_indices == G_inz)[0]
        G_inz = find_index(G_rows, G_cols, int(i), int(k))
        ind2 = xp.where(G_indices == G_inz)[0]

        L_fft = prefactor * xp.multiply(GG_fft[ind2], GL_fft[ind1].conj())
        L_t = xp.fft.ifftn(L_fft)
        LG[L_inz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

        L_fft = prefactor * xp.multiply(GL_fft[ind2], GG_fft[ind1].conj())
        L_t = xp.fft.ifftn(L_fft)
        LL[L_inz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

    return (LG, LL)


def find_index(
    rows: np.ndarray,
    cols: np.ndarray,
    row: int,
    col: int,
):
    """Finds the index of a given row and column in the rows and columns arrays."""
    cond = np.where((rows == row) & (cols == col))[0]
    if cond.size == 0:
        return -1
    if cond.size == 1:
        return cond[0]
    if cond.size > 1:
        raise ValueError("Multiple indices found for the given row and column.")


def find_overlaping_data_for_L(
    G_rows: np.ndarray,
    G_cols: np.ndarray,
    G_nnz_section_offsets: np.ndarray,
    L_rows: np.ndarray,
    L_cols: np.ndarray,
):
    nnz_to_fetch = []
    nnz_rank = []
    for ii, jj in zip(L_rows, L_cols):
        i = G_rows[ii]
        j = G_cols[ii]
        k = G_rows[jj]
        L = G_cols[jj]
        ind1 = find_index(G_rows, G_cols, int(L), int(j))
        ind2 = find_index(G_rows, G_cols, int(i), int(k))
        ind = [ind1, ind2]

        ind_on_rank = find_ranks(G_nnz_section_offsets, ind)

        not_on_rank = ind_on_rank != comm.rank

        nnz_to_fetch.append(ind[not_on_rank])
        nnz_rank.append(ind_on_rank[not_on_rank])

    nnz_to_fetch = np.array(nnz_to_fetch)
    nnz_rank = np.array(nnz_rank)

    unique_nnz_to_fetch, unique_indices = np.unique(nnz_to_fetch, return_index=True)
    unique_nnz_rank = nnz_rank[unique_indices]

    num_unique_nnz_to_fetch = len(unique_nnz_to_fetch)
    list_num_nnz_to_fetch = np.zeros((comm.size,), dtype=int)
    comm.Allgather(num_unique_nnz_to_fetch, list_num_nnz_to_fetch)
    list_nnz_to_fetch = np.array(np.sum(list_num_nnz_to_fetch), dtype=int)
    list_rank_to_fetch_from = np.array(np.sum(list_num_nnz_to_fetch), dtype=int)
    comm.Allgather(unique_nnz_to_fetch, list_nnz_to_fetch)
    comm.Allgather(unique_nnz_rank, list_rank_to_fetch_from)

    nnz_to_fetch = []
    nnz_rank = []
    offset = np.hstack(([0], np.cumsum(list_num_nnz_to_fetch)))
    for i in range(comm.size):
        nnz_to_fetch.append(list_nnz_to_fetch[offset[i] : offset[i + 1]])
        nnz_rank.append(list_rank_to_fetch_from[offset[i] : offset[i + 1]])

    return nnz_to_fetch, nnz_rank
