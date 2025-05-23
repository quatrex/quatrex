# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numba as nb
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm


@nb.njit(parallel=True, fastmath=True)
def compute_pair_sparsity_pattern_faster(
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Computes the sparsity pattern for a pair-interaction matrix A(a,b,c,d) flattened
    into a COO matrix by combining first two and last two index.

    Parameters
    ----------
    rows : NDArray
       The rows of the interaction matrix.
    cols : NDArray
       The columns of the interaction matrix.
    Returns
    -------
    NDArray
       The pair-interaction operator sparsity pattern in a dense bool array.
    """
    nnz = rows.shape[0]
    dense_pair = np.zeros((nnz, nnz), dtype=np.bool)
    for i, (a, b) in enumerate(zip(rows, cols)):
        for j, (c, d) in enumerate(zip(rows, cols)):
            ind1 = np.where((rows == a) & (cols == c))[0]
            ind2 = np.where((rows == b) & (cols == d))[0]
            ind3 = np.where((rows == a) & (cols == d))[0]
            ind4 = np.where((rows == b) & (cols == c))[0]
            dense_pair[i, j] = (
                ind1.size > 0 and ind2.size > 0 and ind3.size > 0 and ind4.size > 0
            )
    return dense_pair


@nb.njit(parallel=True, fastmath=True)
def compute_pair_sparsity_pattern_distributed(
    rows: np.ndarray, cols: np.ndarray, nnz_section_offsets: np.ndarray
) -> np.ndarray:
    """Computes the sparsity pattern for a pair-interaction matrix A(a,b,c,d) flattened
    into a COO matrix by combining first two and last two index.

    Parameters
    ----------
    rows : NDArray
       The rows of the interaction matrix.
    cols : NDArray
       The columns of the interaction matrix.
    nnz_section_offsets : NDArray[int]
       Offsets of the sections in the global data array.
    Returns
    -------
    NDArray
       The part of pair-interaction operator sparsity that is assigned to this rank
    """
    nnz = rows.shape[0]
    local_nnz = nnz_section_offsets[comm.rank + 1] - nnz_section_offsets[comm.rank]
    dense_pair = np.zeros((local_nnz, nnz), dtype=np.bool)
    offset = nnz_section_offsets[comm.rank]
    for i in range(offset, offset + local_nnz):
        ii = i - offset
        for j in range(nnz):
            a = rows[i]
            b = cols[i]
            c = rows[j]
            d = rows[j]
            ind1 = np.where((rows == a) & (cols == c))[0]
            ind2 = np.where((rows == b) & (cols == d))[0]
            ind3 = np.where((rows == a) & (cols == d))[0]
            ind4 = np.where((rows == b) & (cols == c))[0]
            dense_pair[ii, j] = (
                ind1.size > 0 and ind2.size > 0 and ind3.size > 0 and ind4.size > 0
            )
    return dense_pair


def compute_pair_sparsity_pattern(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple:
    """Computes the sparsity pattern for a pair-interaction matrix A(a,b,c,d) flattened
    into a COO matrix by combining first two and last two index.

    Parameters
    ----------
    rows : NDArray
       The rows of the interaction matrix.
    cols : NDArray
       The columns of the interaction matrix.
    Returns
    -------
    NDArray
       The pair-interaction operator sparsity pattern in rows and cols.

    """
    pair_cols = []
    pair_rows = []
    for i, (a, b) in enumerate(zip(rows, cols)):
        for j, (c, d) in enumerate(zip(rows, cols)):
            ind1 = np.where((rows == a) & (cols == c))[0]
            ind2 = np.where((rows == b) & (cols == d))[0]
            ind3 = np.where((rows == a) & (cols == d))[0]
            ind4 = np.where((rows == b) & (cols == c))[0]
            if ind1.size > 0 and ind2.size > 0 and ind3.size > 0 and ind4.size > 0:
                pair_cols.append(i)
                pair_rows.append(j)

    return (np.array(pair_rows), np.array(pair_cols))
