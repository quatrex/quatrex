# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numba as nb
import numpy as np

@nb.njit(parallel=True, fastmath=True)
def compute_pair_sparsity_pattern_faster(
    rows: np.ndarray, cols: np.ndarray, dense: np.ndarray
) -> tuple:
   """Computes the sparsity pattern for a pair-interaction matrix A(a,b,c,d) flattened
   into a COO matrix by combining first two and last two index.

   Parameters
   ----------
   rows : NDArray
      The rows of the interaction matrix.
   cols : NDArray
      The columns of the interaction matrix.
   dense : NDArray
      The dense matrix of the interaction matrix.
   Returns
   -------
   NDArray
      The pair-interaction operator sparsity pattern in a dense bool array.
   """
   nnz = rows.shape[0]
   dense_pair = np.zeros((nnz, nnz), dtype=np.bool)
   for i, (a, b) in enumerate(zip(rows, cols)):
      for j, (c, d) in enumerate(zip(rows, cols)):
         dense_pair[i, j] = (dense[a, c] != 0 
                             and dense[b, d] != 0 
                             and dense[a, d] != 0
                             and dense[b, c] != 0
         )
   return dense_pair


def compute_pair_sparsity_pattern(
    rows: np.ndarray, cols: np.ndarray,
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
   for i,(a,b) in enumerate(zip(rows, cols)):
      for j,(c,d) in enumerate(zip(rows, cols)):
         ind1 = np.where((rows == a) & (cols == c))[0]
         ind2 = np.where((rows == b) & (cols == d))[0]
         ind3 = np.where((rows == a) & (cols == d))[0]
         ind4 = np.where((rows == b) & (cols == c))[0]
         if ind1.size > 0 and ind2.size > 0 and ind3.size > 0 and ind4.size > 0:
            pair_cols.append(i)
            pair_rows.append(j)
         
   return (np.array(pair_rows), np.array(pair_cols))
