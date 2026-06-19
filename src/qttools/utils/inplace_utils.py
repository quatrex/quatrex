# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np

from qttools import NDArray, sparse, xp


def compute_update_indices_sparse(
    M: sparse.csr_matrix, U: sparse.csr_matrix, destination_indexes: NDArray = None
) -> NDArray:
    """Computes the indices for updating the system matrix.

    Parameters
    ----------
    M : sparse.csr_matrix
        The original system matrix.
    U : sparse.csr_matrix
        The update matrix to be applied.
    destination_indexes : NDArray
        The indices in the system matrix where the update should be applied.

    Returns
    -------
    target_indices : NDArray
        The indices in the flattened system matrix corresponding to the
        update positions.

    """

    # Get the CPU versions of M and U
    M = M.get() if hasattr(M, "get") else M
    U = U.get() if hasattr(U, "get") else U

    # Default destination indexes to identity mapping
    if destination_indexes is None:
        destination_indexes = np.arange(M.shape[0], dtype=xp.int64)

    if np.unique(destination_indexes).size != destination_indexes.size:
        raise ValueError(
            "The destination indexes have duplicate entries, cannot compute update indices."
        )

    update_indices = np.zeros_like(U.data, dtype=xp.int64)

    # Iterate over rows of U
    for U_row in range(U.shape[0]):

        # Get the column indices for the current row of U
        row_start = U.indptr[U_row]
        row_end = U.indptr[U_row + 1]
        U_cols = U.indices[row_start:row_end]

        # Get the corresponding row in M
        M_row = destination_indexes[U_row]

        # Get the column indices for the current row of M
        M_row_start = M.indptr[M_row]
        M_row_end = M.indptr[M_row + 1]
        M_cols = M.indices[M_row_start:M_row_end]

        # Check for duplicate column indices in the system matrix row
        if np.unique(M_cols).size != M_cols.size:
            raise ValueError(
                "The system matrix has duplicate column indices in a row, cannot compute update indices."
            )

        # Map U column indices to destination indexes in M
        U_cols_dest = destination_indexes[U_cols]

        # Map U column indices to M column indices
        M_ind_map = np.searchsorted(M_cols, U_cols_dest)
        if (M_cols[M_ind_map] != U_cols_dest).any():
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )
        if np.unique(M_ind_map).size != U_cols_dest.size:
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )

        update_indices[row_start:row_end] = M_ind_map + M_row_start

    return xp.array(update_indices)


def compute_update_indices_dense(
    M: sparse.csr_matrix, destination_indexes: NDArray = None
) -> NDArray:
    """Computes the indices for updating the system matrix.

    Parameters
    ----------
    M : sparse.csr_matrix
        The original system matrix.
    destination_indexes : NDArray
        The indices in the system matrix where the update should be applied.

    Returns
    -------
    target_indices : NDArray
        The indices in the flattened system matrix corresponding to the
        update positions.

    """

    # Get the CPU version of M
    M = M.get() if hasattr(M, "get") else M

    # Default destination indexes to identity mapping
    if destination_indexes is None:
        destination_indexes = np.arange(M.shape[0], dtype=xp.int64)

    if np.unique(destination_indexes).size != destination_indexes.size:
        raise ValueError(
            "The destination indexes have duplicate entries, cannot compute update indices."
        )

    U_size = destination_indexes.shape[0]

    update_indices = np.zeros((U_size**2,), dtype=xp.int64)

    for U_row in range(U_size):

        # Get the corresponding row in M
        M_row = destination_indexes[U_row]
        M_row_start = M.indptr[M_row]
        M_row_end = M.indptr[M_row + 1]
        M_cols = M.indices[M_row_start:M_row_end]

        if np.unique(M_cols).size != M_cols.size:
            raise ValueError(
                "The system matrix has duplicate column indices in a row, cannot compute update indices."
            )

        M_ind_map = np.searchsorted(M_cols, destination_indexes)
        if np.unique(M_ind_map).size != destination_indexes.size:
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )
        if (M_cols[M_ind_map] != destination_indexes).any():
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )

        update_indices[U_row * U_size : (U_row + 1) * U_size] = M_ind_map + M_row_start

    return xp.array(update_indices)
