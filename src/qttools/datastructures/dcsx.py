# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Distributed Extended Compressed Sparse Row (CSX) format with stack support."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.utils.gpu_utils import get_host


class DCSX:
    """Distributed Extended Compressed Sparse Row (CSX) format with
    stack support.

    Parameters
    ----------
    dtype : xp.dtype[xp.generic]
        Data type of the matrix elements.
    rows : int
        Number of rows in the local matrix.
    cols : int
        Number of columns in the local matrix.
    row_offsets : NDArray
        Array of cumulative row counts across all ranks in the block
        communicator.
    local_stack_shape : tuple[int, ...]
        Shape of the local stack for this rank.
    row_ptr : NDArray
        Row pointer array for the local matrix in CSR format.
    row_ind : NDArray
        Row indices for the local matrix in COO format. Added for the
        extended CSX format.
    col_ind : NDArray
        Column indices for the local matrix in CSR format.
    symmetry : str | None, optional
        The symmetry of the matrix. This can be "symmetric",
        "hermitian", "skew-symmetric", "skew-hermitian", or None.
        Default is None.

    """

    def __init__(
        self,
        dtype: xp.dtype[xp.generic],
        rows: int,
        cols: int,
        row_offsets: NDArray,
        local_stack_shape: tuple[int, ...],
        row_ptr: NDArray,
        row_ind: NDArray,
        col_ind: NDArray,
        symmetry: str | None = None,
    ):

        rows = int(rows)
        cols = int(cols)

        # Type of the data
        self.dtype = dtype
        # Type of the indices
        self.index_type = col_ind.dtype
        # Distribution of rows
        self.row_offsets = row_offsets
        self.local_stack_shape = local_stack_shape

        # TODO: Unify the naming between the data structures. In
        # DSDBSparse, `rows` and `cols` refer to the `row_ind` and
        # `col_ind` arrays.
        self.rows = rows
        self.cols = cols

        self.shape = self.local_stack_shape + (self.rows, self.cols)

        self.row_ptr = row_ptr
        self.row_ind = row_ind
        self.col_ind = col_ind
        self._data = None

        self.nnz = len(self.col_ind)

        self.symmetry = symmetry

    def get_tile(self, rows, cols):
        pass

    @property
    def data(self) -> NDArray:
        """Returns the local data."""
        if self._data is None:
            raise ValueError("Data has not been allocated yet.")
        return self._data

    @data.setter
    def data(self, value: NDArray) -> None:
        """Sets the local data."""
        if self._data is None:
            raise ValueError("Data has not been allocated yet.")
        self._data[...] = value

    def allocate_data(self) -> None:
        """Allocates the local data array."""
        if self._data is not None:
            raise ValueError("Data has already been allocated.")
        self._data = xp.zeros(self.local_stack_shape + (self.nnz,), dtype=self.dtype)

    def to_dense(self) -> NDArray:
        """Returns the local dense array.

        Warning
        -------
        This is purely for testing purposes and should not be used in
        production code.

        """
        dense = xp.zeros(
            self.local_stack_shape + (self.cols, self.cols), dtype=self.dtype
        )
        for idx in np.ndindex(self.local_stack_shape):
            data = self.data[idx]
            tmp = sparse.csr_matrix(
                (data, self.col_ind, self.row_ptr), shape=(self.rows, self.cols)
            ).toarray()

            tmp = comm.block.all_gather_v(tmp, axis=0)

            if self.symmetry is not None:
                tmp += xp.triu(symmetry_ops[self.symmetry](tmp), k=1).transpose()

            dense[idx] = tmp

        return dense

    @classmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        local_stack_shape: tuple,
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] = xp.complex128,
        allocate: bool = True,
    ) -> "DCSX":
        """Allocates a DCSX matrix from a sparse array.

        Note
        ----
        This assumes that the input sparse array is already distributed
        correctly.

        Note
        ----
        For the QTBM datastructure, distribution of the stack is handled
        outside since the datastructure does not need to operator
        through the stack.

        Parameters
        ----------
        sparray : sparse.spmatrix
            The sparse array to convert to DCSX format.
        local_stack_shape : tuple
            The shape of the local stack for this rank.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian", or None.
            Default is None.
        dtype : xp.dtype[xp.generic], optional
            The data type of the matrix elements. Default is
            xp.complex128.
        allocate : bool, optional
            Whether to allocate the data array. Default is True.

        Returns
        -------
        DCSX
            The DCSX matrix.

        """

        if comm.stack is None or comm.block is None:
            raise ValueError("Communicators must be initialized.")

        coo = sparray.tocoo()

        index_dtype = coo.col.dtype

        rows = xp.array([coo.shape[0]], dtype=index_dtype)
        cols = xp.array([coo.shape[1]], dtype=index_dtype)

        all_rows = xp.zeros((comm.block.size), dtype=index_dtype)
        all_cols = xp.zeros((comm.block.size), dtype=index_dtype)
        comm.block.all_gather(rows, all_rows)
        comm.block.all_gather(cols, all_cols)

        # Check that the number of columns are the same across all block comm ranks.
        if not xp.all(all_cols == all_cols[0]):
            raise ValueError("The number of columns must be the same across all ranks.")

        # Check that the sum of all rows are the same as the number of columns.
        # This means the matrix is split across the rows.
        if not xp.sum(all_rows) == all_cols[0]:
            raise ValueError(
                "The sum of all rows must be the same as the number of columns."
            )

        row_offsets = xp.zeros((comm.block.size + 1), dtype=index_dtype)
        comm.block.all_gather(rows, row_offsets[1:])
        row_offsets = get_host(xp.cumsum(row_offsets))

        # NOTE: This is not necessary since the inputs should already be
        # upper if needed.
        if symmetry:
            coo = sparse.triu(coo, format="coo", k=row_offsets[comm.block.rank])

        # Canonicalizes the COO format.
        if not coo.has_canonical_format:
            coo.sum_duplicates()

        if not coo.has_canonical_format:
            raise ValueError("COO format is not canonical.")

        row_ind = coo.row
        col_ind = coo.col
        row_ptr = coo.tocsr().indptr

        dsdbcoo = cls(
            dtype=dtype,
            rows=int(rows[0]),
            cols=int(cols[0]),
            row_offsets=row_offsets,
            local_stack_shape=local_stack_shape,
            row_ptr=row_ptr,
            row_ind=row_ind,
            col_ind=col_ind,
            symmetry=symmetry,
        )

        if allocate:
            dsdbcoo.allocate_data()
            dsdbcoo.data = coo.data

        return dsdbcoo
