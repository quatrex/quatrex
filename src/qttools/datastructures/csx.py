# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Extended Compressed Sparse Row (CSX) format with stack support."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.kernels import inplace


class CSX:
    """Extended Compressed Sparse Row (CSX) format with
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

        # Addition cache. Used to cache to avoid repeated finding of the
        # indices for the addition operation.
        self._add_cache: dict[int, NDArray] = {}

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

    def _to_dense(self) -> NDArray:
        """Returns the unsymmetrized dense array.

        Warning
        -------
        This is purely for testing purposes and should not be used in
        production code.

        Note
        ----
        This also unsymmetrizes the matrix if it is symmetric.

        """
        dense = xp.zeros(
            self.local_stack_shape + (self.cols, self.cols), dtype=self.dtype
        )
        for idx in np.ndindex(self.local_stack_shape):
            data = self.data[idx]
            tmp = sparse.csr_matrix(
                (data, self.col_ind, self.row_ptr), shape=(self.rows, self.cols)
            ).toarray()

            if self.symmetry is not None:
                tmp += xp.triu(symmetry_ops[self.symmetry](tmp), k=1).transpose()

            dense[idx] = tmp

        return dense

    def toarray(self) -> NDArray:
        """Returns the local dense array.

        Note
        ----
        This is made to match the `scipy.sparse` API. It does not
        perform any communication.

        Returns
        -------
        NDArray
            The local dense array.

        """
        dense = xp.zeros(
            self.local_stack_shape + (self.cols, self.cols), dtype=self.dtype
        )
        for idx in np.ndindex(self.local_stack_shape):
            data = self.data[idx]
            tmp = sparse.csr_matrix(
                (data, self.col_ind, self.row_ptr), shape=(self.rows, self.cols)
            ).toarray()

            dense[idx] = tmp

        return dense

    def expand_symmetry(
        self,
    ) -> "CSX":
        if self.symmetry is None:
            raise ValueError("Symmetrization is only relevant for symmetric matrices.")

        indices = self.col_ind != self.row_ind
        new_row_ind = [self.row_ind] + [self.col_ind[indices]]
        new_col_ind = [self.col_ind] + [self.row_ind[indices]]
        new_data = [self.data] + [symmetry_ops[self.symmetry](self.data[..., indices])]

        new_row_ind = xp.concatenate(new_row_ind, axis=-1)
        new_col_ind = xp.concatenate(new_col_ind, axis=-1)
        new_data = xp.concatenate(new_data, axis=-1)

        # Sort the indices and data to get the canonical format
        flat_idx = new_row_ind * self.cols + new_col_ind
        sort_idx = xp.argsort(flat_idx)
        new_row_ind = new_row_ind[sort_idx]
        new_col_ind = new_col_ind[sort_idx]
        new_data = new_data[..., sort_idx]

        # TODO: Do not get the row_ptr from scipy
        # We are not passing in the real data since it is higher
        # dimensional.
        coo = sparse.coo_matrix(
            (xp.ones_like(new_row_ind, dtype=bool), (new_row_ind, new_col_ind)),
            shape=(self.rows, self.cols),
            copy=False,
        )
        new_row_ptr = coo.tocsr().indptr

        dcsx = CSX(
            dtype=self.dtype,
            rows=self.rows,
            cols=self.cols,
            local_stack_shape=self.local_stack_shape,
            row_ptr=new_row_ptr,
            row_ind=new_row_ind,
            col_ind=new_col_ind,
        )
        dcsx.allocate_data()
        dcsx.data = new_data

        return dcsx

    def _get_update_indices(self, other: "CSX") -> NDArray:
        """Returns the indices of `self` that correspond to the entries
        of `other`.

        Parameters
        ----------
        other : CSX
            The CSX matrix whose entries we want to find in `self`.

        Returns
        -------
        NDArray
            The indices of `self` that correspond to the entries of
            `other`.

        """
        # flatten row cols into a sortable integer key
        other_keys = other.row_ind * self.cols + other.col_ind
        self_keys = self.row_ind * self.cols + self.col_ind

        update_indices = np.searchsorted(self_keys, other_keys)

        if not np.array_equal(self_keys[update_indices], other_keys):
            raise AssertionError("`other` has entries not present in `self`.")

        return update_indices

    def add_(
        self,
        other: "CSX",
        prefactor: int | float | np.number = 1.0,
        cache: bool = True,
        cache_id: int | None = None,
    ) -> None:
        """Adds another CSX matrix to this one in place.

        Note
        ----
        This method assumes that the sparsity pattern of `other` is a
        subset of the sparsity pattern of `self`. If this is not the
        case, a ValueError will be raised.

        Parameters
        ----------
        other : CSX
            The CSX matrix to add to this one.
        prefactor : int | float | np.number, optional
            A prefactor to multiply `other` by before adding. Default is
            1.0.
        cache : bool, optional
            Whether to cache the indices for the addition operation.
            Default is True.
        cache_id : int | None, optional
            An optional cache ID to use for caching the indices. If
            None, the ID of `other` will be used. Default is None.

        """
        if self.symmetry is not None and other.symmetry is None:
            raise ValueError("Cannot add a non-symmetric matrix to a symmetric matrix.")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                "The shapes of the two matrices must be the same for addition."
            )
        if self.local_stack_shape != other.local_stack_shape:
            raise ValueError(
                "The local stack shapes of the two matrices must be the same for addition."
            )
        if self._data is None:
            raise ValueError("Self data has not been allocated yet.")
        if other._data is None:
            raise ValueError("Other data has not been allocated yet.")

        # In the case of adding a symmetric matrix to a non-symmetric
        # one, we expand the symmetry of the symmetric matrix to match
        # the non-symmetric one.
        if other.symmetry is not None and self.symmetry is None:
            other = other.expand_symmetry()

        if cache_id is None:
            cache_id = id(other)

        # NOTE: We assume that the sparsity of `other` is a subset of
        # the sparsity of `self`.
        if cache and cache_id in self._add_cache:
            update_indices = self._add_cache[cache_id]
        else:
            update_indices = self._get_update_indices(other)
            if cache:
                self._add_cache[cache_id] = update_indices

        # TODO: Update the kernel for higher dimensions
        for stack_index in np.ndindex(self.local_stack_shape):
            inplace.scatter_add_scaled(
                self._data[stack_index],
                other._data[stack_index],
                update_indices,
                prefactor,
                False,
            )

    @classmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        local_stack_shape: tuple,
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] = xp.complex128,
        allocate: bool = True,
    ) -> "CSX":
        """Allocates a CSX matrix from a sparse array.

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
            The sparse array to convert to CSX format.
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
        CSX
            The CSX matrix.

        """

        coo = sparray.tocoo()

        index_dtype = coo.col.dtype

        rows = xp.array([coo.shape[0]], dtype=index_dtype)
        cols = xp.array([coo.shape[1]], dtype=index_dtype)

        # NOTE: This is not necessary since the inputs should already be
        # upper if needed.
        if symmetry:
            coo = sparse.triu(coo, format="coo")

        # Canonicalizes the COO format.
        if not coo.has_canonical_format:
            coo.sum_duplicates()

        if not coo.has_canonical_format:
            raise ValueError("COO format is not canonical.")

        row_ind = coo.row
        col_ind = coo.col
        # TODO: Do not get the row_ptr from scipy
        row_ptr = coo.tocsr().indptr

        dcsx = cls(
            dtype=dtype,
            rows=int(rows[0]),
            cols=int(cols[0]),
            local_stack_shape=local_stack_shape,
            row_ptr=row_ptr,
            row_ind=row_ind,
            col_ind=col_ind,
            symmetry=symmetry,
        )

        if allocate:
            dcsx.allocate_data()
            dcsx.data = coo.data

        return dcsx
