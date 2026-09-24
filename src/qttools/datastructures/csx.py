# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Extended Compressed Sparse Row (CSX) format with stack support."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.kernels import inplace


def _make_canonical_coo(
    row_ind: NDArray,
    col_ind: NDArray,
    data: NDArray,
    cols: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Returns the canonical COO format of the given indices and data.

    Parameters
    ----------
    row_ind : NDArray
        The row indices of the COO format.
    col_ind : NDArray
        The column indices of the COO format.
    data : NDArray
        The data of the COO format.
    cols : int
        The number of columns in the matrix.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The canonical COO format of the given indices and data.

    """
    # Sort the indices and data to get the canonical format
    flat_idx = row_ind * cols + col_ind
    sort_idx = xp.argsort(flat_idx)
    row_ind = row_ind[sort_idx]
    col_ind = col_ind[sort_idx]
    data = data[..., sort_idx]

    return row_ind, col_ind, data


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
        row_ind: NDArray,
        col_ind: NDArray,
        symmetry: str | None = None,
        row_ptr: NDArray | None = None,
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

        # NOTE: We currently do not populate the `row_ptr` array. This
        # is because we do not use it in any of the operations.
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
            tmp = sparse.coo_matrix(
                (data, (self.row_ind, self.col_ind)), shape=(self.rows, self.cols)
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
            self.local_stack_shape + (self.rows, self.cols), dtype=self.dtype
        )
        for idx in np.ndindex(self.local_stack_shape):
            data = self.data[idx]
            tmp = sparse.coo_matrix(
                (data, (self.row_ind, self.col_ind)), shape=(self.rows, self.cols)
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

        new_row_ind, new_col_ind, new_data = _make_canonical_coo(
            row_ind=new_row_ind,
            col_ind=new_col_ind,
            data=new_data,
            cols=self.cols,
        )

        csx = CSX(
            dtype=self.dtype,
            rows=self.rows,
            cols=self.cols,
            local_stack_shape=self.local_stack_shape,
            row_ind=new_row_ind,
            col_ind=new_col_ind,
        )
        csx.allocate_data()
        csx.data = new_data

        return csx

    def _get_update_indices(
        self,
        row_ind: NDArray,
        col_ind: NDArray,
    ) -> NDArray:
        """Returns the indices of `self` that correspond to the entries
        of `other`.

        Parameters
        ----------
        row_ind : NDArray
            The row indices of `other`.
        col_ind : NDArray
            The column indices of `other`.

        Returns
        -------
        NDArray
            The indices of `self` that correspond to the entries of
            `other`.

        """
        # flatten row cols into a sortable integer key
        other_keys = row_ind * self.cols + col_ind
        self_keys = self.row_ind * self.cols + self.col_ind

        update_indices = xp.searchsorted(self_keys, other_keys)

        if not xp.array_equal(self_keys[update_indices], other_keys):
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
            update_indices = self._get_update_indices(other.row_ind, other.col_ind)
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

    def multiply_(
        self,
        other: NDArray,
    ):
        """Multiplies the matrix by a 1D or 2D array in place.

        Note
        ----
        The multiplication with a 1D array scales the columns of the
        matrix, while the multiplication with a 2D array scales the
        rows of the matrix.

        Parameters
        ----------
        other : NDArray
            The array to multiply the matrix by. Can be either a 1D
            array of shape (cols,) or a 2D array of shape (rows, 1).

        """
        if other.ndim not in [1, 2]:
            raise ValueError("Other must be a 1D or 2D array.")

        if other.ndim == 2:
            if other.shape[1] != 1:
                raise ValueError("Other must be a 2D array with shape (N, 1).")
            if other.shape[0] != self.rows:
                raise ValueError("Other must have the same number of rows as self.")

            # scale rowwise
            other = xp.squeeze(other)
            self.data *= other[self.row_ind]

        else:
            if other.shape[0] != self.cols:
                raise ValueError("Other must have the same number of columns as self.")

            # scale columnwise
            self.data *= other[self.col_ind]

    def _get_tile(
        self,
        row_ind: NDArray,
        col_ind: NDArray,
        unsymmetrize: bool = False,
        _exclude_diagonal: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray, tuple[int, int]]:
        """Returns a tile of the matrix as COO format.

        Parameters
        ----------
        row_ind : NDArray
            The row indices of the tile.
        col_ind : NDArray
            The column indices of the tile.
        unsymmetrize : bool, optional
            Whether to unsymmetrize the tile if the matrix is symmetric.
        _exclude_diagonal : bool, optional
            Whether to exclude the diagonal entries from the tile. This
            is used internally when unsymmetrizing a symmetric matrix to
            avoid double counting the diagonal entries.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, tuple[int, int]]
            The data, row indices, column indices, and shape of the tile
            in COO format.

        """

        if unsymmetrize and self.symmetry is None:
            raise ValueError("Cannot unsymmetrize a non-symmetric matrix.")

        tile_shape = (len(row_ind), len(col_ind))

        row_lookup = xp.full(self.rows, -1, dtype=xp.intp)
        row_lookup[row_ind] = xp.arange(tile_shape[0])

        col_lookup = xp.full(self.cols, -1, dtype=xp.intp)
        col_lookup[col_ind] = xp.arange(tile_shape[1])

        new_row = row_lookup[self.row_ind]
        new_col = col_lookup[self.col_ind]

        mask = (new_row >= 0) & (new_col >= 0)
        if _exclude_diagonal:
            mask &= self.row_ind != self.col_ind

        tile_data = self.data[..., mask]
        tile_row = new_row[mask]
        tile_col = new_col[mask]

        # NOTE: This can only be entered if the matrix is symmetric and we
        # want to unsymmetrize it.
        if unsymmetrize:
            # NOTE: Small sanity check. This should never be triggered
            # since we check for this at the beginning of the function.
            if self.symmetry is None:
                raise ValueError("Cannot unsymmetrize a non-symmetric matrix.")

            # NOTE: This is a bit hacky. We switch both the
            # inputs/outputs to get the transpose of the tile. We strip
            # out the diagonal since it's already present in
            # `tile_data/row/col` above.
            tile_data_t, tile_col_t, tile_row_t, _ = self._get_tile(
                row_ind=col_ind,
                col_ind=row_ind,
                unsymmetrize=False,
                _exclude_diagonal=True,
            )
            tile_data_t = symmetry_ops[self.symmetry](tile_data_t)

            tile_data = xp.concatenate([tile_data, tile_data_t], axis=-1)
            tile_row = xp.concatenate([tile_row, tile_row_t])
            tile_col = xp.concatenate([tile_col, tile_col_t])

            tile_row, tile_col, tile_data = _make_canonical_coo(
                row_ind=tile_row,
                col_ind=tile_col,
                data=tile_data,
                cols=tile_shape[1],
            )

        return tile_data, tile_row, tile_col, tile_shape

    def get_tile(
        self,
        row_ind: NDArray | None = None,
        col_ind: NDArray | None = None,
        unsymmetrize: bool = False,
    ) -> "CSX":
        """Returns a tile of the matrix as a new CSX object.

        Parameters
        ----------
        row_ind : NDArray | None
            The row indices of the tile. If None, all rows are included.
        col_ind : NDArray | None
            The column indices of the tile. If None, all columns are included.
        unsymmetrize : bool, optional
            Whether to unsymmetrize the tile if the matrix is symmetric.

        Returns
        -------
        CSX
            The tile as a new CSX object.

        """
        if row_ind is None:
            row_ind = xp.arange(self.rows, dtype=self.index_type)
        if col_ind is None:
            col_ind = xp.arange(self.cols, dtype=self.index_type)

        if xp.min(row_ind) < 0 or xp.max(row_ind) >= self.rows:
            raise ValueError(
                f"Row indices {row_ind} are out of bounds for matrix with {self.rows} rows."
            )
        if xp.min(col_ind) < 0 or xp.max(col_ind) >= self.cols:
            raise ValueError(
                f"Column indices {col_ind} are out of bounds for matrix with {self.cols} columns."
            )

        tile_data, tile_row, tile_col, tile_shape = self._get_tile(
            row_ind=row_ind,
            col_ind=col_ind,
            unsymmetrize=unsymmetrize,
        )

        tile_csx = CSX(
            dtype=self.dtype,
            rows=tile_shape[0],
            cols=tile_shape[1],
            local_stack_shape=self.local_stack_shape,
            row_ind=tile_row,
            col_ind=tile_col,
        )

        tile_csx.allocate_data()
        tile_csx.data = tile_data

        return tile_csx

    @classmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        local_stack_shape: tuple,
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] | None = None,
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
        dtype : xp.dtype[xp.generic] | None, optional
            The data type of the matrix elements. Default is
            None and the data type of the input sparse array is used.
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

        csx = cls(
            dtype=coo.data.dtype if dtype is None else dtype,
            rows=int(rows[0]),
            cols=int(cols[0]),
            local_stack_shape=local_stack_shape,
            row_ind=row_ind,
            col_ind=col_ind,
            symmetry=symmetry,
        )

        if allocate:
            csx.allocate_data()
            csx.data = coo.data

        return csx
