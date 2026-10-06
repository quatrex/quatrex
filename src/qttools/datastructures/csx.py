# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Extended Compressed Sparse Row (CSX) format with stack support."""

from __future__ import annotations

import numpy as np
from scipy.sparse import get_index_dtype

from qttools import NDArray, sparse, xp
from qttools.datastructures.csx_routines import make_canonical_coo
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.kernels import inplace
from qttools.utils.gpu_utils import free_mempool, get_pointer


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

        self._row_ptr = row_ptr
        self._row_ind = row_ind
        self._col_ind = col_ind
        self._data = None

        self.nnz = len(self.col_ind)

        self.symmetry = symmetry

        # Addition cache. Used to cache to avoid repeated finding of the
        # indices for the addition operation.
        self._add_cache: dict[int, NDArray] = {}

    # NOTE: We do not want to allow to overwrite the pointers to the
    # data arrays. This is because we want to manage the memory
    # ourselves.

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
            # NOTE: This we explicitly manage ourself.
            raise ValueError("Data has not been allocated yet.")
        self._data[...] = value

    @property
    def row_ptr(self) -> NDArray:
        """Returns the local row pointer array."""
        if self._row_ptr is None:
            raise ValueError("Row pointer has not been allocated yet.")
        return self._row_ptr

    @row_ptr.setter
    def row_ptr(self, value: NDArray) -> None:
        """Sets the local row pointer array."""
        if self._row_ptr is None:
            self._row_ptr = xp.empty(self.rows + 1, dtype=self.index_type)
        self._row_ptr[...] = value

    @property
    def row_ind(self) -> NDArray:
        """Returns the local row indices array."""
        if self._row_ind is None:
            raise ValueError("Row indices have not been allocated yet.")
        return self._row_ind

    @row_ind.setter
    def row_ind(self, value: NDArray) -> None:
        """Sets the local row indices array."""
        if self._row_ind is None:
            self._row_ind = xp.empty(self.nnz, dtype=self.index_type)
        self._row_ind[...] = value

    @property
    def col_ind(self) -> NDArray:
        """Returns the local column indices array."""
        if self._col_ind is None:
            raise ValueError("Column indices have not been allocated yet.")
        return self._col_ind

    @col_ind.setter
    def col_ind(self, value: NDArray) -> None:
        """Sets the local column indices array."""
        if self._col_ind is None:
            self._col_ind = xp.empty(self.nnz, dtype=self.index_type)
        self._col_ind[...] = value

    def allocate_data(self) -> None:
        """Allocates the local data array."""
        if self._data is None:
            self._data = xp.zeros(
                self.local_stack_shape + (self.nnz,), dtype=self.dtype
            )

    def free_data(self) -> None:
        """Frees the local data."""
        self._data = None
        free_mempool()

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
                (data, (self.row_ind, self.col_ind)),
                shape=(self.rows, self.cols),
                copy=False,
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
                (data, (self.row_ind, self.col_ind)),
                shape=(self.rows, self.cols),
                copy=False,
            ).toarray()

            dense[idx] = tmp

        return dense

    def tocoo(self) -> sparse.coo_matrix:
        """Returns the local matrix in COO format.

        Note
        ----
        This is made to match the `scipy.sparse` API. It does not
        perform any communication.

        Note
        ----
        Only possible with a non-stacked CSX. Could be amended by
        returning a list of COO matrices.

        Returns
        -------
        sparse.coo_matrix
            The local matrix in COO format.

        """
        if self.local_stack_shape != ():
            raise ValueError("Cannot convert a stacked CSX to COO.")

        out = sparse.coo_matrix(
            (self.data, (self.row_ind, self.col_ind)),
            shape=(self.rows, self.cols),
            copy=False,
        )
        # Check that pointers are the same.
        if get_pointer(out.row) != get_pointer(self.row_ind):
            raise ValueError(
                "The row indices of the COO matrix "
                "do not match the row indices of the CSX matrix."
            )
        if get_pointer(out.col) != get_pointer(self.col_ind):
            raise ValueError(
                "The column indices of the COO matrix "
                "do not match the column indices of the CSX matrix."
            )
        if get_pointer(out.data) != get_pointer(self.data):
            raise ValueError(
                "The data of the COO matrix "
                "does not match the data of the CSX matrix."
            )

        return out

    def tocsr(self) -> sparse.csr_matrix:
        """Returns the local matrix in CSR format.

        Note
        ----
        This is made to match the `scipy.sparse` API. It does not
        perform any communication.

        Note
        ----
        Only possible with a non-stacked CSX. Could be amended by
        returning a list of COO matrices.

        Returns
        -------
        sparse.csr_matrix
            The local matrix in CSR format.

        """
        if self.local_stack_shape != ():
            raise ValueError("Cannot convert a stacked CSX to CSR.")

        if self._row_ptr is None:
            self._row_ptr = xp.searchsorted(
                self.row_ind, xp.arange(self.rows + 1)
            ).astype(self.index_type)

        out = sparse.csr_matrix(
            (self.data, self.col_ind, self.row_ptr),
            shape=(self.rows, self.cols),
            copy=False,
        )

        # Check that pointers are the same. This is needed for the solvers.
        if get_pointer(out.indptr) != get_pointer(self.row_ptr):
            raise ValueError(
                "The row pointer of the CSR matrix "
                "does not match the row pointer of the CSX matrix."
            )
        if get_pointer(out.indices) != get_pointer(self.col_ind):
            raise ValueError(
                "The column indices of the CSR matrix "
                "do not match the column indices of the CSX matrix."
            )
        if get_pointer(out.data) != get_pointer(self.data):
            raise ValueError(
                "The data of the CSR matrix "
                "does not match the data of the CSX matrix."
            )

        return out

    def expand_symmetry(
        self,
    ) -> CSX:
        """Symmetrizes the CSX matrix. This returns a new CSX matrix
        that is the symmetrized version of the original matrix.

        Returns
        -------
        CSX
            The symmetrized CSX matrix.

        """
        if self.symmetry is None:
            raise ValueError("Symmetrization is only relevant for symmetric matrices.")

        indices = self.col_ind != self.row_ind
        new_row_ind = [self.row_ind] + [self.col_ind[indices]]
        new_col_ind = [self.col_ind] + [self.row_ind[indices]]
        new_data = [self.data] + [symmetry_ops[self.symmetry](self.data[..., indices])]

        nnz = np.sum([len(ind) for ind in new_row_ind])
        index_type = get_index_dtype(maxval=nnz)
        new_row_ind = [ind.astype(index_type) for ind in new_row_ind]
        new_col_ind = [ind.astype(index_type) for ind in new_col_ind]

        new_row_ind = xp.concatenate(new_row_ind, axis=-1, dtype=index_type)
        new_col_ind = xp.concatenate(new_col_ind, axis=-1, dtype=index_type)
        new_data = xp.concatenate(new_data, axis=-1)

        new_row_ind, new_col_ind, sort_idx = make_canonical_coo(
            row_ind=new_row_ind,
            col_ind=new_col_ind,
        )
        new_data = new_data[..., sort_idx]

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
        r_self = self.row_ind.astype(xp.int64)
        c_self = self.col_ind.astype(xp.int64)

        r_other = row_ind.astype(xp.int64)
        c_other = col_ind.astype(xp.int64)

        self_keys = (r_self << 32) | (c_self & 0xFFFFFFFF)
        other_keys = (r_other << 32) | (c_other & 0xFFFFFFFF)

        update_indices = xp.searchsorted(self_keys, other_keys).astype(self.index_type)

        if not xp.array_equal(self_keys[update_indices], other_keys):
            raise AssertionError("`other` has entries not present in `self`.")

        return update_indices

    def add_(
        self,
        other: CSX,
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

        Warning
        -------
        We do not check if both matrices have the same symmetry. This is
        because we want to allow partial addition of matrices with
        different symmetries. The user is responsible for ensuring that
        the addition is valid.

        Warning
        -------
        We allow the addition of a non-symmetric matrix to a symmetric
        one as long as the sparsity pattern match. This again is to
        allow for partial addition of matrices that result in a
        symmetric matrix. The user is responsible for ensuring that the
        addition is valid.

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

        if other.symmetry is not None and self.symmetry is None:
            raise ValueError("Cannot add a symmetric matrix to a non-symmetric one.")

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
            other = xp.squeeze(other, axis=1)
            self.data *= other[self.row_ind]

        else:
            if other.shape[0] != self.cols:
                raise ValueError("Other must have the same number of columns as self.")

            # scale columnwise
            self.data *= other[self.col_ind]

    def __matmul__(self, other: NDArray) -> NDArray:
        """Matrix multiplication with a 1D or 2D array.

        Parameters
        ----------
        other : NDArray
            The array to multiply the matrix by. Can be either a 1D
            array of shape (cols,) or a 2D array of shape (cols, N).

        Returns
        -------
        NDArray
            The result of the matrix multiplication. Will have the shape
            `self.local_stack_shape + (self.rows,) + other.shape[1:]`.

        """
        if self.symmetry is not None:
            raise ValueError("Cannot multiply a symmetric matrix. Unsymmetrize first.")

        if other.ndim not in [1, 2]:
            raise ValueError("Other must be a 1D or 2D array.")

        # TODO: Allow for more general shapes of `other` and implement
        # the full matrix multiplication.

        # SPMV
        if other.ndim == 1 and other.shape[0] != self.cols:
            raise ValueError("Other must have the same number of columns as self.")

        # SPMM
        if other.ndim == 2 and other.shape[0] != self.cols:
            raise ValueError("Other must have the same number of columns as self.")

        if self._row_ptr is None:
            self._row_ptr = xp.searchsorted(
                self.row_ind, xp.arange(self.rows + 1)
            ).astype(self.index_type)

        out = xp.empty(
            self.local_stack_shape + (self.rows,) + other.shape[1:], dtype=self.dtype
        )

        for idx in np.ndindex(self.local_stack_shape):
            # TODO: Not the most efficient pipeline.
            _mat = sparse.csr_matrix(
                (self.data[idx], self.col_ind, self.row_ptr),
                shape=(self.rows, self.cols),
                copy=False,
            )
            # Check that pointers are the same. This is needed for the solvers.
            if get_pointer(_mat.indptr) != get_pointer(self.row_ptr):
                raise ValueError(
                    "The row pointer of the CSR matrix "
                    "does not match the row pointer of the CSX matrix."
                )
            if get_pointer(_mat.indices) != get_pointer(self.col_ind):
                raise ValueError(
                    "The column indices of the CSR matrix "
                    "do not match the column indices of the CSX matrix."
                )
            # NOTE: scipy somehow copies the data buffer here. I am not
            # sure why.
            out[idx] = _mat.dot(other)

        return out

    def transpose(self) -> CSX:
        """Returns the transpose of the matrix as a new CSX object.

        Returns
        -------
        CSX
            The transpose of the matrix.

        """
        if self.symmetry is not None:
            raise ValueError("Cannot transpose a symmetric matrix.")

        csx = CSX(
            dtype=self.dtype,
            rows=self.cols,
            cols=self.rows,
            local_stack_shape=self.local_stack_shape,
            row_ind=self.col_ind,
            col_ind=self.row_ind,
            symmetry=self.symmetry,
        )
        csx.allocate_data()
        csx.data = self.data

        return csx

    def conjugate(self) -> CSX:
        """Returns the conjugate of the matrix as a new CSX object.

        Returns
        -------
        CSX
            The conjugate of the matrix.

        """
        csx = CSX(
            dtype=self.dtype,
            rows=self.rows,
            cols=self.cols,
            local_stack_shape=self.local_stack_shape,
            row_ind=self.row_ind,
            col_ind=self.col_ind,
            symmetry=self.symmetry,
        )
        csx.allocate_data()
        csx.data = xp.conj(self.data)

        return csx

    def _get_tile(
        self,
        row_ind: NDArray,
        col_ind: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, tuple[int, int]]:
        """Returns a tile of the matrix as COO format.

        Parameters
        ----------
        row_ind : NDArray
            The row indices of the tile.
        col_ind : NDArray
            The column indices of the tile.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, tuple[int, int]]
            The data, row indices, column indices, and shape of the tile
            in COO format.

        """

        if self.symmetry is not None:
            raise ValueError("Cannot get a tile of a symmetric matrix.")

        tile_shape = (len(row_ind), len(col_ind))

        row_lookup = xp.full(self.rows, -1, dtype=self.index_type)
        row_lookup[row_ind] = xp.arange(tile_shape[0])

        col_lookup = xp.full(self.cols, -1, dtype=self.index_type)
        col_lookup[col_ind] = xp.arange(tile_shape[1])

        new_row = row_lookup[self.row_ind]
        new_col = col_lookup[self.col_ind]

        mask = (new_row >= 0) & (new_col >= 0)

        tile_data = self.data[..., mask]
        tile_row = new_row[mask]
        tile_col = new_col[mask]

        return tile_data, tile_row, tile_col, tile_shape

    def get_tile(
        self,
        row_ind: NDArray | None = None,
        col_ind: NDArray | None = None,
    ) -> CSX:
        """Returns a tile of the matrix as a new CSX object.

        Note
        ----
        If the matrix is non-symmetric, this will return the tile
        unchanged. If the matrix is symmetric and `unsymmetrize` is
        True, the tile will be unsymmetrized if wanted.

        Parameters
        ----------
        row_ind : NDArray | None
            The row indices of the tile. If None, all rows are included.
        col_ind : NDArray | None
            The column indices of the tile. If None, all columns are
            included.

        Returns
        -------
        CSX
            The tile as a new CSX object.

        """
        if self.symmetry is not None:
            raise ValueError(
                "Cannot get a tile of a symmetric matrix. "
                "Unsymmetruize the matrix first using `expand_symmetry()`."
            )

        if row_ind is None:
            row_ind = xp.arange(self.rows, dtype=self.index_type)
        if col_ind is None:
            col_ind = xp.arange(self.cols, dtype=self.index_type)

        if len(row_ind) == 0 or len(col_ind) == 0:
            tile_csx = CSX(
                dtype=self.dtype,
                rows=len(row_ind),
                cols=len(col_ind),
                local_stack_shape=self.local_stack_shape,
                row_ind=xp.array([], dtype=self.index_type),
                col_ind=xp.array([], dtype=self.index_type),
            )
            tile_csx.allocate_data()
            return tile_csx

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
        sparray: sparse.spmatrix | None = None,
        row_ind: NDArray | None = None,
        col_ind: NDArray | None = None,
        shape: tuple[int, int] | None = None,
        local_stack_shape: tuple = tuple(),
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] | None = None,
        allocate: bool = True,
    ) -> CSX:
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
        sparray : sparse.spmatrix | None, optional
            The sparse array to convert to CSX format. If None,
            `row_ind` and `col_ind` must be provided.
        row_ind : NDArray | None, optional
            The row indices of the non-zero entries. If None, `sparray`
            must be provided.
        col_ind : NDArray | None, optional
            The column indices of the non-zero entries. If None,
            `sparray` must be provided.
        shape : tuple[int, int] | None, optional
            The shape of the matrix. If None, the shape of `sparray` is
            used. If None and `sparray` is None, this must be provided.
        local_stack_shape : tuple, optional
            The shape of the local stack for this rank. Default is an
            empty tuple, which means no stack.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian", or None.
            Default is None.
        dtype : xp.dtype[xp.generic] | None, optional
            The data type of the matrix elements. Default is None and
            the data type of the input sparse array is used.
        allocate : bool, optional
            Whether to allocate the data array. Default is True. Even if
            allocated, the data will be uninitialized.

        Returns
        -------
        CSX
            The CSX matrix.

        """
        if sparray is None and (row_ind is None or col_ind is None):
            raise ValueError(
                "Either a sparse array or row and column indices must be provided."
            )
        if sparray is not None and (row_ind is not None or col_ind is not None):
            raise ValueError(
                "Cannot provide both a sparse array and row/column indices."
            )

        if sparray is not None:
            sparray = sparray.tocoo()
            if shape is not None and shape != sparray.shape:
                raise ValueError(
                    f"Provided shape {shape} does not match the shape of the sparse array {sparray.shape}."
                )
            shape = sparray.shape
            index_dtype = sparray.col.dtype

            # Canonicalizes the COO format.
            if not sparray.has_canonical_format:
                sparray.sum_duplicates()
            if not sparray.has_canonical_format:
                raise ValueError("COO format is not canonical.")

            row_ind = sparray.row
            col_ind = sparray.col

            dtype = sparray.data.dtype if dtype is None else dtype

        else:
            if dtype is None:
                raise ValueError(
                    "Data type must be provided " "if no sparse array is given."
                )
            if shape is None:
                shape = (int(xp.max(row_ind)) + 1, int(xp.max(col_ind)) + 1)

            index_dtype = row_ind.dtype

        # Small sanity check. This should never be triggered since we
        # check for this at the beginning of the function.
        if row_ind is None or col_ind is None:
            raise ValueError("Both row_ind and col_ind must be provided.")

        rows = xp.array([shape[0]], dtype=index_dtype)
        cols = xp.array([shape[1]], dtype=index_dtype)

        # NOTE: This is not necessary since the inputs should already be
        # upper if needed.
        if symmetry:
            # Check that non lower triangular entries exists.
            if xp.any(row_ind > col_ind):
                raise ValueError(
                    "The input matrix is not upper triangular. "
                    "Cannot create a symmetric CSX matrix."
                )

        csx = cls(
            dtype=dtype,
            rows=int(rows[0]),
            cols=int(cols[0]),
            local_stack_shape=local_stack_shape,
            row_ind=row_ind,
            col_ind=col_ind,
            symmetry=symmetry,
        )

        if allocate:
            csx.allocate_data()
            if sparray is not None:
                csx.data = sparray.data

        return csx
