from mpi4py.MPI import COMM_WORLD as comm

from qttools import sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.gpu_utils import ArrayLike
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import (
    compute_block_sort_index,
    sparsity_pattern_of_product,
)


class DSBCOO(DSBSparse):
    """Distributed stack of sparse matrices in coordinate format.

    This DSBSparse implementation stores the matrix sparsity pattern in
    probably the most straight-forward way: as a list of coordinates.
    Both data and coordinates are sorted by block-row and -column.

    Parameters
    ----------
    data : array_like
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    rows : array_like
        The row indices.
    cols : array_like
        The column indices.
    block_sizes : array_like
        The size of each block in the sparse matrix.
    global_stack_shape : tuple or int
        The global shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.

    """

    def __init__(
        self,
        data: ArrayLike,
        rows: ArrayLike,
        cols: ArrayLike,
        block_sizes: ArrayLike,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DBCOO matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.rows = xp.asarray(rows).astype(int)
        self.cols = xp.asarray(cols).astype(int)

        # Since the data is block-wise contiguous, we can cache block
        # *slices* for faster access.
        self._block_slice_cache = {}

    def _get_items(
        self, stack_index: tuple, rows: xp.ndarray, cols: xp.ndarray
    ) -> ArrayLike:
        """Gets the requested items from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the __getitem__
        method. The index is assumed to already be renormalized.

        If we are in the "stack" distribution state, you will get an
        array of the expected shape, padded with zeros where requested
        elements are not in the matrix.

        If we are in the "nnz" distribution state, and you are
        requesting an element that is not in the matrix, an IndexError
        is raised. If we are in the "nnz" distribution state, you will
        get the requested elements that are on the current rank, and an
        empty array on the ranks that hold none of the requested
        elements.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        rows : int | array_like
            The row indices of the items.
        cols : int | array_like
            The column indices of the items.

        Returns
        -------
        items : array_like
            The requested items.

        """
        inds, value_inds = (
            (self.rows[:, xp.newaxis] == rows) & (self.cols[:, xp.newaxis] == cols)
        ).nonzero()

        data_stack = self.data[*stack_index]

        if self.distribution_state == "stack":
            if len(inds) != rows.size:
                arr = xp.zeros(data_stack.shape[:-1] + (rows.size,), dtype=self.dtype)
                arr[..., value_inds] = data_stack[..., inds]
                return xp.squeeze(arr)

            return xp.squeeze(data_stack[..., inds])

        if len(inds) != rows.size:
            # We cannot know which rank is supposed to hold an element
            # that is not in the matrix, so we raise an error.
            raise IndexError("Requested element not in matrix.")

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        ranks = (self.nnz_section_offsets <= inds[:, xp.newaxis]).sum(-1) - 1

        return data_stack[
            ..., inds[ranks == comm.rank] - self.nnz_section_offsets[comm.rank]
        ]

    def _set_items(
        self, stack_index: tuple, rows: xp.ndarray, cols: xp.ndarray, value: ArrayLike
    ) -> None:
        """Sets the requested items in the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the __setitem__
        method. The index is assumed to already be renormalized.

        If we are in the "stack" distribution state, you need to provide
        an array of the expected shape. The sparsity pattern is not
        modified.

        If we are in the "nnz" distribution state, you need to provide
        the values that are on the current rank. The sparsity pattern is
        not modified.

        In both cases, if you are trying to set a value that is not in
        the matrix, nothing happens.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        rows : int | array_like
            The row indices of the items.
        cols : int | array_like
            The column indices of the items.
        values : array_like
            The values to set.

        """
        inds, value_inds = (
            (self.rows[:, xp.newaxis] == rows) & (self.cols[:, xp.newaxis] == cols)
        ).nonzero()

        if len(inds) == 0:
            # Nothing to do if the element is not in the matrix.
            return

        value = xp.asarray(value)
        if self.distribution_state == "stack":
            if value.ndim == 0:
                self.data[*stack_index][..., inds] = value
                return

            self.data[*stack_index][..., inds] = value[..., value_inds]
            return

        # If nnz are distributed accross the stack, we need to find the
        # rank that holds the data.
        ranks = (self.nnz_section_offsets <= inds[:, xp.newaxis]).sum(-1) - 1

        stack_padding_inds = self._stack_padding_mask.nonzero()[0][stack_index[0]]
        stack_inds, nnz_inds = xp.ix_(
            stack_padding_inds,
            inds[ranks == comm.rank] - self.nnz_section_offsets[comm.rank],
        )
        # We need to access the full data buffer directly to set the
        # value since we are using advanced indexing.
        if value.ndim == 0:
            self._data[stack_inds, stack_index[1:] or Ellipsis, nnz_inds] = value
            return

        self._data[stack_inds, stack_index[1:] or Ellipsis, nnz_inds] = value[
            ..., value_inds[ranks == comm.rank] - value_inds[ranks == comm.rank][0]
        ]
        return

    def _get_block_slice(self, row, col):
        """Gets the slice of data corresponding to a given block.

        This also handles the block slice cache. If there is no data in
        the block, an `slice(None)` is cached.

        Parameters
        ----------
        row : int
            Row index of the block.
        col : int
            Column index of the block.

        Returns
        -------
        block_slice : slice
            The slice of the data corresponding to the block.

        """
        block_slice = self._block_slice_cache.get((row, col), None)

        if block_slice is None:
            # Cache miss, compute the slice.
            mask = (
                (self.rows >= self.block_offsets[row])
                & (self.rows < self.block_offsets[row + 1])
                & (self.cols >= self.block_offsets[col])
                & (self.cols < self.block_offsets[col + 1])
            )
            inds = mask.nonzero()[0]
            if len(inds) == 0:
                # No data in this block, cache an empty slice.
                block_slice = slice(None)
            else:
                # NOTE: The data is sorted by block-row and -column, so
                # we can safely assume that the block is contiguous.
                block_slice = slice(inds[0], inds[-1] + 1)

        self._block_slice_cache[(row, col)] = block_slice
        return block_slice

    def _get_block(self, stack_index: tuple, row: int, col: int) -> ArrayLike:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index of the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.

        Returns
        -------
        block : array_like
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        data_stack = self.data[*stack_index]
        block_slice = self._get_block_slice(row, col)

        if not self.return_dense:
            if block_slice == slice(None):
                # No data in this block, return an empty block.
                return xp.empty(0), xp.empty(0), xp.empty(data_stack.shape[:-1] + (0,))

            rows = self.rows[block_slice] - self.block_offsets[row]
            cols = self.cols[block_slice] - self.block_offsets[col]
            return rows, cols, data_stack[..., block_slice]

        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.block_sizes[row]), int(self.block_sizes[col])),
            dtype=self.dtype,
        )
        if block_slice == slice(None):
            # No data in this block, return an empty block.
            return block

        block[
            ...,
            self.rows[block_slice] - self.block_offsets[row],
            self.cols[block_slice] - self.block_offsets[col],
        ] = data_stack[..., block_slice]

        return block

    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: ArrayLike
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index of the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        block : array_like
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        block_slice = self._get_block_slice(row, col)
        if block_slice == slice(None):
            # No data in this block, nothing to do.
            return

        self.data[*stack_index][..., block_slice] = block[
            ...,
            self.rows[block_slice] - self.block_offsets[row],
            self.cols[block_slice] - self.block_offsets[col],
        ]

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBCOO):
            raise TypeError("Can only add DSBCOO matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")

        if xp.any(self.rows != other.rows):
            raise ValueError("Row indices do not match.")

        if xp.any(self.cols != other.cols):
            raise ValueError("Column indices do not match.")

    def __neg__(self) -> "DSBCOO":
        """Negation of the data."""
        return DSBCOO(
            data=-self.data,
            rows=self.rows,
            cols=self.cols,
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
            return_dense=self.return_dense,
        )

    def __matmul__(self, other: "DSBSparse") -> None:
        """Matrix multiplication of two DSBSparse matrices."""
        if not isinstance(other, DSBSparse):
            if sparse.isspmatrix(other):
                raise NotImplementedError(
                    "Matrix multiplication with sparse matrices  is not implemented."
                )
            raise TypeError("Can only multiply DSBSparse matrices.")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("Matrix shapes do not match.")
        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")
        product_rows, product_cols = sparsity_pattern_of_product(
            (
                sparse.coo_matrix(
                    (xp.ones(self.nnz), (self.rows, self.cols)), shape=self.shape[-2:]
                ),
                sparse.coo_matrix(
                    (xp.ones(other.nnz), (other.rows, other.cols)),
                    shape=other.shape[-2:],
                ),
            )
        )
        block_sort_index = compute_block_sort_index(
            product_rows, product_cols, self.block_sizes
        )
        product = DSBCOO(
            data=xp.zeros(self.stack_shape + (product_rows.size,), dtype=self.dtype),
            rows=product_rows[block_sort_index],
            cols=product_cols[block_sort_index],
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
        )
        for stack_index in xp.ndindex(self.data.shape[:-1]):
            temp_product = sparse.csr_matrix(
                (self.data[stack_index], (self.rows, self.cols)), shape=self.shape[-2:]
            ) @ sparse.csr_matrix(
                (other.data[stack_index], (other.rows, other.cols)),
                shape=other.shape[-2:],
            )
            product.data[stack_index, :] = temp_product[product.rows, product.cols]
        return product

    @DSBSparse.block_sizes.setter
    def block_sizes(self, block_sizes: ArrayLike) -> None:
        """Sets new block sizes for the matrix.

        Parameters
        ----------
        block_sizes : array_like
            The new block sizes.

        """
        if sum(block_sizes) != self.shape[-1]:
            raise ValueError("Block sizes must sum to matrix shape.")
        self._block_sizes = xp.asarray(block_sizes).astype(int)
        self.block_offsets = xp.hstack(([0], xp.cumsum(self._block_sizes)))
        self.num_blocks = len(self.block_sizes)
        # Compute the block-sorting index.
        block_sort_index = compute_block_sort_index(
            self.rows, self.cols, self._block_sizes
        )
        self.data[..., :] = self.data[..., block_sort_index]
        self.rows = self.rows[block_sort_index]
        self.cols = self.cols[block_sort_index]

    def ltranspose(self, copy=False) -> "None | DSBCOO":
        """Performs a local transposition of the matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a new object. Default is False.

        Returns
        -------
        None | DSBCOO
            The transposed matrix. If copy is False, this is None.

        """

        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

        if copy:
            self = DSBCOO(
                self.data.copy(),
                self.rows.copy(),
                self.cols.copy(),
                self.block_sizes,
            )

        if not (
            hasattr(self, "_inds_bcoo2bcoo_t")
            and hasattr(self, "_rows_t")
            and hasattr(self, "_cols_t")
            and hasattr(self, "_block_slice_cache_t")
        ):
            # Transpose.
            rows_t, cols_t = self.cols, self.rows

            # Canonical ordering of the transpose.
            inds_bcoo2canonical_t = xp.lexsort(xp.vstack((cols_t, rows_t)))
            canonical_rows_t = rows_t[inds_bcoo2canonical_t]
            canonical_cols_t = cols_t[inds_bcoo2canonical_t]

            # Compute index for sorting the transpose by block.
            inds_canonical2bcoo_t = compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcoo2bcoo_t = inds_bcoo2canonical_t[inds_canonical2bcoo_t]

            # Cache the necessary objects.
            self._inds_bcoo2bcoo_t = inds_bcoo2bcoo_t
            self._rows_t = rows_t[self._inds_bcoo2bcoo_t]
            self._cols_t = cols_t[self._inds_bcoo2bcoo_t]

            self._block_slice_cache_t = {}

        self.data[:] = self.data[..., self._inds_bcoo2bcoo_t]
        self._inds_bcoo2bcoo_t = xp.argsort(self._inds_bcoo2bcoo_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rows, self._rows_t = self._rows_t, self.rows

        self._block_slice_cache, self._block_slice_cache_t = (
            self._block_slice_cache_t,
            self._block_slice_cache,
        )

    def spy(self) -> tuple[ArrayLike, ArrayLike]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Returns
        -------
        rows : array_like
            Row indices of the non-zero elements.
        cols : array_like
            Column indices of the non-zero elements.

        """
        return self.rows, self.cols

    @classmethod
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: ArrayLike,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBCOO":
        """Creates a new DSBSparse matrix from a scipy.sparse array.

        Parameters
        ----------
        arr : sparse.sparray
            The sparse array to convert.
        block_sizes : array_like
            The size of all the blocks in the matrix.
        global_stack_shape : tuple
            The global shape of the stack of matrices. The provided
            sparse matrix is replicated across the stack.
        densify_blocks : list[tuple], optional
            List of matrix blocks to densify. Default is None. This is
            useful to densify the boundary blocks of the matrix
        pinned : bool, optional
            Whether to pin the memory when using GPU. Default is False.

        Returns
        -------
        DSBCOO
            The new DSBCOO matrix.

        """
        # We only distribute the first dimension of the stack.
        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]

        coo: sparse.coo_matrix = arr.tocoo().copy()

        num_blocks = len(block_sizes)
        block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

        # Densify the selected blocks.
        for i, j in densify_blocks or []:
            # Unsign the block indices.
            i = num_blocks + i if i < 0 else i
            j = num_blocks + j if j < 0 else j
            if not (0 <= i < num_blocks and 0 <= j < num_blocks):
                raise IndexError("Block index out of bounds.")

            indices = [
                (m + block_offsets[i], n + block_offsets[j])
                for m, n in xp.ndindex(int(block_sizes[i]), int(block_sizes[j]))
            ]
            coo.row = xp.append(coo.row, [m for m, __ in indices]).astype(xp.int32)
            coo.col = xp.append(coo.col, [n for __, n in indices]).astype(xp.int32)
            coo.data = xp.append(coo.data, xp.zeros(len(indices), dtype=coo.data.dtype))

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Compute the rowptr map.
        block_sort_index = compute_block_sort_index(coo.row, coo.col, block_sizes)

        data = xp.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        data[..., :] = coo.data[block_sort_index]
        rows = coo.row[block_sort_index]
        cols = coo.col[block_sort_index]

        return cls(
            data=data,
            rows=rows,
            cols=cols,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
    
    def calc_reduce_to_mask(self, rows: ArrayLike, cols: ArrayLike, block_sizes: ArrayLike) -> ArrayLike:
        """Calculate the mask for reducing the matrix to the given rows and columns.

        Parameters
        ----------
        rows : array_like
            The rows to keep.
        cols : array_like
            The columns to keep.
        block_sizes : array_like
            The size of the blocks in the new matrix.

        Returns
        -------
        array_like
            The mask for reducing the matrix.

        """
        mask = xp.zeros(self.nnz, dtype=bool)
        block_sort_index = compute_block_sort_index(
            self.rows, self.cols, block_sizes
        )
        j = 0
        for i, ii in enumerate(block_sort_index):
            if self.rows[ii] == rows[j] and self.cols[ii] == cols[j]:
                mask[i] = True
                j += 1
        return block_sort_index[mask]
        

    def reduce_to(
        self, rows: ArrayLike, cols: ArrayLike, block_sizes: ArrayLike
    ) -> "DSBCOO":
        """Create a reduced matrix to the given rows and columns.

        Parameters
        ----------
        rows : array_like
            The rows to keep.
        cols : array_like
            The columns to keep.
        block_sizes : array_like
            The size of the blocks in the new matrix.

        Returns
        -------
        DSBCOO
            The reduced matrix.

        """
        mask = self.calc_reduce_to_mask(rows, cols, block_sizes)
        return DSBCOO(
            data=self.data[..., mask],
            rows=self.rows[mask],
            cols=self.cols[mask],
            block_sizes=block_sizes,
            global_stack_shape=self.global_stack_shape,
        )
