# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from mpi4py.MPI import COMM_WORLD as comm

from qttools import sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.gpu_utils import ArrayLike
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import (
    compute_block_sort_index,
    compute_ptr_map,
    product_sparsity_pattern,
)


class DSBCSR(DSBSparse):
    """Distributed stack of block-compressed sparse row matrices.

    This DSBSparse implementation uses a block-compressed sparse row
    format to store the sparsity pattern of the matrix. The data is
    sorted by block-row and -column. We use a row pointer map together
    with the column indices to access the blocks efficiently.

    Parameters
    ----------
    data : array_like
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    cols : array_like
        The column indices.
    rowptr_map : dict
        The row pointer map.
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
        cols: ArrayLike,
        rowptr_map: dict,
        block_sizes: ArrayLike,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DBCSR matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.cols = xp.asarray(cols).astype(int)
        self.rowptr_map = rowptr_map

    def _compute_indices(
        self, rows: xp.ndarray, cols: xp.ndarray
    ) -> tuple[xp.ndarray, ...]:
        """Computes the effective indices of the requested items.

        Parameters
        ----------
        rows : array_like
            The row indices of the items.
        cols : array_like
            The column indices of the items.

        Returns
        -------
        inds : array_like
            The indices of the requested items.
        value_inds : array_like
            The indices of the requested items in the value array.

        """
        # Ensure that the indices are at least 1-D arrays.
        rows = xp.atleast_1d(rows)
        cols = xp.atleast_1d(cols)

        brows = (self.block_offsets <= rows[:, xp.newaxis]).sum(-1) - 1
        bcols = (self.block_offsets <= cols[:, xp.newaxis]).sum(-1) - 1

        # Get an ordered list of unique blocks.
        unique_blocks = dict.fromkeys(zip(map(int, brows), map(int, bcols))).keys()
        rowptrs = [self.rowptr_map.get(bcoord, None) for bcoord in unique_blocks]

        inds, value_inds = [], []
        for (brow, bcol), rowptr in zip(unique_blocks, rowptrs):
            if rowptr is None:
                continue

            mask = (brows == brow) & (bcols == bcol)
            mask_inds = xp.where(mask)[0]

            # Renormalize the row indices for this block.
            rr = rows[mask] - self.block_offsets[brow]
            cc = cols[mask]

            # TODO: This could perhaps be done in an efficient way.
            for i, (r, c) in enumerate(zip(rr, cc)):
                ind = xp.where(self.cols[rowptr[r] : rowptr[r + 1]] == c)[0]

                if len(ind) == 0:
                    continue

                value_inds.append(mask_inds[i])
                inds.append(rowptr[r] + ind[0])

        return xp.array(inds, dtype=int), xp.array(value_inds, dtype=int)

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
        inds, value_inds = self._compute_indices(rows, cols)

        data_stack = self.data[stack_index]
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
        self, stack_index: tuple, rows: int | list, cols: int | list, value: ArrayLike
    ) -> None:
        """Sets the requested items in the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the __setitem__
        method. The index is assumed to already be renormalized.

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
        inds, value_inds = self._compute_indices(rows, cols)

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

    def _get_block(self, stack_index: tuple, row: int, col: int) -> ArrayLike:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index of the block in the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.

        Returns
        -------
        block : sparray | np.ndarray
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        data_stack = self.data[*stack_index]
        rowptr = self.rowptr_map.get((row, col), None)

        if not self.return_dense:
            if rowptr is None:
                # No data in this block, return zeros.
                return (
                    xp.zeros(int(self.block_sizes[row]) + 1),
                    xp.empty(0),
                    xp.empty(data_stack.shape[:-1] + (0,)),
                )

            cols = self.cols[rowptr[0] : rowptr[-1]] - self.block_offsets[col]
            return rowptr - rowptr[0], cols, data_stack[..., rowptr[0] : rowptr[-1]]

        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.block_sizes[row]), int(self.block_sizes[col])),
            dtype=self.dtype,
        )
        if rowptr is None:
            # No data in this block, return zeros.
            return block

        for i in range(int(self.block_sizes[row])):
            cols = self.cols[rowptr[i] : rowptr[i + 1]]
            block[..., i, cols - self.block_offsets[col]] = data_stack[
                ..., rowptr[i] : rowptr[i + 1]
            ]

        return block

    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: ArrayLike
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index of the block in the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        block : array_like
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        rowptr = self.rowptr_map.get((row, col), None)
        if rowptr is None:
            # No data in this block, nothing to do.
            return

        for i in range(int(self.block_sizes[row])):
            cols = self.cols[rowptr[i] : rowptr[i + 1]]
            self.data[*stack_index][..., rowptr[i] : rowptr[i + 1]] = block[
                ..., i, cols - self.block_offsets[col]
            ]

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBCSR):
            raise TypeError("Can only add DSBCSR matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")

        if self.rowptr_map.keys() != other.rowptr_map.keys():
            raise ValueError("Block sparsities do not match.")

        if xp.any(self.cols != other.cols):
            raise ValueError("Column indices do not match.")

    def __neg__(self) -> "DSBCSR":
        """Negation of the data."""
        return DSBCSR(
            data=-self.data,
            cols=self.cols,
            rowptr_map=self.rowptr_map,
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
        stack_indices = xp.ndindex(self.data.shape[:-1])
        product_rows, product_cols = product_sparsity_pattern(
            sparse.coo_matrix((xp.ones(self.nnz), (self.spy())), shape=self.shape[-2:]),
            sparse.coo_matrix(
                (xp.ones(other.nnz), (other.spy())), shape=other.shape[-2:]
            ),
        )
        block_sort_index = compute_block_sort_index(
            product_rows, product_cols, self.block_sizes
        )
        product = DSBCSR(
            data=xp.zeros(self.stack_shape + (product_rows.size,), dtype=self.dtype),
            cols=product_cols[block_sort_index],
            rowptr_map=compute_ptr_map(product_rows, product_cols, self.block_sizes),
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
        )
        for stack_index in stack_indices:
            temp_product = sparse.csr_matrix(
                (self.data[stack_index], (self.spy())), shape=self.shape[-2:]
            ) @ sparse.csr_matrix(
                (other.data[stack_index], (other.spy())), shape=other.shape[-2:]
            )
            product.data[stack_index, :] = temp_product[product.spy()]
        return product

    @DSBSparse.block_sizes.setter
    def block_sizes(self, block_sizes: ArrayLike) -> None:
        """Sets new block sizes for the matrix.
        Parameters
        ----------
        block_sizes : array_like
            The new block sizes.
        """
        if self.distribution_state == "nnz":
            raise NotImplementedError(
                "Cannot reassign block-sizes when distributed through nnz."
            )
        if sum(block_sizes) != self.shape[-1]:
            raise ValueError("Block sizes do not match matrix shape.")
        rows, cols = self.spy()
        # Compute canonical ordering of the matrix.
        inds_bcsr2canonical = xp.lexsort(xp.vstack((cols, rows)))
        canonical_rows = rows[inds_bcsr2canonical]
        canonical_cols = cols[inds_bcsr2canonical]
        # Compute the index for sorting by the new block-sizes.
        inds_canonical2bcsr = compute_block_sort_index(
            canonical_rows, canonical_cols, block_sizes
        )
        # Mapping directly from original block-ordering to the new
        # block-ordering is achieved by chaining the two mappings.
        inds_bcsr2bcsr = inds_bcsr2canonical[inds_canonical2bcsr]
        self.data[:] = self.data[..., inds_bcsr2bcsr]
        self.cols = self.cols[inds_bcsr2bcsr]
        # Compute the rowptr map for the new block-sizes.
        self.rowptr_map = compute_ptr_map(canonical_rows, canonical_cols, block_sizes)
        self._block_sizes = xp.asarray(block_sizes, dtype=int)
        self._block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
        self.num_blocks = len(block_sizes)

    def ltranspose(self, copy=False) -> "None | DSBCSR":
        """Performs a local transposition of the matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a new object. Default is False.

        Returns
        -------
        None | DSBCSR
            The transposed matrix. If copy is False, this is None.

        """
        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

        if copy:
            self = DSBCSR(
                self.data.copy(),
                self.cols.copy(),
                self.rowptr_map.copy(),
                self.block_sizes,
            )

        if not (
            hasattr(self, "_inds_bcsr2bcsr_t")
            and hasattr(self, "_rowptr_map_t")
            and hasattr(self, "_cols_t")
        ):
            # These indices are sorted by block-row and -column.
            rows, cols = self.spy()

            # Transpose.
            rows_t, cols_t = cols, rows

            # Canonical ordering of the transpose.
            inds_bcsr2canonical_t = xp.lexsort(xp.vstack((cols_t, rows_t)))
            canonical_rows_t = rows_t[inds_bcsr2canonical_t]
            canonical_cols_t = cols_t[inds_bcsr2canonical_t]

            # Compute index for sorting the transpose by block.
            inds_canonical2bcsr_t = compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Compute the rowptr map for the transpose.
            rowptr_map_t = compute_ptr_map(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Cache the necessary objects.
            self._inds_bcsr2bcsr_t = inds_bcsr2bcsr_t
            self._rowptr_map_t = rowptr_map_t
            self._cols_t = cols_t[self._inds_bcsr2bcsr_t]

        self.data[:] = self.data[..., self._inds_bcsr2bcsr_t]
        self._inds_bcsr2bcsr_t = xp.argsort(self._inds_bcsr2bcsr_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rowptr_map, self._rowptr_map_t = self._rowptr_map_t, self.rowptr_map

    def spy(self) -> tuple[xp.ndarray, xp.ndarray]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Returns
        -------
        rows : np.ndarray
            Row indices of the non-zero elements.
        cols : np.ndarray
            Column indices of the non-zero elements.

        """
        rows = xp.zeros(self.nnz, dtype=int)
        for (row, __), rowptr in self.rowptr_map.items():
            for i in range(int(self.block_sizes[row])):
                rows[rowptr[i] : rowptr[i + 1]] = i + self.block_offsets[row]

        return rows, self.cols

    @classmethod
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: xp.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBCSR":
        """Creates a new DSBSparse matrix from a scipy.sparse array.

        Parameters
        ----------
        arr : sparse.sparray
            The sparse array to convert.
        block_sizes : np.ndarray
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
        DSBCSR
            The new DSBCSR matrix.

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
        rowptr_map = compute_ptr_map(coo.row, coo.col, block_sizes)
        block_sort_index = compute_block_sort_index(coo.row, coo.col, block_sizes)

        data = xp.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        data[:] = coo.data[block_sort_index]
        cols = coo.col[block_sort_index]

        return cls(
            data=data,
            cols=cols,
            rowptr_map=rowptr_map,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
