# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.kernels import dsbcsr_kernels, dsbsparse_kernels
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import densify_selected_blocks, product_sparsity_pattern


class DSBCSR(DSBSparse):
    """Distributed stack of block-compressed sparse row matrices.

    This DSBSparse implementation uses a block-compressed sparse row
    format to store the sparsity pattern of the matrix. The data is
    sorted by block-row and -column. We use a row pointer map together
    with the column indices to access the blocks efficiently.

    Parameters
    ----------
    data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    cols : NDArray
        The column indices.
    rowptr_map : dict
        The row pointer map.
    block_sizes : NDArray
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
        data: NDArray,
        cols: NDArray,
        rowptr_map: dict,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DBCSR matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.cols = cols.astype(int)
        self.rowptr_map = rowptr_map

    def _get_items(self, stack_index: tuple, rows: NDArray, cols: NDArray) -> NDArray:
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
        rows : NDArray
            The row indices of the items.
        cols : NDArray
            The column indices of the items.

        Returns
        -------
        items : NDArray
            The requested items.

        """
        inds, value_inds = dsbcsr_kernels.find_inds(
            self.rowptr_map, self.block_offsets, self.cols, rows, cols
        )

        data_stack = self.data[stack_index]
        if self.distribution_state == "stack":
            arr = xp.zeros(data_stack.shape[:-1] + (rows.size,), dtype=self.dtype)
            arr[..., value_inds] = data_stack[..., inds]
            return xp.squeeze(arr)

        if len(inds) != rows.size:
            # We cannot know which rank is supposed to hold an element
            # that is not in the matrix, so we raise an error.
            raise IndexError("Requested element not in matrix.")

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        ranks = dsbsparse_kernels.find_ranks(self.nnz_section_offsets, inds)

        return data_stack[
            ..., inds[ranks == comm.rank] - self.nnz_section_offsets[comm.rank]
        ]

    def _set_items(
        self, stack_index: tuple, rows: NDArray, cols: NDArray, value: NDArray
    ) -> None:
        """Sets the requested items in the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the __setitem__
        method. The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        rows : NDArray
            The row indices of the items.
        cols : NDArray
            The column indices of the items.
        value : NDArray
            The value to set.

        """
        inds, value_inds = dsbcsr_kernels.find_inds(
            self.rowptr_map, self.block_offsets, self.cols, rows, cols
        )

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
        ranks = dsbsparse_kernels.find_ranks(self.nnz_section_offsets, inds)

        # If the rank does not hold any of the requested elements, we do
        # nothing.
        if not any(ranks == comm.rank):
            return

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

    def _get_block(self, stack_index: tuple, row: int, col: int) -> NDArray | tuple:
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
        block : NDArray | tuple[NDArray, NDArray, NDArray]
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`
            if `return_dense` is True, otherwise it is a tuple of three
            arrays `(rowptr, cols, data)`.

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

        dsbcsr_kernels.densify_block(
            block=block,
            block_offset=self.block_offsets[col],
            self_cols=self.cols,
            rowptr=rowptr,
            data=data_stack,
        )

        return block

    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: NDArray
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
        block : NDArray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        rowptr = self.rowptr_map.get((row, col), None)
        if rowptr is None:
            # No data in this block, nothing to do.
            return

        dsbcsr_kernels.sparsify_block(
            block=block,
            block_offset=self.block_offsets[col],
            self_cols=self.cols,
            rowptr=rowptr,
            data=self.data[*stack_index],
        )

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
        if sparse.isspmatrix(other):
            raise NotImplementedError(
                "Matrix multiplication with sparse matrices  is not implemented."
            )
        if not isinstance(other, DSBSparse):
            raise TypeError("Can only multiply DSBSparse matrices.")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("Matrix shapes do not match.")
        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")
        stack_indices = xp.ndindex(self.data.shape[:-1])
        product_rows, product_cols = product_sparsity_pattern(
            sparse.csr_matrix((xp.ones(self.nnz), (self.spy())), shape=self.shape[-2:]),
            sparse.csr_matrix(
                (xp.ones(other.nnz), (other.spy())), shape=other.shape[-2:]
            ),
        )
        block_sort_index, rowptr_map = dsbcsr_kernels.compute_rowptr_map(
            product_rows, product_cols, self.block_sizes
        )
        product = DSBCSR(
            data=xp.zeros(self.stack_shape + (product_rows.size,), dtype=self.dtype),
            cols=product_cols[block_sort_index],
            rowptr_map=rowptr_map,
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
        )
        # TODO: This is a naive implementation. Should be revisited. Same for dsbcoo.
        for stack_index in stack_indices:
            temp_product = sparse.csr_matrix(
                (self.data[stack_index], (self.spy())), shape=self.shape[-2:]
            ) @ sparse.csr_matrix(
                (other.data[stack_index], (other.spy())), shape=other.shape[-2:]
            )
            product.data[stack_index, :] = temp_product[product.spy()]
        return product

    @DSBSparse.block_sizes.setter
    def block_sizes(self, block_sizes: NDArray) -> None:
        """Sets new block sizes for the matrix.

        Parameters
        ----------
        block_sizes : NDArray
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
        inds_canonical2bcsr, rowptr_map = dsbcsr_kernels.compute_rowptr_map(
            canonical_rows, canonical_cols, block_sizes
        )
        self.rowptr_map = rowptr_map
        # Mapping directly from original block-ordering to the new
        # block-ordering is achieved by chaining the two mappings.
        inds_bcsr2bcsr = inds_bcsr2canonical[inds_canonical2bcsr]
        self.data[:] = self.data[..., inds_bcsr2bcsr]
        self.cols = self.cols[inds_bcsr2bcsr]

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
                self.global_stack_shape,
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

            # Compute index for sorting the transpose by block and the
            # transpose rowptr map.
            inds_canonical2bcsr_t, rowptr_map_t = dsbcsr_kernels.compute_rowptr_map(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Cache the necessary objects.
            self._inds_bcsr2bcsr_t = inds_bcsr2bcsr_t
            self._rowptr_map_t = rowptr_map_t
            self._cols_t = cols_t[self._inds_bcsr2bcsr_t]

        self.data[:] = self.data[..., self._inds_bcsr2bcsr_t]
        self._inds_bcsr2bcsr_t = xp.argsort(self._inds_bcsr2bcsr_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rowptr_map, self._rowptr_map_t = self._rowptr_map_t, self.rowptr_map

        return self if copy else None

    def spy(self) -> tuple[NDArray, NDArray]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Returns
        -------
        rows : NDArray
            Row indices of the non-zero elements.
        cols : NDArray
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
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned: bool = False,
    ) -> "DSBCSR":
        """Creates a new DSBSparse matrix from a scipy.sparse array.

        Parameters
        ----------
        arr : sparse.spmatrix
            The sparse array to convert.
        block_sizes : NDArray
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

        if densify_blocks is not None:
            coo = densify_selected_blocks(coo, block_sizes, densify_blocks)

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Compute block sorting index and the transpose rowptr map.
        block_sort_index, rowptr_map = dsbcsr_kernels.compute_rowptr_map(
            coo.row, coo.col, block_sizes
        )

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
