# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.kernels import dsbcoo_kernels, dsbsparse_kernels
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import densify_selected_blocks, product_sparsity_pattern


class DSBanded(DSBSparse):
    """Distributed stack of banded matrices.

    This DSBSparse implementation stores the matrices in block-based
    "TallNSkinny" or "ShortNFat" format.

    Parameters
    ----------
    data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    global_stack_shape : tuple or int
        The global shape of the stack of sparse matrices. If this is an
        integer, it is interpreted as a one-dimensional stack.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.

    """

    def __init__(
        self,
        data: NDArray,
        half_bandwidth: int,  # @czox's r
        banded_block_size: int,  # @czox's BLK_SIZE
        banded_type: int,  # 0: TallNSkinny, 1: ShortNFat
        block_sizes: NDArray,  # @czox's BIG_BLK_SIZE
        global_stack_shape: tuple | int,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DSBanded matrix."""
        sparse_data = xp.reshape(data, global_stack_shape + (-1,))
        super().__init__(sparse_data, block_sizes, global_stack_shape, return_dense)

        self.half_bandwidth = half_bandwidth
        self.banded_block_size = banded_block_size
        self.half_block_bandwidth = (half_bandwidth + banded_block_size - 1) // banded_block_size  # @czox's r_blk
        num_cols = (2 * self.half_block_bandwidth + 1) * banded_block_size
        assert sparse_data.shape[-1] % num_cols == 0
        num_rows = sparse_data.shape[-1] // num_cols
        assert num_rows % banded_block_size == 0

        self.banded_shape = (num_rows, num_cols)

        assert banded_type in (0, 1)
        self.banded_type = banded_type

        assert return_dense


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
        # inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
        #     self.rows, self.cols, rows, cols
        # )
        # if max_counts not in (0, 1):
        #     raise IndexError(
        #         "Request contains repeated indices. Only unique indices are supported."
        #     )

        data_stack = self.data[*stack_index]

        value_inds = xp.arange(len(rows), dtype=xp.int32)

        block_rows = rows // self.banded_block_size
        block_cols = cols // self.banded_block_size
        block_dist = block_cols - block_rows
        block_colidx = block_dist + self.half_block_bandwidth
        block_coloff = cols % self.banded_block_size

        inds = rows * self.banded_shape[1] + block_colidx * self.banded_block_size + block_coloff

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
        rows : NDArray
            The row indices of the items.
        cols : NDArray
            The column indices of the items.
        value : NDArray
            The value to set.

        """

        # inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
        #     self.rows, self.cols, rows, cols
        # )
        # if max_counts not in (0, 1):
        #     raise IndexError(
        #         "Request contains repeated indices. Only unique indices are supported."
        #     )

        value_inds = xp.arange(len(rows), dtype=xp.int32)

        block_rows = rows // self.banded_block_size
        block_cols = cols // self.banded_block_size
        block_dist = block_cols - block_rows
        block_colidx = block_dist + self.half_block_bandwidth
        block_coloff = cols % self.banded_block_size

        inds = rows * self.banded_shape[1] + block_colidx * self.banded_block_size + block_coloff

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

    def _get_block_slice(self, row: int, col: int) -> slice:
        """Gets the slice of data corresponding to a given block.

        This handles the block slice cache. If there is no data in the
        block, an `slice(None)` is cached.

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

        raise NotImplementedError
    
        block_slice = self._block_slice_cache.get((row, col), None)

        if block_slice is None:
            # Cache miss, compute the slice.
            block_slice = slice(
                *dsbcoo_kernels.compute_block_slice(
                    self.rows, self.cols, self.block_offsets, row, col
                )
            )

        self._block_slice_cache[(row, col)] = block_slice
        return block_slice

    def _get_block(self, stack_index: tuple, row: int, col: int) -> NDArray | tuple:
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
        block : NDArray | tuple[NDArray, NDArray, NDArray]
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])` if
            `return_dense` is True, otherwise it is a tuple of arrays
            `(rows, cols, data)`.

        """
        data_stack = self.data[*stack_index]
        # block_slice = self._get_block_slice(row, col)

        # if not self.return_dense:
        #     if block_slice == slice(None):
        #         # No data in this block, return an empty block.
        #         return xp.empty(0), xp.empty(0), xp.empty(data_stack.shape[:-1] + (0,))

        #     rows = self.rows[block_slice] - self.block_offsets[row]
        #     cols = self.cols[block_slice] - self.block_offsets[col]
        #     return rows, cols, data_stack[..., block_slice]

        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.block_sizes[row]), int(self.block_sizes[col])),
            dtype=self.dtype,
        )
        # if block_slice == slice(None):
        #     # No data in this block, return an empty block.
        #     return block

        # dsbcoo_kernels.densify_block(
        #     block,
        #     self.rows[block_slice] - self.block_offsets[row],
        #     self.cols[block_slice] - self.block_offsets[col],
        #     data_stack[..., block_slice],
        # )

        # if data_stack.ndim > 1:
        data_stack = xp.reshape(data_stack, data_stack.shape[:-1] + self.banded_shape)
        # else:
        #     data_stack = xp.reshape(data_stack, (1, *self.banded_shape))

        if not self.banded_type == 0:
            raise NotImplementedError

        big_block_i = row
        big_block_j = col
        BIG_BLOCK_SIZE_I = int(self.block_sizes[row])
        BIG_BLOCK_SIZE_J = int(self.block_sizes[col])
        BLK_SIZE = int(self.banded_block_size)
        r_block = int(self.half_block_bandwidth)
        A_blk_tallNSkinny = data_stack
        A_dense_block = block

        if len(A_blk_tallNSkinny.shape) == 2:
            A_blk_tallNSkinny = xp.reshape(A_blk_tallNSkinny, (1, *A_blk_tallNSkinny.shape))
        if A_dense_block.ndim == 2:
            A_dense_block = xp.reshape(A_dense_block, (1, *A_dense_block.shape))
        # batch, M, r = A_blk_tallNSkinny.shape
        r = A_blk_tallNSkinny.shape[-1]

        # translate the BIG_BLOCK_SIZE coordinates big_block_i, big_block_j
        # to the ranges of the block rows and columns
        requested_dense_row_start = int(self.block_offsets[big_block_i])
        requested_dense_row_end = int(self.block_offsets[big_block_i + 1])
        requested_dense_col_start = int(self.block_offsets[big_block_j])
        requested_dense_col_end = int(self.block_offsets[big_block_j + 1])

        blk_row_start = requested_dense_row_start // BLK_SIZE
        blk_row_end = (requested_dense_row_end + BLK_SIZE - 1) // BLK_SIZE

        # iterate over the block rows
        for blk_i in range(blk_row_start, blk_row_end):
            # calculate the range of columns in the dense matrix for the current block row
            dense_blk_col_start = blk_i - r_block
            dense_blk_col_end = blk_i + r_block + 1

            dense_col_start = dense_blk_col_start * BLK_SIZE
            dense_col_end = dense_blk_col_end * BLK_SIZE

            # if dense_col_start > requested_dense_col_end, pad with zeros
            # from the left side. Otherwise, extract the subset of the block row.
            left_padding = min(
                max(0, dense_col_start - requested_dense_col_start), BIG_BLOCK_SIZE_J
            )

            # if dense_col_end < requested_dense_col_start, pad with zeros
            # from the right side.
            right_padding = min(
                max(0, requested_dense_col_end - dense_col_end), BIG_BLOCK_SIZE_J
            )

            left_offset = max(0, requested_dense_col_start - dense_col_start)
            right_offset = max(0, dense_col_end - requested_dense_col_end)

            # get the block row from the tallAndSkinny matrix
            start_banded = max(blk_i * BLK_SIZE, requested_dense_row_start)
            end_banded = min((blk_i + 1) * BLK_SIZE, requested_dense_row_end)
            blk_row = A_blk_tallNSkinny[
                ...,
                start_banded : end_banded,
                left_offset : r - right_offset,
            ]

            # apply padding if needed
            if left_padding > 0 or right_padding > 0:
                # blk_row = torch.nn.functional.pad(blk_row, (left_padding, right_padding))
                blk_row = xp.pad(blk_row, ((0, 0), (0, 0), (left_padding, right_padding)), 'constant', constant_values=0)

            # copy the block row to the big block matrix
            start_block = (blk_i - blk_row_start) * BLK_SIZE
            end_block = min(BIG_BLOCK_SIZE_I, (blk_i + 1 - blk_row_start) * BLK_SIZE)
            A_dense_block[
                ...,
                start_block: end_block,
                :,
            ] = blk_row

        return block

    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: NDArray
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
        block : NDArray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        # block_slice = self._get_block_slice(row, col)
        # if block_slice == slice(None):
        #     # No data in this block, nothing to do.
        #     return

        # dsbcoo_kernels.sparsify_block(
        #     block,
        #     self.rows[block_slice] - self.block_offsets[row],
        #     self.cols[block_slice] - self.block_offsets[col],
        #     self.data[*stack_index][..., block_slice],
        # )

        if not self.banded_type == 0:
            raise NotImplementedError
        
        data_stack = self.data[*stack_index]
        data_stack = xp.reshape(data_stack, data_stack.shape[:-1] + self.banded_shape)

        big_block_i = row
        big_block_j = col
        BIG_BLOCK_SIZE_I = int(self.block_sizes[row])
        BIG_BLOCK_SIZE_J = int(self.block_sizes[col])
        BLK_SIZE = int(self.banded_block_size)
        r_block = int(self.half_block_bandwidth)
        A_blk_tallNSkinny = data_stack
        A_dense_block = block

        if len(A_blk_tallNSkinny.shape) == 2:
            A_blk_tallNSkinny = xp.reshape(A_blk_tallNSkinny, (1, *A_blk_tallNSkinny.shape))
        if A_dense_block.ndim == 2:
            A_dense_block = xp.reshape(A_dense_block, (1, *A_dense_block.shape))
        # batch, M, r = A_blk_tallNSkinny.shape
        r = A_blk_tallNSkinny.shape[-1]

        # translate the BIG_BLOCK_SIZE coordinates big_block_i, big_block_j
        # to the ranges of the block rows and columns
        requested_dense_row_start = int(self.block_offsets[big_block_i])
        requested_dense_row_end = int(self.block_offsets[big_block_i + 1])
        requested_dense_col_start = int(self.block_offsets[big_block_j])
        requested_dense_col_end = int(self.block_offsets[big_block_j + 1])

        blk_row_start = requested_dense_row_start // BLK_SIZE
        blk_row_end = (requested_dense_row_end + BLK_SIZE - 1) // BLK_SIZE

        # iterate over the block rows
        for blk_i in range(blk_row_start, blk_row_end):
            # calculate the range of columns in the dense matrix for the current block row
            dense_blk_col_start = blk_i - r_block
            dense_blk_col_end = blk_i + r_block + 1

            dense_col_start = dense_blk_col_start * BLK_SIZE
            dense_col_end = dense_blk_col_end * BLK_SIZE

            # if dense_col_start > requested_dense_col_end, pad with zeros
            # from the left side. Otherwise, extract the subset of the block row.
            left_padding = min(
                max(0, dense_col_start - requested_dense_col_start), BIG_BLOCK_SIZE_J
            )

            # if dense_col_end < requested_dense_col_start, pad with zeros
            # from the right side.
            right_padding = min(
                max(0, requested_dense_col_end - dense_col_end), BIG_BLOCK_SIZE_J
            )

            left_offset = max(0, requested_dense_col_start - dense_col_start)
            right_offset = max(0, dense_col_end - requested_dense_col_end)

            # get the block row from the dense matrix
            start_block = (blk_i - blk_row_start) * BLK_SIZE
            end_block = min(BIG_BLOCK_SIZE_I, (blk_i + 1 - blk_row_start) * BLK_SIZE)
            blk_row = A_dense_block[
                ...,
                start_block : end_block,
                :,
            ]

            # dense 3diag blocks covers the tallAndSkinny matrix, so we need to trim the blk_row
            # from the "corners" of the dense block to fit the tallAndSkinny matrix
            if left_padding > 0 or right_padding > 0:
                blk_row = blk_row[:, :, left_offset : r - right_offset]

            # copy the block row to the tall and skinny matrix
            start_banded = max(blk_i * BLK_SIZE, requested_dense_row_start)
            end_banded = min((blk_i + 1) * BLK_SIZE, requested_dense_row_end)
            A_blk_tallNSkinny[
                ...,
                start_banded : end_banded,
                left_offset : r - right_offset,
            ] = blk_row

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBanded):
            raise TypeError("Can only add DSBanded matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")
        
        if self.banded_type != other.banded_type:
            raise ValueError("Banded types do not match.")
        
        if self.banded_block_size != other.banded_block_size:
            raise ValueError("Banded block sizes do not match.")

    def __neg__(self) -> "DSBanded":
        """Negation of the data."""
        return DSBanded(
            data=-self.data,
            half_bandwidth=self.half_bandwidth,
            banded_block_size=self.banded_block_size,
            banded_type=self.banded_type,
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
            return_dense=self.return_dense,
        )

    def __matmul__(self, other: "DSBanded") -> None:
        """Matrix multiplication of two DSBanded matrices."""

        raise NotImplementedError

        if sparse.isspmatrix(other):
            raise NotImplementedError(
                "Matrix multiplication with sparse matrices is not implemented."
            )
        if not isinstance(other, DSBSparse):
            raise TypeError("Can only multiply DSBSparse matrices.")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("Matrix shapes do not match.")
        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")
        product_rows, product_cols = product_sparsity_pattern(
            sparse.csr_matrix(
                (xp.ones(self.nnz), (self.rows, self.cols)), shape=self.shape[-2:]
            ),
            sparse.csr_matrix(
                (xp.ones(other.nnz), (other.rows, other.cols)),
                shape=other.shape[-2:],
            ),
        )
        block_sort_index = dsbcoo_kernels.compute_block_sort_index(
            product_rows, product_cols, self.block_sizes
        )
        product = DSBCOO(
            data=xp.zeros(self.stack_shape + (product_rows.size,), dtype=self.dtype),
            rows=product_rows[block_sort_index],
            cols=product_cols[block_sort_index],
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
        )
        # TODO: This is a naive implementation. Should be revisited. Same for dsbcsr.
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
            raise ValueError("Block sizes must sum to matrix shape.")
        # Update the block sizes and offsets.
        self._block_sizes = xp.asarray(block_sizes, dtype=int)
        self.block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
        self.num_blocks = len(block_sizes)
        self._block_slice_cache = {}

    def ltranspose(self, copy=False) -> "None | DSBanded":
        """Performs a local transposition of the matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a new object. Default is False.

        Returns
        -------
        None | DSBanded
            The transposed matrix. If copy is False, this is None.

        """

        raise NotImplementedError

        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

        if copy:
            self = DSBanded(
                self.data.copy(),
                self.half_bandwidth,
                self.banded_block_size,
                self.banded_type,
                self.block_sizes,
                self.global_stack_shape,
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
            inds_canonical2bcoo_t = dsbcoo_kernels.compute_block_sort_index(
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

        raise NotImplementedError

        return self.rows, self.cols

    @classmethod
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned: bool = False,
    ) -> "DSBanded":
        """Creates a new DSBanded matrix from a scipy.sparse array.

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
        DSBanded
            The new DSBanded matrix.

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

        # # Compute the block-sorting index.
        # block_sort_index = dsbcoo_kernels.compute_block_sort_index(
        #     coo.row, coo.col, block_sizes
        # )

        half_bandwidth = int(xp.abs(coo.row - coo.col).max())
        banded_block_size = 16
        banded_type = 0
        half_block_bandwidth = (half_bandwidth + banded_block_size - 1) // banded_block_size
        banded_rows = ((coo.shape[0] + banded_block_size - 1) // banded_block_size) * banded_block_size
        banded_cols = (2 * half_block_bandwidth + 1) * banded_block_size
        banded_shape = (banded_rows, banded_cols)

        dense = xp.zeros((banded_rows, banded_cols), dtype=coo.data.dtype)
        dense[coo.row, coo.col] = coo.data
        data = xp.zeros(local_stack_shape + banded_shape, dtype=coo.data.dtype)
        # data[..., :] = coo.data[block_sort_index]
        # rows = coo.row[block_sort_index]
        # cols = coo.col[block_sort_index]

        A = dense
        A_blk_tallNSkinny = data
        BLK_SIZE = banded_block_size
        r_block = half_block_bandwidth

        if len(A.shape) == 2:
            A = xp.reshape(A, (1,) + A.shape)
        batch, M, N = A.shape
        # allocate memory for the compressed matrix
        # A_blk_tallNSkinny = torch.zeros(
        #     (batch, M, (2 * r_block + 1) * BLK_SIZE), dtype=A.dtype, device=A.device
        # )

        # iterate over the block rows
        for blk_i in range(0, M // BLK_SIZE):
            # copy the block row from the dense matrix to the tall and skinny matrix
            # while shifting the elements to the correct positions
            blk_col_start = blk_i - r_block
            blk_col_end = blk_i + r_block + 1

            col_start = blk_col_start * BLK_SIZE
            col_end = blk_col_end * BLK_SIZE

            # calculate the valid range of columns for the current block row and, if needed, pad with zeros
            left_padding = max(0, -col_start)
            right_padding = max(0, col_end - N)
            col_start = max(0, col_start)
            col_end = min(N, col_end)

            # get the block row from the dense matrix
            blk_row = A[:, blk_i * BLK_SIZE : (blk_i + 1) * BLK_SIZE, col_start:col_end]

            # apply padding if needed
            if left_padding > 0 or right_padding > 0:
                # blk_row = torch.nn.functional.pad(blk_row, (left_padding, right_padding))
                blk_row = xp.pad(blk_row, ((0, 0), (0, 0), (left_padding, right_padding)), 'constant', constant_values=0)

            # copy the block row to the tall and skinny matrix
            A_blk_tallNSkinny[..., blk_i * BLK_SIZE : (blk_i + 1) * BLK_SIZE, :] = blk_row

        return cls(
            data=data,
            half_bandwidth=half_bandwidth,
            banded_block_size=banded_block_size,
            banded_type=banded_type,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )


class ShortNFat(DSBSparse):
    """Distributed stack of banded matrices.

    This DSBSparse implementation stores the matrices in block-based
    "TallNSkinny" or "ShortNFat" format.

    Parameters
    ----------
    data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    global_stack_shape : tuple or int
        The global shape of the stack of sparse matrices. If this is an
        integer, it is interpreted as a one-dimensional stack.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.

    """

    def __init__(
        self,
        data: NDArray,
        half_bandwidth: int,  # @czox's r
        banded_block_size: int,  # @czox's BLK_SIZE
        banded_type: int,  # 0: TallNSkinny, 1: ShortNFat
        block_sizes: NDArray,  # @czox's BIG_BLK_SIZE
        global_stack_shape: tuple | int,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DSBanded matrix."""
        sparse_data = xp.reshape(data, global_stack_shape + (-1,))
        super().__init__(sparse_data, block_sizes, global_stack_shape, return_dense)

        self.half_bandwidth = half_bandwidth
        self.banded_block_size = banded_block_size
        self.half_block_bandwidth = (half_bandwidth + banded_block_size - 1) // banded_block_size  # @czox's r_blk
        num_rows = (2 * self.half_block_bandwidth + 1) * banded_block_size
        assert sparse_data.shape[-1] % num_rows == 0
        num_cols = sparse_data.shape[-1] // num_rows
        assert num_cols % banded_block_size == 0

        self.banded_shape = (num_rows, num_cols)

        assert banded_type in (0, 1)
        self.banded_type = banded_type

        assert return_dense


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
        # inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
        #     self.rows, self.cols, rows, cols
        # )
        # if max_counts not in (0, 1):
        #     raise IndexError(
        #         "Request contains repeated indices. Only unique indices are supported."
        #     )

        data_stack = self.data[*stack_index]

        value_inds = xp.arange(len(rows), dtype=xp.int32)

        block_rows = rows // self.banded_block_size
        block_cols = cols // self.banded_block_size
        block_dist = block_cols - block_rows
        block_rowidx = block_dist + self.half_block_bandwidth
        block_rowoff = rows % self.banded_block_size

        inds = (block_rowidx * self.banded_block_size + block_rowoff) * self.banded_shape[1] + cols

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
        rows : NDArray
            The row indices of the items.
        cols : NDArray
            The column indices of the items.
        value : NDArray
            The value to set.

        """

        # inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
        #     self.rows, self.cols, rows, cols
        # )
        # if max_counts not in (0, 1):
        #     raise IndexError(
        #         "Request contains repeated indices. Only unique indices are supported."
        #     )

        value_inds = xp.arange(len(rows), dtype=xp.int32)

        block_rows = rows // self.banded_block_size
        block_cols = cols // self.banded_block_size
        block_dist = block_cols - block_rows
        block_rowidx = block_dist + self.half_block_bandwidth
        block_rowoff = rows % self.banded_block_size

        inds = (block_rowidx * self.banded_block_size + block_rowoff) * self.banded_shape[1] + cols

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

    def _get_block_slice(self, row: int, col: int) -> slice:
        """Gets the slice of data corresponding to a given block.

        This handles the block slice cache. If there is no data in the
        block, an `slice(None)` is cached.

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

        raise NotImplementedError
    
        block_slice = self._block_slice_cache.get((row, col), None)

        if block_slice is None:
            # Cache miss, compute the slice.
            block_slice = slice(
                *dsbcoo_kernels.compute_block_slice(
                    self.rows, self.cols, self.block_offsets, row, col
                )
            )

        self._block_slice_cache[(row, col)] = block_slice
        return block_slice

    def _get_block(self, stack_index: tuple, row: int, col: int) -> NDArray | tuple:
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
        block : NDArray | tuple[NDArray, NDArray, NDArray]
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])` if
            `return_dense` is True, otherwise it is a tuple of arrays
            `(rows, cols, data)`.

        """
        data_stack = self.data[*stack_index]
        # block_slice = self._get_block_slice(row, col)

        # if not self.return_dense:
        #     if block_slice == slice(None):
        #         # No data in this block, return an empty block.
        #         return xp.empty(0), xp.empty(0), xp.empty(data_stack.shape[:-1] + (0,))

        #     rows = self.rows[block_slice] - self.block_offsets[row]
        #     cols = self.cols[block_slice] - self.block_offsets[col]
        #     return rows, cols, data_stack[..., block_slice]

        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.block_sizes[row]), int(self.block_sizes[col])),
            dtype=self.dtype,
        )
        # if block_slice == slice(None):
        #     # No data in this block, return an empty block.
        #     return block

        # dsbcoo_kernels.densify_block(
        #     block,
        #     self.rows[block_slice] - self.block_offsets[row],
        #     self.cols[block_slice] - self.block_offsets[col],
        #     data_stack[..., block_slice],
        # )

        # if data_stack.ndim > 1:
        data_stack = xp.reshape(data_stack, data_stack.shape[:-1] + self.banded_shape)
        # else:
        #     data_stack = xp.reshape(data_stack, (1, *self.banded_shape))

        if not self.banded_type == 1:
            raise NotImplementedError

        big_block_i = row
        big_block_j = col
        BIG_BLOCK_SIZE_I = int(self.block_sizes[row])
        BIG_BLOCK_SIZE_J = int(self.block_sizes[col])
        BLK_SIZE = int(self.banded_block_size)
        r_block = int(self.half_block_bandwidth)
        B_blk_shortNFat = data_stack
        B_dense_block = block

        if len(B_blk_shortNFat) == 2:
            B_blk_shortNFat = xp.reshape(B_blk_shortNFat, (1, *B_blk_shortNFat.shape))
        if B_dense_block.ndim == 2:
            B_dense_block = xp.reshape(B_dense_block, (1, *B_dense_block.shape))
        r = B_blk_shortNFat.shape[-2]

        # translate the BIG_BLOCK_SIZE coordinates big_block_i, big_block_j
        # to the ranges of the block rows and columns
        requested_dense_row_start = int(self.block_offsets[big_block_i])
        requested_dense_row_end = int(self.block_offsets[big_block_i + 1])
        requested_dense_col_start = int(self.block_offsets[big_block_j])
        requested_dense_col_end = int(self.block_offsets[big_block_j + 1])

        blk_col_start = requested_dense_col_start // BLK_SIZE
        blk_col_end = (requested_dense_col_end + BLK_SIZE - 1) // BLK_SIZE

        # iterate over the block columns
        for blk_j in range(blk_col_start, blk_col_end):
            # calculate the range of rows in the short-and-fat matrix for the current block column
            dense_blk_row_start = blk_j - r_block
            dense_blk_row_end = blk_j + r_block + 1

            dense_row_start = dense_blk_row_start * BLK_SIZE
            dense_row_end = dense_blk_row_end * BLK_SIZE

            # if dense_row_start > requested_dense_row_end, pad with zeros
            # from the top. Otherwise, extract the subset of the block column.
            top_padding = min(
                max(0, dense_row_start - requested_dense_row_start), BIG_BLOCK_SIZE_I
            )

            # if dense_row_end < requested_dense_row_start, pad with zeros
            # from the bottom.
            bottom_padding = min(
                max(0, requested_dense_row_end - dense_row_end), BIG_BLOCK_SIZE_I
            )

            top_offset = max(0, requested_dense_row_start - dense_row_start)
            bottom_offset = max(0, dense_row_end - requested_dense_row_end)

            # get the block column from the shortNFat matrix
            start_banded = max(blk_j * BLK_SIZE, requested_dense_col_start)
            end_banded = min((blk_j + 1) * BLK_SIZE, requested_dense_col_end)
            blk_col = B_blk_shortNFat[
                ...,
                top_offset : r - bottom_offset,
                start_banded : end_banded,
            ]

            # apply padding if needed
            if top_padding > 0 or bottom_padding > 0:
                blk_col = xp.pad(blk_col, ((0, 0), (top_padding, bottom_padding), (0, 0)), 'constant', constant_values=0)

            # copy the block column to the big block matrix
            start_block = (blk_j - blk_col_start) * BLK_SIZE
            end_block = min(BIG_BLOCK_SIZE_J, (blk_j + 1 - blk_col_start) * BLK_SIZE)
            B_dense_block[
                ...,
                :,
                start_block: end_block,
            ] = blk_col

        return block

    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: NDArray
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
        block : NDArray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        # block_slice = self._get_block_slice(row, col)
        # if block_slice == slice(None):
        #     # No data in this block, nothing to do.
        #     return

        # dsbcoo_kernels.sparsify_block(
        #     block,
        #     self.rows[block_slice] - self.block_offsets[row],
        #     self.cols[block_slice] - self.block_offsets[col],
        #     self.data[*stack_index][..., block_slice],
        # )

        if not self.banded_type == 1:
            raise NotImplementedError
        
        data_stack = self.data[*stack_index]
        data_stack = xp.reshape(data_stack, data_stack.shape[:-1] + self.banded_shape)

        big_block_i = row
        big_block_j = col
        BIG_BLOCK_SIZE_I = int(self.block_sizes[row])
        BIG_BLOCK_SIZE_J = int(self.block_sizes[col])
        BLK_SIZE = int(self.banded_block_size)
        r_block = int(self.half_block_bandwidth)
        B_blk_shortNFat = data_stack
        B_dense_block = block

        if len(B_blk_shortNFat.shape) == 2:
            B_blk_shortNFat = xp.reshape(B_blk_shortNFat, (1, *B_blk_shortNFat.shape))
        if B_dense_block.ndim == 2:
            B_dense_block = xp.reshape(B_dense_block, (1, *B_dense_block.shape))
        r = B_blk_shortNFat.shape[-2]

        # translate the BIG_BLOCK_SIZE coordinates big_block_i, big_block_j
        # to the ranges of the block rows and columns
        requested_dense_row_start = int(self.block_offsets[big_block_i])
        requested_dense_row_end = int(self.block_offsets[big_block_i + 1])
        requested_dense_col_start = int(self.block_offsets[big_block_j])
        requested_dense_col_end = int(self.block_offsets[big_block_j + 1])

        blk_col_start = requested_dense_col_start // BLK_SIZE
        blk_col_end = (requested_dense_col_end + BLK_SIZE - 1) // BLK_SIZE

        # iterate over the block columns
        for blk_j in range(blk_col_start, blk_col_end):
            # calculate the range of columns in the dense matrix for the current block column
            dense_blk_row_start = blk_j - r_block
            dense_blk_row_end = blk_j + r_block + 1

            dense_row_start = dense_blk_row_start * BLK_SIZE
            dense_row_end = dense_blk_row_end * BLK_SIZE

            # if dense_row_start > requested_dense_row_end, pad with zeros
            # from the top. Otherwise, extract the subset of the block column.
            top_padding = min(
                max(0, dense_row_start - requested_dense_row_start), BIG_BLOCK_SIZE_I
            )

            # if dense_row_end < requested_dense_row_start, pad with zeros
            # from the bottom.
            bottom_padding = min(
                max(0, requested_dense_row_end - dense_row_end), BIG_BLOCK_SIZE_I
            )

            top_offset = max(0, requested_dense_row_start - dense_row_start)
            bottom_offset = max(0, dense_row_end - requested_dense_row_end)

            # get the block row from the dense matrix
            start_block = (blk_j - blk_col_start) * BLK_SIZE
            end_block = min(BIG_BLOCK_SIZE_J, (blk_j + 1 - blk_col_start) * BLK_SIZE)
            blk_col = B_dense_block[
                ...,
                :,
                start_block : end_block,
            ]

            # dense 3diag blocks covers the shortNFat matrix, so we need to trim the blk_col
            # from the "corners" of the dense block to fit the shortNFat matrix
            if top_padding > 0 or bottom_padding > 0:
                blk_col = blk_col[:, top_offset : r - bottom_offset]

            # copy the block row to the tall and skinny matrix
            start_banded = max(blk_j * BLK_SIZE, requested_dense_col_start)
            end_banded = min((blk_j + 1) * BLK_SIZE, requested_dense_col_end)
            B_blk_shortNFat[
                ...,
                top_offset : r - bottom_offset,
                start_banded : end_banded,
            ] = blk_col

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBanded):
            raise TypeError("Can only add DSBanded matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")
        
        if self.banded_type != other.banded_type:
            raise ValueError("Banded types do not match.")
        
        if self.banded_block_size != other.banded_block_size:
            raise ValueError("Banded block sizes do not match.")

    def __neg__(self) -> "DSBanded":
        """Negation of the data."""
        return DSBanded(
            data=-self.data,
            half_bandwidth=self.half_bandwidth,
            banded_block_size=self.banded_block_size,
            banded_type=self.banded_type,
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
            return_dense=self.return_dense,
        )

    def __matmul__(self, other: "DSBanded") -> None:
        """Matrix multiplication of two DSBanded matrices."""

        raise NotImplementedError

        if sparse.isspmatrix(other):
            raise NotImplementedError(
                "Matrix multiplication with sparse matrices is not implemented."
            )
        if not isinstance(other, DSBSparse):
            raise TypeError("Can only multiply DSBSparse matrices.")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError("Matrix shapes do not match.")
        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")
        product_rows, product_cols = product_sparsity_pattern(
            sparse.csr_matrix(
                (xp.ones(self.nnz), (self.rows, self.cols)), shape=self.shape[-2:]
            ),
            sparse.csr_matrix(
                (xp.ones(other.nnz), (other.rows, other.cols)),
                shape=other.shape[-2:],
            ),
        )
        block_sort_index = dsbcoo_kernels.compute_block_sort_index(
            product_rows, product_cols, self.block_sizes
        )
        product = DSBCOO(
            data=xp.zeros(self.stack_shape + (product_rows.size,), dtype=self.dtype),
            rows=product_rows[block_sort_index],
            cols=product_cols[block_sort_index],
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
        )
        # TODO: This is a naive implementation. Should be revisited. Same for dsbcsr.
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
            raise ValueError("Block sizes must sum to matrix shape.")
        # Update the block sizes and offsets.
        self._block_sizes = xp.asarray(block_sizes, dtype=int)
        self.block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
        self.num_blocks = len(block_sizes)
        self._block_slice_cache = {}

    def ltranspose(self, copy=False) -> "None | DSBanded":
        """Performs a local transposition of the matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a new object. Default is False.

        Returns
        -------
        None | DSBanded
            The transposed matrix. If copy is False, this is None.

        """

        raise NotImplementedError

        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

        if copy:
            self = DSBanded(
                self.data.copy(),
                self.half_bandwidth,
                self.banded_block_size,
                self.banded_type,
                self.block_sizes,
                self.global_stack_shape,
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
            inds_canonical2bcoo_t = dsbcoo_kernels.compute_block_sort_index(
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

        raise NotImplementedError

        return self.rows, self.cols

    @classmethod
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned: bool = False,
    ) -> "ShortNFat":
        """Creates a new ShortNFat matrix from a scipy.sparse array.

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
        ShortNFat
            The new ShortNFat matrix.

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

        # # Compute the block-sorting index.
        # block_sort_index = dsbcoo_kernels.compute_block_sort_index(
        #     coo.row, coo.col, block_sizes
        # )

        half_bandwidth = int(xp.abs(coo.row - coo.col).max())
        banded_block_size = 16
        banded_type = 1
        half_block_bandwidth = (half_bandwidth + banded_block_size - 1) // banded_block_size
        banded_rows = (2 * half_block_bandwidth + 1) * banded_block_size
        banded_cols = ((coo.shape[1] + banded_block_size - 1) // banded_block_size) * banded_block_size
        banded_shape = (banded_rows, banded_cols)

        dense = xp.zeros((banded_rows, banded_cols), dtype=coo.data.dtype)
        dense[coo.row, coo.col] = coo.data
        data = xp.zeros(local_stack_shape + banded_shape, dtype=coo.data.dtype)
        # data[..., :] = coo.data[block_sort_index]
        # rows = coo.row[block_sort_index]
        # cols = coo.col[block_sort_index]

        B = dense
        B_blk_shortNFat = data
        BLK_SIZE = banded_block_size
        r_block = half_block_bandwidth

        if len(B.shape) == 2:
            B = xp.reshape(B, (1,) + B.shape)
        batch, M, N = B.shape
        # allocate memory for the compressed matrix
        # A_blk_tallNSkinny = torch.zeros(
        #     (batch, M, (2 * r_block + 1) * BLK_SIZE), dtype=A.dtype, device=A.device
        # )

        # iterate over the block rows
        for blk_j in range(0, N // BLK_SIZE):
            # copy the block row from the dense matrix to the tall and skinny matrix
            # while shifting the elements to the correct positions
            blk_row_start = blk_j - r_block
            blk_row_end = blk_j + r_block + 1

            row_start = blk_row_start * BLK_SIZE
            row_end = blk_row_end * BLK_SIZE

            # calculate the valid range of columns for the current block row and, if needed, pad with zeros
            top_padding = max(0, -row_start)
            bottom_padding = max(0, row_end - M)
            row_start = max(0, row_start)
            row_end = min(M, row_end)

            # get the block row from the dense matrix
            blk_col = B[:, row_start:row_end, blk_j * BLK_SIZE : (blk_j + 1) * BLK_SIZE]

            # apply padding if needed
            if top_padding > 0 or bottom_padding > 0:
                # blk_col = torch.nn.functional.pad(blk_col, (0, 0, top_padding, bottom_padding))
                blk_col = xp.pad(blk_col, ((0, 0), (top_padding, bottom_padding), (0, 0)), 'constant', constant_values=0)

            # copy the block row to the tall and skinny matrix
            B_blk_shortNFat[..., :, blk_j * BLK_SIZE : (blk_j + 1) * BLK_SIZE] = blk_col

        return cls(
            data=data,
            half_bandwidth=half_bandwidth,
            banded_block_size=banded_block_size,
            banded_type=banded_type,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
