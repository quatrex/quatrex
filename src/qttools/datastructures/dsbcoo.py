# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.
from typing import Callable

from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, host_xp, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.kernels import dsbcoo_kernels, dsbsparse_kernels
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import densify_selected_blocks

profiler = Profiler()


class DSBCOO(DSBSparse):
    """Distributed stack of sparse matrices in coordinate format.

    This DSBSparse implementation stores the matrix sparsity pattern in
    probably the most straight-forward way: as a list of coordinates.
    Both data and coordinates are sorted by block-row and -column.

    Parameters
    ----------
    data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    rows : NDArray
        The row indices. This should be an array of shape `(nnz,)`.
    cols : NDArray
        The column indices. This should be an array of shape `(nnz,)`.
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
        rows: NDArray,
        cols: NDArray,
        block_sizes: NDArray,
        global_stack_shape: tuple | int,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DSBCOO matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.rows = rows.astype(xp.int32)
        self.cols = cols.astype(xp.int32)

        self._diag_inds = xp.where(self.rows == self.cols)[0]
        self._diag_value_inds = self.rows[self._diag_inds]
        ranks = dsbsparse_kernels.find_ranks(self.nnz_section_offsets, self._diag_inds)
        if not any(ranks == comm.rank):
            self._diag_inds_nnz = None
            self._diag_value_inds_nnz = None
            return
        self._diag_inds_nnz = (
            self._diag_inds[ranks == comm.rank] - self.nnz_section_offsets[comm.rank]
        )
        self._diag_value_inds_nnz = (
            self._diag_value_inds[ranks == comm.rank]
            - self._diag_value_inds[ranks == comm.rank][0]
        )

    @profiler.profile(level="debug")
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
        inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
            self.rows, self.cols, rows, cols
        )
        if max_counts not in (0, 1):
            raise IndexError(
                "Request contains repeated indices. Only unique indices are supported."
            )

        data_stack = self.data[*stack_index]

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

    @profiler.profile(level="debug")
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
        inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
            self.rows, self.cols, rows, cols
        )
        if max_counts not in (0, 1):
            raise IndexError(
                "Request contains repeated indices. Only unique indices are supported."
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

    @profiler.profile(level="debug")
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
        block_slice = self._block_config[self.num_blocks].block_slice_cache.get(
            (row, col), None
        )

        if block_slice is None:
            # Cache miss, compute the slice.
            block_slice = slice(
                *dsbcoo_kernels.compute_block_slice(
                    self.rows, self.cols, self.block_offsets, row, col
                )
            )
            self._block_config[self.num_blocks].block_slice_cache[
                (row, col)
            ] = block_slice

        return block_slice

    @profiler.profile(level="debug")
    def _get_block(
        self, arg: tuple | NDArray, row: int, col: int, is_index: bool = True
    ) -> NDArray | tuple:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        arg : tuple | NDArray
            The index of the stack or a view of the data stack. The
            is_index flag indicates whether the argument is an index or
            a view.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        is_index : bool, optional
            Whether the argument is an index or a view. Default is True.

        Returns
        -------
        block : NDArray | tuple[NDArray, NDArray, NDArray]
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])` if
            `return_dense` is True, otherwise it is a tuple of arrays
            `(rows, cols, data)`.

        """

        if is_index:
            data_stack = self.data[*arg]
        else:
            data_stack = arg
        block_slice = self._get_block_slice(row, col)

        if not self.return_dense:
            if block_slice.start is None and block_slice.stop is None:
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
        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, return an empty block.
            return block

        dsbcoo_kernels.densify_block(
            block,
            self.rows,
            self.cols,
            data_stack,
            block_slice,
            self.block_offsets[row],
            self.block_offsets[col],
        )

        return block

    def _get_sparse_block(
        self,
        arg: tuple | NDArray,
        row: int,
        col: int,
        is_index: bool = True,
    ) -> sparse.spmatrix | tuple:
        """Gets a block from the data structure in a sparse representation.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        arg : tuple | NDArray
            The index of the stack or a view of the data stack. The
            is_index flag indicates whether the argument is an index or
            a view.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        is_index : bool, optional
            Whether the argument is an index or a view. Default is True.

        Returns
        -------
        block : spmatrix | tuple
            The block at the requested index. It is a sparse
            representation of the block.

        """
        if is_index:
            data_stack = self.data[*arg]
        else:
            data_stack = arg
        block_slice = self._get_block_slice(row, col)

        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, return an empty block.
            return xp.empty(data_stack.shape[:-1] + (0,)), (xp.empty(0), xp.empty(0))

        rows = self.rows[block_slice] - self.block_offsets[row]
        cols = self.cols[block_slice] - self.block_offsets[col]
        return data_stack[..., block_slice], (rows, cols)

    @profiler.profile(level="debug")
    def _set_block(
        self,
        arg: tuple | NDArray,
        row: int,
        col: int,
        block: NDArray,
        is_index: bool = True,
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Parameters
        ----------
        arg : tuple | NDArray
            The index of the stack or a view of the data stack. The
            is_index flag indicates whether the argument is an index or
            a view.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        block : NDArray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.
        is_index : bool, optional
            Whether the argument is an index or a view. Default is True.

        """
        if is_index:
            data_stack = self.data[*arg]
        else:
            data_stack = arg
        block_slice = self._get_block_slice(row, col)
        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, nothing to do.
            return

        dsbcoo_kernels.sparsify_block(
            block,
            self.rows[block_slice] - self.block_offsets[row],
            self.cols[block_slice] - self.block_offsets[col],
            data_stack[..., block_slice],
        )

    @profiler.profile(level="debug")
    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBCOO):
            raise TypeError("Can only add DSBCOO matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if host_xp.any(self.block_sizes != other.block_sizes):
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

    # @profiler.profile(level="api")
    # def diagonal(self, stack_index: tuple = (Ellipsis,), val: NDArray = None) -> NDArray:
    #     """Returns or sets the diagonal elements of the matrix.

    #     This temporarily sets the return_dense state to True.

    #     Returns
    #     -------
    #     diagonal : NDArray
    #         The diagonal elements of the matrix.

    #     """
    #     if val is None:
    #         # Getter
    #         data_stack = self.data[*stack_index]
    #         if self.distribution_state == "stack":
    #             diagonal = xp.zeros((data_stack.shape[:-1] + (self.shape[-1],)), dtype=self.dtype)
    #             diagonal[..., self._diag_value_inds] = data_stack[..., self._diag_inds]
    #             return diagonal
    #         else:
    #             if self._diag_inds_nnz is not None:
    #                 return data_stack[..., self._diag_inds_nnz]
    #             return xp.empty(0)
    #     else:
    #         # Setter
    #         if self.distribution_state == "stack":
    #             self.data[*stack_index][..., self._diag_inds] = val[..., self._diag_value_inds]
    #         else:
    #             if self._diag_inds_nnz is not None:
    #                 stack_padding_inds = self._stack_padding_mask.nonzero()[0][stack_index[0]]
    #                 stack_inds, nnz_inds = xp.ix_(stack_padding_inds, self._diag_inds_nnz)
    #                 # We need to access the full data buffer directly to set the
    #                 # value since we are using advanced indexing.
    #                 self._data[stack_inds, stack_index[1:] or Ellipsis, nnz_inds] = val[..., self._diag_value_inds_nnz]
    #             return

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

        num_blocks = len(block_sizes)
        # Check if configuration already exists.
        if num_blocks in self._block_config:
            # Compute canonical ordering of the matrix.
            inds_bcoo2canonical = xp.lexsort(xp.vstack((self.cols, self.rows)))

            if self._block_config[num_blocks].inds_canonical2block is None:
                canonical_rows = self.rows[inds_bcoo2canonical]
                canonical_cols = self.cols[inds_bcoo2canonical]
                # Compute the index for sorting by the new block-sizes.
                inds_canonical2bcoo = dsbcoo_kernels.compute_block_sort_index(
                    canonical_rows, canonical_cols, block_sizes
                )
                self._block_config[num_blocks].inds_canonical2block = (
                    inds_canonical2bcoo
                )

            # Mapping directly from original block-ordering to the new
            # block-ordering is achieved by chaining the two mappings.
            inds_bcoo2bcoo = inds_bcoo2canonical[
                self._block_config[num_blocks].inds_canonical2block
            ]
            data = self.data.reshape(-1, self.data.shape[-1])
            for stack_idx in range(data.shape[0]):
                data[stack_idx] = data[stack_idx, inds_bcoo2bcoo]
            self.rows = self.rows[inds_bcoo2bcoo]
            self.cols = self.cols[inds_bcoo2bcoo]

            self.num_blocks = num_blocks
            return

        if sum(block_sizes) != self.shape[-1]:
            raise ValueError("Block sizes must sum to matrix shape.")
        # Compute canonical ordering of the matrix.
        inds_bcoo2canonical = xp.lexsort(xp.vstack((self.cols, self.rows)))
        canonical_rows = self.rows[inds_bcoo2canonical]
        canonical_cols = self.cols[inds_bcoo2canonical]
        # Compute the index for sorting by the new block-sizes.
        inds_canonical2bcoo = dsbcoo_kernels.compute_block_sort_index(
            canonical_rows, canonical_cols, block_sizes
        )
        # Mapping directly from original block-ordering to the new
        # block-ordering is achieved by chaining the two mappings.
        inds_bcoo2bcoo = inds_bcoo2canonical[inds_canonical2bcoo]
        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, inds_bcoo2bcoo]
        self.rows = self.rows[inds_bcoo2bcoo]
        self.cols = self.cols[inds_bcoo2bcoo]
        # Update the block sizes and offsets.
        block_sizes = host_xp.asarray(block_sizes, dtype=host_xp.int32)
        block_offsets = host_xp.hstack(
            ([0], host_xp.cumsum(block_sizes)), dtype=host_xp.int32
        )
        self.num_blocks = num_blocks
        self._add_block_config(self.num_blocks, block_sizes, block_offsets)

    @profiler.profile(level="api")
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

        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, self._inds_bcoo2bcoo_t]
        self._inds_bcoo2bcoo_t = xp.argsort(self._inds_bcoo2bcoo_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rows, self._rows_t = self._rows_t, self.rows

        (
            self._block_config[self.num_blocks].block_slice_cache,
            self._block_slice_cache_t,
        ) = (
            self._block_slice_cache_t,
            self._block_config[self.num_blocks].block_slice_cache,
        )

        return self if copy else None

    @profiler.profile(level="api")
    def symmetrize(self, op: Callable[[NDArray, NDArray], NDArray] = xp.add) -> None:
        """Symmetrizes the matrix.

        NOTE: Assumes that the natrix's sparsity pattern is symmetric.

        Parameters
        ----------
        op : callable, optional
            The operation to perform on the symmetric elements. Default
            is addition.

        """
        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot symmetrize when distributed through nnz.")

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

        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = 0.5 * op(
                data[stack_idx], data[stack_idx, self._inds_bcoo2bcoo_t].conj()
            )

    @profiler.profile(level="api")
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
        return self.rows, self.cols

    @classmethod
    @profiler.profile(level="api")
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned: bool = False,
    ) -> "DSBCOO":
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
        DSBCOO
            The new DSBCOO matrix.

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

        # Compute the block-sorting index.
        block_sort_index = dsbcoo_kernels.compute_block_sort_index(
            coo.row, coo.col, block_sizes
        )

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
