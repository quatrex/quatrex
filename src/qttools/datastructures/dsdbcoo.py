# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import copy
from typing import Callable

from qttools import NDArray, block_comm, host_xp, sparse, stack_comm, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.kernels import dsbcoo_kernels, dsbsparse_kernels
from qttools.profiling.profiler import Profiler
from qttools.utils.mpi_utils import get_section_sizes

profiler = Profiler()


class DSDBCOO(DSDBSparse):
    """A Distributed Stack of Distributed Block-accessible COO matrices.

    Note
    ----
    It is the caller's responsibility to ensure that the data is
    distributed correctly across the ranks.

    Parameters
    ----------
    local_data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, local_nnz)`.
    local_rows : NDArray
        The local row indices of the COO matrix.
    local_cols : NDArray
        The local column indices of the COO matrix.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    global_stack_shape : tuple or int
        The global shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    block_comm : MPI.Comm
        The communicator for the block distribution.
    stack_comm : MPI.Comm
        The communicator for the stack distribution.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.

    """

    def __init__(
        self,
        local_data: NDArray,
        local_rows: NDArray,
        local_cols: NDArray,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ):
        """Initializes a DSDBCOO matrix."""
        super().__init__(local_data, block_sizes, global_stack_shape, return_dense)

        self.rows = xp.asarray(local_rows, dtype=xp.int32)
        self.cols = xp.asarray(local_cols, dtype=xp.int32)

        self._diag_inds = xp.where(self.rows == self.cols)[0]
        self._diag_value_inds = self.rows[self._diag_inds]
        ranks = dsbsparse_kernels.find_ranks(self.nnz_section_offsets, self._diag_inds)
        if not any(ranks == stack_comm.rank):
            self._diag_inds_nnz = None
            self._diag_value_inds_nnz = None
            return
        self._diag_inds_nnz = (
            self._diag_inds[ranks == stack_comm.rank]
            - self.nnz_section_offsets[stack_comm.rank]
        )
        self._diag_value_inds_nnz = (
            self._diag_value_inds[ranks == stack_comm.rank]
            - self._diag_value_inds[ranks == stack_comm.rank][0]
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

        Note
        ----
        Note that in stack distribution, every rank in the
        block-communicator will return either zeros or the requested
        data if present.

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
        # We need to shift the local rows and cols to the global
        # coordinates.
        inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
            self.rows + self.global_block_offset,
            self.cols + self.global_block_offset,
            rows,
            cols,
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

        # NOTE: This causes problems with the block-distribution, so
        # this is commented out for now. If you request elements outside
        # the sparsity pattern, you may get trouble.
        # if len(inds) != rows.size:
        #     # We cannot know which rank is supposed to hold an element
        #     # that is not in the matrix, so we raise an error.
        #     raise IndexError("Requested element not in matrix.")

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        ranks = dsbsparse_kernels.find_ranks(self.nnz_section_offsets, inds)

        return data_stack[
            ...,
            inds[ranks == stack_comm.rank] - self.nnz_section_offsets[stack_comm.rank],
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
        # We need to shift the local rows and cols to the global
        # coordinates.
        inds, value_inds, max_counts = dsbcoo_kernels.find_inds(
            self.rows + self.global_block_offset,
            self.cols + self.global_block_offset,
            rows,
            cols,
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
        if not any(ranks == stack_comm.rank):
            return

        stack_padding_inds = self._stack_padding_mask.nonzero()[0][stack_index[0]]
        stack_inds, nnz_inds = xp.ix_(
            stack_padding_inds,
            inds[ranks == stack_comm.rank] - self.nnz_section_offsets[stack_comm.rank],
        )
        # We need to access the full data buffer directly to set the
        # value since we are using advanced indexing.
        if value.ndim == 0:
            self._data[stack_inds, stack_index[1:] or Ellipsis, nnz_inds] = value
            return

        self._data[stack_inds, stack_index[1:] or Ellipsis, nnz_inds] = value[
            ...,
            value_inds[ranks == stack_comm.rank]
            - value_inds[ranks == stack_comm.rank][0],
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
                    self.rows, self.cols, self.local_block_offsets, row, col
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

            rows = self.rows[block_slice] - self.local_block_offsets[row]
            cols = self.cols[block_slice] - self.local_block_offsets[col]
            return rows, cols, data_stack[..., block_slice]

        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.local_block_sizes[row]), int(self.local_block_sizes[col])),
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
            self.local_block_offsets[row],
            self.local_block_offsets[col],
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

        rows = self.rows[block_slice] - self.local_block_offsets[row]
        cols = self.cols[block_slice] - self.local_block_offsets[col]
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
            self.rows[block_slice] - self.local_block_offsets[row],
            self.cols[block_slice] - self.local_block_offsets[col],
            data_stack[..., block_slice],
        )

    @profiler.profile(level="debug")
    def _check_commensurable(self, other: "DSDBCOO") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSDBCOO):
            raise TypeError("Can only add DSDBCOO matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if xp.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")

        if xp.any(self.rows != other.rows):
            raise ValueError("Row indices do not match.")

        if xp.any(self.cols != other.cols):
            raise ValueError("Column indices do not match.")

    def __iadd__(self, other: "DSDBCOO | sparse.spmatrix") -> "DSDBCOO":
        """In-place addition of two DSDBCOO matrices."""
        if sparse.issparse(other):
            csr = other.tocsr()
            self.data += csr[
                self.rows + self.global_block_offset,
                self.cols + self.global_block_offset,
            ]
            return self

        self._check_commensurable(other)
        self._data += other._data
        return self

    def __isub__(self, other: "DSDBCOO | sparse.spmatrix") -> "DSDBCOO":
        """In-place subtraction of two DSDBCOO matrices."""
        if sparse.issparse(other):
            csr = other.tocsr()
            self.data -= csr[
                self.rows + self.global_block_offset,
                self.cols + self.global_block_offset,
            ]
            return self

        self._check_commensurable(other)
        self._data -= other._data
        return self

    def __neg__(self) -> "DSDBCOO":
        """Negation of the data."""
        return DSDBCOO(
            local_data=-self.data,
            local_rows=self.rows,
            local_cols=self.cols,
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
            return_dense=self.return_dense,
        )

    # @profiler.profile(level="api")
    # def diagonal(self, val: NDArray = None) -> NDArray:
    #     """Returns or sets the diagonal elements of the matrix.

    #     This temporarily sets the return_dense state to True. Note that
    #     this will cause communication in the block-communicator.

    #     Returns
    #     -------
    #     diagonal : NDArray
    #         The diagonal elements of the whole matrix.

    #     """
    #     if val is None:
    #         # Getter
    #         local_diagonal = xp.zeros(
    #             (*self.shape[:-2], self.shape[-1]), dtype=self.dtype
    #         )
    #         local_diagonal[..., self._diag_inds] = self.data[..., self._diag_mask]
    #         return xp.concatenate(block_comm.allgather(local_diagonal), axis=-1)
    #     else:
    #         # Setter
    #         # NOTE: val must contain the local partition's diagonal.
    #         self.data[..., self._diag_mask] = val[self._diag_inds]

    @DSDBSparse.block_sizes.setter
    def block_sizes(self, block_sizes: NDArray) -> None:
        """Sets new block sizes for the matrix.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        block_sizes = host_xp.asarray(block_sizes, dtype=host_xp.int32)

        if self.distribution_state == "nnz":
            raise NotImplementedError(
                "Cannot reassign block-sizes when distributed through nnz."
            )

        num_blocks = len(block_sizes)
        if num_blocks in self._block_config and num_blocks == self.num_blocks:
            return

        if sum(block_sizes) != self.shape[-1]:
            raise ValueError("Block sizes must sum to matrix shape.")

        block_section_sizes, __ = get_section_sizes(len(block_sizes), block_comm.size)
        block_section_offsets = host_xp.hstack(
            ([0], host_xp.cumsum(block_section_sizes))
        )

        local_block_sizes = block_sizes[block_section_offsets[block_comm.rank] :]
        if sum(local_block_sizes[: block_section_sizes[block_comm.rank]]) != sum(
            self.local_block_sizes[: self.num_local_blocks]
        ):
            raise ValueError(
                f"Block sizes {block_sizes} are inconsistent with the current distribution."
            )

        # Check if configuration already exists.
        if num_blocks in self._block_config:
            # Compute canonical ordering of the matrix.
            inds_bcoo2canonical = xp.lexsort(xp.vstack((self.cols, self.rows)))

            if self._block_config[num_blocks].inds_canonical2block is None:
                canonical_rows = self.rows[inds_bcoo2canonical]
                canonical_cols = self.cols[inds_bcoo2canonical]
                # Compute the index for sorting by the new block-sizes.
                inds_canonical2bcoo = dsbcoo_kernels.compute_block_sort_index(
                    canonical_rows, canonical_cols, local_block_sizes
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

            self.block_section_offsets = block_section_offsets
            # We need to know our local block sizes and those of all
            # subsequent ranks.
            self.num_local_blocks = block_section_sizes[block_comm.rank]
            self.local_block_sizes = block_sizes[
                self.block_section_offsets[block_comm.rank] :
            ]
            self.local_block_offsets = host_xp.hstack(
                ([0], host_xp.cumsum(self.local_block_sizes))
            )

            self.num_blocks = num_blocks
            return

        # Compute canonical ordering of the matrix.
        inds_bcoo2canonical = xp.lexsort(xp.vstack((self.cols, self.rows)))
        canonical_rows = self.rows[inds_bcoo2canonical]
        canonical_cols = self.cols[inds_bcoo2canonical]
        # Compute the index for sorting by the new block-sizes.
        inds_canonical2bcoo = dsbcoo_kernels.compute_block_sort_index(
            canonical_rows, canonical_cols, local_block_sizes
        )
        # Mapping directly from original block-ordering to the new
        # block-ordering is achieved by chaining the two mappings.
        inds_bcoo2bcoo = inds_bcoo2canonical[inds_canonical2bcoo]
        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, inds_bcoo2bcoo]
        self.rows = self.rows[inds_bcoo2bcoo]
        self.cols = self.cols[inds_bcoo2bcoo]

        # Update the block sizes and offsets as in the initializer.
        self.num_blocks = num_blocks

        block_offsets = host_xp.hstack(([0], host_xp.cumsum(block_sizes)))

        self.block_section_offsets = block_section_offsets
        # We need to know our local block sizes and those of all
        # subsequent ranks.
        self.num_local_blocks = block_section_sizes[block_comm.rank]
        self.local_block_sizes = block_sizes[
            self.block_section_offsets[block_comm.rank] :
        ]
        self.local_block_offsets = host_xp.hstack(
            ([0], host_xp.cumsum(self.local_block_sizes))
        )
        self._add_block_config(num_blocks, block_sizes, block_offsets)
        self._block_config[num_blocks].inds_canonical2block = inds_canonical2bcoo

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
        rows = block_comm.allgather(self.rows)
        cols = block_comm.allgather(self.cols)
        rank_max = xp.hstack(
            block_comm.allgather(sum(self.local_block_sizes[: self.num_local_blocks]))
        )
        rank_offset = xp.hstack(([0], xp.cumsum(rank_max)))

        for i in range(1, block_comm.size):
            rows[i] += rank_offset[i]
            cols[i] += rank_offset[i]
        return xp.hstack(rows), xp.hstack(cols)

    def symmetrize(self, op: Callable[[NDArray, NDArray], NDArray] = xp.add) -> None:
        """Symmetrizes the matrix with a given operation.

        This is done by setting the data to the result of the operation
        applied to the data and its conjugate transpose.

        Note
        ----
        This assumes that the matrix's sparsity pattern is symmetric.

        Parameters
        ----------
        op : callable, optional
            The operation to apply to the data and its conjugate
            transpose. Default is `xp.add`, so that the matrix is
            Hermitian after calling.

        """
        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot symmetrize when distributed through nnz.")

        if not hasattr(self, "_inds_bcoo2bcoo_t"):
            # Transpose.
            rows_t, cols_t = self.cols, self.rows

            # Canonical ordering of the transpose.
            inds_bcoo2canonical_t = xp.lexsort(xp.vstack((cols_t, rows_t)))
            canonical_rows_t = rows_t[inds_bcoo2canonical_t]
            canonical_cols_t = cols_t[inds_bcoo2canonical_t]

            # Compute index for sorting the transpose by block.
            inds_canonical2bcoo_t = dsbcoo_kernels.compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.local_block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcoo2bcoo_t = inds_bcoo2canonical_t[inds_canonical2bcoo_t]

            # Cache the necessary objects.
            self._inds_bcoo2bcoo_t = inds_bcoo2bcoo_t

        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = 0.5 * op(
                data[stack_idx], data[stack_idx, self._inds_bcoo2bcoo_t].conj()
            )

    @classmethod
    @profiler.profile(level="api")
    def zeros_like(cls, dsdbsparse: "DSDBCOO") -> "DSDBCOO":
        """Creates a new DSDBCOO matrix with the same shape and dtype.

        All non-zero elements are set to zero, but the sparsity pattern
        is preserved.

        Parameters
        ----------
        dsdbcoo : DSDBCOO
            The matrix to copy the shape and dtype from.

        Returns
        -------
        dsdbcoo
            The new DSDBCOO matrix.

        """
        out = copy.deepcopy(dsdbsparse)
        out.data[:] = 0
        return out

    @classmethod
    @profiler.profile(level="api")
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ) -> "DSDBCOO":
        """Constructs a DSDBCOO matrix from a COO matrix.

        This essentially distributes the COO matrix across the
        participating ranks.

        Parameters
        ----------
        sparray : sparse.coo_matrix
            The COO matrix to distribute.
        block_sizes : NDArray
            The block sizes of the block-sparse matrix.
        global_stack_shape : tuple
            The global shape of the stack.

        Returns
        -------
        DSDBCOO
            The new DSDBCOO matrix.

        """
        if stack_comm is None or block_comm is None:
            raise ValueError("Communicators must be initialized.")

        # We only distribute the first dimension of the stack.
        stack_section_sizes, __ = get_section_sizes(
            global_stack_shape[0], stack_comm.size
        )
        section_size = stack_section_sizes[stack_comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]

        coo: sparse.coo_matrix = sparray.tocoo().copy()

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Compute the block-sorting index.
        block_sort_index = dsbcoo_kernels.compute_block_sort_index(
            coo.row, coo.col, block_sizes
        )

        data = coo.data[block_sort_index]
        rows = coo.row[block_sort_index]
        cols = coo.col[block_sort_index]

        # Determine the local slice of the data.
        # NOTE: This is arrow-wise partitioning.
        # TODO: Allow more options, e.g., block row-wise partitioning.
        section_sizes, __ = get_section_sizes(len(block_sizes), block_comm.size)
        section_offsets = host_xp.hstack(([0], host_xp.cumsum(section_sizes)))

        block_offsets = host_xp.hstack(([0], host_xp.cumsum(block_sizes)))
        start_idx = block_offsets[section_offsets[block_comm.rank]]
        end_idx = block_offsets[section_offsets[block_comm.rank + 1]]
        local_mask = ((rows >= start_idx) & (cols >= start_idx)) & (
            (rows < end_idx) | (cols < end_idx)
        )

        local_data = xp.zeros(
            local_stack_shape + (int(local_mask.sum()),), dtype=coo.data.dtype
        )
        local_data[..., :] = data[local_mask]
        local_rows = rows[local_mask] - start_idx
        local_cols = cols[local_mask] - start_idx

        return cls(
            local_data=local_data,
            local_rows=local_rows,
            local_cols=local_cols,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

    def to_dense(self):
        """Converts the local data to a dense array.

        This is dumb, unless used for testing and debugging.

        Returns
        -------
        arr : NDArray
            The dense array of shape `(*local_stack_shape, *shape)`.

        """
        if self.distribution_state != "stack":
            raise ValueError(
                "Conversion to dense is only supported in 'stack' distribution state."
            )

        # Gather rows, cols, and data.
        rows = block_comm.allgather(self.rows)
        cols = block_comm.allgather(self.cols)
        data = xp.concatenate(block_comm.allgather(self.data), axis=-1)

        rank_max = xp.hstack(
            block_comm.allgather(sum(self.local_block_sizes[: self.num_local_blocks]))
        )
        rank_offset = xp.hstack(([0], xp.cumsum(rank_max)))

        for i in range(1, block_comm.size):
            rows[i] += rank_offset[i]
            cols[i] += rank_offset[i]

        rows = xp.hstack(rows)
        cols = xp.hstack(cols)

        arr = xp.zeros(self.shape, dtype=self.dtype)
        arr[..., rows, cols] = data

        return arr
