# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import copy
from typing import Callable

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.kernels.datastructure import dsdbcsr_kernels, dsdbsparse_kernels
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import get_section_sizes

profiler = Profiler()


def _upper_triangle(rows: NDArray, cols: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Returns upper triangular rows and cols."""
    mask = cols < rows
    temp = rows[mask]
    rows[mask] = cols[mask]
    cols[mask] = temp
    return rows, cols, mask


class DSDBCSR(DSDBSparse):
    """A Distributed Stack of Distributed Block-accessible CSR matrices.

    This DSDBSparse implementation uses a block-compressed sparse row
    format to store the sparsity pattern of the matrix. The data is
    sorted by block-row and -column. We use a row pointer map together
    with the column indices to access the blocks efficiently.

    Note
    ----
    It is the caller's responsibility to ensure that the data is
    distributed correctly across the ranks.

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
    symmetry : bool, optional
        Whether the matrix is symmetric. Default is False.
    symmetry_op : callable, optional
        The operation to use for the symmetry. Default is
        `xp.conj`.

    """

    def __init__(
        self,
        data: NDArray,
        cols: NDArray,
        rowptr_map: dict,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        return_dense: bool = True,
        symmetry: bool | None = False,
        symmetry_op: Callable = xp.conj,
    ) -> None:
        """Initializes the DBCSR matrix."""

        if comm.block.size != 1:
            raise NotImplementedError(
                "DSDBCSR is not yet implemented for distributed stacks."
            )

        super().__init__(
            data,
            block_sizes,
            global_stack_shape,
            return_dense,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )

        self.cols = cols.astype(int)
        self.rowptr_map = rowptr_map

        inds = xp.arange(self.shape[-1])
        self._diag_inds, self._diag_value_inds = dsdbcsr_kernels.find_inds(
            self.rowptr_map, xp.asarray(self.block_offsets), self.cols, inds, inds
        )
        ranks = dsdbsparse_kernels.find_ranks(self.nnz_section_offsets, self._diag_inds)
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
        if self.symmetry:
            rows, cols, mask_transposed = _upper_triangle(rows, cols)

            inds, value_inds = dsdbcsr_kernels.find_inds(
                self.rowptr_map,
                xp.asarray(self.block_offsets),
                self.cols,
                rows[~mask_transposed],
                cols[~mask_transposed],
            )
            value_inds = (~mask_transposed).nonzero()[0][value_inds]
            inds_t, value_inds_t = dsdbcsr_kernels.find_inds(
                self.rowptr_map,
                xp.asarray(self.block_offsets),
                self.cols,
                rows[mask_transposed],
                cols[mask_transposed],
            )  # need to split the function call into two, because we might want to get (i,j) and (j,i) at the same time
            value_inds_t = mask_transposed.nonzero()[0][value_inds_t]
        else:
            inds, value_inds = dsdbcsr_kernels.find_inds(
                self.rowptr_map, xp.asarray(self.block_offsets), self.cols, rows, cols
            )

        data_stack = self.data[stack_index]
        if self.distribution_state == "stack":
            arr = xp.zeros(data_stack.shape[:-1] + (rows.size,), dtype=self.dtype)

            if self.symmetry:
                arr[..., value_inds] = data_stack[..., inds]
                arr[..., value_inds_t] = self.symmetry_op(data_stack[..., inds_t])
            else:
                arr[..., value_inds] = data_stack[..., inds]
            return xp.squeeze(arr)

        if len(inds) != rows.size:
            # We cannot know which rank is supposed to hold an element
            # that is not in the matrix, so we raise an error.
            raise IndexError("Requested element not in matrix.")

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        ranks = dsdbsparse_kernels.find_ranks(self.nnz_section_offsets, inds)

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
        if self.symmetry:
            # items of upper triangle of the matrix
            rows, cols, mask_transposed = _upper_triangle(rows, cols)

        inds, value_inds = dsdbcsr_kernels.find_inds(
            self.rowptr_map, xp.asarray(self.block_offsets), self.cols, rows, cols
        )

        if len(inds) == 0:
            # Nothing to do if the element is not in the matrix.
            return

        value = xp.asarray(value)
        if self.distribution_state == "stack":
            if value.ndim == 0:
                self.data[*stack_index][..., inds] = value
                if self.symmetry:
                    self.data[*stack_index][..., inds[mask_transposed]] = (
                        self.symmetry_op(value)
                    )
                return

            self.data[*stack_index][..., inds] = value[..., value_inds]
            return

        # If nnz are distributed accross the stack, we need to find the
        # rank that holds the data.
        ranks = dsdbsparse_kernels.find_ranks(self.nnz_section_offsets, inds)

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
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`
            if `return_dense` is True, otherwise it is a tuple of three
            arrays `(rowptr, cols, data)`.

        """
        if self.symmetry and (col < row):
            block = self._get_block(arg, row=col, col=row, is_index=is_index)
            return xp.ascontiguousarray(self.symmetry_op(block.swapaxes(-1, -2)))

        if is_index:
            data_stack = self.data[*arg]
        else:
            data_stack = arg

        rowptr = self.rowptr_map.get((row, col), None)

        if not self.return_dense:
            if self.symmetry:
                # TODO: If really needed, this will need some more thinking.
                raise IndexError("Not implemented")
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

        dsdbcsr_kernels.densify_block(
            block=block,
            block_offset=self.block_offsets[col],
            self_cols=self.cols,
            rowptr=rowptr,
            data=data_stack,
        )
        if self.symmetry and (col == row):
            block += self.symmetry_op(block.swapaxes(-1, -2))
            block[..., *xp.diag_indices(block.shape[-1])] /= 2

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
        if self.symmetry:
            # TODO: If really needed, this will need some more thinking.
            raise NotImplementedError("Not implemented")
        if is_index:
            data_stack = self.data[*arg]
        else:
            data_stack = arg
        rowptr = self.rowptr_map.get((row, col), None)

        if rowptr is None:
            # No data in this block, return zeros.
            return (
                xp.empty(data_stack.shape[:-1] + (0,)),
                xp.empty(0),
                xp.zeros(int(self.block_sizes[row]) + 1),
            )

        cols = self.cols[rowptr[0] : rowptr[-1]] - self.block_offsets[col]
        return data_stack[..., rowptr[0] : rowptr[-1]], cols, rowptr - rowptr[0]

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
        if self.symmetry and (col < row):
            # TODO: Probably worth testing if the block is symmetric.
            self._set_block(
                arg,
                row=col,
                col=row,
                block=self.symmetry_op(block.swapaxes(-1, -2)),
                is_index=is_index,
            )

        if is_index:
            data_stack = self.data[*arg]
        else:
            data_stack = arg

        rowptr = self.rowptr_map.get((row, col), None)
        if rowptr is None:
            # No data in this block, nothing to do.
            return

        dsdbcsr_kernels.sparsify_block(
            block=block,
            block_offset=self.block_offsets[col],
            self_cols=self.cols,
            rowptr=rowptr,
            data=data_stack,
        )

    @profiler.profile(level="debug")
    def _check_commensurable(self, other: "DSDBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSDBCSR):
            raise TypeError("Can only add DSDBCSR matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if np.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")

        if self.rowptr_map.keys() != other.rowptr_map.keys():
            raise ValueError("Block sparsities do not match.")

        if xp.any(self.cols != other.cols):
            raise ValueError("Column indices do not match.")

    def __iadd__(self, other: "DSDBCSR | sparse.spmatrix") -> "DSDBCSR":
        """In-place addition of two DSDBCSR matrices."""
        if sparse.issparse(other):
            raise NotImplementedError(
                "In-place addition is not implemented for DSDBCSR matrices."
            )

        self._check_commensurable(other)
        self._data += other._data
        return self

    def __isub__(self, other: "DSDBCSR | sparse.spmatrix") -> "DSDBCSR":
        """In-place subtraction of two DSDBCSR matrices."""
        if sparse.issparse(other):
            raise NotImplementedError(
                "In-place subtraction is not implemented for DSDBCSR matrices."
            )

        self._check_commensurable(other)
        self._data -= other._data
        return self

    def __neg__(self) -> "DSDBCSR":
        """Negation of the data."""
        return DSDBCSR(
            data=-self.data,
            cols=self.cols,
            rowptr_map=self.rowptr_map,
            block_sizes=self.block_sizes,
            global_stack_shape=self.global_stack_shape,
            return_dense=self.return_dense,
            symmetry=self.symmetry,
            symmetry_op=self.symmetry_op,
        )

    @DSDBSparse.block_sizes.setter
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

            if num_blocks == self.num_blocks:
                return

            if self._block_config[num_blocks].inds_canonical2block is None:
                rows, cols = self.spy()
                inds_bcsr2canonical = xp.lexsort(xp.vstack((cols, rows)))
                canonical_rows = rows[inds_bcsr2canonical]
                canonical_cols = cols[inds_bcsr2canonical]
                # Compute the index for sorting by the new block-sizes.
                inds_canonical2bcsr, rowptr_map = dsdbcsr_kernels.compute_rowptr_map(
                    canonical_rows, canonical_cols, block_sizes
                )
                self._block_config[num_blocks].inds_canonical2block = (
                    inds_canonical2bcsr
                )
                self._block_config[num_blocks].rowptr_map = rowptr_map

            self.rowptr_map = self._block_config[num_blocks].rowptr_map

            # Mapping directly from original block-ordering to the new
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr = inds_bcsr2canonical[
                self._block_config[num_blocks].inds_canonical2block
            ]
            data = self.data.reshape(-1, self.data.shape[-1])
            for stack_idx in range(data.shape[0]):
                data[stack_idx] = data[stack_idx, inds_bcsr2bcsr]
            self.cols = self.cols[inds_bcsr2bcsr]

            self.num_blocks = num_blocks
            return

        if sum(block_sizes) != self.shape[-1]:
            raise ValueError("Block sizes do not match matrix shape.")
        rows, cols = self.spy()
        # Compute canonical ordering of the matrix.
        inds_bcsr2canonical = xp.lexsort(xp.vstack((cols, rows)))
        canonical_rows = rows[inds_bcsr2canonical]
        canonical_cols = cols[inds_bcsr2canonical]
        # Compute the index for sorting by the new block-sizes.
        inds_canonical2bcsr, rowptr_map = dsdbcsr_kernels.compute_rowptr_map(
            canonical_rows, canonical_cols, block_sizes
        )
        self.rowptr_map = rowptr_map
        # Mapping directly from original block-ordering to the new
        # block-ordering is achieved by chaining the two mappings.
        inds_bcsr2bcsr = inds_bcsr2canonical[inds_canonical2bcsr]
        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, inds_bcsr2bcsr]
        self.cols = self.cols[inds_bcsr2bcsr]

        block_sizes = np.asarray(block_sizes, dtype=np.int32)
        block_offsets = np.hstack(([0], np.cumsum(block_sizes)), dtype=np.int32)
        self.num_blocks = num_blocks
        self._add_block_config(self.num_blocks, block_sizes, block_offsets)

    @profiler.profile(level="api")
    def ltranspose(self, copy=False) -> "None | DSDBCSR":
        """Performs a local transposition of the matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a new object. Default is False.

        Returns
        -------
        None | DSDBCSR
            The transposed matrix. If copy is False, this is None.

        """
        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

        if self.symmetry:
            if copy:
                self = DSDBCSR(
                    self.data.copy(),
                    self.cols.copy(),
                    self.rowptr_map.copy(),
                    self.block_sizes,
                    self.global_stack_shape,
                    symmetry=self.symmetry,
                    symmetry_op=self.symmetry_op,
                )
            self.data[:] = self.symmetry_op(self.data)
            return self if copy else None

        if copy:
            self = DSDBCSR(
                self.data.copy(),
                self.cols.copy(),
                self.rowptr_map.copy(),
                self.block_sizes,
                self.global_stack_shape,
                symmetry=self.symmetry,
                symmetry_op=self.symmetry_op,
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
            inds_canonical2bcsr_t, rowptr_map_t = dsdbcsr_kernels.compute_rowptr_map(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Cache the necessary objects.
            self._inds_bcsr2bcsr_t = inds_bcsr2bcsr_t
            self._rowptr_map_t = rowptr_map_t
            self._cols_t = cols_t[self._inds_bcsr2bcsr_t]

        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, self._inds_bcsr2bcsr_t]
        self._inds_bcsr2bcsr_t = xp.argsort(self._inds_bcsr2bcsr_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rowptr_map, self._rowptr_map_t = self._rowptr_map_t, self.rowptr_map

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
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

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
            inds_canonical2bcsr_t, rowptr_map_t = dsdbcsr_kernels.compute_rowptr_map(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Cache the necessary objects.
            self._inds_bcsr2bcsr_t = inds_bcsr2bcsr_t
            self._rowptr_map_t = rowptr_map_t
            self._cols_t = cols_t[self._inds_bcsr2bcsr_t]

        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = 0.5 * op(
                data[stack_idx], data[stack_idx, self._inds_bcsr2bcsr_t].conj()
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
        rows = xp.zeros(self.cols.size, dtype=int)
        for (row, __), rowptr in self.rowptr_map.items():
            for i in range(int(self.block_sizes[row])):
                rows[rowptr[i] : rowptr[i + 1]] = i + self.block_offsets[row]

        return rows, self.cols

    @classmethod
    @profiler.profile(level="api")
    def zeros_like(cls, dsdbsparse: "DSDBSparse") -> "DSDBSparse":
        """Creates a new DSDBSparse matrix with the same shape and dtype.

        All non-zero elements are set to zero, but the sparsity pattern
        is preserved.

        Parameters
        ----------
        dsdbsparse : DSDBSparse
            The matrix to copy the shape and dtype from.

        Returns
        -------
        DSDBSparse
            The new DSDBSparse matrix.

        """
        # TODO: deepcopy should be removed
        # Problem with symmetry operators
        out = copy.deepcopy(dsdbsparse)
        if out._data is None:
            out.allocate_data()
        out._data[:] = 0.0
        return out

    @classmethod
    @profiler.profile(level="api")
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: bool | None = False,
        symmetry_op: Callable = xp.conj,
    ) -> "DSDBCSR":
        """Creates a new DSDBSparse matrix from a scipy.sparse array.

        Parameters
        ----------
        arr : sparse.spmatrix
            The sparse array to convert.
        block_sizes : NDArray
            The size of all the blocks in the matrix.
        global_stack_shape : tuple
            The global shape of the stack of matrices. The provided
            sparse matrix is replicated across the stack.
        symmetry : bool, optional
            Whether to enforce symmetry in the matrix. Default is False.
        symmetry_op : callable, optional
            The operation to use for the symmetry. Default is
            `xp.conj`.

        Returns
        -------
        DSDBCSR
            The new DSDBCSR matrix.

        """
        # We only distribute the first dimension of the stack.
        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]

        coo: sparse.coo_matrix = arr.tocoo().copy()

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        if symmetry:
            coo = sparse.triu(coo, format="coo")

        # Compute block sorting index and the transpose rowptr map.
        block_sort_index, rowptr_map = dsdbcsr_kernels.compute_rowptr_map(
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
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )

    @profiler.profile(level="api")
    def to_dense(self) -> NDArray:
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

        original_return_dense = self.return_dense
        self.return_dense = True

        arr = xp.zeros(self.shape, dtype=self.dtype)
        for i, j in xp.ndindex(self.num_blocks, self.num_blocks):
            arr[
                ...,
                self.block_offsets[i] : self.block_offsets[i + 1],
                self.block_offsets[j] : self.block_offsets[j + 1],
            ] = self._get_block((Ellipsis,), i, j)

        self.return_dense = original_return_dense

        return arr
