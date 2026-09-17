# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the distributed block-accessible COO matrix data structure."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, symmetry_ops
from qttools.kernels.datastructure import dsdbcoo_kernels, dsdbsparse_kernels
from qttools.utils.mpi_utils import get_section_sizes


class DSDBCOO(DSDBSparse):
    """A Distributed Stack of Distributed Block-accessible COO matrices.

    Note
    ----
    It is the caller's responsibility to ensure that the data is
    distributed correctly across the ranks.

    Parameters
    ----------
    dtype : xp.dtype[xp.generic]
        The data type of the matrix.
    rows : NDArray
        The local row indices of the COO matrix.
    cols : NDArray
        The local column indices of the COO matrix.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    local_stack_shape : tuple or int
        The local shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    global_stack_shape : tuple or int
        The global shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    symmetry : str | None, optional
        The symmetry of the matrix. This can be "symmetric",
        "hermitian", "skew-symmetric", "skew-hermitian", or None.
        Default is None.

    """

    def __init__(
        self,
        dtype: xp.dtype[xp.generic],
        rows: NDArray,
        cols: NDArray,
        block_sizes: NDArray,
        local_stack_shape: tuple | int,
        global_stack_shape: tuple | int,
        symmetry: str | None = None,
    ):
        """Initializes a DSDBCOO matrix."""

        # NOTE: The idea is that outside scipy decides
        # the correct data type for the indices.
        # We just need to make sure to follow.
        index_type = rows.dtype
        if cols.dtype != index_type:
            raise TypeError(
                "Row and column indices must have the same dtype "
                f"but got {rows.dtype} and {cols.dtype}."
            )
        if len(rows) != len(cols):
            raise ValueError(
                "Row and column indices must have the same length "
                f"but got {len(rows)} and {len(cols)}."
            )

        super().__init__(
            dtype=dtype,
            block_sizes=block_sizes,
            nnz=len(rows),
            local_stack_shape=local_stack_shape,
            global_stack_shape=global_stack_shape,
            index_type=index_type,
            symmetry=symmetry,
        )

        self.rows = rows
        self.cols = cols

        # NOTE: If the symmetry is not enforced and we want to symmetrize
        # later, we need to check if the sparsity pattern is symmetric now.
        if symmetry is None:
            self._symmetric_pattern = self._check_sparsity_pattern_symmetric()

        self._set_nnz_indices()
        self._set_diagonal_indices()

    def _set_nnz_indices(self) -> None:
        """Sets the `nnz` distributed local indices of the matrix."""
        self.rows_nnz = self.rows[
            self.nnz_section_offsets[comm.stack.rank] : self.nnz_section_offsets[
                comm.stack.rank + 1
            ]
        ] + self.index_type.type(self.global_block_offset)
        self.cols_nnz = self.cols[
            self.nnz_section_offsets[comm.stack.rank] : self.nnz_section_offsets[
                comm.stack.rank + 1
            ]
        ] + self.index_type.type(self.global_block_offset)

    def _set_diagonal_indices(self) -> None:
        """Sets the diagonal indices of the matrix."""

        if self.num_blocks in self._diag_cache:
            (
                self._diag_inds,
                self._diag_value_inds,
                self._diag_inds_nnz,
                self._diag_value_inds_nnz,
            ) = self._diag_cache[self.num_blocks]
            return

        self._diag_inds = xp.where(self.rows == self.cols)[0].astype(self.index_type)
        self._diag_value_inds = self.rows[self._diag_inds]
        ranks = dsdbsparse_kernels.find_ranks(self.nnz_section_offsets, self._diag_inds)

        if not any(ranks == comm.stack.rank):
            self._diag_inds_nnz = None
            self._diag_value_inds_nnz = None
        else:
            self._diag_inds_nnz = (
                self._diag_inds[ranks == comm.stack.rank]
                - self.nnz_section_offsets[comm.stack.rank]
            )
            self._diag_value_inds_nnz = (
                self._diag_value_inds[ranks == comm.stack.rank]
                - self._diag_value_inds[ranks == comm.stack.rank][0]
            )
        self._diag_cache[self.num_blocks] = (
            self._diag_inds,
            self._diag_value_inds,
            self._diag_inds_nnz,
            self._diag_value_inds_nnz,
        )

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
                *dsdbcoo_kernels.compute_block_slice(
                    self.rows, self.cols, self.local_block_offsets, row, col
                )
            )
            self._block_config[self.num_blocks].block_slice_cache[
                (row, col)
            ] = block_slice

        return block_slice

    def _get_block(
        self,
        stack_index: tuple,
        row: int,
        col: int,
    ) -> NDArray:
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
        block : NDArray
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        if self.symmetry and (col < row):
            block = self._get_block(stack_index, row=col, col=row)
            return xp.ascontiguousarray(
                symmetry_ops[self.symmetry](block.swapaxes(-1, -2))
            )

        data_stack = self.data[*stack_index]

        block_slice = self._get_block_slice(row, col)

        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.local_block_sizes[row]), int(self.local_block_sizes[col])),
            dtype=self.dtype,
        )
        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, return an empty block.
            return block

        dsdbcoo_kernels.densify_block(
            block,
            self.rows,
            self.cols,
            data_stack,
            block_slice,
            self.local_block_offsets[row],
            self.local_block_offsets[col],
        )
        if self.symmetry and (col == row):
            block += symmetry_ops[self.symmetry](block.swapaxes(-1, -2))
            block[..., *xp.diag_indices(block.shape[-1])] /= 2

        return block

    def _set_block(
        self,
        stack_index: tuple,
        row: int,
        col: int,
        block: NDArray,
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Note
        ----
        The input block is not tested for symmetry even if the matrix is
        symmetric.

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
        if self.symmetry and (col < row):
            self._set_block(
                stack_index,
                row=col,
                col=row,
                block=symmetry_ops[self.symmetry](block.swapaxes(-1, -2)),
            )
            return

        data_stack = self.data[*stack_index]

        block_slice = self._get_block_slice(row, col)
        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, nothing to do.
            return

        dsdbcoo_kernels.sparsify_block(
            block,
            self.rows[block_slice] - self.local_block_offsets[row],
            self.cols[block_slice] - self.local_block_offsets[col],
            data_stack[..., block_slice],
        )

    @DSDBSparse.block_sizes.setter
    def block_sizes(self, block_sizes: NDArray) -> None:
        """Sets new block sizes for the matrix.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        num_blocks = len(block_sizes)

        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))
        local_block_sizes = block_sizes[block_section_offsets[comm.block.rank] :]
        num_local_blocks = block_section_sizes[comm.block.rank]

        if sum(local_block_sizes[:num_local_blocks]) != sum(
            self.local_block_sizes[: self.num_local_blocks]
        ):
            raise ValueError(
                f"Block sizes {block_sizes} are inconsistent with the current distribution."
            )

        if self.distribution_state == "nnz":
            raise NotImplementedError(
                "Cannot reassign block-sizes when distributed through nnz."
            )

        if num_blocks in self._block_config and num_blocks == self.num_blocks:
            return

        if sum(block_sizes) != self.shape[-1]:
            raise ValueError("Block sizes must sum to matrix shape.")

        if (self.num_blocks, num_blocks) in self._block_change_cache:
            inds_bcoo2bcoo = self._block_change_cache[self.num_blocks, num_blocks]

        else:
            # Compute canonical ordering of the matrix.
            inds_bcoo2canonical = xp.lexsort(xp.vstack((self.cols, self.rows)))
            canonical_rows = self.rows[inds_bcoo2canonical]
            canonical_cols = self.cols[inds_bcoo2canonical]
            # Compute the index for sorting by the new block-sizes.
            inds_canonical2bcoo = dsdbcoo_kernels.compute_block_sort_index(
                canonical_rows, canonical_cols, local_block_sizes
            )
            # Mapping directly from original block-ordering to the new
            # block-ordering is achieved by chaining the two mappings.
            inds_bcoo2bcoo = inds_bcoo2canonical[inds_canonical2bcoo]
            block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

            self._add_block_config(num_blocks, block_sizes, block_offsets)
            self._block_change_cache[self.num_blocks, num_blocks] = inds_bcoo2bcoo

        # NOTE: The batched loop is due to the fancy indexing consuming
        # a lot of memory.
        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, inds_bcoo2bcoo]
        self.rows = self.rows[inds_bcoo2bcoo]
        self.cols = self.cols[inds_bcoo2bcoo]

        # Update the block sizes and offsets as in the initializer.
        self.num_blocks = num_blocks
        self.block_section_offsets = block_section_offsets
        # We need to know our local block sizes and those of all
        # subsequent ranks.
        self.num_local_blocks = num_local_blocks
        self.local_block_sizes = local_block_sizes
        self.local_block_offsets = np.hstack(([0], np.cumsum(self.local_block_sizes)))
        # self.global_block_offset is already set in the initializer and does not change.

        self._set_nnz_indices()
        self._set_diagonal_indices()

    def spy(self) -> tuple[NDArray, NDArray]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Note
        ----
        In the block distributed case, this returns the local
        sparsity pattern.

        Returns
        -------
        rows : NDArray
            Row indices of the non-zero elements.
        cols : NDArray
            Column indices of the non-zero elements.

        """
        return (
            self.rows + self.global_block_offset,
            self.cols + self.global_block_offset,
        )

    def _check_sparsity_pattern_symmetric(self) -> bool:
        """Checks if the sparsity pattern is symmetric.

        Returns
        -------
        bool
            Whether the sparsity pattern is symmetric.

        """
        # NOTE: This uses the global shape, but the local rows and cols.
        # In the block distributed case, this is an upper bound, but sufficient for the check.
        num_rows = self.shape[-2]

        # NOTE: This is upcasted to not overflow
        idx_original = (self.rows.astype(xp.int64) * num_rows) + self.cols
        idx_swapped = (self.cols.astype(xp.int64) * num_rows) + self.rows

        idx_original.sort()
        idx_swapped.sort()

        return xp.all(idx_original == idx_swapped)

    def symmetrize(self, symmetry: str) -> None:
        """Symmetrizes the matrix with a given symmetry.

        Note
        ----
        This assumes that the matrix's sparsity pattern is symmetric.

        Parameters
        ----------
        symmetry : str
            The symmetry to enforce. This can be "symmetric",
            "hermitian", "skew-symmetric", or "skew-hermitian".

        """
        if symmetry not in symmetry_ops:
            raise ValueError(
                f"Symmetry must be one of {list(symmetry_ops.keys())} but got {symmetry}."
            )

        if self.symmetry is not None:
            if symmetry != self.symmetry:
                raise ValueError(
                    f"Matrix is already {self.symmetry}. Cannot enforce {symmetry}."
                )
            # Already symmetric, nothing to do.
            return

        if self.distribution_state == "nnz":
            raise NotImplementedError("Cannot symmetrize when distributed through nnz.")

        if not self._symmetric_pattern:
            raise ValueError(
                "Sparsity pattern is not symmetric. This will lead to incorrect results."
            )

        if not hasattr(self, "_inds_bcoo2bcoo_t"):
            # Transpose.
            rows_t, cols_t = self.cols, self.rows

            # Canonical ordering of the transpose.
            inds_bcoo2canonical_t = xp.lexsort(xp.vstack((cols_t, rows_t)))
            canonical_rows_t = rows_t[inds_bcoo2canonical_t]
            canonical_cols_t = cols_t[inds_bcoo2canonical_t]

            # Compute index for sorting the transpose by block.
            inds_canonical2bcoo_t = dsdbcoo_kernels.compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.local_block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcoo2bcoo_t = inds_bcoo2canonical_t[inds_canonical2bcoo_t]

            # Cache the necessary objects.
            self._inds_bcoo2bcoo_t = inds_bcoo2bcoo_t

        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = 0.5 * (
                symmetry_ops[symmetry](data[stack_idx, self._inds_bcoo2bcoo_t])
                + data[stack_idx]
            )

    @classmethod
    def empty_like(cls, dsdbsparse: "DSDBCOO") -> "DSDBCOO":
        """Creates a new DSDBCOO matrix with the same shape and
        dtype.

        Note
        ----
        There is no data allocated in the new matrix. The sparsity
        pattern is the same as the original matrix.

        Parameters
        ----------
        dsdbsparse : DSDBCOO
            The matrix to copy the shape and dtype from.

        Returns
        -------
        DSDBCOO
            The new DSDBCOO matrix.

        """
        return cls(
            dtype=dsdbsparse.dtype,
            rows=dsdbsparse.rows,
            cols=dsdbsparse.cols,
            block_sizes=dsdbsparse.block_sizes,
            local_stack_shape=dsdbsparse.local_stack_shape,
            global_stack_shape=dsdbsparse.global_stack_shape,
            symmetry=dsdbsparse.symmetry,
        )

    @classmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] = xp.complex128,
        allocate: bool = True,
    ) -> "DSDBCOO":
        """Constructs a DSDBCOO matrix from a sparse matrix.

        This essentially distributed the matrix across the stack and
        block communicators.

        Parameters
        ----------
        sparray : sparse.spmatrix
            The sparse matrix from which to use the sparsity pattern.
        block_sizes : NDArray
            The block sizes of the block-sparse matrix.
        global_stack_shape : tuple
            The global shape of the stack.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian", or None.
            Default is None.
        dtype : xp.dtype, optional
            The data type of the matrix. Default is `xp.complex128`.
        allocate : bool, optional
            Whether to allocate the data of the resulting matrix.
            Default is True.

        Returns
        -------
        DSDBCOO
            The new DSDBCOO matrix.

        """

        if comm.stack is None or comm.block is None:
            raise ValueError("Communicators must be initialized.")

        # We only distribute the first dimension of the stack.
        stack_section_sizes, __ = get_section_sizes(
            global_stack_shape[0], comm.stack.size
        )
        section_size = stack_section_sizes[comm.stack.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]

        coo: sparse.coo_matrix = sparray.tocoo()

        # Canonicalizes the COO format.
        if symmetry:
            coo = sparse.triu(coo, format="coo")

        if not coo.has_canonical_format:
            coo.sum_duplicates()

        # Compute the block-sorting index.
        block_sort_index = dsdbcoo_kernels.compute_block_sort_index(
            coo.row, coo.col, block_sizes
        )

        _rows = coo.row[block_sort_index]
        _cols = coo.col[block_sort_index]

        # Determine the local slice of the data.
        # NOTE: This is arrow-wise partitioning.
        section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        section_offsets = np.hstack(([0], np.cumsum(section_sizes)))

        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
        start_idx = block_offsets[section_offsets[comm.block.rank]]
        end_idx = block_offsets[section_offsets[comm.block.rank + 1]]
        local_mask = ((_rows >= start_idx) & (_cols >= start_idx)) & (
            (_rows < end_idx) | (_cols < end_idx)
        )

        rows = _rows[local_mask] - start_idx
        cols = _cols[local_mask] - start_idx

        dsdbcoo = cls(
            dtype=dtype,
            rows=rows,
            cols=cols,
            block_sizes=block_sizes,
            local_stack_shape=local_stack_shape,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        if allocate:
            _data = coo.data[block_sort_index]
            dsdbcoo.allocate_data()
            dsdbcoo.data = _data[local_mask]

        return dsdbcoo

    def to_dense(self):
        """Converts the local data to a dense array.

        This is dumb, unless used for testing and debugging.

        Warning
        -------
        This creates a very large dense matrix.

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
        rows = comm.block._mpi_comm.allgather(self.rows)
        cols = comm.block._mpi_comm.allgather(self.cols)
        data = xp.concatenate(comm.block._mpi_comm.allgather(self.data), axis=-1)

        rank_max = xp.hstack(
            comm.block._mpi_comm.allgather(
                sum(self.local_block_sizes[: self.num_local_blocks])
            )
        )
        rank_offset = xp.hstack(([0], xp.cumsum(rank_max)))

        for i in range(1, comm.block.size):
            rows[i] += rank_offset[i]
            cols[i] += rank_offset[i]

        rows = xp.hstack(rows)
        cols = xp.hstack(cols)

        arr = xp.zeros(self.shape, dtype=self.dtype)
        arr[..., rows, cols] = data

        if self.symmetry is not None:
            arr += symmetry_ops[self.symmetry](arr.swapaxes(-1, -2))
            arr[..., *xp.diag_indices(arr.shape[-1])] /= 2

        return arr
