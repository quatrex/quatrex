# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the distributed block-accessible CSR matrix data structure."""

import warnings

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, symmetry_ops
from qttools.kernels.datastructure import dsdbcsr_kernels, dsdbsparse_kernels
from qttools.utils.mpi_utils import get_section_sizes


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
    dtype : xp.dtype[xp.generic]
        The data type of the matrix.
    cols : NDArray
        The column indices.
    rowptr_map : dict
        The row pointer map.
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
        cols: NDArray,
        rowptr_map: dict,
        block_sizes: NDArray,
        local_stack_shape: tuple | int,
        global_stack_shape: tuple,
        symmetry: str | None = None,
    ) -> None:
        """Initializes the DBCSR matrix."""

        # NOTE: The idea is that outside scipy decides
        # the correct data type for the indices.
        # We just need to make sure to follow suite.
        index_type = cols.dtype
        for item in rowptr_map.values():
            if item.dtype != index_type:
                raise TypeError(
                    "rowptr map and column indices must have the same dtype "
                    f"but got {item.dtype} and {cols.dtype}."
                )

        if comm.block.size != 1:
            raise NotImplementedError(
                "DSDBCSR is not yet implemented for distributed stacks."
            )

        super().__init__(
            dtype=dtype,
            block_sizes=block_sizes,
            nnz=len(cols),
            local_stack_shape=local_stack_shape,
            global_stack_shape=global_stack_shape,
            index_type=index_type,
            symmetry=symmetry,
        )

        self.cols = cols
        self.rowptr_map = rowptr_map

        self._set_diagonal_indices()

    def _set_diagonal_indices(self) -> None:
        """Sets the diagonal indices of the matrix."""
        inds = xp.arange(self.shape[-1], dtype=self.index_type)
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

    def _get_block(self, stack_index: tuple, row: int, col: int) -> NDArray:
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

        rowptr = self.rowptr_map.get((row, col), None)

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

        data_stack = self.data[*stack_index]

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

        # NOTE: caching is not implemented for CSR

        # Compute canonical ordering of the matrix.
        rows, cols = self.spy()
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
        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

        self._add_block_config(num_blocks, block_sizes, block_offsets)

        # NOTE: The batched loop is due to the fancy indexing consuming
        # a lot of memory.
        data = self.data.reshape(-1, self.data.shape[-1])
        for stack_idx in range(data.shape[0]):
            data[stack_idx] = data[stack_idx, inds_bcsr2bcsr]
        self.cols = self.cols[inds_bcsr2bcsr]

        # Update the block sizes and offsets as in the initializer.
        self.num_blocks = num_blocks
        self.block_section_offsets = block_section_offsets
        # We need to know our local block sizes and those of all
        # subsequent ranks.
        self.num_local_blocks = num_local_blocks
        self.local_block_sizes = local_block_sizes
        self.local_block_offsets = np.hstack(([0], np.cumsum(self.local_block_sizes)))
        # self.global_block_offset is already set in the initializer and does not change.

        self._set_diagonal_indices()

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
            data[stack_idx] = 0.5 * (
                symmetry_ops[symmetry](data[stack_idx, self._inds_bcsr2bcsr_t])
                + data[stack_idx]
            )

    def spy(self) -> tuple[NDArray, NDArray]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Note
        ----
        In the block distributed case, this returns the local
        sparsity pattern.

        Warning
        -------
        This not performant.

        Returns
        -------
        rows : NDArray
            Row indices of the non-zero elements.
        cols : NDArray
            Column indices of the non-zero elements.

        """
        if comm.rank == 0:
            warnings.warn("The spy method is not efficient for large matrices.")

        rows = xp.zeros(self.cols.size, dtype=self.index_type)
        for (row, __), rowptr in self.rowptr_map.items():
            for i in range(int(self.block_sizes[row])):
                rows[rowptr[i] : rowptr[i + 1]] = i + self.block_offsets[row]

        return rows + self.global_block_offset, self.cols + self.global_block_offset

    @classmethod
    def empty_like(cls, dsdbsparse: "DSDBCSR") -> "DSDBCSR":
        """Creates a new DSDBCSR matrix with the same shape and dtype.

        Note
        ----
        There is no data allocated in the new matrix. The sparsity
        pattern is the same as the original matrix.

        Parameters
        ----------
        dsdbsparse : DSDBCSR
            The matrix to copy the shape and dtype from.

        Returns
        -------
        DSDBCSR
            The new DSDBCSR matrix.

        """
        return cls(
            dtype=dsdbsparse.dtype,
            cols=dsdbsparse.cols.copy(),
            rowptr_map=dsdbsparse.rowptr_map.copy(),
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
    ) -> "DSDBCSR":
        """Creates a new DSDBCSR matrix from a scipy.sparse array.

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
        DSDBCSR
            The new DSDBCSR matrix.

        """

        # We only distribute the first dimension of the stack.
        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]

        coo: sparse.coo_matrix = sparray.tocoo().copy()

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        if symmetry:
            coo = sparse.triu(coo, format="coo")

        # Compute block sorting index and the transpose rowptr map.
        index_type = coo.row.dtype
        block_sort_index, rowptr_map = dsdbcsr_kernels.compute_rowptr_map(
            coo.row, coo.col, block_sizes.astype(index_type)
        )
        cols = coo.col[block_sort_index]

        dsdbcsr = cls(
            dtype=dtype,
            cols=cols,
            rowptr_map=rowptr_map,
            block_sizes=block_sizes,
            local_stack_shape=local_stack_shape,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )
        if allocate:
            dsdbcsr.allocate_data()
            dsdbcsr._data[:] = 0
            dsdbcsr.data = coo.data[block_sort_index]

        return dsdbcsr

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

        arr = xp.zeros(self.shape, dtype=self.dtype)
        for i, j in xp.ndindex(self.num_blocks, self.num_blocks):
            arr[
                ...,
                self.block_offsets[i] : self.block_offsets[i + 1],
                self.block_offsets[j] : self.block_offsets[j + 1],
            ] = self._get_block((Ellipsis,), i, j)

        return arr
