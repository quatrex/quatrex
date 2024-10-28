import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.gpu_utils import ArrayLike, get_device, xp
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import compute_block_sort_index


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
        Default is False.

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

    def _normalize_index(self, index: tuple) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        if not isinstance(index, tuple):
            raise IndexError("Invalid index.")

        if not len(index) == 2:
            raise IndexError("Invalid index.")

        row, col = index

        row = self.shape[-2] + row if row < 0 else row
        col = self.shape[-1] + col if col < 0 else col
        if not (0 <= row < self.shape[-2] and 0 <= col < self.shape[-1]):
            raise IndexError("Index out of bounds.")

        return row, col

    def __getitem__(self, index: tuple) -> ArrayLike:
        """Gets a single value or from the data structure."""
        row, col = self._normalize_index(index)
        ind = xp.where((self.rows == row) & (self.cols == col))[0]

        if self.distribution_state == "stack":
            if len(ind) == 0:
                return xp.zeros(self.data.shape[:-1], dtype=self.dtype)

            return self.data[..., ind[0]]

        if len(ind) == 0:
            # We cannot know which rank is supposed to hold an element
            # that is not in the matrix, so we raise an error.
            raise IndexError("Requested element not in matrix.")

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        rank = xp.where(self.nnz_section_offsets <= ind[0])[0][-1]

        if rank == comm.rank:
            return self.data[..., ind[0] - self.nnz_section_offsets[rank]]

        raise IndexError(
            f"Requested data not on this rank ({comm.rank}). It is on rank {rank}."
        )

    def __setitem__(self, index: tuple, value: ArrayLike) -> None:
        """Sets a single value or block in the data structure."""
        row, col = self._normalize_index(index)
        ind = xp.where((self.rows == row) & (self.cols == col))[0]
        if len(ind) == 0:
            # Nothing to do if the element is not in the matrix.
            return

        if self.distribution_state == "stack":
            self.data[..., ind[0]] = value
            return

        # If nnz are distributed accross the stack, we need to find the
        # rank that holds the data.
        rank = xp.where(self.nnz_section_offsets <= ind[0])[0][-1]

        if rank == comm.rank:
            # We need to access the full data buffer directly to set the
            # value since we are using advanced indexing.
            self._data[
                self._stack_padding_mask, ..., ind[0] - self.nnz_section_offsets[rank]
            ] = value
            return

        raise IndexError(
            f"Requested data not on this rank ({comm.rank}). It is on rank {rank}."
        )

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

    def __iadd__(self, other: "DSBSparse | sparse.sparray") -> "DSBCOO":
        """In-place addition of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()
            sparray_data = lil[self.rows, self.cols].toarray()
            self.data[:] += sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] += other.data[:]
        return self

    def __isub__(self, other: "DSBSparse | sparse.sparray") -> "DSBCOO":
        """In-place subtraction of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()
            sparray_data = lil[self.rows, self.cols].toarray()
            self.data[:] -= sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, other: "DSBSparse") -> "DSBCOO":
        """In-place multiplication of two DSBSparse matrices."""
        self._check_commensurable(other)
        self.data[:] *= other.data[:]
        return self

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
        raise NotImplementedError("Matrix multiplication is not implemented.")

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
        arr: sparse.sparray,
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

        coo: sparse.coo_array = arr.tocoo().copy()

        num_blocks = len(block_sizes)
        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

        # Densify the selected blocks.
        for i, j in densify_blocks or []:
            # Unsign the block indices.
            i = num_blocks + i if i < 0 else i
            j = num_blocks + j if j < 0 else j
            if not (0 <= i < num_blocks and 0 <= j < num_blocks):
                raise IndexError("Block index out of bounds.")

            indices = [
                (m + block_offsets[i], n + block_offsets[j])
                for m, n in np.ndindex(block_sizes[i], block_sizes[j])
            ]
            coo.row = np.append(coo.row, [m for m, __ in indices])
            coo.col = np.append(coo.col, [n for __, n in indices])
            coo.data = np.append(coo.data, np.zeros(len(indices), dtype=coo.data.dtype))

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Compute the rowptr map.
        block_sort_index = compute_block_sort_index(
            get_device(coo.row), get_device(coo.col), get_device(block_sizes)
        )

        data = xp.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        data[..., :] = get_device(coo.data)[block_sort_index]
        rows = get_device(coo.row)[block_sort_index]
        cols = get_device(coo.col)[block_sort_index]

        return cls(
            data=data,
            rows=rows,
            cols=cols,
            block_sizes=get_device(block_sizes),
            global_stack_shape=global_stack_shape,
        )
