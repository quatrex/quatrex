import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import compute_block_sort_index


class DSBCOO(DSBSparse):
    """Distributed stack of sparse matrices in coordinate format.

    This DSBSparse implementation stores the matrix sparsity pattern in
    probably the most straight-forward way: as a list of coordinates.
    Both data and coordinates are sorted by block-row and -column.

    Parameters
    ----------
    data : np.ndarray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    rows : np.ndarray
        The row indices.
    cols : np.ndarray
        The column indices.
    block_sizes : np.ndarray
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
        data: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DBCOO matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.rows = np.asarray(rows).astype(int)
        self.cols = np.asarray(cols).astype(int)

    def _get_block(self, row: int, col: int) -> np.ndarray:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
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
        block = np.zeros(
            (
                *self.stack_shape,  # Stack dimensions.
                self.block_sizes[row],
                self.block_sizes[col],
            ),
            dtype=self.dtype,
        )
        mask = (
            (self.rows >= self.block_offsets[row])
            & (self.rows < self.block_offsets[row + 1])
            & (self.cols >= self.block_offsets[col])
            & (self.cols < self.block_offsets[col + 1])
        )

        if not np.any(mask):
            # No data in this block, return zeros.
            return block

        block[
            ...,
            self.rows[mask] - self.block_offsets[row],
            self.cols[mask] - self.block_offsets[col],
        ] = self.data[..., mask]

        return block

    def _set_block(self, row: int, col: int, block: np.ndarray) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Parameters
        ----------
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        block : np.ndarray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        mask = (
            (self.rows >= self.block_offsets[row])
            & (self.rows < self.block_offsets[row + 1])
            & (self.cols >= self.block_offsets[col])
            & (self.cols < self.block_offsets[col + 1])
        )

        if not np.any(mask):
            # No data in this block, nothing to do.
            return

        self.data[..., mask] = block[
            ...,
            self.rows[mask] - self.block_offsets[row],
            self.cols[mask] - self.block_offsets[col],
        ]

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBCOO):
            raise TypeError("Can only add DSBCOO matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if np.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")

        if np.any(self.rows != other.rows):
            raise ValueError("Row indices do not match.")

        if np.any(self.cols != other.cols):
            raise ValueError("Column indices do not match.")

    def __iadd__(self, other: "DSBSparse | sparse.sparray") -> "DSBCOO":
        """In-place addition of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()
            sparray_data = lil[self.rows, self.cols]
            self.data[:] += sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] += other.data[:]
        return self

    def __isub__(self, other: "DSBSparse | sparse.sparray") -> "DSBCOO":
        """In-place subtraction of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()
            sparray_data = lil[self.rows, self.cols]
            self.data[:] -= sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, other: "DSBSparse") -> None:
        """In-place multiplication of two DSBSparse matrices."""
        self._check_commensurable(other)
        self.data *= other.data

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

        Returns-------
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
        ):
            # Transpose.
            rows_t, cols_t = self.cols, self.rows

            # Canonical ordering of the transpose.
            inds_bcoo2canonical_t = np.lexsort((cols_t, rows_t))
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

        self.data[:] = self.data[..., self._inds_bcoo2bcoo_t]
        self._inds_bcoo2bcoo_t = np.argsort(self._inds_bcoo2bcoo_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rows, self._rows_t = self._rows_t, self.rows

    def spy(self) -> tuple[np.ndarray, np.ndarray]:
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
        rows = np.zeros(self.nnz, dtype=int)
        for (row, __), rowptr in self.rowptr_map.items():
            for i in range(self.block_sizes[row]):
                rows[rowptr[i] : rowptr[i + 1]] = i + self.block_offsets[row]

        return rows, self.cols

    @classmethod
    def from_sparray(
        cls,
        arr: sparse.sparray,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBCOO":
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
        DSBCOO
            The new DSBCOO matrix.

        """
        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        local_stack_shape = stack_section_sizes[comm.rank]

        if isinstance(local_stack_shape, int):
            local_stack_shape = (local_stack_shape,)

        coo: sparse.coo_array = arr.tocoo()

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
        block_sort_index = compute_block_sort_index(coo.row, coo.col, block_sizes)

        data = np.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
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
