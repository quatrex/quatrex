# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.mpi_utils import get_section_sizes


def _compute_block_sort_index(
    coo_rows: np.ndarray, coo_cols: np.ndarray, block_sizes: np.ndarray
) -> np.ndarray:
    """Computes the block-sorting index for a sparse matrix.

    Parameters
    ----------
    coo_rows : np.ndarray
        The row indices of the matrix in coordinate format.
    coo_cols : np.ndarray
        The column indices of the matrix in coordinate format.
    block_sizes : np.ndarray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : np.ndarray
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = len(block_sizes)
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

    sort_index = np.zeros(len(coo_cols), dtype=int)
    offset = 0
    for i, j in np.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not np.any(mask):
            # Skip empty blocks.
            continue

        bnnz = np.sum(mask)

        # Sort the data by block-row and -column.
        sort_index[offset : offset + bnnz] = np.argwhere(mask).squeeze()

        offset += bnnz

    return sort_index


def _compute_ptr_map(
    coo_rows: np.ndarray, coo_cols: np.ndarray, block_sizes: np.ndarray
) -> dict:
    """Computes the rowptr map for a sparse matrix.

    Parameters
    ----------
    coo_rows : np.ndarray
        The row indices of the matrix in coordinate format.
    coo_cols : np.ndarray
        The column indices of the matrix in coordinate format.
    block_sizes : np.ndarray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    rowptr_map : dict
        The row pointer map, describing the block-sparse matrix in
        blockwise column-sparse-row format.

    """
    num_blocks = len(block_sizes)
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

    # NOTE: This is a naive implementation and can be parallelized.
    rowptr_map = {}
    offset = 0
    for i, j in np.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not np.any(mask):
            # Skip empty blocks.
            continue

        # Compute the rowptr map.
        rowptr, __ = np.histogram(
            coo_rows[mask] - block_offsets[i],
            bins=np.arange(block_sizes[i] + 1),
        )
        rowptr = np.hstack(([0], np.cumsum(rowptr))) + offset
        rowptr_map[(i, j)] = rowptr

        bnnz = np.sum(mask)
        offset += bnnz

    return rowptr_map


class DSBCSR(DSBSparse):
    """Distributed stack of block-compressed sparse row matrices.

    This DSBSparse implementation uses a block-compressed sparse row
    format to store the sparsity pattern of the matrix. The data is
    sorted by block-row and -column. We use a row pointer map together
    with the column indices to access the blocks efficiently.

    Parameters
    ----------
    data : np.ndarray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    cols : np.ndarray
        The column indices.
    rowptr_map : dict
        The row pointer map.
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
        cols: np.ndarray,
        rowptr_map: dict,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DBCSR matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.cols = np.asarray(cols).astype(int)
        self.rowptr_map = rowptr_map

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
        rowptr = self.rowptr_map.get((row, col), None)
        if rowptr is None:
            # No data in this block.
            return block

        for i in range(self.block_sizes[row]):
            cols = self.cols[rowptr[i] : rowptr[i + 1]]
            block[..., i, cols - self.block_offsets[col]] = self.data[
                ..., rowptr[i] : rowptr[i + 1]
            ]
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
        rowptr = self.rowptr_map.get((row, col), None)
        if rowptr is None:
            # No data in this block, nothing to do.
            return

        for i in range(self.block_sizes[row]):
            cols = self.cols[rowptr[i] : rowptr[i + 1]]
            self.data[..., rowptr[i] : rowptr[i + 1]] = block[
                ..., i, cols - self.block_offsets[col]
            ]

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBCSR):
            raise TypeError("Can only add DSBCSR matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if np.any(self.block_sizes != other.block_sizes):
            raise ValueError("Block sizes do not match.")

        if self.rowptr_map.keys() != other.rowptr_map.keys():
            raise ValueError("Block sparsities do not match.")

        if np.any(self.cols != other.cols):
            raise ValueError("Column indices do not match.")

    def __iadd__(self, other: "DSBSparse | sparse.sparray") -> "DSBCSR":
        """In-place addition of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()

            sparray_data = np.zeros(self.nnz, dtype=self.dtype)
            for i, (row, col) in enumerate(zip(*self.spy())):
                sparray_data[i] = lil[row, col]
            self.data[:] += sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] += other.data[:]
        return self

    def __isub__(self, other: "DSBSparse | sparse.sparray") -> "DSBCSR":
        """In-place subtraction of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()

            sparray_data = np.zeros(self.nnz, dtype=self.dtype)
            for i, (row, col) in enumerate(zip(*self.spy())):
                sparray_data[i] = lil[row, col]
            self.data[:] -= sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, other: "DSBSparse") -> None:
        """In-place multiplication of two DSBSparse matrices."""
        self._check_commensurable(other)
        self.data *= other.data

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
        raise NotImplementedError("Matrix multiplication is not implemented.")

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
            inds_bcsr2canonical_t = np.lexsort((cols_t, rows_t))
            canonical_rows_t = rows_t[inds_bcsr2canonical_t]
            canonical_cols_t = cols_t[inds_bcsr2canonical_t]

            # Compute index for sorting the transpose by block.
            inds_canonical2bcsr_t = _compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Compute the rowptr map for the transpose.
            rowptr_map_t = _compute_ptr_map(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Cache the necessary objects.
            self._inds_bcsr2bcsr_t = inds_bcsr2bcsr_t
            self._rowptr_map_t = rowptr_map_t
            self._cols_t = cols_t[self._inds_bcsr2bcsr_t]

        self.data[:] = self.data[..., self._inds_bcsr2bcsr_t]
        self._inds_bcsr2bcsr_t = np.argsort(self._inds_bcsr2bcsr_t)
        self.cols, self._cols_t = self._cols_t, self.cols
        self.rowptr_map, self._rowptr_map_t = self._rowptr_map_t, self.rowptr_map

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
        rowptr_map = _compute_ptr_map(coo.row, coo.col, block_sizes)
        block_sort_index = _compute_block_sort_index(coo.row, coo.col, block_sizes)

        data = np.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        data[..., :] = coo.data[block_sort_index]
        cols = coo.col[block_sort_index]

        return cls(data, cols, rowptr_map, block_sizes, global_stack_shape)
