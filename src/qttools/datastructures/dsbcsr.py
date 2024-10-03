# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.gpu_utils import ArrayLike, get_device, xp
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import compute_block_sort_index, compute_ptr_map


class DSBCSR(DSBSparse):
    """Distributed stack of block-compressed sparse row matrices.

    This DSBSparse implementation uses a block-compressed sparse row
    format to store the sparsity pattern of the matrix. The data is
    sorted by block-row and -column. We use a row pointer map together
    with the column indices to access the blocks efficiently.

    Parameters
    ----------
    data : array_like
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    cols : array_like
        The column indices.
    rowptr_map : dict
        The row pointer map.
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
        cols: ArrayLike,
        rowptr_map: dict,
        block_sizes: ArrayLike,
        global_stack_shape: tuple,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DBCSR matrix."""
        super().__init__(data, block_sizes, global_stack_shape, return_dense)

        self.cols = xp.asarray(cols).astype(int)
        self.rowptr_map = rowptr_map

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
        """Gets a single value accross the stack."""
        row, col = self._normalize_index(index)

        brow = xp.where(self.block_offsets <= row)[0][-1]
        bcol = xp.where(self.block_offsets <= col)[0][-1]
        rowptr = self.rowptr_map.get((brow, bcol), None)

        if rowptr is None:
            if self.distribution_state == "stack":
                return xp.zeros(self.data.shape[:-1], dtype=self.dtype)
            # We cannot know which rank is supposed to hold an element
            # that is not in the matrix, so we raise an error.
            raise IndexError("Requested element not in matrix.")

        row -= self.block_offsets[brow]  # Renormalize the row index for this block.
        ind = xp.where(self.cols[rowptr[row] : rowptr[row + 1]] == col)[0]

        if self.distribution_state == "stack":
            if len(ind) == 0:
                return xp.zeros(self.data.shape[:-1], dtype=self.dtype)

            return self.data[..., rowptr[row] + ind[0]]

        if len(ind) == 0:
            # We cannot know which rank is supposed to hold an element
            # that is not in the matrix, so we raise an error.
            raise IndexError("Requested element not in matrix.")

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        nnz_section_offsets = xp.hstack(
            ([0], xp.cumsum([max(self.nnz_section_sizes)] * comm.size))
        )
        rank = xp.where(nnz_section_offsets <= rowptr[row] + ind[0])[0][-1]

        if rank == comm.rank:
            return self.data[..., rowptr[row] + ind[0] - nnz_section_offsets[rank]]

        raise IndexError(
            f"Requested data not on this rank ({comm.rank}). It is on rank {rank}."
        )

    def __setitem__(self, index: tuple, value: ArrayLike) -> None:
        """Sets a single value in the matrix."""
        row, col = self._normalize_index(index)

        brow = xp.where(self.block_offsets <= row)[0][-1]
        bcol = xp.where(self.block_offsets <= col)[0][-1]
        rowptr = self.rowptr_map.get((brow, bcol), None)

        if rowptr is None:
            return

        row -= self.block_offsets[brow]  # Renormalize the row index for this block.
        ind = xp.where(self.cols[rowptr[row] : rowptr[row + 1]] == col)[0]

        if len(ind) == 0:
            return

        if self.distribution_state == "stack":
            self.data[..., rowptr[row] + ind[0]] = value
            return

        # If nnz are distributed accross the ranks, we need to find the
        # rank that holds the data.
        nnz_section_offsets = xp.hstack(
            ([0], xp.cumsum([max(self.nnz_section_sizes)] * comm.size))
        )
        rank = xp.where(nnz_section_offsets <= rowptr[row] + ind[0])[0][-1]

        if rank == comm.rank:
            self._data[
                self._stack_padding_mask,
                ...,
                rowptr[row] + ind[0] - nnz_section_offsets[rank],
            ] = value
            return

        raise IndexError(
            f"Requested data not on this rank ({comm.rank}). It is on rank {rank}."
        )

    def _get_block(self, stack_index: tuple, row: int, col: int) -> ArrayLike:
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
        block : sparray | np.ndarray
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        rowptr = self.rowptr_map.get((row, col), None)
        data_stack = self.data[*stack_index]
        block = xp.zeros(
            data_stack.shape[:-1]
            + (int(self.block_sizes[row]), int(self.block_sizes[col])),
            dtype=self.dtype,
        )
        if rowptr is None:
            # No data in this block, return zeros.
            return block

        for i in range(int(self.block_sizes[row])):
            cols = self.cols[rowptr[i] : rowptr[i + 1]]
            block[..., i, cols - self.block_offsets[col]] = data_stack[
                ..., rowptr[i] : rowptr[i + 1]
            ]

        return block

    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: ArrayLike
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
        block : array_like
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        rowptr = self.rowptr_map.get((row, col), None)
        if rowptr is None:
            # No data in this block, nothing to do.
            return

        for i in range(int(self.block_sizes[row])):
            cols = self.cols[rowptr[i] : rowptr[i + 1]]
            self.data[*stack_index][..., rowptr[i] : rowptr[i + 1]] = block[
                ..., i, cols - self.block_offsets[col]
            ]

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

    def __iadd__(self, other: "DSBSparse | sparse.sparray") -> "DSBCSR":
        """In-place addition of two DSBSparse matrices."""
        if sparse.issparse(other):
            lil = other.tolil()

            sparray_data = xp.zeros(self.nnz, dtype=self.dtype)
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

            sparray_data = xp.zeros(self.nnz, dtype=self.dtype)
            for i, (row, col) in enumerate(zip(*self.spy())):
                sparray_data[i] = lil[row, col]
            self.data[:] -= sparray_data
            return self

        self._check_commensurable(other)
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, other: "DSBSparse") -> "DSBCSR":
        """In-place multiplication of two DSBSparse matrices."""
        self._check_commensurable(other)
        self.data[:] *= other.data[:]
        return self

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
            inds_bcsr2canonical_t = xp.lexsort(xp.vstack((cols_t, rows_t)))
            canonical_rows_t = rows_t[inds_bcsr2canonical_t]
            canonical_cols_t = cols_t[inds_bcsr2canonical_t]

            # Compute index for sorting the transpose by block.
            inds_canonical2bcsr_t = compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Compute the rowptr map for the transpose.
            rowptr_map_t = compute_ptr_map(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Cache the necessary objects.
            self._inds_bcsr2bcsr_t = inds_bcsr2bcsr_t
            self._rowptr_map_t = rowptr_map_t
            self._cols_t = cols_t[self._inds_bcsr2bcsr_t]

        self.data[:] = self.data[..., self._inds_bcsr2bcsr_t]
        self._inds_bcsr2bcsr_t = xp.argsort(self._inds_bcsr2bcsr_t)
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
        rows = xp.zeros(self.nnz, dtype=int)
        for (row, __), rowptr in self.rowptr_map.items():
            for i in range(int(self.block_sizes[row])):
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
                for m, n in xp.ndindex(block_sizes[i], block_sizes[j])
            ]
            coo.row = np.append(coo.row, [m for m, __ in indices])
            coo.col = np.append(coo.col, [n for __, n in indices])
            coo.data = np.append(coo.data, np.zeros(len(indices), dtype=coo.data.dtype))

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Compute the rowptr map.
        rowptr_map = compute_ptr_map(
            get_device(coo.row), get_device(coo.col), get_device(block_sizes)
        )
        block_sort_index = compute_block_sort_index(
            get_device(coo.row), get_device(coo.col), get_device(block_sizes)
        )

        data = xp.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        data[..., :] = get_device(coo.data)[block_sort_index]
        cols = get_device(coo.col)[block_sort_index]

        return cls(
            data=data,
            cols=cols,
            rowptr_map=rowptr_map,
            block_sizes=get_device(block_sizes),
            global_stack_shape=global_stack_shape,
        )
