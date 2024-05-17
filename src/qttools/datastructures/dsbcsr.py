# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.utils.mpi_utils import get_num_elements_per_section


class DSBCSR(DSBSparse):
    """Distributed block compressed sparse row matrix."""

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

    def __setitem__(self, key: tuple, block_stack: np.ndarray) -> None:
        """Sets a block in the matrix."""
        if self._distribution_state == "nnz":
            raise NotImplementedError("Cannot get blocks when distributed through nnz.")
        if not self._return_dense:
            raise NotImplementedError("Sparse array not yet implemented.")

        if len(key) < 2:
            raise ValueError("At least the two block indices are required.")

        if len(key) >= 2:
            *stack_index, brow, bcol = key

        if len(stack_index) > len(self.stack_shape):
            raise ValueError(
                f"Too many stack indices for stack shape '{self.stack_shape}'."
            )

        stack_index += (slice(None),) * (len(self.stack_shape) - len(stack_index))

        brow, bcol = self._unsign_block_index(brow, bcol)

        if block_stack.shape[-2:] != (
            self.block_sizes[brow],
            self.block_sizes[bcol],
        ):
            raise ValueError("Block shape does not match.")

        rowptr = self.rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            return

        for row in range(self.block_sizes[brow]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            self.data[*stack_index, rowptr[row] : rowptr[row + 1]] = block_stack[
                ..., row, cols - self.block_offsets[bcol]
            ]

    def __getitem__(self, key: tuple) -> sparse.sparray | np.ndarray:
        """Gets a block from the matrix.

        The two last indices are always the block indices.

        """
        if self._distribution_state == "nnz":
            raise NotImplementedError("Cannot get blocks when distributed through nnz.")
        if not self._return_dense:
            raise NotImplementedError("Sparse array not yet implemented.")

        if len(key) < 2:
            raise ValueError("At least the two block indices are required.")

        if len(key) >= 2:
            *stack_index, brow, bcol = key

        if len(stack_index) > len(self.stack_shape):
            raise ValueError(
                f"Too many stack indices for stack shape '{self.stack_shape}'."
            )

        stack_index += (slice(None),) * (len(self.stack_shape) - len(stack_index))

        brow, bcol = self._unsign_block_index(brow, bcol)

        block = np.zeros(
            (
                *self.data[*stack_index].shape[:-1],  # Stack dimensions.
                self.block_sizes[brow],
                self.block_sizes[bcol],
            ),
            dtype=self._padded_data.dtype,
        )
        rowptr = self.rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            return block

        for row in range(self.block_sizes[brow]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            block[..., row, cols - self.block_offsets[bcol]] = self.data[
                *stack_index, rowptr[row] : rowptr[row + 1]
            ]
        return block

    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DSBCSR):
            raise TypeError("Can only add DBCSR matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if self.block_sizes != other.block_sizes:
            raise ValueError("Block sizes do not match.")

        if self.rowptr_map.keys() != other.rowptr_map.keys():
            raise ValueError("Block sparsities do not match.")

        if self.cols != other.cols:
            raise ValueError("Column indices do not match.")

    def __iadd__(self, other: "DSBSparse | sparse.sparray") -> None:
        """Adds another DBSparse matrix to the current matrix."""

        if isinstance(other, sparse.sparray):
            coo = other.tocoo()
            coo.sum_duplicates()
            rows, cols = self.spy()

        self._check_commensurable(other)
        self.data += other.data

    def __isub__(self, other: "DSBSparse | sparse.sparray") -> None:
        """Subtracts another DBSparse matrix from the current matrix."""
        self.__iadd__(self, -other)

    def __imul__(self, other: "DSBSparse") -> None:
        """Multiplies another DBSparse matrix to the current matrix."""
        self._check_commensurable(other)
        self.data *= other.data

    def __neg__(self) -> "DSBCSR":
        """Negates the matrix."""
        return DSBCSR(-self.data, self.cols, self.rowptr_map, self.block_sizes)

    def __matmul__(self, other: "DSBSparse") -> None:
        ...

    def ltranspose(self, copy=False) -> "None | DSBSparse":
        """Returns the transpose of the matrix."""
        if self._distribution_state == "nnz":
            raise NotImplementedError("Cannot transpose when distributed through nnz.")

        if copy:
            self = DSBCSR(
                self._padded_data.copy(),
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
            inds_canonical2bcsr_t = self._compute_block_sort_index(
                canonical_rows_t, canonical_cols_t, self.block_sizes
            )

            # Mapping directly from original ordering to transpose
            # block-ordering is achieved by chaining the two mappings.
            inds_bcsr2bcsr_t = inds_bcsr2canonical_t[inds_canonical2bcsr_t]

            # Compute the rowptr map for the transpose.
            rowptr_map_t = self._compute_ptr_map(
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
        """Returns the COO sparsity pattern."""
        rows = np.zeros(self.nnz, dtype=int)
        for (brow, __), rowptr in self.rowptr_map.items():
            for i in range(self.block_sizes[brow]):
                rows[rowptr[i] : rowptr[i + 1]] = i + self.block_offsets[brow]

        return rows, self.cols

    def to_dense(self) -> np.ndarray:
        """Returns the dense matrix representation."""
        temp = self._return_dense

        self._return_dense = True

        arr = np.zeros(self.shape, dtype=self._padded_data.dtype)

        for i, j in np.ndindex(self.num_blocks, self.num_blocks):
            arr[
                ...,
                self.block_offsets[i] : self.block_offsets[i + 1],
                self.block_offsets[j] : self.block_offsets[j + 1],
            ] = self[i, j]

        self._return_dense = temp

        return arr

    @classmethod
    def from_sparray(
        cls,
        a: sparse.sparray,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBCSR":
        stack_section_sizes, __ = get_num_elements_per_section(
            global_stack_shape[0], comm.size
        )
        local_stack_shape = stack_section_sizes[comm.rank]

        if isinstance(local_stack_shape, int):
            local_stack_shape = (local_stack_shape,)

        coo: sparse.coo_array = a.tocoo()

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
        rowptr_map = cls._compute_ptr_map(coo.row, coo.col, block_sizes)
        block_sort_index = cls._compute_block_sort_index(coo.row, coo.col, block_sizes)

        data = np.zeros(local_stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        data[..., :] = coo.data[block_sort_index]
        cols = coo.col[block_sort_index]

        return cls(data, cols, rowptr_map, block_sizes, global_stack_shape)

    @classmethod
    def zeros_like(cls, a: "DSBCSR") -> "DSBCSR":
        """Returns a DBCSR matrix with the same sparsity pattern."""
        return cls(
            np.zeros_like(a.data),
            a.cols,
            a.rowptr_map,
            a.block_sizes,
            a.global_stack_shape,
            a.return_dense,
        )

    @staticmethod
    def _compute_block_sort_index(
        coo_rows: np.ndarray, coo_cols: np.ndarray, block_sizes: np.ndarray
    ) -> np.ndarray:
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

    @staticmethod
    def _compute_ptr_map(
        coo_rows: np.ndarray, coo_cols: np.ndarray, block_sizes: np.ndarray
    ) -> dict:
        """Computes the rowptr map for the given COO matrix."""
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
