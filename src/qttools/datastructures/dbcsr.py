# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.
# All rights reserved.


import numpy as np
from scipy import sparse

from qttools.datastructures.dbsparse import DBSparse


class DBCSR(DBSparse):
    """Distributed block compressed sparse row matrix."""

    def __init__(
        self,
        data: np.ndarray,
        cols: np.ndarray,
        rowptr_map: dict,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        return_dense: bool = False,
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

        if block_stack.shape != (
            *self.masked_data[*stack_index].shape[:-1],
            self.block_sizes[brow],
            self.block_sizes[bcol],
        ):
            raise ValueError("Block shape does not match.")

        rowptr = self.rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            return

        for row in range(self.block_sizes[brow]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            self.masked_data[*stack_index, rowptr[row] : rowptr[row + 1]] = block_stack[
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
                *self.masked_data[*stack_index].shape[:-1],  # Stack dimensions.
                self.block_sizes[brow],
                self.block_sizes[bcol],
            ),
            dtype=self.data.dtype,
        )
        rowptr = self.rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            return block

        for row in range(self.block_sizes[brow]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            block[..., row, cols - self.block_offsets[bcol]] = self.masked_data[
                *stack_index, rowptr[row] : rowptr[row + 1]
            ]
        return block

    def _check_commensurable(self, other: "DBSparse") -> None:
        """Checks if the other matrix is commensurate."""
        if not isinstance(other, DBCSR):
            raise TypeError("Can only add DBCSR matrices.")

        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match.")

        if self.block_sizes != other.block_sizes:
            raise ValueError("Block sizes do not match.")

        if self.rowptr_map.keys() != other.rowptr_map.keys():
            raise ValueError("Rowptr maps do not match.")

        if self.cols != other.cols:
            raise ValueError("Column indices do not match.")

    def __iadd__(self, other: "DBSparse") -> None:
        """Adds another DBSparse matrix to the current matrix."""
        self._check_commensurable(other)
        self.data += other.data

    def __imul__(self, other: "DBSparse") -> None:
        """Multiplies another DBSparse matrix to the current matrix."""
        self._check_commensurable(other)
        self.data *= other.data

    def __neg__(self) -> "DBCSR":
        """Negates the matrix."""
        return DBCSR(-self.data, self.cols, self.rowptr_map, self.block_sizes)

    def __matmul__(self, other: "DBSparse") -> None:
        ...

    def ltranspose(self, copy=False) -> "None | DBSparse":
        ...

    def to_dense(self) -> np.ndarray:
        """Returns the dense matrix representation."""
        temp = self._return_dense

        self._return_dense = True

        arr = np.zeros(self.shape, dtype=self.data.dtype)

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
        stack_shape: tuple | None = None,
        global_stack_shape: tuple | None = None,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ):
        coo: sparse.coo_array = a.tocoo()

        num_blocks = len(block_sizes)
        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

        # Densify the selected blocks.
        for i, j in densify_blocks or []:
            indices = [
                (m + block_offsets[i], n + block_offsets[j])
                for m, n in np.ndindex(block_sizes[i], block_sizes[j])
            ]
            coo.row = np.append(coo.row, [m for m, __ in indices])
            coo.col = np.append(coo.col, [n for __, n in indices])
            coo.data = np.append(coo.data, np.zeros(len(indices), dtype=coo.data.dtype))

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Initialize the data and column index arrays.
        if isinstance(stack_shape, int):
            stack_shape = (stack_shape,)

        data = np.zeros(stack_shape + (coo.nnz,), dtype=coo.data.dtype)
        cols = np.zeros(coo.nnz, dtype=int)

        # NOTE: This is a naive implementation and can be parallelized.
        rowptr_map = {}
        offset = 0
        for i, j in np.ndindex(num_blocks, num_blocks):
            inds = (
                (block_offsets[i] <= coo.row)
                & (coo.row < block_offsets[i + 1])
                & (block_offsets[j] <= coo.col)
                & (coo.col < block_offsets[j + 1])
            )
            bnnz = np.sum(inds)

            if bnnz == 0:
                # Skip empty blocks.
                continue

            # Sort the data by block-row and -column.
            data[..., offset : offset + bnnz] = coo.data[inds]
            cols[offset : offset + bnnz] = coo.col[inds]

            # Compute the rowptr map.
            rowptr, __ = np.histogram(
                coo.row[inds] - block_offsets[i],
                bins=np.arange(block_sizes[i] + 1),
            )
            rowptr = np.hstack(([0], np.cumsum(rowptr))) + offset
            rowptr_map[(i, j)] = rowptr

            offset += bnnz

        if global_stack_shape is None:
            global_stack_shape = stack_shape

        return cls(data, cols, rowptr_map, block_sizes, global_stack_shape)

    @classmethod
    def zeros_like(cls, a: "DBCSR") -> "DBCSR":
        """Returns a DBCSR matrix with the same sparsity pattern."""
        return cls(np.zeros_like(a.data), a.cols, a.rowptr_map, a.block_sizes)
