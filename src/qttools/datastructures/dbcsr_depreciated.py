import numpy as np
import scipy.sparse as sps
from line_profiler import profile


class DBCSR:
    """Distributed block compressed sparse row matrix."""

    def __init__(
        self,
        data: np.ndarray,
        cols: np.ndarray,
        rowptr_map: dict,
        block_sizes: np.ndarray,
    ) -> None:
        """Initializes the DBCSR matrix."""
        self.data = np.asarray(data)
        self.cols = np.asarray(cols).astype(int)
        self.rowptr_map = rowptr_map
        self.block_sizes = np.asarray(block_sizes).astype(int)

        self.num_blocks = len(block_sizes)
        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        self.stack_shape = data.shape[:-1]
        self.shape = self.stack_shape + (np.sum(block_sizes), np.sum(block_sizes))
        self.nnz = self.data.shape[-1]

    def _unsign_block_index(self, brow: int, bcol: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        brow = self.num_blocks + brow if brow < 0 else brow
        bcol = self.num_blocks + bcol if bcol < 0 else bcol
        if not (0 <= brow < self.num_blocks and 0 <= bcol < self.num_blocks):
            raise IndexError("Block index out of bounds.")

        return brow, bcol

    def to_dense(self) -> np.ndarray:
        """Returns the dense matrix representation."""
        arr = np.zeros(self.shape, dtype=self.data.dtype)

        for i, j in np.ndindex(self.num_blocks, self.num_blocks):
            arr[
                ...,
                self.block_offsets[i] : self.block_offsets[i + 1],
                self.block_offsets[j] : self.block_offsets[j + 1],
            ] = self.get_block(i, j)

        return arr

    def get_block(self, brow: int, bcol: int) -> np.ndarray:
        """Returns the block at the given block-row and -column."""
        brow, bcol = self._unsign_block_index(brow, bcol)

        rowptr = self.rowptr_map.get((brow, bcol), None)
        block = np.zeros(
            self.stack_shape + (self.block_sizes[brow], self.block_sizes[bcol]),
            dtype=self.data.dtype,
        )
        if rowptr is None:
            return block

        for row in range(self.block_sizes[brow]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            block[..., row, cols - self.block_offsets[bcol]] = self.data[
                ..., rowptr[row] : rowptr[row + 1]
            ]
        return block

    def set_block(self, brow: int, bcol: int, block: np.ndarray) -> None:
        """Sets the block at the given block-row and -column."""
        if block.shape[-2:] != (
            self.block_sizes[brow],
            self.block_sizes[bcol],
        ):
            raise ValueError("Block shape does not match.")

        brow, bcol = self._unsign_block_index(brow, bcol)

        rowptr = self.rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            return

        for row in range(self.block_sizes[brow]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            self.data[..., rowptr[row] : rowptr[row + 1]] = block[
                ..., row, cols - self.block_offsets[bcol]
            ]

    @classmethod
    def zeros_like(cls, dbcsr: "DBCSR") -> "DBCSR":
        """Returns a zero-initialized DBCSR matrix with the same
        sparsity pattern."""
        return cls(
            np.zeros_like(dbcsr.data),
            dbcsr.cols,
            dbcsr.rowptr_map,
            dbcsr.block_sizes,
        )

    @profile
    @classmethod
    def from_coo(
        cls,
        coo: sps.coo_array,
        block_sizes: np.ndarray,
        stackshape: int | tuple,
    ) -> "DBCSR":
        """Converts a coo matrix to a DBCSR matrix."""
        # Sort the data by block-row and -column
        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
        num_blocks = len(block_sizes)

        rowptr_map = {}

        if isinstance(stackshape, int):
            stackshape = (stackshape,)

        data = np.zeros(stackshape + (coo.nnz,), dtype=coo.data.dtype)
        cols = np.zeros(coo.nnz, dtype=int)

        # NOTE: This is a naive implementation and can be optimized.
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
                continue

            data[..., offset : offset + bnnz] = coo.data[inds]
            cols[offset : offset + bnnz] = coo.col[inds]

            rowptr, __ = np.histogram(
                coo.row[inds] - block_offsets[i],
                bins=np.arange(block_sizes[i] + 1),
            )
            rowptr = np.hstack(([0], np.cumsum(rowptr))) + offset
            rowptr_map[(i, j)] = rowptr

            offset += bnnz

        return cls(data, cols, rowptr_map, block_sizes)
