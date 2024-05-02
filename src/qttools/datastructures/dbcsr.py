import numpy as np

from qttools.datastructures.dbsparse import DBSparse


class DBCSR(DBSparse):
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

        self._num_blocks = len(block_sizes)
        self._block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        self._stack_shape = data.shape[:-1]
        self._shape = self.stack_shape + (np.sum(block_sizes), np.sum(block_sizes))
        self._nnz = self.data.shape[-1]

    def from_sparray(
        a: sparray,
        blocksizes: np.ndarray,
        stackshape: tuple = (1,),
        densify_blocks=None,
        pinned=False,
    ): ...

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

    def zeros_like(cls, dbcsr: "DBCSR") -> DBSparse:
        return cls(
            np.zeros_like(dbcsr.data),
            dbcsr.cols,
            dbcsr.rowptr_map,
            dbcsr.block_sizes,
        )

    def block_diagonal(
        offset: int = 0, dense: bool = False
    ) -> list[sparray] | list[np.ndarray]: ...

    def diagonal() -> np.ndarray: ...

    def local_transpose(copy=False) -> None | DBSparse: ...

    def distributed_transpose() -> None: ...

    def _unsign_block_index(self, brow: int, bcol: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        brow = self.num_blocks + brow if brow < 0 else brow
        bcol = self.num_blocks + bcol if bcol < 0 else bcol
        if not (0 <= brow < self.num_blocks and 0 <= bcol < self.num_blocks):
            raise IndexError("Block index out of bounds.")

        return brow, bcol

    def __setitem__(
        self, idx: tuple[int, int], block: np.ndarray
    ) -> None:
        if block.shape[-2:] != (
            self.block_sizes[idx[0]],
            self.block_sizes[idx[1]],
        ):
            raise ValueError("Block shape does not match.")

        idx[0], idx[1] = self._unsign_block_index(idx[0], idx[1])

        rowptr = self.rowptr_map.get((idx[0], idx[1]), None)
        if rowptr is None:
            return

        for row in range(self.block_sizes[idx[0]]):
            cols = self.cols[rowptr[row] : rowptr[row + 1]]
            self.data[..., rowptr[row] : rowptr[row + 1]] = block[
                ..., row, cols - self.block_offsets[idx[1]]
            ]

    def __getitem__(
        self, idx: tuple[int, int]
    ) -> sparray:

    def __iadd__(self, other: DBSparse) -> None: ...

    def __imul__(self, other: DBSparse) -> None: ...

    def __neg__(self) -> None: ...

    def __matmul__(self, other: DBSparse) -> None: ...

    @property
    def num_blocks(self) -> np.uint:
        return self._num_blocks

    @property
    def block_offsets(self) -> np.uint:
        return self._block_offsets

    @property
    def stack_shape(self) -> np.uint:
        return self._stack_shape

    @property
    def shape(self) -> np.uint:
        return self._shape

    @property
    def nzz(self) -> np.uint:
        return self._nnz
