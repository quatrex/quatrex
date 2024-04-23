try:
    import cupy as xp
    import cupyx

    _GPU = True

except ImportError:
    import numpy as xp

    _GPU = False


import numpy as np
from numpy.typing import ArrayLike


class COOGroup:
    """Store a pack of COO matrices with identical sparsity patterns.
    - Storing sparsity pattern once
    - Packing values in a contiguous array
    """

    def __init__(
        self,
        length: int,
        rows: ArrayLike,
        cols: ArrayLike,
        data: ArrayLike = None,
        blocksizes: ArrayLike = None,
        pinned: bool = False,
    ) -> None:
        self.nnz = rows.shape[0]
        self.length = length
        self.blocksizes = xp.array(blocksizes)
        self.shape = (length, rows.max() + 1, cols.max() + 1)

        self.rows = xp.array(rows, dtype=np.uint32)
        self.cols = xp.array(cols, dtype=np.uint32)

        if data is not None:
            self.data[:] = xp.array(data, dtype=np.complex128)
        if _GPU and pinned:
            self.data = cupyx.zeros_pinned((self.length, self.nnz), dtype=np.complex128)
        else:
            self.data = xp.zeros((self.length, self.nnz), dtype=np.complex128)

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self.blocksizes[0] + row if row < 0 else row
        col = self.blocksizes[0] + col if col < 0 else col
        if not (
            0 <= row < self.blocksizes.shape[0] and 0 <= col < self.blocksizes.shape[0]
        ):
            raise IndexError("Block index out of bounds.")

        return row, col

    def get_block(brow, bcol):
        """Get a block of the matrix."""
        brow, bcol = self._unsign_index(brow, bcol)

        start_row_idx = self.blocksizes[:brow].sum()
        start_col_idx = self.blocksizes[:bcol].sum()

        stop_row_idx = start_row_idx + self.blocksizes[brow]
        stop_col_idx = start_col_idx + self.blocksizes[bcol]

        idx = np.where(
            zip(self.rows, self.cols),
            (start_row_idx <= self.rows < stop_row_idx)
            & (start_col_idx <= self.cols < stop_col_idx),
        )
        rows = self.rows[idx] - start_row_idxs
        cols = self.cols[idx] - start_col_idxs

        arr = xp.zeros(
            (self.length, self.blocksizes[brow], self.blocksizes[bcol]),
            dtype=np.complex128,
        )
        arr[:, rows, cols] = self.data[:, idx]

        return arr

    def set_block(brow, bcol, val): ...
