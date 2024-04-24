# try:
#     import cupy as xp
#     import cupyx

#     _GPU = True

# except ImportError:
#     import numpy as xp

#     _GPU = False


import numpy as np
from numpy.typing import ArrayLike


class SBCOO:
    """Stack of block-accessible COO matrices with identical sparsity.

    To rapidly access the blocks of the matrix, a block-mapping is
    computed (or provided) during instantiation.

    The block-setters and -getters support accessing and modifying dense
    and sparse views of blocks of the matrix.

    Basic arithmetic is supported.

    Parameters
    ----------
    rows : ArrayLike
        The row indices of the non-zero elements.
    cols : ArrayLike
        The column indices of the non-zero elements.
    blocksizes : ArrayLike
        The sizes of the blocks.
    data : ArrayLike, optional
        The non-zero elements of the matrix. If not provided, the data
        is initialized to zero.
    stackshape : tuple, optional
        The shape of the matrix stack. Default is (1,). If the data is
        provided, this parameter is ignored and the stack shape is
        inferred from the shape of the data. The last dimension of the
        data must be the same as the number of non-zero elements, the
        rest of the shape is the stack shape.
    blockmap : ArrayLike, optional
        The blocks of the matrix. If not provided, the blocks are
        computed from the data.
    dtype : type, optional
        The data type of the stored data matrix. Default is
        np.complex128.
    pinned : bool, optional
        Whether to use pinned memory for the data. Default is False.

    """

    def __init__(
        self,
        rows: ArrayLike,
        cols: ArrayLike,
        blocksizes: ArrayLike,
        data: ArrayLike | None = None,
        stackshape: tuple | None = None,
        blockmap: ArrayLike | None = None,
        dtype: type = np.complex128,
        pinned: bool = False,
    ) -> None:
        """Initializes the SBCOO matrix."""
        if data is not None:
            if data.shape[-1] != rows.shape[0]:
                raise ValueError(
                    "The last dimension of the data must be the same as "
                    "the number of non-zero elements."
                )
            if stackshape is not None:
                if data.shape[:-1] != stackshape:
                    raise ValueError(
                        "The shape of the data must match the stack shape."
                    )
            else:
                stackshape = data.shape[:-1]

        else:
            if stackshape is None:
                stackshape = (1,)
            # TODO: Implement pinned memory support.
            data = np.zeros(stackshape + (rows.shape[0],), dtype=dtype)

        self.rows = np.asarray(rows, dtype=np.uint32)
        self.cols = np.asarray(cols, dtype=np.uint32)
        self.nnz = self.rows.shape[0]

        self.blocksizes = np.asarray(blocksizes, dtype=np.uint32)
        self.blockoffsets = np.hstack(([0], np.cumsum(blocksizes)))
        self.shape = (self.blockoffsets[-1], self.blockoffsets[-1])
        self.num_blocks = self.blocksizes.shape[0]

        # TODO: Implement pinned memory support.
        self.data = np.asarray(data, dtype=dtype)
        self.stackshape = stackshape

        if blockmap is not None:
            blockmap = np.asarray(blockmap, dtype=np.uint32)
        else:
            blockmap = self._compute_blockmap()

        self.blockmap = blockmap

    def _compute_blockmap(self) -> np.ndarray:
        """Computes blockmap from rows, cols and blocksizes."""
        blockmap = np.zeros(
            (self.num_blocks, self.num_blocks, self.nnz), dtype=np.bool8
        )
        for i, j in np.ndindex(self.num_blocks, self.num_blocks):
            blockmap[i, j] = (
                (self.blockoffsets[i] <= self.rows)
                & (self.rows < self.blockoffsets[i + 1])
                & (self.blockoffsets[j] <= self.cols)
                & (self.cols < self.blockoffsets[j + 1])
            )
        return blockmap

    def _unsign_block_index(self, brow: int, bcol: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        brow = self.num_blocks + brow if brow < 0 else brow
        bcol = self.num_blocks + bcol if bcol < 0 else bcol
        if not (0 <= brow < self.num_blocks and 0 <= bcol < self.num_blocks):
            raise IndexError("Block index out of bounds.")

        return brow, bcol

    def get_block(self, brow: int, bcol: int) -> np.ndarray:
        """Returns the block at the given block-row and -column."""
        brow, bcol = self._unsign_block_index(brow, bcol)

        block = np.zeros(
            (*self.stackshape, self.blocksizes[brow], self.blocksizes[bcol]),
            dtype=self.data.dtype,
        )
        inds = self.blockmap[brow, bcol]
        rows = self.rows[inds] - self.blockoffsets[brow]
        cols = self.cols[inds] - self.blockoffsets[bcol]
        block[..., rows, cols] = self.data[..., inds]

        return block

    def set_block(self, brow: int, bcol: int, block: ArrayLike) -> None:
        """Sets the block at the given block-row and -column."""
        brow, bcol = self._unsign_block_index(brow, bcol)
        block = np.asarray(block, dtype=self.data.dtype)

        if block.shape != (
            *self.stackshape,
            self.blocksizes[brow],
            self.blocksizes[bcol],
        ):
            raise ValueError("Block shape does not match.")

        inds = self.blockmap[brow, bcol]
        rows = self.rows[inds] - self.blockoffsets[brow]
        cols = self.cols[inds] - self.blockoffsets[bcol]
        self.data[..., inds] = block[..., rows, cols]

    def to_dense(self) -> np.ndarray:
        """Returns the dense matrix representation."""
        dense = np.zeros(
            (*self.stackshape, *self.shape),
            dtype=self.data.dtype,
        )
        dense[..., self.rows, self.cols] = self.data

        return dense
