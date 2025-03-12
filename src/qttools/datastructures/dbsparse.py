# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.kernels import dsbcoo_kernels
from qttools.utils.mpi_utils import check_gpu_aware_mpi, get_section_sizes

GPU_AWARE_MPI = check_gpu_aware_mpi()


def find_split_indices(rows: NDArray, cols: NDArray, block_sizes: NDArray) -> NDArray:
    """Finds the indices at which to split the block-sorted data vector.

    Parameters
    ----------
    rows : NDArray
        The row indices of the block-sorted data.
    cols : NDArray
        The column indices of the block-sorted data.
    block_sizes : NDArray
        The block sizes of the block-sparse matrix.

    """
    section_sizes, __ = get_section_sizes(len(block_sizes), comm.size)
    section_offsets = xp.hstack(([0], xp.cumsum(xp.array(section_sizes))))

    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
    split_indices = []

    for i in range(comm.size - 1):
        ind = section_offsets[i + 1]
        start, __ = dsbcoo_kernels.compute_block_slice(
            rows, cols, block_offsets, ind, ind
        )
        split_indices.append(int(start))

    return split_indices


class DBSparse(ABC):
    """Base class for nnz-distributed block-accessible sparse matrices."""

    def __init__(self):
        """Initializes a DBSparse object."""
        self.blocks = None
        self.block_sizes = None
        self.num_blocks: int = None
        self.local_blocks = None
        self.local_block_sizes = None
        self.num_local_blocks: int = None

    @abstractmethod
    def _get_block(self, row: int, col: int) -> NDArray:
        """Gets the block at the specified row and column."""
        ...

    @abstractmethod
    def _set_block(self, row: int, col: int, block: NDArray) -> None:
        """Sets the block at the specified row and column."""
        ...

    @classmethod
    @abstractmethod
    def from_sparray(
        cls, sparray: sparse.coo_matrix, block_sizes: NDArray
    ) -> "DBSparse":
        """Constructs a DBSparse matrix from a COO matrix."""
        ...

    @abstractmethod
    def to_dense(self) -> NDArray:
        """Converts the DBSparse matrix to a dense matrix."""
        ...


class DBCOO(DBSparse):
    """A distributed, block-accessible COO matrix.

    Parameters
    ----------
    local_data : NDArray
        The local data of the COO matrix.
    local_rows : NDArray
        The local row indices of the COO matrix.
    local_cols : NDArray
        The local column indices of the COO matrix.
    block_sizes : NDArray
        The block sizes of the block-sparse matrix.

    """

    def __init__(
        self,
        local_data: NDArray,
        local_rows: NDArray,
        local_cols: NDArray,
        block_sizes: NDArray,
    ):
        """Initializes a DBCOO matrix."""
        # Important stuff first.
        self.local_data = xp.asarray(local_data, dtype=local_data.dtype)
        self.dtype = self.local_data.dtype
        self.local_rows = xp.asarray(local_rows, dtype=int)
        self.local_cols = xp.asarray(local_cols, dtype=int)

        self.block_sizes = block_sizes
        self.num_blocks = len(block_sizes)

        section_sizes, __ = get_section_sizes(len(block_sizes))
        section_offsets = xp.hstack(([0], xp.cumsum(xp.array(section_sizes))))

        self.num_local_blocks = section_sizes[comm.rank]
        self.local_block_sizes = block_sizes[..., int(section_offsets[comm.rank]) :]

        self.block_offsets = xp.hstack(([0], xp.cumsum(self.block_sizes)))
        self.local_block_offsets = xp.hstack(([0], xp.cumsum(self.local_block_sizes)))

        # Since the data is block-wise contiguous, we can cache block
        # *slices* for faster access.
        self._block_slice_cache = {}

    @property
    def local_blocks(self) -> "_DBlockIndexer":
        """Returns a block indexer."""
        return _DBlockIndexer(self)

    def _get_block_slice(self, row: int, col: int) -> slice:
        """Gets the slice of data corresponding to a given block.

        This handles the block slice cache. If there is no data in the
        block, an `slice(None)` is cached.

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
            block_slice = slice(
                *dsbcoo_kernels.compute_block_slice(
                    self.local_rows, self.local_cols, self.local_block_offsets, row, col
                )
            )

        self._block_slice_cache[(row, col)] = block_slice
        return block_slice

    def _get_block(self, row: int, col: int) -> NDArray:
        """Gets the block at the specified row and column."""
        block_slice = self._get_block_slice(row, col)

        block = xp.zeros(
            (int(self.local_block_sizes[row]), int(self.local_block_sizes[col])),
            dtype=self.dtype,
        )
        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, return an empty block.
            return block

        dsbcoo_kernels.densify_block(
            block,
            self.local_rows[block_slice] - self.local_block_offsets[row],
            self.local_cols[block_slice] - self.local_block_offsets[col],
            self.local_data[block_slice],
        )

        return block

    def _set_block(self, row: int, col: int, block: NDArray) -> None:
        """Sets the block at the specified row and column."""
        block_slice = self._get_block_slice(row, col)
        if block_slice.start is None and block_slice.stop is None:
            # No data in this block, nothing to do.
            return

        dsbcoo_kernels.sparsify_block(
            block,
            self.local_rows[block_slice] - self.local_block_offsets[row],
            self.local_cols[block_slice] - self.local_block_offsets[col],
            self.local_data[block_slice],
        )

    def to_dense(self):
        """Converts the DBCOO matrix to a dense matrix."""
        # Gather rows, cols, and data.
        rows = comm.allgather(self.local_rows)
        cols = comm.allgather(self.local_cols)
        data = xp.hstack(comm.allgather(self.local_data))

        rank_max = xp.hstack(
            comm.allgather(sum(self.local_block_sizes[: self.num_local_blocks]))
        )
        rank_offset = xp.hstack(([0], xp.cumsum(rank_max)))

        for i in range(1, comm.size):
            rows[i] += rank_offset[i]
            cols[i] += rank_offset[i]

        rows = xp.hstack(rows)
        cols = xp.hstack(cols)

        # Create the dense matrix
        # dense = xp.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=self.dtype)
        size = int(sum(self.block_sizes))
        dense = xp.zeros((size, size), dtype=self.dtype)
        dense[rows, cols] = data

        return dense

    @classmethod
    def from_sparray(cls, sparray: sparse.coo_matrix, block_sizes: NDArray) -> "DBCOO":
        """Constructs a DBCOO matrix from a COO matrix.

        This essentially distributes the COO matrix across the
        participating ranks.

        Parameters
        ----------
        sparray : sparse.coo_matrix
            The COO matrix to distribute.
        block_sizes : NDArray
            The block sizes of the block-sparse matrix.

        Returns
        -------
        DBCOO
            The distributed, block-accessible COO matrix.

        """
        coo: sparse.coo_matrix = sparray.tocoo().copy()

        # Canonicalizes the COO format.
        coo.sum_duplicates()

        # Compute the block-sorting index.
        block_sort_index = dsbcoo_kernels.compute_block_sort_index(
            coo.row, coo.col, block_sizes
        )

        data = xp.zeros((coo.nnz,), dtype=coo.data.dtype)
        data[..., :] = coo.data[block_sort_index]
        rows = coo.row[block_sort_index]
        cols = coo.col[block_sort_index]

        # Find indices
        # NOTE: This is arrow-wise partitioning.
        # TODO: Allow more options, e.g., block row-wise partitioning.
        section_sizes, __ = get_section_sizes(len(block_sizes), comm.size)
        section_offsets = xp.hstack(([0], xp.cumsum(xp.array(section_sizes))))
        block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
        start_idx = block_offsets[section_offsets[comm.rank]]
        end_idx = block_offsets[section_offsets[comm.rank + 1]]
        indices = xp.logical_and(
            xp.logical_and(rows >= start_idx, cols >= start_idx),
            xp.logical_or(rows < end_idx, cols < end_idx),
        )
        local_data = data[indices]
        local_rows = rows[indices]
        local_cols = cols[indices]

        # Normalize the row and column indices.
        # local_rows -= local_rows.min()
        # local_cols -= local_cols.min()
        local_rows -= start_idx
        local_cols -= start_idx

        return cls(
            local_data=local_data,
            local_rows=local_rows,
            local_cols=local_cols,
            block_sizes=block_sizes,
        )


class _DBlockIndexer:
    """A utility class to locate blocks in the distributed stack.

    This uses the `_get_block` and `_set_block` methods of the
    underlying DBSparse object to locate and set blocks in the stack.
    It further allows slicing and more advanced indexing by repeatedly
    calling the low-level methods.

    Parameters
    ----------
    dbsparse : DBSparse
        The underlying datastructure

    """

    def __init__(self, dbsparse: DBSparse) -> None:
        """Initializes the block indexer."""
        self._dbsparse = dbsparse
        self._num_local_blocks = dbsparse.num_local_blocks

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self._num_local_blocks + row if row < 0 else row
        col = self._num_local_blocks + col if col < 0 else col
        # NOTE: Bounds checking is not needed now.
        # if not (0 <= row < self._num_blocks and 0 <= col < self._num_blocks):
        #     raise IndexError("Block index out of bounds.")

        return row, col

    def _normalize_index(self, index: tuple) -> tuple:
        """Normalizes the block index."""
        if len(index) != 2:
            raise IndexError("Exactly two block indices are required.")

        row, col = index
        if isinstance(row, slice) or isinstance(col, slice):
            raise NotImplementedError("Slicing is not supported.")

        return self._unsign_index(row, col)

    def __getitem__(self, index: tuple) -> NDArray | tuple:
        """Gets the requested block from the data structure."""
        row, col = self._normalize_index(index)
        return self._dbsparse._get_block(row, col)

    def __setitem__(self, index: tuple, block: NDArray) -> None:
        """Sets the requested block in the data structure."""
        row, col = self._normalize_index(index)
        self._dbsparse._set_block(row, col, block)
