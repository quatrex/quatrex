# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

import numpy as np
import numpy.lib.stride_tricks as npst
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.utils.mpi_utils import get_num_elements_per_section


class DSBSparse(ABC):
    """Base class for distributed stacks of sparse block matrices.

    In NEGF, all scattering self-energies, Green's functions, and system
    matrices are sparse matrices in the real space basis (due to the
    orbital interaction cutoff). Since they are also energy and k-point
    dependent, we represent the entire object as a stack of sparse
    matrices with identical sparsity pattern. For each energy and
    k-point, we have exactly one data vector, while we only need to keep
    one global sparsity pattern.

    Due to the large amount of total data, and to facilitate parallel
    processing, the entire data structure needs to be distributed across
    the participating MPI ranks. This can either be done by distributing
    smaller stacks of entire sparse matrices across the ranks, or by
    distributing the non-zero elements of the sparse matrices across the
    ranks. In NEGF, we use both approaches; stack-distribution to
    compute the Green's functions, and nnz-distribution to compute the
    scattering self-energies accross the ranks.

    To allow for (almost) in-place transposition of the data through the
    network, even if the number of ranks does not evenly divide the
    stack-size / number of non-zero elements, the data is stored with
    some padding on each rank.

    DSBSparse implementations should provide the following methods:
    - `_set_block(row, col, block)`: Sets a block throughout the stack.
    - `_get_block(row, col)`: Gets a block from the stack.
    - `__iadd__(other)`: In-place addition.
    - `__imul__(other)`: In-place multiplication.
    - `__imatmul__(other)`: In-place matrix multiplication.
    - `__neg__()`: In-place negation.
    - `ltranspose()`: Local transposition.
    - `to_dense()`: Convert to dense.
    - `from_sparray()`: Create from a scipy.sparse array.
    - `zeros_like()`: Create a new object with the same shape and dtype
      and no non-zero elements.

    Note that only in-place arithmetic operations are required by this
    interface. We never want to implicitly create a new object.

    Parameters
    ----------
    data : np.ndarray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    block_sizes : np.ndarray
        The size of each block in the sparse matrix.
    global_stack_shape : tuple
        The global shape of the stack.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is False.

    """

    def __init__(
        self,
        data: np.ndarray,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        return_dense: bool = False,
    ) -> None:
        """Initializes the DBSparse matrix."""
        if isinstance(global_stack_shape, int):
            global_stack_shape = (global_stack_shape,)

        if data.ndim != 2 or len(global_stack_shape) != 1:
            raise NotImplementedError("Currently only 2D data is supported.")

        self.global_stack_shape = global_stack_shape

        # Determine how the data is distributed across the ranks.
        stack_section_sizes, total_stack_size = get_num_elements_per_section(
            global_stack_shape[0], comm.size
        )
        nnz_section_sizes, total_nnz_size = get_num_elements_per_section(
            data.shape[-1], comm.size
        )

        self.stack_section_sizes = stack_section_sizes
        self.nnz_section_sizes = nnz_section_sizes
        self.total_stack_size = total_stack_size
        self.total_nnz_size = total_nnz_size

        # Per default, we have the data is distributed in stack format.
        self.distribution_state = "stack"

        # Pad local data with zeros to ensure that all ranks have the
        # same data size for the in-place Alltoall communication.
        self._data = np.zeros(
            (max(stack_section_sizes), total_nnz_size), dtype=data.dtype
        )
        self._data[: data.shape[0], : data.shape[1]] = data

        self.dtype = data.dtype

        self.stack_shape = data.shape[:-1]
        self.nnz = data.shape[-1]
        self.shape = self.stack_shape + (np.sum(block_sizes), np.sum(block_sizes))

        self.block_sizes = np.asarray(block_sizes).astype(int)
        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        self.num_blocks = len(block_sizes)
        self.return_dense = return_dense

    @property
    def blocks(self) -> "_DSBlockIndexer":
        """Returns a block indexer."""
        return _DSBlockIndexer(self)

    @property
    def data(self) -> np.ndarray:
        """Returns the local slice of the data, masking the padding.

        This does not return a copy of the data, but a view. This is
        also why we do not need a setter method (one can just set
        `.data` directly).

        """
        if self.distribution_state == "stack":
            return self._data[
                : self.stack_section_sizes[comm.rank],
                : sum(self.nnz_section_sizes),
            ]
        return self._data[
            : sum(self.stack_section_sizes), : self.nnz_section_sizes[comm.rank]
        ]

    @abstractmethod
    def _set_block(self, row: int, col: int, block: np.ndarray) -> None:
        """Sets a block throughout the stack in the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
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
        ...

    @abstractmethod
    def _get_block(self, row: int, col: int) -> sparse.sparray | np.ndarray:
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
        block : sparse.sparray | np.ndarray
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        ...

    @abstractmethod
    def __iadd__(self, other: "DSBSparse") -> None:
        """In-place addition of two DSBSparse matrices."""
        ...

    @abstractmethod
    def __imul__(self, other: "DSBSparse") -> None:
        """In-place multiplication of two DSBSparse matrices."""
        ...

    @abstractmethod
    def __neg__(self) -> None:
        """Negation of the data."""
        ...

    @abstractmethod
    def __matmul__(self, other: "DSBSparse") -> None:
        """Matrix multiplication of two DSBSparse matrices."""
        ...

    def block_diagonal(
        self, offset: int = 0
    ) -> list[sparse.sparray] | list[np.ndarray]:
        """Returns the block diagonal of the matrix.

        Parameters
        ----------
        offset : int, optional
            Offset from the main diagonal. Positive values indicate
            superdiagonals, negative values indicate subdiagonals.
            Default is 0.

        Returns
        -------
        blocks : list
            List of block diagonal elements. The length of the list is
            the number of blocks on the main diagonal minus the offset.
            Depending on return_dense, the elements are either sparse
            or dense arrays.

        """
        blocks = []
        for b in range(self.num_blocks - abs(offset)):
            blocks.append(self.blocks[b, b + offset])

        return blocks

    def diagonal(self) -> np.ndarray:
        """Returns the diagonal elements of the matrix.

        This temporarily sets the return_dense state to True.

        Returns
        -------
        diagonal : np.ndarray
            The diagonal elements of the matrix.

        """
        # Store the current return_dense state and set it to True.
        original_return_dense = self.return_dense
        self.return_dense = True

        diagonals = []
        for b in range(self.num_blocks):
            diagonals.append(np.diagonal(self.blocks[b, b], axis1=-2, axis2=-1))

        # Restore the original return_dense state.
        self.return_dense = original_return_dense
        return np.hstack(diagonals)

    def _stack_to_nnz_dtranspose(self) -> None:
        """Transpose the data."""
        original_buffer_shape = self._data.shape
        self._data = np.ascontiguousarray(
            npst.as_strided(
                self._data,
                shape=(
                    comm.size,
                    self._data.shape[0],
                    self._data.shape[1] // comm.size,
                ),
                strides=(
                    (self._data.shape[1] // comm.size) * self._data.itemsize,
                    self._data.shape[1] * self._data.itemsize,
                    self._data.itemsize,
                ),
            )
        )
        comm.Alltoall(MPI.IN_PLACE, self._data)

        self._data = self._data.reshape(
            original_buffer_shape[0] * comm.size, original_buffer_shape[1] // comm.size
        )

    def _nnz_to_stack_dtranspose(self) -> None:
        """Transpose the data."""
        original_buffer_shape = self._data.shape
        self._data = self._data.reshape(
            comm.size,
            self._data.shape[0] // comm.size,
            self._data.shape[1],
        )
        comm.Alltoall(MPI.IN_PLACE, self._data)
        self._data = self._data.transpose(1, 0, 2)
        self._data = self._data.reshape(
            original_buffer_shape[0] // comm.size, original_buffer_shape[1] * comm.size
        )

    def dtranspose(self) -> None:
        """Performs a distributed transposition of the datastructure."""
        if self.distribution_state == "stack":
            self._stack_to_nnz_dtranspose()
            self.distribution_state = "nnz"
        else:
            self._nnz_to_stack_dtranspose()
            self.distribution_state = "stack"

    @abstractmethod
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
        ...

    @abstractmethod
    def ltranspose(self, copy=False) -> "None | DSBSparse":
        """Performs a local transposition of the matrix.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a new object. Default is False.

        Returns
        -------
        None | DSBSparse
            The transposed matrix. If copy is False, this is None.

        """
        ...

    def to_dense(self) -> np.ndarray:
        """Converts the local data to a dense array.

        This is dumb, unless used for testing and debugging.

        Returns
        -------
        arr : np.ndarray
            The dense array of shape `(*local_stack_shape, *shape)`.
        """
        original_return_dense = self.return_dense
        self.return_dense = True

        arr = np.zeros(self.shape, dtype=self.dtype)
        for i, j in np.ndindex(self.num_blocks, self.num_blocks):
            arr[
                ...,
                self.block_offsets[i] : self.block_offsets[i + 1],
                self.block_offsets[j] : self.block_offsets[j + 1],
            ] = self._get_block(i, j)

        self.return_dense = original_return_dense

        return arr

    @classmethod
    @abstractmethod
    def from_sparray(
        cls,
        a: sparse.sparray,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBSparse": ...

    @classmethod
    @abstractmethod
    def zeros_like(cls, a: "DSBSparse") -> "DSBSparse": ...


class _DSBlockIndexer:
    """A utility class to locate blocks in the distributed stack.

    This uses the `_get_block` and `_set_block` methods of the
    underlying DSBSparse object to locate and set blocks in the stack.
    It further allows slicing and more advanced indexing by repeatedly
    calling the low-level methods.

    Parameters
    ----------
    dsbsparse : DSBSparse
        The underlying datastructure

    """

    def __init__(self, dsbsparse: DSBSparse) -> None:
        self.dsbsparse = dsbsparse
        self.num_blocks = dsbsparse.num_blocks
        self.block_sizes = dsbsparse.block_sizes
        self.return_dense = dsbsparse.return_dense

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds.

        Parameters
        ----------
        row : int
            Block row index.
        col : int
            Block column index.

        Returns
        -------
        tuple
            Renormalized block indices.

        """
        row = self.dsbsparse.num_blocks + row if row < 0 else row
        col = self.dsbsparse.num_blocks + col if col < 0 else col
        if not (0 <= row < self.num_blocks and 0 <= col < self.num_blocks):
            raise IndexError("Block index out of bounds.")

        return row, col

    def __getitem__(self, key: tuple | slice) -> int:
        """Gets the requested block(s) from the data structure."""
        if self.dsbsparse.distribution_state == "nnz":
            raise NotImplementedError(
                "Block indexing is not supported in 'stack' distribution state."
            )

        row, col = key
        return self.dsbsparse._get_block(row, col)

        # if len(stack_index) > len(self.stack_shape):
        #     raise ValueError(
        #         f"Too many stack indices for stack shape '{self.stack_shape}'."
        #     )

        # stack_index += (slice(None),) * (len(self.stack_shape) - len(stack_index))

    def __setitem__(self, key: tuple | slice, value: int) -> None:
        """Sets the requested block(s) in the data structure."""
        # if self.dsbsparse.distribution_state == "stack":
        #     raise NotImplementedError(
        #         "Block indexing is not supported in stack distribution state."
        #     )

        # if self.distribution_state == "nnz":
        #     raise NotImplementedError("Cannot get blocks when distributed through nnz.")
        # if not self.return_dense:
        #     raise NotImplementedError("Sparse array not yet implemented.")

        # if len(key) < 2:
        #     raise ValueError("At least the two block indices are required.")

        # if len(key) >= 2:
        #     *stack_index, brow, bcol = key

        # if len(stack_index) > len(self.stack_shape):
        #     raise ValueError(
        #         f"Too many stack indices for stack shape '{self.stack_shape}'."
        #     )

        # stack_index += (slice(None),) * (len(self.stack_shape) - len(stack_index))

        # brow, bcol = self._unsign_block_index(brow, bcol)

    # if block.shape[-2:] != (
    #             self.block_sizes[brow],
    #             self.block_sizes[bcol],
    #         ):
    #             raise ValueError("Block shape does not match.")
