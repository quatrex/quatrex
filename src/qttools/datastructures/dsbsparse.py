# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import copy
from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.utils.gpu_utils import ArrayLike, get_host, synchronize_current_stream, xp
from qttools.utils.mpi_utils import get_section_sizes


def _block_view(arr: ArrayLike, axis: int):
    """Gets a block view of an array along a given axis.

    This is a helper function to get a block view of an array along a
    given axis. This is useful for the distributed transposition of
    arrays, where we need to transpose the data through the network.

    This is stolen from `skimage.util.view_as_blocks`.

    Parameters
    ----------
    arr : array_like
        The array to get the block view of.
    axis : int
        The axis along which to get the block view.

    """
    block_shape = list(arr.shape)
    block_shape[axis] //= comm.size

    new_shape = (comm.size,) + tuple(block_shape)
    new_strides = (arr.strides[axis] * block_shape[axis],) + arr.strides

    return xp.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)


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

    When calling the `dtranspose` method, the data is transposed through
    the network. This is done by first reshaping the local data, then
    performing an Alltoall communication, and finally reshaping the data
    back to the correct new shape. The local reshaping of the data
    cannot be done entirely in-place. This can lead to pronounced memory
    peaks if all ranks start reshaping concurrently, which can be
    mitigated by using more ranks and by not forcing a synchronization
    barrier right before calling `dtranspose`.

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
    data : array_like
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
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
        block_sizes: ArrayLike,
        global_stack_shape: tuple | int,
        return_dense: bool = False,
    ) -> None:
        """Initializes the DBSparse matrix."""
        if isinstance(global_stack_shape, int):
            global_stack_shape = (global_stack_shape,)

        self.global_stack_shape = global_stack_shape

        # Determine how the data is distributed across the ranks.
        stack_section_sizes, total_stack_size = get_section_sizes(
            global_stack_shape[0], comm.size
        )
        nnz_section_sizes, total_nnz_size = get_section_sizes(data.shape[-1], comm.size)

        self.stack_section_sizes = stack_section_sizes
        self.nnz_section_sizes = nnz_section_sizes
        self.total_stack_size = total_stack_size
        self.total_nnz_size = total_nnz_size

        # Per default, we have the data is distributed in stack format.
        self.distribution_state = "stack"

        # Pad local data with zeros to ensure that all ranks have the
        # same data size for the in-place Alltoall communication.
        self._data = xp.zeros(
            (max(stack_section_sizes), *global_stack_shape[1:], total_nnz_size),
            dtype=data.dtype,
        )
        self._data[: data.shape[0], ..., : data.shape[-1]] = data

        self.dtype = data.dtype

        self.stack_shape = data.shape[:-1]
        self.nnz = data.shape[-1]
        self.shape = self.stack_shape + (int(sum(block_sizes)), int(sum(block_sizes)))

        self.block_sizes = xp.asarray(block_sizes).astype(int)
        self.block_offsets = xp.hstack(([0], xp.cumsum(self.block_sizes)))
        self.num_blocks = len(block_sizes)
        self.return_dense = return_dense

    @property
    def blocks(self) -> "_DSBlockIndexer":
        """Returns a block indexer."""
        return _DSBlockIndexer(self)

    @property
    def data(self) -> ArrayLike:
        """Returns the local slice of the data, masking the padding.

        This does not return a copy of the data, but a view. This is
        also why we do not need a setter method (one can just set
        `.data` directly).

        """
        if self.distribution_state == "stack":
            return self._data[
                : self.stack_section_sizes[comm.rank],
                ...,
                : sum(self.nnz_section_sizes),
            ]
        return self._data[
            : sum(self.stack_section_sizes), ..., : self.nnz_section_sizes[comm.rank]
        ]

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"block_sizes={self.block_sizes}, "
            f"global_stack_shape={self.global_stack_shape}, "
            f'distribution_state="{self.distribution_state}")'
        )

    @abstractmethod
    def _set_block(self, row: int, col: int, block: ArrayLike) -> None:
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
        block : array_like
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        ...

    @abstractmethod
    def _get_block(self, row: int, col: int) -> ArrayLike:
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
        block : array_like
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

    def block_diagonal(self, offset: int = 0) -> list[ArrayLike]:
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

    def diagonal(self) -> ArrayLike:
        """Returns the diagonal elements of the matrix.

        This temporarily sets the return_dense state to True.

        Returns
        -------
        diagonal : array_like
            The diagonal elements of the matrix.

        """
        # Store the current return_dense state and set it to True.
        original_return_dense = self.return_dense
        self.return_dense = True

        diagonals = []
        for b in range(self.num_blocks):
            diagonals.append(xp.diagonal(self.blocks[b, b], axis1=-2, axis2=-1))

        # Restore the original return_dense state.
        self.return_dense = original_return_dense
        return xp.hstack(diagonals)

    def _stack_to_nnz_dtranspose(self) -> None:
        """Transposes the data from stack to nnz distribution."""
        # Preserve old shape and compute new shape.
        old_shape = self._data.shape
        new_shape = (
            old_shape[0] * comm.size,
            *old_shape[1:-1],
            old_shape[-1] // comm.size,
        )

        self._data = _block_view(self._data, axis=-1)
        # We need to make sure that the block-view is memory-contiguous.
        self._data = xp.ascontiguousarray(self._data)

        synchronize_current_stream()
        comm.Alltoall(MPI.IN_PLACE, self._data)

        self._data = self._data.reshape(new_shape)

    def _nnz_to_stack_dtranspose(self) -> None:
        """Transposes the data from nnz to stack distribution."""
        # Preserve old shape and compute new shape.
        old_shape = self._data.shape
        new_shape = (
            old_shape[0] // comm.size,
            *old_shape[1:-1],
            old_shape[-1] * comm.size,
        )

        # Here the data is already contiguous in memory.
        self._data = _block_view(self._data, axis=0)

        synchronize_current_stream()
        comm.Alltoall(MPI.IN_PLACE, self._data)

        # The blocks we receive are now flipped, so transpose them back.
        self._data = self._data.swapaxes(0, -2)
        self._data = self._data.reshape(new_shape)

    def dtranspose(self) -> None:
        """Performs a distributed transposition of the datastructure.

        This is done by reshaping the local data, then performing an
        in-place Alltoall communication, and finally reshaping the data
        back to the correct new shape.

        The local reshaping of the data cannot be done entirely
        in-place. This can lead to pronounced memory peaks if all ranks
        start reshaping concurrently, which can be mitigated by using
        more ranks and by not forcing a synchronization barrier right
        before calling `dtranspose`.

        """
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
        if self.distribution_state != "stack":
            raise ValueError(
                "Conversion to dense is only supported in 'stack' distribution state."
            )

        original_return_dense = self.return_dense
        self.return_dense = True

        arr = xp.zeros(self.shape, dtype=self.dtype)
        for i, j in xp.ndindex(self.num_blocks, self.num_blocks):
            arr[
                ...,
                self.block_offsets[i] : self.block_offsets[i + 1],
                self.block_offsets[j] : self.block_offsets[j + 1],
            ] = self._get_block(i, j)

        self.return_dense = original_return_dense

        return get_host(arr)

    @classmethod
    @abstractmethod
    def from_sparray(
        cls,
        arr: sparse.sparray,
        block_sizes: ArrayLike,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBSparse":
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
        DSBSparse
            The new DSBSparse matrix.

        """
        ...

    @classmethod
    def zeros_like(cls, dsbsparse: "DSBSparse") -> "DSBSparse":
        """Creates a new DSBSparse matrix with the same shape and dtype.

        All non-zero elements are set to zero, but the sparsity pattern
        is preserved.

        Parameters
        ----------
        dsbsparse : DSBSparse
            The matrix to copy the shape and dtype from.

        Returns
        -------
        DSBSparse
            The new DSBSparse matrix.

        """
        out = copy.deepcopy(dsbsparse)
        out.data[:] = 0.0
        return out


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
        """Initializes the block indexer."""
        self.dsbsparse = dsbsparse
        self.num_blocks = dsbsparse.num_blocks
        self.block_sizes = dsbsparse.block_sizes
        self.return_dense = dsbsparse.return_dense

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self.dsbsparse.num_blocks + row if row < 0 else row
        col = self.dsbsparse.num_blocks + col if col < 0 else col
        if not (0 <= row < self.num_blocks and 0 <= col < self.num_blocks):
            raise IndexError("Block index out of bounds.")

        return row, col

    def _normalize_index(self, index: tuple) -> tuple:
        """Normalizes the block index."""
        if self.dsbsparse.distribution_state != "stack":
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if len(index) != 2:
            raise IndexError("Exactly two block indices are required.")

        row, col = index
        if isinstance(row, slice) or isinstance(col, slice):
            raise NotImplementedError("Slicing is not supported.")

        row, col = self._unsign_index(row, col)
        return row, col

    def __getitem__(self, index: tuple) -> ArrayLike:
        """Gets the requested block from the data structure."""
        row, col = self._normalize_index(index)
        return self.dsbsparse._get_block(row, col)

    def __setitem__(self, index: tuple, block: ArrayLike) -> None:
        """Sets the requested block in the data structure."""
        row, col = self._normalize_index(index)
        self.dsbsparse._set_block(row, col, block)
