# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import copy
from abc import ABC, abstractmethod

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import ArrayLike, NDArray, sparse, xp
from qttools.utils.gpu_utils import get_host, synchronize_current_stream
from qttools.utils.mpi_utils import check_gpu_aware_mpi, get_section_sizes

GPU_AWARE_MPI = check_gpu_aware_mpi()


def _block_view(arr: NDArray, axis: int, num_blocks: int = comm.size) -> NDArray:
    """Gets a block view of an array along a given axis.

    This is a helper function to get a block view of an array along a
    given axis. This is useful for the distributed transposition of
    arrays, where we need to transpose the data through the network.

    This is stolen from `skimage.util.view_as_blocks`.

    Parameters
    ----------
    arr : NDArray
        The array to get the block view of.
    axis : int
        The axis along which to get the block view.
    num_blocks : int, optional
        The number of blocks to divide the array into. Default is the
        number of MPI ranks in the communicator.

    Returns
    -------
    block_view : NDArray
        The specified block view of the array.

    """
    block_shape = list(arr.shape)

    if block_shape[axis] % num_blocks != 0:
        raise ValueError("The array shape is not divisible by the number of blocks.")

    block_shape[axis] //= num_blocks

    new_shape = (num_blocks,) + tuple(block_shape)
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

    - `_set_block(stack_index, row, col, block)`: Sets a block.
    - `_get_block(stack_index, row, col)`: Gets a block the stack.
    - `_getitems(stack_index, row, col)`: Gets items from the data.
    - `_setitems(stack_index, row, col)`: Sets items in the data.
    - `_check_commensurable(other)`: Checks if two DSBSparse matrices
       are commensurable.
    - `__imatmul__(other)`: In-place matrix multiplication.
    - `__neg__()`: In-place negation.
    - `ltranspose()`: Local transposition.
    - `from_sparray()`: Create from scipy.sparse array.

    Note that only in-place arithmetic operations are required by this
    interface. We never want to implicitly create a new object.

    Parameters
    ----------
    data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, nnz)`. It is the caller's responsibility
        to ensure that the data is distributed correctly across the
        ranks.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    global_stack_shape : tuple or int
        The global shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.

    """

    def __init__(
        self,
        data: NDArray,
        block_sizes: NDArray,
        global_stack_shape: tuple | int,
        return_dense: bool = True,
    ) -> None:
        """Initializes the DSBSparse matrix."""
        if isinstance(global_stack_shape, int):
            global_stack_shape = (global_stack_shape,)

        if global_stack_shape[0] < comm.size:
            raise ValueError(
                f"Number of MPI ranks {comm.size} exceeds stack shape {global_stack_shape}."
            )

        self.global_stack_shape = global_stack_shape

        # Determine how the data is distributed across the ranks.
        stack_section_sizes, total_stack_size = get_section_sizes(
            global_stack_shape[0], comm.size, strategy="balanced"
        )
        self.stack_section_sizes = xp.array(stack_section_sizes)
        self.total_stack_size = total_stack_size

        nnz_section_sizes, total_nnz_size = get_section_sizes(
            data.shape[-1], comm.size, strategy="greedy"
        )
        self.nnz_section_sizes = xp.array(nnz_section_sizes)
        self.nnz_section_offsets = xp.hstack(([0], xp.cumsum(self.nnz_section_sizes)))
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

        # For the weird padding convention we use, we need to keep track
        # of this padding mask.
        # NOTE: We should maybe consistently use the greedy strategy for
        # the stack distribution as well.
        self._stack_padding_mask = xp.zeros(total_stack_size, dtype=bool)
        for i, size in enumerate(stack_section_sizes):
            offset = i * max(stack_section_sizes)
            self._stack_padding_mask[offset : offset + size] = True
        self.dtype = data.dtype

        self.stack_shape = data.shape[:-1]
        self.nnz = data.shape[-1]
        self.shape = self.stack_shape + (int(sum(block_sizes)), int(sum(block_sizes)))

        self._block_sizes = xp.asarray(block_sizes).astype(int)
        self.block_offsets = xp.hstack(([0], xp.cumsum(self.block_sizes)))
        self.num_blocks = len(block_sizes)
        self.return_dense = return_dense

    @property
    def block_sizes(self) -> ArrayLike:
        """Returns the block sizes."""
        return self._block_sizes

    @block_sizes.setter
    @abstractmethod
    def block_sizes(self, block_sizes: ArrayLike) -> None:
        """Sets the block sizes."""
        ...

    def _normalize_index(self, index: tuple) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        if not isinstance(index, tuple):
            raise IndexError("Invalid index.")

        if not len(index) == 2:
            raise IndexError("Invalid index.")

        row, col = index

        row = xp.asarray(row, dtype=int)
        col = xp.asarray(col, dtype=int)

        # Ensure that the indices are at least 1-D arrays.
        row = xp.atleast_1d(row)
        col = xp.atleast_1d(col)

        row = xp.where(row < 0, self.shape[-2] + row, row)
        col = xp.where(col < 0, self.shape[-1] + col, col)
        if not (
            ((0 <= row) & (row < self.shape[-2])).all()
            and ((0 <= col) & (col < self.shape[-1])).all()
        ):
            raise IndexError("Index out of bounds.")

        return row, col

    def __getitem__(self, index: tuple[ArrayLike, ArrayLike]) -> NDArray:
        """Gets a single value accross the stack."""
        index = self._normalize_index(index)
        return self._get_items((Ellipsis,), *index)

    def __setitem__(self, index: tuple[ArrayLike, ArrayLike], value: NDArray) -> None:
        """Sets a single value in the matrix."""
        index = self._normalize_index(index)
        self._set_items((Ellipsis,), *index, value)

    @property
    def blocks(self) -> "_DSBlockIndexer":
        """Returns a block indexer."""
        return _DSBlockIndexer(self)

    @property
    def stack(self) -> "_DStackIndexer":
        """Returns a stack indexer."""
        return _DStackIndexer(self)

    @property
    def data(self) -> NDArray:
        """Returns the local slice of the data, masking the padding."""
        if self.distribution_state == "stack":
            return self._data[
                : self.stack_section_sizes[comm.rank],
                ...,
                : sum(self.nnz_section_sizes),
            ]
        return self._data[
            self._stack_padding_mask, ..., : self.nnz_section_sizes[comm.rank]
        ]

    @data.setter
    def data(self, value: NDArray) -> None:
        """Sets the local slice of the data."""
        if self.distribution_state == "stack":
            self._data[
                : self.stack_section_sizes[comm.rank],
                ...,
                : sum(self.nnz_section_sizes),
            ] = value
        else:
            self._data[
                self._stack_padding_mask, ..., : self.nnz_section_sizes[comm.rank]
            ] = value

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
    def _get_items(self, stack_index: tuple, rows: NDArray, cols: NDArray) -> NDArray:
        """Gets the requested items from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the __getitem__
        method. The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        rows : NDArray
            The row indices of the items.
        cols : NDArray
            The column indices of the items.

        Returns
        -------
        items : NDArray
            The requested items.

        """
        ...

    @abstractmethod
    def _set_items(
        self, stack_index: tuple, rows: NDArray, cols: NDArray, values: NDArray
    ) -> None:
        """Sets the requested items in the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the __setitem__
        method. The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        rows : NDArray
            The row indices of the items.
        cols : NDArray
            The column indices of the items.
        values : NDArray
            The values to set.

        """
        ...

    @abstractmethod
    def _set_block(
        self, stack_index: tuple, row: int, col: int, block: NDArray
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        block : NDArray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        ...

    @abstractmethod
    def _get_block(self, stack_index: tuple, row: int, col: int) -> NDArray | tuple:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index in the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.

        Returns
        -------
        block : NDArray | tuple
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`
            if `return_dense` is True. Otherwise, it is a sparse
            representation of the block.

        """
        ...

    @abstractmethod
    def _check_commensurable(self, other: "DSBSparse") -> None:
        """Checks if two DSBSparse matrices are commensurable."""
        ...

    def __iadd__(self, other: "DSBSparse | sparse.spmatrix") -> "DSBSparse":
        """In-place addition of two DSBSparse matrices."""
        if sparse.issparse(other):
            csr = other.tocsr()
            self.data[:] += csr[*self.spy()]
            return self

        self._check_commensurable(other)
        self.data[:] += other.data[:]
        return self

    def __isub__(self, other: "DSBSparse | sparse.spmatrix") -> "DSBSparse":
        """In-place subtraction of two DSBSparse matrices."""
        if sparse.issparse(other):
            csr = other.tocsr()
            self.data[:] -= csr[*self.spy()]
            return self

        self._check_commensurable(other)
        self.data[:] -= other.data[:]
        return self

    def __imul__(self, other: "DSBSparse") -> "DSBSparse":
        """In-place multiplication of two DSBSparse matrices."""
        self._check_commensurable(other)
        self.data[:] *= other.data[:]
        return self

    @abstractmethod
    def __neg__(self) -> "DSBSparse":
        """Negation of the data."""
        ...

    @abstractmethod
    def __matmul__(self, other: "DSBSparse") -> "DSBSparse":
        """Matrix multiplication of two DSBSparse matrices."""
        ...

    def block_diagonal(self, offset: int = 0) -> list[NDArray]:
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

    def diagonal(self) -> NDArray:
        """Returns the diagonal elements of the matrix.

        This temporarily sets the return_dense state to True.

        Returns
        -------
        diagonal : NDArray
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
        return xp.concatenate(diagonals, axis=-1)

    def _dtranspose(
        self, block_axis: int, concatenate_axis: int, discard: bool = False
    ) -> None:
        """Performs the distributed transposition of the data.

        This is a helper method that performs the distributed transposition
        depending on the current distribution state.

        Parameters
        ----------
        block_axis : int
            The axis along which the blocks view is created.
        concatenate_axis : int
            The axis along which the received blocks are concatenated.
        discard : bool, optional
            Whether to perform a "fake" transposition. Default is False.

        """
        # old_shape = self._data.shape
        # new_shape = (
        #     old_shape[0] // comm.size,
        #     *old_shape[1:-1],
        #     old_shape[-1] * comm.size,
        # )

        self._data = _block_view(self._data, axis=block_axis)
        if discard:
            self._data = xp.concatenate(self._data, axis=concatenate_axis)
            self._data[:] = 0.0
            return

        # We need to make sure that the block-view is memory-contiguous.
        # This does nothing if the data is already contiguous.
        self._data = xp.ascontiguousarray(self._data)

        synchronize_current_stream()
        if xp.__name__ == "numpy" or GPU_AWARE_MPI:
            comm.Alltoall(MPI.IN_PLACE, self._data)
        else:
            _data_host = get_host(self._data)
            comm.Alltoall(MPI.IN_PLACE, _data_host)
            self._data = xp.array(_data_host)

        self._data = xp.concatenate(self._data, axis=concatenate_axis)

        # NOTE: There are a few things commented out here, since there
        # may be an alternative way to do the correct reshaping after
        # the Alltoall communication. The concatenatation needs to be
        # checked, as it may copy some data.

        # self._data = np.moveaxis(self._data, concatenate_axis, -2).reshape(new_shape)

    def dtranspose(self, discard: bool = False) -> None:
        """Performs a distributed transposition of the datastructure.

        This is done by reshaping the local data, then performing an
        in-place Alltoall communication, and finally reshaping the data
        back to the correct new shape.

        The local reshaping of the data cannot be done entirely
        in-place. This can lead to pronounced memory peaks if all ranks
        start reshaping concurrently, which can be mitigated by using
        more ranks and by not forcing a synchronization barrier right
        before calling `dtranspose`.

        Parameters
        ----------
        discard : bool, optional
            Whether to perform a "fake" transposition. Default is False.
            This is useful if you want to get the correct data shape
            after a transposition, but do not want to perform the actual
            all-to-all communication.

        """
        if self.distribution_state == "stack":
            self._dtranspose(block_axis=-1, concatenate_axis=0, discard=discard)
            self.distribution_state = "nnz"
        else:
            self._dtranspose(block_axis=0, concatenate_axis=-1, discard=discard)
            self.distribution_state = "stack"

    @abstractmethod
    def spy(self) -> tuple[NDArray, NDArray]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Returns
        -------
        rows : NDArray
            Row indices of the non-zero elements.
        cols : NDArray
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

    def to_dense(self) -> NDArray:
        """Converts the local data to a dense array.

        This is dumb, unless used for testing and debugging.

        Returns
        -------
        arr : NDArray
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
            ] = self._get_block((Ellipsis,), i, j)

        self.return_dense = original_return_dense

        return arr

    @classmethod
    @abstractmethod
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned: bool = False,
    ) -> "DSBSparse":
        """Creates a new DSBSparse matrix from a scipy.sparse array.

        Parameters
        ----------
        arr : sparse.spmatrix
            The sparse array to convert.
        block_sizes : NDArray
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


class _DStackIndexer:
    """A utility class to locate substacks in the distributed stack.

    Parameters
    ----------
    dsbsparse : DSBSparse
        The underlying datastructure

    """

    def __init__(self, dsbsparse: DSBSparse) -> None:
        """Initializes the stack indexer."""
        self._dsbsparse = dsbsparse

    def __getitem__(self, index: tuple) -> "_DStackView":
        """Gets a substack view."""
        return _DStackView(self._dsbsparse, index)


class _DStackView:
    """A utility class to create substack views.

    Parameters
    ----------
    dsbsparse : DSBSparse
        The underlying datastructure.
    stack_index : tuple
        The index of the substack to address.

    """

    def __init__(self, dsbsparse: DSBSparse, stack_index: tuple) -> None:
        """Initializes the stack indexer."""
        self._dsbsparse = dsbsparse
        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)
        stack_index = self._replace_ellipsis(stack_index)
        self._stack_index = stack_index

    def _replace_ellipsis(self, stack_index: tuple) -> tuple:
        """Replaces ellipsis with the correct number of slices.

        Note
        ----
        This replacement of ellipsis is nicked from
        https://github.com/dask/dask/blob/main/dask/array/slicing.py

        See the license at
        https://github.com/dask/dask/blob/main/LICENSE.txt

        Parameters
        ----------
        stack_index : tuple
            The stack index to replace the ellipsis in.

        Returns
        -------
        stack_index : tuple
            The stack index with the ellipsis replaced.

        """
        is_ellipsis = [i for i, ind in enumerate(stack_index) if ind is Ellipsis]
        if is_ellipsis:
            if len(is_ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis ('...')")

            loc = is_ellipsis[0]
            extra_dimensions = (self._dsbsparse.data.ndim - 1) - (
                len(stack_index) - sum(i is None for i in stack_index) - 1
            )
            stack_index = (
                stack_index[:loc]
                + (slice(None, None, None),) * extra_dimensions
                + stack_index[loc + 1 :]
            )
        return stack_index

    def __getitem__(self, index: tuple[ArrayLike, ArrayLike]) -> NDArray:
        """Gets the requested data from the substack."""
        rows, cols = self._dsbsparse._normalize_index(index)
        return self._dsbsparse._get_items(self._stack_index, rows, cols)

    def __setitem__(self, index: tuple[ArrayLike, ArrayLike], values: NDArray) -> None:
        """Sets the requested data in the substack."""
        rows, cols = self._dsbsparse._normalize_index(index)
        self._dsbsparse._set_items(self._stack_index, rows, cols, values)

    @property
    def blocks(self) -> "_DSBlockIndexer":
        """Returns a block indexer on the substack."""
        return _DSBlockIndexer(self._dsbsparse, self._stack_index)


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
    stack_index : tuple, optional
        The stack index to slice the blocks from. Default is Ellipsis,
        i.e. we return the whole stack of blocks.

    """

    def __init__(self, dsbsparse: DSBSparse, stack_index: tuple = (Ellipsis,)) -> None:
        """Initializes the block indexer."""
        self._dsbsparse = dsbsparse
        self._num_blocks = dsbsparse.num_blocks
        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)
        self._stack_index = stack_index

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self._dsbsparse.num_blocks + row if row < 0 else row
        col = self._dsbsparse.num_blocks + col if col < 0 else col
        if not (0 <= row < self._num_blocks and 0 <= col < self._num_blocks):
            raise IndexError("Block index out of bounds.")

        return row, col

    def _normalize_index(self, index: tuple) -> tuple:
        """Normalizes the block index."""
        if self._dsbsparse.distribution_state != "stack":
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

    def __getitem__(self, index: tuple) -> NDArray | tuple:
        """Gets the requested block from the data structure."""
        row, col = self._normalize_index(index)
        return self._dsbsparse._get_block(self._stack_index, row, col)

    def __setitem__(self, index: tuple, block: NDArray) -> None:
        """Sets the requested block in the data structure."""
        row, col = self._normalize_index(index)
        self._dsbsparse._set_block(self._stack_index, row, col, block)
