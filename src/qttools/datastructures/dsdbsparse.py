# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the abstract base class for distributed block-accessible sparse matrices."""

import numbers
import warnings
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

from qttools import ArrayLike, NDArray, sparse, xp
from qttools.comm import comm
from qttools.utils.gpu_utils import free_mempool, synchronize_device
from qttools.utils.mpi_utils import get_section_sizes

symmetry_ops = {
    "symmetric": lambda x: x,
    "hermitian": xp.conj,
    "skew-symmetric": lambda x: -x,
    "skew-hermitian": lambda x: -xp.conj(x),
}


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


class BlockConfig:
    """Configuration of block-sizes and block-slices for a DSDBSparse matrix.

    Parameters
    ----------
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    block_offsets : NDArray
        The block offsets of the block-sparse matrix.
    index_type : xp.int32 | xp.int64
        The index type to use for the sparse matrix.
        This is relevant for the low level kernels to avoid
        unnecessary type conversions.
    rowptr_map : dict, optional
        A mapping from block-coordinates to row-pointers. Default is
        None.
    block_slice_cache : dict, optional
        A cache for the block slices. Default is None.

    """

    def __init__(
        self,
        block_sizes: NDArray,
        block_offsets: NDArray,
        index_type: xp.int32 | xp.int64,
        rowptr_map: dict | None = None,
        block_slice_cache: dict | None = None,
    ):
        """Initializes the block config."""
        self.block_sizes = block_sizes.astype(index_type)
        self.block_offsets = block_offsets.astype(index_type)
        self.rowptr_map = rowptr_map or {}
        self.block_slice_cache = block_slice_cache or {}


class DSDBSparse(ABC):
    """Base class for Distributed Stack of Distributed Block-accessible
    Sparse matrices.

    Parameters
    ----------
    dtype : xp.dtype[xp.generic]
        The data type of the matrix.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    nnz : int
        The number of non-zero elements in the sparse matrix.
    local_stack_shape : tuple or int
        The local shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    global_stack_shape : tuple or int
        The global shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    index_type : xp.int32 | xp.int64
        The index type to use for the sparse matrix. This is relevant
        for the low level kernels to avoid unnecessary type conversions.
    symmetry : str | None, optional
        The symmetry of the matrix. This can be "symmetric",
        "hermitian", "skew-symmetric", "skew-hermitian", or None.
        Default is None.

    """

    def __init__(
        self,
        dtype: xp.dtype[xp.generic],
        block_sizes: NDArray,
        nnz: int,
        local_stack_shape: tuple | int,
        global_stack_shape: tuple | int,
        index_type: xp.int32 | xp.int64,
        symmetry: str | None = None,
    ):
        """Initializes a DSBDSparse matrix."""

        # Set the block and stack communicators.
        if comm.block is None or comm.stack is None:
            raise ValueError(
                "Block and stack communicators must be initialized via "
                "the `setup_context` method."
            )

        if symmetry not in list(symmetry_ops.keys()) + [None]:
            raise ValueError(
                f"Invalid symmetry '{symmetry}'."
                f"Must be one of {list(symmetry_ops.keys()) + [None]}."
            )

        # Type of the data
        self.dtype = dtype
        # Type of the indices
        self.index_type = index_type
        self.symmetry = symmetry
        # Per default, we have the data is distributed in stack format.
        self.distribution_state = "stack"
        self._data = None

        if not isinstance(local_stack_shape, type(global_stack_shape)):
            raise ValueError(
                "local_stack_shape and global_stack_shape must be of the same type."
            )

        if isinstance(global_stack_shape, int):
            global_stack_shape = (global_stack_shape,)

        if isinstance(local_stack_shape, int):
            local_stack_shape = (local_stack_shape,)

        if global_stack_shape[0] < comm.stack.size:
            raise ValueError(
                f"Number of MPI ranks in stack communicator {comm.stack.size} "
                f"exceeds stack shape {global_stack_shape[0]}."
            )

        self.local_stack_shape = local_stack_shape
        self.global_stack_shape = global_stack_shape

        # This is the shape of this matrix in the comm.stack.
        # NOTE: This is the local shape of the stack.
        self.shape = self.local_stack_shape + (
            int(sum(block_sizes)),
            int(sum(block_sizes)),
        )

        # --- Things concerning stack distribution ---------------------

        # Determine how the data is distributed across the stack.
        stack_section_sizes, total_stack_size = get_section_sizes(
            global_stack_shape[0], comm.stack.size, strategy="balanced"
        )
        self.stack_section_sizes = stack_section_sizes
        self.total_stack_size = total_stack_size

        self.stack_section_offsets = xp.hstack(
            ([0], np.cumsum(stack_section_sizes))
        ).astype(index_type)

        # --- Things concerning nnz distribution ---------------------

        # Determine how the data is distributed across the nnz.
        nnz_section_sizes, total_nnz_size = get_section_sizes(
            nnz, comm.stack.size, strategy="greedy"
        )
        self.nnz_section_sizes = nnz_section_sizes
        self.total_nnz_size = total_nnz_size

        self.nnz_section_offsets = xp.hstack(
            ([0], np.cumsum(nnz_section_sizes))
        ).astype(index_type)

        # --- Things concerning both distributions ---------------------

        self.data_slice_stack = (
            slice(None, int(self.stack_section_sizes[comm.stack.rank])),
            ...,
            slice(None, int(self.nnz_section_offsets[-1])),
        )
        self.data_slice_nnz = (
            slice(None, int(self.global_stack_shape[0])),
            ...,
            slice(None, int(self.nnz_section_sizes[comm.stack.rank])),
        )

        # For the weird padding convention we use, we need to keep track
        # of this padding mask.
        # NOTE: We should maybe consistently use the greedy strategy for
        # the stack distribution as well.
        self._stack_padding_mask = xp.zeros(total_stack_size, dtype=bool)
        for i, size in enumerate(stack_section_sizes):
            offset = i * max(stack_section_sizes)
            self._stack_padding_mask[offset : offset + size] = True

        # --- Things concerning block distribution ---------------------

        # Block-sizes is an settable property.
        self.num_blocks = len(block_sizes)

        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

        block_section_sizes, __ = get_section_sizes(self.num_blocks, comm.block.size)
        self.block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        # We need to know our local block sizes and those of all
        # subsequent ranks.
        self.num_local_blocks = block_section_sizes[comm.block.rank]
        self.local_block_sizes = block_sizes[
            self.block_section_offsets[comm.block.rank] :
        ]
        self.local_block_offsets = np.hstack(([0], np.cumsum(self.local_block_sizes)))

        self.global_block_offset = sum(
            block_sizes[: self.block_section_offsets[comm.block.rank]]
        )
        self.num_local_diag = sum(
            block_sizes[
                self.block_section_offsets[
                    comm.block.rank
                ] : self.block_section_offsets[comm.block.rank + 1]
            ]
        )

        # --- Things concerning block indexing and slicing --------------

        self._block_config: dict[int, BlockConfig] = {}
        # This is a cache for the block change. It contains
        # the mapping from one block configuration to another.
        # NOTE: Currently, it is assumed that each configuration
        # is uniquely identified by the number of blocks. This is
        # not necessarily true, but it is a reasonable assumption for now.
        self._block_change_cache: dict[(int, int), NDArray] = {}
        self._add_block_config(self.num_blocks, block_sizes, block_offsets)

        self._block_indexer = _DSDBlockIndexer(self)
        self._stack_view = _DStackView(
            dsdbsparse=self, stack_shape=self.local_stack_shape, stack_index=(...,)
        )

        # Diagonal indices.
        self._diag_inds = None
        self._diag_value_inds = None
        self._diag_inds_nnz = None
        self._diag_value_inds_nnz = None
        self._diag_cache: dict[int, NDArray] = {}

    def _add_block_config(
        self,
        num_blocks: int,
        block_sizes: NDArray,
        block_offsets: NDArray,
        block_slice_cache: dict = None,
    ):
        """Adds a block configuration to the block config cache.

        The assumption is that the number of blocks uniquely identifies
        the block configuration.

        Parameters
        ----------
        num_blocks : int
            The number of blocks in the block configuration.
        block_sizes : NDArray
            The size of each block in the block configuration.
        block_offsets : NDArray
            The block offsets of the block configuration.
        block_slice_cache : dict, optional
            A cache for the block slices. Default is None.

        """
        self._block_config[num_blocks] = BlockConfig(
            block_sizes,
            block_offsets,
            self.index_type,
            block_slice_cache=block_slice_cache,
        )

    @property
    def block_sizes(self) -> ArrayLike:
        """Returns the global block sizes."""
        return self._block_config[self.num_blocks].block_sizes

    @block_sizes.setter
    @abstractmethod
    def block_sizes(self, block_sizes: ArrayLike) -> None:
        """Sets the global block sizes."""
        ...

    @property
    def block_offsets(self) -> ArrayLike:
        """Returns the block sizes."""
        return self._block_config[self.num_blocks].block_offsets

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

    @property
    def blocks(self) -> "_DSDBlockIndexer":
        """Returns a block indexer."""
        return self._block_indexer

    @property
    def stack(self) -> "_DStackView":
        """Returns a stack indexer."""
        return self._stack_view

    @property
    def data(self) -> NDArray:
        """Returns the local slice of the data, masking the padding."""
        if self.distribution_state == "stack":
            return self._data[self.data_slice_stack]
        return self._data[self.data_slice_nnz]

    @data.setter
    def data(self, value: NDArray) -> None:
        """Sets the local slice of the data."""
        if self.distribution_state == "stack":
            self._data[self.data_slice_stack] = value
        else:
            self._data[self.data_slice_nnz] = value

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"block_sizes={self.block_sizes}, "
            f"global_stack_shape={self.global_stack_shape}, "
            f'distribution_state="{self.distribution_state}", '
            f"stack_comm_rank={comm.stack.rank}, "
            f"block_comm_rank={comm.block.rank})"
        )

    @abstractmethod
    def _set_block(
        self,
        stack_index: tuple,
        row: int,
        col: int,
        block: NDArray,
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Note
        ----
        The input block is not tested for symmetry even if the matrix is
        symmetric.

        Parameters
        ----------
        stack_index : tuple
            The index of the stack.
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
    def _get_block(
        self,
        stack_index: tuple,
        row: int,
        col: int,
    ) -> NDArray:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        stack_index : tuple
            The index of the stack.
        row : int
            Row index of the block.
        col : int
            Column index of the block.

        Returns
        -------
        block : NDArray
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.

        """
        ...

    def diagonal(self, stack_index: tuple = (Ellipsis,)) -> NDArray:
        """Returns or sets the diagonal elements of the matrix.

        Note
        ----
        In the block distributed case, this returns the local
        diagonal elements.

        Parameters
        ----------
        stack_index : tuple, optional
            The index in the stack. Default is (Ellipsis,).

        Returns
        -------
        diagonal : NDArray
            The diagonal elements of the matrix.

        """
        if self._diag_inds is None or self._diag_value_inds is None:
            raise NotImplementedError("Diagonal not implemented.")

        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)

        # Getter
        data_stack = self.data[*stack_index]
        if self.distribution_state == "stack":
            local_diagonal = xp.zeros(
                (
                    data_stack.shape[:-1]
                    + (sum(self.local_block_sizes[: self.num_local_blocks]),)
                ),
                dtype=self.dtype,
            )
            local_diagonal[..., self._diag_value_inds] = data_stack[
                ..., self._diag_inds
            ]
            return local_diagonal
        else:
            if self._diag_inds_nnz is not None:
                return data_stack[..., self._diag_inds_nnz]
            return xp.empty((data_stack.shape[:-1] + (0,)))

    def fill_diagonal(self, val: NDArray, stack_index: tuple = (Ellipsis,)) -> NDArray:
        """Returns or sets the diagonal elements of the matrix.

        Parameters
        ----------
        val : NDArray
            The value(s) to set along the diagonal.
        stack_index : tuple, optional
            The index in the stack. Default is (Ellipsis,).

        Returns
        -------
        diagonal : NDArray
            The diagonal elements of the matrix.

        """
        if self._diag_inds is None or self._diag_value_inds is None:
            raise NotImplementedError("Diagonal not implemented.")

        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)

        # Setter
        val = xp.asarray(val)
        if self.distribution_state == "stack":
            if val.ndim == 0:
                self.data[*stack_index][..., self._diag_inds] = val
            else:
                self.data[*stack_index][..., self._diag_inds] = val[
                    ..., self._diag_value_inds
                ]
            return

        if self._diag_inds_nnz is not None:
            if val.ndim == 0:
                self.data[*stack_index][..., self._diag_inds_nnz] = val
            else:
                self.data[*stack_index][..., self._diag_inds_nnz] = val[
                    ..., self._diag_value_inds_nnz
                ]
        return

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

        if discard:
            self._data = _block_view(
                self._data, axis=block_axis, num_blocks=comm.stack.size
            )
            self._data = xp.concatenate(self._data, axis=concatenate_axis)
            self._data[:] = 0.0
            return

        # We need to make sure that the block-view is memory-contiguous.
        # This does nothing if the data is already contiguous.
        self._data = _block_view(
            self._data, axis=block_axis, num_blocks=comm.stack.size
        )
        self._data = xp.ascontiguousarray(self._data)
        synchronize_device()

        receive_buffer = xp.empty_like(self._data)
        comm.stack.all_to_all(self._data, receive_buffer)
        self._data = receive_buffer

        self._data = xp.concatenate(self._data, axis=concatenate_axis)
        synchronize_device()

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
            # Shuffle data to make it contiguous in memory
            _data = xp.zeros_like(self._data)
            _data[: self.global_stack_shape[0]] = self._data[self._stack_padding_mask]
            self._data = _data

        else:
            # Undo the shuffle
            _data = xp.zeros_like(self._data)
            _data[self._stack_padding_mask] = self._data[: self.global_stack_shape[0]]
            self._data = _data

            self._dtranspose(block_axis=0, concatenate_axis=-1, discard=discard)
            self.distribution_state = "stack"

    @abstractmethod
    def spy(self) -> tuple[NDArray, NDArray]:
        """Returns the row and column indices of the non-zero elements.

        This is essentially the same as converting the sparsity pattern
        to coordinate format. The returned sparsity pattern is not
        sorted.

        Note
        ----
        In the block distributed case, this returns the local
        sparsity pattern including the offset.

        Returns
        -------
        rows : NDArray
            Row indices of the non-zero elements.
        cols : NDArray
            Column indices of the non-zero elements.

        """
        ...

    @abstractmethod
    def symmetrize(self, symmetry: str) -> None:
        """Symmetrizes the matrix with a given symmetry.

        Note
        ----
        This assumes that the matrix's sparsity pattern is symmetric.

        Parameters
        ----------
        symmetry : str
            The symmetry to enforce. This can be "symmetric",
            "hermitian", "skew-symmetric", or "skew-hermitian".

        """
        ...

    @abstractmethod
    def to_dense(self) -> NDArray:
        """Converts the local data to a dense array.

        This is dumb, unless used for testing and debugging.

        Returns
        -------
        arr : NDArray
            The dense array of shape `(*local_stack_shape, *shape)`.

        """
        ...

    def free_data(self) -> None:
        """Frees the local data."""
        self._data = None
        free_mempool()

    def allocate_data(self, stack_size: int | None = None) -> None:
        """Allocates the local data.

        Note
        ----
        This should not be called with a non-None stack size if the
        data will be dtransposed.
        The data is not zeroed. It is the user responsibility to
        ensure that the data is initialized correctly.

        Parameters
        ----------
        stack_size : int, optional
            The size of the stack dimension to allocate. If None, the
            full stack size is used. Default is None.

        """
        free_mempool()

        # NOTE: Dangerous
        # but we assume there is no padding
        # and no all-to-all will be performed
        # TODO: We should have a non distributed
        # version without padding and cheaper initialization
        # As with the block size, it should have a stack size
        # setter which updates attributes
        if stack_size is not None:
            self.data_slice_stack = (
                slice(None, int(stack_size)),
                ...,
                slice(None, int(self.nnz_section_offsets[-1])),
            )
            self.shape = (int(stack_size), *self.shape[1:])
            self.local_stack_shape = self.shape[:-2]

        if stack_size == 0:
            warnings.warn(
                "Stack size of 0 is not valid."
                "Allocating data with a unit stack section size."
            )
            stack_size = 1

        if stack_size is None:
            stack_size = int(max(self.stack_section_sizes))
        # NOTE: Edge case for when the stack size is 0
        if stack_size == 0:
            warnings.warn(
                "Stack size of 0 is not valid."
                "Allocating data with a unit stack section size."
            )
            stack_size = 1

        if self._data is None:
            self._data = xp.empty(
                (
                    stack_size,
                    *self.global_stack_shape[1:],
                    self.total_nnz_size,
                ),
                dtype=self.dtype,
            )

    @classmethod
    @abstractmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] = xp.complex128,
        allocate: bool = True,
    ) -> "DSDBSparse":
        """Creates a new DSDBSparse matrix from a scipy.sparse array.

        This essentially distributed the matrix across the stack and
        block communicators.

        Parameters
        ----------
        sparray : sparse.spmatrix
            The sparse matrix from which to use the sparsity pattern.
        block_sizes : NDArray
            The block sizes of the block-sparse matrix.
        global_stack_shape : tuple
            The global shape of the stack.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian", or None.
            Default is None.
        dtype : xp.dtype, optional
            The data type of the matrix. Default is `xp.complex128`.
        allocate : bool, optional
            Whether to allocate the data of the resulting matrix.
            Default is True.

        Returns
        -------
        DSDBSparse
            The new DSDBSparse matrix.

        """
        ...

    @classmethod
    @abstractmethod
    def empty_like(cls, dsdbsparse: "DSDBSparse") -> "DSDBSparse":
        """Creates a new DSDBSparse matrix with the same shape and
        dtype.

        Note
        ----
        There is no data allocated in the new matrix. The sparsity
        pattern is the same as the original matrix.

        Parameters
        ----------
        dsdbsparse : DSDBSparse
            The matrix to copy the shape and dtype from.

        Returns
        -------
        DSDBSparse
            The new DSDBSparse matrix.

        """
        ...


def _replace_ellipsis(stack_index: tuple, ndim: int) -> tuple:
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
    ndim : int
        The number of dimensions of the data.

    Returns
    -------
    stack_index : tuple
        The stack index with the ellipsis replaced.

    """

    if not isinstance(stack_index, tuple):
        stack_index = (stack_index,)

    is_ellipsis = [i for i, ind in enumerate(stack_index) if ind is Ellipsis]
    if is_ellipsis:
        if len(is_ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        loc = is_ellipsis[0]
        extra_dimensions = (ndim - 1) - (
            len(stack_index) - sum(i is None for i in stack_index) - 1
        )
        stack_index = (
            stack_index[:loc]
            + (slice(None, None, None),) * extra_dimensions
            + stack_index[loc + 1 :]
        )
    return stack_index


def _compose_single(lhs: int | slice, rhs: int | slice, length: int) -> int | slice:
    """Composes two unidimensional indices or slices.

    Example:
        length = 30  # dimension of length 30
        lhs = slice(2, 27, 2)  # selects indices [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        rhs = slice(1, 11, 3)  # selects indices [1, 4, 7, 10] from the previous selection,
                               # i.e., [4, 10, 16, 22] from the original dimension
        result = _compose_single(lhs, rhs, 30)
        # result is slice(4, 22, 6), which selects indices [4, 10, 16, 22] from the original dimension.

    Parameters
    ----------
    lhs : int | slice
        The first index or slice.
    rhs : int | slice
        The second index or slice.
    length : int
        The length of the dimension being indexed.

    Returns
    -------
    int | slice
        The composed index or slice.
    """
    out = range(length)[lhs][rhs]
    return out if isinstance(out, int) else slice(out.start, out.stop, out.step)


def _compose(
    shape: tuple[int, ...],
    first: tuple[int, ...] | int | slice,
    second: tuple[int, ...] | int | slice,
) -> tuple[int | slice, ...]:
    """Composes two multidimensional indices or slices.

    Parameters
    ----------
    shape : tuple
        The shape of the array being indexed.
    first : tuple | int | slice
        The first index or slice.
    second : tuple | int | slice
        The second index or slice.

    Returns
    -------
    tuple
        The composed index or slice.
    """

    def ensure_tuple(ndslice):
        return ndslice if isinstance(ndslice, tuple) else (ndslice,)

    # Ensure both are tuples for easier processing
    first = ensure_tuple(first)
    second = ensure_tuple(second)

    # Initialize output with first index/slice and fill the rest with full slices
    out = list(first) + [slice(None)] * (len(shape) - len(first))

    # We only need to compose the slice dimensions (not the indices).
    # NOTE: It is implied that for any dimensions excluded here, the corresponding index in `second` is 0.
    remaining_dims = [i for i, s in enumerate(out) if isinstance(s, slice)]

    for i, rhs in zip(remaining_dims, second):
        out[i] = _compose_single(out[i], rhs, length=shape[i])
    return tuple(out)


def _local_stack_shape(
    stack_index: tuple,
    stack_shape: tuple,
) -> tuple:
    """Computes the shape of the addressed substack.

    Parameters
    ----------
    stack_index : tuple
        The index of the substack to address.
    stack_shape : tuple
        The shape of the local stack to address.

    """
    sizes = []
    for i, s in enumerate(stack_index):
        if isinstance(s, slice):
            if s.step not in (1, None):
                raise NotImplementedError(
                    f"Non-unit strides are not supported, step: {s.step}."
                )
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else stack_shape[i]
            sizes.append(stop - start)
        elif isinstance(s, (int, numbers.Integral)):
            sizes.append(1)
        else:
            raise IndexError(
                f"Expected slice or int in stack index but got {type(s)} ({s=})."
            )
    return tuple(sizes) + stack_shape[len(stack_index) :]


class _StackView(ABC):
    """Abstract base class for stack views.

    Supports further indexing via `[...]` or `.stack[...]`, which
    compose with the current index and return a new `_StackView`.

    Note
    ----
    Only continuous views are supported.

    Warning
    -------
    If the underlying datastructure is modified, the view may become
    invalid.

    Parameters
    ----------
    local_stack_shape : tuple
        The shape of the stack to index into.
    stack_index : tuple
        The base index of the substack.

    """

    def __init__(
        self,
        stack_shape: tuple,
        stack_index: tuple,
    ) -> None:
        self._stack_shape = stack_shape
        self._stack_index = _replace_ellipsis(stack_index, len(stack_shape))

    def __getitem__(self, index: tuple) -> "_StackView":
        """Composes `index` with the current index and returns a new
        view of the same concrete type."""
        index = _replace_ellipsis(index, len(self._stack_shape))
        if self._stack_index is not None:
            index = _compose(self._stack_shape, self._stack_index, index)
        return self._reindexed(index)

    @abstractmethod
    def _reindexed(self, stack_index: tuple) -> "_StackView":
        """Returns a new view of the same concrete type, addressing
        `stack_index`."""
        ...

    @property
    def stack(self) -> "_StackView":
        """Returns self, so `.stack[...]` chains."""
        return self

    @cached_property
    def local_stack_shape(self) -> tuple:
        """Returns the shape of the addressed substack."""
        return _local_stack_shape(self._stack_index, self._stack_shape)

    @cached_property
    def blocks(self) -> "_BlockIndexer":
        """Returns a (lazily built, cached) block indexer for the substack."""
        return self._make_block_indexer()

    @abstractmethod
    def _make_block_indexer(self) -> "_BlockIndexer":
        """Constructs the block indexer for this view."""
        ...


class _DStackView(_StackView):
    """A view onto a (possibly full) substack of a `DSDBSparse` matrix.

    Supports further indexing via `[...]` or `.stack[...]`, which
    compose with the current index and return a new `_DStackView`.

    Note
    ----
    Only continuous views are supported.

    Warning
    -------
    If the underlying datastructure is modified, the view may become invalid.

    Parameters
    ----------
    dsdbsparse : DSDBSparse
        The underlying datastructure.
    local_stack_shape : tuple
        The shape of the stack to index into.
    stack_index : tuple
        The base index of the substack.

    """

    _DELEGATED = (
        "symmetry",
        "distribution_state",
        "dtype",
        "num_blocks",
        "block_sizes",
        "num_local_blocks",
    )

    def __init__(
        self,
        dsdbsparse: DSDBSparse,
        stack_shape: tuple,
        stack_index: tuple,
    ) -> None:
        super().__init__(stack_shape, stack_index)
        self._dsdbsparse = dsdbsparse

    def _reindexed(self, stack_index: tuple) -> "_DStackView":
        return _DStackView(self._dsdbsparse, self._stack_shape, stack_index)

    def __getattr__(self, name: str):
        if name in self._DELEGATED:
            return getattr(self._dsdbsparse, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    @property
    def data(self) -> NDArray:
        """Returns the local slice of the data, masking the padding."""
        return self._dsdbsparse.data[self._stack_index]

    @data.setter
    def data(self, value: NDArray) -> None:
        """Sets the local slice of the data."""
        self._dsdbsparse.data[self._stack_index] = value

    def _make_block_indexer(self) -> "_DSDBlockIndexer":
        """Constructs the block indexer for this view."""
        return _DSDBlockIndexer(
            dsdbsparse=self._dsdbsparse, stack_index=self._stack_index
        )

    @property
    def block_section_offsets(self) -> list[int]:
        """Returns the block section offsets."""
        return self._dsdbsparse.block_section_offsets


class _BlockIndexer(ABC):
    """Abstract base class for block indexers.

    Parameters
    ----------
    stack_index : tuple, optional
        The stack index to slice the blocks from. Default is Ellipsis,
        i.e. we return the whole stack of blocks.

    """

    def __init__(
        self,
        stack_index: tuple = (Ellipsis,),
    ) -> None:
        """Initializes the block indexer."""
        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)
        self._stack_index = stack_index

    def __getitem__(self, index: tuple) -> NDArray | tuple:
        """Gets the requested block from the data structure."""
        ...

    def __setitem__(self, index: tuple, block: NDArray) -> None:
        """Sets the requested block in the data structure."""
        ...


class _DSDBlockIndexer(_BlockIndexer):
    """A utility class to locate blocks in the distributed stack.

    This uses the `_get_block` and `_set_block` methods of the
    underlying DSDBSparse object to locate and set blocks in the stack.

    This is only intended to give blocks from the current rank in the
    block communicator.

    Parameters
    ----------
    dsdbsparse : DSDBSparse
        The underlying datastructure
    stack_index : tuple, optional
        The stack index to slice the blocks from. Default is Ellipsis,
        i.e. we return the whole stack of blocks.

    """

    def __init__(
        self,
        dsdbsparse: DSDBSparse,
        stack_index: tuple = (Ellipsis,),
    ) -> None:
        """Initializes the block indexer."""
        super().__init__(stack_index)
        self._dsdbsparse = dsdbsparse

    def _normalize_index(self, index: tuple) -> tuple:
        """Normalizes the block index."""
        if self._dsdbsparse.distribution_state != "stack":
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if len(index) != 2:
            raise IndexError("Exactly two block indices are required.")

        row, col = index
        if isinstance(row, slice) or isinstance(col, slice):
            raise NotImplementedError("Slicing is not supported.")

        if row < 0 or col < 0:
            raise IndexError("Negative block indices are not supported.")

        if row >= len(self._dsdbsparse.local_block_sizes) or col >= len(
            self._dsdbsparse.local_block_sizes
        ):
            raise IndexError("Block index out of bounds.")

        return row, col

    def __getitem__(self, index: tuple) -> NDArray | tuple:
        """Gets the requested block from the data structure."""
        row, col = self._normalize_index(index)
        return self._dsdbsparse._get_block(self._stack_index, row, col)

    def __setitem__(self, index: tuple, block: NDArray) -> None:
        """Sets the requested block in the data structure."""
        row, col = self._normalize_index(index)
        self._dsdbsparse._set_block(self._stack_index, row, col, block)
