# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import itertools
import time
from abc import ABC, abstractmethod
from typing import Callable

from mpi4py import MPI

from qttools import (
    NCCL_AVAILABLE,
    ArrayLike,
    NDArray,
    block_comm,
    global_comm,
    host_xp,
    nccl_stack_comm,
    sparse,
    stack_comm,
    xp,
)
from qttools.datastructures.dsbsparse import BlockConfig, _block_view
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import get_host, synchronize_device
from qttools.utils.mpi_utils import check_gpu_aware_mpi, get_section_sizes

profiler = Profiler()

GPU_AWARE_MPI = check_gpu_aware_mpi()


def _flatten_list(nested_lists: list[list]) -> list:
    """Flattens a list of lists.

    This should do the same as sum(l, start=[]) but is more explicit and
    apparently faster as well.

    Parameters
    ----------
    nested_lists : list[list]
        The list of lists to flatten.

    Returns
    -------
    list
        The flattened list.

    """
    return list(itertools.chain.from_iterable(nested_lists))


class DSDBSparse(ABC):
    """Base class for Distributed Stack of Distributed Block-accessible Sparse matrices.

    Parameters
    ----------
    local_data : NDArray
        The local slice of the data. This should be an array of shape
        `(*local_stack_shape, local_nnz)`. It is the caller's
        responsibility to ensure that the data is distributed correctly
        across the ranks.
    block_sizes : NDArray
        The size of each block in the sparse matrix.
    global_stack_shape : tuple or int
        The global shape of the stack. If this is an integer, it is
        interpreted as a one-dimensional stack.
    block_comm : MPI.Comm
        The communicator for the block distribution.
    stack_comm : MPI.Comm
        The communicator for the stack distribution.
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.

    """

    def __init__(
        self,
        local_data: NDArray,
        block_sizes: NDArray,
        global_stack_shape: tuple | int,
        return_dense: bool = True,
    ):
        """Initializes a DSBDSparse matrix."""

        # --- Things concerning stack distribution ---------------------

        if isinstance(global_stack_shape, int):
            global_stack_shape = (global_stack_shape,)

        if global_stack_shape[0] < stack_comm.size:
            raise ValueError(
                f"Number of MPI ranks in stack communicator {stack_comm.size} "
                f"exceeds stack shape {global_stack_shape}."
            )

        self.global_stack_shape = global_stack_shape

        # Set the block and stack communicators.
        if block_comm is None or stack_comm is None:
            raise ValueError(
                "Block and stack communicators must be initialized via "
                "the BLOCK_COMM_SIZE environment variable."
            )

        # Determine how the data is distributed across the stack.
        stack_section_sizes, total_stack_size = get_section_sizes(
            global_stack_shape[0], stack_comm.size, strategy="balanced"
        )
        self.stack_section_sizes = stack_section_sizes
        self.total_stack_size = total_stack_size

        nnz_section_sizes, total_nnz_size = get_section_sizes(
            local_data.shape[-1], stack_comm.size, strategy="greedy"
        )
        self.nnz_section_sizes = nnz_section_sizes
        self.nnz_section_offsets = xp.hstack(([0], host_xp.cumsum(nnz_section_sizes)))
        self.total_nnz_size = total_nnz_size

        # Per default, we have the data is distributed in stack format.
        self.distribution_state = "stack"

        # Pad local data with zeros to ensure that all ranks have the
        # same data size for the all-to-all communication.
        self._data = xp.zeros(
            (max(stack_section_sizes), *global_stack_shape[1:], total_nnz_size),
            dtype=local_data.dtype,
        )
        self._data[: local_data.shape[0], ..., : local_data.shape[-1]] = local_data

        # For the weird padding convention we use, we need to keep track
        # of this padding mask.
        # NOTE: We should maybe consistently use the greedy strategy for
        # the stack distribution as well.
        self._stack_padding_mask = xp.zeros(total_stack_size, dtype=bool)
        for i, size in enumerate(stack_section_sizes):
            offset = i * max(stack_section_sizes)
            self._stack_padding_mask[offset : offset + size] = True

        self.stack_shape = local_data.shape[:-1]
        self.local_nnz = local_data.shape[-1]
        # This is the shape of this matrix in the stack_comm.
        self.shape = self.stack_shape + (int(sum(block_sizes)), int(sum(block_sizes)))

        # --- Things concerning block distribution ---------------------
        # Block-sizes is an settable property.
        self.num_blocks = len(block_sizes)

        block_offsets = host_xp.hstack(([0], host_xp.cumsum(block_sizes)))

        block_section_sizes, __ = get_section_sizes(self.num_blocks, block_comm.size)
        self.block_section_offsets = host_xp.hstack(
            ([0], host_xp.cumsum(block_section_sizes))
        )

        # We need to know our local block sizes and those of all
        # subsequent ranks.
        self.num_local_blocks = block_section_sizes[block_comm.rank]
        self.local_block_sizes = block_sizes[
            self.block_section_offsets[block_comm.rank] :
        ]
        self.local_block_offsets = host_xp.hstack(
            ([0], host_xp.cumsum(self.local_block_sizes))
        )

        self.global_block_offset = sum(
            block_sizes[: self.block_section_offsets[block_comm.rank]]
        )

        self._block_config: dict[int, BlockConfig] = {}
        self._add_block_config(self.num_blocks, block_sizes, block_offsets)

        self.dtype = local_data.dtype
        self.return_dense = return_dense

        self._block_indexer = _DSDBlockIndexer(self)
        self._sparse_block_indexer = _DSDBlockIndexer(self, return_dense=False)
        self._stack_indexer = _DStackIndexer(self)

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
            block_sizes, block_offsets, block_slice_cache
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

    @profiler.profile(level="debug")
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
    def local_blocks(self) -> "_DSDBlockIndexer":
        """Returns a block indexer."""
        return self._block_indexer

    @property
    def sparse_local_blocks(self) -> "_DSDBlockIndexer":
        """Returns a block indexer."""
        return self._sparse_block_indexer

    @property
    def stack(self) -> "_DStackIndexer":
        """Returns a stack indexer."""
        return self._stack_indexer

    @property
    def data(self) -> NDArray:
        """Returns the local slice of the data, masking the padding."""
        if self.distribution_state == "stack":
            return self._data[
                : self.stack_section_sizes[stack_comm.rank],
                ...,
                : sum(self.nnz_section_sizes),
            ]
        return self._data[
            self._stack_padding_mask,
            ...,
            : self.nnz_section_sizes[stack_comm.rank],
        ]

    @data.setter
    def data(self, value: NDArray) -> None:
        """Sets the local slice of the data."""
        if self.distribution_state == "stack":
            self._data[
                : self.stack_section_sizes[stack_comm.rank],
                ...,
                : sum(self.nnz_section_sizes),
            ] = value
        else:
            self._data[
                self._stack_padding_mask,
                ...,
                : self.nnz_section_sizes[stack_comm.rank],
            ] = value

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"block_sizes={self.block_sizes}, "
            f"global_stack_shape={self.global_stack_shape}, "
            f'distribution_state="{self.distribution_state}", '
            f"stack_comm_rank={stack_comm.rank}, "
            f"block_comm_rank={block_comm.rank})"
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
        self,
        arg: tuple | NDArray,
        row: int,
        col: int,
        block: NDArray,
        is_index: bool = True,
    ) -> None:
        """Sets a block throughout the stack in the data structure.

        The index is assumed to already be renormalized.

        Parameters
        ----------
        arg : tuple | NDArray
            The index of the stack or a view of the data stack. The
            is_index flag indicates whether the argument is an index or
            a view.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        block : NDArray
            The block to set. This must be an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])`.
        is_index : bool, optional
            Whether the argument is an index or a view. Default is True.

        """
        ...

    @abstractmethod
    def _get_block(
        self, arg: tuple | NDArray, row: int, col: int, is_index: bool = True
    ) -> NDArray | tuple:
        """Gets a block from the data structure.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        arg : tuple | NDArray
            The index of the stack or a view of the data stack. The
            is_index flag indicates whether the argument is an index or
            a view.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        is_index : bool, optional
            Whether the argument is an index or a view. Default is True.

        Returns
        -------
        block : NDArray | tuple[NDArray, NDArray, NDArray]
            The block at the requested index. This is an array of shape
            `(*local_stack_shape, block_sizes[row], block_sizes[col])` if
            `return_dense` is True, otherwise it is a tuple of arrays
            `(rows, cols, data)`.

        """
        ...

    @abstractmethod
    def _get_sparse_block(
        self,
        arg: tuple | NDArray,
        row: int,
        col: int,
        is_index: bool = True,
    ) -> sparse.spmatrix | tuple:
        """Gets a block from the data structure in a sparse representation.

        This is supposed to be a low-level method that does not perform
        any checks on the input. These are handled by the block indexer.
        The index is assumed to already be renormalized.

        Parameters
        ----------
        arg : tuple | NDArray
            The index of the stack or a view of the data stack. The
            is_index flag indicates whether the argument is an index or
            a view.
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        is_index : bool, optional
            Whether the argument is an index or a view. Default is True.

        Returns
        -------
        block : spmatrix | tuple
            The block at the requested index. It is a sparse
            representation of the block.

        """
        ...

    @abstractmethod
    def _check_commensurable(self, other: "DSDBSparse") -> None:
        """Checks if two DSDBSparse matrices are commensurable."""
        ...

    def __imul__(self, other: "DSDBSparse") -> "DSDBSparse":
        """In-place multiplication of two DSDBSparse matrices."""
        self._check_commensurable(other)
        self._data *= other._data
        return self

    @abstractmethod
    def __iadd__(self, other: "DSDBSparse | sparse.spmatrix") -> "DSDBSparse":
        """In-place addition of two DSDBSparse matrices."""
        ...

    @abstractmethod
    def __isub__(self, other: "DSDBSparse | sparse.spmatrix") -> "DSDBSparse":
        """In-place subtraction of two DSDBSparse matrices."""
        ...

    @abstractmethod
    def __neg__(self) -> "DSDBSparse":
        """Negation of the data."""
        ...

    @profiler.profile(level="api")
    def block_diagonal(self, offset: int = 0) -> list[NDArray]:
        """Returns the block diagonal of the matrix.

        Note that this will cause communication in the
        block-communicator.

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
        local_blocks = []
        stack_view = self.stack[...]
        if block_comm.rank != block_comm.size - 1:
            # Only the last rank in the block-communicator needs to make
            # sure that the offset does not exceed the number of local
            # blocks.
            num_blocks = self.num_local_blocks
        else:
            num_blocks = self.num_local_blocks - abs(offset)

        col_offset = offset if offset > 0 else 0
        row_offset = abs(offset) if offset < 0 else 0

        for b in range(num_blocks):
            local_blocks.append(stack_view.local_blocks[b + row_offset, b + col_offset])

        return _flatten_list(block_comm.allgather(local_blocks))

    @profiler.profile(level="api")
    def diagonal(self, val: NDArray = None) -> NDArray:
        """Returns or sets the diagonal elements of the matrix.

        This temporarily sets the return_dense state to True. Note that
        this will cause communication in the block-communicator.

        Returns
        -------
        diagonal : NDArray
            The diagonal elements of the whole matrix.

        """
        if val is None:
            # Getter
            # Store the current return_dense state and set it to True.
            original_return_dense = self.return_dense
            self.return_dense = True

            diagonals = []
            stack_view = self.stack[...]
            for b in range(self.num_local_blocks):
                diagonals.append(
                    xp.diagonal(stack_view.local_blocks[b, b], axis1=-2, axis2=-1)
                )

            # Restore the original return_dense state.
            self.return_dense = original_return_dense
            local_diagonal = xp.concatenate(diagonals, axis=-1)

            return xp.concatenate(block_comm.allgather(local_diagonal), axis=-1)
        else:
            # Setter
            raise NotImplementedError("Setting the diagonal is not implemented.")

    @profiler.profile(level="debug")
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

        if discard:
            t_discard_start = time.perf_counter()
            self._data = _block_view(
                self._data, axis=block_axis, num_blocks=stack_comm.size
            )
            self._data = xp.concatenate(self._data, axis=concatenate_axis)
            self._data[:] = 0.0
            synchronize_device()
            t_discard_end = time.perf_counter()
            global_comm.Barrier()
            t_discard_end_all = time.perf_counter()
            if global_comm.rank == 0:
                print(
                    f"        Discard time: {t_discard_end - t_discard_start:.3f} seconds",
                    flush=True,
                )
                print(
                    f"        Discard time all: {t_discard_end_all - t_discard_start:.3f} seconds",
                    flush=True,
                )
            return

        # We need to make sure that the block-view is memory-contiguous.
        # This does nothing if the data is already contiguous.
        t_block_view_start = time.perf_counter()
        self._data = _block_view(
            self._data, axis=block_axis, num_blocks=stack_comm.size
        )
        self._data = xp.ascontiguousarray(self._data)
        synchronize_device()
        t_block_view_end = time.perf_counter()
        global_comm.Barrier()
        t_block_view_end_all = time.perf_counter()
        if global_comm.rank == 0:
            print(
                f"        Block view time: {t_block_view_end - t_block_view_start:.3f} seconds",
                flush=True,
            )
            print(
                f"        Block view time all: {t_block_view_end_all - t_block_view_start:.3f} seconds",
                flush=True,
            )

        if NCCL_AVAILABLE:
            # Always use NCCL if available.
            receive_buffer = xp.empty_like(self._data)
            synchronize_device()
            global_comm.Barrier()
            t_nccl_start = time.perf_counter()
            nccl_stack_comm.all_to_all(self._data, receive_buffer)
            synchronize_device()
            t_nccl_end = time.perf_counter()
            global_comm.Barrier()
            t_nccl_end_all = time.perf_counter()
            if global_comm.rank == 0:
                print(
                    f"        NCCL Alltoall time: {t_nccl_end - t_nccl_start:.3f} seconds",
                    flush=True,
                )
                print(
                    f"        NCCL Alltoall time all: {t_nccl_end_all - t_nccl_start:.3f} seconds",
                    flush=True,
                )
            self._data = receive_buffer
        elif xp.__name__ == "numpy" or GPU_AWARE_MPI:
            # Use MPI if we are not on GPU or if we have GPU-aware MPI.
            receive_buffer = xp.empty_like(self._data)
            synchronize_device()
            global_comm.Barrier()
            t_nccl_start = time.perf_counter()
            stack_comm.Alltoall(self._data, receive_buffer)
            synchronize_device()
            t_nccl_end = time.perf_counter()
            global_comm.Barrier()
            t_nccl_end_all = time.perf_counter()
            if global_comm.rank == 0:
                print(
                    f"        MPI Alltoall time: {t_nccl_end - t_nccl_start:.3f} seconds",
                    flush=True,
                )
                print(
                    f"        MPI Alltoall time all: {t_nccl_end_all - t_nccl_start:.3f} seconds",
                    flush=True,
                )
            self._data = receive_buffer
        else:
            # Use the host memory if we are on GPU and do not have
            # GPU-aware MPI.
            _data_host = get_host(self._data)
            stack_comm.Alltoall(MPI.IN_PLACE, _data_host)
            self._data = xp.array(_data_host)

        t_concatenate_start = time.perf_counter()
        self._data = xp.concatenate(self._data, axis=concatenate_axis)
        synchronize_device()
        t_concatenate_end = time.perf_counter()
        global_comm.Barrier()
        t_concatenate_end_all = time.perf_counter()
        if global_comm.rank == 0:
            print(
                f"        Concatenate time: {t_concatenate_end - t_concatenate_start:.3f} seconds",
                flush=True,
            )
            print(
                f"        Concatenate time all: {t_concatenate_end_all - t_concatenate_start:.3f} seconds",
                flush=True,
            )

        # NOTE: There are a few things commented out here, since there
        # may be an alternative way to do the correct reshaping after
        # the Alltoall communication. The concatenatation needs to be
        # checked, as it may copy some data.

        # self._data = np.moveaxis(self._data, concatenate_axis, -2).reshape(new_shape)

    @profiler.profile(level="api")
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
    def symmetrize(self, op: Callable[[NDArray, NDArray], NDArray] = xp.add) -> None:
        """Symmetrizes the matrix with a given operation.

        This is done by setting the data to the result of the operation
        applied to the data and its conjugate transpose.

        Note
        ----
        This assumes that the matrix's sparsity pattern is symmetric.

        Parameters
        ----------
        op : callable, optional
            The operation to apply to the data and its conjugate
            transpose. Default is `xp.add`, so that the matrix is
            Hermitian after calling.

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

    @classmethod
    @abstractmethod
    def from_sparray(
        cls,
        arr: sparse.spmatrix,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ) -> "DSDBSparse":
        """Creates a new DSDBSparse matrix from a scipy.sparse array.

        Parameters
        ----------
        arr : sparse.spmatrix
            The sparse array to convert.
        block_sizes : NDArray
            The size of all the blocks in the matrix.
        global_stack_shape : tuple
            The global shape of the stack of matrices. The provided
            sparse matrix is replicated across the stack.

        Returns
        -------
        DSDBSparse
            The new DSDBSparse matrix.

        """
        ...

    @classmethod
    @abstractmethod
    def zeros_like(cls, dsdbsparse: "DSDBSparse") -> "DSDBSparse":
        """Creates a new DSDBSparse matrix with the same shape and dtype.

        All non-zero elements are set to zero, but the sparsity pattern
        is preserved.

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


class _DStackIndexer:
    """A utility class to locate substacks in the distributed stack.

    Parameters
    ----------
    dsdbsparse : DSDBSparse
        The underlying datastructure

    """

    def __init__(self, dsdbsparse: DSDBSparse) -> None:
        """Initializes the stack indexer."""
        self._dsdbsparse = dsdbsparse

    def __getitem__(self, index: tuple) -> "_DStackView":
        """Gets a substack view."""
        return _DStackView(self._dsdbsparse, index)


class _DStackView:
    """A utility class to create substack views.

    Parameters
    ----------
    dsdbsparse : DSDBSparse
        The underlying datastructure.
    stack_index : tuple
        The index of the substack to address.

    """

    def __init__(self, dsdbsparse: DSDBSparse, stack_index: tuple) -> None:
        """Initializes the stack indexer."""
        self._dsdbsparse = dsdbsparse
        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)
        stack_index = self._replace_ellipsis(stack_index)
        self._stack_index = stack_index
        self._block_indexer = _DSDBlockIndexer(
            self._dsdbsparse, self._stack_index, cache_stack=True
        )
        self._sparse_block_indexer = _DSDBlockIndexer(
            self._dsdbsparse, self._stack_index, return_dense=False, cache_stack=True
        )

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
            extra_dimensions = (self._dsdbsparse._data.ndim - 1) - (
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
        rows, cols = self._dsdbsparse._normalize_index(index)
        return self._dsdbsparse._get_items(self._stack_index, rows, cols)

    def __setitem__(self, index: tuple[ArrayLike, ArrayLike], values: NDArray) -> None:
        """Sets the requested data in the substack."""
        rows, cols = self._dsdbsparse._normalize_index(index)
        self._dsdbsparse._set_items(self._stack_index, rows, cols, values)

    @property
    def num_local_blocks(self) -> int:
        """Returns the number of local blocks."""
        return self._dsdbsparse.num_local_blocks

    @property
    def local_blocks(self) -> "_DSDBlockIndexer":
        """Returns a block indexer on the substack."""
        return self._block_indexer

    @property
    def sparse_local_blocks(self) -> "_DSDBlockIndexer":
        """Returns a sparse block indexer on the substack."""
        return self._sparse_block_indexer


@decorate_methods(profiler.profile(level="debug"))
class _DSDBlockIndexer:
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
    return_dense : bool, optional
        Whether to return dense arrays when accessing the blocks.
        Default is True.
    cache_stack : bool, optional
        Whether to propagate only the stack index to the block
        access methods, or to provide the data stack outright. Default
        is False.

    """

    def __init__(
        self,
        dsdbsparse: DSDBSparse,
        stack_index: tuple = (Ellipsis,),
        return_dense: bool = True,
        cache_stack: bool = False,
    ) -> None:
        """Initializes the block indexer."""
        self._dsdbsparse = dsdbsparse
        if not isinstance(stack_index, tuple):
            stack_index = (stack_index,)
        if cache_stack:
            self._arg = self._dsdbsparse.data[stack_index]
            self._is_index = False
        else:
            self._arg = stack_index
            self._is_index = True
        self._return_dense = return_dense

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
        if self._return_dense:
            return self._dsdbsparse._get_block(self._arg, row, col, self._is_index)
        return self._dsdbsparse._get_sparse_block(self._arg, row, col, self._is_index)

    def __setitem__(self, index: tuple, block: NDArray) -> None:
        """Sets the requested block in the data structure."""
        row, col = self._normalize_index(index)
        self._dsdbsparse._set_block(self._arg, row, col, block, self._is_index)
