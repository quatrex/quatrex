# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Distributed Extended Compressed Sparse Row (CSX) format with stack support."""

from __future__ import annotations

import numpy as np
from scipy.sparse import get_index_dtype

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.csx import CSX, symmetry_ops
from qttools.datastructures.csx_routines import make_canonical_coo
from qttools.utils.gpu_utils import get_host


class DCSX:
    """Distributed Extended Compressed Sparse Row (CSX) format with
    stack support.

    Parameters
    ----------
    dtype : xp.dtype[xp.generic]
        Data type of the matrix elements.
    num_rows : int
        Number of rows in the local matrix.
    num_cols : int
        Number of columns in the local matrix.
    row_offsets : NDArray
        Array of cumulative row counts across all ranks in the block
        communicator.
    local_stack_shape : tuple[int, ...]
        Shape of the local stack for this rank.
    row_ptr : NDArray
        Row pointer array for the local matrix in CSR format.
    row_ind : NDArray
        Row indices for the local matrix in COO format. Added for the
        extended CSX format.
    col_ind : NDArray
        Column indices for the local matrix in CSR format.
    symmetry : str | None, optional
        The symmetry of the matrix. This can be "symmetric",
        "hermitian", "skew-symmetric", "skew-hermitian", or None.
        Default is None.

    """

    _DELEGATED = [
        "dtype",
        "num_rows",
        "num_cols",
        "index_type",
        "local_stack_shape",
        "shape",
        "row_ptr",
        "row_ind",
        "col_ind",
        "nnz",
        "symmetry",
        "allocate_data",
        "free_data",
        "toarray",
        "_get_update_indices",
        "multiply_",
        "tocoo",
        "tocsr",
        "add_",
        "_data",
        "get_tile",
    ]

    def __init__(
        self,
        _csx: CSX,
        row_offsets: NDArray,
    ):

        self._csx = _csx

        # Distribution of rows
        self.row_offsets = row_offsets

        # Graph analysis attributes. They are populated by the
        # `_graph_analysis` method and used in the `expand_symmetry`
        # method.
        self.is_neighbour: NDArray | None = None
        self.num_neighbour_indices: NDArray | None = None
        self.neighbour_indices: dict[int, NDArray] | None = None

    def __getattr__(self, name: str):
        if name in self._DELEGATED:
            return getattr(self._csx, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value) -> None:
        if name in self._DELEGATED:
            raise AttributeError(
                f"{type(self).__name__!r} object attribute {name!r} is read-only "
                f"(it is delegated to the underlying CSX object)"
            )
        super().__setattr__(name, value)

    @property
    def data(self) -> NDArray:
        """Returns the local data."""
        if self._csx._data is None:
            raise ValueError("Data has not been allocated yet.")
        return self._csx._data

    @data.setter
    def data(self, value: NDArray) -> None:
        """Sets the local data."""
        if self._csx._data is None:
            raise ValueError("Data has not been allocated yet.")
        self._csx._data[...] = value

    def _to_dense(self) -> NDArray:
        """Returns the unsymmetrized dense array.

        Warning
        -------
        This is purely for testing purposes and should not be used in
        production code.

        Note
        ----
        This also unsymmetrizes the matrix if it is symmetric.

        Note
        ----
        This returns the global dense array, not just the local part.

        """
        dense = xp.zeros(
            self.local_stack_shape + (self.num_cols, self.num_cols), dtype=self.dtype
        )
        for idx in np.ndindex(self.local_stack_shape):
            data = self.data[idx]
            tmp = sparse.coo_matrix(
                (data, (self.row_ind, self.col_ind)),
                shape=(self.num_rows, self.num_cols),
            ).toarray()

            tmp = comm.block.all_gather_v(tmp, axis=0)

            if self.symmetry is not None:
                tmp += xp.triu(symmetry_ops[self.symmetry](tmp), k=1).transpose()

            dense[idx] = tmp

        return dense

    def _communicate_quantity_blocking(self, quantity: NDArray) -> dict[int, NDArray]:
        """
        Communicates a quantity to the neighbours.

        The quantity can either be the row indices, column indices, or
        the data. The quantity is communicated to the neighbours based
        on the graph analysis performed in `_graph_analysis`.

        Parameters
        ----------
        quantity : NDArray
            The quantity to communicate.

        Returns
        -------
        dict[int, NDArray]
            A dictionary mapping the rank of each neighbour to the
            received quantity.

        """
        if (
            self.is_neighbour is None
            or self.num_neighbour_indices is None
            or self.neighbour_indices is None
        ):
            raise ValueError("Graph analysis has not been performed yet.")

        my_rank = comm.block.rank
        recv_buffer = {}

        # NOTE: We do not deadlock here since we send in ascending
        # order.
        # Blocking receives from lower-ranked neighbours
        for rank in range(my_rank):
            if self.is_neighbour[my_rank, rank]:
                recv_buffer[rank] = xp.empty(
                    quantity.shape[:-1] + (self.num_neighbour_indices[my_rank, rank],),
                    dtype=quantity.dtype,
                )
                comm.block.recv(buf=recv_buffer[rank], source=rank)

        # Blocking sends to higher-ranked neighbours
        for rank in range(my_rank + 1, comm.block.size):
            if self.is_neighbour[my_rank, rank]:
                send_buffer = xp.ascontiguousarray(
                    quantity[..., self.neighbour_indices[rank]]
                )
                comm.block.send(buf=send_buffer, dest=rank)

        return recv_buffer

    def _communicate_quantity_nonblocking(
        self, quantity: NDArray
    ) -> dict[int, NDArray]:
        """
        Communicates a quantity to the neighbours.

        The quantity can either be the row indices, column indices, or
        the data. The quantity is communicated to the neighbours based
        on the graph analysis performed in `_graph_analysis`.

        Parameters
        ----------
        quantity : NDArray
            The quantity to communicate.

        Returns
        -------
        dict[int, NDArray]
            A dictionary mapping the rank of each neighbour to the
            received quantity.

        """
        if (
            self.is_neighbour is None
            or self.num_neighbour_indices is None
            or self.neighbour_indices is None
        ):
            raise ValueError("Graph analysis has not been performed yet.")

        my_rank = comm.block.rank
        recv_buffer = {}

        for rank in range(comm.block.size):
            if rank < my_rank and self.is_neighbour[my_rank, rank]:
                recv_buffer[rank] = xp.empty(
                    quantity.shape[:-1] + (self.num_neighbour_indices[my_rank, rank],),
                    dtype=quantity.dtype,
                )

        # First communicate the col indices
        requests = []
        send_buffers = []
        comm.block.group_start(comm.block._backend)
        for rank in range(comm.block.size):
            if rank == my_rank:
                continue

            # Post a send
            if rank > my_rank and self.is_neighbour[my_rank, rank]:
                send_buffer = xp.ascontiguousarray(
                    quantity[..., self.neighbour_indices[rank]]
                )
                send_buffers.append(send_buffer)
                requests.append(comm.block.isend(buf=send_buffer, dest=rank))

            # Post a receive
            if rank < my_rank and self.is_neighbour[my_rank, rank]:
                requests.append(comm.block.irecv(buf=recv_buffer[rank], source=rank))

        comm.block.group_end(comm.block._backend, requests)

        return recv_buffer

    def _communicate_quantity(self, quantity: NDArray) -> dict[int, NDArray]:
        """
        Communicates a quantity to the neighbours.

        The quantity can either be the row indices, column indices, or
        the data. The quantity is communicated to the neighbours based
        on the graph analysis performed in `_graph_analysis`.

        Parameters
        ----------
        quantity : NDArray
            The quantity to communicate.

        Returns
        -------
        dict[int, NDArray]
            A dictionary mapping the rank of each neighbour to the
            received quantity.

        """

        # When using host mpi, we need to do the communication with
        # blocking sends and receives.
        if comm.block._backend == "host_mpi":
            return self._communicate_quantity_blocking(quantity)

        else:
            return self._communicate_quantity_nonblocking(quantity)

    def _graph_analysis(self):
        """Performs a graph analysis to determine the neighbours of this rank.

        This populates the following attributes:
        - `is_neighbour`: A boolean array indicating whether each rank is a neighbour.
        - `num_neighbour_indices`: An array indicating the number of indices to send to each neighbour.
        - `neighbour_indices`: A dictionary mapping each neighbour rank to the indices to send to that neighbour.

        Note
        ----
        This method should only be called for symmetric matrices. For
        non-symmetric matrices, the graph analysis is not yet relevant.

        """

        if (
            self.is_neighbour is not None
            or self.num_neighbour_indices is not None
            or self.neighbour_indices is not None
        ):
            raise ValueError("Graph analysis has already been performed.")

        if self.symmetry is None:
            raise ValueError("Graph analysis is only relevant for symmetric matrices.")

        is_neighbour = np.zeros((1, comm.block.size), dtype=bool)
        num_neighbour_indices = np.zeros((1, comm.block.size), dtype=self.index_type)
        neighbour_indices = {}

        for rank in range(comm.block.size):
            indices = xp.argwhere(
                (self.col_ind < self.row_offsets[rank + 1])
                & (self.col_ind >= self.row_offsets[rank])
            ).ravel()

            # Do not include self connections in the neighbour list.
            if len(indices) > 0 and not (rank == comm.block.rank):
                is_neighbour[0, rank] = True
                num_neighbour_indices[0, rank] = len(indices)
                neighbour_indices[rank] = indices

        self.is_neighbour = np.empty(
            (comm.block.size, comm.block.size), dtype=is_neighbour.dtype
        )
        self.num_neighbour_indices = np.empty(
            (comm.block.size, comm.block.size), dtype=num_neighbour_indices.dtype
        )
        comm.block.all_gather(is_neighbour, self.is_neighbour, backend="device_mpi")
        comm.block.all_gather(
            num_neighbour_indices, self.num_neighbour_indices, backend="device_mpi"
        )
        # symmetrize the graph
        self.is_neighbour |= self.is_neighbour.T
        self.num_neighbour_indices += self.num_neighbour_indices.T

        self.neighbour_indices = neighbour_indices

    def _expand_sparsity(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Expands the sparsity pattern of the DCSX matrix to include
        the symmetric entries.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]
            The expanded row indices, column indices, the sorting
            indices, and the local indices. Last two outputs are for
            reuse in the `expand_symmetry` method.

        """
        if self.symmetry is None:
            raise ValueError("Symmetrization is only relevant for symmetric matrices.")
        if (
            self.is_neighbour is None
            or self.num_neighbour_indices is None
            or self.neighbour_indices is None
        ):
            self._graph_analysis()

        if self.neighbour_indices is None:
            raise ValueError("Graph analysis has not been performed yet.")

        recv_row_indices = self._communicate_quantity(self.row_ind)
        recv_col_indices = self._communicate_quantity(self.col_ind)

        # TODO: This could be optimized to avoid the concatenation and
        # sorting, but for now we will keep it simple.

        # Convert received global entries into their transposed local coordinates.
        new_row_ind = [self.row_ind] + [
            recv_col_indices[rank] - self.row_offsets[comm.block.rank]
            for rank in recv_col_indices.keys()
        ]
        new_col_ind = [self.col_ind] + [
            recv_row_indices[rank] + self.row_offsets[rank]
            for rank in recv_row_indices.keys()
        ]

        # NOTE: Need to account for the local symmetric entries.
        local_indices = xp.argwhere(
            (self.col_ind < self.row_offsets[comm.block.rank + 1])
            & (self.col_ind >= self.row_offsets[comm.block.rank])
        ).ravel()

        # Filter out the diagonal.
        local_indices = local_indices[
            self.col_ind[local_indices]
            != self.row_ind[local_indices] + self.row_offsets[comm.block.rank]
        ]
        new_row_ind.append(
            self.col_ind[local_indices] - self.row_offsets[comm.block.rank]
        )
        new_col_ind.append(
            self.row_ind[local_indices] + self.row_offsets[comm.block.rank]
        )

        nnz = np.sum([len(ind) for ind in new_row_ind])
        index_type = get_index_dtype(maxval=nnz)
        new_row_ind = [ind.astype(index_type) for ind in new_row_ind]
        new_col_ind = [ind.astype(index_type) for ind in new_col_ind]

        new_row_ind = xp.concatenate(new_row_ind, axis=-1, dtype=index_type)
        new_col_ind = xp.concatenate(new_col_ind, axis=-1, dtype=index_type)

        # Sort the indices and data to get the canonical format
        new_row_ind, new_col_ind, sort_idx = make_canonical_coo(
            row_ind=new_row_ind,
            col_ind=new_col_ind,
        )

        return new_row_ind, new_col_ind, sort_idx, local_indices

    def expand_sparsity(
        self,
    ) -> tuple[NDArray, NDArray]:
        """Expands the sparsity pattern of the DCSX matrix to include
        the symmetric entries.

        Returns
        -------
        tuple[NDArray, NDArray]
            The expanded row indices and column indices. The sorting
            indices can be used to sort the data array accordingly.

        """
        new_row_ind, new_col_ind, __, __ = self._expand_sparsity()
        return new_row_ind, new_col_ind

    def expand_symmetry(
        self,
    ) -> DCSX:
        """Symmetrizes the DCSX matrix. This returns a new DCSX matrix
        that is the symmetrized version of the original matrix.

        Returns
        -------
        DCSX
            The symmetrized DCSX matrix.

        """
        if self.symmetry == "upper-triangular":
            raise ValueError(
                "Symmetrization is not supported for upper-triangular matrices."
            )
        (new_row_ind, new_col_ind, sort_idx, local_indices) = self._expand_sparsity()

        # NOTE: All of the below could be cached, but then we start to
        # lose the memory advantage and we could just save the full
        # row_ind/col_ind/data arrays. So we will not do that for now.
        recv_data = self._communicate_quantity(self.data)

        # TODO: This could be optimized to avoid the concatenation and
        # sorting, but for now we will keep it simple.
        new_data = [self.data] + [
            symmetry_ops[self.symmetry](recv_data[rank]) for rank in recv_data.keys()
        ]

        # NOTE: Need to account for the local symmetric entries.
        new_data.append(symmetry_ops[self.symmetry](self.data[..., local_indices]))

        new_data = xp.concatenate(new_data, axis=-1)
        new_data = new_data[..., sort_idx]

        _csx = CSX(
            dtype=self.dtype,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            local_stack_shape=self.local_stack_shape,
            row_ind=new_row_ind,
            col_ind=new_col_ind,
        )

        dcsx = DCSX(
            _csx=_csx,
            row_offsets=self.row_offsets,
        )
        dcsx.allocate_data()
        dcsx.data = new_data

        return dcsx

    def __matmul__(self, other: NDArray) -> NDArray:
        """Matrix multiplication with a 1D or 2D array.

        Note
        ----
        This method currently only implements local matrix
        multiplication. This means `other` is expected to be a local
        array on each rank.

        Parameters
        ----------
        other : NDArray
            The array to multiply the matrix by. Can be either a 1D
            array of shape (cols,) or a 2D array of shape (cols, N).

        Returns
        -------
        NDArray
            The result of the matrix multiplication. Will have the shape
            `self.local_stack_shape + (self.num_rows,) + other.shape[1:]`.

        """
        return self._csx @ other

    @classmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        local_stack_shape: tuple = tuple(),
        symmetry: str | None = None,
        allocate: bool = True,
    ) -> DCSX:
        """Allocates a DCSX matrix from a sparse array.

        Note
        ----
        This assumes that the input sparse array is already distributed
        correctly.

        Note
        ----
        For the QTBM datastructure, distribution of the stack is handled
        outside since the datastructure does not need to operator
        through the stack.

        Parameters
        ----------
        sparray : sparse.spmatrix
            The sparse array to convert to DCSX format.
        local_stack_shape : tuple, optional
            The shape of the local stack for this rank. Default is an
            empty tuple, which means no stack.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian",
            "upper-triangular", or None. Default is None.
        allocate : bool, optional
            Whether to allocate the data array. Default is True.

        Returns
        -------
        DCSX
            The DCSX matrix.

        """
        if comm.stack is None or comm.block is None:
            raise ValueError("Communicators must be initialized.")

        sparray = sparray.tocoo()
        shape = sparray.shape
        index_dtype = sparray.col.dtype

        # Canonicalizes the COO format.
        if not sparray.has_canonical_format:
            sparray.sum_duplicates()
        if not sparray.has_canonical_format:
            raise ValueError("COO format is not canonical.")

        row_ind = sparray.row
        col_ind = sparray.col

        dtype = sparray.data.dtype

        num_rows = xp.array([shape[0]], dtype=index_dtype)
        num_cols = xp.array([shape[1]], dtype=index_dtype)

        all_rows = xp.zeros((comm.block.size), dtype=index_dtype)
        all_cols = xp.zeros((comm.block.size), dtype=index_dtype)
        comm.block.all_gather(num_rows, all_rows)
        comm.block.all_gather(num_cols, all_cols)

        # Check that the number of columns are the same across all block comm ranks.
        if not xp.all(all_cols == all_cols[0]):
            raise ValueError("The number of columns must be the same across all ranks.")

        # Check that the sum of all rows are the same as the number of columns.
        # This means the matrix is split across the rows.
        if not xp.sum(all_rows) == all_cols[0]:
            raise ValueError(
                "The sum of all rows must be the same as the number of columns."
            )

        row_offsets = xp.zeros((comm.block.size + 1), dtype=index_dtype)
        comm.block.all_gather(num_rows, row_offsets[1:])
        row_offsets = get_host(xp.cumsum(row_offsets))

        _csx = CSX.from_indices(
            row_ind=row_ind,
            col_ind=col_ind,
            dtype=dtype,
            shape=shape,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
            allocate=allocate,
        )
        if allocate:
            _csx.data = sparray.data

        dcsx = cls(
            _csx=_csx,
            row_offsets=row_offsets,
        )

        return dcsx

    @classmethod
    def from_indices(
        cls,
        row_ind: NDArray,
        col_ind: NDArray,
        dtype: xp.dtype[xp.generic],
        shape: tuple[int, int],
        local_stack_shape: tuple = tuple(),
        symmetry: str | None = None,
        allocate: bool = True,
    ) -> DCSX:
        """Allocates a DCSX matrix from a sparse array.

        Note
        ----
        This assumes that the input sparse array is already distributed
        correctly.

        Parameters
        ----------
        row_ind : NDArray
            The row indices of the non-zero entries.
        col_ind : NDArray
            The column indices of the non-zero entries.
        dtype : xp.dtype[xp.generic]
            The data type of the matrix elements.
        shape : tuple[int, int]
            The shape of the matrix.
        local_stack_shape : tuple, optional
            The shape of the local stack for this rank. Default is an
            empty tuple, which means no stack.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian", "upper-triangular", or None. Default is None.
        allocate : bool, optional
            Whether to allocate the data array. Default is True.

        Returns
        -------
        DCSX
            The DCSX matrix.

        """

        if comm.stack is None or comm.block is None:
            raise ValueError("Communicators must be initialized.")

        index_dtype = row_ind.dtype

        num_rows = xp.array([shape[0]], dtype=index_dtype)
        num_cols = xp.array([shape[1]], dtype=index_dtype)

        all_rows = xp.zeros((comm.block.size), dtype=index_dtype)
        all_cols = xp.zeros((comm.block.size), dtype=index_dtype)
        comm.block.all_gather(num_rows, all_rows)
        comm.block.all_gather(num_cols, all_cols)

        # Check that the number of columns are the same across all block comm ranks.
        if not xp.all(all_cols == all_cols[0]):
            raise ValueError("The number of columns must be the same across all ranks.")

        # Check that the sum of all rows are the same as the number of columns.
        # This means the matrix is split across the rows.
        if not xp.sum(all_rows) == all_cols[0]:
            raise ValueError(
                "The sum of all rows must be the same as the number of columns."
            )

        row_offsets = xp.zeros((comm.block.size + 1), dtype=index_dtype)
        comm.block.all_gather(num_rows, row_offsets[1:])
        row_offsets = get_host(xp.cumsum(row_offsets))

        _csx = CSX.from_indices(
            row_ind=row_ind,
            col_ind=col_ind,
            dtype=dtype,
            shape=shape,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
            allocate=allocate,
        )

        dcsx = cls(
            _csx=_csx,
            row_offsets=row_offsets,
        )

        return dcsx
