# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Distributed Extended Compressed Sparse Row (CSX) format with stack support."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.csx import CSX
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.utils.gpu_utils import get_host


class DCSX:
    """Distributed Extended Compressed Sparse Row (CSX) format with
    stack support.

    Parameters
    ----------
    dtype : xp.dtype[xp.generic]
        Data type of the matrix elements.
    rows : int
        Number of rows in the local matrix.
    cols : int
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
        "rows",
        "cols",
        "index_type",
        "local_stack_shape",
        "shape",
        "row_ptr",
        "row_ind",
        "col_ind",
        "nnz",
        "symmetry",
        "allocate_data",
        "toarray",
        "_get_update_indices",
        "multiply_",
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
        self.recv_row_indices: dict[int, NDArray] | None = None
        self.recv_col_indices: dict[int, NDArray] | None = None

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
            self.local_stack_shape + (self.cols, self.cols), dtype=self.dtype
        )
        for idx in np.ndindex(self.local_stack_shape):
            data = self.data[idx]
            tmp = sparse.coo_matrix(
                (data, (self.row_ind, self.col_ind)), shape=(self.rows, self.cols)
            ).toarray()

            tmp = comm.block.all_gather_v(tmp, axis=0)

            if self.symmetry is not None:
                tmp += xp.triu(symmetry_ops[self.symmetry](tmp), k=1).transpose()

            dense[idx] = tmp

        return dense

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

        if (
            self.is_neighbour is None
            or self.num_neighbour_indices is None
            or self.neighbour_indices is None
        ):
            raise ValueError("Graph analysis has not been performed yet.")

        recv_buffer = {}

        for rank in range(comm.block.size):
            if rank < comm.block.rank and self.is_neighbour[comm.block.rank, rank]:
                recv_buffer[rank] = xp.empty(
                    quantity.shape[:-1]
                    + (self.num_neighbour_indices[comm.block.rank, rank],),
                    dtype=quantity.dtype,
                )

        # First communicate the col indices
        requests = []
        send_buffers = []
        comm.block.group_start(comm.block._config["send_recv"])
        for rank in range(comm.block.size):
            if rank == comm.block.rank:
                continue

            # Post a send
            if rank > comm.block.rank and self.is_neighbour[comm.block.rank, rank]:
                send_buffer = xp.ascontiguousarray(
                    quantity[..., self.neighbour_indices[rank]]
                )
                send_buffers.append(send_buffer)
                requests.append(comm.block.isend(buf=send_buffer, dest=rank))

            # Post a receive
            if rank < comm.block.rank and self.is_neighbour[comm.block.rank, rank]:
                requests.append(comm.block.irecv(buf=recv_buffer[rank], source=rank))

        comm.block.group_end(comm.block._config["send_recv"], requests)

        return recv_buffer

    def _graph_analysis(self):
        """Performs a graph analysis to determine the neighbours of this rank.

        This populates the following attributes:
        - `is_neighbour`: A boolean array indicating whether each rank is a neighbour.
        - `num_neighbour_indices`: An array indicating the number of indices to send to each neighbour.
        - `neighbour_indices`: A dictionary mapping each neighbour rank to the indices to send to that neighbour.
        - `recv_row_indices`: A dictionary mapping each neighbour rank to the row indices received from that neighbour.
        - `recv_col_indices`: A dictionary mapping each neighbour rank to the column indices received from that neighbour.

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

        is_neighbour = xp.zeros((1, comm.block.size), dtype=bool)
        num_neighbour_indices = xp.zeros((1, comm.block.size), dtype=self.index_type)
        neighbour_indices = {}

        col_ind = self.col_ind
        for rank in range(comm.block.size):
            indices = xp.argwhere(
                (col_ind < self.row_offsets[rank + 1])
                & (col_ind >= self.row_offsets[rank])
            ).ravel()

            # Do not include self connections in the neighbour list.
            if len(indices) > 0:
                is_neighbour[0, rank] = True
                num_neighbour_indices[0, rank] = len(indices)
                neighbour_indices[rank] = indices

        self.is_neighbour = comm.block.all_gather_v(is_neighbour, axis=0)
        self.num_neighbour_indices = comm.block.all_gather_v(
            num_neighbour_indices, axis=0
        )
        # symmetrize the graph
        self.is_neighbour |= self.is_neighbour.T
        self.num_neighbour_indices += self.num_neighbour_indices.T

        self.neighbour_indices = neighbour_indices

        # In the symmetric case, we need now to communicate the row/col indices
        # to the neighbouring ranks

        # Allocate the receiv buffers for the row/col indices
        self.recv_row_indices = self._communicate_quantity(self.row_ind)
        self.recv_col_indices = self._communicate_quantity(self.col_ind)

    def expand_symmetry(
        self,
    ) -> "DCSX":
        """Symmetrizes the DCSX matrix. This returns a new DCSX matrix
        that is the symmetrized version of the original matrix.

        Note
        ----
        This method should only be called for symmetric matrices. For
        non-symmetric matrices, the symmetrization is not yet relevant.

        Note
        ----
        This method is intended to be used in the matrix assembly
        process in QTBM.

        Returns
        -------
        DCSX
            The symmetrized DCSX matrix.

        """
        if self.symmetry is None:
            raise ValueError("Symmetrization is only relevant for symmetric matrices.")

        if (
            self.is_neighbour is None
            or self.num_neighbour_indices is None
            or self.neighbour_indices is None
        ):
            self._graph_analysis()

        if self.recv_row_indices is None or self.recv_col_indices is None:
            raise ValueError("Graph analysis has not been performed yet.")

        if self.neighbour_indices is None:
            raise ValueError("Graph analysis has not been performed yet.")

        recv_data = self._communicate_quantity(self.data)

        # NOTE: All of the below could be cached, but then we start to
        # lose the memory advantage and we could just save the full
        # row_ind/col_ind/data arrays. So we will not do that for now.

        # TODO: This could be optimized to avoid the concatenation and
        # sorting, but for now we will keep it simple.

        # Convert received global entries into their transposed local coordinates.
        new_row_ind = [self.row_ind] + [
            self.recv_col_indices[rank] - self.row_offsets[comm.block.rank]
            for rank in recv_data.keys()
        ]
        new_col_ind = [self.col_ind] + [
            self.recv_row_indices[rank] + self.row_offsets[rank]
            for rank in recv_data.keys()
        ]

        new_data = [self.data] + [
            symmetry_ops[self.symmetry](recv_data[rank]) for rank in recv_data.keys()
        ]

        # NOTE: Need to account for the local symmetric entries.
        if comm.block.rank in self.neighbour_indices:
            local_neighbour_indices = self.neighbour_indices[comm.block.rank]

            # Filter out the diagonal.
            local_neighbour_indices = local_neighbour_indices[
                self.col_ind[local_neighbour_indices]
                != self.row_ind[local_neighbour_indices]
                + self.row_offsets[comm.block.rank]
            ]
            new_row_ind.append(
                self.col_ind[local_neighbour_indices]
                - self.row_offsets[comm.block.rank]
            )
            new_col_ind.append(
                self.row_ind[local_neighbour_indices]
                + self.row_offsets[comm.block.rank]
            )
            new_data.append(
                symmetry_ops[self.symmetry](self.data[..., local_neighbour_indices])
            )

        new_row_ind = xp.concatenate(new_row_ind, axis=-1)
        new_col_ind = xp.concatenate(new_col_ind, axis=-1)
        new_data = xp.concatenate(new_data, axis=-1)

        # Sort the indices and data to get the canonical format
        flat_idx = new_row_ind * self.cols + new_col_ind
        sort_idx = xp.argsort(flat_idx)
        new_row_ind = new_row_ind[sort_idx]
        new_col_ind = new_col_ind[sort_idx]
        new_data = new_data[..., sort_idx]

        _csx = CSX(
            dtype=self.dtype,
            rows=self.rows,
            cols=self.cols,
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

    def add_(
        self,
        other: "DCSX",
        prefactor: int | float | np.number = 1.0,
        cache: bool = True,
        cache_id: int | None = None,
    ) -> None:
        """Adds another DCSX matrix to this one in place.

        Note
        ----
        This method assumes that the sparsity pattern of `other` is a
        subset of the sparsity pattern of `self`. If this is not the
        case, a ValueError will be raised.

        Parameters
        ----------
        other : DCSX
            The DCSX matrix to add to this one.
        prefactor : int | float | np.number, optional
            A prefactor to multiply `other` by before adding. Default is
            1.0.
        cache : bool, optional
            Whether to cache the indices for the addition operation.
            Default is True.
        cache_id : int | None, optional
            An optional cache ID to use for caching the indices. If
            None, the ID of `other` will be used. Default is None.

        """
        # In the case of adding a symmetric matrix to a non-symmetric
        # one, we expand the symmetry of the symmetric matrix to match
        # the non-symmetric one.
        # NOTE: We route here to the `DCSX` version of `expand_symmetry`
        # since it involves communication between ranks.
        if other.symmetry is not None and self.symmetry is None:

            # Get the ID here since `expand_symmetry` will create a new
            # object, but we want to keep the cache ID of the original
            # object.
            # i.e. `expand_symmetry` will be called again, but the
            # update indices are the same again.
            if cache_id is None:
                cache_id = id(other)

            other = other.expand_symmetry()

        # Afterwards we can just use the `CSX` version of `add_` since
        # the symmetries are now the same.
        self._csx.add_(
            other._csx,
            prefactor=prefactor,
            cache=cache,
            cache_id=cache_id,
        )

    def get_tile(
        self,
        row_ind: NDArray | None = None,
        col_ind: NDArray | None = None,
        unsymmetrize: bool = False,
    ) -> "CSX":
        """Returns a tile of the matrix as a new CSX object.

        Note
        ----
        The output is a non-distributed CSX object. If needed, one needs
        to manually gather the results.

        Note
        ----
        We cannot directly return the method of the underlying `CSX`
        object since the unsymmetrization needs to be done on the full
        matrix, not just the local part.

        Parameters
        ----------
        row_ind : NDArray | None
            The row indices of the tile. If None, all rows are included.
        col_ind : NDArray | None
            The column indices of the tile. If None, all columns are
            included.
        unsymmetrize : bool, optional
            Whether to unsymmetrize the tile if the matrix is symmetric.

        Returns
        -------
        CSX
            The tile as a new CSX object.

        """
        if not unsymmetrize:
            return self._csx.get_tile(
                row_ind=row_ind,
                col_ind=col_ind,
            )

        # NOTE: This is expensive. It would be possible to only
        # communicate the relevant entries, but for now we will keep it
        # simple.
        tmp = self.expand_symmetry()
        return tmp.get_tile(
            row_ind=row_ind,
            col_ind=col_ind,
        )

    @classmethod
    def from_sparray(
        cls,
        sparray: sparse.spmatrix,
        local_stack_shape: tuple,
        symmetry: str | None = None,
        dtype: xp.dtype[xp.generic] | None = None,
        allocate: bool = True,
    ) -> "DCSX":
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
        local_stack_shape : tuple
            The shape of the local stack for this rank.
        symmetry : str | None, optional
            The symmetry of the matrix. This can be "symmetric",
            "hermitian", "skew-symmetric", "skew-hermitian", or None.
            Default is None.
        dtype : xp.dtype[xp.generic] | None, optional
            The data type of the matrix elements. Default is
            None and the data type of the input sparse array is used.
        allocate : bool, optional
            Whether to allocate the data array. Default is True.

        Returns
        -------
        DCSX
            The DCSX matrix.

        """

        if comm.stack is None or comm.block is None:
            raise ValueError("Communicators must be initialized.")

        coo = sparray.tocoo()

        index_dtype = coo.col.dtype

        rows = xp.array([coo.shape[0]], dtype=index_dtype)
        cols = xp.array([coo.shape[1]], dtype=index_dtype)

        all_rows = xp.zeros((comm.block.size), dtype=index_dtype)
        all_cols = xp.zeros((comm.block.size), dtype=index_dtype)
        comm.block.all_gather(rows, all_rows)
        comm.block.all_gather(cols, all_cols)

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
        comm.block.all_gather(rows, row_offsets[1:])
        row_offsets = get_host(xp.cumsum(row_offsets))

        _csx = CSX.from_sparray(
            coo,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
            dtype=coo.data.dtype if dtype is None else dtype,
            allocate=allocate,
        )

        dcsx = cls(
            _csx=_csx,
            row_offsets=row_offsets,
        )

        return dcsx
