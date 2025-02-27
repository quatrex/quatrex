# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from mpi4py import MPI

from qttools import NDArray, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.utils.solvers_utils import get_batches


class RGF(GFSolver):
    """Selected inversion solver based on the Schur complement.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        100.

    """

    def __init__(
        self,
        a: DSBSparse,
        max_batch_size: int = 100,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """Initializes the selected inversion solver."""
        self.max_batch_size = max_batch_size

        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        # --- Initialize the reduced system ---
        # `rs`` stand for reduced system
        # To use AllGather in the comm step, need all ranks to have the
        # same numbers of elements ot communicates
        self.n_diag_rs = 2 * comm_size
        self.n_diag_local = len(a.local_block_sizes)

        block_sizes_rs = xp.array([0 for i in self.n_diag_rs])

        if comm_rank == 0:
            block_sizes_rs[1] = a.local_block_sizes[-1]
        elif comm_rank == comm_size - 1:
            block_sizes_rs[-2] = a.local_block_sizes[0]
        else:
            block_sizes_rs[2 * comm_rank] = a.local_block_sizes[0]
            block_sizes_rs[2 * comm_rank + 1] = a.local_block_sizes[-1]

        # Aggregate the block sizes of the reduced system
        comm.Allreduce(MPI.IN_PLACE, block_sizes_rs, op=MPI.SUM)

        # Allocate the reduced system
        self._self._A_diagonal_blocks = [None] * a.num_blocks
        self._self._A_lower_diagonal_blocks = [None] * (a.num_blocks - 2)
        self._self._A_upper_diagonal_blocks = [None] * (a.num_blocks - 2)
        for i in range(1, self.n_diag_rs - 1):
            self._self._A_diagonal_blocks[i] = xp.empty(
                (block_sizes_rs[i], block_sizes_rs[i]),
                dtype=a.local_blocks[0, 0].dtype,
            )
            if i < self.n_diag_rs - 1:
                self._self._A_lower_diagonal_blocks[i] = xp.empty(
                    (
                        block_sizes_rs[i + 1],
                        block_sizes_rs[i],
                    ),
                    dtype=a.local_blocks[1, 0].dtype,
                )
                self._self._A_upper_diagonal_blocks[i] = xp.empty(
                    (
                        block_sizes_rs[i],
                        block_sizes_rs[i + 1],
                    ),
                    dtype=a.local_blocks[0, 1].dtype,
                )

        # --- Allocate permutation buffer blocks ---
        self.buffer_lower: list = None
        self.buffer_upper: list = None
        if comm_rank != 0 and comm_rank != comm_size - 1:

            self.buffer = []
            for i in range(self.n_diag_local):
                self.buffer_lower.append(
                    xp.zeros(
                        (a.blocks[i, i].shape[0], a.blocks[i, i].shape[1]),
                        dtype=a.blocks[i, i].dtype,
                    )
                )

                self.buffer_upper.append(
                    xp.zeros(
                        (a.blocks[i, i].shape[0], a.blocks[i, i].shape[1]),
                        dtype=a.blocks[i, i].dtype,
                    )
                )

        # Buffers for the Schur-complement
        self.xr_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

    def selected_inv(
        self,
        a: DSBSparse,
        obc_blocks: OBCBlocks | None = None,
        out: DSBSparse | None = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None | DSBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : DSBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSBSparse object.

        """
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)
            if self.buffer is None:
                if comm_rank == 0:
                    # Direction: downward Schur-complement
                    self._downward_schur_inv(
                        a=a,
                        obc=obc_blocks,
                        stack_slice=stack_slice,
                        invert_last_block=False,
                    )
                elif comm_rank == comm_size - 1:
                    # Direction: upward Schur-complement
                    self._upward_schur_inv(
                        a=a,
                        obc=obc_blocks,
                        stack_slice=stack_slice,
                        invert_last_block=False,
                    )
                else:
                    raise ValueError(
                        f"Rank {comm_rank}, should have a buffer allocated as it is a 'middle process'."
                    )
            else:
                # Permuted Schur-complement
                self._permuted_schur_inv(
                    a=a,
                    stack_slice=stack_slice,
                    obc=obc_blocks,
                )

            # Map the local partition boundary blocks to the reduced system
            self.map_local_partition_to_rs(
                a=a,
                comm=comm,
            )

            # Communicate the reduced system to all processes
            self.aggregate_rs(
                comm=comm,
            )

            # Perform selected-inversion of reduced system
            self._solve_rs()

            # Map back the reduced system to the local partition
            self.map_rs_to_local_partition(
                a=a,
                comm=comm,
            )

            if self.buffer is None:
                if comm_rank == 0:
                    # Direction: upward sell-inv
                    self._downward_sellinv_inv(
                        a=a,
                        stack_slice=stack_slice,
                    )
                elif comm_rank == comm_size - 1:
                    # Direction: downward sell-inv
                    self._upward_sellinv_inv(
                        a=a,
                        stack_slice=stack_slice,
                    )
                else:
                    raise ValueError(
                        f"Rank {comm_rank}, should have a buffer allocated as it is a 'middle process'."
                    )
            else:
                # Permuted Sell-inv
                self._permuted_sellinv_inv(
                    a=a,
                    stack_slice=stack_slice,
                )

    def _downward_schur_inv(
        self,
        a: DSBSparse,
        obc: OBCBlocks,
        stack_slice: slice,
        invert_last_block: bool,
    ):
        for n_i in range(0, self.n_diag_local - 1):
            a_jj = (
                a.local_blocks[n_i + 1, n_i + 1]
                if obc is None
                else a.local_blocks[n_i + 1, n_i + 1] - obc[stack_slice]
            )

            self.xr_diag_blocks[n_i] = xp.linalg.inv(a_jj)

            self.xr_diag_blocks[n_i + 1] = (
                a.local_block[n_i + 1, n_i + 1]
                - a.local_block[n_i + 1, n_i]
                @ self.xr_diag_blocks[n_i]
                @ a.local_block[n_i, n_i + 1]
            )

        if invert_last_block:
            self.xr_diag_blocks[-1] = xp.linalg.inv(a.local_block[-1, -1])

    def _upward_schur_inv(
        self,
        a: DSBSparse,
        obc: OBCBlocks,
        stack_slice: slice,
        invert_last_block: bool,
    ):
        for n_i in range(self.n_diag_local - 1, 0, -1):
            self.xr_diag_blocks[n_i] = xp.linalg.inv(a.local_block[n_i, n_i])

            self.xr_diag_blocks[n_i - 1] = (
                a.local_block[n_i - 1, n_i - 1]
                - a.local_block[n_i - 1, n_i]
                @ a.local_block[n_i, n_i]
                @ a.local_block[n_i, n_i - 1]
            )

        if invert_last_block:
            self.xr_diag_blocks[0] = xp.linalg.inv(a.local_block[0, 0])

    def _permuted_schur_inv(
        self,
        a: DSBSparse,
        obc: OBCBlocks,
        stack_slice: slice,
    ):
        self.buffer_lower[0] = a.local_block[0, 1]
        self.buffer_upper[0] = a.local_block[1, 0]

        for n_i in range(1, self.n_diag_local - 1):
            # Inverse current diagonal block
            self.xr_diag_blocks[n_i] = xp.linalg.inv(a.local_block[n_i, n_i])

            # Update next diagonal block
            self.xr_diag_blocks[n_i + 1] = (
                a.local_block[n_i + 1, n_i + 1]
                - a.local_block[n_i + 1, n_i]
                @ a.local_block[n_i, n_i]
                @ a.local_block[n_i, n_i + 1]
            )

            # Update lower buffer block
            self.buffer_lower[n_i] = (
                -self.buffer_lower[n_i - 1]
                @ a.local_block[n_i, n_i]
                @ a.local_block[n_i, n_i + 1]
            )

            # Update upper buffer block
            self.buffer_upper[n_i] = (
                -a.local_block[n_i + 1, n_i]
                @ a.local_block[n_i, n_i]
                @ self.buffer_upper[n_i - 1]
            )

            # Update 0-block (first)
            self.xr_diag_blocks[0] = (
                a.local_block[0, 0]
                - self.buffer_lower[n_i - 1]
                @ a.local_block[n_i, n_i]
                @ self.buffer_upper[n_i - 1]
            )

    def map_local_partition_to_rs(
        self,
        a: DSBSparse,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        if comm_rank == 0:
            self._A_diagonal_blocks[1] = a.local_blocks[-1, -1]
            self._A_lower_diagonal_blocks[1] = a.local_blocks[-1, -2]
            self._A_upper_diagonal_blocks[1] = a.local_blocks[-2, -1]
        elif comm_rank == comm_size - 1:
            self._A_diagonal_blocks[-2] = a.local_blocks[0, 0]
        else:
            self._A_diagonal_blocks[2 * comm_rank] = a.local_blocks[0, 0]
            self._A_diagonal_blocks[2 * comm_rank + 1] = a.local_blocks[-1, -1]

            self._A_lower_diagonal_blocks[2 * comm_rank] = self.buffer_upper[-2]
            self._A_upper_diagonal_blocks[2 * comm_rank] = self.buffer_lower[-2]

            self._A_lower_diagonal_blocks[2 * comm_rank + 1] = a.local_blocks[-1, -2]
            self._A_upper_diagonal_blocks[2 * comm_rank + 1] = a.local_blocks[-2, -1]

    def aggregate_rs(
        self,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        comm.Allgather(
            MPI.IN_PLACE,
            self._A_diagonal_blocks_comm,
        )
        comm.Allgather(
            MPI.IN_PLACE,
            self._A_lower_diagonal_blocks_comm,
        )
        comm.Allgather(
            MPI.IN_PLACE,
            self._A_upper_diagonal_blocks_comm,
        )

    def _solve_rs(
        self,
        stack_slice: slice,
    ):
        # Forward pass
        for n_i in range(0, self._A_diagonal_blocks.shape[0] - 1):
            self._A_diagonal_blocks[n_i] = xp.linalg.inv(self._A_diagonal_blocks[n_i])

            self._A_diagonal_blocks[n_i + 1] = (
                self._A_diagonal_blocks[n_i + 1]
                - self._A_lower_diagonal_blocks[n_i]
                @ self._A_diagonal_blocks[n_i]
                @ self._A_upper_diagonal_blocks[n_i]
            )

        self._A_diagonal_blocks[-1] = xp.linalg.inv(self._A_diagonal_blocks[-1])

        # Backward pass
        if self._A_diagonal_blocks.shape[0] > 1:
            # If there is only a single diagonal block, we don't need these buffers.
            temp_lower = xp.empty_like(self._A_lower_diagonal_blocks[0])

        for n_i in range(self._A_diagonal_blocks.shape[0] - 2, -1, -1):
            temp_lower[:, :] = self._A_lower_diagonal_blocks[n_i]

            self._A_lower_diagonal_blocks[n_i] = (
                -self._A_diagonal_blocks[n_i + 1]
                @ self._A_lower_diagonal_blocks[n_i]
                @ self._A_diagonal_blocks[n_i]
            )

            self._A_upper_diagonal_blocks[n_i] = (
                -self._A_diagonal_blocks[n_i]
                @ self._A_upper_diagonal_blocks[n_i]
                @ self._A_diagonal_blocks[n_i + 1]
            )

            self._A_diagonal_blocks[n_i] = (
                self._A_diagonal_blocks[n_i]
                - self._A_upper_diagonal_blocks[n_i]
                @ temp_lower
                @ self._A_diagonal_blocks[n_i]
            )

    def map_rs_to_local_partition(
        self,
        a: DSBSparse,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        if comm_rank == 0:
            a.local_blocks[-1, -1] = self._A_diagonal_blocks[0]
            a.local_blocks[-1, -2] = self._A_lower_diagonal_blocks[0]
            a.local_blocks[-2, -1] = self._A_upper_diagonal_blocks[0]
        elif comm_rank == comm_size - 1:
            a.local_blocks[0, 0] = self._A_diagonal_blocks[-1]
        else:
            a.local_blocks[0, 0] = self._A_diagonal_blocks[2 * comm_rank - 1]
            a.local_blocks[-1, -1] = self._A_diagonal_blocks[2 * comm_rank]

            self.buffer_upper[-2] = self._A_lower_diagonal_blocks[2 * comm_rank - 1]
            self.buffer_lower[-2] = self._A_upper_diagonal_blocks[2 * comm_rank - 1]

            a.local_blocks[-1, -2] = self._A_lower_diagonal_blocks[2 * comm_rank]
            a.local_blocks[-2, -1] = self._A_upper_diagonal_blocks[2 * comm_rank]

    def _downward_sellinv_inv(
        self,
        a: DSBSparse,
        stack_slice: slice,
    ):
        if self.n_diag_local > 1:
            # If there is only a single diagonal block, we don't need these buffers.
            # TODO, THis doesn't work in the case of non-uniform block sizes
            temp_lower = xp.empty_like(a.local_block[1, 0])

        for n_i in range(self.n_diag_local - 2, -1, -1):
            temp_lower[:, :] = a.local_block[n_i, n_i - 1]

            a.local_block[n_i, n_i - 1] = (
                -self.xr_diag_blocks[n_i + 1]
                @ a.local_block[n_i, n_i - 1]
                @ self.xr_diag_blocks[n_i]
            )

            a.local_block[n_i - 1, n_i] = (
                -self.xr_diag_blocks[n_i]
                @ a.local_block[n_i - 1, n_i]
                @ self.xr_diag_blocks[n_i + 1]
            )

            self.xr_diag_blocks[n_i] = (
                self.xr_diag_blocks[n_i]
                - a.local_block[n_i - 1, n_i] @ temp_lower @ self.xr_diag_blocks[n_i]
            )

            # Streaming/Sparsifying back to DSBSparse
            a.local_block[n_i, n_i] = self.xr_diag_blocks[n_i]

    def _upward_sellinv_inv(
        self,
        a: DSBSparse,
        stack_slice: slice,
    ):
        if self.n_diag_local > 1:
            # If there is only a single diagonal block, we don't need these buffers.
            # TODO, THis doesn't work in the case of non-uniform block sizes
            temp_upper = xp.empty_like(a.local_block[0, 1])

        for n_i in range(1, self.n_diag_local):
            temp_upper[:, :] = a.local_block[n_i - 1, n_i]

            a.local_block[n_i, n_i - 1] = (
                -self.xr_diag_blocks[n_i]
                @ a.local_block[n_i, n_i - 1]
                @ self.xr_diag_blocks[n_i - 1]
            )

            a.local_block[n_i - 1, n_i] = (
                -self.xr_diag_blocks[n_i - 1]
                @ a.local_block[n_i - 1, n_i]
                @ self.xr_diag_blocks[n_i]
            )

            self.xr_diag_blocks[n_i] = (
                self.xr_diag_blocks[n_i]
                - a.local_block[n_i, n_i - 1] @ temp_upper @ self.xr_diag_blocks[n_i]
            )

            # Streaming/Sparsifying back to DSBSparse
            a.local_block[n_i, n_i] = self.xr_diag_blocks[n_i]

    def _permuted_sellinv_inv(
        self,
        a: DSBSparse,
        stack_slice: slice,
    ):
        # TODO, THis doesn't work in the case of non-uniform block sizes
        B1 = xp.empty_like(a.local_block[1, 0])
        B2 = xp.empty_like(self.buffer_upper[0])

        C1 = xp.empty_like(a.local_block[0, 1])
        C2 = xp.empty_like(self.buffer_lower[0])

        D1 = xp.empty_like(a.local_block[1, 0])
        D2 = xp.empty_like(self.buffer_lower[0])

        for n_i in range(self.n_diag_local - 2, 0, -1):
            B1[:, :] = (
                a.local_block[n_i - 1, n_i] @ self.xr_diag_blocks[n_i + 1]
                + self.buffer_upper[n_i - 1] @ self.buffer_lower[n_i]
            )

            B2[:, :] = (
                a.local_block[n_i - 1, n_i] @ self.buffer_upper[n_i]
                + self.buffer_upper[n_i - 1] @ a.local_block[0, 0]
            )

            C1[:, :] = (
                self.xr_diag_blocks[n_i + 1] @ a.local_block[n_i, n_i - 1]
                + self.buffer_upper[n_i] @ self.buffer_lower[n_i - 1]
            )

            C2[:, :] = (
                self.buffer_lower[n_i] @ a.local_block[n_i, n_i - 1]
                + a.local_block[0, 0] @ self.buffer_lower[n_i - 1]
            )

            a.local_block[n_i - 1, n_i] = -self.xr_diag_blocks[n_i] @ B1[:, :]
            self.buffer_upper[n_i - 1] = -self.xr_diag_blocks[n_i] @ B2[:, :]

            D1[:, :] = a.local_block[n_i, n_i - 1]
            D2[:, :] = self.buffer_lower[n_i - 1]

            a.local_block[n_i, n_i - 1] = -C1[:, :] @ self.xr_diag_blocks[n_i]
            self.buffer_lower[n_i - 1] = -C2[:, :] @ self.xr_diag_blocks[n_i]

            self.xr_diag_blocks[n_i] = (
                self.xr_diag_blocks[n_i]
                + self.xr_diag_blocks[n_i]
                @ (B1[:, :] @ D1[:, :] + B2[:, :] @ D2[:, :])
                @ self.xr_diag_blocks[n_i]
            )

            # Streaming/Sparsifying back to DSBSparse
            a.local_block[n_i, n_i] = self.xr_diag_blocks[n_i]

        a.local_block[1, 0] = self.buffer_upper[0]
        a.local_block[0, 1] = self.buffer_lower[0]
