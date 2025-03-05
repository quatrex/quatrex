# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import itertools

from mpi4py import MPI

from qttools import NDArray, xp
from qttools.datastructures.dbsparse import DBSparse
from qttools.greens_function_solver.solver import GFSolver


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


class ReducedSystem:
    """Auxiliary class to handle the reduced system.

    This is basically a container for the diagonal and off-diagonal
    blocks of the reduced system with some extra functionality.

    Parameters
    ----------
    comm : MPI.Comm
        The intranode MPI communicator.

    Attributes
    ----------
    diag_blocks : list[NDArray | None]
        The diagonal blocks of the reduced system.
    upper_blocks : list[NDArray | None]
        The upper off-diagonal blocks of the reduced system.
    lower_blocks : list[NDArray | None]
        The lower off-diagonal blocks of the reduced system

    """

    def __init__(self, comm: MPI.Comm) -> None:
        """Initializes the reduced system."""
        self.num_diags = 2 * (comm.size - 1)

        self.diag_blocks: list[NDArray | None] = [None] * self.num_diags
        self.upper_blocks: list[NDArray | None] = [None] * self.num_diags
        self.lower_blocks: list[NDArray | None] = [None] * self.num_diags

        self.comm = comm

    def gather(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
    ):
        """Gathers the reduced system across all ranks."""
        i = a.num_local_blocks - 1
        j = i + 1

        diag_blocks = []
        upper_blocks = []
        lower_blocks = []
        if self.comm.rank == 0:
            diag_blocks.append(x_diag_blocks[-1])
            lower_blocks.append(a.local_blocks[j, i])
            upper_blocks.append(a.local_blocks[i, j])
        elif self.comm.rank == self.comm.size - 1:
            diag_blocks.append(x_diag_blocks[0])
        else:
            diag_blocks.append(x_diag_blocks[0])
            diag_blocks.append(x_diag_blocks[-1])

            lower_blocks.append(buffer_upper[-2])
            lower_blocks.append(a.local_blocks[j, i])

            upper_blocks.append(buffer_lower[-2])
            upper_blocks.append(a.local_blocks[i, j])

        self.diag_blocks = _flatten_list(self.comm.allgather(diag_blocks))
        self.upper_blocks = _flatten_list(self.comm.allgather(upper_blocks))
        self.lower_blocks = _flatten_list(self.comm.allgather(lower_blocks))

    def solve(self):
        """Solves the reduced system on all ranks."""

        # Forwards pass.
        for i in range(self.num_diags - 1):
            self.diag_blocks[i] = xp.linalg.inv(self.diag_blocks[i])

            self.diag_blocks[i + 1] = (
                self.diag_blocks[i + 1]
                - self.lower_blocks[i] @ self.diag_blocks[i] @ self.upper_blocks[i]
            )

        # Invert the last diagonal block.
        self.diag_blocks[-1] = xp.linalg.inv(self.diag_blocks[-1])

        # Backwards pass.
        for i in range(self.num_diags - 2, -1, -1):
            temp_lower = self.lower_blocks[i]

            self.lower_blocks[i] = (
                -self.diag_blocks[i + 1] @ self.lower_blocks[i] @ self.diag_blocks[i]
            )

            self.upper_blocks[i] = (
                -self.diag_blocks[i] @ self.upper_blocks[i] @ self.diag_blocks[i + 1]
            )

            self.diag_blocks[i] = (
                self.diag_blocks[i]
                - self.upper_blocks[i] @ temp_lower @ self.diag_blocks[i]
            )

    def scatter(
        self,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
        out: DBSparse,
    ):
        i = out.num_local_blocks - 1
        j = i + 1
        if self.comm.rank == 0:
            x_diag_blocks[-1] = self.diag_blocks[0]
            out.local_blocks[i, i] = self.diag_blocks[0]

            out.local_blocks[j, i] = self.lower_blocks[0]
            out.local_blocks[i, j] = self.upper_blocks[0]
        elif self.comm.rank == self.comm.size - 1:
            x_diag_blocks[0] = self.diag_blocks[-1]
            out.local_blocks[0, 0] = self.diag_blocks[-1]
        else:
            x_diag_blocks[0] = self.diag_blocks[2 * self.comm.rank - 1]
            x_diag_blocks[-1] = self.diag_blocks[2 * self.comm.rank]
            out.local_blocks[0, 0] = x_diag_blocks[0]
            out.local_blocks[i, i] = x_diag_blocks[-1]

            buffer_upper[-2] = self.lower_blocks[2 * self.comm.rank - 1]
            buffer_lower[-2] = self.upper_blocks[2 * self.comm.rank - 1]

            out.local_blocks[j, i] = self.lower_blocks[2 * self.comm.rank]
            out.local_blocks[i, j] = self.upper_blocks[2 * self.comm.rank]


class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        100.

    """

    def __init__(self, max_batch_size: int = 100) -> None:
        """Initializes the selected inversion solver."""
        self.max_batch_size = max_batch_size

    def _downward_schur(
        self, a: DBSparse, x_diag_blocks: list[NDArray], invert_last_block: bool
    ):
        x_diag_blocks[0] = a.local_blocks[0, 0]
        for i in range(a.num_local_blocks - 1):
            j = i + 1
            x_diag_blocks[i] = xp.linalg.inv(x_diag_blocks[i])

            x_diag_blocks[j] = (
                a.local_blocks[j, j]
                - a.local_blocks[j, i] @ x_diag_blocks[i] @ a.local_blocks[i, j]
            )

        if invert_last_block:
            x_diag_blocks[-1] = xp.linalg.inv(x_diag_blocks[-1])

    def _upward_schur(
        self, a: DBSparse, x_diag_blocks: list[NDArray], invert_last_block: bool
    ):
        x_diag_blocks[-1] = a.local_blocks[-1, -1]
        for i in range(a.num_local_blocks - 1, 0, -1):
            j = i - 1
            x_diag_blocks[i] = xp.linalg.inv(x_diag_blocks[i])

            x_diag_blocks[j] = (
                a.local_blocks[j, j]
                - a.local_blocks[j, i] @ x_diag_blocks[i] @ a.local_blocks[i, j]
            )

        if invert_last_block:
            x_diag_blocks[0] = xp.linalg.inv(x_diag_blocks[0])

    def _permuted_schur(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_lower: list[NDArray],
        buffer_upper: list[NDArray],
    ):
        buffer_lower[0] = a.local_blocks[0, 1]
        buffer_upper[0] = a.local_blocks[1, 0]

        x_diag_blocks[0] = a.local_blocks[0, 0]
        x_diag_blocks[1] = a.local_blocks[1, 1]

        for i in range(1, a.num_local_blocks - 1):
            # Invert current diagonal block.
            x_diag_blocks[i] = xp.linalg.inv(x_diag_blocks[i])

            # Update next diagonal block.
            x_diag_blocks[i + 1] = (
                a.local_blocks[i + 1, i + 1]
                - a.local_blocks[i + 1, i] @ x_diag_blocks[i] @ a.local_blocks[i, i + 1]
            )

            # Update lower buffer block.
            buffer_lower[i] = (
                -buffer_lower[i - 1] @ x_diag_blocks[i] @ a.local_blocks[i, i + 1]
            )

            # Update upper buffer block.
            buffer_upper[i] = (
                -a.local_blocks[i + 1, i] @ x_diag_blocks[i] @ buffer_upper[i - 1]
            )

            # Update first block.
            x_diag_blocks[0] = (
                x_diag_blocks[0]
                - buffer_lower[i - 1] @ x_diag_blocks[i] @ buffer_upper[i - 1]
            )

    def _downward_selinv(
        self, a: DBSparse, x_diag_blocks: list[NDArray], out: DBSparse
    ):

        for i in range(a.num_local_blocks - 2, -1, -1):
            j = i + 1

            x_ji = -x_diag_blocks[j] @ a.local_blocks[j, i] @ x_diag_blocks[i]
            out.local_blocks[j, i] = x_ji

            out.local_blocks[i, j] = (
                -x_diag_blocks[i] @ a.local_blocks[i, j] @ x_diag_blocks[j]
            )
            x_diag_blocks[i] = (
                x_diag_blocks[i] - x_diag_blocks[i] @ a.local_blocks[i, j] @ x_ji
            )
            # Streaming/Sparsifying back to DDSBSparse
            out.local_blocks[i, i] = x_diag_blocks[i]

    def _upward_selinv(self, a: DBSparse, x_diag_blocks: list[NDArray], out: DBSparse):

        for i in range(1, a.num_local_blocks):
            j = i - 1
            x_ji = -x_diag_blocks[j] @ a.local_blocks[j, i] @ x_diag_blocks[i]
            out.local_blocks[j, i] = x_ji

            out.local_blocks[i, j] = (
                -x_diag_blocks[i] @ a.local_blocks[i, j] @ x_diag_blocks[j]
            )
            x_diag_blocks[i] = (
                x_diag_blocks[i] - x_diag_blocks[i] @ a.local_blocks[i, j] @ x_ji
            )
            # Streaming/Sparsifying back to DDSBSparse
            out.local_blocks[i, i] = x_diag_blocks[i]

    def _permuted_selinv(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_lower: list[NDArray],
        buffer_upper: list[NDArray],
        out: DBSparse,
    ):
        for i in range(a.num_local_blocks - 2, 0, -1):

            B1 = (
                a.local_blocks[i, i + 1] @ x_diag_blocks[i + 1]
                + buffer_upper[i - 1] @ buffer_lower[i]
            )

            B2 = (
                a.local_blocks[i, i + 1] @ buffer_upper[i]
                + buffer_upper[i - 1] @ x_diag_blocks[0]
            )

            C1 = (
                x_diag_blocks[i + 1] @ a.local_blocks[i + 1, i]
                + buffer_upper[i] @ buffer_lower[i - 1]
            )

            C2 = (
                buffer_lower[i] @ a.local_blocks[i + 1, i]
                + x_diag_blocks[0] @ buffer_lower[i - 1]
            )

            out.local_blocks[i, i + 1] = -x_diag_blocks[i] @ B1
            buffer_upper[i - 1] = -x_diag_blocks[i] @ B2

            D1 = a.local_blocks[i + 1, i]
            D2 = buffer_lower[i - 1]

            out.local_blocks[i + 1, i] = -C1 @ x_diag_blocks[i]
            buffer_lower[i - 1] = -C2 @ x_diag_blocks[i]

            x_diag_blocks[i] = (
                x_diag_blocks[i]
                + x_diag_blocks[i] @ (B1 @ D1 + B2 @ D2) @ x_diag_blocks[i]
            )

            # Streaming/Sparsifying back to DDSBSparse
            out.local_blocks[i, i] = x_diag_blocks[i]

        a.local_blocks[1, 0] = buffer_upper[0]
        a.local_blocks[0, 1] = buffer_lower[0]

    def selected_inv(
        self,
        a: DBSparse,
        out: DBSparse,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None | DBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DDSBSparse
            Matrix to invert.
        out : DDSBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DDSBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DDSBSparse object.

        """

        # Initialize temporary buffers.
        reduced_system = ReducedSystem(comm)

        buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        x_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks

        # NOTE: This section should be fixed.
        if comm.rank == 0:
            # Direction: downward Schur-complement
            self._downward_schur(a, x_diag_blocks, invert_last_block=False)
        elif comm.rank == comm.size - 1:
            # Direction: upward Schur-complement
            self._upward_schur(a, x_diag_blocks, invert_last_block=False)
        else:
            # Permuted Schur-complement
            self._permuted_schur(a, x_diag_blocks, buffer_lower, buffer_upper)

        # Construct the reduced system.
        reduced_system.gather(a, x_diag_blocks, buffer_upper, buffer_lower)
        # Perform selected-inversion on the reduced system.
        reduced_system.solve()
        # Scatter the result to the output matrix.
        reduced_system.scatter(x_diag_blocks, buffer_upper, buffer_lower, out)

        if comm.rank == 0:
            # Direction: upward sell-inv
            self._downward_selinv(a, x_diag_blocks, out)
        elif comm.rank == comm.size - 1:
            # Direction: downward sell-inv
            self._upward_selinv(a, x_diag_blocks, out)
        else:
            # Permuted Sell-inv
            self._permuted_selinv(a, x_diag_blocks, buffer_lower, buffer_upper, out)

    def selected_solve(
        self,
        a,
        sigma_lesser,
        sigma_greater,
        obc_blocks=None,
        out=None,
        return_retarded=False,
        return_current=False,
    ):
        # TODO: Implement selected_solve.
        ...
