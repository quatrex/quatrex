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

    def __init__(
        self, comm: MPI.Comm, solve_lesser: bool = False, solve_greater: bool = False
    ) -> None:
        """Initializes the reduced system."""
        self.num_diags = 2 * (comm.size - 1)

        self.diag_blocks: list[NDArray | None] = [None] * self.num_diags
        self.upper_blocks: list[NDArray | None] = [None] * self.num_diags
        self.lower_blocks: list[NDArray | None] = [None] * self.num_diags

        self.solve_lesser = solve_lesser
        if self.solve_lesser:
            self.diag_blocks_lesser: list[NDArray | None] = [None] * self.num_diags
            self.upper_blocks_lesser: list[NDArray | None] = [None] * self.num_diags
            self.lower_blocks_lesser: list[NDArray | None] = [None] * self.num_diags

        self.solve_greater = solve_greater
        if self.solve_greater:
            self.diag_blocks_greater: list[NDArray | None] = [None] * self.num_diags
            self.upper_blocks_greater: list[NDArray | None] = [None] * self.num_diags
            self.lower_blocks_greater: list[NDArray | None] = [None] * self.num_diags

        self.comm = comm

    def gather(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        bl_buffer_upper: list[NDArray] = None,
        bl_buffer_lower: list[NDArray] = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
        bg_buffer_upper: list[NDArray] = None,
        bg_buffer_lower: list[NDArray] = None,
    ):
        """Gathers the reduced system across all ranks."""

        (diag_blocks, upper_blocks, lower_blocks) = self._map_reduced_system(
            a, x_diag_blocks, buffer_upper, buffer_lower
        )
        if self.solve_lesser:
            (diag_blocks_lesser, upper_blocks_lesser, lower_blocks_lesser) = (
                self._map_reduced_system(
                    bl, xl_diag_blocks, bl_buffer_upper, bl_buffer_lower
                )
            )
        if self.solve_greater:
            (diag_blocks_greater, upper_blocks_greater, lower_blocks_greater) = (
                self._map_reduced_system(
                    bg, xg_diag_blocks, bg_buffer_upper, bg_buffer_lower
                )
            )

        self.diag_blocks = _flatten_list(self.comm.allgather(diag_blocks))
        self.upper_blocks = _flatten_list(self.comm.allgather(upper_blocks))
        self.lower_blocks = _flatten_list(self.comm.allgather(lower_blocks))
        if self.solve_lesser:
            self.diag_blocks_lesser = _flatten_list(
                self.comm.allgather(diag_blocks_lesser)
            )
            self.upper_blocks_lesser = _flatten_list(
                self.comm.allgather(upper_blocks_lesser)
            )
            self.lower_blocks_lesser = _flatten_list(
                self.comm.allgather(lower_blocks_lesser)
            )
        if self.solve_greater:
            self.diag_blocks_greater = _flatten_list(
                self.comm.allgather(diag_blocks_greater)
            )
            self.upper_blocks_greater = _flatten_list(
                self.comm.allgather(upper_blocks_greater)
            )
            self.lower_blocks_greater = _flatten_list(
                self.comm.allgather(lower_blocks_greater)
            )

    def _map_reduced_system(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
    ):
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

        return diag_blocks, upper_blocks, lower_blocks

    def solve(self):
        """Solves the reduced system on all ranks."""

        # Forwards pass.
        for i in range(self.num_diags - 1):
            # Inverse the curent block
            self.diag_blocks[i] = xp.linalg.inv(self.diag_blocks[i])
            if self.solve_lesser:
                self.diag_blocks_lesser[i] = (
                    self.diag_blocks[i]
                    @ self.diag_blocks_lesser[i]
                    @ self.diag_blocks[i].T
                )
            if self.solve_greater:
                self.diag_blocks_greater[i] = (
                    self.diag_blocks[i]
                    @ self.diag_blocks_greater[i]
                    @ self.diag_blocks[i].T
                )

            # Update the next diagonal block
            temp_1 = self.lower_blocks[i] @ self.diag_blocks[i]
            if self.solve_lesser or self.solve_greater:
                temp_2 = self.diag_blocks[i].T @ self.lower_blocks[i].T

            self.diag_blocks[i + 1] = (
                self.diag_blocks[i + 1] - temp_1 @ self.upper_blocks[i]
            )
            if self.solve_lesser:
                self.diag_blocks_lesser[i + 1] = (
                    self.diag_blocks_lesser[i + 1]
                    + self.lower_blocks[i]
                    @ self.diag_blocks_lesser[i]
                    @ self.lower_blocks[i].T
                    - self.lower_blocks_lesser[i] @ temp_2
                    - temp_1 @ self.upper_blocks_lesser[i]
                )
            if self.solve_greater:
                self.diag_blocks_greater[i + 1] = (
                    self.diag_blocks_greater[i + 1]
                    + self.lower_blocks[i]
                    @ self.diag_blocks_greater[i]
                    @ self.lower_blocks[i].T
                    - self.lower_blocks_greater[i] @ temp_2
                    - temp_1 @ self.upper_blocks_greater[i]
                )

        # Invert the last diagonal block.
        self.diag_blocks[-1] = xp.linalg.inv(self.diag_blocks[-1])
        if self.solve_lesser:
            self.diag_blocks_lesser[-1] = (
                self.diag_blocks[-1]
                @ self.diag_blocks_lesser[-1]
                @ self.diag_blocks[-1].T
            )
        if self.solve_greater:
            self.diag_blocks_greater[-1] = (
                self.diag_blocks[-1]
                @ self.diag_blocks_greater[-1]
                @ self.diag_blocks[-1].T
            )

        # Backwards pass.
        for i in range(self.num_diags - 2, -1, -1):
            temp_1 = self.diag_blocks[i] @ self.upper_blocks[i]
            temp_3 = self.diag_blocks[i + 1] @ self.lower_blocks[i]

            if self.solve_lesser:
                temp_upper_lesser = self.upper_blocks_lesser[i]
                temp_4 = self.lower_blocks[i].T @ self.diag_blocks[i + 1].T
                self.upper_blocks_lesser[i] = (
                    -temp_1 @ self.diag_blocks_lesser[i + 1]
                    - self.diag_blocks_lesser[i] @ temp_4
                    + self.diag_blocks[i]
                    @ self.upper_blocks_lesser[i]
                    @ self.diag_blocks[i + 1].T
                )

                temp_lower_lesser = self.lower_blocks_lesser[i]
                temp_2 = self.upper_blocks[i].T @ self.diag_blocks[i].T
                self.lower_blocks_lesser[i] = (
                    -self.diag_blocks_lesser[i + 1] @ temp_2
                    - temp_3 @ self.diag_blocks_lesser[i]
                    + self.diag_blocks[i + 1]
                    @ self.lower_blocks_lesser[i]
                    @ self.diag_blocks[i].T
                )

                self.diag_blocks_lesser[i] = (
                    self.diag_blocks_lesser[i]
                    + temp_1 @ self.diag_blocks_lesser[i + 1] @ temp_2
                    + temp_1 @ temp_3 @ self.diag_blocks_lesser[i]
                    + self.diag_blocks_lesser[i].T @ temp_4 @ temp_2
                    - temp_1
                    @ self.diag_blocks[i + 1]
                    @ temp_lower_lesser
                    @ self.diag_blocks[i].T
                    - self.diag_blocks[i]
                    @ temp_upper_lesser
                    @ self.diag_blocks[i + 1].T
                    @ temp_2
                )

            if self.solve_greater:
                temp_upper_greater = self.upper_blocks_greater[i]
                temp_4 = self.lower_blocks[i].T @ self.diag_blocks[i + 1].T
                self.upper_blocks_greater[i] = (
                    -temp_1 @ self.diag_blocks_greater[i + 1]
                    - self.diag_blocks_greater[i] @ temp_4
                    + self.diag_blocks[i]
                    @ self.upper_blocks_greater[i]
                    @ self.diag_blocks[i + 1].T
                )

                temp_lower_greater = self.lower_blocks_greater[i]
                temp_2 = self.upper_blocks[i].T @ self.diag_blocks[i].T
                self.lower_blocks_greater[i] = (
                    -self.diag_blocks_greater[i + 1] @ temp_2
                    - temp_3 @ self.diag_blocks_greater[i]
                    + self.diag_blocks[i + 1]
                    @ self.lower_blocks_greater[i]
                    @ self.diag_blocks[i].T
                )

                self.diag_blocks_greater[i] = (
                    self.diag_blocks_greater[i]
                    + temp_1 @ self.diag_blocks_greater[i + 1] @ temp_2
                    + temp_1 @ temp_3 @ self.diag_blocks_greater[i]
                    + self.diag_blocks_greater[i].T @ temp_4 @ temp_2
                    - temp_1
                    @ self.diag_blocks[i + 1]
                    @ temp_lower_greater
                    @ self.diag_blocks[i].T
                    - self.diag_blocks[i]
                    @ temp_upper_greater
                    @ self.diag_blocks[i + 1].T
                    @ temp_2
                )

            temp_lower = self.lower_blocks[i]
            self.lower_blocks[i] = -temp_3 @ self.diag_blocks[i]
            self.upper_blocks[i] = -temp_1 @ self.diag_blocks[i + 1]
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
        xl_diag_blocks: list[NDArray] = None,
        xl_buffer_lower: list[NDArray] = None,
        xl_buffer_upper: list[NDArray] = None,
        xl_out: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_buffer_lower: list[NDArray] = None,
        xg_buffer_upper: list[NDArray] = None,
        xg_out: DBSparse = None,
    ):
        self._mapback_reduced_system(
            x_diag_blocks,
            buffer_upper,
            buffer_lower,
            out,
            self.diag_blocks,
            self.upper_blocks,
            self.lower_blocks,
        )

        if self.solve_lesser:
            self._mapback_reduced_system(
                xl_diag_blocks,
                xl_buffer_upper,
                xl_buffer_lower,
                xl_out,
                self.diag_blocks_lesser,
                self.upper_blocks_lesser,
                self.lower_blocks_lesser,
            )
        if self.solve_greater:
            self._mapback_reduced_system(
                xg_diag_blocks,
                xg_buffer_upper,
                xg_buffer_lower,
                xg_out,
                self.diag_blocks_greater,
                self.upper_blocks_greater,
                self.lower_blocks_greater,
            )

    def _mapback_reduced_system(
        self,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
        out: DBSparse,
        diag_block_reduced_system: list[NDArray],
        upper_block_reduced_system: list[NDArray],
        lower_block_reduced_system: list[NDArray],
    ):
        i = out.num_local_blocks - 1
        j = i + 1
        if self.comm.rank == 0:
            x_diag_blocks[-1] = diag_block_reduced_system[0]
            out.local_blocks[i, i] = diag_block_reduced_system[0]

            out.local_blocks[j, i] = lower_block_reduced_system[0]
            out.local_blocks[i, j] = upper_block_reduced_system[0]
        elif self.comm.rank == self.comm.size - 1:
            x_diag_blocks[0] = diag_block_reduced_system[-1]
            out.local_blocks[0, 0] = diag_block_reduced_system[-1]
        else:
            x_diag_blocks[0] = diag_block_reduced_system[2 * self.comm.rank - 1]
            x_diag_blocks[-1] = diag_block_reduced_system[2 * self.comm.rank]
            out.local_blocks[0, 0] = x_diag_blocks[0]
            out.local_blocks[i, i] = x_diag_blocks[-1]

            buffer_upper[-2] = lower_block_reduced_system[2 * self.comm.rank - 1]
            buffer_lower[-2] = upper_block_reduced_system[2 * self.comm.rank - 1]

            out.local_blocks[j, i] = lower_block_reduced_system[2 * self.comm.rank]
            out.local_blocks[i, j] = upper_block_reduced_system[2 * self.comm.rank]


class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        100.

    """

    def __init__(
        self,
        solve_lesser: bool = False,
        solve_greater: bool = False,
        max_batch_size: int = 100,
    ) -> None:
        """Initializes the selected inversion solver."""
        self.solve_lesser = solve_lesser
        self.solve_greater = solve_greater
        self.max_batch_size = max_batch_size

    def _downward_schur(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        invert_last_block: bool,
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
    ):
        x_diag_blocks[0] = a.local_blocks[0, 0]
        if self.solve_lesser:
            xl_diag_blocks[0] = bl.local_blocks[0, 0]
        if self.solve_greater:
            xg_diag_blocks[0] = bg.local_blocks[0, 0]

        for i in range(a.num_local_blocks - 1):
            j = i + 1
            x_diag_blocks[i] = xp.linalg.inv(x_diag_blocks[i])
            if self.solve_lesser:
                xl_diag_blocks[i] = (
                    x_diag_blocks[i] @ xl_diag_blocks[i] @ x_diag_blocks[i].T
                )
            if self.solve_greater:
                xg_diag_blocks[i] = (
                    x_diag_blocks[i] @ xg_diag_blocks[i] @ x_diag_blocks[i].T
                )

            temp_1 = a.local_blocks[j, i] @ x_diag_blocks[i]
            if self.solve_lesser or self.solve_greater:
                temp_2 = x_diag_blocks[i].T @ a.local_blocks[j, i].T
            if self.solve_lesser:
                xl_diag_blocks[j] = (
                    bl.local_blocks[j, j]
                    + a.local_blocks[j, i] @ xl_diag_blocks[i] @ a.local_blocks[j, i].T
                    - bl.local_blocks[j, i] @ temp_2
                    - temp_1 @ bl.local_blocks[i, j]
                )
            if self.solve_greater:
                xg_diag_blocks[j] = (
                    bg.local_blocks[j, j]
                    + a.local_blocks[j, i] @ xg_diag_blocks[i] @ a.local_blocks[j, i].T
                    - bl.local_blocks[j, i] @ temp_2
                    - temp_1 @ bl.local_blocks[i, j]
                )
            x_diag_blocks[j] = a.local_blocks[j, j] - temp_1 @ a.local_blocks[i, j]

        if invert_last_block:
            x_diag_blocks[-1] = xp.linalg.inv(x_diag_blocks[-1])
            if self.solve_lesser:
                xl_diag_blocks[-1] = (
                    x_diag_blocks[-1] @ xl_diag_blocks[-1] @ x_diag_blocks[-1].T
                )
            if self.solve_greater:
                xg_diag_blocks[-1] = (
                    x_diag_blocks[-1] @ xg_diag_blocks[-1] @ x_diag_blocks[-1].T
                )

    def _upward_schur(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        invert_last_block: bool,
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
    ):
        x_diag_blocks[-1] = a.local_blocks[-1, -1]
        if self.solve_lesser:
            xl_diag_blocks[-1] = bl.local_blocks[-1, -1]
        if self.solve_greater:
            xg_diag_blocks[-1] = bg.local_blocks[-1, -1]

        for i in range(a.num_local_blocks - 1, 0, -1):
            j = i - 1
            x_diag_blocks[i] = xp.linalg.inv(x_diag_blocks[i])
            if self.solve_lesser:
                xl_diag_blocks[i] = (
                    x_diag_blocks[i] @ xl_diag_blocks[i] @ x_diag_blocks[i].T
                )
            if self.solve_greater:
                xg_diag_blocks[i] = (
                    x_diag_blocks[i] @ xg_diag_blocks[i] @ x_diag_blocks[i].T
                )

            temp_1 = a.local_blocks[j, i] @ x_diag_blocks[i]
            if self.solve_lesser or self.solve_greater:
                temp_2 = x_diag_blocks[i].T @ a.local_blocks[j, i].T
            if self.solve_lesser:
                xl_diag_blocks[j] = (
                    bl.local_blocks[j, j]
                    + a.local_blocks[j, i] @ xl_diag_blocks[i] @ a.local_blocks[j, i].T
                    - bl.local_blocks[j, i] @ temp_2
                    - temp_1 @ bl.local_blocks[i, j]
                )
            if self.solve_greater:
                xg_diag_blocks[j] = (
                    bg.local_blocks[j, j]
                    + a.local_blocks[j, i] @ xg_diag_blocks[i] @ a.local_blocks[j, i].T
                    - bg.local_blocks[j, i] @ temp_2
                    - temp_1 @ bg.local_blocks[i, j]
                )
            x_diag_blocks[j] = a.local_blocks[j, j] - temp_1 @ a.local_blocks[i, j]

        if invert_last_block:
            x_diag_blocks[0] = xp.linalg.inv(x_diag_blocks[0])
            if self.solve_lesser:
                xl_diag_blocks[0] = (
                    x_diag_blocks[0] @ xl_diag_blocks[0] @ x_diag_blocks[0].T
                )
            if self.solve_greater:
                xg_diag_blocks[0] = (
                    x_diag_blocks[0] @ xg_diag_blocks[0] @ x_diag_blocks[0].T
                )

    def _permuted_schur(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_lower: list[NDArray],
        buffer_upper: list[NDArray],
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        bl_buffer_lower: list[NDArray] = None,
        bl_buffer_upper: list[NDArray] = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
        bg_buffer_lower: list[NDArray] = None,
        bg_buffer_upper: list[NDArray] = None,
    ):
        buffer_lower[0] = a.local_blocks[0, 1]
        buffer_upper[0] = a.local_blocks[1, 0]
        x_diag_blocks[0] = a.local_blocks[0, 0]
        x_diag_blocks[1] = a.local_blocks[1, 1]
        if self.solve_lesser:
            bl_buffer_lower[0] = bl.local_blocks[0, 1]
            bl_buffer_upper[0] = bl.local_blocks[1, 0]
            xl_diag_blocks[0] = bl.local_blocks[0, 0]
            xl_diag_blocks[1] = bl.local_blocks[1, 1]
        if self.solve_greater:
            bg_buffer_lower[0] = bg.local_blocks[0, 1]
            bg_buffer_upper[0] = bg.local_blocks[1, 0]
            xg_diag_blocks[0] = bg.local_blocks[0, 0]
            xg_diag_blocks[1] = bg.local_blocks[1, 1]

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

            if self.solve_lesser:
                xl_diag_blocks[i] = (
                    x_diag_blocks[i] @ xl_diag_blocks[i] @ x_diag_blocks[i].T
                )
                xl_diag_blocks[i + 1] = (
                    bl.local_blocks[i + 1, i + 1]
                    + a.local_blocks[i + 1, i]
                    @ xl_diag_blocks[i]
                    @ a.local_blocks[i + 1, i].T
                    - bl.local_blocks[i + 1, i][i]
                    @ x_diag_blocks[i].T
                    @ a.local_blocks[i + 1, i].T
                    - a.local_blocks[i + 1, i]
                    @ x_diag_blocks[i]
                    @ bl.local_blocks[i, i + 1]
                )
                bl_buffer_upper[i] = (
                    a.local_blocks[i + 1, i] @ xl_diag_blocks[i] @ buffer_lower[i - 1].T
                    - bl.local_blocks[i + 1, i][i]
                    @ x_diag_blocks[i].T
                    @ buffer_lower[i - 1].T
                    - a.local_blocks[i + 1, i]
                    @ x_diag_blocks[i]
                    @ bl_buffer_upper[i - 1]
                )
                bl_buffer_lower[i] = (
                    buffer_lower[i - 1] @ xl_diag_blocks[i] @ a.local_blocks[i + 1, i].T
                    - bl_buffer_lower[i - 1]
                    @ x_diag_blocks[i].T
                    @ a.local_blocks[i + 1, i].T
                    - buffer_lower[i - 1] @ x_diag_blocks[i] @ bl.local_blocks[i, i + 1]
                )
                xl_diag_blocks[0] = (
                    xl_diag_blocks[0]
                    + buffer_lower[i - 1] @ xl_diag_blocks[i] @ buffer_lower[i - 1].T
                    - bl_buffer_lower[i - 1]
                    @ x_diag_blocks[i].T
                    @ buffer_lower[i - 1].T
                    - buffer_lower[i - 1] @ x_diag_blocks[i] @ bl_buffer_upper[i - 1]
                )

            if self.solve_greater:
                xg_diag_blocks[i] = (
                    x_diag_blocks[i] @ xg_diag_blocks[i] @ x_diag_blocks[i].T
                )
                xg_diag_blocks[i + 1] = (
                    bg.local_blocks[i + 1, i + 1]
                    + a.local_blocks[i + 1, i]
                    @ xg_diag_blocks[i]
                    @ a.local_blocks[i + 1, i].T
                    - bl.local_blocks[i + 1, i][i]
                    @ x_diag_blocks[i].T
                    @ a.local_blocks[i + 1, i].T
                    - a.local_blocks[i + 1, i]
                    @ x_diag_blocks[i]
                    @ bl.local_blocks[i, i + 1]
                )
                bg_buffer_upper[i] = (
                    a.local_blocks[i + 1, i] @ xg_diag_blocks[i] @ buffer_lower[i - 1].T
                    - bl.local_blocks[i + 1, i][i]
                    @ x_diag_blocks[i].T
                    @ buffer_lower[i - 1].T
                    - a.local_blocks[i + 1, i]
                    @ x_diag_blocks[i]
                    @ bg_buffer_upper[i - 1]
                )
                bg_buffer_lower[i] = (
                    buffer_lower[i - 1] @ xg_diag_blocks[i] @ a.local_blocks[i + 1, i].T
                    - bg_buffer_lower[i - 1]
                    @ x_diag_blocks[i].T
                    @ a.local_blocks[i + 1, i].T
                    - buffer_lower[i - 1] @ x_diag_blocks[i] @ bl.local_blocks[i, i + 1]
                )
                xg_diag_blocks[0] = (
                    xg_diag_blocks[0]
                    + buffer_lower[i - 1] @ xg_diag_blocks[i] @ buffer_lower[i - 1].T
                    - bg_buffer_lower[i - 1]
                    @ x_diag_blocks[i].T
                    @ buffer_lower[i - 1].T
                    - buffer_lower[i - 1] @ x_diag_blocks[i] @ bg_buffer_upper[i - 1]
                )

    def _downward_selinv(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        out: DBSparse,
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        xl_out: DBSparse = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_out: DBSparse = None,
    ):

        for i in range(a.num_local_blocks - 2, -1, -1):
            j = i + 1

            temp_1 = x_diag_blocks[i] @ a.local_blocks[i, j]
            temp_3 = x_diag_blocks[j] @ a.local_blocks[j, i]

            if self.solve_lesser:
                temp_4 = a.local_blocks[j, i].T @ x_diag_blocks[j].T
                xl_upper_block = (
                    -temp_1 @ xl_diag_blocks[j]
                    - xl_diag_blocks[i] @ temp_4
                    + x_diag_blocks[i] @ bl.local_blocks[i, j] @ x_diag_blocks[j].T
                )

                temp_2 = a.local_blocks[i, j].T @ x_diag_blocks[i].T
                xl_lower_block = (
                    -xl_diag_blocks[j] @ temp_2
                    - temp_3 @ xl_diag_blocks[i]
                    + x_diag_blocks[j] @ bl.local_blocks[j, i] @ x_diag_blocks[i].T
                )

                xl_diag_blocks[i] = (
                    xl_diag_blocks[i]
                    + temp_1 @ xl_diag_blocks[i + 1] @ temp_2
                    + temp_1 @ temp_3 @ xl_diag_blocks[i]
                    + xl_diag_blocks[i].T @ temp_4 @ temp_2
                    - temp_1
                    @ x_diag_blocks[i + 1]
                    @ bl.local_blocks[j, i]
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ bl.local_blocks[i, j]
                    @ x_diag_blocks[i + 1].T
                    @ temp_2
                )
                # Streaming/Sparsifying back to DDSBSparse
                xl_out.local_blocks[j, i] = xl_lower_block
                xl_out.local_blocks[i, j] = xl_upper_block
                xl_out.local_blocks[i, i] = xl_diag_blocks[i]

            if self.solve_greater:
                temp_4 = a.local_blocks[j, i].T @ x_diag_blocks[i + 1].T
                xg_upper_block = (
                    -temp_1 @ xg_diag_blocks[i + 1]
                    - xg_diag_blocks[i] @ temp_4
                    + x_diag_blocks[i] @ bg.local_blocks[i, j] @ x_diag_blocks[i + 1].T
                )

                temp_2 = a.local_blocks[i, j].T @ x_diag_blocks[i].T
                xg_lower_block = (
                    -xg_diag_blocks[i + 1] @ temp_2
                    - temp_3 @ xg_diag_blocks[i]
                    + x_diag_blocks[i + 1] @ bg.local_blocks[j, i] @ x_diag_blocks[i].T
                )

                xg_diag_blocks[i] = (
                    xg_diag_blocks[i]
                    + temp_1 @ xg_diag_blocks[i + 1] @ temp_2
                    + temp_1 @ temp_3 @ xg_diag_blocks[i]
                    + xg_diag_blocks[i].T @ temp_4 @ temp_2
                    - temp_1
                    @ x_diag_blocks[i + 1]
                    @ bg.local_blocks[j, i]
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ bg.local_blocks[i, j]
                    @ x_diag_blocks[i + 1].T
                    @ temp_2
                )
                # Streaming/Sparsifying back to DDSBSparse
                xg_out.local_blocks[j, i] = xg_lower_block
                xg_out.local_blocks[i, j] = xg_upper_block
                xg_out.local_blocks[i, i] = xg_diag_blocks[i]

            x_lower_block = -temp_3 @ x_diag_blocks[i]
            x_upper_block = -temp_1 @ x_diag_blocks[j]
            x_diag_blocks[i] = (
                x_diag_blocks[i]
                - x_upper_block @ a.local_blocks[j, i] @ x_diag_blocks[i]
            )
            # # Streaming/Sparsifying back to DDSBSparse
            out.local_blocks[j, i] = x_lower_block
            out.local_blocks[i, j] = x_upper_block
            out.local_blocks[i, i] = x_diag_blocks[i]

    def _upward_selinv(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        out: DBSparse,
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        xl_out: DBSparse = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_out: DBSparse = None,
    ):

        for i in range(1, a.num_local_blocks):
            j = i - 1
            temp_1 = x_diag_blocks[j] @ a.local_blocks[j, i]
            temp_3 = x_diag_blocks[i] @ a.local_blocks[i, j]

            if self.solve_lesser:
                temp_4 = a.local_blocks[i, j].T @ x_diag_blocks[i].T
                xl_upper_block = (
                    -temp_1 @ xl_diag_blocks[i]
                    - xl_diag_blocks[j] @ temp_4
                    + x_diag_blocks[j] @ bl.local_blocks[j, i] @ x_diag_blocks[i].T
                )

                temp_2 = a.local_blocks[j, i].T @ x_diag_blocks[j].T
                xl_lower_block = (
                    -xl_diag_blocks[i] @ temp_2
                    - temp_3 @ xl_diag_blocks[j]
                    + x_diag_blocks[i] @ bl.local_blocks[i, j] @ x_diag_blocks[j].T
                )

                xl_diag_blocks[i] = (
                    xl_diag_blocks[i]
                    + temp_3 @ xl_diag_blocks[j] @ temp_4
                    + temp_3 @ temp_1 @ xl_diag_blocks[i]
                    + xl_diag_blocks[i].T @ temp_2 @ temp_4
                    - temp_3
                    @ x_diag_blocks[j]
                    @ bl.local_blocks[j, i]
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ bl.local_blocks[i, j]
                    @ x_diag_blocks[j].T
                    @ temp_4
                )
                # Streaming/Sparsifying back to DDSBSparse
                xl_out.local_blocks[j, i] = xl_upper_block
                xl_out.local_blocks[i, j] = xl_lower_block
                xl_out.local_blocks[i, i] = xl_diag_blocks[i]

            if self.solve_greater:
                temp_4 = a.local_blocks[i, j].T @ x_diag_blocks[i].T
                xg_upper_block = (
                    -temp_1 @ xg_diag_blocks[i]
                    - xg_diag_blocks[j] @ temp_4
                    + x_diag_blocks[j] @ bg.local_blocks[j, i] @ x_diag_blocks[i].T
                )

                temp_2 = a.local_blocks[j, i].T @ x_diag_blocks[j].T
                xg_lower_block = (
                    -xg_diag_blocks[i] @ temp_2
                    - temp_3 @ xg_diag_blocks[j]
                    + x_diag_blocks[i] @ bg.local_blocks[i, j] @ x_diag_blocks[j].T
                )

                xg_diag_blocks[i] = (
                    xg_diag_blocks[i]
                    + temp_3 @ xg_diag_blocks[j] @ temp_4
                    + temp_3 @ temp_1 @ xg_diag_blocks[i]
                    + xg_diag_blocks[i].T @ temp_2 @ temp_4
                    - temp_3
                    @ x_diag_blocks[j]
                    @ bg.local_blocks[j, i]
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ bg.local_blocks[i, j]
                    @ x_diag_blocks[j].T
                    @ temp_4
                )
                # Streaming/Sparsifying back to DDSBSparse
                xg_out.local_blocks[j, i] = xg_upper_block
                xg_out.local_blocks[i, j] = xg_lower_block
                xg_out.local_blocks[i, i] = xg_diag_blocks[i]

            x_upper_block = -temp_1 @ x_diag_blocks[i]
            x_lower_block = -temp_3 @ x_diag_blocks[j]
            x_diag_blocks[i] = (
                x_diag_blocks[i]
                - x_lower_block @ a.local_blocks[j, i] @ x_diag_blocks[i]
            )
            # Streaming/Sparsifying back to DDSBSparse
            out.local_blocks[j, i] = x_upper_block
            out.local_blocks[i, j] = x_lower_block
            out.local_blocks[i, i] = x_diag_blocks[i]

    def _permuted_selinv(
        self,
        a: DBSparse,
        x_diag_blocks: list[NDArray],
        buffer_lower: list[NDArray],
        buffer_upper: list[NDArray],
        out: DBSparse,
        bl: DBSparse = None,
        xl_diag_blocks: list[NDArray] = None,
        xl_buffer_lower: list[NDArray] = None,
        xl_buffer_upper: list[NDArray] = None,
        xl_out: DBSparse = None,
        bg: DBSparse = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_buffer_lower: list[NDArray] = None,
        xg_buffer_upper: list[NDArray] = None,
        xg_out: DBSparse = None,
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

            if self.solve_lesser:
                temp_B_13 = xl_buffer_upper[i - 1]
                temp_B_31 = xl_buffer_lower[i - 1]

                bl_upper_block = (
                    -x_diag_blocks[i]
                    @ (
                        a.local_blocks[i, i + 1] @ xl_diag_blocks[i + 1]
                        + buffer_upper[i - 1] @ xl_buffer_lower[i]
                    )
                    - xl_diag_blocks[i]
                    @ (
                        a.local_blocks[i + 1, i].T @ x_diag_blocks[i + 1].T
                        + buffer_lower[i - 1].T @ buffer_upper[i].T
                    )
                    + x_diag_blocks[i]
                    @ (
                        bl.local_blocks[i, i + 1] @ x_diag_blocks[i + 1].T
                        + xl_buffer_upper[i - 1] @ buffer_upper[i].T
                    )
                )
                xl_buffer_upper[i - 1] = (
                    -x_diag_blocks[i]
                    @ (
                        a.local_blocks[i, i + 1] @ xl_buffer_upper[i]
                        + buffer_upper[i - 1] @ xl_diag_blocks[0]
                    )
                    - xl_diag_blocks[i]
                    @ (
                        a.local_blocks[i + 1, i].T @ buffer_lower[i].T
                        + buffer_lower[i - 1].T @ x_diag_blocks[0].T
                    )
                    + x_diag_blocks[i]
                    @ (
                        bl.local_blocks[i, i + 1] @ buffer_lower[i].T
                        + xl_buffer_upper[i - 1] @ x_diag_blocks[0].T
                    )
                )

                bl_lower_block = (
                    -(
                        xl_diag_blocks[i + 1] @ a.local_blocks[i, i + 1].T
                        + xl_buffer_upper[i] @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    - (C1) @ xl_diag_blocks[i]
                    + (
                        x_diag_blocks[i + 1] @ bl.local_blocks[i + 1, i]
                        + buffer_upper[i] @ xl_buffer_lower[i - 1]
                    )
                    @ x_diag_blocks[i].T
                )
                xl_buffer_lower[i - 1] = (
                    -(
                        xl_buffer_lower[i] @ a.local_blocks[i, i + 1].T
                        + xl_diag_blocks[0] @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    - (C2) @ xl_diag_blocks[i]
                    + (
                        buffer_lower[i] @ bl.local_blocks[i + 1, i]
                        + x_diag_blocks[0] @ xl_buffer_lower[i - 1]
                    )
                    @ x_diag_blocks[i].T
                )

                xl_diag_blocks[i] = (
                    xl_diag_blocks[i]
                    + x_diag_blocks[i]
                    @ (
                        (
                            a.local_blocks[i, i + 1] @ xl_diag_blocks[i + 1]
                            + buffer_upper[i - 1] @ xl_buffer_lower[i]
                        )
                        @ a.local_blocks[i, i + 1].T
                        + (
                            a.local_blocks[i, i + 1] @ xl_buffer_upper[i]
                            + buffer_upper[i - 1] @ xl_diag_blocks[0]
                        )
                        @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    + x_diag_blocks[i]
                    @ ((B1) @ a.local_blocks[i + 1, i] + (B2) @ buffer_lower[i - 1])
                    @ xl_diag_blocks[i]
                    + xl_diag_blocks[i].T
                    @ (
                        (
                            a.local_blocks[i + 1, i].T @ x_diag_blocks[i + 1].T
                            + buffer_lower[i - 1].T @ buffer_upper[i].T
                        )
                        @ a.local_blocks[i, i + 1].T
                        + (
                            a.local_blocks[i + 1, i].T @ buffer_lower[i].T
                            + buffer_lower[i - 1].T @ x_diag_blocks[0].T
                        )
                        @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ ((B1) @ bl.local_blocks[i + 1, i] + (B2) @ temp_B_31)
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ (
                        (
                            bl.local_blocks[i, i + 1] @ x_diag_blocks[i + 1].T
                            + temp_B_13 @ buffer_upper[i].T
                        )
                        @ a.local_blocks[i, i + 1].T
                        + (
                            bl.local_blocks[i, i + 1] @ buffer_lower[i].T
                            + temp_B_13 @ x_diag_blocks[0].T
                        )
                        @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                )
                # Streaming/Sparsifying back to DDSBSparse
                xl_out.local_blocks[i + 1, i] = bl_lower_block
                xl_out.local_blocks[i, i + 1] = bl_upper_block
                xl_out.local_blocks[i, i] = xl_diag_blocks[i]

            if self.solve_greater:
                temp_B_13 = xg_buffer_upper[i - 1]
                temp_B_31 = xg_buffer_lower[i - 1]

                bg_upper_block = (
                    -x_diag_blocks[i]
                    @ (
                        a.local_blocks[i, i + 1] @ xg_diag_blocks[i + 1]
                        + buffer_upper[i - 1] @ xg_buffer_lower[i]
                    )
                    - xg_diag_blocks[i]
                    @ (
                        a.local_blocks[i + 1, i].T @ x_diag_blocks[i + 1].T
                        + buffer_lower[i - 1].T @ buffer_upper[i].T
                    )
                    + x_diag_blocks[i]
                    @ (
                        bg.local_blocks[i, i + 1] @ x_diag_blocks[i + 1].T
                        + xg_buffer_upper[i - 1] @ buffer_upper[i].T
                    )
                )
                xg_buffer_upper[i - 1] = (
                    -x_diag_blocks[i]
                    @ (
                        a.local_blocks[i, i + 1] @ xg_buffer_upper[i]
                        + buffer_upper[i - 1] @ xg_diag_blocks[0]
                    )
                    - xg_diag_blocks[i]
                    @ (
                        a.local_blocks[i + 1, i].T @ buffer_lower[i].T
                        + buffer_lower[i - 1].T @ x_diag_blocks[0].T
                    )
                    + x_diag_blocks[i]
                    @ (
                        bg.local_blocks[i, i + 1] @ buffer_lower[i].T
                        + xg_buffer_upper[i - 1] @ x_diag_blocks[0].T
                    )
                )

                bg_lower_block = (
                    -(
                        xg_diag_blocks[i + 1] @ a.local_blocks[i, i + 1].T
                        + xg_buffer_upper[i] @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    - (C1) @ xg_diag_blocks[i]
                    + (
                        x_diag_blocks[i + 1] @ bg.local_blocks[i + 1, i]
                        + buffer_upper[i] @ xg_buffer_lower[i - 1]
                    )
                    @ x_diag_blocks[i].T
                )
                xg_buffer_lower[i - 1] = (
                    -(
                        xg_buffer_lower[i] @ a.local_blocks[i, i + 1].T
                        + xg_diag_blocks[0] @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    - (C2) @ xg_diag_blocks[i]
                    + (
                        buffer_lower[i] @ bg.local_blocks[i + 1, i]
                        + x_diag_blocks[0] @ xg_buffer_lower[i - 1]
                    )
                    @ x_diag_blocks[i].T
                )

                xg_diag_blocks[i] = (
                    xg_diag_blocks[i]
                    + x_diag_blocks[i]
                    @ (
                        (
                            a.local_blocks[i, i + 1] @ xg_diag_blocks[i + 1]
                            + buffer_upper[i - 1] @ xg_buffer_lower[i]
                        )
                        @ a.local_blocks[i, i + 1].T
                        + (
                            a.local_blocks[i, i + 1] @ xg_buffer_upper[i]
                            + buffer_upper[i - 1] @ xg_diag_blocks[0]
                        )
                        @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    + x_diag_blocks[i]
                    @ ((B1) @ a.local_blocks[i + 1, i] + (B2) @ buffer_lower[i - 1])
                    @ xg_diag_blocks[i]
                    + xg_diag_blocks[i].T
                    @ (
                        (
                            a.local_blocks[i + 1, i].T @ x_diag_blocks[i + 1].T
                            + buffer_lower[i - 1].T @ buffer_upper[i].T
                        )
                        @ a.local_blocks[i, i + 1].T
                        + (
                            a.local_blocks[i + 1, i].T @ buffer_lower[i].T
                            + buffer_lower[i - 1].T @ x_diag_blocks[0].T
                        )
                        @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ ((B1) @ bg.local_blocks[i + 1, i] + (B2) @ temp_B_31)
                    @ x_diag_blocks[i].T
                    - x_diag_blocks[i]
                    @ (
                        (
                            bg.local_blocks[i, i + 1] @ x_diag_blocks[i + 1].T
                            + temp_B_13 @ buffer_upper[i].T
                        )
                        @ a.local_blocks[i, i + 1].T
                        + (
                            bg.local_blocks[i, i + 1] @ buffer_lower[i].T
                            + temp_B_13 @ x_diag_blocks[0].T
                        )
                        @ buffer_upper[i - 1].T
                    )
                    @ x_diag_blocks[i].T
                )
                # Streaming/Sparsifying back to DDSBSparse
                xg_out.local_blocks[i + 1, i] = bg_lower_block
                xg_out.local_blocks[i, i + 1] = bg_upper_block
                xg_out.local_blocks[i, i] = xg_diag_blocks[i]

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
        if self.solve_lesser:
            bl.local_blocks[1, 0] = xl_buffer_upper[0]
            bl.local_blocks[0, 1] = xl_buffer_lower[0]
        if self.solve_greater:
            bg.local_blocks[1, 0] = xg_buffer_upper[0]
            bg.local_blocks[0, 1] = xg_buffer_lower[0]

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

        x_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

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
        a: DBSparse,
        out: DBSparse,
        bl: DBSparse = None,
        xl_out: DBSparse = None,
        bg: DBSparse = None,
        xg_out: DBSparse = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        if bl is None and bg is None:
            return self.selected_inv(a, out, comm)

        # Initialize temporary buffers.
        reduced_system = ReducedSystem(
            comm=comm, solve_lesser=bl is not None, solve_greater=bg is not None
        )

        x_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks
        if reduced_system.solve_lesser:
            xl_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
            bl_buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
            bl_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks
        else:
            xl_diag_blocks = None
            bl_buffer_lower = None
            bl_buffer_upper = None
        if reduced_system.solve_greater:
            xg_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
            bg_buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
            bg_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks
        else:
            xg_diag_blocks = None
            bg_buffer_lower = None
            bg_buffer_upper = None

        if comm.rank == 0:
            # Direction: downward Schur-complement
            self._downward_schur(
                a=a,
                x_diag_blocks=x_diag_blocks,
                invert_last_block=False,
                bl=bl,
                xl_diag_blocks=xl_diag_blocks,
                bg=bg,
                xg_diag_blocks=xg_diag_blocks,
            )
        elif comm.rank == comm.size - 1:
            # Direction: upward Schur-complement
            self._upward_schur(
                a=a,
                x_diag_blocks=x_diag_blocks,
                invert_last_block=False,
                bl=bl,
                xl_diag_blocks=xl_diag_blocks,
                bg=bg,
                xg_diag_blocks=xg_diag_blocks,
            )
        else:
            # Permuted Schur-complement
            self._permuted_schur(
                a=a,
                x_diag_blocks=x_diag_blocks,
                buffer_lower=buffer_lower,
                buffer_upper=buffer_upper,
                bl=bl,
                xl_diag_blocks=xl_diag_blocks,
                bl_buffer_lower=bl_buffer_lower,
                bl_buffer_upper=bl_buffer_upper,
                bg=bg,
                xg_diag_blocks=xg_diag_blocks,
                bg_buffer_lower=bg_buffer_lower,
                bg_buffer_upper=bg_buffer_upper,
            )

        # Construct the reduced system.
        reduced_system.gather(
            a=a,
            x_diag_blocks=x_diag_blocks,
            buffer_lower=buffer_lower,
            buffer_upper=buffer_upper,
            bl=bl,
            xl_diag_blocks=xl_diag_blocks,
            bl_buffer_lower=bl_buffer_lower,
            bl_buffer_upper=bl_buffer_upper,
            bg=bg,
            xg_diag_blocks=xg_diag_blocks,
            bg_buffer_lower=bg_buffer_lower,
            bg_buffer_upper=bg_buffer_upper,
        )
        # Perform selected-inversion on the reduced system.
        reduced_system.solve()
        # Scatter the result to the output matrix.
        reduced_system.scatter(
            x_diag_blocks=x_diag_blocks,
            buffer_lower=buffer_lower,
            buffer_upper=buffer_upper,
            out=out,
            xl_diag_blocks=xl_diag_blocks,
            xl_buffer_lower=bl_buffer_lower,
            xl_buffer_upper=bl_buffer_upper,
            xl_out=xl_out,
            xg_diag_blocks=xg_diag_blocks,
            xg_buffer_lower=bg_buffer_lower,
            xg_buffer_upper=bg_buffer_upper,
            xg_out=xg_out,
        )

        if comm.rank == 0:
            # Direction: upward sell-inv
            self._downward_selinv(
                a=a,
                x_diag_blocks=x_diag_blocks,
                out=out,
                bl=bl,
                xl_diag_blocks=xl_diag_blocks,
                xl_out=xl_out,
                bg=bg,
                xg_diag_blocks=xg_diag_blocks,
                xg_out=xg_out,
            )
        elif comm.rank == comm.size - 1:
            # Direction: downward sell-inv
            self._upward_selinv(
                a=a,
                x_diag_blocks=x_diag_blocks,
                out=out,
                bl=bl,
                xl_diag_blocks=xl_diag_blocks,
                xl_out=xl_out,
                bg=bg,
                xg_diag_blocks=xg_diag_blocks,
                xg_out=xg_out,
            )
        else:
            # Permuted Sell-inv
            self._permuted_selinv(
                a=a,
                x_diag_blocks=x_diag_blocks,
                buffer_lower=buffer_lower,
                buffer_upper=buffer_upper,
                out=out,
                bl=bl,
                xl_diag_blocks=xl_diag_blocks,
                xl_buffer_lower=bl_buffer_lower,
                xl_buffer_upper=bl_buffer_upper,
                xl_out=xl_out,
                bg=bg,
                xg_diag_blocks=xg_diag_blocks,
                xg_buffer_lower=bg_buffer_lower,
                xg_buffer_upper=bg_buffer_upper,
                xg_out=xg_out,
            )
