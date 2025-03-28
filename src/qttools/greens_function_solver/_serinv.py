# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

"""
This backend implements the methods present in Serinv for the specific
case of solving lesser, greater, and retarded Green's functions.

For more information see https://github.com/vincent-maillou/serinv
"""

import functools
import itertools

from qttools import NDArray, block_comm, nccl_block_comm, xp
from qttools.datastructures.dsdbsparse import DSDBSparse, _DStackView
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.kernels import linalg


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
    solve_lesser : bool, optional
        Whether to solve the quadratic system associated with the lesser
        right-hand-side, by default False.
    solve_greater : bool, optional
        Whether to solve the quadratic system associated with the
        greater right-hand-side, by default False.

    Attributes
    ----------
    comm : MPI.Comm
        The intranode MPI communicator.
    num_diags : int
        The number of diagonal blocks in the reduced system.
    diag_blocks : list[NDArray | None]
        The diagonal blocks of the reduced system.
    upper_blocks : list[NDArray | None]
        The upper off-diagonal blocks of the reduced system.
    lower_blocks : list[NDArray | None]
        The lower off-diagonal blocks of the reduced system
    solve_lesser : bool
        Whether to solve the quadratic system associated with the lesser
        righ-hand-sideS.
    diag_blocks_lesser : list[NDArray | None]
        The diagonal blocks of the reduced system associated with the
        lesser right-hand-side.
    upper_blocks_lesser : list[NDArray | None]
        The upper off-diagonal blocks of the reduced system associated
        with the lesser right-hand-side.
    lower_blocks_lesser : list[NDArray | None]
        The lower off-diagonal blocks of the reduced system associated
        with the lesser right-hand-side.
    solve_greater : bool
        Whether to solve the quadratic system associated with the
        greater right-hand-side.
    diag_blocks_greater : list[NDArray | None]
        The diagonal blocks of the reduced system associated with the
        greater right-hand-side.
    upper_blocks_greater : list[NDArray | None]
        The upper off-diagonal blocks of the reduced system associated
        with the greater right-hand-side.
    lower_blocks_greater : list[NDArray | None]
        The lower off-diagonal blocks of the reduced system associated
        with the greater right-hand-side.

    """

    def __init__(self, selected_solve: bool = False) -> None:
        """Initializes the reduced system."""
        self.num_diags = 2 * (block_comm.size - 1)

        self.xr_diag_blocks: list[NDArray | None] = [None] * self.num_diags
        self.xr_upper_blocks: list[NDArray | None] = [None] * self.num_diags
        self.xr_lower_blocks: list[NDArray | None] = [None] * self.num_diags

        self.selected_solve = selected_solve
        if self.selected_solve:
            self.xl_diag_blocks: list[NDArray | None] = [None] * self.num_diags
            self.xl_upper_blocks: list[NDArray | None] = [None] * self.num_diags
            self.xl_lower_blocks: list[NDArray | None] = [None] * self.num_diags

            self.xg_diag_blocks: list[NDArray | None] = [None] * self.num_diags
            self.xg_upper_blocks: list[NDArray | None] = [None] * self.num_diags
            self.xg_lower_blocks: list[NDArray | None] = [None] * self.num_diags

    def gather(
        self,
        a: DSDBSparse | _DStackView,
        xr_diag_blocks: list[NDArray],
        xr_buffer_upper: list[NDArray],
        xr_buffer_lower: list[NDArray],
        sigma_lesser: DSDBSparse | _DStackView = None,
        xl_diag_blocks: list[NDArray] = None,
        xl_buffer_upper: list[NDArray] = None,
        xl_buffer_lower: list[NDArray] = None,
        sigma_greater: DSDBSparse | _DStackView = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_buffer_upper: list[NDArray] = None,
        xg_buffer_lower: list[NDArray] = None,
    ):
        """Gathers the reduced system across all ranks.

        Parameters
        ----------
        a : DSDBSparse
            The system matrix A in A X A^T = I/B.
        x_diag_blocks : list[NDArray]
            The diagonal blocks of the system matrix.
        buffer_upper : list[NDArray]
            The upper off-diagonal blocks of the system matrix.
        buffer_lower : list[NDArray]
            The lower off-diagonal blocks of the system matrix.
        bl : DSDBSparse, optional
            The system matrix Bl in A X Bl A^T = I/Bl, by default None.
        xl_diag_blocks : list[NDArray], optional
            The diagonal blocks of the system matrix Bl, by default None.
        bl_buffer_upper : list[NDArray], optional
            The upper off-diagonal blocks of the system matrix Bl, by default None.
        bl_buffer_lower : list[NDArray], optional
            The lower off-diagonal blocks of the system matrix Bl, by default None.
        bg : DSDBSparse, optional
            The system matrix Bg in A X Bg A^T = I/Bg, by default None.
        xg_diag_blocks : list[NDArray], optional
            The diagonal blocks of the system matrix Bg, by default None.
        bg_buffer_upper : list[NDArray], optional
            The upper off-diagonal blocks of the system matrix Bg, by default None.
        bg_buffer_lower : list[NDArray], optional
            The lower off-diagonal blocks of the system matrix Bg, by default None.
        """

        xr_diag_blocks, xr_upper_blocks, xr_lower_blocks = self._map_reduced_system(
            a, xr_diag_blocks, xr_buffer_upper, xr_buffer_lower
        )

        self.xr_diag_blocks = _flatten_list(block_comm.allgather(xr_diag_blocks))
        self.xr_upper_blocks = _flatten_list(block_comm.allgather(xr_upper_blocks))
        self.xr_lower_blocks = _flatten_list(block_comm.allgather(xr_lower_blocks))

        if self.selected_solve:
            xl_diag_blocks, xl_upper_blocks, xl_lower_blocks = self._map_reduced_system(
                sigma_lesser,
                xl_diag_blocks,
                xl_buffer_upper,
                xl_buffer_lower,
                is_retarded=False,
            )
            self.xl_diag_blocks = _flatten_list(block_comm.allgather(xl_diag_blocks))
            self.xl_upper_blocks = _flatten_list(block_comm.allgather(xl_upper_blocks))
            self.xl_lower_blocks = _flatten_list(block_comm.allgather(xl_lower_blocks))

            xg_diag_blocks, xg_upper_blocks, xg_lower_blocks = self._map_reduced_system(
                sigma_greater,
                xg_diag_blocks,
                xg_buffer_upper,
                xg_buffer_lower,
                is_retarded=False,
            )
            self.xg_diag_blocks = _flatten_list(block_comm.allgather(xg_diag_blocks))
            self.xg_upper_blocks = _flatten_list(block_comm.allgather(xg_upper_blocks))
            self.xg_lower_blocks = _flatten_list(block_comm.allgather(xg_lower_blocks))

    def gather_nccl(
        self,
        a: DSDBSparse | _DStackView,
        xr_diag_blocks: list[NDArray],
        xr_buffer_upper: list[NDArray],
        xr_buffer_lower: list[NDArray],
        sigma_lesser: DSDBSparse | _DStackView = None,
        xl_diag_blocks: list[NDArray] = None,
        xl_buffer_upper: list[NDArray] = None,
        xl_buffer_lower: list[NDArray] = None,
        sigma_greater: DSDBSparse | _DStackView = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_buffer_upper: list[NDArray] = None,
        xg_buffer_lower: list[NDArray] = None,
    ):
        """Gathers the reduced system across all ranks.

        Parameters
        ----------
        a : DSDBSparse
            The system matrix A in A X A^T = I/B.
        x_diag_blocks : list[NDArray]
            The diagonal blocks of the system matrix.
        buffer_upper : list[NDArray]
            The upper off-diagonal blocks of the system matrix.
        buffer_lower : list[NDArray]
            The lower off-diagonal blocks of the system matrix.
        bl : DSDBSparse, optional
            The system matrix Bl in A X Bl A^T = I/Bl, by default None.
        xl_diag_blocks : list[NDArray], optional
            The diagonal blocks of the system matrix Bl, by default None.
        bl_buffer_upper : list[NDArray], optional
            The upper off-diagonal blocks of the system matrix Bl, by default None.
        bl_buffer_lower : list[NDArray], optional
            The lower off-diagonal blocks of the system matrix Bl, by default None.
        bg : DSDBSparse, optional
            The system matrix Bg in A X Bg A^T = I/Bg, by default None.
        xg_diag_blocks : list[NDArray], optional
            The diagonal blocks of the system matrix Bg, by default None.
        bg_buffer_upper : list[NDArray], optional
            The upper off-diagonal blocks of the system matrix Bg, by default None.
        bg_buffer_lower : list[NDArray], optional
            The lower off-diagonal blocks of the system matrix Bg, by default None.
        """

        if isinstance(a, DSDBSparse):
            ssz = a._data.shape[:-1]
            bsz = a.block_sizes[0]
        else:
            ssz = a._block_indexer._arg.shape[:-1]
            bsz = a._dsdbsparse.block_sizes[0]
        count = functools.reduce(lambda x, y: x * y, ssz, 2 * bsz * bsz)

        xr_diag_blocks, xr_upper_blocks, xr_lower_blocks = (
            self._map_reduced_system_nccl(
                a, xr_diag_blocks, xr_buffer_upper, xr_buffer_lower
            )
        )

        # self.xr_diag_blocks = _flatten_list(block_comm.allgather(xr_diag_blocks))
        # self.xr_upper_blocks = _flatten_list(block_comm.allgather(xr_upper_blocks))
        # self.xr_lower_blocks = _flatten_list(block_comm.allgather(xr_lower_blocks))
        nccl_block_comm.all_gather(
            xr_diag_blocks[2 * block_comm.rank].reshape(-1),
            xr_diag_blocks.reshape(-1),
            count,
        )
        nccl_block_comm.all_gather(
            xr_upper_blocks[2 * block_comm.rank].reshape(-1),
            xr_upper_blocks.reshape(-1),
            count,
        )
        nccl_block_comm.all_gather(
            xr_lower_blocks[2 * block_comm.rank].reshape(-1),
            xr_lower_blocks.reshape(-1),
            count,
        )
        self.xr_diag_blocks = xr_diag_blocks[1:-1]
        self.xr_upper_blocks = xr_upper_blocks[1:-2]
        self.xr_lower_blocks = xr_lower_blocks[1:-2]

        if self.selected_solve:
            xl_diag_blocks, xl_upper_blocks, xl_lower_blocks = (
                self._map_reduced_system_nccl(
                    sigma_lesser,
                    xl_diag_blocks,
                    xl_buffer_upper,
                    xl_buffer_lower,
                    is_retarded=False,
                )
            )
            # self.xl_diag_blocks = _flatten_list(block_comm.allgather(xl_diag_blocks))
            # self.xl_upper_blocks = _flatten_list(block_comm.allgather(xl_upper_blocks))
            # self.xl_lower_blocks = _flatten_list(block_comm.allgather(xl_lower_blocks))
            nccl_block_comm.all_gather(
                xl_diag_blocks[2 * block_comm.rank].reshape(-1),
                xl_diag_blocks.reshape(-1),
                count,
            )
            nccl_block_comm.all_gather(
                xl_upper_blocks[2 * block_comm.rank].reshape(-1),
                xl_upper_blocks.reshape(-1),
                count,
            )
            nccl_block_comm.all_gather(
                xl_lower_blocks[2 * block_comm.rank].reshape(-1),
                xl_lower_blocks.reshape(-1),
                count,
            )
            self.xl_diag_blocks = xl_diag_blocks[1:-1]
            self.xl_upper_blocks = xl_upper_blocks[1:-2]
            self.xl_lower_blocks = xl_lower_blocks[1:-2]

            xg_diag_blocks, xg_upper_blocks, xg_lower_blocks = (
                self._map_reduced_system_nccl(
                    sigma_greater,
                    xg_diag_blocks,
                    xg_buffer_upper,
                    xg_buffer_lower,
                    is_retarded=False,
                )
            )
            # self.xg_diag_blocks = _flatten_list(block_comm.allgather(xg_diag_blocks))
            # self.xg_upper_blocks = _flatten_list(block_comm.allgather(xg_upper_blocks))
            # self.xg_lower_blocks = _flatten_list(block_comm.allgather(xg_lower_blocks))
            nccl_block_comm.all_gather(
                xg_diag_blocks[2 * block_comm.rank].reshape(-1),
                xg_diag_blocks.reshape(-1),
                count,
            )
            nccl_block_comm.all_gather(
                xg_upper_blocks[2 * block_comm.rank].reshape(-1),
                xg_upper_blocks.reshape(-1),
                count,
            )
            nccl_block_comm.all_gather(
                xg_lower_blocks[2 * block_comm.rank].reshape(-1),
                xg_lower_blocks.reshape(-1),
                count,
            )
            self.xg_diag_blocks = xg_diag_blocks[1:-1]
            self.xg_upper_blocks = xg_upper_blocks[1:-2]
            self.xg_lower_blocks = xg_lower_blocks[1:-2]

    def _map_reduced_system(
        self,
        a: DSDBSparse | _DStackView,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
        is_retarded: bool = True,
    ):
        """Maps the local partition to the reduced system.

        Parameters
        ----------
        a : DSDBSparse
            Local partition of the matrix to map.
        x_diag_blocks : list[NDArray]
            Local (densified) diagonal blocks of the matrix to map.
        buffer_upper : list[NDArray]
            Buffer blocks from the permutation of the matrix to map.
        buffer_lower : list[NDArray]
            Buffer blocks from the permutation of the matrix to map.

        """
        i = a.num_local_blocks - 1
        j = i + 1

        diag_blocks = []
        upper_blocks = []
        lower_blocks = []
        if block_comm.rank == 0:
            diag_blocks.append(x_diag_blocks[-1])
            lower_blocks.append(a.local_blocks[j, i])
            upper_blocks.append(a.local_blocks[i, j])
        elif block_comm.rank == block_comm.size - 1:
            diag_blocks.append(x_diag_blocks[0])
        else:
            diag_blocks.append(x_diag_blocks[0])
            diag_blocks.append(x_diag_blocks[-1])

            lower_blocks.append(buffer_upper[-2])
            lower_blocks.append(a.local_blocks[j, i])

            if is_retarded:
                upper_blocks.append(buffer_lower[-2])
            else:
                upper_blocks.append(-buffer_upper[-2].conj().swapaxes(-2, -1))
            upper_blocks.append(a.local_blocks[i, j])

        return diag_blocks, upper_blocks, lower_blocks

    def _map_reduced_system_nccl(
        self,
        a: DSDBSparse | _DStackView,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
        is_retarded: bool = True,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Maps the local partition to the reduced system.

        Parameters
        ----------
        a : DSDBSparse
            Local partition of the matrix to map.
        x_diag_blocks : list[NDArray]
            Local (densified) diagonal blocks of the matrix to map.
        buffer_upper : list[NDArray]
            Buffer blocks from the permutation of the matrix to map.
        buffer_lower : list[NDArray]
            Buffer blocks from the permutation of the matrix to map.

        """
        i = a.num_local_blocks - 1
        j = i + 1

        if isinstance(a, DSDBSparse):
            ssz = a._data.shape[:-1]
            bsz = a.block_sizes[0]
            dtype = a.dtype
        else:
            ssz = a._block_indexer._arg.shape[:-1]
            bsz = a._dsdbsparse.block_sizes[0]
            dtype = a._dsdbsparse.dtype

        diag_blocks = xp.empty((2 * block_comm.size, *ssz, bsz, bsz), dtype=dtype)
        upper_blocks = xp.empty((2 * block_comm.size, *ssz, bsz, bsz), dtype=dtype)
        lower_blocks = xp.empty((2 * block_comm.size, *ssz, bsz, bsz), dtype=dtype)

        if block_comm.rank == 0:
            # diag_blocks.append(x_diag_blocks[-1])
            diag_blocks[1] = x_diag_blocks[-1]
            # lower_blocks.append(a.local_blocks[j, i])
            lower_blocks[1] = a.local_blocks[j, i]
            # upper_blocks.append(a.local_blocks[i, j])
            upper_blocks[1] = a.local_blocks[i, j]
        elif block_comm.rank == block_comm.size - 1:
            # diag_blocks.append(x_diag_blocks[0])
            diag_blocks[-2] = x_diag_blocks[0]
        else:
            # diag_blocks.append(x_diag_blocks[0])
            # diag_blocks.append(x_diag_blocks[-1])
            diag_blocks[2 * block_comm.rank] = x_diag_blocks[0]
            diag_blocks[2 * block_comm.rank + 1] = x_diag_blocks[-1]

            # lower_blocks.append(buffer_upper[-2])
            # lower_blocks.append(a.local_blocks[j, i])
            lower_blocks[2 * block_comm.rank] = buffer_upper[-2]
            lower_blocks[2 * block_comm.rank + 1] = a.local_blocks[j, i]

            if is_retarded:
                # upper_blocks.append(buffer_lower[-2])
                upper_blocks[2 * block_comm.rank] = buffer_lower[-2]
            else:
                # upper_blocks.append(-buffer_upper[-2].conj().swapaxes(-2, -1))
                upper_blocks[2 * block_comm.rank] = (
                    -buffer_upper[-2].conj().swapaxes(-2, -1)
                )
            # upper_blocks.append(a.local_blocks[i, j])
            upper_blocks[2 * block_comm.rank + 1] = a.local_blocks[i, j]

        return diag_blocks, upper_blocks, lower_blocks

    def solve(self):
        """Solves the reduced system on all ranks."""

        # NOTE: I think for general cases, where OBC could be applied in
        # the middle of the device, there should probably be some
        # OBCBlocks here as well.

        # Forwards pass.
        for i in range(self.num_diags - 1):
            # Inverse the curent block
            self.xr_diag_blocks[i] = linalg.inv(self.xr_diag_blocks[i])
            xr_ii_dagger = self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
            if self.selected_solve:
                self.xl_diag_blocks[i] = (
                    self.xr_diag_blocks[i] @ self.xl_diag_blocks[i] @ xr_ii_dagger
                )

                self.xg_diag_blocks[i] = (
                    self.xr_diag_blocks[i] @ self.xg_diag_blocks[i] @ xr_ii_dagger
                )

            # Precompute some terms that are used multiple times.
            xr_ji_xr_ij = self.xr_lower_blocks[i] @ self.xr_diag_blocks[i]
            xr_ji = self.xr_lower_blocks[i]
            xr_ji_dagger = xr_ji.conj().swapaxes(-2, -1)
            if self.selected_solve:
                xr_ji_xr_ij_xl_ij = xr_ji_xr_ij @ self.xl_upper_blocks[i]
                xr_ji_xr_ij_xg_ij = xr_ji_xr_ij @ self.xg_upper_blocks[i]

            # Update the next diagonal block
            # temp_1 = self.xr_lower_blocks[i] @ self.xr_diag_blocks[i]
            # if self.selected_solve:
            #     temp_2 = self.xr_diag_blocks[i].conj().swapaxes(
            #         -2, -1
            #     ) @ self.xr_lower_blocks[i].conj().swapaxes(-2, -1)

            self.xr_diag_blocks[i + 1] = (
                self.xr_diag_blocks[i + 1] - xr_ji_xr_ij @ self.xr_upper_blocks[i]
            )
            if self.selected_solve:
                self.xl_diag_blocks[i + 1] = (
                    self.xl_diag_blocks[i + 1]
                    + xr_ji
                    # + self.xr_lower_blocks[i]
                    @ self.xl_diag_blocks[i] @ xr_ji_dagger
                    # @ self.xr_lower_blocks[i].conj().swapaxes(-2, -1)
                    + xr_ji_xr_ij_xl_ij.conj().swapaxes(-2, -1)
                    # - self.xl_lower_blocks[i] @ temp_2
                    - xr_ji_xr_ij_xl_ij
                    # - temp_1 @ self.xl_upper_blocks[i]
                )

                self.xg_diag_blocks[i + 1] = (
                    self.xg_diag_blocks[i + 1]
                    + xr_ji
                    # + self.xr_lower_blocks[i]
                    @ self.xg_diag_blocks[i] @ xr_ji_dagger
                    # @ self.xr_lower_blocks[i].conj().swapaxes(-2, -1)
                    + xr_ji_xr_ij_xg_ij.conj().swapaxes(-2, -1)
                    # - self.xg_lower_blocks[i] @ temp_2
                    - xr_ji_xr_ij_xg_ij
                    # - temp_1 @ self.xg_upper_blocks[i]
                )

        # Invert the last diagonal block.
        self.xr_diag_blocks[-1] = linalg.inv(self.xr_diag_blocks[-1])
        xr_ii_dagger = self.xr_diag_blocks[-1].conj().swapaxes(-2, -1)
        if self.selected_solve:
            self.xl_diag_blocks[-1] = (
                self.xr_diag_blocks[-1] @ self.xl_diag_blocks[-1] @ xr_ii_dagger
            )
            self.xg_diag_blocks[-1] = (
                self.xr_diag_blocks[-1] @ self.xg_diag_blocks[-1] @ xr_ii_dagger
            )
            self.xl_diag_blocks[-1] = 0.5 * (
                self.xl_diag_blocks[-1]
                - self.xl_diag_blocks[-1].conj().swapaxes(-2, -1)
            )
            self.xg_diag_blocks[-1] = 0.5 * (
                self.xg_diag_blocks[-1]
                - self.xg_diag_blocks[-1].conj().swapaxes(-2, -1)
            )

        # Backwards pass.
        for i in range(self.num_diags - 2, -1, -1):
            # j = i + 1

            # Get the blocks that are used multiple times.
            xr_ii = self.xr_diag_blocks[i]
            xr_jj = self.xr_diag_blocks[i + 1]
            xr_ij = self.xr_upper_blocks[i]
            xr_ji = self.xr_lower_blocks[i]
            xl_ii = self.xl_diag_blocks[i]
            xl_jj = self.xl_diag_blocks[i + 1]
            xg_ii = self.xg_diag_blocks[i]
            xg_jj = self.xg_diag_blocks[i + 1]
            tmp_xl_ij = self.xl_upper_blocks[i]
            tmp_xg_ij = self.xg_upper_blocks[i]

            # Precompute the transposes that are used multiple times.
            xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
            xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
            xr_ij_dagger = xr_ij.conj().swapaxes(-2, -1)

            # Precompute the terms that are used multiple times.
            xr_ji_dagger_xr_jj_dagger = xr_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
            xr_ij_dagger_xr_ii_dagger = xr_ij_dagger @ xr_ii_dagger
            xr_ii_xr_ij = xr_ii @ xr_ij
            xr_jj_xr_ji = xr_jj @ xr_ji
            xr_ii_xr_ij_xr_jj = xr_ii_xr_ij @ xr_jj
            xr_jj_dagger_xr_ij_dagger_xr_ii_dagger = xr_ii_xr_ij_xr_jj.conj().swapaxes(
                -2, -1
            )
            xr_ii_xr_ij_xr_jj_xr_ji = xr_ii_xr_ij @ xr_jj_xr_ji
            xr_ii_xr_ij_xl_jj = xr_ii_xr_ij @ xl_jj
            xr_ii_xr_ij_xg_jj = xr_ii_xr_ij @ xg_jj

            temp_1_l = xr_ii @ tmp_xl_ij @ xr_jj_dagger_xr_ij_dagger_xr_ii_dagger
            temp_1_l -= temp_1_l.conj().swapaxes(-2, -1)

            temp_1_g = xr_ii @ tmp_xg_ij @ xr_jj_dagger_xr_ij_dagger_xr_ii_dagger
            temp_1_g -= temp_1_g.conj().swapaxes(-2, -1)

            # temp_1 = self.xr_diag_blocks[i] @ self.xr_upper_blocks[i]
            # temp_3 = self.xr_diag_blocks[i + 1] @ self.xr_lower_blocks[i]

            if self.selected_solve:
                # temp_upper_lesser = self.xl_upper_blocks[i]
                # temp_4 = temp_3.conj().swapaxes(-2, -1)
                # temp_4 = self.xr_lower_blocks[i].conj().swapaxes(
                #     -2, -1
                # ) @ self.xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                self.xl_upper_blocks[i] = (
                    -xr_ii_xr_ij_xl_jj
                    - xl_ii @ xr_ji_dagger_xr_jj_dagger
                    + xr_ii @ tmp_xl_ij @ xr_jj_dagger
                )
                # self.xl_upper_blocks[i] = (
                #     - xr_ii_xr_ij_xl_jj
                #     # -temp_1 @ self.xl_diag_blocks[i + 1]
                #     - self.xl_diag_blocks[i] @ temp_4
                #     + self.xr_diag_blocks[i]
                #     @ self.xl_upper_blocks[i]
                #     @ self.xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                # )

                # temp_lower_lesser = -temp_upper_lesser.conj().swapaxes(-2, -1)
                # temp_lower_lesser = self.xl_lower_blocks[i]
                # temp_2 = temp_1.conj().swapaxes(-2, -1)
                # temp_2 = self.xr_upper_blocks[i].conj().swapaxes(
                #     -2, -1
                # ) @ self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
                self.xl_lower_blocks[i] = (
                    -self.xl_upper_blocks[i].conj().swapaxes(-2, -1)
                )
                # self.xl_lower_blocks[i] = (
                #     -self.xl_diag_blocks[i + 1] @ temp_2
                #     - temp_3 @ self.xl_diag_blocks[i]
                #     + self.xr_diag_blocks[i + 1]
                #     @ self.xl_lower_blocks[i]
                #     @ self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
                # )

                temp_2_l = xr_ii_xr_ij_xr_jj_xr_ji @ xl_ii
                self.xl_diag_blocks[i] = (
                    xl_ii
                    + xr_ii_xr_ij_xl_jj @ xr_ij_dagger_xr_ii_dagger
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
                )
                self.xl_diag_blocks[i] = 0.5 * (
                    self.xl_diag_blocks[i]
                    - self.xl_diag_blocks[i].conj().swapaxes(-2, -1)
                )
                # self.xl_diag_blocks[i] = (
                #     self.xl_diag_blocks[i]
                #     + temp_1 @ self.xl_diag_blocks[i + 1] @ temp_2
                #     + temp_1 @ temp_3 @ self.xl_diag_blocks[i]
                #     + self.xl_diag_blocks[i] @ temp_4 @ temp_2
                #     - temp_1
                #     @ self.xr_diag_blocks[i + 1]
                #     @ temp_lower_lesser
                #     @ self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
                #     - self.xr_diag_blocks[i]
                #     @ temp_upper_lesser
                #     @ self.xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                #     @ temp_2
                # )

                # temp_upper_greater = self.xg_upper_blocks[i]
                # temp_4 doesn't change
                # temp_4 = self.xr_lower_blocks[i].conj().swapaxes(
                #     -2, -1
                # ) @ self.xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                self.xg_upper_blocks[i] = (
                    -xr_ii_xr_ij_xg_jj
                    - xg_ii @ xr_ji_dagger_xr_jj_dagger
                    + xr_ii @ tmp_xg_ij @ xr_jj_dagger
                )
                # self.xg_upper_blocks[i] = (
                #     -temp_1 @ self.xg_diag_blocks[i + 1]
                #     - self.xg_diag_blocks[i] @ temp_4
                #     + self.xr_diag_blocks[i]
                #     @ self.xg_upper_blocks[i]
                #     @ self.xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                # )

                # temp_lower_greater = -temp_upper_greater.conj().swapaxes(-2, -1)
                # temp_lower_greater = self.xg_lower_blocks[i]
                # temp_2 doesn't change
                # temp_2 = self.xr_upper_blocks[i].conj().swapaxes(
                #     -2, -1
                # ) @ self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
                self.xg_lower_blocks[i] = (
                    -self.xg_upper_blocks[i].conj().swapaxes(-2, -1)
                )
                # self.xg_lower_blocks[i] = (
                #     -self.xg_diag_blocks[i + 1] @ temp_2
                #     - temp_3 @ self.xg_diag_blocks[i]
                #     + self.xr_diag_blocks[i + 1]
                #     @ self.xg_lower_blocks[i]
                #     @ self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
                # )

                temp_2_g = xr_ii_xr_ij_xr_jj_xr_ji @ xg_ii
                self.xg_diag_blocks[i] = (
                    xg_ii
                    + xr_ii_xr_ij_xg_jj @ xr_ij_dagger_xr_ii_dagger
                    - temp_1_g
                    + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
                )
                self.xg_diag_blocks[i] = 0.5 * (
                    self.xg_diag_blocks[i]
                    - self.xg_diag_blocks[i].conj().swapaxes(-2, -1)
                )
                # self.xg_diag_blocks[i] = (
                #     self.xg_diag_blocks[i]
                #     + temp_1 @ self.xg_diag_blocks[i + 1] @ temp_2
                #     + temp_1 @ temp_3 @ self.xg_diag_blocks[i]
                #     + self.xg_diag_blocks[i] @ temp_4 @ temp_2
                #     - temp_1
                #     @ self.xr_diag_blocks[i + 1]
                #     @ temp_lower_greater
                #     @ self.xr_diag_blocks[i].conj().swapaxes(-2, -1)
                #     - self.xr_diag_blocks[i]
                #     @ temp_upper_greater
                #     @ self.xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                #     @ temp_2
                # )

            # temp_lower = self.xr_lower_blocks[i]
            self.xr_lower_blocks[i] = -xr_jj_xr_ji @ self.xr_diag_blocks[i]
            self.xr_upper_blocks[i] = -xr_ii_xr_ij @ self.xr_diag_blocks[i + 1]
            # self.xr_diag_blocks[i] = (
            #     self.xr_diag_blocks[i]
            #     - self.xr_upper_blocks[i] @ temp_lower @ self.xr_diag_blocks[i]
            # )
            self.xr_diag_blocks[i] = xr_ii + xr_ii_xr_ij_xr_jj_xr_ji @ xr_ii

    def scatter(
        self,
        xr_diag_blocks: list[NDArray],
        xr_buffer_upper: list[NDArray],
        xr_buffer_lower: list[NDArray],
        xr_out: DSDBSparse | _DStackView,
        return_retarded: bool = True,
        xl_diag_blocks: list[NDArray] = None,
        xl_buffer_lower: list[NDArray] = None,
        xl_buffer_upper: list[NDArray] = None,
        xl_out: DSDBSparse | _DStackView = None,
        xg_diag_blocks: list[NDArray] = None,
        xg_buffer_lower: list[NDArray] = None,
        xg_buffer_upper: list[NDArray] = None,
        xg_out: DSDBSparse | _DStackView = None,
    ):
        """Scatters the reduced system across all ranks.

        Parameters
        ----------
        x_diag_blocks : list[NDArray]
            The diagonal blocks of the reduced system.
        buffer_upper : list[NDArray]
            The upper off-diagonal blocks of the reduced system.
        buffer_lower : list[NDArray]
            The lower off-diagonal blocks of the reduced system.
        out : DSDBSparse
            The distributed block-sparse matrix to scatter to.
        xl_diag_blocks : list[NDArray], optional
            The diagonal blocks of the reduced system associated with
            the lesser right-hand-side, by default None.
        xl_buffer_lower : list[NDArray], optional
            The lower off-diagonal blocks of the reduced system
            associated with the lesser right-hand-side, by default None.
        xl_buffer_upper : list[NDArray], optional
            The upper off-diagonal blocks of the reduced system
            associated with the lesser right-hand-side, by default None.
        xl_out : DSDBSparse, optional
            The distributed block-sparse matrix to scatter to associated
            with the lesser right-hand-side, by default None.
        xg_diag_blocks : list[NDArray], optional
            The diagonal blocks of the reduced system associated with
            the greater right-hand-side, by default None.
        xg_buffer_lower : list[NDArray], optional
            The lower off-diagonal blocks of the reduced system
            associated with the greater right-hand-side, by default
            None.
        xg_buffer_upper : list[NDArray], optional
            The upper off-diagonal blocks of the reduced system
            associated with the greater right-hand-side, by default
            None.
        xg_out : DSDBSparse, optional
            The distributed block-sparse matrix to scatter to associated
            with the greater right-hand-side, by default None.

        """
        self._mapback_reduced_system(
            x_diag_blocks=xr_diag_blocks,
            buffer_upper=xr_buffer_upper,
            buffer_lower=xr_buffer_lower,
            x_out=xr_out,
            write_x_out=return_retarded,
            diag_block_reduced_system=self.xr_diag_blocks,
            upper_block_reduced_system=self.xr_upper_blocks,
            lower_block_reduced_system=self.xr_lower_blocks,
        )

        if self.selected_solve:
            self._mapback_reduced_system(
                x_diag_blocks=xl_diag_blocks,
                buffer_upper=xl_buffer_upper,
                buffer_lower=xl_buffer_lower,
                x_out=xl_out,
                write_x_out=True,
                diag_block_reduced_system=self.xl_diag_blocks,
                upper_block_reduced_system=self.xl_upper_blocks,
                lower_block_reduced_system=self.xl_lower_blocks,
                is_retarded=False,
            )
            self._mapback_reduced_system(
                x_diag_blocks=xg_diag_blocks,
                buffer_upper=xg_buffer_upper,
                buffer_lower=xg_buffer_lower,
                x_out=xg_out,
                write_x_out=True,
                diag_block_reduced_system=self.xg_diag_blocks,
                upper_block_reduced_system=self.xg_upper_blocks,
                lower_block_reduced_system=self.xg_lower_blocks,
                is_retarded=False,
            )

    def _mapback_reduced_system(
        self,
        x_diag_blocks: list[NDArray],
        buffer_upper: list[NDArray],
        buffer_lower: list[NDArray],
        x_out: DSDBSparse | _DStackView,
        write_x_out: bool,
        diag_block_reduced_system: list[NDArray],
        upper_block_reduced_system: list[NDArray],
        lower_block_reduced_system: list[NDArray],
        is_retarded: bool = True,
    ):
        """Maps the reduced system back to the local partition.

        Parameters
        ----------
        x_diag_blocks : list[NDArray]
            Local (densified) diagonal blocks of the matrix to map.
        buffer_upper : list[NDArray]
            Buffer blocks from the permutation of the matrix to map.
        buffer_lower : list[NDArray]
            Buffer blocks from the permutation of the matrix to map.
        out : DSDBSparse
            Local partition of the matrix to map.
        diag_block_reduced_system : list[NDArray]
            The diagonal blocks of the reduced system.
        upper_block_reduced_system : list[NDArray]
            The upper off-diagonal blocks of the reduced system.
        lower_block_reduced_system : list[NDArray]
            The lower off-diagonal blocks of the reduced system.
        """
        if block_comm.rank == 0:
            x_diag_blocks[-1] = diag_block_reduced_system[0]
            if not write_x_out:
                return
            i = x_out.num_local_blocks - 1
            j = i + 1
            x_out.local_blocks[i, i] = diag_block_reduced_system[0]

            x_out.local_blocks[j, i] = lower_block_reduced_system[0]
            x_out.local_blocks[i, j] = upper_block_reduced_system[0]
        elif block_comm.rank == block_comm.size - 1:
            x_diag_blocks[0] = diag_block_reduced_system[-1]
            if not write_x_out:
                return

            x_out.local_blocks[0, 0] = diag_block_reduced_system[-1]
        else:
            x_diag_blocks[0] = diag_block_reduced_system[2 * block_comm.rank - 1]
            x_diag_blocks[-1] = diag_block_reduced_system[2 * block_comm.rank]

            buffer_upper[-2] = lower_block_reduced_system[2 * block_comm.rank - 1]
            if is_retarded:
                buffer_lower[-2] = upper_block_reduced_system[2 * block_comm.rank - 1]

            if not write_x_out:
                return

            i = x_out.num_local_blocks - 1
            j = i + 1
            x_out.local_blocks[0, 0] = x_diag_blocks[0]
            x_out.local_blocks[i, i] = x_diag_blocks[-1]

            x_out.local_blocks[j, i] = lower_block_reduced_system[2 * block_comm.rank]
            x_out.local_blocks[i, j] = upper_block_reduced_system[2 * block_comm.rank]


def downward_schur(
    a: DSDBSparse | _DStackView,
    xr_diag_blocks: list[NDArray],
    obc_blocks: OBCBlocks,
    sigma_lesser: DSDBSparse | _DStackView = None,
    xl_diag_blocks: list[NDArray] = None,
    sigma_greater: DSDBSparse | _DStackView = None,
    xg_diag_blocks: list[NDArray] = None,
    stack_slice: slice = Ellipsis,
    invert_last_block: bool = True,
    selected_solve: bool = False,
):
    """Performs the downward Schur complement decomposition."""
    obc_r = obc_blocks.retarded[0]
    a_00 = (
        a.local_blocks[0, 0]
        if obc_r is None
        else a.local_blocks[0, 0] - obc_r[stack_slice]
    )
    xr_diag_blocks[0] = a_00
    if selected_solve:
        obc_l = obc_blocks.lesser[0]
        sl_00 = (
            sigma_lesser.local_blocks[0, 0]
            if obc_l is None
            else sigma_lesser.local_blocks[0, 0] + obc_l[stack_slice]
        )
        obc_g = obc_blocks.greater[0]
        sg_00 = (
            sigma_greater.local_blocks[0, 0]
            if obc_g is None
            else sigma_greater.local_blocks[0, 0] + obc_g[stack_slice]
        )

        xl_diag_blocks[0] = sl_00
        xg_diag_blocks[0] = sg_00

    for i in range(a.num_local_blocks - 1):
        j = i + 1
        xr_diag_blocks[i] = linalg.inv(xr_diag_blocks[i])
        xr_ii_dagger = xr_diag_blocks[i].conj().swapaxes(-2, -1)
        if selected_solve:
            xl_diag_blocks[i] = xr_diag_blocks[i] @ xl_diag_blocks[i] @ xr_ii_dagger
            xg_diag_blocks[i] = xr_diag_blocks[i] @ xg_diag_blocks[i] @ xr_ii_dagger

        # Get the blocks that are used multiple times.
        a_ji = a.local_blocks[j, i]
        xr_ii = xr_diag_blocks[i]

        # Precompute the transposes that are used multiple times.
        a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

        # Precompute some terms that are used multiple times.
        a_ji_xr_ii = a_ji @ xr_ii
        # temp_1 = a.local_blocks[j, i] @ xr_diag_blocks[i]
        if selected_solve:
            a_ji_xr_ii_sl_ij = a_ji_xr_ii @ sigma_lesser.local_blocks[i, j]
            a_ji_xr_ii_sg_ij = a_ji_xr_ii @ sigma_greater.local_blocks[i, j]
            # temp_2 = xr_diag_blocks[i].conj().swapaxes(-2, -1) @ a.local_blocks[
            #     j, i
            # ].conj().swapaxes(-2, -1)

        obc_r = obc_blocks.retarded[j]
        a_jj = (
            a.local_blocks[j, j]
            if obc_r is None
            else a.local_blocks[j, j] - obc_r[stack_slice]
        )

        xr_diag_blocks[j] = a_jj - a_ji_xr_ii @ a.local_blocks[i, j]

        if selected_solve:
            obc_l = obc_blocks.lesser[j]
            sl_jj = (
                sigma_lesser.local_blocks[j, j]
                if obc_l is None
                else sigma_lesser.local_blocks[j, j] + obc_l[stack_slice]
            )
            obc_g = obc_blocks.greater[j]
            sg_jj = (
                sigma_greater.local_blocks[j, j]
                if obc_g is None
                else sigma_greater.local_blocks[j, j] + obc_g[stack_slice]
            )

            xl_diag_blocks[j] = (
                sl_jj
                + a_ji
                # + a.local_blocks[j, i]
                @ xl_diag_blocks[i] @ a_ji_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                # - sigma_lesser.local_blocks[j, i] @ temp_2
                - a_ji_xr_ii_sl_ij
                # - temp_1 @ sigma_lesser.local_blocks[i, j]
            )
            xg_diag_blocks[j] = (
                sg_jj
                + a_ji
                # + a.local_blocks[j, i]
                @ xg_diag_blocks[i] @ a_ji_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                # - sigma_greater.local_blocks[j, i] @ temp_2
                - a_ji_xr_ii_sg_ij
                # - temp_1 @ sigma_greater.local_blocks[i, j]
            )

    if invert_last_block:
        xr_diag_blocks[-1] = linalg.inv(xr_diag_blocks[-1])
        if selected_solve:
            xl_diag_blocks[-1] = (
                xr_diag_blocks[-1]
                @ xl_diag_blocks[-1]
                @ xr_diag_blocks[-1].conj().swapaxes(-2, -1)
            )

            xg_diag_blocks[-1] = (
                xr_diag_blocks[-1]
                @ xg_diag_blocks[-1]
                @ xr_diag_blocks[-1].conj().swapaxes(-2, -1)
            )


def upward_schur(
    a: DSDBSparse | _DStackView,
    xr_diag_blocks: list[NDArray],
    obc_blocks: OBCBlocks,
    sigma_lesser: DSDBSparse | _DStackView = None,
    xl_diag_blocks: list[NDArray] = None,
    sigma_greater: DSDBSparse | _DStackView = None,
    xg_diag_blocks: list[NDArray] = None,
    stack_slice: slice = Ellipsis,
    invert_last_block: bool = True,
    selected_solve: bool = False,
):
    """Performs the upward Schur complement decomposition."""
    n = a.num_local_blocks - 1

    obc_r = obc_blocks.retarded[n]
    a_nn = (
        a.local_blocks[n, n]
        if obc_r is None
        else a.local_blocks[n, n] - obc_r[stack_slice]
    )
    xr_diag_blocks[-1] = a_nn
    if selected_solve:
        obc_l = obc_blocks.lesser[n]
        sl_nn = (
            sigma_lesser.local_blocks[n, n]
            if obc_l is None
            else sigma_lesser.local_blocks[n, n] + obc_l[stack_slice]
        )
        obc_g = obc_blocks.greater[n]
        sg_nn = (
            sigma_greater.local_blocks[n, n]
            if obc_g is None
            else sigma_greater.local_blocks[n, n] + obc_g[stack_slice]
        )

        xl_diag_blocks[-1] = sl_nn
        xg_diag_blocks[-1] = sg_nn

    for i in range(n, 0, -1):
        j = i - 1
        xr_diag_blocks[i] = linalg.inv(xr_diag_blocks[i])
        xr_ii_dagger = xr_diag_blocks[i].conj().swapaxes(-2, -1)
        if selected_solve:
            xl_diag_blocks[i] = xr_diag_blocks[i] @ xl_diag_blocks[i] @ xr_ii_dagger
            xg_diag_blocks[i] = xr_diag_blocks[i] @ xg_diag_blocks[i] @ xr_ii_dagger

        # Get the blocks that are used multiple times.
        a_ji = a.local_blocks[j, i]
        xr_ii = xr_diag_blocks[i]

        # Precompute the transposes that are used multiple times.
        a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

        # Precompute some terms that are used multiple times.
        a_ji_xr_ii = a_ji @ xr_ii
        # temp_1 = a.local_blocks[j, i] @ xr_diag_blocks[i]
        if selected_solve:
            a_ji_xr_ii_sl_ij = a_ji_xr_ii @ sigma_lesser.local_blocks[i, j]
            a_ji_xr_ii_sg_ij = a_ji_xr_ii @ sigma_greater.local_blocks[i, j]
            # temp_2 = xr_diag_blocks[i].conj().swapaxes(-2, -1) @ a.local_blocks[
            #     j, i
            # ].conj().swapaxes(-2, -1)

        obc_r = obc_blocks.retarded[j]
        a_jj = (
            a.local_blocks[j, j]
            if obc_r is None
            else a.local_blocks[j, j] - obc_r[stack_slice]
        )

        xr_diag_blocks[j] = a_jj - a_ji_xr_ii @ a.local_blocks[i, j]

        if selected_solve:
            obc_l = obc_blocks.lesser[j]
            sl_jj = (
                sigma_lesser.local_blocks[j, j]
                if obc_l is None
                else sigma_lesser.local_blocks[j, j] + obc_l[stack_slice]
            )
            obc_g = obc_blocks.greater[j]
            sg_jj = (
                sigma_greater.local_blocks[j, j]
                if obc_g is None
                else sigma_greater.local_blocks[j, j] + obc_g[stack_slice]
            )

            xl_diag_blocks[j] = (
                sl_jj
                + a_ji
                # + a.local_blocks[j, i]
                @ xl_diag_blocks[i] @ a_ji_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                # - sigma_lesser.local_blocks[j, i] @ temp_2
                - a_ji_xr_ii_sl_ij
                # - temp_1 @ sigma_lesser.local_blocks[i, j]
            )
            xg_diag_blocks[j] = (
                sg_jj
                + a_ji
                # + a.local_blocks[j, i]
                @ xg_diag_blocks[i] @ a_ji_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                # - sigma_greater.local_blocks[j, i] @ temp_2
                - a_ji_xr_ii_sg_ij
                # - temp_1 @ sigma_greater.local_blocks[i, j]
            )

    if invert_last_block:
        xr_diag_blocks[0] = linalg.inv(xr_diag_blocks[0])
        if selected_solve:
            xl_diag_blocks[0] = (
                xr_diag_blocks[0]
                @ xl_diag_blocks[0]
                @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
            )
            xg_diag_blocks[0] = (
                xr_diag_blocks[0]
                @ xg_diag_blocks[0]
                @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
            )


def permuted_schur(
    a: DSDBSparse | _DStackView,
    xr_diag_blocks: list[NDArray],
    xr_buffer_lower: list[NDArray],
    xr_buffer_upper: list[NDArray],
    obc_blocks: OBCBlocks,
    sigma_lesser: DSDBSparse | _DStackView = None,
    xl_diag_blocks: list[NDArray] = None,
    xl_buffer_lower: list[NDArray] = None,
    xl_buffer_upper: list[NDArray] = None,
    sigma_greater: DSDBSparse | _DStackView = None,
    xg_diag_blocks: list[NDArray] = None,
    xg_buffer_lower: list[NDArray] = None,
    xg_buffer_upper: list[NDArray] = None,
    stack_slice: slice = Ellipsis,
    selected_solve: bool = False,
):
    """Performs the permuted Schur complement decomposition."""
    xr_buffer_lower[0] = a.local_blocks[0, 1]
    xr_buffer_upper[0] = a.local_blocks[1, 0]

    obc_r = obc_blocks.retarded[0]
    a_00 = (
        a.local_blocks[0, 0]
        if obc_r is None
        else a.local_blocks[0, 0] - obc_r[stack_slice]
    )
    xr_diag_blocks[0] = a_00

    obc_r = obc_blocks.retarded[1]
    a_11 = (
        a.local_blocks[1, 1]
        if obc_r is None
        else a.local_blocks[1, 1] - obc_r[stack_slice]
    )
    xr_diag_blocks[1] = a_11
    if selected_solve:
        # xl_buffer_lower[0] = sigma_lesser.local_blocks[0, 1]
        xl_buffer_upper[0] = sigma_lesser.local_blocks[1, 0]
        # xl_buffer_lower[0] = -xl_buffer_upper[0].conj().swapaxes(-2, -1)

        obc_l = obc_blocks.lesser[0]
        sl_00 = (
            sigma_lesser.local_blocks[0, 0]
            if obc_l is None
            else sigma_lesser.local_blocks[0, 0] + obc_l[stack_slice]
        )
        obc_l = obc_blocks.lesser[1]
        sl_11 = (
            sigma_lesser.local_blocks[1, 1]
            if obc_l is None
            else sigma_lesser.local_blocks[1, 1] + obc_l[stack_slice]
        )

        xl_diag_blocks[0] = sl_00
        xl_diag_blocks[1] = sl_11

        # xg_buffer_lower[0] = sigma_greater.local_blocks[0, 1]
        xg_buffer_upper[0] = sigma_greater.local_blocks[1, 0]
        # xg_buffer_lower[0] = -xg_buffer_upper[0].conj().swapaxes(-2, -1)

        obc_g = obc_blocks.greater[0]
        sg_00 = (
            sigma_greater.local_blocks[0, 0]
            if obc_g is None
            else sigma_greater.local_blocks[0, 0] + obc_g[stack_slice]
        )
        obc_g = obc_blocks.greater[1]
        sg_11 = (
            sigma_greater.local_blocks[1, 1]
            if obc_g is None
            else sigma_greater.local_blocks[1, 1] + obc_g[stack_slice]
        )
        xg_diag_blocks[0] = sg_00
        xg_diag_blocks[1] = sg_11

    for i in range(1, a.num_local_blocks - 1):
        j = i + 1
        # Invert current diagonal block.
        xr_diag_blocks[i] = linalg.inv(xr_diag_blocks[i])

        # Get the blocks that are used multiple times.
        a_ji = a.local_blocks[j, i]
        xr_ii = xr_diag_blocks[i]
        xr_ii_dagger = xr_diag_blocks[i].conj().swapaxes(-2, -1)
        xr_ji_xr_ii = xr_buffer_lower[i - 1] @ xr_ii

        # Precompute the transposes that are used multiple times.
        a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

        # Precompute some terms that are used multiple times.
        a_ji_xr_ii = a_ji @ xr_ii
        if selected_solve:
            sigma_lesser_ij = sigma_lesser.local_blocks[i, j]
            sigma_greater_ij = sigma_greater.local_blocks[i, j]
            a_ji_xr_ii_sl_ij = a_ji_xr_ii @ sigma_lesser_ij
            a_ji_xr_ii_sg_ij = a_ji_xr_ii @ sigma_greater_ij

        # Update next diagonal block.
        obc_r = obc_blocks.retarded[j]
        a_jj = (
            a.local_blocks[j, j]
            if obc_r is None
            else a.local_blocks[j, j] - obc_r[stack_slice]
        )
        xr_diag_blocks[j] = a_jj - a_ji_xr_ii @ a.local_blocks[i, j]
        # Update lower buffer block.
        xr_buffer_lower[i] = -xr_ji_xr_ii @ a.local_blocks[i, j]
        # Update upper buffer block.
        xr_buffer_upper[i] = -a_ji_xr_ii @ xr_buffer_upper[i - 1]
        # Update first block.
        xr_diag_blocks[0] = xr_diag_blocks[0] - xr_ji_xr_ii @ xr_buffer_upper[i - 1]

        if selected_solve:
            xl_diag_blocks[i] = xr_diag_blocks[i] @ xl_diag_blocks[i] @ xr_ii_dagger

            obc_l = obc_blocks.lesser[j]
            sl_jj = (
                sigma_lesser.local_blocks[j, j]
                if obc_l is None
                else sigma_lesser.local_blocks[j, j] + obc_l[stack_slice]
            )

            xl_diag_blocks[j] = (
                sl_jj
                + a_ji
                # + a.local_blocks[j, i]
                @ xl_diag_blocks[i] @ a_ji_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                # - sigma_lesser.local_blocks[j, i]
                # @ xr_ii_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                - a_ji_xr_ii_sl_ij
                # - a.local_blocks[j, i]
                # @ xr_diag_blocks[i]
                # @ sigma_lesser.local_blocks[i, j]
            )
            xl_buffer_upper[i] = (
                a_ji
                # a.local_blocks[j, i]
                @ xl_diag_blocks[i] @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                + sigma_lesser_ij.conj().swapaxes(-2, -1)
                # - sigma_lesser.local_blocks[j, i]
                @ xr_ji_xr_ii.conj().swapaxes(-2, -1)
                # @ xr_ii_dagger
                # @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                - a_ji_xr_ii @ xl_buffer_upper[i - 1]
            )
            # xl_buffer_lower[i] = -xl_buffer_upper[i].conj().swapaxes(-2, -1)
            # xl_buffer_lower[i] = (
            #     xr_buffer_lower[i - 1] @ xl_diag_blocks[i] @ a_ji_dagger
            #     # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
            #     - xl_buffer_lower[i - 1] @ a_ji_xr_ii.conj().swapaxes(-2, -1)
            #     # @ xr_ii_dagger
            #     # @ a_ji_dagger
            #     # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
            #     - xr_ji_xr_ii
            #     # - xr_buffer_lower[i - 1]
            #     # @ xr_ii
            #     @ sigma_lesser_ij
            # )
            xl_diag_blocks[0] = (
                xl_diag_blocks[0]
                + xr_buffer_lower[i - 1]
                @ xl_diag_blocks[i]
                @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                + xl_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                @ xr_ji_xr_ii.conj().swapaxes(-2, -1)
                # - xl_buffer_lower[i - 1] @ xr_ji_xr_ii.conj().swapaxes(-2, -1)
                # @ xr_ii_dagger
                # @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                - xr_ji_xr_ii @ xl_buffer_upper[i - 1]
            )

            xg_diag_blocks[i] = xr_ii @ xg_diag_blocks[i] @ xr_ii_dagger

            obc_g = obc_blocks.greater[j]
            sg_jj = (
                sigma_greater.local_blocks[j, j]
                if obc_g is None
                else sigma_greater.local_blocks[j, j] + obc_g[stack_slice]
            )
            xg_diag_blocks[j] = (
                sg_jj
                + a_ji @ xg_diag_blocks[i] @ a_ji_dagger
                + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                # - sigma_greater.local_blocks[j, i]
                # @ xr_ii_dagger
                # @ a.local_blocks[j, i].conj().swapaxes(-2, -1)
                - a_ji_xr_ii_sg_ij
                # - a.local_blocks[j, i]
                # @ xr_diag_blocks[i]
                # @ sigma_greater.local_blocks[i, j]
            )
            xg_buffer_upper[i] = (
                a_ji
                # a.local_blocks[j, i]
                @ xg_diag_blocks[i] @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                - sigma_greater.local_blocks[j, i] @ xr_ji_xr_ii.conj().swapaxes(-2, -1)
                # @ xr_ii_dagger
                # @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                - a_ji_xr_ii @ xg_buffer_upper[i - 1]
            )
            # xg_buffer_lower[i] = -xg_buffer_upper[i].conj().swapaxes(-2, -1)
            # xg_buffer_lower[i] = (
            #     xr_buffer_lower[i - 1] @ xg_diag_blocks[i] @ a_ji_dagger
            #     - xg_buffer_lower[i - 1] @ a_ji_xr_ii.conj().swapaxes(-2, -1)
            #     # @ xr_ii_dagger
            #     # @ a_ji_dagger
            #     - xr_buffer_lower[i - 1] @ xr_ii @ sigma_greater_ij
            # )
            xg_diag_blocks[0] = (
                xg_diag_blocks[0]
                + xr_buffer_lower[i - 1]
                @ xg_diag_blocks[i]
                @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                + xg_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                @ xr_ji_xr_ii.conj().swapaxes(-2, -1)
                # - xg_buffer_lower[i - 1] @ xr_ji_xr_ii.conj().swapaxes(-2, -1)
                # @ xr_ii_dagger
                # @ xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                - xr_ji_xr_ii @ xg_buffer_upper[i - 1]
            )


def downward_selinv(
    a: DSDBSparse | _DStackView,
    xr_diag_blocks: list[NDArray],
    xr_out: DSDBSparse | _DStackView,
    sigma_lesser: DSDBSparse | _DStackView = None,
    xl_diag_blocks: list[NDArray] = None,
    xl_out: DSDBSparse | _DStackView = None,
    sigma_greater: DSDBSparse | _DStackView = None,
    xg_diag_blocks: list[NDArray] = None,
    xg_out: DSDBSparse | _DStackView = None,
    selected_solve: bool = False,
    return_retarded: bool = True,
):
    """Performs the downward selected inversion."""
    for i in range(a.num_local_blocks - 2, -1, -1):
        j = i + 1

        # Get the blocks that are used multiple times.
        xr_ii = xr_diag_blocks[i]
        xr_jj = xr_diag_blocks[j]
        a_ij = a.local_blocks[i, j]
        a_ji = a.local_blocks[j, i]
        xl_ii = xl_diag_blocks[i]
        xl_jj = xl_diag_blocks[j]
        xg_ii = xg_diag_blocks[i]
        xg_jj = xg_diag_blocks[j]
        sigma_lesser_ij = sigma_lesser.local_blocks[i, j]
        sigma_greater_ij = sigma_greater.local_blocks[i, j]

        # Precompute the transposes that are used multiple times.
        xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
        xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
        a_ij_dagger = a_ij.conj().swapaxes(-2, -1)

        # Precompute the terms that are used multiple times.
        a_ji_dagger_xr_jj_dagger = a_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
        a_ij_dagger_xr_ii_dagger = a_ij_dagger @ xr_ii_dagger
        xr_ii_a_ij = xr_ii @ a_ij
        xr_jj_a_ji = xr_jj @ a_ji
        xr_ii_a_ij_xr_jj = xr_ii_a_ij @ xr_jj
        xr_jj_dagger_aij_dagger_xr_ii_dagger = xr_ii_a_ij_xr_jj.conj().swapaxes(-2, -1)
        xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji
        xr_ii_a_ij_xl_jj = xr_ii_a_ij @ xl_jj
        xr_ii_a_ij_xg_jj = xr_ii_a_ij @ xg_jj

        temp_1_l = xr_ii @ sigma_lesser_ij @ xr_jj_dagger_aij_dagger_xr_ii_dagger
        temp_1_l -= temp_1_l.conj().swapaxes(-2, -1)

        temp_1_g = xr_ii @ sigma_greater_ij @ xr_jj_dagger_aij_dagger_xr_ii_dagger
        temp_1_g -= temp_1_g.conj().swapaxes(-2, -1)

        # temp_1 = xr_diag_blocks[i] @ a.local_blocks[i, j]
        # temp_3 = xr_diag_blocks[j] @ a.local_blocks[j, i]

        if selected_solve:

            xl_ij = (
                -xr_ii_a_ij_xl_jj
                - xl_ii @ a_ji_dagger_xr_jj_dagger
                + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
            )

            xl_out.local_blocks[i, j] = xl_ij
            xl_out.local_blocks[j, i] = -xl_ij.conj().swapaxes(-2, -1)

            xg_ij = (
                -xr_ii_a_ij_xg_jj
                - xg_ii @ a_ji_dagger_xr_jj_dagger
                + xr_ii @ sigma_greater_ij @ xr_jj_dagger
            )

            xg_out.local_blocks[i, j] = xg_ij
            xg_out.local_blocks[j, i] = -xg_ij.conj().swapaxes(-2, -1)

            # temp_4 = a.local_blocks[j, i].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     j
            # ].conj().swapaxes(-2, -1)
            # xl_upper_block = (
            #     -temp_1 @ xl_diag_blocks[j]
            #     - xl_diag_blocks[i] @ temp_4
            #     + xr_diag_blocks[i]
            #     @ sigma_lesser.local_blocks[i, j]
            #     @ xr_diag_blocks[j].conj().swapaxes(-2, -1)
            # )

            # temp_2 = a.local_blocks[i, j].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     i
            # ].conj().swapaxes(-2, -1)
            # xl_lower_block = (
            #     -xl_diag_blocks[j] @ temp_2
            #     - temp_3 @ xl_diag_blocks[i]
            #     + xr_diag_blocks[j]
            #     @ sigma_lesser.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )

            # xl_diag_blocks[i] = (
            #     xl_diag_blocks[i]
            #     + temp_1 @ xl_diag_blocks[i + 1] @ temp_2
            #     + temp_1 @ temp_3 @ xl_diag_blocks[i]
            #     + xl_diag_blocks[i] @ temp_4 @ temp_2
            #     - temp_1
            #     @ xr_diag_blocks[i + 1]
            #     @ sigma_lesser.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - xr_diag_blocks[i]
            #     @ sigma_lesser.local_blocks[i, j]
            #     @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
            #     @ temp_2
            # )
            # # Streaming/Sparsifying back to DSDBSparse
            # xl_out.local_blocks[j, i] = xl_lower_block
            # xl_out.local_blocks[i, j] = xl_upper_block
            # xl_out.local_blocks[i, i] = xl_diag_blocks[i]

            # temp_4 = a.local_blocks[j, i].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     i + 1
            # ].conj().swapaxes(-2, -1)
            # xg_upper_block = (
            #     -temp_1 @ xg_diag_blocks[i + 1]
            #     - xg_diag_blocks[i] @ temp_4
            #     + xr_diag_blocks[i]
            #     @ sigma_greater.local_blocks[i, j]
            #     @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
            # )

            # temp_2 = a.local_blocks[i, j].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     i
            # ].conj().swapaxes(-2, -1)
            # xg_lower_block = (
            #     -xg_diag_blocks[i + 1] @ temp_2
            #     - temp_3 @ xg_diag_blocks[i]
            #     + xr_diag_blocks[i + 1]
            #     @ sigma_greater.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )

            # xg_diag_blocks[i] = (
            #     xg_diag_blocks[i]
            #     + temp_1 @ xg_diag_blocks[i + 1] @ temp_2
            #     + temp_1 @ temp_3 @ xg_diag_blocks[i]
            #     + xg_diag_blocks[i] @ temp_4 @ temp_2
            #     - temp_1
            #     @ xr_diag_blocks[i + 1]
            #     @ sigma_greater.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - xr_diag_blocks[i]
            #     @ sigma_greater.local_blocks[i, j]
            #     @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
            #     @ temp_2
            # )
            # # Streaming/Sparsifying back to DSDBSparse
            # xg_out.local_blocks[j, i] = xg_lower_block
            # xg_out.local_blocks[i, j] = xg_upper_block
            # xg_out.local_blocks[i, i] = xg_diag_blocks[i]

            temp_2_l = xr_ii_a_ij_xr_jj_a_ji @ xl_ii

            temp_2_g = xr_ii_a_ij_xr_jj_a_ji @ xg_ii

            xl_diag_blocks[i] = (
                xl_ii
                + xr_ii_a_ij_xl_jj @ a_ij_dagger_xr_ii_dagger
                - temp_1_l
                + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
            )
            xl_out.local_blocks[i, i] = 0.5 * (
                xl_diag_blocks[i] - xl_diag_blocks[i].conj().swapaxes(-2, -1)
            )
            xg_diag_blocks[i] = (
                xg_ii
                + xr_ii_a_ij_xg_jj @ a_ij_dagger_xr_ii_dagger
                - temp_1_g
                + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
            )
            xg_out.local_blocks[i, i] = 0.5 * (
                xg_diag_blocks[i] - xg_diag_blocks[i].conj().swapaxes(-2, -1)
            )

        x_lower_block = -xr_jj_a_ji @ xr_diag_blocks[i]
        x_upper_block = -xr_ii_a_ij @ xr_diag_blocks[j]
        # xr_diag_blocks[i] = (
        #     xr_diag_blocks[i] - x_upper_block @ a.local_blocks[j, i] @ xr_diag_blocks[i]
        # )
        xr_diag_blocks[i] = xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
        if not return_retarded:
            continue

        # # Streaming/Sparsifying back to DSDBSparse
        xr_out.local_blocks[j, i] = x_lower_block
        xr_out.local_blocks[i, j] = x_upper_block
        xr_out.local_blocks[i, i] = xr_diag_blocks[i]


def upward_selinv(
    a: DSDBSparse,
    xr_diag_blocks: list[NDArray],
    xr_out: DSDBSparse,
    sigma_lesser: DSDBSparse = None,
    xl_diag_blocks: list[NDArray] = None,
    xl_out: DSDBSparse = None,
    sigma_greater: DSDBSparse = None,
    xg_diag_blocks: list[NDArray] = None,
    xg_out: DSDBSparse = None,
    selected_solve: bool = False,
    return_retarded: bool = True,
):
    """Performs the upward selected inversion."""
    for i in range(1, a.num_local_blocks):
        j = i - 1

        # Get the blocks that are used multiple times.
        xr_ii = xr_diag_blocks[i]
        xr_jj = xr_diag_blocks[j]
        a_ij = a.local_blocks[i, j]
        a_ji = a.local_blocks[j, i]
        xl_ii = xl_diag_blocks[i]
        xl_jj = xl_diag_blocks[j]
        xg_ii = xg_diag_blocks[i]
        xg_jj = xg_diag_blocks[j]
        sigma_lesser_ij = sigma_lesser.local_blocks[i, j]
        sigma_greater_ij = sigma_greater.local_blocks[i, j]

        # Precompute the transposes that are used multiple times.
        xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
        xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
        a_ij_dagger = a_ij.conj().swapaxes(-2, -1)

        # Precompute the terms that are used multiple times.
        a_ji_dagger_xr_jj_dagger = a_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
        a_ij_dagger_xr_ii_dagger = a_ij_dagger @ xr_ii_dagger
        xr_ii_a_ij = xr_ii @ a_ij
        xr_jj_a_ji = xr_jj @ a_ji
        xr_ii_a_ij_xr_jj = xr_ii_a_ij @ xr_jj
        xr_jj_dagger_aij_dagger_xr_ii_dagger = xr_ii_a_ij_xr_jj.conj().swapaxes(-2, -1)
        xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji
        xr_ii_a_ij_xl_jj = xr_ii_a_ij @ xl_jj
        xr_ii_a_ij_xg_jj = xr_ii_a_ij @ xg_jj

        temp_1_l = xr_ii @ sigma_lesser_ij @ xr_jj_dagger_aij_dagger_xr_ii_dagger
        temp_1_l -= temp_1_l.conj().swapaxes(-2, -1)

        temp_1_g = xr_ii @ sigma_greater_ij @ xr_jj_dagger_aij_dagger_xr_ii_dagger
        temp_1_g -= temp_1_g.conj().swapaxes(-2, -1)

        # temp_1 = xr_diag_blocks[j] @ a.local_blocks[j, i]
        # temp_3 = xr_diag_blocks[i] @ a.local_blocks[i, j]

        if selected_solve:

            xl_ij = (
                -xr_ii_a_ij_xl_jj
                - xl_ii @ a_ji_dagger_xr_jj_dagger
                + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
            )

            xl_out.local_blocks[i, j] = xl_ij
            xl_out.local_blocks[j, i] = -xl_ij.conj().swapaxes(-2, -1)

            xg_ij = (
                -xr_ii_a_ij_xg_jj
                - xg_ii @ a_ji_dagger_xr_jj_dagger
                + xr_ii @ sigma_greater_ij @ xr_jj_dagger
            )

            xg_out.local_blocks[i, j] = xg_ij
            xg_out.local_blocks[j, i] = -xg_ij.conj().swapaxes(-2, -1)

            # temp_4 = a.local_blocks[i, j].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     i
            # ].conj().swapaxes(-2, -1)
            # xl_upper_block = (
            #     -temp_1 @ xl_diag_blocks[i]
            #     - xl_diag_blocks[j] @ temp_4
            #     + xr_diag_blocks[j]
            #     @ sigma_lesser.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )

            # temp_2 = a.local_blocks[j, i].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     j
            # ].conj().swapaxes(-2, -1)
            # xl_lower_block = (
            #     -xl_diag_blocks[i] @ temp_2
            #     - temp_3 @ xl_diag_blocks[j]
            #     + xr_diag_blocks[i]
            #     @ sigma_lesser.local_blocks[i, j]
            #     @ xr_diag_blocks[j].conj().swapaxes(-2, -1)
            # )

            # xl_diag_blocks[i] = (
            #     xl_diag_blocks[i]
            #     + temp_3 @ xl_diag_blocks[j] @ temp_4
            #     + temp_3 @ temp_1 @ xl_diag_blocks[i]
            #     + xl_diag_blocks[i] @ temp_2 @ temp_4
            #     - temp_3
            #     @ xr_diag_blocks[j]
            #     @ sigma_lesser.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - xr_diag_blocks[i]
            #     @ sigma_lesser.local_blocks[i, j]
            #     @ xr_diag_blocks[j].conj().swapaxes(-2, -1)
            #     @ temp_4
            # )
            # # Streaming/Sparsifying back to DSDBSparse
            # xl_out.local_blocks[j, i] = xl_upper_block
            # xl_out.local_blocks[i, j] = xl_lower_block
            # xl_out.local_blocks[i, i] = xl_diag_blocks[i]

            # temp_4 = a.local_blocks[i, j].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     i
            # ].conj().swapaxes(-2, -1)
            # xg_upper_block = (
            #     -temp_1 @ xg_diag_blocks[i]
            #     - xg_diag_blocks[j] @ temp_4
            #     + xr_diag_blocks[j]
            #     @ sigma_greater.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )

            # temp_2 = a.local_blocks[j, i].conj().swapaxes(-2, -1) @ xr_diag_blocks[
            #     j
            # ].conj().swapaxes(-2, -1)
            # xg_lower_block = (
            #     -xg_diag_blocks[i] @ temp_2
            #     - temp_3 @ xg_diag_blocks[j]
            #     + xr_diag_blocks[i]
            #     @ sigma_greater.local_blocks[i, j]
            #     @ xr_diag_blocks[j].conj().swapaxes(-2, -1)
            # )

            # xg_diag_blocks[i] = (
            #     xg_diag_blocks[i]
            #     + temp_3 @ xg_diag_blocks[j] @ temp_4
            #     + temp_3 @ temp_1 @ xg_diag_blocks[i]
            #     + xg_diag_blocks[i] @ temp_2 @ temp_4
            #     - temp_3
            #     @ xr_diag_blocks[j]
            #     @ sigma_greater.local_blocks[j, i]
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - xr_diag_blocks[i]
            #     @ sigma_greater.local_blocks[i, j]
            #     @ xr_diag_blocks[j].conj().swapaxes(-2, -1)
            #     @ temp_4
            # )
            # # Streaming/Sparsifying back to DSDBSparse
            # xg_out.local_blocks[j, i] = xg_upper_block
            # xg_out.local_blocks[i, j] = xg_lower_block
            # xg_out.local_blocks[i, i] = xg_diag_blocks[i]

            temp_2_l = xr_ii_a_ij_xr_jj_a_ji @ xl_ii

            temp_2_g = xr_ii_a_ij_xr_jj_a_ji @ xg_ii

            xl_diag_blocks[i] = (
                xl_ii
                + xr_ii_a_ij_xl_jj @ a_ij_dagger_xr_ii_dagger
                - temp_1_l
                + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
            )
            xl_out.local_blocks[i, i] = 0.5 * (
                xl_diag_blocks[i] - xl_diag_blocks[i].conj().swapaxes(-2, -1)
            )
            xg_diag_blocks[i] = (
                xg_ii
                + xr_ii_a_ij_xg_jj @ a_ij_dagger_xr_ii_dagger
                - temp_1_g
                + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
            )
            xg_out.local_blocks[i, i] = 0.5 * (
                xg_diag_blocks[i] - xg_diag_blocks[i].conj().swapaxes(-2, -1)
            )

        x_upper_block = -xr_jj_a_ji @ xr_diag_blocks[i]
        x_lower_block = -xr_ii_a_ij @ xr_diag_blocks[j]
        # xr_diag_blocks[i] = (
        #     xr_diag_blocks[i] - x_lower_block @ a.local_blocks[j, i] @ xr_diag_blocks[i]
        # )
        xr_diag_blocks[i] = xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
        if not return_retarded:
            continue

        # Streaming/Sparsifying back to DSDBSparse
        xr_out.local_blocks[j, i] = x_upper_block
        xr_out.local_blocks[i, j] = x_lower_block
        xr_out.local_blocks[i, i] = xr_diag_blocks[i]


def permuted_selinv(
    a: DSDBSparse | _DStackView,
    xr_diag_blocks: list[NDArray],
    xr_buffer_lower: list[NDArray],
    xr_buffer_upper: list[NDArray],
    xr_out: DSDBSparse | _DStackView,
    sigma_lesser: DSDBSparse | _DStackView = None,
    xl_diag_blocks: list[NDArray] = None,
    xl_buffer_lower: list[NDArray] = None,
    xl_buffer_upper: list[NDArray] = None,
    xl_out: DSDBSparse | _DStackView = None,
    sigma_greater: DSDBSparse | _DStackView = None,
    xg_diag_blocks: list[NDArray] = None,
    xg_buffer_lower: list[NDArray] = None,
    xg_buffer_upper: list[NDArray] = None,
    xg_out: DSDBSparse | _DStackView = None,
    selected_solve: bool = False,
    return_retarded: bool = True,
):
    """Performs the permuted selected inversion."""
    for i in range(a.num_local_blocks - 2, 0, -1):
        # j = i + 1

        # # Get the blocks that are used multiple times.
        # xr_ii = xr_diag_blocks[i]
        # xr_jj = xr_diag_blocks[j]
        # a_ij = a.local_blocks[i, j]
        # a_ji = a.local_blocks[j, i]
        # xl_ii = xl_diag_blocks[i]
        # xl_jj = xl_diag_blocks[j]
        # xg_ii = xg_diag_blocks[i]
        # xg_jj = xg_diag_blocks[j]
        # sigma_lesser_ij = sigma_lesser.local_blocks[i, j]
        # sigma_greater_ij = sigma_greater.local_blocks[i, j]

        # # Precompute the transposes that are used multiple times.
        # xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
        # xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
        # a_ij_dagger = a_ij.conj().swapaxes(-2, -1)

        # # Precompute the terms that are used multiple times.
        # a_ji_dagger_xr_jj_dagger = a_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
        # a_ij_dagger_xr_ii_dagger = a_ij_dagger @ xr_ii_dagger
        # xr_ii_a_ij = xr_ii @ a_ij
        # xr_jj_a_ji = xr_jj @ a_ji
        # xr_ii_a_ij_xr_jj = xr_ii_a_ij @ xr_jj
        # xr_jj_dagger_aij_dagger_xr_ii_dagger = xr_ii_a_ij_xr_jj.conj().swapaxes(-2, -1)
        # xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji
        # xr_ii_a_ij_xl_jj = xr_ii_a_ij @ xl_jj
        # xr_ii_a_ij_xg_jj = xr_ii_a_ij @ xg_jj

        # temp_1_l = xr_ii @ sigma_lesser_ij @ xr_jj_dagger_aij_dagger_xr_ii_dagger
        # temp_1_l -= temp_1_l.conj().swapaxes(-2, -1)

        # temp_1_g = xr_ii @ sigma_greater_ij @ xr_jj_dagger_aij_dagger_xr_ii_dagger
        # temp_1_g -= temp_1_g.conj().swapaxes(-2, -1)

        B1 = (
            a.local_blocks[i, i + 1] @ xr_diag_blocks[i + 1]
            + xr_buffer_upper[i - 1] @ xr_buffer_lower[i]
        )
        B2 = (
            a.local_blocks[i, i + 1] @ xr_buffer_upper[i]
            + xr_buffer_upper[i - 1] @ xr_diag_blocks[0]
        )
        C1 = (
            xr_diag_blocks[i + 1] @ a.local_blocks[i + 1, i]
            + xr_buffer_upper[i] @ xr_buffer_lower[i - 1]
        )
        C2 = (
            xr_buffer_lower[i] @ a.local_blocks[i + 1, i]
            + xr_diag_blocks[0] @ xr_buffer_lower[i - 1]
        )

        if selected_solve:
            temp_B_13 = xl_buffer_upper[i - 1]
            # temp_B_31 = xl_buffer_lower[i - 1]
            temp_B_31 = -xl_buffer_upper[i - 1].conj().swapaxes(-2, -1)

            bl_upper_block = (
                -xr_diag_blocks[i]
                @ (
                    a.local_blocks[i, i + 1] @ xl_diag_blocks[i + 1]
                    - xr_buffer_upper[i - 1]
                    @ xl_buffer_upper[i].conj().swapaxes(-2, -1)
                    # + xr_buffer_upper[i - 1] @ xl_buffer_lower[i]
                )
                - xl_diag_blocks[i]
                @ (
                    a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                    @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                    + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                    @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                )
                + xr_diag_blocks[i]
                @ (
                    sigma_lesser.local_blocks[i, i + 1]
                    @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                    + xl_buffer_upper[i - 1]
                    @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                )
            )
            xl_buffer_upper[i - 1] = (
                -xr_diag_blocks[i]
                @ (
                    a.local_blocks[i, i + 1] @ xl_buffer_upper[i]
                    + xr_buffer_upper[i - 1] @ xl_diag_blocks[0]
                )
                - xl_diag_blocks[i]
                @ (
                    a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                    @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                    + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                    @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                )
                + xr_diag_blocks[i]
                @ (
                    sigma_lesser.local_blocks[i, i + 1]
                    @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                    + xl_buffer_upper[i - 1] @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                )
            )

            # bl_lower_block = (
            #     -(
            #         xl_diag_blocks[i + 1]
            #         @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
            #         + xl_buffer_upper[i]
            #         @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - (C1) @ xl_diag_blocks[i]
            #     + (
            #         xr_diag_blocks[i + 1] @ sigma_lesser.local_blocks[i + 1, i]
            #         + xr_buffer_upper[i] @ xl_buffer_lower[i - 1]
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )
            # xl_buffer_lower[i - 1] = (
            #     -(
            #         xl_buffer_lower[i]
            #         @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
            #         + xl_diag_blocks[0] @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - (C2) @ xl_diag_blocks[i]
            #     + (
            #         xr_buffer_lower[i] @ sigma_lesser.local_blocks[i + 1, i]
            #         + xr_diag_blocks[0] @ xl_buffer_lower[i - 1]
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )

            xl_diag_blocks[i] = (
                xl_diag_blocks[i]
                + xr_diag_blocks[i]
                @ (
                    (
                        a.local_blocks[i, i + 1] @ xl_diag_blocks[i + 1]
                        - xr_buffer_upper[i - 1]
                        @ xl_buffer_upper[i].conj().swapaxes(-2, -1)
                        # + xr_buffer_upper[i - 1] @ xl_buffer_lower[i]
                    )
                    @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
                    + (
                        a.local_blocks[i, i + 1] @ xl_buffer_upper[i]
                        + xr_buffer_upper[i - 1] @ xl_diag_blocks[0]
                    )
                    @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                )
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
                + xr_diag_blocks[i]
                @ ((B1) @ a.local_blocks[i + 1, i] + (B2) @ xr_buffer_lower[i - 1])
                @ xl_diag_blocks[i]
                + xl_diag_blocks[i]
                @ (
                    (
                        a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                        @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                        + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                        @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                    )
                    @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
                    + (
                        a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                        @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                        + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                        @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                    )
                    @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                )
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
                - xr_diag_blocks[i]
                @ ((B1) @ sigma_lesser.local_blocks[i + 1, i] + (B2) @ temp_B_31)
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
                - xr_diag_blocks[i]
                @ (
                    (
                        sigma_lesser.local_blocks[i, i + 1]
                        @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                        + temp_B_13 @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                    )
                    @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
                    + (
                        sigma_lesser.local_blocks[i, i + 1]
                        @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                        + temp_B_13 @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                    )
                    @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                )
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            )
            # Streaming/Sparsifying back to DSDBSparse
            # xl_out.local_blocks[i + 1, i] = bl_lower_block
            xl_out.local_blocks[i, i + 1] = bl_upper_block
            xl_out.local_blocks[i + 1, i] = -bl_upper_block.conj().swapaxes(-2, -1)
            xl_out.local_blocks[i, i] = 0.5 * (
                xl_diag_blocks[i] - xl_diag_blocks[i].conj().swapaxes(-2, -1)
            )

            temp_B_13 = xg_buffer_upper[i - 1]
            # temp_B_31 = xg_buffer_lower[i - 1]
            temp_B_31 = -xg_buffer_upper[i - 1].conj().swapaxes(-2, -1)

            bg_upper_block = (
                -xr_diag_blocks[i]
                @ (
                    a.local_blocks[i, i + 1] @ xg_diag_blocks[i + 1]
                    - xr_buffer_upper[i - 1]
                    @ xg_buffer_upper[i].conj().swapaxes(-2, -1)
                    # + xr_buffer_upper[i - 1] @ xg_buffer_lower[i]
                )
                - xg_diag_blocks[i]
                @ (
                    a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                    @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                    + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                    @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                )
                + xr_diag_blocks[i]
                @ (
                    sigma_greater.local_blocks[i, i + 1]
                    @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                    + xg_buffer_upper[i - 1]
                    @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                )
            )
            xg_buffer_upper[i - 1] = (
                -xr_diag_blocks[i]
                @ (
                    a.local_blocks[i, i + 1] @ xg_buffer_upper[i]
                    + xr_buffer_upper[i - 1] @ xg_diag_blocks[0]
                )
                - xg_diag_blocks[i]
                @ (
                    a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                    @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                    + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                    @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                )
                + xr_diag_blocks[i]
                @ (
                    sigma_greater.local_blocks[i, i + 1]
                    @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                    + xg_buffer_upper[i - 1] @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                )
            )

            # bg_lower_block = (
            #     -(
            #         xg_diag_blocks[i + 1]
            #         @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
            #         + xg_buffer_upper[i]
            #         @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - (C1) @ xg_diag_blocks[i]
            #     + (
            #         xr_diag_blocks[i + 1] @ sigma_greater.local_blocks[i + 1, i]
            #         + xr_buffer_upper[i] @ xg_buffer_lower[i - 1]
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )
            # xg_buffer_lower[i - 1] = (
            #     -(
            #         xg_buffer_lower[i]
            #         @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
            #         + xg_diag_blocks[0] @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            #     - (C2) @ xg_diag_blocks[i]
            #     + (
            #         xr_buffer_lower[i] @ sigma_greater.local_blocks[i + 1, i]
            #         + xr_diag_blocks[0] @ xg_buffer_lower[i - 1]
            #     )
            #     @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            # )

            xg_diag_blocks[i] = (
                xg_diag_blocks[i]
                + xr_diag_blocks[i]
                @ (
                    (
                        a.local_blocks[i, i + 1] @ xg_diag_blocks[i + 1]
                        - xr_buffer_upper[i - 1]
                        @ xg_buffer_upper[i].conj().swapaxes(-2, -1)
                        # + xr_buffer_upper[i - 1] @ xg_buffer_lower[i]
                    )
                    @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
                    + (
                        a.local_blocks[i, i + 1] @ xg_buffer_upper[i]
                        + xr_buffer_upper[i - 1] @ xg_diag_blocks[0]
                    )
                    @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                )
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
                + xr_diag_blocks[i]
                @ ((B1) @ a.local_blocks[i + 1, i] + (B2) @ xr_buffer_lower[i - 1])
                @ xg_diag_blocks[i]
                + xg_diag_blocks[i]
                @ (
                    (
                        a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                        @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                        + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                        @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                    )
                    @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
                    + (
                        a.local_blocks[i + 1, i].conj().swapaxes(-2, -1)
                        @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                        + xr_buffer_lower[i - 1].conj().swapaxes(-2, -1)
                        @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                    )
                    @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                )
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
                - xr_diag_blocks[i]
                @ ((B1) @ sigma_greater.local_blocks[i + 1, i] + (B2) @ temp_B_31)
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
                - xr_diag_blocks[i]
                @ (
                    (
                        sigma_greater.local_blocks[i, i + 1]
                        @ xr_diag_blocks[i + 1].conj().swapaxes(-2, -1)
                        + temp_B_13 @ xr_buffer_upper[i].conj().swapaxes(-2, -1)
                    )
                    @ a.local_blocks[i, i + 1].conj().swapaxes(-2, -1)
                    + (
                        sigma_greater.local_blocks[i, i + 1]
                        @ xr_buffer_lower[i].conj().swapaxes(-2, -1)
                        + temp_B_13 @ xr_diag_blocks[0].conj().swapaxes(-2, -1)
                    )
                    @ xr_buffer_upper[i - 1].conj().swapaxes(-2, -1)
                )
                @ xr_diag_blocks[i].conj().swapaxes(-2, -1)
            )
            # Streaming/Sparsifying back to DSDBSparse
            # xg_out.local_blocks[i + 1, i] = bg_lower_block
            xg_out.local_blocks[i, i + 1] = bg_upper_block
            xg_out.local_blocks[i + 1, i] = -bg_upper_block.conj().swapaxes(-2, -1)
            xg_out.local_blocks[i, i] = 0.5 * (
                xg_diag_blocks[i] - xg_diag_blocks[i].conj().swapaxes(-2, -1)
            )

        if return_retarded:
            xr_out.local_blocks[i, i + 1] = -xr_diag_blocks[i] @ B1

        xr_buffer_upper[i - 1] = -xr_diag_blocks[i] @ B2

        D1 = a.local_blocks[i + 1, i]
        D2 = xr_buffer_lower[i - 1]

        if return_retarded:
            xr_out.local_blocks[i + 1, i] = -C1 @ xr_diag_blocks[i]

        xr_buffer_lower[i - 1] = -C2 @ xr_diag_blocks[i]

        xr_diag_blocks[i] = (
            xr_diag_blocks[i]
            + xr_diag_blocks[i] @ (B1 @ D1 + B2 @ D2) @ xr_diag_blocks[i]
        )
        # Streaming/Sparsifying back to DSDBSparse
        if return_retarded:
            xr_out.local_blocks[i, i] = xr_diag_blocks[i]

    if return_retarded:
        xr_out.local_blocks[1, 0] = xr_buffer_upper[0]
        xr_out.local_blocks[0, 1] = xr_buffer_lower[0]
    if selected_solve:
        xl_out.local_blocks[1, 0] = xl_buffer_upper[0]
        # xl_out.local_blocks[0, 1] = xl_buffer_lower[0]
        xl_out.local_blocks[0, 1] = -xl_buffer_upper[0].conj().swapaxes(-2, -1)

        xg_out.local_blocks[1, 0] = xg_buffer_upper[0]
        # xg_out.local_blocks[0, 1] = xg_buffer_lower[0]
        xg_out.local_blocks[0, 1] = -xg_buffer_upper[0].conj().swapaxes(-2, -1)
