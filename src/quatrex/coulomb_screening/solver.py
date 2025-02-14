# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.datastructures.routines import btd_matmul, btd_sandwich
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import compute_num_connected_blocks
from quatrex.coulomb_screening.utils import assemble_boundary_blocks


def _check_block_sizes(rows: NDArray, columns: NDArray, block_sizes: NDArray) -> bool:
    """Checks if matrix elements lie within the block-tridiagonal.

    Parameters
    ----------
    rows : NDArray
        The row indices of the matrix elements.
    columns : NDArray
        The column indices of the matrix elements.
    block_sizes : NDArray
        The block sizes.

    Returns
    -------
    bool
        True if the matrix elements lie within the block-tridiagonal,
        False otherwise.

    """
    nnz_in_blocks = 0
    for i in range(len(block_sizes)):
        bmin = sum(block_sizes[:i])
        bmax = sum(block_sizes[: i + 1])
        mask = (rows >= bmin) & (rows < bmax) & (columns >= bmin) & (columns < bmax)
        nnz_in_blocks += mask.sum()
        if i > 0:
            bmin2 = sum(block_sizes[: i - 1])
            bmax2 = sum(block_sizes[:i])
            mask_lower = (
                (rows >= bmin) & (rows < bmax) & (columns >= bmin2) & (columns < bmax2)
            )
            mask_upper = (
                (rows >= bmin2) & (rows < bmax2) & (columns >= bmin) & (columns < bmax)
            )
            nnz_in_blocks += mask_lower.sum() + mask_upper.sum()
    return rows.size == nnz_in_blocks


def _spillover_matmul(
    a: sparse.spmatrix, b: sparse.spmatrix, block_sizes
) -> sparse.coo_matrix:
    """Multiplies two sparse matrices with spillover correction."""
    c = (a @ b).tolil()

    a = a.tolil()
    b = b.tolil()

    # Left spillover
    i_ = slice(None, block_sizes[0])
    j_ = slice(block_sizes[0], sum(block_sizes[:2]))
    c[i_, i_] += a[j_, i_] @ b[i_, j_]

    # Right spillover
    i_ = slice(-block_sizes[-1], None)
    j_ = slice(-sum(block_sizes[-2:]), -block_sizes[-1])
    c[i_, i_] += a[j_, i_] @ b[i_, j_]

    return c.tocoo()


class CoulombScreeningSolver(SubsystemSolver):
    """Solves the dynamics of the screened Coulomb interaction.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies : NDArray
        The energies at which to solve.

    """

    system = "coulomb_screening"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the solver."""
        super().__init__(quatrex_config, compute_config, energies)

        # Load the Coulomb matrix.
        self.coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)

        # Load block sizes.
        self.small_block_sizes = distributed_load(
            quatrex_config.input_dir / "block_sizes.npy"
        ).astype(xp.int32)
        if not _check_block_sizes(
            self.coulomb_matrix_sparray.row,
            self.coulomb_matrix_sparray.col,
            self.small_block_sizes,
        ):
            raise ValueError("Block sizes do not match Coulomb matrix.")

        self.num_connected_blocks = (
            quatrex_config.coulomb_screening.num_connected_blocks
        )
        if self.num_connected_blocks == "auto":
            self.num_connected_blocks = compute_num_connected_blocks(
                sparsity_pattern, self.small_block_sizes
            )

        if len(self.small_block_sizes) % self.num_connected_blocks != 0:
            # Not implemented yet.
            raise ValueError(
                f"Number of blocks must be divisible by {self.num_connected_blocks}."
            )

        self.block_sizes = (
            self.small_block_sizes[
                : len(self.small_block_sizes) // self.num_connected_blocks
            ]
            * self.num_connected_blocks
        )
        # Check that the provided block sizes match the coulomb matrix.
        if self.small_block_sizes.sum() != self.coulomb_matrix_sparray.shape[0]:
            raise ValueError(
                "Block sizes do not match Coulomb matrix. "
                f"{self.small_block_sizes.sum()} != {self.coulomb_matrix_sparray.shape[0]}"
            )
        # Create DBSparse matrix from the Coulomb matrix.
        # TODO: Waste of memory. Not an energy-dependent matrix.
        self.coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.small_block_sizes,
            global_stack_shape=self.energies.shape,
        )
        self.coulomb_matrix.data = 0.0
        self.coulomb_matrix += self.coulomb_matrix_sparray

        v_times_p_sparsity_pattern = _spillover_matmul(
            sparsity_pattern, sparsity_pattern, self.small_block_sizes
        )
        # Allocate memory for the Coulomb matrix times polarization.
        self.v_times_p_retarded = compute_config.dsbsparse_type.from_sparray(
            v_times_p_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
        )
        self.v_times_p_retarded.data = 0.0

        l_sparsity_pattern = _spillover_matmul(
            v_times_p_sparsity_pattern, sparsity_pattern, self.block_sizes
        )
        # Allocate memory for the L_lesser and L_greater matrices.
        self.l_lesser = compute_config.dsbsparse_type.from_sparray(
            l_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
        )
        self.l_lesser.data = 0.0

        self.l_greater = compute_config.dsbsparse_type.zeros_like(self.l_lesser)
        self.system_matrix = compute_config.dsbsparse_type.zeros_like(
            self.v_times_p_retarded
        )

        # Boundary conditions.
        self.left_occupancies = bose_einstein(
            self.local_energies,
            quatrex_config.coulomb_screening.temperature,
        )
        self.right_occupancies = bose_einstein(
            self.local_energies,
            quatrex_config.coulomb_screening.temperature,
        )

        self.obc_blocks = OBCBlocks(num_blocks=self.block_sizes.size)

        # Allocate memory for the OBC blocks.
        self.boundary_blocks_left = {
            "diag": xp.zeros_like(self.system_matrix.blocks[0, 0]),
            "right": xp.zeros_like(self.system_matrix.blocks[0, 0]),
            "below": xp.zeros_like(self.system_matrix.blocks[0, 0]),
        }
        self.boundary_blocks_right = {
            "diag": xp.zeros_like(self.system_matrix.blocks[-1, -1]),
            "above": xp.zeros_like(self.system_matrix.blocks[-1, -1]),
            "left": xp.zeros_like(self.system_matrix.blocks[-1, -1]),
        }

    def _set_block_sizes(self, block_sizes: NDArray) -> None:
        """Sets the block sizes of all matrices.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        self.v_times_p_retarded.block_sizes = block_sizes
        self.system_matrix.block_sizes = block_sizes
        self.l_lesser.block_sizes = block_sizes
        self.l_greater.block_sizes = block_sizes

    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""

        # Compute surface Green's functions.
        x_00 = self.obc(
            self.boundary_blocks_left["diag"],
            self.boundary_blocks_left["right"],
            self.boundary_blocks_left["below"],
            "left",
        )
        # Twist it, flip it, bop it.
        x_nn = self.obc(
            xp.flip(self.boundary_blocks_right["diag"], axis=(-2, -1)),
            xp.flip(self.boundary_blocks_right["left"], axis=(-2, -1)),
            xp.flip(self.boundary_blocks_right["above"], axis=(-2, -1)),
            "right",
        )
        x_nn = xp.flip(x_nn, axis=(-2, -1))

        self.obc_blocks.retarded[0] = (
            self.system_matrix.blocks[1, 0] @ x_00 @ self.system_matrix.blocks[0, 1]
        )
        self.obc_blocks.retarded[-1] = (
            self.system_matrix.blocks[-2, -1] @ x_nn @ self.system_matrix.blocks[-1, -2]
        )

        # Compute and apply the lesser boundary self-energy.
        a_00 = self.boundary_blocks_left["below"] @ x_00 @ self.l_lesser.blocks[0, 1]
        a_nn = self.boundary_blocks_right["above"] @ x_nn @ self.l_lesser.blocks[-1, -2]
        w_00 = self.lyapunov(
            x_00 @ self.boundary_blocks_left["below"],
            x_00
            @ (self.l_lesser.blocks[0, 0] - (a_00 - a_00.conj().swapaxes(-1, -2)))
            @ x_00.conj().swapaxes(-1, -2),
            "left",
        )
        w_nn = self.lyapunov(
            x_nn @ self.boundary_blocks_right["above"],
            x_nn
            @ (self.l_lesser.blocks[-1, -1] - (a_nn - a_nn.conj().swapaxes(-1, -2)))
            @ x_nn.conj().swapaxes(-1, -2),
            "right",
        )

        self.obc_blocks.lesser[0] = self.system_matrix.blocks[
            1, 0
        ] @ w_00 @ self.system_matrix.blocks[1, 0].conj().swapaxes(-1, -2) - (
            a_00 - a_00.conj().swapaxes(-1, -2)
        )
        self.obc_blocks.lesser[-1] = self.system_matrix.blocks[
            -2, -1
        ] @ w_nn @ self.system_matrix.blocks[-2, -1].conj().swapaxes(-1, -2) - (
            a_nn - a_nn.conj().swapaxes(-1, -2)
        )

        # Compute and apply the greater boundary self-energy.
        a_00 = self.boundary_blocks_left["below"] @ x_00 @ self.l_greater.blocks[0, 1]
        a_nn = (
            self.boundary_blocks_right["above"] @ x_nn @ self.l_greater.blocks[-1, -2]
        )
        w_00 = self.lyapunov(
            x_00 @ self.boundary_blocks_left["below"],
            x_00
            @ (self.l_greater.blocks[0, 0] - (a_00 - a_00.conj().swapaxes(-1, -2)))
            @ x_00.conj().swapaxes(-1, -2),
            "left",
        )
        w_nn = self.lyapunov(
            x_nn @ self.boundary_blocks_right["above"],
            x_nn
            @ (self.l_greater.blocks[-1, -1] - (a_nn - a_nn.conj().swapaxes(-1, -2)))
            @ x_nn.conj().swapaxes(-1, -2),
            "right",
        )

        self.obc_blocks.greater[0] = self.system_matrix.blocks[
            1, 0
        ] @ w_00 @ self.system_matrix.blocks[1, 0].conj().swapaxes(-1, -2) - (
            a_00 - a_00.conj().swapaxes(-1, -2)
        )
        self.obc_blocks.greater[-1] = self.system_matrix.blocks[
            -2, -1
        ] @ w_nn @ self.system_matrix.blocks[-2, -1].conj().swapaxes(-1, -2) - (
            a_nn - a_nn.conj().swapaxes(-1, -2)
        )

    def _assemble_system_matrix(self) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data = 0.0
        self.system_matrix += sparse.eye(self.system_matrix.shape[-1])
        self.system_matrix -= self.v_times_p_retarded

    def solve(
        self,
        p_lesser: DSBSparse,
        p_greater: DSBSparse,
        p_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Solves for the screened Coulomb interaction.

        Parameters
        ----------
        p_lesser : DSBSparse
            The lesser polarization.
        p_greater : DSBSparse
            The greater polarization.
        p_retarded : DSBSparse
            The retarded polarization.
        out : tuple[DSBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """
        times = []

        # Compute the product of the Coulomb matrix with the polarization.
        times.append(time.perf_counter())
        # Change the block sizes to match the Coulomb matrix.
        self._set_block_sizes(self.small_block_sizes)

        btd_matmul(
            self.coulomb_matrix,
            p_retarded,
            out=self.v_times_p_retarded,
            spillover_correction=True,
        )
        btd_sandwich(
            self.coulomb_matrix,
            p_lesser,
            out=self.l_lesser,
            spillover_correction=True,
        )
        btd_sandwich(
            self.coulomb_matrix,
            p_greater,
            out=self.l_greater,
            spillover_correction=True,
        )

        t_obc_multiply = time.perf_counter() - times.pop()

        # Assemble the matrices.
        times.append(time.perf_counter())
        # Assemble the system matrix.
        self._assemble_system_matrix()
        # Assemble the boundary blocks.
        assemble_boundary_blocks(
            self.boundary_blocks_left["diag"],
            self.boundary_blocks_left["right"],
            self.boundary_blocks_left["below"],
            self.boundary_blocks_right["diag"],
            self.boundary_blocks_right["above"],
            self.boundary_blocks_right["left"],
            self.system_matrix,
            nbc=self.num_connected_blocks,
        )
        # Go back to normal block sizes.
        self._set_block_sizes(self.block_sizes)
        t_assemble = time.perf_counter() - times.pop()

        # Apply the OBC algorithm.
        times.append(time.perf_counter())

        self._compute_obc()
        t_obc = time.perf_counter() - times.pop()

        # Solve the system
        times.append(time.perf_counter())
        self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=self.l_lesser,
            sigma_greater=self.l_greater,
            obc_blocks=self.obc_blocks,
            out=out,
            return_retarded=True,
        )
        t_solve = time.perf_counter() - times.pop()

        # Compute the retarded Screened interaction (mainly used for debugging).
        # Currently doesn't work.
        # obc_multiply(out[2], (out[2], self.coulomb_matrix), self.block_sizes)

        w_lesser, w_greater, *__ = out
        w_lesser.data = 0.5 * (
            w_lesser.data - w_lesser.ltranspose(copy=True).data.conj()
        )
        w_greater.data = 0.5 * (
            w_greater.data - w_greater.ltranspose(copy=True).data.conj()
        )

        # if comm.rank == 0:
        #     w_greater.data[0,:] = 0.0
        #     w_lesser.data[0,:] = 0.0

        (
            print(
                f"OBC Multiply: {t_obc_multiply}, Assemble: {t_assemble}, OBC: {t_obc}, Solve: {t_solve}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )
