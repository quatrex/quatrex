# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.datastructures.routines import btd_matmul, btd_sandwich
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.core.subsystem import SubsystemSolver
from quatrex.coulomb_screening.utils import assemble_boundary_blocks

# from qttools.utils.sparse_utils import product_sparsity_pattern


def check_block_sizes(rows: NDArray, columns: NDArray, block_sizes: NDArray) -> bool:
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


def obc_multiply(
    buffer: DSBSparse, matrices: tuple[DSBSparse, ...], block_sizes: NDArray
) -> None:
    """Multiplies two DSBSparse matrices, accounting for OBC spillover.

    Parameters
    ----------
    buffer : DSBSparse
        The output buffer.
    matrices : tuple[DSBSparse, ...]
        The matrices to multiply.
    block_sizes : NDArray
        The block sizes.

    """
    for mat in matrices:
        if not check_block_sizes(
            mat.rows,
            mat.cols,
            block_sizes,
        ):
            raise ValueError(
                "The matrix elements are not contained within the "
                "block-tridiagonal structure."
            )
    if len(matrices) == 2:
        a, b = matrices
        p = a @ b
        # Corrections accounting for the fact that the matrices should
        # have open ends.
        p.blocks[0, 0] += a.blocks[1, 0] @ b.blocks[0, 1]
        p.blocks[-1, -1] += a.blocks[-2, -1] @ b.blocks[-1, -2]
        try:
            buffer.data = p.data
        except ValueError:
            # TODO: Still slow
            buffer.data = p[*buffer.spy()]
    elif len(matrices) == 3:
        a, b, c = matrices
        p = a @ b @ c
        # Corrections accounting for the fact that the matrices should
        # have open ends.
        # Left side.
        p.blocks[0, 0] += (
            a.blocks[1, 0] @ b.blocks[0, 1] @ c.blocks[0, 0]
            + a.blocks[0, 0] @ b.blocks[1, 0] @ c.blocks[0, 1]
            + a.blocks[1, 0] @ b.blocks[0, 0] @ c.blocks[0, 1]
        )
        p.blocks[0, 1] += a.blocks[1, 0] @ b.blocks[0, 1] @ c.blocks[0, 1]
        p.blocks[1, 0] += a.blocks[1, 0] @ b.blocks[1, 0] @ c.blocks[0, 1]
        # Right side.
        p.blocks[-1, -1] += (
            a.blocks[-2, -1] @ b.blocks[-1, -2] @ c.blocks[-1, -1]
            + a.blocks[-1, -1] @ b.blocks[-2, -1] @ c.blocks[-1, -2]
            + a.blocks[-2, -1] @ b.blocks[-1, -1] @ c.blocks[-1, -2]
        )
        p.blocks[-1, -2] += a.blocks[-2, -1] @ b.blocks[-1, -2] @ c.blocks[-1, -2]
        p.blocks[-2, -1] += a.blocks[-2, -1] @ b.blocks[-2, -1] @ c.blocks[-1, -2]
        try:
            buffer.data = p.data
        except ValueError:
            # TODO: Still slow
            buffer.data = p[*buffer.spy()]
    else:
        raise ValueError("Invalid number of matrices.")


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
        if not check_block_sizes(
            self.coulomb_matrix_sparray.row,
            self.coulomb_matrix_sparray.col,
            self.small_block_sizes,
        ):
            raise ValueError("Block sizes do not match Coulomb matrix.")
        self.new_block_size_factor = 3
        if len(self.small_block_sizes) % self.new_block_size_factor != 0:
            # Not implemented yet.
            raise ValueError(
                f"Number of blocks must be divisible by {self.new_block_size_factor}."
            )
        self.block_sizes = (
            self.small_block_sizes[
                : len(self.small_block_sizes) // self.new_block_size_factor
            ]
            * self.new_block_size_factor
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

        # Create a dummy identity matrix.
        # dummy_identity_data = xp.ones_like(self.coulomb_matrix_sparray.data) * 1e-16
        # dummy_identity_data[
        #     self.coulomb_matrix_sparray.row == self.coulomb_matrix_sparray.col
        # ] = 1.0
        # dummy_identity_sparray = sparse.coo_matrix(
        #     (
        #         dummy_identity_data,
        #         (self.coulomb_matrix_sparray.row, self.coulomb_matrix_sparray.col),
        #     ),
        # )
        # self.dummy_identity = compute_config.dsbsparse_type.from_sparray(
        #     dummy_identity_sparray,
        #     block_sizes=self.small_block_sizes,
        #     global_stack_shape=self.energies.shape,
        #     densify_blocks=[(i, i) for i in range(len(self.small_block_sizes))],
        # )

        # Load the device Hamiltonian for finding new sparsity pattern
        # dummy_hamiltonian = distributed_load(
        #     quatrex_config.input_dir / "hamiltonian.npz"
        # )

        sparsity_pattern = sparsity_pattern.tocsr()

        v_times_p_sparsity_pattern = sparsity_pattern @ sparsity_pattern
        # Allocate memory for the Coulomb matrix times polarization.
        self.v_times_p_retarded = compute_config.dsbsparse_type.from_sparray(
            v_times_p_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )

        l_sparsity_pattern = v_times_p_sparsity_pattern @ sparsity_pattern
        # Allocate memory for the L_lesser and L_greater matrices.
        self.l_lesser = compute_config.dsbsparse_type.from_sparray(
            l_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )
        self.l_greater = compute_config.dsbsparse_type.zeros_like(self.l_lesser)
        self.system_matrix = compute_config.dsbsparse_type.zeros_like(
            self.v_times_p_retarded
        )

        # Compute new sparsity pattern.
        # rows, cols = product_sparsity_pattern(
        #     sparse.csr_matrix(
        #         (
        #             xp.ones(self.coulomb_matrix.nnz),
        #             (self.coulomb_matrix.rows, self.coulomb_matrix.cols),
        #         )
        #     ),
        #     sparse.csr_matrix(
        #         (
        #             xp.ones_like(dummy_hamiltonian.data, dtype=xp.float64),
        #             (dummy_hamiltonian.row, dummy_hamiltonian.col),
        #         )
        #     ),
        #     sparse.csr_matrix(
        #         (
        #             xp.ones(self.coulomb_matrix.nnz),
        #             (self.coulomb_matrix.rows, self.coulomb_matrix.cols),
        #         )
        #     ),
        # )

        # Load the overlap matrix.
        # try:
        #     self.overlap_sparray = distributed_load(
        #         quatrex_config.input_dir / "overlap.npz"
        #     ).astype(xp.complex128)
        # except FileNotFoundError:
        #     # No overlap provided. Assume orthonormal basis.
        #     self.overlap_sparray = sparse.eye(
        #         self.coulomb_matrix_sparray.shape[0],
        #         format="coo",
        #         dtype=self.coulomb_matrix_sparray.dtype,
        #     )
        # self.overlap_sparray = self.overlap_sparray.tocoo()
        # # Check that the overlap matrix and Coulomb matrix match.
        # if self.overlap_sparray.shape != self.coulomb_matrix_sparray.shape:
        #     raise ValueError("Overlap matrix and Coulomb matrix have different shapes.")

        # Construct the bare system matrix.
        # self.bare_system_matrix = compute_config.dsbsparse_type.zeros_like(
        #     self.v_times_p_retarded
        # )
        # Add the overlap matrix to the bare system matrix.
        # self.bare_system_matrix += sparse.eye(sparsity_pattern.shape[0])
        # Allocate memory for the system matrix.

        # Boundary conditions.
        self.left_occupancies = bose_einstein(
            self.local_energies,
            quatrex_config.coulomb_screening.temperature,
        )
        self.right_occupancies = bose_einstein(
            self.local_energies,
            quatrex_config.coulomb_screening.temperature,
        )
        # Allocate memory for the OBC blocks.
        self.obc_blocks_left = {}
        self.obc_blocks_right = {}
        self.obc_blocks_left["diag"] = xp.zeros_like(self.system_matrix.blocks[0, 0])
        self.obc_blocks_left["right"] = xp.zeros_like(
            self.obc_blocks_left["diag"],
        )
        self.obc_blocks_left["below"] = xp.zeros_like(
            self.obc_blocks_left["diag"],
        )
        self.obc_blocks_right["diag"] = xp.zeros_like(self.system_matrix.blocks[-1, -1])
        self.obc_blocks_right["above"] = xp.zeros_like(
            self.obc_blocks_right["diag"],
        )
        self.obc_blocks_right["left"] = xp.zeros_like(
            self.obc_blocks_right["diag"],
        )

    def _set_block_sizes(self, block_sizes: NDArray) -> None:
        """Sets the block sizes of all matrices.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        self.v_times_p_retarded.block_sizes = block_sizes
        # self.bare_system_matrix.block_sizes = block_sizes
        self.system_matrix.block_sizes = block_sizes
        self.l_lesser.block_sizes = block_sizes
        self.l_greater.block_sizes = block_sizes

    def _apply_obc(self, l_lesser: DSBSparse, l_greater: DSBSparse) -> None:
        """Applies open boundary conditions.

        Parameters
        ----------
        l_lesser : DSBSparse
            The lesser quasi self-energy.
        l_greater : DSBSparse
            The greater quasi self-energy.

        """

        # Compute surface Green's functions.
        x_00 = self.obc(
            self.obc_blocks_left["diag"],
            self.obc_blocks_left["right"],
            self.obc_blocks_left["below"],
            "left",
        )
        x_nn = self.obc(
            self.obc_blocks_right["diag"],
            self.obc_blocks_right["left"],
            self.obc_blocks_right["above"],
            "right",
        )

        # Apply the retarded boundary self-energy.
        self.system_matrix.blocks[0, 0] -= (
            self.system_matrix.blocks[1, 0] @ x_00 @ self.system_matrix.blocks[0, 1]
        )
        self.system_matrix.blocks[-1, -1] -= (
            self.system_matrix.blocks[-2, -1] @ x_nn @ self.system_matrix.blocks[-1, -2]
        )

        # Compute and apply the lesser boundary self-energy.
        a_00 = self.obc_blocks_left["below"] @ x_00 @ self.l_lesser.blocks[0, 1]
        a_nn = self.obc_blocks_right["above"] @ x_nn @ self.l_lesser.blocks[-1, -2]
        w_00 = self.lyapunov(
            x_00 @ self.obc_blocks_left["below"],
            x_00
            @ (l_lesser.blocks[0, 0] - (a_00 - a_00.conj().swapaxes(-1, -2)))
            @ x_00.conj().swapaxes(-1, -2),
            "left",
        )
        w_nn = self.lyapunov(
            x_nn @ self.obc_blocks_right["above"],
            x_nn
            @ (l_lesser.blocks[-1, -1] - (a_nn - a_nn.conj().swapaxes(-1, -2)))
            @ x_nn.conj().swapaxes(-1, -2),
            "right",
        )

        l_lesser.blocks[0, 0] += self.system_matrix.blocks[
            1, 0
        ] @ w_00 @ self.system_matrix.blocks[1, 0].conj().swapaxes(-1, -2) - (
            a_00 - a_00.conj().swapaxes(-1, -2)
        )
        l_lesser.blocks[-1, -1] += self.system_matrix.blocks[
            -2, -1
        ] @ w_nn @ self.system_matrix.blocks[-2, -1].conj().swapaxes(-1, -2) - (
            a_nn - a_nn.conj().swapaxes(-1, -2)
        )

        # Compute and apply the greater boundary self-energy.
        a_00 = self.obc_blocks_left["below"] @ x_00 @ self.l_greater.blocks[0, 1]
        a_nn = self.obc_blocks_right["above"] @ x_nn @ self.l_greater.blocks[-1, -2]
        w_00 = self.lyapunov(
            x_00 @ self.obc_blocks_left["below"],
            x_00
            @ (l_greater.blocks[0, 0] - (a_00 - a_00.conj().swapaxes(-1, -2)))
            @ x_00.conj().swapaxes(-1, -2),
            "left",
        )
        w_nn = self.lyapunov(
            x_nn @ self.obc_blocks_right["above"],
            x_nn
            @ (l_greater.blocks[-1, -1] - (a_nn - a_nn.conj().swapaxes(-1, -2)))
            @ x_nn.conj().swapaxes(-1, -2),
            "right",
        )

        l_greater.blocks[0, 0] += self.system_matrix.blocks[
            1, 0
        ] @ w_00 @ self.system_matrix.blocks[1, 0].conj().swapaxes(-1, -2) - (
            a_00 - a_00.conj().swapaxes(-1, -2)
        )
        l_greater.blocks[-1, -1] += self.system_matrix.blocks[
            -2, -1
        ] @ w_nn @ self.system_matrix.blocks[-2, -1].conj().swapaxes(-1, -2) - (
            a_nn - a_nn.conj().swapaxes(-1, -2)
        )

    def _assemble_system_matrix(self, v_times_p_retarded: DSBSparse) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data = 0.0
        self.system_matrix += sparse.eye(self.system_matrix.shape[-1])
        self.system_matrix -= v_times_p_retarded

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
        # obc_multiply(
        #     self.v_times_p_retarded,
        #     (self.coulomb_matrix, p_retarded, self.dummy_identity),
        #     self.small_block_sizes,
        # )
        # obc_multiply(
        #     self.l_lesser,
        #     (self.coulomb_matrix, p_lesser, self.coulomb_matrix),
        #     self.small_block_sizes,
        # )
        # obc_multiply(
        #     self.l_greater,
        #     (self.coulomb_matrix, p_greater, self.coulomb_matrix),
        #     self.small_block_sizes,
        # )

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
        self._assemble_system_matrix(self.v_times_p_retarded)
        # Assemble the boundary blocks.
        assemble_boundary_blocks(
            self.obc_blocks_left["diag"],
            self.obc_blocks_left["right"],
            self.obc_blocks_left["below"],
            self.obc_blocks_right["diag"],
            self.obc_blocks_right["above"],
            self.obc_blocks_right["left"],
            self.system_matrix,
            self.new_block_size_factor,
        )
        # Go back to normal block sizes.
        self._set_block_sizes(self.block_sizes)
        t_assemble = time.perf_counter() - times.pop()

        # Apply the OBC algorithm.
        times.append(time.perf_counter())
        self._apply_obc(self.l_lesser, self.l_greater)
        t_obc = time.perf_counter() - times.pop()

        # Solve the system
        times.append(time.perf_counter())
        self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=self.l_lesser,
            sigma_greater=self.l_greater,
            out=out,
            return_retarded=True,
        )
        t_solve = time.perf_counter() - times.pop()

        # Compute the retarded Screened interaction (mainly used for debugging).
        # Currently doesn't work.
        # obc_multiply(out[2], (out[2], self.coulomb_matrix), self.block_sizes)

        w_lesser, w_greater, _ = out
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


# TODO: Maybe implement another abstract class that don't necessitates solve with lesser and greater
class CoulombScreeningSolver_X(SubsystemSolver):
    """Solves the dynamics of the screened Coulomb interaction.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies_x : NDArray
        The energies at which to solve.

    """

    system = "coulomb_screening"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies_x: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the solver."""
        super().__init__(quatrex_config, compute_config, energies_x)

        # Load the Coulomb matrix.
        self.coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)

        # Load block sizes.
        self.small_block_sizes = distributed_load(
            quatrex_config.input_dir / "block_sizes.npy"
        ).astype(xp.int32)
        if not check_block_sizes(
            self.coulomb_matrix_sparray.row,
            self.coulomb_matrix_sparray.col,
            self.small_block_sizes,
        ):
            raise ValueError("Block sizes do not match Coulomb matrix.")
        self.new_block_size_factor = 3
        if len(self.small_block_sizes) % self.new_block_size_factor != 0:
            # Not implemented yet.
            raise ValueError(
                f"Number of blocks must be divisible by {self.new_block_size_factor}."
            )
        self.block_sizes = (
            self.small_block_sizes[
                : len(self.small_block_sizes) // self.new_block_size_factor
            ]
            * self.new_block_size_factor
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

        sparsity_pattern = sparsity_pattern.tocsr()

        v_times_p_sparsity_pattern = sparsity_pattern @ sparsity_pattern
        # Allocate memory for the Coulomb matrix times polarization.
        self.v_times_p_retarded = compute_config.dsbsparse_type.from_sparray(
            v_times_p_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )

        l_sparsity_pattern = v_times_p_sparsity_pattern @ sparsity_pattern
        # Allocate memory for the L_lesser and L_greater matrices.
        self.l_lesser = compute_config.dsbsparse_type.from_sparray(
            l_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )
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
        # Allocate memory for the OBC blocks.
        self.obc_blocks_left = {}
        self.obc_blocks_right = {}
        self.obc_blocks_left["diag"] = xp.zeros_like(self.system_matrix.blocks[0, 0])
        self.obc_blocks_left["right"] = xp.zeros_like(
            self.obc_blocks_left["diag"],
        )
        self.obc_blocks_left["below"] = xp.zeros_like(
            self.obc_blocks_left["diag"],
        )
        self.obc_blocks_right["diag"] = xp.zeros_like(self.system_matrix.blocks[-1, -1])
        self.obc_blocks_right["above"] = xp.zeros_like(
            self.obc_blocks_right["diag"],
        )
        self.obc_blocks_right["left"] = xp.zeros_like(
            self.obc_blocks_right["diag"],
        )

    def _set_block_sizes_x(self, block_sizes: NDArray) -> None:
        """Sets the block sizes of all matrices.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        self.v_times_p_retarded.block_sizes = block_sizes
        self.system_matrix.block_sizes = block_sizes
        self.l_lesser.block_sizes = block_sizes

    def _apply_obc_x(self, l_lesser: DSBSparse) -> None:
        """Applies open boundary conditions.

        Parameters
        ----------
        l_lesser : DSBSparse
            The lesser quasi self-energy.

        """

        # Compute surface Green's functions.
        x_00 = self.obc(
            self.obc_blocks_left["diag"],
            self.obc_blocks_left["right"],
            self.obc_blocks_left["below"],
            "left",
        )
        x_nn = self.obc(
            self.obc_blocks_right["diag"],
            self.obc_blocks_right["left"],
            self.obc_blocks_right["above"],
            "right",
        )

        # Apply the retarded boundary self-energy.
        self.system_matrix.blocks[0, 0] -= (
            self.system_matrix.blocks[1, 0] @ x_00 @ self.system_matrix.blocks[0, 1]
        )
        self.system_matrix.blocks[-1, -1] -= (
            self.system_matrix.blocks[-2, -1] @ x_nn @ self.system_matrix.blocks[-1, -2]
        )

        # Compute and apply the lesser boundary self-energy.
        a_00 = self.obc_blocks_left["below"] @ x_00 @ self.l_lesser.blocks[0, 1]
        a_nn = self.obc_blocks_right["above"] @ x_nn @ self.l_lesser.blocks[-1, -2]
        w_00 = self.lyapunov(
            x_00 @ self.obc_blocks_left["below"],
            x_00
            @ (l_lesser.blocks[0, 0] - (a_00 - a_00.conj().swapaxes(-1, -2)))
            @ x_00.conj().swapaxes(-1, -2),
            "left",
        )
        w_nn = self.lyapunov(
            x_nn @ self.obc_blocks_right["above"],
            x_nn
            @ (l_lesser.blocks[-1, -1] - (a_nn - a_nn.conj().swapaxes(-1, -2)))
            @ x_nn.conj().swapaxes(-1, -2),
            "right",
        )

        l_lesser.blocks[0, 0] += self.system_matrix.blocks[
            1, 0
        ] @ w_00 @ self.system_matrix.blocks[1, 0].conj().swapaxes(-1, -2) - (
            a_00 - a_00.conj().swapaxes(-1, -2)
        )
        l_lesser.blocks[-1, -1] += self.system_matrix.blocks[
            -2, -1
        ] @ w_nn @ self.system_matrix.blocks[-2, -1].conj().swapaxes(-1, -2) - (
            a_nn - a_nn.conj().swapaxes(-1, -2)
        )

    def _assemble_system_matrix(self, v_times_p_retarded: DSBSparse) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data = 0.0
        self.system_matrix += sparse.eye(self.system_matrix.shape[-1])
        self.system_matrix -= v_times_p_retarded

    def solve(
        self,
        p_lesser: DSBSparse,
        p_stupid: DSBSparse,
        p_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Solves for the screened Coulomb interaction.

        Parameters
        ----------
        p_lesser : DSBSparse
            The lesser polarization.
        p_retarded : DSBSparse
            The retarded polarization.
        out : tuple[DSBSparse, ...]
            The output matrices. The order is (lesser, retarded).

        """
        times = []

        # Compute the product of the Coulomb matrix with the polarization.
        times.append(time.perf_counter())
        # Change the block sizes to match the Coulomb matrix.
        self._set_block_sizes_x(self.small_block_sizes)

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

        t_obc_multiply = time.perf_counter() - times.pop()

        # Assemble the matrices.
        times.append(time.perf_counter())
        # Assemble the system matrix.
        self._assemble_system_matrix(self.v_times_p_retarded)
        # Assemble the boundary blocks.
        assemble_boundary_blocks(
            self.obc_blocks_left["diag"],
            self.obc_blocks_left["right"],
            self.obc_blocks_left["below"],
            self.obc_blocks_right["diag"],
            self.obc_blocks_right["above"],
            self.obc_blocks_right["left"],
            self.system_matrix,
            self.new_block_size_factor,
        )
        # Go back to normal block sizes.
        self._set_block_sizes_x(self.block_sizes)
        t_assemble = time.perf_counter() - times.pop()

        # Apply the OBC algorithm.
        times.append(time.perf_counter())
        self._apply_obc_x(self.l_lesser)
        t_obc = time.perf_counter() - times.pop()

        # Solve the system
        times.append(time.perf_counter())
        # TODO: Implement solve_x
        # self.solver.selected_solve_x(
        #     a=self.system_matrix,
        #     sigma_lesser=self.l_lesser,
        #     out=out,
        #     return_retarded=True,
        # )
        self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=self.l_lesser,
            sigma_greater=self.l_lesser,
            out=(out[0], out[0], out[1]),
            return_retarded=True,
        )
        t_solve = time.perf_counter() - times.pop()

        # Compute the retarded Screened interaction (mainly used for debugging).
        # Currently doesn't work.
        # obc_multiply(out[2], (out[2], self.coulomb_matrix), self.block_sizes)

        w_lesser, _ = out
        w_lesser.data = 0.5 * (
            w_lesser.data - w_lesser.ltranspose(copy=True).data.conj()
        )

        (
            print(
                f"OBC Multiply: {t_obc_multiply}, Assemble: {t_assemble}, OBC: {t_obc}, Solve: {t_solve}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )
