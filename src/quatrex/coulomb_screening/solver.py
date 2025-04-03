# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NCCL_AVAILABLE, USE_NCCL, NDArray, _DType, host_xp, nccl_comm, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.datastructures.routines import bd_matmul, bd_sandwich
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import get_host, synchronize_device
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from qttools.utils.sparse_utils import product_sparsity_pattern_dsdbsparse

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import (
    compute_num_connected_blocks,
    get_periodic_superblocks,
    homogenize,
)

profiler = Profiler()


@profiler.profile(level="debug")
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


@profiler.profile(level="debug")
def _spillover_matmul(
    a: sparse.spmatrix, b: sparse.spmatrix, block_sizes
) -> sparse.coo_matrix:
    """Multiplies two sparse matrices with spillover correction."""
    c = (a @ b).tocsr()

    a = a.tocsr()
    b = b.tocsr()

    # Left spillover
    i_ = slice(None, int(block_sizes[0]))
    j_ = slice(int(block_sizes[0]), int(sum(block_sizes[:2])))
    c[i_, i_] += a[j_, i_] @ b[i_, j_]

    # Right spillover
    i_ = slice(int(-block_sizes[-1]), None)
    j_ = slice(int(-sum(block_sizes[-2:])), int(-block_sizes[-1]))
    c[i_, i_] += a[j_, i_] @ b[i_, j_]

    return c.tocoo()


@profiler.profile(level="debug")
def _compute_sparsity_pattern(
    *matrices: DSBSparse, dtype: _DType = None
) -> sparse.coo_matrix:
    """Computes the sparsity pattern of the product of several DSBSparse matrices."""
    rows, cols = product_sparsity_pattern_dsdbsparse(*matrices, spillover=True)
    shape = matrices[0].shape[-2:]
    dtype = dtype or matrices[0].dtype
    return sparse.coo_matrix(
        (xp.ones_like(rows), (rows, cols)), shape=shape, dtype=dtype
    )


@decorate_methods(profiler.profile(level="api"), exclude=["solve"])
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
        if quatrex_config.device.construct_from_unit_cell:
            coulomb_matrix_unit_cells = distributed_load(
                quatrex_config.input_dir / "coulomb_matrix_unit_cells.npy"
            ).astype(xp.complex128)
            self.small_block_sizes = host_xp.asarray(
                [
                    coulomb_matrix_unit_cells.shape[-1]
                    * quatrex_config.device.unit_cell_per_supercell[
                        "xyz".index(quatrex_config.device.transport_direction)
                    ]
                ]
                * quatrex_config.device.number_of_supercells
            )

        else:
            # Load block sizes.
            self.small_block_sizes = get_host(
                distributed_load(quatrex_config.input_dir / "block_sizes.npy").astype(
                    xp.int32
                )
            )

        # The coulomb matrix is only used to compute the sparsity
        # pattern of the system matrix and the l_lesser and l_greater
        # matrices. Will convert to complex128 later.
        self.coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.float32),
            block_sizes=self.small_block_sizes,
            global_stack_shape=(comm.size,),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=xp.conj,
        )

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
        if self.small_block_sizes.sum() != self.coulomb_matrix.shape[-2]:
            raise ValueError(
                "Block sizes do not match Coulomb matrix. "
                f"{self.small_block_sizes.sum()} != {self.coulomb_matrix.shape[-2]}"
            )

        v_times_p_sparsity_pattern = _compute_sparsity_pattern(
            self.coulomb_matrix, self.coulomb_matrix, dtype=xp.float32
        )

        # Allocate memory for the System matrix (1 - V @ P).
        self.system_matrix = compute_config.dsbsparse_type.from_sparray(
            v_times_p_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
        )
        self.system_matrix.data = 0.0
        # Explicitely try to free the memory for the sparsity pattern.
        del v_times_p_sparsity_pattern

        l_sparsity_pattern = _compute_sparsity_pattern(
            self.coulomb_matrix,
            self.coulomb_matrix,
            self.coulomb_matrix,
            dtype=xp.float32,
        )

        # Allocate memory for the L_lesser and L_greater matrices.
        self.l_lesser = compute_config.dsbsparse_type.from_sparray(
            l_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )
        self.l_lesser.data = 0.0
        # Explicitely try to free the memory for the sparsity pattern.
        del l_sparsity_pattern

        self.l_greater = compute_config.dsbsparse_type.zeros_like(self.l_lesser)

        # Load the Coulomb matrix.
        if quatrex_config.device.construct_from_unit_cell:
            coulomb_matrix_sparray, __ = create_hamiltonian(
                cutoff_hr(
                    coulomb_matrix_unit_cells,
                    R_cutoff=quatrex_config.device.unit_cell_per_supercell,
                ),
                quatrex_config.device.number_of_supercells,
                quatrex_config.device.transport_direction,
                quatrex_config.device.unit_cell_per_supercell,
                return_sparse=True,
            )
            coulomb_matrix_sparray = coulomb_matrix_sparray.astype(xp.complex128)
            coulomb_matrix_sparray.sum_duplicates()

        else:
            coulomb_matrix_sparray = distributed_load(
                quatrex_config.input_dir / "coulomb_matrix.npz"
            ).astype(xp.complex128)

        self.coulomb_matrix._data = xp.zeros_like(
            self.coulomb_matrix._data, dtype=xp.complex128
        )
        self.coulomb_matrix.dtype = self.coulomb_matrix._data.dtype
        self.coulomb_matrix += coulomb_matrix_sparray
        # Explicitely try to free the memory for the sparsity pattern.
        del coulomb_matrix_sparray

        # Make sure the Coulomb matrix is hermitian.
        self.coulomb_matrix.symmetrize()
        self.coulomb_matrix._data /= quatrex_config.coulomb_screening.epsilon_r

        # Boundary conditions.
        self.left_occupancies = bose_einstein(
            self.local_energies,
            quatrex_config.coulomb_screening.temperature,
        )
        self.right_occupancies = bose_einstein(
            self.local_energies,
            quatrex_config.coulomb_screening.temperature,
        )

        self.dos_peak_limit = quatrex_config.coulomb_screening.dos_peak_limit

        self.obc_blocks = OBCBlocks(num_blocks=self.block_sizes.size)

        self.block_sections = quatrex_config.coulomb_screening.obc.block_sections

        self.flatband = quatrex_config.electron.flatband
        self.solve_call_count = 0
        self.filtering_iteration_limit = (
            quatrex_config.coulomb_screening.filtering_iteration_limit
        )

    def _set_block_sizes(self, block_sizes: NDArray) -> None:
        """Sets the block sizes of all matrices.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        self.system_matrix.block_sizes = block_sizes
        self.l_lesser.block_sizes = block_sizes
        self.l_greater.block_sizes = block_sizes

    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""

        t_obc_r_start = time.perf_counter()

        m_10, m_00, m_01 = get_periodic_superblocks(
            a_ii=self.system_matrix.blocks[0, 0],
            a_ji=self.system_matrix.blocks[1, 0],
            a_ij=self.system_matrix.blocks[0, 1],
            block_sections=self.block_sections,
        )
        m_mn, m_nn, m_nm = get_periodic_superblocks(
            # Twist it, flip it, ...
            a_ii=xp.flip(self.system_matrix.blocks[-1, -1], axis=(-2, -1)),
            a_ji=xp.flip(self.system_matrix.blocks[-2, -1], axis=(-2, -1)),
            a_ij=xp.flip(self.system_matrix.blocks[-1, -2], axis=(-2, -1)),
            block_sections=self.block_sections,
        )
        # ... bop it.
        m_nn = xp.flip(m_nn, axis=(-2, -1))
        m_nm = xp.flip(m_nm, axis=(-2, -1))
        m_mn = xp.flip(m_mn, axis=(-2, -1))

        x_00 = self.obc(a_ii=m_00, a_ij=m_01, a_ji=m_10, contact="left")
        x_nn = self.obc(
            # Twist it, flip it, ...
            a_ii=xp.flip(m_nn, axis=(-2, -1)),
            a_ij=xp.flip(m_nm, axis=(-2, -1)),
            a_ji=xp.flip(m_mn, axis=(-2, -1)),
            contact="right",
        )
        # ... bop it.
        x_nn = xp.flip(x_nn, axis=(-2, -1))

        m_10_x_00 = m_10 @ x_00
        m_mn_x_nn = m_mn @ x_nn
        self.obc_blocks.retarded[0] = m_10_x_00 @ m_01
        self.obc_blocks.retarded[-1] = m_mn_x_nn @ m_nm

        synchronize_device()
        t_obc_r_end = time.perf_counter()
        comm.Barrier()
        t_obc_r_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"        OBC retarded: {t_obc_r_end-t_obc_r_start:.3f}", flush=True)
            print(
                f"        OBC retarded all: {t_obc_r_end_all-t_obc_r_start:.3f}",
                flush=True,
            )

        t_lyapunov_start = time.perf_counter()
        # Compute and apply the left lesser/greater boundary self-energy.
        a_00_lesser = m_10_x_00 @ self.l_lesser.blocks[0, 1]
        a_00_greater = m_10_x_00 @ self.l_greater.blocks[0, 1]

        q_00_lesser = (
            x_00
            @ (
                self.l_lesser.blocks[0, 0]
                - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))
            )
            @ x_00.conj().swapaxes(-1, -2)
        )
        q_00_greater = (
            x_00
            @ (
                self.l_greater.blocks[0, 0]
                - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))
            )
            @ x_00.conj().swapaxes(-1, -2)
        )

        b_00 = x_00 @ m_10
        q_00 = xp.stack((q_00_lesser, q_00_greater))

        w_00_lesser, w_00_greater = self.lyapunov(b_00, q_00, "left")

        self.obc_blocks.lesser[0] = m_10 @ w_00_lesser @ m_10.conj().swapaxes(
            -1, -2
        ) - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))

        self.obc_blocks.greater[0] = m_10 @ w_00_greater @ m_10.conj().swapaxes(
            -1, -2
        ) - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))

        # Compute and apply the right lesser/greater boundary self-energy.
        a_nn_lesser = m_mn_x_nn @ self.l_lesser.blocks[-1, -2]
        a_nn_greater = m_mn_x_nn @ self.l_greater.blocks[-1, -2]

        q_nn_lesser = (
            x_nn
            @ (
                self.l_lesser.blocks[-1, -1]
                - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))
            )
            @ x_nn.conj().swapaxes(-1, -2)
        )
        q_nn_greater = (
            x_nn
            @ (
                self.l_greater.blocks[-1, -1]
                - (a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2))
            )
            @ x_nn.conj().swapaxes(-1, -2)
        )

        b_nn = x_nn @ m_mn

        q_nn = xp.stack((q_nn_lesser, q_nn_greater))

        w_nn_lesser, w_nn_greater = self.lyapunov(b_nn, q_nn, "right")

        self.obc_blocks.lesser[-1] = m_mn @ w_nn_lesser @ m_mn.conj().swapaxes(
            -1, -2
        ) - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))

        self.obc_blocks.greater[-1] = m_mn @ w_nn_greater @ m_mn.conj().swapaxes(
            -1, -2
        ) - (a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2))

        synchronize_device()
        t_lyapunov_end = time.perf_counter()
        comm.Barrier()
        t_lyapunov_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"        Lyapunov: {t_lyapunov_end-t_lyapunov_start:.3f}", flush=True
            )
            print(
                f"        Lyapunov all: {t_lyapunov_end_all-t_lyapunov_start:.3f}",
                flush=True,
            )

    def _assemble_system_matrix(self, p_retarded: DSBSparse) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data = 0.0

        bd_matmul(
            self.coulomb_matrix,
            p_retarded,
            out=self.system_matrix,
            spillover_correction=True,
        )

        # self.system_matrix._data = -self.system_matrix._data
        xp.negative(self.system_matrix._data, out=self.system_matrix._data)
        self.system_matrix += sparse.eye(self.system_matrix.shape[-1])

    def _filter_peaks(self, out: tuple[DSBSparse, ...]) -> None:
        """Filters out peaks in the Green's functions.

        Parameters
        ----------
        out : tuple[DSBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """

        w_lesser, w_greater, *__ = out
        local_dos = []
        for b in range(w_lesser.num_blocks):
            w_lesser_density = xp.diagonal(
                w_lesser.blocks[b, b], axis1=-2, axis2=-1
            ).imag.mean(-1)
            w_greater_density = (
                -xp.diagonal(w_greater.blocks[b, b], axis1=-2, axis2=-1).imag
            ).mean(-1)
            local_dos.append(0.5 * (w_greater_density - w_lesser_density))

        if not NCCL_AVAILABLE or not USE_NCCL:
            dos = xp.hstack(comm.allgather(local_dos))
        else:
            # NOTE: NCCL does not expose all_gather_v. This is a hack.
            local_dos = xp.array(local_dos)
            pad_width = w_lesser.total_stack_size // comm.size - local_dos.shape[1]
            local_dos = xp.pad(local_dos, ((0, 0), (0, pad_width)))
            dos = xp.empty(
                (local_dos.shape[0], w_lesser.total_stack_size), dtype=local_dos.dtype
            )
            synchronize_device()
            nccl_comm.all_gather(local_dos, dos, local_dos.size)
            synchronize_device()
            dos = dos[..., w_lesser._stack_padding_mask]

        dos_gradient = xp.abs(xp.gradient(dos, self.energies, axis=1))
        mask = xp.max(dos_gradient, axis=0) > self.dos_peak_limit

        section_sizes, __ = get_section_sizes(self.energies.size, comm.size)
        section_offsets = host_xp.hstack(([0], host_xp.cumsum(section_sizes)))
        local_mask = mask[section_offsets[comm.rank] : section_offsets[comm.rank + 1]]

        w_lesser.data[local_mask] = 0.0
        w_greater.data[local_mask] = 0.0

    @profiler.profile(level="basic")
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
        t_set_blocksize_start = time.perf_counter()

        self.system_matrix.allocate_data()
        self.l_lesser.allocate_data()
        self.l_greater.allocate_data()

        # Change the block sizes to match the Coulomb matrix.
        self._set_block_sizes(self.small_block_sizes)
        synchronize_device()
        t_set_blocksize_end = time.perf_counter()
        comm.Barrier()
        t_set_blocksize_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Set block sizes: {t_set_blocksize_end-t_set_blocksize_start:.3f}",
                flush=True,
            )
            print(
                f"    Set block sizes all: {t_set_blocksize_end_all-t_set_blocksize_start:.3f}",
                flush=True,
            )

        # Compute the product of the Coulomb matrix with the polarization.

        # Assemble the system matrix (Includes matrix multiplication).
        t_assembly_start = time.perf_counter()
        self._assemble_system_matrix(p_retarded)
        synchronize_device()
        t_assembly_end = time.perf_counter()
        comm.Barrier()
        t_assembly_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Assembly: {t_assembly_end-t_assembly_start:.3f}", flush=True)
            print(
                f"    Assembly all: {t_assembly_end_all-t_assembly_start:.3f}",
                flush=True,
            )

        t_sandwich_start = time.perf_counter()
        bd_sandwich(
            self.coulomb_matrix,
            p_lesser,
            out=self.l_lesser,
            spillover_correction=True,
        )
        bd_sandwich(
            self.coulomb_matrix,
            p_greater,
            out=self.l_greater,
            spillover_correction=True,
        )
        synchronize_device()
        t_sandwich_end = time.perf_counter()
        comm.Barrier()
        t_sandwich_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Sandwich: {t_sandwich_end-t_sandwich_start:.3f}", flush=True)
            print(
                f"    Sandwich all: {t_sandwich_end_all-t_sandwich_start:.3f}",
                flush=True,
            )

        if self.flatband:
            t_homogenize_start = time.perf_counter()
            homogenize(self.system_matrix)
            homogenize(self.l_lesser)
            homogenize(self.l_greater)
            t_homogenize_end = time.perf_counter()
            comm.Barrier()
            t_homogenize_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"    Homogenize: {t_homogenize_end-t_homogenize_start:.3f}",
                    flush=True,
                )
                print(
                    f"    Homogenize all: {t_homogenize_end_all-t_homogenize_start:.3f}",
                    flush=True,
                )

        # Go back to normal block sizes.
        t_set_blocksize_start = time.perf_counter()
        self._set_block_sizes(self.block_sizes)
        t_set_blocksize_end = time.perf_counter()
        comm.Barrier()
        t_set_blocksize_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Set block sizes: {t_set_blocksize_end-t_set_blocksize_start:.3f}",
                flush=True,
            )
            print(
                f"    Set block sizes all: {t_set_blocksize_end_all-t_set_blocksize_start:.3f}",
                flush=True,
            )

        # Apply the OBC algorithm.
        t_obc_start = time.perf_counter()
        self._compute_obc()
        synchronize_device()
        t_obc_end = time.perf_counter()
        comm.Barrier()
        t_obc_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    OBC: {t_obc_end-t_obc_start:.3f}", flush=True)
            print(f"    OBC all: {t_obc_end_all-t_obc_start:.3f}", flush=True)

        # Solve the system
        t_solve_start = time.perf_counter()
        self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=self.l_lesser,
            sigma_greater=self.l_greater,
            obc_blocks=self.obc_blocks,
            out=out,
            return_retarded=False,
        )
        synchronize_device()
        t_solve_end = time.perf_counter()
        comm.Barrier()
        t_solve_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Solve: {t_solve_end-t_solve_start:.3f}", flush=True)
            print(f"    Solve all: {t_solve_end_all-t_solve_start:.3f}", flush=True)

        t_filter_start = time.perf_counter()
        # Only filter the peaks for the first few iterations.
        if self.solve_call_count < self.filtering_iteration_limit:
            self._filter_peaks(out)

        self.system_matrix.free_data()
        self.l_lesser.free_data()
        self.l_greater.free_data()

        w_lesser, w_greater, *__ = out
        if comm.rank == 0:
            w_greater.data[0, :] = 0.0
            w_lesser.data[0, :] = 0.0

        synchronize_device()
        t_filter_end = time.perf_counter()
        comm.Barrier()
        t_filter_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Filter: {t_filter_end-t_filter_start:.3f}", flush=True)
            print(f"    Filter all: {t_filter_end_all-t_filter_start:.3f}", flush=True)

        self.solve_call_count += 1
