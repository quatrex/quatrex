# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import (
    create_stream,
    debug_gpu_memory_usage,
    free_mempool,
    get_host,
    synchronize_device,
    synchronize_stream,
)
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from qttools.utils.solvers_utils import get_batches
from qttools.utils.stack_utils import scale_stack
from quatrex.bandstructure.band_edges import (
    find_band_edges,
    find_dos_peaks,
    find_renormalized_eigenvalues,
)
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import get_periodic_superblocks, homogenize

profiler = Profiler()


@profiler.profile(level="debug")
def _btd_subtract(a: DSDBSparse, b: DSDBSparse) -> None:
    """Subtracts b from a on the block-tridiagonal.

    This is an in-place operation, i.e. a is modified.

    Parameters
    ----------
    a : DSDBSparse
        The matrix to subtract from.
    b : DSDBSparse
        The matrix to subtract.

    """
    a_ = a.stack[...]
    b_ = b.stack[...]
    for i in range(a.num_local_blocks):
        j = i + 1
        a_.blocks[i, i] -= xp.asarray(b_.blocks[i, i])

        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        a_.blocks[i, j] -= xp.asarray(b_.blocks[i, j])
        a_.blocks[j, i] -= xp.asarray(b_.blocks[j, i])


@decorate_methods(profiler.profile(level="api"), exclude=["solve"])
class ElectronSolver(SubsystemSolver):
    """Solves the electron dynamics.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies : np.ndarray
        The energies at which to solve.

    """

    system = "electron"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: NDArray,
        sparsity_pattern: sparse.coo_matrix = None,
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(quatrex_config, compute_config, energies)

        self.local_energies = get_local_slice(energies, comm.stack)

        debug_gpu_memory_usage("Before constructing hamiltonian sparsity pattern")

        # Load the device Hamiltonian.
        synchronize_device()
        comm.barrier()
        t_ham_load_start = time.perf_counter()
        if quatrex_config.device.construct_from_unit_cell:
            hamiltonian_unit_cells = distributed_load(
                quatrex_config.input_dir / "hamiltonian_unit_cells.npy"
            ).astype(xp.complex128)

            debug_gpu_memory_usage("After loading hamiltonian unit cells")

            # Determine the local slice of the data.
            # NOTE: This is arrow-wise partitioning.
            # TODO: Allow more options, e.g., block row-wise partitioning.
            section_sizes, __ = get_section_sizes(
                quatrex_config.device.number_of_supercells, comm.block.size
            )
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            start_block = section_offsets[comm.block.rank]
            end_block = section_offsets[comm.block.rank + 1]

            hamiltonian_sparray, block_sizes = create_hamiltonian(
                cutoff_hr(
                    hamiltonian_unit_cells,
                    R_cutoff=quatrex_config.device.unit_cell_per_supercell,
                ),
                quatrex_config.device.number_of_supercells,
                quatrex_config.device.transport_direction,
                quatrex_config.device.unit_cell_per_supercell,
                block_start=start_block,
                block_end=end_block,
                return_sparse=True,
            )
            debug_gpu_memory_usage(
                "After creating initial hamiltonian sparsity pattern"
            )
            hamiltonian_sparray = hamiltonian_sparray.astype(xp.complex128)
            free_mempool()
            debug_gpu_memory_usage(
                "After converting hamiltonian sparsity pattern to complex128"
            )
            hamiltonian_sparray.sum_duplicates()
            free_mempool()
            debug_gpu_memory_usage(
                "After summing duplicates in hamiltonian sparsity pattern"
            )
            block_sizes = get_host(block_sizes)
            self.block_sizes = np.asarray(
                [block_sizes[0]] * quatrex_config.device.number_of_supercells
            )

        else:
            hamiltonian_sparray = distributed_load(
                quatrex_config.input_dir / "hamiltonian.npz"
            ).astype(xp.complex128)
            self.block_sizes = get_host(
                distributed_load(quatrex_config.input_dir / "block_sizes.npy")
            )

        debug_gpu_memory_usage("After constructing hamiltonian sparsity pattern")
        if global_comm.rank == 0:
            print(
                f"    Hamiltonian sparsity pattern shape: {hamiltonian_sparray.shape}",
                flush=True,
            )
            print(f"    Hamiltonian nnz: {hamiltonian_sparray.nnz}", flush=True)
            print(f"    Hamiltonian dtype: {hamiltonian_sparray.dtype}", flush=True)
            print(
                f"    Hamiltonian rows dtype: {hamiltonian_sparray.row.dtype}",
                flush=True,
            )
            print(
                f"    Hamiltonian cols dtype: {hamiltonian_sparray.col.dtype}",
                flush=True,
            )
            print(f"    Block sizes: {self.block_sizes}", flush=True)

        synchronize_device()
        t_ham_load_end = time.perf_counter()
        comm.barrier()
        t_ham_load_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Load Hamiltonian: {t_ham_load_end-t_ham_load_start}",
                flush=True,
            )
            print(
                f"    Load Hamiltonian all: {t_ham_load_end_all-t_ham_load_start}",
                flush=True,
            )

        self.hamiltonian = compute_config.dsdbsparse_type.from_sparray(
            hamiltonian_sparray.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=(comm.stack.size,),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=xp.conj,
        )
        self.hamiltonian.to_host()

        debug_gpu_memory_usage("After creating Hamiltonian DSDBSparse")

        synchronize_device()
        t_ham_create_end = time.perf_counter()
        comm.barrier()
        t_ham_create_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Create Hamiltonian: {t_ham_create_end-t_ham_load_end_all}",
                flush=True,
            )
            print(
                f"    Create Hamiltonian all: {t_ham_create_end_all-t_ham_load_end_all}",
                flush=True,
            )

        # Make sure that the the system matrix sparsity is a superset of
        # self-energy and Hamiltonian sparsity.
        if sparsity_pattern is None:
            sparsity_pattern = hamiltonian_sparray.copy()
        else:
            sparsity_pattern += hamiltonian_sparray
        del hamiltonian_sparray
        free_mempool()
        sparsity_pattern.sum_duplicates()
        sparsity_pattern = sparsity_pattern.astype(xp.complex128)
        sparsity_pattern = sparsity_pattern.tocoo()
        free_mempool()

        debug_gpu_memory_usage("After constructing system matrix sparsity pattern")

        # Allocate memory for the system matrix.
        self.system_matrix = compute_config.dsdbsparse_type.from_sparray(
            # sparsity_pattern.astype(xp.complex128),
            sparsity_pattern,
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            allocate=False,
        )
        del sparsity_pattern
        free_mempool()

        debug_gpu_memory_usage("After creating system matrix DSDBSparse")

        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.hamiltonian.shape[-2]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.hamiltonian.shape[-2]}"
            )

        if quatrex_config.device.construct_from_unit_cell:
            try:
                overlap_unit_cells = distributed_load(
                    quatrex_config.input_dir / "overlap_unit_cells.npy"
                ).astype(xp.complex128)
                overlap_sparray, __ = create_hamiltonian(
                    cutoff_hr(
                        overlap_unit_cells,
                        R_cutoff=quatrex_config.device.unit_cell_per_supercell,
                    ),
                    quatrex_config.device.number_of_supercells,
                    quatrex_config.device.transport_direction,
                    quatrex_config.device.unit_cell_per_supercell,
                    return_sparse=True,
                )
                self.overlap_sparray = overlap_sparray.astype(xp.complex128)
            except FileNotFoundError:
                # No overlap provided. Assume orthonormal basis.
                self.overlap_sparray = sparse.eye(
                    self.hamiltonian.shape[-2],
                    format="coo",
                    dtype=self.hamiltonian.dtype,
                )

        else:
            # Load the overlap matrix.
            try:
                self.overlap_sparray = distributed_load(
                    quatrex_config.input_dir / "overlap.npz"
                ).astype(xp.complex128)
            except FileNotFoundError:
                # No overlap provided. Assume orthonormal basis.
                self.overlap_sparray = sparse.eye(
                    self.hamiltonian.shape[-2],
                    format="coo",
                    dtype=self.hamiltonian.dtype,
                )

        # Check that the overlap matrix and Hamiltonian matrix match.
        if self.overlap_sparray.shape != self.hamiltonian.shape[-2:]:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )

        # Make sure that the Hamiltonian and overlap matrices are
        # Hermitian.
        if not self.hamiltonian.symmetry:
            self.hamiltonian.symmetrize()
        self.overlap_sparray = (
            0.5 * (self.overlap_sparray + self.overlap_sparray.conj().T)
        ).tocoo()

        # Load the potential.
        try:
            self.potential = distributed_load(
                quatrex_config.input_dir / "potential.npy"
            )
            if self.potential.size != self.hamiltonian.shape[-2]:
                raise ValueError(
                    "Potential matrix and Hamiltonian have different shapes."
                )
        except FileNotFoundError:
            # No potential provided. Assume zero potential.
            self.potential = xp.zeros(
                self.hamiltonian.shape[-2], dtype=self.hamiltonian.dtype
            )
        self.eta = quatrex_config.electron.eta

        # Contacts.
        self.flatband = quatrex_config.electron.flatband
        if self.flatband and comm.rank == 0:
            print("Flatband conditions detected", flush=True)

        self.eta_obc = quatrex_config.electron.eta_obc

        if quatrex_config.electron.solver.compute_current and comm.block.size > 1:
            raise NotImplementedError(
                "Current computation not implemented in distributed mode."
            )

        self.compute_meir_wingreen_current = (
            quatrex_config.electron.solver.compute_current
        )

        self.dos_peak_limit = quatrex_config.electron.dos_peak_limit

        # Band edges and Fermi levels.
        # TODO: This only works for small potential variations accross
        # the device.
        # TODO: During this initialization we should compute the contact
        # band structures and extract the correct fermi levels & band
        # edges from there.
        self.band_edge_tracking = quatrex_config.electron.band_edge_tracking
        self.delta_fermi_level_conduction_band = (
            quatrex_config.electron.conduction_band_edge
            - quatrex_config.electron.fermi_level
        )
        self.left_mid_gap_energy = 0.5 * (
            quatrex_config.electron.conduction_band_edge
            + quatrex_config.electron.valence_band_edge
        )
        self.left_fermi_level = quatrex_config.electron.left_fermi_level
        self.right_fermi_level = quatrex_config.electron.right_fermi_level

        potential = self.left_fermi_level - self.right_fermi_level
        self.right_mid_gap_energy = self.left_mid_gap_energy - potential
        self.temperature = quatrex_config.electron.temperature

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

        # Prepare Buffers for OBC.
        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = quatrex_config.electron.obc.block_sections

        self.call_count = 0
        self.filtering_iteration_limit = (
            quatrex_config.electron.filtering_iteration_limit
        )

        self._sigma_stream = create_stream()
        self._system_stream = create_stream()

        # Batching.
        self.max_batch_size = quatrex_config.electron.max_batch_size

    @staticmethod
    def load_hamiltonian(
        quatrex_config: QuatrexConfig,
    ) -> tuple[sparse.coo_matrix, NDArray]:

        # Load the device Hamiltonian.
        synchronize_device()
        comm.barrier()
        t_ham_load_start = time.perf_counter()
        if quatrex_config.device.construct_from_unit_cell:
            hamiltonian_unit_cells = distributed_load(
                quatrex_config.input_dir / "hamiltonian_unit_cells.npy"
            ).astype(xp.complex128)

            # Determine the local slice of the data.
            # NOTE: This is arrow-wise partitioning.
            # TODO: Allow more options, e.g., block row-wise partitioning.
            section_sizes, __ = get_section_sizes(
                quatrex_config.device.number_of_supercells, comm.block.size
            )
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            start_block = section_offsets[comm.block.rank]
            end_block = section_offsets[comm.block.rank + 1]

            hamiltonian_sparray, block_sizes = create_hamiltonian(
                cutoff_hr(
                    hamiltonian_unit_cells,
                    R_cutoff=quatrex_config.device.unit_cell_per_supercell,
                ),
                quatrex_config.device.number_of_supercells,
                quatrex_config.device.transport_direction,
                quatrex_config.device.unit_cell_per_supercell,
                block_start=start_block,
                block_end=end_block,
                return_sparse=True,
            )
            hamiltonian_sparray = hamiltonian_sparray.astype(xp.complex128)
            hamiltonian_sparray.sum_duplicates()
            block_sizes = get_host(block_sizes)
            block_sizes = np.asarray(
                [block_sizes[0]] * quatrex_config.device.number_of_supercells
            )

        else:
            hamiltonian_sparray = distributed_load(
                quatrex_config.input_dir / "hamiltonian.npz"
            ).astype(xp.complex128)
            block_sizes = get_host(
                distributed_load(quatrex_config.input_dir / "block_sizes.npy")
            )

        synchronize_device()
        t_ham_load_end = time.perf_counter()
        comm.barrier()
        t_ham_load_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Load Hamiltonian: {t_ham_load_end-t_ham_load_start}",
                flush=True,
            )
            print(
                f"    Load Hamiltonian all: {t_ham_load_end_all-t_ham_load_start}",
                flush=True,
            )

        return hamiltonian_sparray, block_sizes

    @staticmethod
    def get_block(
        coo: sparse.coo_matrix, block_sizes: NDArray, index: tuple
    ) -> NDArray:
        """Gets a block from a COO matrix."""
        block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
        row, col = index
        row = row + len(block_sizes) if row < 0 else row
        col = col + len(block_sizes) if col < 0 else col
        mask = (
            (block_offsets[row] <= coo.row)
            & (coo.row < block_offsets[row + 1])
            & (block_offsets[col] <= coo.col)
            & (coo.col < block_offsets[col + 1])
        )
        block = xp.zeros(
            (int(block_sizes[row]), int(block_sizes[col])), dtype=coo.dtype
        )
        block[
            coo.row[mask] - block_offsets[row],
            coo.col[mask] - block_offsets[col],
        ] = coo.data[mask]

        return block

    def update_potential(self, new_potential: NDArray) -> None:
        """Updates the potential matrix.

        Parameters
        ----------
        new_potential : NDArray
            The new potential matrix.

        """
        self.potential = new_potential

    def _update_fermi_levels(
        self, left_band_edges: NDArray, right_band_edges: NDArray
    ) -> None:
        """Updates the Fermi levels.

        Parameters
        ----------
        out : tuple[DSDBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """
        self.left_mid_gap_energy = xp.mean(left_band_edges)
        self.right_mid_gap_energy = xp.mean(right_band_edges)

        __, left_conduction_band_edge = left_band_edges
        __, right_conduction_band_edge = right_band_edges

        (
            print(
                f"Updating conduction band edges: "
                f"{left_conduction_band_edge}, {right_conduction_band_edge}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )

        self.left_fermi_level = (
            left_conduction_band_edge - self.delta_fermi_level_conduction_band
        )
        self.right_fermi_level = (
            right_conduction_band_edge - self.delta_fermi_level_conduction_band
        )

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level,
            self.temperature,
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level,
            self.temperature,
        )

    def _get_block(self, coo: sparse.coo_matrix, index: tuple) -> NDArray:
        """Gets a block from a COO matrix."""
        row, col = index
        row = row + len(self.block_sizes) if row < 0 else row
        col = col + len(self.block_sizes) if col < 0 else col
        mask = (
            (self.block_offsets[row] <= coo.row)
            & (coo.row < self.block_offsets[row + 1])
            & (self.block_offsets[col] <= coo.col)
            & (coo.col < self.block_offsets[col + 1])
        )
        block = xp.zeros(
            (int(self.block_sizes[row]), int(self.block_sizes[col])), dtype=coo.dtype
        )
        block[
            coo.row[mask] - self.block_offsets[row],
            coo.col[mask] - self.block_offsets[col],
        ] = coo.data[mask]

        return block

    def _compute_obc(self, stack_slice: slice) -> None:
        """Computes open boundary conditions."""
        if comm.block.rank == 0:
            # Extract the overlap matrix blocks.
            s_00 = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (0, 0))
            s_01 = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (0, 1))
            s_10 = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (1, 0))

            m_10, m_00, m_01 = get_periodic_superblocks(
                a_ii=self.system_matrix.blocks[0, 0],
                a_ji=self.system_matrix.blocks[1, 0],
                a_ij=self.system_matrix.blocks[0, 1],
                block_sections=self.block_sections,
            )

            g_00 = self.obc(
                a_ii=m_00 + s_00,
                a_ij=m_01 + s_01,
                a_ji=m_10 + s_10,
                contact="left",
                stack_slice=stack_slice,
            )
            # Apply the retarded boundary self-energy.
            sigma_00 = m_10 @ g_00 @ m_01
            self.obc_blocks.retarded[0] = sigma_00
            gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))

            # Compute and apply the lesser boundary self-energy.
            self.obc_blocks.lesser[0] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies[stack_slice]
            )
            # Compute and apply the greater boundary self-energy.
            self.obc_blocks.greater[0] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies[stack_slice] - 1
            )
        if comm.block.rank == comm.block.size - 1:
            # Extract the overlap matrix blocks.
            s_nn = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (-1, -1))
            s_nm = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (-1, -2))
            s_mn = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (-2, -1))

            n = self.system_matrix.num_local_blocks - 1
            m = n - 1

            m_mn, m_nn, m_nm = get_periodic_superblocks(
                # Twist it, flip it, ...
                a_ii=xp.flip(self.system_matrix.blocks[n, n], axis=(-2, -1)),
                a_ji=xp.flip(self.system_matrix.blocks[m, n], axis=(-2, -1)),
                a_ij=xp.flip(self.system_matrix.blocks[n, m], axis=(-2, -1)),
                block_sections=self.block_sections,
            )
            # ... bop it.
            m_nn = xp.flip(m_nn, axis=(-2, -1))
            m_nm = xp.flip(m_nm, axis=(-2, -1))
            m_mn = xp.flip(m_mn, axis=(-2, -1))
            g_nn = self.obc(
                # Twist it, flip it, ...
                a_ii=xp.flip(m_nn + s_nn, axis=(-2, -1)),
                a_ij=xp.flip(m_nm + s_nm, axis=(-2, -1)),
                a_ji=xp.flip(m_mn + s_mn, axis=(-2, -1)),
                contact="right",
                stack_slice=stack_slice,
            )
            # ... bop it.
            g_nn = xp.flip(g_nn, axis=(-2, -1))

            # NOTE: Here we could possibly do peak/discontinuity detection
            # on the surface Green's function DOS (not same as actual DOS).

            # Apply the retarded boundary self-energy.
            sigma_nn = m_mn @ g_nn @ m_nm

            self.obc_blocks.retarded[-1] = sigma_nn

            gamma_nn = 1j * (sigma_nn - sigma_nn.conj().swapaxes(-2, -1))

            self.obc_blocks.lesser[-1] = 1j * scale_stack(
                gamma_nn.copy(), self.right_occupancies[stack_slice]
            )

            self.obc_blocks.greater[-1] = 1j * scale_stack(
                gamma_nn.copy(), self.right_occupancies[stack_slice] - 1
            )

    def _assemble_system_matrix(
        self, sse_retarded: DSDBSparse, stack_slice: slice
    ) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_retarded : DSDBSparse
            The retarded scattering self-energy.

        """
        # self.hamiltonian.to_device(
        #     delete_host=False, stream=self._system_stream, sync=False
        # )
        self.system_matrix.data = 0.0
        self.system_matrix += self.overlap_sparray
        scale_stack(
            self.system_matrix.data,
            self.local_energies[stack_slice] + 1j * self.eta,
        )

        self.system_matrix -= sparse.diags(self.potential, format="csr")
        _btd_subtract(self.system_matrix, sse_retarded)
        # _btd_subtract(self.system_matrix, sse_retarded.stack[stack_slice])
        # synchronize_stream(self._system_stream)
        _btd_subtract(self.system_matrix, self.hamiltonian)

    def _filter_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Filters out peaks in the Green's functions.

        Parameters
        ----------
        out : tuple[DSDBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """
        g_lesser, g_greater, g_retarded = out
        # local_dos = [
        #     (-xp.diagonal(block, axis1=-2, axis2=-1).imag).mean(-1)
        #     for block in g_retarded.block_diagonal()
        # ]

        g_retarded_diag = g_retarded.diagonal()
        block_sizes = g_retarded.block_sizes
        block_offsets = g_retarded.block_offsets
        local_dos = []
        for i, (bsz, boff) in enumerate(zip(block_sizes, block_offsets)):
            g_retarded_density = -g_retarded_diag[..., boff : boff + bsz].imag.mean(-1)
            local_dos.append(g_retarded_density)

        local_dos = xp.array(local_dos)
        dos = comm.stack.all_gather_v(
            local_dos, axis=1, mask=g_lesser._stack_padding_mask
        )

        dos_gradient = xp.abs(xp.gradient(dos, self.energies, axis=1))
        mask = (xp.max(dos_gradient, axis=0) > self.dos_peak_limit) | (
            xp.max(dos, axis=0) > 10
        )

        section_sizes, __ = get_section_sizes(self.energies.size, comm.stack.size)
        section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
        local_mask = mask[
            section_offsets[comm.stack.rank] : section_offsets[comm.stack.rank + 1]
        ]

        g_lesser.data[local_mask] = 0.0
        g_greater.data[local_mask] = 0.0
        g_retarded.data[local_mask] = 0.0

    @profiler.profile(level="basic")
    def solve(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        sse_retarded: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ):
        """Solves for the electron Green's function.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser self-energy.
        sse_greater : DSDBSparse
            The greater self-energy.
        sse_retarded : DSDBSparse
            The retarded self-energy.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """

        if self.flatband:
            time_homogenize_start = time.perf_counter()
            homogenize(sse_greater)
            homogenize(sse_lesser)
            homogenize(sse_retarded)
            synchronize_device()
            time_homogenize_end = time.perf_counter()
            comm.barrier()
            time_homogenize_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"    Homogenize: {time_homogenize_end-time_homogenize_start}",
                    flush=True,
                )
                print(
                    f"    Homogenize all: {time_homogenize_end_all-time_homogenize_start}",
                    flush=True,
                )

        batch_sizes, batch_offsets = get_batches(
            sse_retarded.shape[0], self.max_batch_size
        )

        if comm.rank == 0:
            print(f"Max batch size: {self.max_batch_size}", flush=True)
            print(f"Total size: {sse_retarded.shape[0]}", flush=True)
            print(f"Batch sizes: {batch_sizes}", flush=True)
            print(f"Batch offsets: {batch_offsets}", flush=True)

        for i in range(len(batch_sizes)):

            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))
            sse_lesser_tmp = sse_lesser.stack[stack_slice]
            sse_greater_tmp = sse_greater.stack[stack_slice]
            sse_retarded_tmp = sse_retarded.stack[stack_slice]

            reallocate = False
            if i > 0 and batch_sizes[i] != batch_sizes[i - 1]:
                reallocate = True

            if comm.rank == 0:
                print(
                    f"Processing slice {stack_slice} of {sse_retarded.shape[0]}, batch size {batch_sizes[i]}",
                    flush=True,
                )

            t_assemble_start = time.perf_counter()
            self.hamiltonian.set_to_host()
            if reallocate:
                self.system_matrix.free_data()
            self.system_matrix.allocate_data(stack_size=batch_sizes[i])
            self._assemble_system_matrix(sse_retarded_tmp, stack_slice)
            synchronize_stream(None)
            t_assemble_end = time.perf_counter()
            comm.barrier()
            t_assemble_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    Assemble: {t_assemble_end-t_assemble_start}", flush=True)
                print(
                    f"    Assemble all: {t_assemble_end_all-t_assemble_start}",
                    flush=True,
                )

            if i == 0 and self.band_edge_tracking == "eigenvalues":
                t_band_edges_start = time.perf_counter()
                left_band_edges, right_band_edges = find_renormalized_eigenvalues(
                    hamiltonian=self.hamiltonian,
                    overlap=self.overlap_sparray,
                    potential=self.potential,
                    sigma_retarded=sse_retarded,
                    energies=self.energies,
                    conduction_band_guesses=(
                        self.left_fermi_level + self.delta_fermi_level_conduction_band,
                        self.right_fermi_level + self.delta_fermi_level_conduction_band,
                    ),
                    mid_gap_energies=(
                        self.left_mid_gap_energy,
                        self.right_mid_gap_energy,
                    ),
                    band_edge_config=self.compute_config.band_edge,
                )
                self._update_fermi_levels(left_band_edges, right_band_edges)

                # synchronize_device()
                synchronize_stream(None)
                t_band_edges_end = time.perf_counter()
                comm.barrier()
                t_band_edges_end_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"    Band edges: {t_band_edges_end-t_band_edges_start}",
                        flush=True,
                    )
                    print(
                        f"    Band edges all: {t_band_edges_end_all-t_band_edges_start}",
                        flush=True,
                    )

            if i == 0:
                sse_lesser.to_host(
                    delete_device=False, stream=self._sigma_stream, sync=False
                )
                sse_greater.to_host(
                    delete_device=False, stream=self._sigma_stream, sync=False
                )
                sse_retarded.to_host(
                    delete_device=False, stream=self._sigma_stream, sync=False
                )

            t_obc_start = time.perf_counter()
            self.hamiltonian.set_to_device()
            self._compute_obc(stack_slice)
            # synchronize_device()
            synchronize_stream(None)
            t_obc_end = time.perf_counter()
            comm.barrier()
            t_obc_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    OBC: {t_obc_end-t_obc_start}", flush=True)
                print(f"    OBC all: {t_obc_end_all-t_obc_start}", flush=True)

            out_l, out_g, out_r = out
            out_slice = (
                out_l.stack[stack_slice],
                out_g.stack[stack_slice],
                out_r.stack[stack_slice],
            )

            t_solve_start = time.perf_counter()
            if comm.block.size > 1:
                self.solver_dist.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=sse_lesser_tmp,
                    sigma_greater=sse_greater_tmp,
                    obc_blocks=self.obc_blocks,
                    out=out_slice,
                    return_retarded=True,
                )
            else:
                self.meir_wingreen_current = self.solver.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=sse_lesser_tmp,
                    sigma_greater=sse_greater_tmp,
                    obc_blocks=self.obc_blocks,
                    out=out_slice,
                    return_retarded=True,
                    return_current=self.compute_meir_wingreen_current,
                )
                synchronize_stream(None)
                t_solve_end = time.perf_counter()
                comm.barrier()
                t_solve_end_all = time.perf_counter()
                if comm.rank == 0:
                    print(f"    Solve: {t_solve_end-t_solve_start}", flush=True)
                    print(f"    Solve all: {t_solve_end_all-t_solve_start}", flush=True)

        t_filter_peaks_start = time.perf_counter()
        if self.call_count < self.filtering_iteration_limit:
            self._filter_peaks(out)
        # synchronize_device()
        synchronize_stream(None)
        t_filter_peaks_end = time.perf_counter()
        comm.barrier()
        t_filter_peaks_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Filter peaks: {t_filter_peaks_end-t_filter_peaks_start}",
                flush=True,
            )
            print(
                f"    Filter peaks all: {t_filter_peaks_end_all-t_filter_peaks_start}",
                flush=True,
            )

        if self.band_edge_tracking == "dos-peaks":

            t_dos_peaks_start = time.perf_counter()

            _, _, g_retarded = out
            left_band_edges = np.empty((2,), dtype=float)
            right_band_edges = np.empty((2,), dtype=float)

            if comm.block.rank == 0:
                s_00 = self._get_block(self.overlap_sparray, (0, 0))
                g_00 = g_retarded.blocks[0, 0]

                local_left_dos = -xp.mean(
                    xp.diagonal(g_00 @ s_00, axis1=-2, axis2=-1).imag, axis=-1
                )

                left_dos = comm.stack.all_gather_v(
                    local_left_dos,
                    axis=0,
                    mask=g_retarded._stack_padding_mask,
                )

                e_0_left = find_dos_peaks(left_dos, self.energies)
                left_band_edges = np.array(
                    find_band_edges(e_0_left, self.left_mid_gap_energy)
                )

            if comm.block.rank == comm.block.size - 1:
                s_nn = self._get_block(self.overlap_sparray, (-1, -1))
                n = g_retarded.num_local_blocks - 1
                g_nn = g_retarded.blocks[n, n]
                local_right_dos = -xp.mean(
                    xp.diagonal(g_nn @ s_nn, axis1=-2, axis2=-1).imag, axis=-1
                )

                right_dos = comm.stack.all_gather_v(
                    local_right_dos,
                    axis=0,
                    mask=g_retarded._stack_padding_mask,
                )

                e_0_right = find_dos_peaks(right_dos, self.energies)
                right_band_edges = np.array(
                    find_band_edges(e_0_right, self.right_mid_gap_energy)
                )

            comm.block.bcast(left_band_edges, root=0, backend="device_mpi")
            comm.block.bcast(
                right_band_edges, root=comm.block.size - 1, backend="device_mpi"
            )

            self._update_fermi_levels(left_band_edges, right_band_edges)
            synchronize_device()
            t_dos_peaks_end = time.perf_counter()
            comm.barrier()
            t_dos_peaks_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    DOS peaks: {t_dos_peaks_end-t_dos_peaks_start}", flush=True)
                print(
                    f"    DOS peaks all: {t_dos_peaks_end_all-t_dos_peaks_start}",
                    flush=True,
                )

        self.call_count += 1
