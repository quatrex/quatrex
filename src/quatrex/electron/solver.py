# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.kernels.mixed_precision import compress, decompress
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import create_stream
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from qttools.utils.solvers_utils import get_batches
from qttools.utils.stack_utils import scale_stack
from quatrex.bandstructure.band_edges import (
    find_band_edges,
    find_dos_peaks,
    find_renormalized_eigenvalues,
)
from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import get_periodic_superblocks, homogenize
from quatrex.device.inputs import assemble_matrix

profiler = Profiler()


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
        a_.blocks[i, i] -= b_.blocks[i, i]

        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        a_.blocks[i, j] -= b_.blocks[i, j]
        a_.blocks[j, i] -= b_.blocks[j, i]


class ElectronSolver(SubsystemSolver):
    """Solves the electron dynamics.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    energies : np.ndarray
        The energies at which to solve.

    """

    system = "electron"

    def __init__(
        self,
        config: QuatrexConfig,
        energies: NDArray,
        rows,
        cols,
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(config, energies)

        self.local_energies = get_local_slice(energies, comm.stack)

        # Load the device Hamiltonian.
        self.hamiltonian, hamiltonian_sparsity_pattern = assemble_matrix(
            config=config,
            matrix_name="hamiltonian",
            sparsity_pattern=None,
            shift_kpoints=False,
        )
        self.hamiltonian.to_host()

        # Make sure that the the system matrix sparsity is a superset of
        # self-energy and Hamiltonian sparsity.

        sparsity_pattern = sparse.coo_matrix(
            (xp.ones_like(rows, dtype=xp.float32), (rows, cols))
        )

        sparsity_pattern += hamiltonian_sparsity_pattern
        sparsity_pattern = sparsity_pattern.tocoo()

        del hamiltonian_sparsity_pattern
        self.block_sizes = self.hamiltonian.block_sizes

        self.orthogonal_basis = config.device.orthogonal_basis
        if not self.orthogonal_basis:
            # TODO: Overlap matrix is not supported correctly. The code
            # should look like this.

            # Load the device Overlap.
            self.overlap, overlap_sparsity_pattern = assemble_matrix(
                config=config,
                matrix_name="overlap",
                sparsity_pattern=(sparsity_pattern.row, sparsity_pattern.col),
                shift_kpoints=False,
            )
            self.overlap.to_host()

            # Make sure that the the system matrix sparsity is a superset of
            # self-energy and overlap sparsity.
            # TODO: This is not correct
            # sparsity_pattern += overlap_sparsity_pattern

            # Check that the overlap matrix and Hamiltonian matrix match.
            if self.overlap.shape != self.hamiltonian.shape:
                raise ValueError(
                    "Overlap matrix and Hamiltonian matrix have different shapes."
                )

            raise NotImplementedError("Currently, overlap matrices are not supported.")

        else:
            self.overlap_sparray = sparse.eye(
                self.hamiltonian.shape[-2],
                format="coo",
                dtype=self.hamiltonian.dtype,
            )

        # Allocate memory for the system matrix.
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            sparsity_pattern.row,
            sparsity_pattern.col,
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([int(k) for k in config.device.kpoint_grid if k > 1]),
            bits=config.compute.num_bits,
        )
        self.system_matrix.free_data()  # Free any previously allocated data
        del sparsity_pattern

        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.hamiltonian.shape[-2]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.hamiltonian.shape[-2]}"
            )

        # Load the potential.
        try:
            self.potential = distributed_load(config.input_dir / "potential.npy")
        except FileNotFoundError:
            # No potential provided. Assume zero potential.
            self.potential = xp.zeros(
                self.hamiltonian.shape[-2], dtype=self.hamiltonian.dtype
            )
        if self.potential.size != self.hamiltonian.shape[-2]:
            raise ValueError("Potential matrix and Hamiltonian have different shapes.")
        self.eta = config.electron.eta
        self.eta_obc = config.electron.eta_obc

        # Contacts.
        self.flatband = config.electron.flatband
        if self.flatband and comm.rank == 0:
            print("Flatband conditions detected", flush=True)

        if config.electron.solver.compute_current and comm.block.size > 1:
            raise NotImplementedError(
                "Current computation not implemented in distributed mode."
            )

        self.compute_meir_wingreen_current = config.electron.solver.compute_current

        self.dos_peak_limit = config.electron.dos_peak_limit

        # Band edges and Fermi levels.
        # TODO: This only works for small potential variations accross
        # the device.
        # TODO: During this initialization we should compute the contact
        # band structures and extract the correct fermi levels & band
        # edges from there.
        self.band_edge_tracking = config.electron.band_edge_tracking
        self.delta_fermi_level_conduction_band = (
            config.electron.conduction_band_edge - config.electron.fermi_level
        )
        self.left_mid_gap_energy = 0.5 * (
            config.electron.conduction_band_edge + config.electron.valence_band_edge
        )
        self.left_fermi_level = config.electron.left_fermi_level
        self.right_fermi_level = config.electron.right_fermi_level

        potential = self.left_fermi_level - self.right_fermi_level
        self.right_mid_gap_energy = self.left_mid_gap_energy - potential
        self.temperature = config.electron.temperature

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

        # Prepare Buffers for OBC.
        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = config.electron.obc.block_sections

        self.call_count = 0
        self.filtering_iteration_limit = config.electron.filtering_iteration_limit

        self._sigma_stream = create_stream()
        self._system_stream = create_stream()
        self.max_batch_size = config.electron.max_batch_size

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

    @profiler.profile(label="ElectronSolver: OBC", level="default", comm=comm)
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

            # TODO: use residuals to filter "bad" energies
            g_00, *__ = self.obc(
                (m_00 + s_00, m_01 + s_01, m_10 + s_10),
                contact="left-" + str(stack_slice),
            )

            # Apply the retarded boundary self-energy.
            sigma_00 = m_10 @ g_00 @ m_01

            if self.obc_blocks.retarded[0] is None:
                self.obc_blocks.retarded[0] = xp.empty(
                    (
                        self.local_energies.size,
                        sigma_00.shape[-2],
                        sigma_00.shape[-1],
                    ),
                    dtype=sigma_00.dtype,
                )
                self.obc_blocks.lesser[0] = xp.empty_like(self.obc_blocks.retarded[0])
                self.obc_blocks.greater[0] = xp.empty_like(self.obc_blocks.retarded[0])

            self.obc_blocks.retarded[0][stack_slice] = sigma_00
            gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))

            # Compute and apply the lesser boundary self-energy.
            self.obc_blocks.lesser[0][stack_slice] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies[stack_slice]
            )
            # Compute and apply the greater boundary self-energy.
            self.obc_blocks.greater[0][stack_slice] = 1j * scale_stack(
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
            g_nn, *__ = self.obc(
                # Twist it, flip it, ...
                (
                    xp.flip(m_nn + s_nn, axis=(-2, -1)),
                    xp.flip(m_nm + s_nm, axis=(-2, -1)),
                    xp.flip(m_mn + s_mn, axis=(-2, -1)),
                ),
                contact="right-" + str(stack_slice),
            )
            # ... bop it.
            g_nn = xp.flip(g_nn, axis=(-2, -1))

            # NOTE: Here we could possibly do peak/discontinuity detection
            # on the surface Green's function DOS (not same as actual DOS).

            # Apply the retarded boundary self-energy.
            sigma_nn = m_mn @ g_nn @ m_nm

            if self.obc_blocks.retarded[-1] is None:
                self.obc_blocks.retarded[-1] = xp.empty(
                    (
                        self.local_energies.size,
                        sigma_nn.shape[-2],
                        sigma_nn.shape[-1],
                    ),
                    dtype=sigma_nn.dtype,
                )
                self.obc_blocks.lesser[-1] = xp.empty_like(self.obc_blocks.retarded[-1])
                self.obc_blocks.greater[-1] = xp.empty_like(
                    self.obc_blocks.retarded[-1]
                )

            self.obc_blocks.retarded[-1][stack_slice] = sigma_nn

            gamma_nn = 1j * (sigma_nn - sigma_nn.conj().swapaxes(-2, -1))

            self.obc_blocks.lesser[-1][stack_slice] = 1j * scale_stack(
                gamma_nn.copy(), self.right_occupancies[stack_slice]
            )

            self.obc_blocks.greater[-1][stack_slice] = 1j * scale_stack(
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

        self.system_matrix.data = 0

        if self.config.compute.num_bits is not None:
            _data = decompress(self.system_matrix.data, self.system_matrix.bits)
        else:
            _data = self.system_matrix.data

        if not self.orthogonal_basis:
            raise NotImplementedError("Non-orthogonal basis not implemented.")
            # TODO: This is not correct in the case of kpoints
            self.system_matrix += self.overlap_sparray

        tmp = (
            self.local_energies[stack_slice][:, None]
            + 1j * self.eta
            - self.potential[None, :]
        )
        self.system_matrix.fill_diagonal(tmp, data=_data)

        if self.config.compute.num_bits is not None:
            self.system_matrix.data = compress(_data, self.system_matrix.bits)

        _btd_subtract(self.system_matrix, sse_retarded)
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

    @profiler.profile(label="ElectronSolver", level="default", comm=comm)
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
            with profiler.profile_range(
                label="ElectronSolver: Homogenize", level="default", comm=comm
            ):
                homogenize(sse_greater)
                homogenize(sse_lesser)
                homogenize(sse_retarded)

        if self.max_batch_size is None:
            max_batch_size = sse_retarded.shape[0]
        else:
            max_batch_size = self.max_batch_size

        batch_sizes, batch_offsets = get_batches(sse_retarded.shape[0], max_batch_size)

        self.meir_wingreen_current = []

        self.hamiltonian.set_to_host()

        for i in range(len(batch_sizes)):

            stack_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))
            sse_lesser_tmp = sse_lesser.stack[stack_slice]
            sse_greater_tmp = sse_greater.stack[stack_slice]
            sse_retarded_tmp = sse_retarded.stack[stack_slice]

            with profiler.profile_range(
                label="ElectronSolver: Assemble", level="default", comm=comm
            ):
                reallocate = False
                if i > 0 and batch_sizes[i] != batch_sizes[i - 1]:
                    reallocate = True

                if reallocate:
                    self.system_matrix.free_data()
                self.system_matrix.allocate_data(stack_size=batch_sizes[i])

                self._assemble_system_matrix(sse_retarded_tmp, stack_slice)

            if i == 0 and self.band_edge_tracking == "eigenvalues":
                with profiler.profile_range(
                    label="ElectronSolver: Band edges", level="default", comm=comm
                ):
                    left_band_edges, right_band_edges = find_renormalized_eigenvalues(
                        hamiltonian=self.hamiltonian,
                        overlap=self.overlap_sparray,
                        potential=self.potential,
                        sigma_retarded=sse_retarded,
                        energies=self.energies,
                        conduction_band_guesses=(
                            self.left_fermi_level
                            + self.delta_fermi_level_conduction_band,
                            self.right_fermi_level
                            + self.delta_fermi_level_conduction_band,
                        ),
                        mid_gap_energies=(
                            self.left_mid_gap_energy,
                            self.right_mid_gap_energy,
                        ),
                        band_edge_config=self.config.compute.band_edge,
                    )
                    self._update_fermi_levels(left_band_edges, right_band_edges)

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

            self._compute_obc(stack_slice)

            out_l, out_g, out_r = out
            out_slice = (
                out_l.stack[stack_slice],
                out_g.stack[stack_slice],
                out_r.stack[stack_slice],
            )
            obc_blocks_tmp = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
            for j in range(self.system_matrix.num_local_blocks):
                if self.obc_blocks.retarded[j] is not None:
                    obc_blocks_tmp.retarded[j] = self.obc_blocks.retarded[j][
                        stack_slice
                    ]
                    obc_blocks_tmp.lesser[j] = self.obc_blocks.lesser[j][stack_slice]
                    obc_blocks_tmp.greater[j] = self.obc_blocks.greater[j][stack_slice]

            if self.system_matrix.bits is not None:
                _tmp = decompress(self.system_matrix.data, self.system_matrix.bits)
                if not xp.all(xp.isfinite(_tmp)):
                    print(
                        f"Warning: Non-finite values {xp.any(xp.isnan(_tmp))} {xp.any(xp.isinf(_tmp))}  detected in system G. {comm.rank} {self.call_count}",
                        flush=True,
                    )
                _tmp = decompress(sse_lesser_tmp.data, self.system_matrix.bits)
                if not xp.all(xp.isfinite(_tmp)):
                    print(
                        f"Warning: Non-finite values {xp.any(xp.isnan(_tmp))} {xp.any(xp.isinf(_tmp))}  detected in sse lesser. {comm.rank} {self.call_count}",
                        flush=True,
                    )
                _tmp = decompress(sse_greater_tmp.data, self.system_matrix.bits)
                if not xp.all(xp.isfinite(_tmp)):
                    print(
                        f"Warning: Non-finite values {xp.any(xp.isnan(_tmp))} {xp.any(xp.isinf(_tmp))}  detected in sse greater. {comm.rank} {self.call_count}",
                        flush=True,
                    )

            with profiler.profile_range(
                label="ElectronSolver: Solve", level="default", comm=comm
            ):
                if comm.block.size > 1:
                    self.solver_dist.selected_solve(
                        a=self.system_matrix,
                        sigma_lesser=sse_lesser_tmp,
                        sigma_greater=sse_greater_tmp,
                        obc_blocks=self.obc_blocks_tmp,
                        out=out_slice,
                        return_retarded=True,
                    )

                else:
                    current = self.solver.selected_solve(
                        a=self.system_matrix,
                        sigma_lesser=sse_lesser_tmp,
                        sigma_greater=sse_greater_tmp,
                        obc_blocks=obc_blocks_tmp,
                        out=out_slice,
                        return_retarded=True,
                        return_current=self.compute_meir_wingreen_current,
                        ozaki=self.config.compute.g_rgf_ozaki,
                        slices=self.config.compute.g_rgf_slices,
                    )
                    self.meir_wingreen_current.append(current)

            if self.system_matrix.bits is not None:
                g_lesser_, g_greater_, g_retarded_ = out_slice

                _tmp = decompress(g_lesser_.data, self.system_matrix.bits)
                if not xp.all(xp.isfinite(_tmp)):
                    print(
                        f"Warning: Non-finite values {xp.any(xp.isnan(_tmp))} {xp.any(xp.isinf(_tmp))} detected in lesser Green's function. {comm.rank} {self.call_count}",
                        flush=True,
                    )
                    g_lesser_.data = 0

                _tmp = decompress(g_greater_.data, self.system_matrix.bits)
                if not xp.all(xp.isfinite(_tmp)):
                    print(
                        f"Warning: Non-finite values {xp.any(xp.isnan(_tmp))} {xp.any(xp.isinf(_tmp))}  detected in greater Green's function. {comm.rank} {self.call_count}",
                        flush=True,
                    )
                    g_greater_.data = 0
                _tmp = decompress(g_retarded_.data, self.system_matrix.bits)
                if not xp.all(xp.isfinite(_tmp)):
                    print(
                        f"Warning: Non-finite values {xp.any(xp.isnan(_tmp))} {xp.any(xp.isinf(_tmp))}  detected in retarded Green's function. {comm.rank} {self.call_count}",
                        flush=True,
                    )
                    g_retarded_.data = 0

        with profiler.profile_range(
            label="ElectronSolver: Filter", level="default", comm=comm
        ):
            self.system_matrix.free_data()
            # if self.call_count < self.filtering_iteration_limit:
            #     self._filter_peaks(out)

        if self.band_edge_tracking == "dos-peaks":

            with profiler.profile_range(
                label="ElectronSolver: DOS peaks", level="default", comm=comm
            ):
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

        if self.compute_meir_wingreen_current:

            self.meir_wingreen_current = xp.concatenate(
                self.meir_wingreen_current, axis=0
            )
            self.meir_wingreen_current = self.meir_wingreen_current.reshape(
                (-1, *self.meir_wingreen_current.shape[1:])
            )

        self.call_count += 1
