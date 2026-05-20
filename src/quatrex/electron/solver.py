# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import synchronize_device
from qttools.utils.mpi_utils import distributed_load, get_local_slice
from qttools.utils.stack_utils import scale_stack
from quatrex.bandstructure.band_edges import (
    find_band_edges,
    find_dos_peaks,
    find_renormalized_eigenvalues,
)
from quatrex.bandstructure.contact import (
    contact_fermi_level,
    find_charge_neutral_fermi_level,
)
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import density
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import (
    filtering_peaks_mask,
    get_periodic_superblocks,
    homogenize,
)
from quatrex.device.inputs import load_matrix

from ._potential_generation import (
    compute_charge_for_fermi_levels,
    find_energy_shift,
    generate_potential_profile,
)

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
        a_.blocks[i, i] -= b_.blocks[i, i]

        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        a_.blocks[i, j] -= b_.blocks[i, j]
        a_.blocks[j, i] -= b_.blocks[j, i]


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
        sparsity_pattern: sparse.coo_matrix,
        grid: NDArray,
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(quatrex_config, compute_config, energies)

        self.local_energies = get_local_slice(energies, comm.stack)

        self.lattice_vectors = distributed_load(
            quatrex_config.input_dir / "lattice_vectors.npy"
        )

        # Load the device Hamiltonian.
        self.hamiltonian, hamiltonian_sparsity_pattern = load_matrix(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            matrix_name="hamiltonian",
            sparsity_pattern=None,
            shift_kpoints=False,
        )

        # Make sure that the the system matrix sparsity is a superset of
        # self-energy and Hamiltonian sparsity.
        sparsity_pattern += hamiltonian_sparsity_pattern

        del hamiltonian_sparsity_pattern
        self.block_sizes = self.hamiltonian.block_sizes

        self.orthogonal_basis = quatrex_config.device.orthogonal_basis
        if not self.orthogonal_basis:
            # TODO: Overlap matrix is not supported correctly. The code
            # should look like this.

            # Load the device Overlap.
            self.overlap, overlap_sparsity_pattern = load_matrix(
                quatrex_config=quatrex_config,
                compute_config=compute_config,
                matrix_name="overlap",
                sparsity_pattern=None,
                shift_kpoints=False,
            )

            # Make sure that the the system matrix sparsity is a superset of
            # self-energy and overlap sparsity.
            sparsity_pattern += overlap_sparsity_pattern
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
        self.system_matrix = compute_config.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([int(k) for k in quatrex_config.device.kpoint_grid if k > 1]),
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

        # Eta can be a float or an array, if it is an array the elements are iterated over
        # each call to solve, until the end is reached and the last element is used for all
        # subsequent calls.
        self.eta = quatrex_config.electron.eta
        if isinstance(self.eta, float):
            self.eta = (self.eta,)
        self.eta_obc = quatrex_config.electron.eta_obc

        # Contacts.
        self.flatband = quatrex_config.electron.flatband
        if self.flatband and comm.rank == 0:
            print("Flatband conditions detected", flush=True)

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

        self.bias = self.left_fermi_level - self.right_fermi_level
        self.right_mid_gap_energy = self.left_mid_gap_energy - self.bias
        self.temperature = quatrex_config.electron.temperature

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

        self.grid = grid

        # Load the potential.
        try:
            self.potential = distributed_load(
                quatrex_config.input_dir / "potential.npy"
            )
        except FileNotFoundError:
            transport_direction = "xyz".index(quatrex_config.device.transport_direction)
            self.flat_length = (
                self.lattice_vectors[transport_direction, transport_direction] * 4
            )  # Flat for 1 transport cells at each end
            # self.flat_length = self.lattice_vectors[transport_direction, transport_direction] * 8  # Flat for 2 transport cells at each end
            # self.flat_length = self.lattice_vectors[transport_direction, transport_direction] * 16  # Flat for 4 transport cells at each end
            # self.flat_length = self.lattice_vectors[transport_direction, transport_direction] * 12  # Flat for 3 transport cells at each end
            # self.flat_length = self.lattice_vectors[transport_direction, transport_direction] * 20  # Flat for 5 transport cells at each end
            # self.flat_length = self.lattice_vectors[transport_direction, transport_direction] * 28  # Flat for 7 transport cells at each end
            self.potential = generate_potential_profile(
                self.grid,
                transport_direction=transport_direction,
                bias=self.bias,
                potential_function="linear",
                # potential_function="tanh",
                flat_length=self.flat_length,
            )
            # # No potential provided. Assume zero potential.
            # self.potential = xp.zeros(
            #     self.hamiltonian.shape[-2], dtype=self.hamiltonian.dtype
            # )
        if self.potential.size != self.hamiltonian.shape[-2]:
            raise ValueError("Potential matrix and Hamiltonian have different shapes.")

        # Prepare Buffers for OBC.
        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = quatrex_config.electron.obc.block_sections
        self.block_sections_contact_gf = quatrex_config.device.neighbor_cell_cutoff[
            "xyz".index(quatrex_config.device.transport_direction)
        ]

        self.fermi_level_mixing = quatrex_config.electron.fermi_level_mixing

        self.fermi_levels = []
        self.charge_densities = []
        # Charge per unit volume
        self.left_target_charge = (
            self.quatrex_config.electron.doping
            * xp.linalg.det(
                # So far this only works for 2D systems.
                self.lattice_vectors[:2, :2]
            )
            * 1e-16
        )  # A^2 to cm^2

        # NOTE: This only works for uniform block sizes and when the unit cell is commensurate with the blocks.
        uc_size = self.quatrex_config.device.neighbor_cell_cutoff[
            "xyz".index(self.quatrex_config.device.transport_direction)
        ]
        block_size = self.block_sizes[0]
        assert (
            block_size % uc_size == 0
        ), "Block size must be divisible by unit cell size."
        self.small_block_size = block_size // uc_size
        # Try bigger block sizes
        # self.small_block_size *= uc_size
        # self.left_target_charge *= uc_size

        self.call_count = 0
        self.filtering_iteration_limit = (
            quatrex_config.electron.filtering_iteration_limit
        )

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
        self.right_fermi_level = self.left_fermi_level - self.bias

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

    def _compute_obc(self) -> None:
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
            )
            # Apply the retarded boundary self-energy.
            sigma_00 = m_10 @ g_00 @ m_01
            self.obc_blocks.retarded[0] = sigma_00
            gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))

            # Compute and apply the lesser boundary self-energy.
            self.obc_blocks.lesser[0] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies
            )
            # Compute and apply the greater boundary self-energy.
            self.obc_blocks.greater[0] = 1j * scale_stack(
                gamma_00.copy(), self.left_occupancies - 1
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
                gamma_nn.copy(), self.right_occupancies
            )

            self.obc_blocks.greater[-1] = 1j * scale_stack(
                gamma_nn.copy(), self.right_occupancies - 1
            )

    def _assemble_system_matrix(self, sse_retarded: DSDBSparse) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_retarded : DSDBSparse
            The retarded scattering self-energy.

        """
        self.system_matrix.data = 0.0
        if self.orthogonal_basis:
            self.system_matrix.fill_diagonal(1.0)
        else:
            # TODO: This is not correct in the case of kpoints
            self.system_matrix += self.overlap_sparray

        eta = self.eta[self.call_count if self.call_count < len(self.eta) else -1]
        if comm.rank == 0:
            print(f"Using eta = {eta}", flush=True)
        scale_stack(
            self.system_matrix.data,
            self.local_energies + 1j * eta,
        )
        self.system_matrix -= sparse.diags(self.potential, format="csr")
        _btd_subtract(self.system_matrix, sse_retarded)
        _btd_subtract(self.system_matrix, self.hamiltonian)

    def _block_resolved_dos(self, x: DSDBSparse) -> NDArray:
        """Computes the block-resolved density of states from a Green's function.
        It is in units of states per eV per cm^2, assuming a 2D system."""
        kp_dims = len(x.shape[1:-2])
        ne = len(self.energies)
        block_dos = density(x=x, overlap=self.overlap_sparray)
        # Mean over k-points
        block_dos = block_dos.mean(tuple(range(1, kp_dims + 1)))
        # Make it block resolved
        block_dos = block_dos.reshape((ne, -1, self.small_block_size)).sum(-1)
        # # Correct units
        # block_dos = block_dos / np.linalg.det(self.lattice_vectors[:2, :2]) * 1e16
        return block_dos

    def _potential_update_boundary(self, left_dos, right_dos):
        """Creates a new potential by enforcing charge neutrality at the contacts."""
        charge_neutral_fl = contact_fermi_level(
            temperature=self.temperature,
            dos=left_dos,
            energies=self.energies,
            doping_density=self.left_target_charge,
            midgap_energy=self.left_mid_gap_energy,
        )
        charge_neutral_fr = contact_fermi_level(
            temperature=self.temperature,
            dos=right_dos,
            energies=self.energies,
            doping_density=self.left_target_charge,
            midgap_energy=self.right_mid_gap_energy,
        )
        old_pot_start = self.potential[0]
        old_pot_end = self.potential[-1]
        old_bias = old_pot_start - old_pot_end
        new_bias = (
            old_bias
            - (charge_neutral_fl - charge_neutral_fr)
            + (self.left_fermi_level - self.right_fermi_level)
        )
        self.potential = generate_potential_profile(
            self.grid,
            transport_direction="xyz".index(
                self.quatrex_config.device.transport_direction
            ),
            bias=new_bias,
            # potential_function="linear",
            potential_function="tanh",
            flat_length=self.flat_length,
        )
        self.potential += old_pot_start - self.potential[0]
        self.potential += self.left_fermi_level - charge_neutral_fl
        self.left_mid_gap_energy += self.left_fermi_level - charge_neutral_fl
        self.right_mid_gap_energy += self.right_fermi_level - charge_neutral_fl

    def _potential_update(self, ldos, excess_charge, midgap_energies):
        """Creates a new potential by enforcing charge neutrality across the device."""
        fermi_levels = self.energies.copy()
        # Find charge for each Fermi level.
        charge_per_fermi_level = compute_charge_for_fermi_levels(
            fermi_levels, ldos, midgap_energies, self.energies
        )
        # Find the Fermi level that corresponds to the target charge, and the current excess charge. Then
        # return the difference as the energy shift to apply to the potential.
        energy_shifts = find_energy_shift(
            charge_per_fermi_level, fermi_levels, self.left_target_charge, excess_charge
        )
        # Apply the energy shift as a rigid shift to the potential.
        # Problem: energy shifts are block resolved, but the potential is orbital resolved. For now, we just apply
        # the same shift to all orbitals. This can maybe change in the future.
        bs = (
            self.small_block_size
            if self.small_block_size % 2 == 1
            else self.small_block_size - 1
        )
        energy_shifts = np.repeat(energy_shifts, self.small_block_size)
        self.potential -= energy_shifts
        # self.potential -= xp.convolve(np.pad(energy_shifts, pad_width=bs//2, mode="edge"), np.ones(bs)/bs, mode="valid")
        # self.potential -= xp.convolve(np.pad(energy_shifts, pad_width=bs//2, mode="wrap"), np.ones(bs)/bs, mode="valid")
        # self.potential -= xp.convolve(np.pad(energy_shifts, pad_width=bs//2, mode="reflect"), np.ones(bs)/bs, mode="valid")
        self.potential -= xp.convolve(
            np.pad(
                energy_shifts,
                pad_width=bs // 2,
                mode="constant",
                constant_values=self.left_target_charge / self.small_block_size,
            ),
            np.ones(bs) / bs,
            mode="valid",
        )

    def _update_potential_and_solve(
        self,
        ldos,
        excess_charge,
        midgap_energies,
        sse_retarded: DSDBSparse,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Updates the potential and solves for the Green's function."""

        # Update the potential based on the contact DOS
        if self.band_edge_tracking == "potential-update-boundary":
            self._potential_update_boundary(ldos[..., 0], ldos[..., -1])
        elif self.band_edge_tracking == "potential-update":
            self._potential_update(ldos, excess_charge, midgap_energies)

        t_assemble_start = time.perf_counter()
        self._assemble_system_matrix(sse_retarded)
        synchronize_device()
        t_assemble_end = time.perf_counter()
        comm.barrier()
        t_assemble_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Assemble: {t_assemble_end - t_assemble_start}", flush=True)
            print(
                f"    Assemble all: {t_assemble_end_all - t_assemble_start}", flush=True
            )

        t_obc_start = time.perf_counter()
        self._compute_obc()
        synchronize_device()
        t_obc_end = time.perf_counter()
        comm.barrier()
        t_obc_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    OBC: {t_obc_end - t_obc_start}", flush=True)
            print(f"    OBC all: {t_obc_end_all - t_obc_start}", flush=True)

        if comm.block.size > 1:
            t_solve_start = time.perf_counter()
            self.solver_dist.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    Solve: {t_solve_end - t_solve_start}", flush=True)
                print(f"    Solve all: {t_solve_end_all - t_solve_start}", flush=True)

        else:
            t_solve_start = time.perf_counter()
            self.meir_wingreen_current = self.solver.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
                return_current=self.compute_meir_wingreen_current,
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    Solve: {t_solve_end - t_solve_start}", flush=True)
                print(f"    Solve all: {t_solve_end_all - t_solve_start}", flush=True)

        t_filter_peaks_start = time.perf_counter()
        # Free the system matrix data to save memory
        # self.system_matrix.free_data()
        if self.call_count < self.filtering_iteration_limit:
            g_lesser, g_greater, g_retarded = out
            local_mask = filtering_peaks_mask(
                g_retarded, self.energies, self.dos_peak_limit
            )
            g_lesser.data[local_mask] = 0.0
            g_greater.data[local_mask] = 0.0
            g_retarded.data[local_mask] = 0.0

        synchronize_device()
        t_filter_peaks_end = time.perf_counter()
        comm.barrier()
        t_filter_peaks_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Filter peaks: {t_filter_peaks_end - t_filter_peaks_start}",
                flush=True,
            )
            print(
                f"    Filter peaks all: {t_filter_peaks_end_all - t_filter_peaks_start}",
                flush=True,
            )

    def _update_fermi_and_solve(
        self,
        left_dos,
        right_dos,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Updates the Fermi levels and solves for the Green's function."""
        self.left_fermi_level = contact_fermi_level(
            temperature=self.temperature,
            dos=left_dos,
            energies=self.energies,
            doping_density=self.left_target_charge,
            midgap_energy=self.left_mid_gap_energy,
        )
        # self.right_fermi_level = self.left_fermi_level - self.bias
        self.right_fermi_level = contact_fermi_level(
            temperature=self.temperature,
            dos=right_dos,
            energies=self.energies,
            doping_density=self.left_target_charge,
            midgap_energy=self.right_mid_gap_energy,
        )
        # Update lesser/greater self-energies with new Fermi levels.
        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level,
            self.temperature,
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level,
            self.temperature,
        )
        sigma_00 = self.obc_blocks.retarded[0]
        gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))
        # Compute and apply the lesser boundary self-energy.
        self.obc_blocks.lesser[0] = 1j * scale_stack(
            gamma_00.copy(), self.left_occupancies
        )
        # Compute and apply the greater boundary self-energy.
        self.obc_blocks.greater[0] = 1j * scale_stack(
            gamma_00.copy(), self.left_occupancies - 1
        )
        sigma_nn = self.obc_blocks.retarded[-1]
        gamma_nn = 1j * (sigma_nn - sigma_nn.conj().swapaxes(-2, -1))
        self.obc_blocks.lesser[-1] = 1j * scale_stack(
            gamma_nn.copy(), self.right_occupancies
        )
        self.obc_blocks.greater[-1] = 1j * scale_stack(
            gamma_nn.copy(), self.right_occupancies - 1
        )
        t_solve_start = time.perf_counter()
        self.meir_wingreen_current = self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=sse_lesser,
            sigma_greater=sse_greater,
            obc_blocks=self.obc_blocks,
            out=out,
            return_retarded=True,
            return_current=self.compute_meir_wingreen_current,
        )
        synchronize_device()
        t_solve_end = time.perf_counter()
        comm.barrier()
        t_solve_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Second solve: {t_solve_end - t_solve_start}", flush=True)
            print(
                f"    Second solve all: {t_solve_end_all - t_solve_start}", flush=True
            )
        t_filter_peaks_start = time.perf_counter()
        # Free the system matrix data to save memory
        # self.system_matrix.free_data()
        if self.call_count < self.filtering_iteration_limit:
            g_lesser, g_greater, g_retarded = out
            local_mask = filtering_peaks_mask(
                g_retarded, self.energies, self.dos_peak_limit
            )
            g_lesser.data[local_mask] = 0.0
            g_greater.data[local_mask] = 0.0
            g_retarded.data[local_mask] = 0.0

        synchronize_device()
        t_filter_peaks_end = time.perf_counter()
        comm.barrier()
        t_filter_peaks_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Filter peaks: {t_filter_peaks_end - t_filter_peaks_start}",
                flush=True,
            )
            print(
                f"    Filter peaks all: {t_filter_peaks_end_all - t_filter_peaks_start}",
                flush=True,
            )

    def _find_band_edges(self, ldos):
        """Finds the band edges based on the local density of states."""
        # Loop through the blocks and find the band edges for each block
        band_edges = []
        block_potential = self.potential.reshape((-1, self.small_block_size)).mean(-1)
        block_potential -= block_potential[0]
        for block in range(ldos.shape[1]):
            peaks = find_dos_peaks(ldos[:, block], self.energies)
            edges = find_band_edges(
                peaks, self.left_mid_gap_energy + block_potential[block]
            )
            band_edges.append(edges)

        return band_edges

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
                    f"    Homogenize: {time_homogenize_end - time_homogenize_start}",
                    flush=True,
                )
                print(
                    f"    Homogenize all: {time_homogenize_end_all - time_homogenize_start}",
                    flush=True,
                )

        t_assemble_start = time.perf_counter()
        self.system_matrix.allocate_data()

        self._assemble_system_matrix(sse_retarded)
        synchronize_device()
        t_assemble_end = time.perf_counter()
        comm.barrier()
        t_assemble_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Assemble: {t_assemble_end - t_assemble_start}", flush=True)
            print(
                f"    Assemble all: {t_assemble_end_all - t_assemble_start}", flush=True
            )

        if self.band_edge_tracking == "eigenvalues":
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
                mid_gap_energies=(self.left_mid_gap_energy, self.right_mid_gap_energy),
                band_edge_config=self.compute_config.band_edge,
            )
            self._update_fermi_levels(left_band_edges, right_band_edges)

            synchronize_device()
            t_band_edges_end = time.perf_counter()
            comm.barrier()
            t_band_edges_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"    Band edges: {t_band_edges_end - t_band_edges_start}",
                    flush=True,
                )
                print(
                    f"    Band edges all: {t_band_edges_end_all - t_band_edges_start}",
                    flush=True,
                )

        elif (
            self.band_edge_tracking in ["charge-neutrality", "secant-method"]
            and self.call_count <= 1
        ):
            t_cn_start = time.perf_counter()
            # Charge per unit volume
            left_fermi_level, left_mid_gap_energy = find_charge_neutral_fermi_level(
                hamiltonian=self.hamiltonian,
                overlap=self.overlap_sparray,
                potential=self.potential,
                sigma_retarded=sse_retarded,
                local_energies=self.local_energies,
                energies=self.energies,
                temperature=self.temperature,
                target_charge=self.left_target_charge,
                mid_gap_energy=self.left_mid_gap_energy,
                block_sections=self.block_sections_contact_gf,
                side="left",
            )
            if self.call_count > 0:
                # Mix the Fermi level with the previous one.
                self.left_fermi_level = (
                    self.fermi_level_mixing * left_fermi_level
                    + (1 - self.fermi_level_mixing) * self.left_fermi_level
                )
            else:
                self.left_fermi_level = left_fermi_level
            self.left_mid_gap_energy = left_mid_gap_energy

            self.right_mid_gap_energy = self.left_mid_gap_energy - self.bias
            self.right_fermi_level = self.left_fermi_level - self.bias

            self.left_occupancies = fermi_dirac(
                self.local_energies - self.left_fermi_level,
                self.temperature,
            )
            self.right_occupancies = fermi_dirac(
                self.local_energies - self.right_fermi_level,
                self.temperature,
            )

            synchronize_device()
            t_cn_end = time.perf_counter()
            comm.barrier()
            t_cn_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    CN: {t_cn_end - t_cn_start}", flush=True)
                print(f"    CN all: {t_cn_end_all - t_cn_start}", flush=True)

        elif self.band_edge_tracking == "potential-update" and self.call_count < 1:
            t_cn_start = time.perf_counter()
            # Charge per unit volume
            # Fermi level is updated once and then kept fixed. The potential is updated to get charge neutrality in the contacts.
            left_fermi_level, left_mid_gap_energy = find_charge_neutral_fermi_level(
                hamiltonian=self.hamiltonian,
                overlap=self.overlap_sparray,
                potential=self.potential,
                sigma_retarded=sse_retarded,
                local_energies=self.local_energies,
                energies=self.energies,
                temperature=self.temperature,
                target_charge=self.left_target_charge,
                mid_gap_energy=self.left_mid_gap_energy,
                block_sections=self.block_sections_contact_gf,
                side="left",
            )
            self.left_fermi_level = left_fermi_level
            self.left_mid_gap_energy = left_mid_gap_energy

            self.right_mid_gap_energy = self.left_mid_gap_energy - self.bias
            self.right_fermi_level = self.left_fermi_level - self.bias

            self.left_occupancies = fermi_dirac(
                self.local_energies - self.left_fermi_level,
                self.temperature,
            )
            self.right_occupancies = fermi_dirac(
                self.local_energies - self.right_fermi_level,
                self.temperature,
            )

            synchronize_device()
            t_cn_end = time.perf_counter()
            comm.barrier()
            t_cn_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    CN: {t_cn_end - t_cn_start}", flush=True)
                print(f"    CN all: {t_cn_end_all - t_cn_start}", flush=True)

        # Print the updated Fermi levels.
        (
            print(
                f"Updated Fermi levels: left={self.left_fermi_level}, right={self.right_fermi_level}\nLeft mid-gap={self.left_mid_gap_energy}, right mid-gap={self.right_mid_gap_energy}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )

        t_obc_start = time.perf_counter()
        self._compute_obc()
        synchronize_device()
        t_obc_end = time.perf_counter()
        comm.barrier()
        t_obc_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    OBC: {t_obc_end - t_obc_start}", flush=True)
            print(f"    OBC all: {t_obc_end_all - t_obc_start}", flush=True)

        if comm.block.size > 1:
            t_solve_start = time.perf_counter()
            self.solver_dist.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    Solve: {t_solve_end - t_solve_start}", flush=True)
                print(f"    Solve all: {t_solve_end_all - t_solve_start}", flush=True)

        else:
            t_solve_start = time.perf_counter()
            self.meir_wingreen_current = self.solver.selected_solve(
                a=self.system_matrix,
                sigma_lesser=sse_lesser,
                sigma_greater=sse_greater,
                obc_blocks=self.obc_blocks,
                out=out,
                return_retarded=True,
                return_current=self.compute_meir_wingreen_current,
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    Solve: {t_solve_end - t_solve_start}", flush=True)
                print(f"    Solve all: {t_solve_end_all - t_solve_start}", flush=True)

        t_filter_peaks_start = time.perf_counter()
        # Free the system matrix data to save memory
        # Don't free the data here since we might need to do another solve
        # in the case of potential update.
        # self.system_matrix.free_data()
        if self.call_count < self.filtering_iteration_limit:
            g_lesser, g_greater, g_retarded = out
            local_mask = filtering_peaks_mask(
                g_retarded, self.energies, self.dos_peak_limit
            )
            g_lesser.data[local_mask] = 0.0
            g_greater.data[local_mask] = 0.0
            g_retarded.data[local_mask] = 0.0

        synchronize_device()
        t_filter_peaks_end = time.perf_counter()
        comm.barrier()
        t_filter_peaks_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    Filter peaks: {t_filter_peaks_end - t_filter_peaks_start}",
                flush=True,
            )
            print(
                f"    Filter peaks all: {t_filter_peaks_end_all - t_filter_peaks_start}",
                flush=True,
            )

        if self.band_edge_tracking in [
            "dos-peaks",
            "secant-method",
            "potential-update",
        ] or (self.call_count >= 1 and self.band_edge_tracking == "charge-neutrality"):
            t_dos_peaks_start = time.perf_counter()

            if self.band_edge_tracking == "potential-update":
                g_lesser, _, g_retarded = out
            else:
                _, _, g_retarded = out
            left_band_edges = np.empty((2,), dtype=float)
            right_band_edges = np.empty((2,), dtype=float)

            # NOTE: This will not work if comm.block.size > 1
            if comm.block.size > 1:
                raise NotImplementedError(
                    "Band edge tracking with block distribution is not implemented."
                )
            # NOTE: Also assumes it is block homogeneous
            # Check that g_retarded.block_sizes are all the same
            if not np.all(g_retarded.block_sizes == g_retarded.block_sizes[0]):
                raise NotImplementedError(
                    "Band edge tracking with non-uniform block sizes is not implemented."
                )

            # TODO: Check sign
            ldos = -self._block_resolved_dos(g_retarded) / xp.pi

            band_edges = self._find_band_edges(ldos)
            left_dos = ldos[..., 0]
            left_band_edges = band_edges[0]
            right_dos = ldos[..., -1]
            right_band_edges = band_edges[-1]

            # NOTE: This will not work if comm.block.size > 1
            if self.band_edge_tracking == "charge-neutrality":
                # Update the mid band gap
                self.left_mid_gap_energy = xp.mean(left_band_edges)
                left_fermi_level = contact_fermi_level(
                    temperature=self.temperature,
                    dos=left_dos,
                    energies=self.energies,
                    doping_density=self.left_target_charge,
                    midgap_energy=self.left_mid_gap_energy,
                )
            elif self.band_edge_tracking == "secant-method":
                # Update the mid band gap
                self.left_mid_gap_energy = xp.mean(left_band_edges)
                self.right_mid_gap_energy = xp.mean(right_band_edges)
                # Add the current Fermi level guess to the list of tracked Fermi levels for the secant method.
                self.fermi_levels.append(self.left_fermi_level)
                # Compute the charge of the current iteration.
                neq_distribution = fermi_dirac(
                    self.energies - self.left_fermi_level, self.temperature
                ) - fermi_dirac(
                    self.energies - self.left_mid_gap_energy, self.temperature
                )
                charge_density = xp.trapz(
                    left_dos * neq_distribution, self.energies, axis=0
                )
                # If the charge density and target charge are too far off, redo the caculation of the Green's function
                # with the correct Fermi level.
                # if xp.abs(charge_density - self.left_target_charge) / xp.abs(self.left_target_charge) > 1.0:
                if True:
                    if comm.rank == 0:
                        print(
                            "Charge density is too far from target charge. Redoing calculation with correct Fermi level.",
                            flush=True,
                        )
                    # Zero the output Green's functions to be safe.
                    for g in out:
                        g.data[:] = 0.0
                    self._update_fermi_and_solve(
                        left_dos=left_dos,
                        right_dos=right_dos,
                        sse_lesser=sse_lesser,
                        sse_greater=sse_greater,
                        out=out,
                    )
                    # Replace last Fermi level with the new one.
                    self.fermi_levels[-1] = self.left_fermi_level
                    # Recompute the charge density with the updated Green's function.
                    neq_distribution = fermi_dirac(
                        self.energies - self.left_fermi_level, self.temperature
                    ) - fermi_dirac(
                        self.energies - self.left_mid_gap_energy, self.temperature
                    )
                    charge_density = xp.trapz(
                        left_dos * neq_distribution, self.energies, axis=0
                    )
                if comm.rank == 0:
                    print(
                        f"Left Fermi level: {self.left_fermi_level}\n",
                        f"Left Mid-gap energy: {self.left_mid_gap_energy}\n",
                        f"Right Fermi level: {self.right_fermi_level}\n",
                        f"Right Mid-gap energy: {self.right_mid_gap_energy}\n",
                        f"Current charge: {charge_density}, target charge: {self.left_target_charge}",
                        flush=True,
                    )
                self.charge_densities.append(charge_density - self.left_target_charge)
                if len(self.fermi_levels) > 1:
                    # update_value = self.charge_densities[-1] * (self.fermi_levels[-1] - self.fermi_levels[-2]) / (self.charge_densities[-1] - self.charge_densities[-2])
                    # left_fermi_level = self.left_fermi_level - update_value

                    # Use the secant method to compute the next Fermi level guess.
                    update_value = (
                        self.charge_densities[-1]
                        * (self.fermi_levels[-1] - self.fermi_levels[-2])
                        / (self.charge_densities[-1] - self.charge_densities[-2])
                    )
                    if np.abs(update_value) > 0.2:
                        if comm.rank == 0:
                            print(
                                f"Large update value detected: {update_value}. Falling back to using previous charge neutrality.",
                                flush=True,
                            )
                        left_fermi_level = contact_fermi_level(
                            temperature=self.temperature,
                            dos=left_dos,
                            energies=self.energies,
                            doping_density=self.left_target_charge,
                            midgap_energy=self.left_mid_gap_energy,
                        )
                    else:
                        left_fermi_level = self.left_fermi_level - update_value

                    # charge_diff = self.charge_densities[-1] - self.charge_densities[-2]
                    # # If the charge difference is smaller than a certain threshold,
                    # # we fallback to a simple update to avoid numerical issues.
                    # if abs(charge_diff) < 1e-4:
                    #     if comm.rank == 0:
                    #         print(
                    #             f"Small charge difference detected: {charge_diff}. Falling back to using previous charge neutrality.",
                    #             flush=True,
                    #         )
                    #     left_fermi_level = contact_fermi_level(
                    #         temperature=self.temperature,
                    #         dos=left_dos,
                    #         energies=self.energies,
                    #         doping_density=self.left_target_charge,
                    #         midgap_energy=self.left_mid_gap_energy,
                    #     )
                    # else:
                    #     left_fermi_level = (
                    #         self.fermi_levels[-2] * self.charge_densities[-1]
                    #         - self.fermi_levels[-1] * self.charge_densities[-2]
                    #     ) / charge_diff

            elif self.band_edge_tracking == "potential-update":
                # TODO: What about mid-gap energy? Fermi stays the same but mid-gap energy changes
                # Update the mid band gap
                self.left_mid_gap_energy = xp.mean(left_band_edges)
                self.right_mid_gap_energy = xp.mean(right_band_edges)
                # Also compute the excess charge
                # TODO: Check sign
                el_ldos = self._block_resolved_dos(g_lesser) / (2 * np.pi)
                excess_charge = []
                midgap_energies = xp.mean(band_edges, axis=1)
                for b in range(el_ldos.shape[-1]):
                    mask = self.energies > midgap_energies[b]
                    excess_charge.append(
                        np.trapezoid(el_ldos[:, b][mask], self.energies[mask])
                    )
                # Zero the output Green's functions to be safe.
                for g in out:
                    g.data[:] = 0.0
                self._update_potential_and_solve(
                    ldos=ldos,
                    excess_charge=excess_charge,
                    midgap_energies=midgap_energies,
                    sse_retarded=sse_retarded,
                    sse_lesser=sse_lesser,
                    sse_greater=sse_greater,
                    out=out,
                )
                if comm.rank == 0:
                    print(
                        f"Left Fermi level: {self.left_fermi_level}\n",
                        f"Left Mid-gap energy: {self.left_mid_gap_energy}\n",
                        f"Right Fermi level: {self.right_fermi_level}\n",
                        f"Right Mid-gap energy: {self.right_mid_gap_energy}\n",
                        flush=True,
                    )

            if self.band_edge_tracking == "dos-peaks":
                comm.block.bcast(left_band_edges, root=0, backend="device_mpi")
                comm.block.bcast(
                    right_band_edges, root=comm.block.size - 1, backend="device_mpi"
                )
                self._update_fermi_levels(left_band_edges, right_band_edges)
            elif self.band_edge_tracking in ["charge-neutrality", "secant-method"]:
                comm.block.bcast(left_fermi_level, root=0, backend="device_mpi")
                # Mix the Fermi level with the previous one.
                self.left_fermi_level = (
                    self.fermi_level_mixing * left_fermi_level
                    + (1 - self.fermi_level_mixing) * self.left_fermi_level
                )
                # What should I do here? It was commented out before (maybe for printing)
                self.right_fermi_level = self.left_fermi_level - self.bias
                self.left_occupancies = fermi_dirac(
                    self.local_energies - self.left_fermi_level,
                    self.temperature,
                )
                self.right_occupancies = fermi_dirac(
                    self.local_energies - self.right_fermi_level,
                    self.temperature,
                )

            synchronize_device()
            t_dos_peaks_end = time.perf_counter()
            comm.barrier()
            t_dos_peaks_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"    DOS peaks: {t_dos_peaks_end - t_dos_peaks_start}", flush=True
                )
                print(
                    f"    DOS peaks all: {t_dos_peaks_end_all - t_dos_peaks_start}",
                    flush=True,
                )
        # Free the system matrix data to save memory
        self.system_matrix.free_data()

        self.call_count += 1
