# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from qttools.utils.stack_utils import scale_stack
from quatrex.bandstructure.band_edges import find_renormalized_eigenvalues
from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.utils import get_periodic_superblocks, homogenize
from quatrex.device.inputs import assemble_matrix

profiler = Profiler()


def _btd_add(a: DSDBSparse, b: DSDBSparse) -> None:
    """Adds b to a on the block-tridiagonal.

    This is an in-place operation, i.e. a is modified.

    Parameters
    ----------
    a : DSDBSparse
        The matrix to add to.
    b : DSDBSparse
        The matrix to add.

    """
    a_ = a.stack[...]
    b_ = b.stack[...]
    for i in range(a.num_local_blocks):
        j = i + 1
        a_.blocks[i, i] += b_.blocks[i, i]

        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        a_.blocks[i, j] += b_.blocks[i, j]
        a_.blocks[j, i] += b_.blocks[j, i]


def _btd_apply_potential(
    a: DSDBSparse, overlap: DSDBSparse, potential: NDArray
) -> None:
    """Applies the potential to a on the block-tridiagonal.

    This is an in-place operation, i.e. a is modified.

    Parameters
    ----------
    a : DSDBSparse
        The matrix to apply the potential to.
    overlap : DSDBSparse
        The overlap matrix.
    potential : NDArray
        The potential to apply.

    """
    a_ = a.stack[...]
    overlap_ = overlap.stack[...]
    offset = 0
    for i in range(a.num_local_blocks):
        j = i + 1
        s_ii = overlap_.blocks[i, i]
        potential_i = potential[offset : offset + s_ii.shape[-1]]

        a_.blocks[i, i] -= (
            s_ii * potential_i[..., np.newaxis] + s_ii * potential_i
        ) / 2

        offset += s_ii.shape[-1]

        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        s_ij = overlap_.blocks[i, j]
        s_ji = overlap_.blocks[j, i]
        potential_j = potential[offset : offset + s_ij.shape[-1]]

        a_.blocks[i, j] -= (
            s_ij * potential_i[..., np.newaxis] + s_ij * potential_j
        ) / 2
        a_.blocks[j, i] -= (
            s_ji * potential_j[..., np.newaxis] + s_ji * potential_i
        ) / 2


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
        sparsity_pattern: sparse.coo_matrix,
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

        # Make sure that the the system matrix sparsity is a superset of
        # self-energy and Hamiltonian sparsity.
        sparsity_pattern += hamiltonian_sparsity_pattern

        del hamiltonian_sparsity_pattern
        self.block_sizes = self.hamiltonian.block_sizes

        try:
            # Attempt to load the device overlap matrix.
            self.overlap, overlap_sparsity_pattern = assemble_matrix(
                config=config,
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

            if comm.rank == 0:
                print("Non-orthogonal basis detected.", flush=True)

        except FileNotFoundError:
            self.overlap = None
            if comm.rank == 0:
                print("No overlap matrix found. Assuming orthogonal basis.", flush=True)

        # Allocate memory for the system matrix.
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([int(k) for k in config.device.kpoint_grid if k > 1]),
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

        self.left_fermi_level = (
            left_conduction_band_edge - self.delta_fermi_level_conduction_band
        )
        self.right_fermi_level = (
            right_conduction_band_edge - self.delta_fermi_level_conduction_band
        )

        (
            print(
                f"Updating conduction band edges: "
                f"{left_conduction_band_edge:.6f}, {right_conduction_band_edge:.6f}\n",
                f"Updating Fermi levels: {self.left_fermi_level:.6f}, {self.right_fermi_level:.6f}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level,
            self.temperature,
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level,
            self.temperature,
        )

    @profiler.profile(label="ElectronSolver: OBC", level="default", comm=comm)
    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""
        if comm.block.rank == 0:

            m_10, m_00, m_01 = get_periodic_superblocks(
                a_ii=self.system_matrix.blocks[0, 0],
                a_ji=self.system_matrix.blocks[1, 0],
                a_ij=self.system_matrix.blocks[0, 1],
                block_sections=self.block_sections,
            )

            if self.overlap is None:
                s_00 = 1j * self.eta_obc * xp.eye(m_00.shape[-1], dtype=m_00.dtype)
                s_01 = xp.zeros_like(m_01, dtype=m_01.dtype)
                s_10 = xp.zeros_like(m_10, dtype=m_10.dtype)
            else:
                # Extract the overlap matrix blocks.
                s_00 = 1j * self.eta_obc * self.overlap.blocks[0, 0]
                s_01 = 1j * self.eta_obc * self.overlap.blocks[0, 1]
                s_10 = 1j * self.eta_obc * self.overlap.blocks[1, 0]

            # TODO: use residuals to filter "bad" energies
            g_00, *__ = self.obc(
                (m_00 + s_00, m_01 + s_01, m_10 + s_10),
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
            n = self.system_matrix.num_local_blocks - 1
            m = n - 1

            m_mn, m_nn, m_nm = get_periodic_superblocks(
                # Twist it, flip it, ...
                a_ii=xp.flip(self.system_matrix.blocks[n, n], axis=(-2, -1)),
                a_ji=xp.flip(self.system_matrix.blocks[m, n], axis=(-2, -1)),
                a_ij=xp.flip(self.system_matrix.blocks[n, m], axis=(-2, -1)),
                block_sections=self.block_sections,
            )

            if self.overlap is None:
                s_nn = 1j * self.eta_obc * xp.eye(m_nn.shape[-1], dtype=m_nn.dtype)
                s_nm = xp.zeros_like(m_nm, dtype=m_nm.dtype)
                s_mn = xp.zeros_like(m_mn, dtype=m_mn.dtype)
            else:
                # Extract the overlap matrix blocks.
                s_nn = 1j * self.eta_obc * self.overlap.blocks[n, n]
                s_nm = 1j * self.eta_obc * self.overlap.blocks[n, m]
                s_mn = 1j * self.eta_obc * self.overlap.blocks[m, n]

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

    def _subtract_hamiltonian_and_self_energy(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        sse_retarded_hermitian: DSDBSparse,
    ) -> None:
        r"""Subtracts the Hamiltonian and the self-energy from the system matrix on the block-tridiagonal.

        $$\mathbf{M} \mathrel{{-}{=}} \mathbf{H} + \mathbf{\Sigma}^R + \frac{1}{2} \left(\mathbf{\Sigma}^{>} - \mathbf{\Sigma}^{<} \right)$$

        This modifies the system matrix in-place, i.e. the result is stored in `self.system_matrix`.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser self-energy to subtract.
        sse_greater : DSDBSparse
            The greater self-energy to subtract.
        sse_retarded_hermitian : DSDBSparse
            The retarded self-energy to subtract.

        """
        system_matrix_ = self.system_matrix.stack[...]
        hamiltonian_ = self.hamiltonian.stack[...]
        sse_retarded_hermitian_ = sse_retarded_hermitian.stack[...]
        sse_lesser_ = sse_lesser.stack[...]
        sse_greater_ = sse_greater.stack[...]
        for i in range(self.system_matrix.num_local_blocks):
            j = i + 1
            system_matrix_.blocks[i, i] -= (
                sse_retarded_hermitian_.blocks[i, i]
                + 0.5 * (sse_greater_.blocks[i, i] - sse_lesser_.blocks[i, i])
                + hamiltonian_.blocks[i, i]
            )

            if (
                j >= self.system_matrix.num_local_blocks
                and comm.block.rank == comm.block.size - 1
            ):
                # The last rank does not have these blocks.
                continue

            system_matrix_.blocks[i, j] -= (
                sse_retarded_hermitian_.blocks[i, j]
                + 0.5 * (sse_greater_.blocks[i, j] - sse_lesser_.blocks[i, j])
                + hamiltonian_.blocks[i, j]
            )
            system_matrix_.blocks[j, i] -= (
                sse_retarded_hermitian_.blocks[j, i]
                + 0.5 * (sse_greater_.blocks[j, i] - sse_lesser_.blocks[j, i])
                + hamiltonian_.blocks[j, i]
            )

    def _assemble_system_matrix(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        sse_retarded_hermitian: DSDBSparse,
    ) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser scattering self-energy.
        sse_greater : DSDBSparse
            The greater scattering self-energy.
        sse_retarded_hermitian : DSDBSparse
            The hermitian part of the retarded scattering self-energy.

        """
        self.system_matrix.data = 0.0
        if self.overlap is None:
            self.system_matrix.fill_diagonal(1.0)
        else:
            _btd_add(self.system_matrix, self.overlap)

        scale_stack(
            self.system_matrix.data,
            self.local_energies + 1j * self.eta,
        )

        if self.overlap is None:
            self.system_matrix -= sparse.diags(self.potential, format="csr")
        else:
            _btd_apply_potential(self.system_matrix, self.overlap, self.potential)

        self._subtract_hamiltonian_and_self_energy(
            sse_lesser,
            sse_greater,
            sse_retarded_hermitian,
        )

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
        sse_retarded_hermitian: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ):
        """Solves for the electron Green's function.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser self-energy.
        sse_greater : DSDBSparse
            The greater self-energy.
        sse_retarded_hermitian : DSDBSparse
            The hermitian part of the retarded self-energy.
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
                homogenize(sse_retarded_hermitian)

        with profiler.profile_range(
            label="ElectronSolver: Assemble", level="default", comm=comm
        ):
            self.system_matrix.allocate_data()

            self._assemble_system_matrix(
                sse_lesser, sse_greater, sse_retarded_hermitian
            )

        if self.band_edge_tracking:
            with profiler.profile_range(
                label="ElectronSolver: Band edges", level="default", comm=comm
            ):
                left_band_edges, right_band_edges = find_renormalized_eigenvalues(
                    hamiltonian=self.hamiltonian,
                    overlap=self.overlap,
                    potential=self.potential,
                    sigma_lesser=sse_lesser,
                    sigma_greater=sse_greater,
                    sigma_retarded_hermitian=sse_retarded_hermitian,
                    energies=self.energies,
                    conduction_band_guesses=(
                        self.left_fermi_level + self.delta_fermi_level_conduction_band,
                        self.right_fermi_level + self.delta_fermi_level_conduction_band,
                    ),
                    mid_gap_energies=(
                        self.left_mid_gap_energy,
                        self.right_mid_gap_energy,
                    ),
                    band_edge_config=self.config.compute.band_edge,
                )
                self._update_fermi_levels(left_band_edges, right_band_edges)

        self._compute_obc()

        with profiler.profile_range(
            label="ElectronSolver: Solve", level="default", comm=comm
        ):
            if comm.block.size > 1:
                self.meir_wingreen_current = self.solver_dist.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=sse_lesser,
                    sigma_greater=sse_greater,
                    obc_blocks=self.obc_blocks,
                    out=out,
                    return_retarded=True,
                    return_current=self.compute_meir_wingreen_current,
                )

            else:
                self.meir_wingreen_current = self.solver.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=sse_lesser,
                    sigma_greater=sse_greater,
                    obc_blocks=self.obc_blocks,
                    out=out,
                    return_retarded=True,
                    return_current=self.compute_meir_wingreen_current,
                )

        with profiler.profile_range(
            label="ElectronSolver: Filter", level="default", comm=comm
        ):
            self.system_matrix.free_data()
            if self.call_count < self.filtering_iteration_limit:
                self._filter_peaks(out)

        self.call_count += 1
