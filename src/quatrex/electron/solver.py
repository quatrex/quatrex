# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.utils.mpi_utils import distributed_load
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
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(quatrex_config, compute_config, energies)

        # Load the device Hamiltonian.
        self.hamiltonian_sparray = distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        ).astype(xp.complex128)
        self.block_sizes = distributed_load(
            quatrex_config.input_dir / "block_sizes.npy"
        )
        self.block_offsets = xp.hstack(([0], xp.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.hamiltonian_sparray.shape[0]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.hamiltonian_sparray.shape[0]}"
            )
        # Load the overlap matrix.
        try:
            self.overlap_sparray = distributed_load(
                quatrex_config.input_dir / "overlap.npz"
            ).astype(xp.complex128)
        except FileNotFoundError:
            # No overlap provided. Assume orthonormal basis.
            self.overlap_sparray = sparse.eye(
                self.hamiltonian_sparray.shape[0],
                format="coo",
                dtype=self.hamiltonian_sparray.dtype,
            )

        # Check that the overlap matrix and Hamiltonian matrix match.
        if self.overlap_sparray.shape != self.hamiltonian_sparray.shape:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )

        # Construct the bare system matrix.
        self.bare_system_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
        )
        self.bare_system_matrix.data = 0.0

        self.bare_system_matrix += self.overlap_sparray
        scale_stack(
            self.bare_system_matrix.data,
            self.local_energies + 1j * quatrex_config.electron.eta,
        )
        self.bare_system_matrix -= self.hamiltonian_sparray

        # Load the potential.
        try:
            self.potential = distributed_load(
                quatrex_config.input_dir / "potential.npy"
            )
            if self.potential.size != self.hamiltonian_sparray.shape[0]:
                raise ValueError(
                    "Potential matrix and Hamiltonian have different shapes."
                )
        except FileNotFoundError:
            # No potential provided. Assume zero potential.
            self.potential = xp.zeros(
                self.hamiltonian_sparray.shape[0], dtype=self.hamiltonian_sparray.dtype
            )

        self.bare_system_matrix -= sparse.diags(self.potential)

        self.system_matrix = compute_config.dsbsparse_type.zeros_like(
            self.bare_system_matrix
        )

        # Contacts.
        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc

        self.compute_meir_wingreen_current = (
            quatrex_config.electron.solver.compute_current
        )

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
        self.left_mid_gap_energy = quatrex_config.electron.left_fermi_level
        self.right_mid_gap_energy = quatrex_config.electron.right_fermi_level

        self.temperature = quatrex_config.electron.temperature

        self.left_fermi_level = quatrex_config.electron.left_fermi_level
        self.right_fermi_level = quatrex_config.electron.right_fermi_level

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

        # Prepare Buffers for OBC.
        self.obc_blocks = OBCBlocks(num_blocks=self.block_sizes.size)

    def update_potential(self, new_potential: NDArray) -> None:
        """Updates the potential matrix.

        Parameters
        ----------
        new_potential : NDArray
            The new potential matrix.

        """
        potential_diff_matrix = sparse.diags(new_potential - self.potential)
        self.bare_system_matrix -= potential_diff_matrix
        self.potential = new_potential

    def _update_fermi_levels(self, e_0_left: NDArray, e_0_right: NDArray) -> None:
        """Updates the Fermi levels.

        Parameters
        ----------
        out : tuple[DSBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """
        left_band_edges = find_band_edges(e_0_left, self.left_mid_gap_energy)
        right_band_edges = find_band_edges(e_0_right, self.right_mid_gap_energy)

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

    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""

        # Extract the overlap matrix blocks.
        s_00 = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (0, 0))
        s_01 = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (0, 1))
        s_10 = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (1, 0))
        s_nn = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (-1, -1))
        s_nm = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (-1, -2))
        s_mn = 1j * self.eta_obc * self._get_block(self.overlap_sparray, (-2, -1))

        # Compute surface Green's functions.
        g_00 = self.obc(
            self.system_matrix.blocks[0, 0] + s_00,
            self.system_matrix.blocks[0, 1] + s_01,
            self.system_matrix.blocks[1, 0] + s_10,
            "left",
        )
        # Twist it, flip it, bop it.
        g_nn = self.obc(
            xp.flip(self.system_matrix.blocks[-1, -1] + s_nn, axis=(-2, -1)),
            xp.flip(self.system_matrix.blocks[-1, -2] + s_nm, axis=(-2, -1)),
            xp.flip(self.system_matrix.blocks[-2, -1] + s_mn, axis=(-2, -1)),
            "right",
        )
        g_nn = xp.flip(g_nn, axis=(-2, -1))

        # NOTE: Here we could possibly do peak/discontinuity detection
        # on the surface Green's function DOS (not same as actual DOS).

        # Apply the retarded boundary self-energy.
        sigma_00 = (
            self.system_matrix.blocks[1, 0] @ g_00 @ self.system_matrix.blocks[0, 1]
        )
        sigma_nn = (
            self.system_matrix.blocks[-2, -1] @ g_nn @ self.system_matrix.blocks[-1, -2]
        )

        self.obc_blocks.retarded[0] = sigma_00
        self.obc_blocks.retarded[-1] = sigma_nn

        gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))
        gamma_nn = 1j * (sigma_nn - sigma_nn.conj().swapaxes(-2, -1))

        # Compute and apply the lesser boundary self-energy.
        self.obc_blocks.lesser[0] = 1j * scale_stack(
            gamma_00.copy(), self.left_occupancies
        )
        self.obc_blocks.lesser[-1] = 1j * scale_stack(
            gamma_nn.copy(), self.right_occupancies
        )

        # Compute and apply the greater boundary self-energy.
        self.obc_blocks.greater[0] = 1j * scale_stack(
            gamma_00.copy(), self.left_occupancies - 1
        )
        self.obc_blocks.greater[-1] = 1j * scale_stack(
            gamma_nn.copy(), self.right_occupancies - 1
        )

    def _assemble_system_matrix(self, sse_retarded: DSBSparse) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_retarded : DSBSparse
            The retarded scattering self-energy.

        """
        self.system_matrix.data = self.bare_system_matrix.data
        self.system_matrix -= sse_retarded

    def _filter_peaks(self, out: tuple[DSBSparse, ...]) -> None:
        """Filters out peaks in the Green's functions.

        Parameters
        ----------
        out : tuple[DSBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
        """

        # NOTE: This filtering only works in flatband systems.
        # Otherwise, the dos and charge densities should be summed per
        # transport cell (in the old code the trace is taken during
        # RGF).
        # TODO: make parameters settable.
        if self.flatband:
            g_lesser, g_greater, g_retarded = out
            dos = -g_retarded.diagonal().imag.sum(1)
            ne = g_lesser.diagonal().imag.sum(1)
            nh = -g_greater.diagonal().imag.sum(1)
            f1 = xp.abs(dos - (ne + nh) / 2) / (xp.abs(dos) + 1e-6)
            f2 = xp.abs(dos - (ne + nh) / 2) / (xp.abs((ne + nh) / 2) + 1e-6)

            # peaks = find_peaks(xp.abs(dos), height=0.5)[0]
            # TODO: Should communicate boundary energies to correctly
            # filter makes arrays of bools
            # f3 = xp.zeros_like(f1, dtype=bool)
            # f3[peaks] = True

            mask = (f1 > 1e-1) | (f2 > 1e-1) | (dos < 0)

            assert g_lesser.distribution_state == "stack"
            g_lesser.data[mask] = 0.0
            g_greater.data[mask] = 0.0
            g_retarded.data[mask] = 0.0

    def solve(
        self,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ):
        """Solves for the electron Green's function.

        Parameters
        ----------
        sse_lesser : DSBSparse
            The lesser self-energy.
        sse_greater : DSBSparse
            The greater self-energy.
        sse_retarded : DSBSparse
            The retarded self-energy.
        out : tuple[DSBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """
        times = []

        times.append(time.perf_counter())
        self._assemble_system_matrix(sse_retarded)
        t_assemble = time.perf_counter() - times.pop()

        if self.band_edge_tracking == "eigenvalues":
            e_0_left, e_0_right = find_renormalized_eigenvalues(
                self.hamiltonian_sparray,
                self.overlap_sparray,
                self.potential,
                sse_retarded,
                self.energies,
                (
                    self.left_fermi_level + self.delta_fermi_level_conduction_band,
                    self.right_fermi_level + self.delta_fermi_level_conduction_band,
                ),
                (self.left_mid_gap_energy, self.right_mid_gap_energy),
            )
            self._update_fermi_levels(e_0_left, e_0_right)

        times.append(time.perf_counter())
        self._compute_obc()
        t_obc = time.perf_counter() - times.pop()

        times.append(time.perf_counter())
        self.meir_wingreen_current = self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=sse_lesser,
            sigma_greater=sse_greater,
            obc_blocks=self.obc_blocks,
            out=out,
            return_retarded=True,
            return_current=self.compute_meir_wingreen_current,
        )
        t_solve = time.perf_counter() - times.pop()
        g_lesser, g_greater, g_retarded = out

        # Make sure the Green's functions are skew-Hermitian.
        g_lesser.data = 0.5 * (
            g_lesser.data - g_lesser.ltranspose(copy=True).data.conj()
        )
        g_greater.data = 0.5 * (
            g_greater.data - g_greater.ltranspose(copy=True).data.conj()
        )

        self._filter_peaks(out)

        if self.band_edge_tracking == "dos-peaks":
            s_00 = self._get_block(self.overlap_sparray, (0, 0))
            s_nn = self._get_block(self.overlap_sparray, (-1, -1))
            g_00 = g_retarded.blocks[0, 0]
            g_nn = g_retarded.blocks[-1, -1]

            local_left_dos = -xp.mean(
                xp.diagonal(g_00 @ s_00, axis1=-2, axis2=-1).imag, axis=-1
            )
            local_right_dos = -xp.mean(
                xp.diagonal(g_nn @ s_nn, axis1=-2, axis2=-1).imag, axis=-1
            )
            left_dos = xp.hstack(comm.allgather(local_left_dos)) / (2 * xp.pi)
            right_dos = xp.hstack(comm.allgather(local_right_dos)) / (2 * xp.pi)

            e_0_left = find_dos_peaks(left_dos, self.energies)
            e_0_right = find_dos_peaks(right_dos, self.energies)

            self._update_fermi_levels(e_0_left, e_0_right)

        (
            print(
                f"Assemble: {t_assemble}, OBC: {t_obc}, Solve: {t_solve}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )
