# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.observables import current_conservation, density, device_current
from quatrex.core.transport import TransportSolver
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.device import Device
from quatrex.device.inputs import assemble_matrix, get_block_sizes
from quatrex.electron import (
    ElectronSolver,
    SigmaCoulombScreening,
    SigmaFock,
    SigmaPhonon,
    SigmaPhoton,
)
from quatrex.grid import get_electron_energies
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton

profiler = Profiler()


class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.

    """

    def __init__(self, config: QuatrexConfig, electron_energies: NDArray) -> None:
        """Initializes the SCBA data."""
        # Load orbital positions, energy vector and block-sizes.

        grid, __, __ = Device.load_structure(config)
        block_sizes = get_block_sizes(config, grid)

        kpoint_grid = config.device.kpoint_grid
        # Find the maximum interaction cutoff.
        max_interaction_cutoff = 0.0
        if config.scba.coulomb_screening:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.coulomb_screening.interaction_cutoff,
            )
        if config.scba.photon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.photon.interaction_cutoff,
            )
        if config.scba.phonon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.phonon.interaction_cutoff,
            )
        if max_interaction_cutoff == 0.0:
            raise NotImplementedError(
                "At least one interaction must be enabled in the SCBA."
                "Ballistic transport is not properly supported yet."
            )

        if comm.rank == 0:
            print(f"Max Interaction Cutoff: {max_interaction_cutoff}", flush=True)

        with profiler.profile_range(
            label="SCBA: Sparsity Pattern", level="default", comm=comm
        ):
            # Determine the local slice of the data.
            # NOTE: This is arrow-wise partitioning.
            # TODO: Allow more options, e.g., block row-wise partitioning.
            section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
            start_idx = block_offsets[section_offsets[comm.block.rank]]
            end_idx = block_offsets[section_offsets[comm.block.rank + 1]]

            self.sparsity_pattern = compute_sparsity_pattern(
                grid,
                max_interaction_cutoff,
                transport_direction=config.device.transport_direction,
                start_idx=start_idx,
                end_idx=end_idx,
            )

        dsdbsparse_type = config.compute.dsdbsparse_type

        self.g_retarded = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
        )
        self.g_retarded.data[:] = 0.0  # Initialize to zero.

        self.g_lesser = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry=config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )
        self.g_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_lesser_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_retarded_hermitian_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_retarded_hermitian = dsdbsparse_type.zeros_like(self.g_lesser)
        if config.scba.symmetric:
            self.sigma_retarded_hermitian.symmetry_op = lambda a: a.conj()
            self.sigma_retarded_hermitian_prev.symmetry_op = lambda a: a.conj()

        if config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_retarded_hermitian = dsdbsparse_type.zeros_like(self.g_lesser)
            self.p_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
            self.p_greater = dsdbsparse_type.zeros_like(self.g_lesser)

            if config.scba.symmetric:
                self.p_retarded_hermitian.symmetry_op = lambda a: a.conj()

            num_connected_blocks = config.coulomb_screening.num_connected_blocks
            if num_connected_blocks == "auto":
                num_connected_blocks = compute_num_connected_blocks(
                    self.sparsity_pattern, block_sizes
                )

            if comm.rank == 0:
                print(f"Number of connected blocks: {num_connected_blocks}", flush=True)

            # TODO: This only works for constant block sizes.
            coulomb_screening_block_sizes = (
                block_sizes[: len(block_sizes) // num_connected_blocks]
                * num_connected_blocks
            )

            self.w_lesser = dsdbsparse_type.from_sparray(
                self.sparsity_pattern.astype(xp.complex128),
                block_sizes=coulomb_screening_block_sizes,
                global_stack_shape=electron_energies.shape
                + tuple([k for k in kpoint_grid if k > 1]),
                symmetry=config.scba.symmetric,
                symmetry_op=lambda a: -a.conj(),
            )
            self.w_greater = dsdbsparse_type.zeros_like(self.w_lesser)

        # TODO: The interactions with photons and phonons are not yet
        # implemented.
        if config.scba.photon:
            raise NotImplementedError

        if config.scba.phonon and config.phonon.model == "negf":
            raise NotImplementedError


@dataclass
class Observables:
    """Observable quantities for the SCBA."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)

    valence_band_edges: NDArray = None
    conduction_band_edges: NDArray = None

    excess_charge_density: NDArray = None

    electron_electron_scattering_rate: NDArray = None
    electron_photon_scattering_rate: NDArray = None
    electron_phonon_scattering_rate: NDArray = None

    sigma_lesser_density: NDArray = None
    sigma_greater_density: NDArray = None

    # --- Coulomb screening --------------------------------------------
    w_lesser_density: NDArray = None
    w_greater_density: NDArray = None

    p_lesser_density: NDArray = None
    p_greater_density: NDArray = None

    # --- Photons ------------------------------------------------------
    pi_photon_retarded_density: NDArray = None
    pi_photon_lesser_density: NDArray = None
    pi_photon_greater_density: NDArray = None

    d_photon_retarded_density: NDArray = None
    d_photon_lesser_density: NDArray = None
    d_photon_greater_density: NDArray = None

    photon_current_density: NDArray = None

    # --- Phonons ------------------------------------------------------
    pi_phonon_retarded_density: NDArray = None
    pi_phonon_lesser_density: NDArray = None
    pi_phonon_greater_density: NDArray = None
    d_phonon_retarded_density: NDArray = None
    d_phonon_lesser_density: NDArray = None
    d_phonon_greater_density: NDArray = None

    thermal_current: NDArray = None


class SCBA(TransportSolver):
    """Self-consistent Born approximation (SCBA) solver.

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        """Initializes an SCBA instance."""
        self.config = config

        self.observables = Observables()
        electron_energies = xp.zeros((comm.size,))
        self.data = SCBAData(config, electron_energies=electron_energies)  # dummy data
        self.mixing_factor = self.config.scba.mixing_factor

        # ----- Electrons ----------------------------------------------
        self.electron_energies = get_electron_energies(config)

        min_energy = self.electron_energies[0]
        max_energy = self.electron_energies[-1]
        num_energies = len(self.electron_energies)
        energy_resolution = self.electron_energies[1] - self.electron_energies[0]
        num_energies_per_rank = num_energies // comm.stack.size
        if comm.rank == 0:
            print(
                f"Energy window: {min_energy} to {max_energy} eV with {num_energies} grid points.",
                flush=True,
            )
            print(f"Resolution is {energy_resolution:.6f} eV.", flush=True)
            print(
                f"comm.stack size: {comm.stack.size}, comm.block size: {comm.block.size}",
                flush=True,
            )
            print(
                f"Each comm.block has {num_energies_per_rank} grid points.", flush=True
            )

        self.electron_solver = ElectronSolver(
            self.config,
            self.electron_energies,
            sparsity_pattern=self.data.sparsity_pattern,
        )

        # ----- Coulomb screening --------------------------------------
        if self.config.scba.coulomb_screening:
            # Load the Coulomb matrix.
            coulomb_matrix, __ = assemble_matrix(
                config=config,
                matrix_name="coulomb_matrix",
                sparsity_pattern=self.data.sparsity_pattern,
                shift_kpoints=True,
            )

            # Make sure the Coulomb matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if not coulomb_matrix.symmetry:
                coulomb_matrix.symmetrize()
            coulomb_matrix._data /= config.coulomb_screening.epsilon_r

            energies_path = self.config.input_dir / "coulomb_screening_energies.npy"
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = distributed_load(energies_path)
            else:
                self.coulomb_screening_energies = (
                    self.electron_energies - self.electron_energies[0]
                )
                # Remove the zero energy to avoid division by zero.
                self.coulomb_screening_energies += 1e-6

            (
                coulomb_matrix.dtranspose()
                if coulomb_matrix.distribution_state != "nnz"
                else None
            )
            self.sigma_fock = SigmaFock(
                self.config,
                coulomb_matrix,
                self.electron_energies,
            )
            # Have to transpose the coulomb matrix back to the original distribution.
            (
                coulomb_matrix.dtranspose()
                if coulomb_matrix.distribution_state == "nnz"
                else None
            )

            # NOTE: No sparsity information required here.
            self.p_coulomb_screening = PCoulombScreening(
                self.config,
                self.coulomb_screening_energies,
            )
            self.coulomb_screening_solver = CoulombScreeningSolver(
                self.config,
                coulomb_matrix,
                self.coulomb_screening_energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
            self.sigma_coulomb_screening = SigmaCoulombScreening(
                self.config,
                self.electron_energies,
            )

        # ----- Photons ------------------------------------------------
        if self.config.scba.photon:
            energies_path = self.config.input_dir / "photon_energies.npy"
            self.photon_energies = distributed_load(energies_path)
            self.pi_photon = PiPhoton(...)
            self.photon_solver = PhotonSolver(self.config, self.photon_energies)
            self.sigma_photon = SigmaPhoton(...)

        # ----- Phonons ------------------------------------------------
        if self.config.scba.phonon:
            if self.config.phonon.model == "negf":
                energies_path = self.config.input_dir / "phonon_energies.npy"
                self.phonon_energies = distributed_load(energies_path)
                self.pi_phonon = PiPhonon(...)
                self.phonon_solver = PhononSolver(config, self.phonon_energies)
                self.sigma_phonon = SigmaPhonon(...)

            elif self.config.phonon.model == "pseudo-scattering":
                self.sigma_phonon = SigmaPhonon(config, self.electron_energies)

        self.data = SCBAData(
            config, electron_energies=self.electron_energies
        )  # real data

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_lesser_prev.data[:] = self.data.sigma_lesser.data
        self.data.sigma_greater_prev.data[:] = self.data.sigma_greater.data
        self.data.sigma_retarded_hermitian_prev.data[:] = (
            self.data.sigma_retarded_hermitian.data
        )

        self.data.sigma_retarded_hermitian.data[:] = 0.0
        self.data.sigma_lesser.data[:] = 0.0
        self.data.sigma_greater.data[:] = 0.0

    @profiler.profile(label="SCBA: Symmetrize Sigma", level="default", comm=comm)
    def _symmetrize_sigma(self) -> None:
        # Symmetrization.
        if not self.config.scba.symmetric:
            self.data.sigma_lesser.symmetrize(xp.subtract)
            self.data.sigma_greater.symmetrize(xp.subtract)
            # Make the self-energy Hermitian
            # This is done before adding the skew hermitian part coming
            # from the lesser and greater self-energies
            self.data.sigma_retarded_hermitian.symmetrize(xp.add)

        if self.config.scba.align_self_energy_to_complex_axes:
            self.data.sigma_lesser._data.real = 0
            self.data.sigma_greater._data.real = 0
            # Make sure that the imaginary part comes only from
            # sigma_greater - sigma_lesser.
            self.data.sigma_retarded_hermitian._data.imag = 0

    @profiler.profile(label="SCBA: Update Sigma", level="default", comm=comm)
    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""
        self.data.sigma_lesser.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_lesser_prev.data
            + self.mixing_factor * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_greater_prev.data
            + self.mixing_factor * self.data.sigma_greater.data
        )
        self.data.sigma_retarded_hermitian.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_retarded_hermitian_prev.data
            + self.mixing_factor * self.data.sigma_retarded_hermitian.data
        )

    @profiler.profile(label="SCBA: Convergence test", level="default", comm=comm)
    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        diff = (
            self.data.sigma_retarded_hermitian.data
            - self.data.sigma_retarded_hermitian_prev.data
        )
        local_max_diff = get_host(xp.max(xp.abs(diff)))
        max_diff = np.empty_like(local_max_diff)
        global_comm.Allreduce(local_max_diff, max_diff, op=MPI.MAX)

        meir_wingreen_current = self.observables.electron_current.get(
            "meir-wingreen", [0, 0]
        )
        i_left = xp.real(meir_wingreen_current[..., 0])
        i_right = xp.real(meir_wingreen_current[..., -1])

        dE = self.electron_energies[1] - self.electron_energies[0]
        current_diff = xp.abs(xp.sum(i_left) * dE - xp.sum(i_right) * dE)

        current_conservation_abs, current_conservation_rel = current_conservation(
            self.data.g_lesser,
            self.data.g_greater,
            self.data.sigma_lesser,
            self.data.sigma_greater,
        )

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            print(f"Contact Current Difference: {current_diff}", flush=True)
            print(f"Current Conservation abs: {current_conservation_abs}", flush=True)
            print(f"Current Conservation rel: {current_conservation_rel}", flush=True)

        return False  # TODO: :-)

    @profiler.profile(label="SCBA: Phonon interactions", level="default", comm=comm)
    def _compute_phonon_interaction(self):
        """Computes the phonon interaction."""
        if self.config.phonon.model == "negf":
            raise NotImplementedError

        elif self.config.phonon.model == "pseudo-scattering":
            self.sigma_phonon.compute(
                self.data.g_lesser,
                self.data.g_greater,
                out=(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded_hermitian,
                ),
            )

    @profiler.profile(label="SCBA: Photon interactions", level="default", comm=comm)
    def _compute_photon_interaction(self):
        """Computes the photon interaction."""
        raise NotImplementedError

    @profiler.profile(label="SCBA: Electron interactions", level="default", comm=comm)
    def _compute_coulomb_screening_interaction(self):
        """Computes the Coulomb screening interaction."""

        self.data.p_greater.allocate_data()
        self.data.p_lesser.allocate_data()
        self.data.p_retarded_hermitian.allocate_data()

        self.p_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            out=(
                self.data.p_lesser,
                self.data.p_greater,
                self.data.p_retarded_hermitian,
            ),
        )

        self.data.w_greater.allocate_data()
        self.data.w_lesser.allocate_data()

        self.coulomb_screening_solver.solve(
            self.data.p_lesser,
            self.data.p_greater,
            self.data.p_retarded_hermitian,
            out=(self.data.w_lesser, self.data.w_greater),
        )

        self._compute_coulomb_screening_observables()

        self.data.p_lesser.free_data()
        self.data.p_greater.free_data()
        self.data.p_retarded_hermitian.free_data()

        self.sigma_fock.compute(
            self.data.g_lesser,
            out=(self.data.sigma_retarded_hermitian,),
        )

        self.sigma_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            self.data.w_lesser,
            self.data.w_greater,
            out=(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded_hermitian,
            ),
        )

        self.data.w_greater.free_data()
        self.data.w_lesser.free_data()

    @profiler.profile(label="SCBA: G observables", level="default", comm=comm)
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        if self.config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.electron_ldos *= 2  # Spin
        if self.config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.electron_density *= 2  # Spin
        if self.config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.hole_density *= 2  # Spin

        if self.config.outputs.device_currents:
            self.observables.electron_current["device"] = device_current(
                self.data.g_lesser, self.electron_solver.hamiltonian
            )
            if self.config.electron.solver.compute_current:

                local_current = self.electron_solver.meir_wingreen_current
                meir_wingreen_current = comm.stack.all_gather_v(
                    local_current,
                    axis=0,
                    mask=self.data.g_lesser._stack_padding_mask,
                )

                self.observables.electron_current["meir-wingreen"] = (
                    meir_wingreen_current
                )

        if self.config.outputs.self_energy_density:
            self.observables.sigma_lesser_density = density(
                self.data.sigma_lesser,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)
            self.observables.sigma_greater_density = -density(
                self.data.sigma_greater,
                self.electron_solver.overlap,
            ) / (2 * xp.pi)

    @profiler.profile(label="SCBA: W observables", level="default", comm=comm)
    def _compute_coulomb_screening_observables(self) -> None:

        # NOTE: The overlap is maybe missing here (it is not used)
        if self.config.outputs.polarization_density:
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )
            self.observables.p_greater_density = -density(self.data.p_greater) / (
                2 * xp.pi
            )

        if self.config.outputs.coulomb_screening_density:
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )
            self.observables.w_greater_density = -density(self.data.w_greater) / (
                2 * xp.pi
            )

    @profiler.profile(label="SCBA: Write outputs", level="default", comm=comm)
    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if comm.rank != 0:
            return

        outputs = {}

        if self.config.outputs.electron_ldos:
            outputs[f"electron_ldos_{iteration}.npy"] = self.observables.electron_ldos
        if self.config.outputs.electron_density:
            outputs[f"electron_density_{iteration}.npy"] = (
                self.observables.electron_density
            )
        if self.config.outputs.hole_density:
            outputs[f"hole_density_{iteration}.npy"] = self.observables.hole_density

        if self.config.outputs.device_currents:
            outputs[f"device_current_{iteration}.npy"] = (
                self.observables.electron_current["device"]
            )
            if self.config.electron.solver.compute_current:
                outputs[f"meir_wingreen_current_{iteration}.npy"] = (
                    self.observables.electron_current["meir-wingreen"]
                )

        if self.config.scba.coulomb_screening:
            if self.config.outputs.polarization_density:
                outputs.update(
                    {
                        f"p_lesser_density_{iteration}.npy": self.observables.p_lesser_density,
                        f"p_greater_density_{iteration}.npy": self.observables.p_greater_density,
                    }
                )
            if self.config.outputs.coulomb_screening_density:
                outputs.update(
                    {
                        f"w_lesser_density_{iteration}.npy": self.observables.w_lesser_density,
                        f"w_greater_density_{iteration}.npy": self.observables.w_greater_density,
                    }
                )

        if self.config.outputs.self_energy_density:
            outputs.update(
                {
                    f"sigma_lesser_density_{iteration}.npy": self.observables.sigma_lesser_density,
                    f"sigma_greater_density_{iteration}.npy": self.observables.sigma_greater_density,
                }
            )

        print(f"Writing output for iteration {iteration}...", flush=True)

        if not os.path.exists(self.config.output_dir):
            os.mkdir(self.config.output_dir)

        for filename, data in outputs.items():
            xp.save(self.config.output_dir / filename, data)

    def _compute_excess_charge_densities(self):
        """Computes the charge density from the local density of states.

        Returns
        -------
        excess_electron_density : NDArray
            The excess electron density computed from the local density
            of states.
        excess_hole_density : NDArray
            The excess hole density computed from the local density of
            states.

        """
        if (
            self.observables.electron_density is None
            or self.observables.hole_density is None
        ):
            raise ValueError(
                "Electron and hole densities must be computed "
                "before computing excess charge densities."
            )

        mid_gap_energy = (
            self.config.electron.conduction_band_edge
            + self.config.electron.valence_band_edge
        ) / 2
        mid_gap_energy = self.electron_solver.potential + mid_gap_energy

        electron_density = self.observables.electron_density.copy()
        hole_density = self.observables.hole_density.copy()

        mask = self.electron_energies[:, None] > mid_gap_energy
        electron_density[~mask] = 0
        hole_density[mask] = 0

        excess_electron_density = np.trapezoid(
            electron_density, self.electron_energies, axis=0
        )
        excess_hole_density = np.trapezoid(hole_density, self.electron_energies, axis=0)

        return excess_electron_density, excess_hole_density

    def set_potential(self, potential: NDArray):
        """Sets the potential for the SCBA calculation.

        Parameters
        ----------
        potential : NDArray
            The new potential values to be set in the system matrix.

        """
        if potential.shape[0] != np.sum(self.data.orbitals_per_atom):
            potential = np.repeat(potential, self.data.orbitals_per_atom)

        self.electron_solver.potential = potential

    def get_charge_density(self) -> NDArray:
        """Gets the charge density.

        This runs the SCBA to convergence (or to the maximum number of
        iterations) and then computes the charge density from the
        spectral electron and hole densities.

        Returns
        -------
        charge_density : NDArray
            The computed charge density for the device.

        """
        electron_density, hole_density = self._compute_excess_charge_densities()
        charge_density = electron_density - hole_density

        # From orbital to atom resolved charge density.
        orbital_offsets = np.hstack(([0], np.cumsum(self.data.orbitals_per_atom)))
        charge_density = np.add.reduceat(charge_density, orbital_offsets[:-1])

        return charge_density

    @profiler.profile(label="SCBA", level="default", comm=comm)
    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None

        for i in range(self.config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            with profiler.profile_range(
                label="SCBA: Iteration", level="default", comm=comm
            ):
                self.electron_solver.solve(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded_hermitian,
                    out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
                )
                self._compute_electron_observables()

                # Stash current into previous self-energy buffer.
                self._stash_sigma()

                with profiler.profile_range(
                    label="SCBA: stack->nnz transpose", level="default", comm=comm
                ):
                    # Transpose to nnz distribution.
                    # NOTE: While computing all interactions, we only ever need
                    # to access the Green's function and the self-energies in
                    # their nnz-distributed state.
                    for m in (self.data.g_lesser, self.data.g_greater):
                        m.dtranspose(discard=False)  # This must not be discarded.
                        assert m.distribution_state == "nnz"
                    for m in (
                        self.data.sigma_lesser,
                        self.data.sigma_greater,
                        self.data.sigma_retarded_hermitian,
                    ):
                        m.dtranspose(discard=True)  # These can be safely discarded.
                        assert m.distribution_state == "nnz"

                if self.config.scba.coulomb_screening:
                    self._compute_coulomb_screening_interaction()

                if self.config.scba.photon:
                    self._compute_photon_interaction()

                if self.config.scba.phonon:
                    self._compute_phonon_interaction()

                with profiler.profile_range(
                    label="SCBA: stack->nnz transpose back", level="default", comm=comm
                ):
                    for m in (self.data.g_lesser, self.data.g_greater):
                        m.dtranspose(discard=True)  # These can be safely discarded.
                        assert m.distribution_state == "stack"
                    for m in (
                        self.data.sigma_lesser,
                        self.data.sigma_greater,
                        self.data.sigma_retarded_hermitian,
                    ):
                        m.dtranspose(discard=False)  # This must not be discarded.
                        assert m.distribution_state == "stack"

            # Symmetrize the self-energy.
            self._symmetrize_sigma()

            if self._has_converged():
                if comm.rank == 0:
                    print(f"SCBA converged after {i} iterations.", flush=True)
                break

            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()

            if xp.__name__ == "cupy":
                free_memory, total_memory = xp.cuda.Device().mem_info
                usage = np.array((total_memory - free_memory) / total_memory)
                average_usage = np.empty(1)
                max_usage = np.empty(1)
                global_comm.Allreduce(usage, average_usage, op=MPI.SUM)
                global_comm.Allreduce(usage, max_usage, op=MPI.MAX)
                average_usage /= comm.size

                if comm.rank == 0:
                    print(
                        f"Rank-average device memory usage: {average_usage[0] * 100:.4f}%",
                        flush=True,
                    )
                    print(
                        f"Max device memory usage: {max_usage[0] * 100:.4f}%",
                        flush=True,
                    )

            if i % self.config.scba.output_interval == 0:
                self._write_iteration_outputs(i)

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                print(f"SCBA did not converge after {i} iterations.")
