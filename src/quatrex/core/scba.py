# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import contact_currents, density
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.coulomb_screening import (
    CoulombScreeningSolver,
    CoulombScreeningSolver_X,
    PCoulombScreening,
    PCoulombScreening_X,
)
from quatrex.electron import (
    ElectronSolver,
    ElectronSolver_X,
    SigmaCoulombScreening,
    SigmaCoulombScreening_X,
    SigmaFock,
    SigmaFock_X,
    SigmaPhonon,
    SigmaPhoton,
)
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton


def _compute_sparsity_pattern(
    positions: NDArray,
    cutoff_distance: float,
    strategy: str = "box",
) -> sparse.coo_matrix:
    """Computes the sparsity pattern for the interaction matrix.

    Parameters
    ----------
    grid : NDArray
        The grid points.
    interaction_cutoff : float
        The interaction cutoff.
    strategy : str, optional
        The strategy to use, by default "box", where only the distance
        along the transport direction is considered. The other option is
        "sphere", where the usual Euclidean distance between points
        matters.

    Returns
    -------
    sparse.coo_matrix
        The sparsity pattern.

    """
    if strategy == "sphere":

        def distance(x, y):
            """Euclidean distance."""
            return xp.linalg.norm(x - y, axis=-1)

    elif strategy == "box":

        def distance(x, y):
            """Distance along transport direction."""
            return xp.abs(x[..., 0] - y[..., 0])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows, cols = [], []
    for i, position in enumerate(positions):
        distances = distance(positions, position)
        interacting = xp.where(distances < cutoff_distance)[0]
        cols.extend(interacting)
        rows.extend([i] * len(interacting))

    rows, cols = xp.array(rows), xp.array(cols)
    return sparse.coo_matrix((xp.ones_like(rows, dtype=xp.float32), (rows, cols)))


class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
    ) -> None:
        """Initializes the SCBA data."""
        # Load orbital positions, energy vector and block-sizes.
        grid = distributed_load(quatrex_config.input_dir / "grid.npy")
        electron_energies = distributed_load(
            quatrex_config.input_dir / "electron_energies.npy"
        )
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")

        # Generate energy grid for _x quantity.
        if quatrex_config.electron.use_energy_x:
            # TODO: Don't hardcode this.
            overlap_energy = 1  # eV
            # TODO: There is probably a heuristic for finding better electron_energies_x.
            mid_fermi_energy = 0.5 * (
                quatrex_config.electron.left_fermi_level
                + quatrex_config.electron.right_fermi_level
            )
            energies_lesser = electron_energies[
                electron_energies < mid_fermi_energy + overlap_energy
            ]
            energies_greater = electron_energies[
                electron_energies > mid_fermi_energy - overlap_energy
            ]
            electron_energies_x = xp.concatenate([energies_lesser, energies_greater])

        # Find the maximum interaction cutoff.
        max_interaction_cutoff = 0.0
        if quatrex_config.scba.coulomb_screening:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                quatrex_config.coulomb_screening.interaction_cutoff,
            )
        if quatrex_config.scba.photon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                quatrex_config.photon.interaction_cutoff,
            )
        if quatrex_config.scba.phonon and quatrex_config.phonon.model == "negf":
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                quatrex_config.phonon.interaction_cutoff,
            )

        (
            print(f"Max Interaction Cutoff: {max_interaction_cutoff}", flush=True)
            if comm.rank == 0
            else None
        )
        self.sparsity_pattern = _compute_sparsity_pattern(grid, max_interaction_cutoff)
        # self.sparsity_pattern = distributed_load(
        #     quatrex_config.input_dir / "hamiltonian.npz"
        # ).astype(xp.complex128)

        dsbsparse_type = compute_config.dsbsparse_type

        if quatrex_config.electron.use_energy_x:
            self.g_retarded = dsbsparse_type.from_sparray(
                self.sparsity_pattern.astype(xp.complex128),
                block_sizes=block_sizes,
                global_stack_shape=electron_energies_x.shape,
                densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
            )
            self.g_retarded._data[:] = 0.0  # Initialize to zero.
            self.g_x = dsbsparse_type.zeros_like(self.g_retarded)

            self.sigma_retarded_prev = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_x_prev = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_retarded = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_x = dsbsparse_type.zeros_like(self.g_retarded)
        else:
            self.g_retarded = dsbsparse_type.from_sparray(
                self.sparsity_pattern.astype(xp.complex128),
                block_sizes=block_sizes,
                global_stack_shape=electron_energies.shape,
                densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
            )
            self.g_retarded._data[:] = 0.0  # Initialize to zero.
            self.g_lesser = dsbsparse_type.zeros_like(self.g_retarded)
            self.g_greater = dsbsparse_type.zeros_like(self.g_retarded)

            self.sigma_retarded_prev = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_lesser_prev = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_greater_prev = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_retarded = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_lesser = dsbsparse_type.zeros_like(self.g_retarded)
            self.sigma_greater = dsbsparse_type.zeros_like(self.g_retarded)

        if quatrex_config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction and densify those.
            self.p_retarded = dsbsparse_type.zeros_like(self.g_retarded)
            self.p_lesser = dsbsparse_type.zeros_like(self.g_retarded)
            if not quatrex_config.electron.use_energy_x:
                self.p_greater = dsbsparse_type.zeros_like(self.g_retarded)

            # TODO: Only multiples of three are supported for now.
            coulomb_screening_block_sizes = block_sizes[: len(block_sizes) // 3] * 3

            self.w_retarded = dsbsparse_type.from_sparray(
                self.sparsity_pattern.astype(xp.complex128),
                block_sizes=coulomb_screening_block_sizes,
                global_stack_shape=self.g_retarded.global_stack_shape,
                densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
            )
            self.w_retarded._data[:] = 0.0  # Initialize to zero.
            self.w_lesser = dsbsparse_type.zeros_like(self.w_retarded)
            if not quatrex_config.electron.use_energy_x:
                self.w_greater = dsbsparse_type.zeros_like(self.w_retarded)

        # TODO: The interactions with photons and phonons are not yet
        # implemented.
        if quatrex_config.scba.photon:
            raise NotImplementedError

        if quatrex_config.scba.phonon and quatrex_config.phonon.model == "negf":
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

    sigma_retarded_density: NDArray = None
    sigma_lesser_density: NDArray = None
    sigma_greater_density: NDArray = None

    # --- Coulomb screening --------------------------------------------
    w_retarded_density: NDArray = None
    w_lesser_density: NDArray = None
    w_greater_density: NDArray = None

    p_retarded_density: NDArray = None
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


class SCBA:
    """Self-consistent Born approximation (SCBA) solver.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    compute_config : Path, optional
        Compute configuration file, by default None. If None, the
        default compute parameters are used.

    """

    @time_range()
    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes an SCBA instance."""
        self.quatrex_config = quatrex_config

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        self.data = SCBAData(quatrex_config, compute_config)
        self.observables = Observables()

        # ----- Electrons ----------------------------------------------
        self.electron_energies = distributed_load(
            self.quatrex_config.input_dir / "electron_energies.npy"
        )
        if self.quatrex_config.electron.use_energy_x:
            # TODO: Don't hardcode this.
            overlap_energy = 1  # eV
            # TODO: There is probably a heuristic for finding better electron_energies_x.
            mid_fermi_energy = 0.5 * (
                quatrex_config.electron.left_fermi_level
                + quatrex_config.electron.right_fermi_level
            )
            energies_lesser = self.electron_energies[
                self.electron_energies < mid_fermi_energy + overlap_energy
            ]
            energies_greater = self.electron_energies[
                self.electron_energies > mid_fermi_energy - overlap_energy
            ]
            number_of_overlap_energies = sum(
                (self.electron_energies < mid_fermi_energy + overlap_energy)
                & (self.electron_energies > mid_fermi_energy - overlap_energy)
            )
            electron_energies_x = xp.concatenate([energies_lesser, energies_greater])
            self.electron_solver = ElectronSolver_X(
                self.quatrex_config,
                self.compute_config,
                self.electron_energies,
                energies_lesser,
                energies_greater,
                number_of_overlap_energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
        else:
            self.electron_solver = ElectronSolver(
                self.quatrex_config,
                self.compute_config,
                self.electron_energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )

        # ----- Coulomb screening --------------------------------------
        if self.quatrex_config.scba.coulomb_screening:
            energies_path = (
                self.quatrex_config.input_dir / "coulomb_screening_energies.npy"
            )
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = distributed_load(energies_path)
            else:
                self.coulomb_screening_energies = (
                    self.electron_energies - self.electron_energies[0]
                )
                # Remove the zero energy to avoid division by zero.
                self.coulomb_screening_energies += 1e-6

            # NOTE: No sparsity information required here.
            if self.quatrex_config.electron.use_energy_x:
                self.sigma_fock = SigmaFock_X(
                    self.quatrex_config,
                    self.compute_config,
                    len(energies_lesser),
                    self.electron_energies,
                    sparsity_pattern=self.data.sparsity_pattern,
                )
                self.p_coulomb_screening = PCoulombScreening_X(
                    self.quatrex_config,
                    energies_lesser,
                    number_of_overlap_energies,
                )
                coulomb_screening_energies_x = (
                    electron_energies_x - electron_energies_x[-1]
                )
                coulomb_screening_energies_x += 1e-6
                self.coulomb_screening_solver = CoulombScreeningSolver_X(
                    self.quatrex_config,
                    self.compute_config,
                    coulomb_screening_energies_x,
                    sparsity_pattern=self.data.sparsity_pattern,
                )
                self.sigma_coulomb_screening = SigmaCoulombScreening_X(
                    self.quatrex_config,
                    self.compute_config,
                    self.electron_energies,
                    electron_energies_x,
                    len(energies_lesser),
                    number_of_overlap_energies,
                    sparsity_pattern=self.data.sparsity_pattern,
                )
            else:
                self.sigma_fock = SigmaFock(
                    self.quatrex_config,
                    self.compute_config,
                    self.electron_energies,
                    sparsity_pattern=self.data.sparsity_pattern,
                )
                # NOTE: No sparsity information required here.
                self.p_coulomb_screening = PCoulombScreening(
                    self.quatrex_config,
                    self.coulomb_screening_energies,
                )
                self.coulomb_screening_solver = CoulombScreeningSolver(
                    self.quatrex_config,
                    self.compute_config,
                    self.coulomb_screening_energies,
                    sparsity_pattern=self.data.sparsity_pattern,
                )
                self.sigma_coulomb_screening = SigmaCoulombScreening(
                    self.quatrex_config,
                    self.compute_config,
                    self.electron_energies,
                    sparsity_pattern=self.data.sparsity_pattern,
                )

        # ----- Photons ------------------------------------------------
        if self.quatrex_config.scba.photon:
            energies_path = self.quatrex_config.input_dir / "photon_energies.npy"
            self.photon_energies = distributed_load(energies_path)
            self.pi_photon = PiPhoton(...)
            self.photon_solver = PhotonSolver(
                self.quatrex_config,
                self.compute_config,
                self.photon_energies,
                ...,
            )
            self.sigma_photon = SigmaPhoton(...)

        # ----- Phonons ------------------------------------------------
        if self.quatrex_config.scba.phonon:
            if self.quatrex_config.phonon.model == "negf":
                energies_path = self.quatrex_config.input_dir / "phonon_energies.npy"
                self.phonon_energies = distributed_load(energies_path)
                self.pi_phonon = PiPhonon(...)
                self.phonon_solver = PhononSolver(
                    self.quatrex_config,
                    self.compute_config,
                    self.phonon_energies,
                    ...,
                )
                self.sigma_phonon = SigmaPhonon(...)

            elif self.quatrex_config.phonon.model == "pseudo-scattering":
                self.sigma_phonon = SigmaPhonon(quatrex_config, self.electron_energies)

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_lesser_prev._data[:] = self.data.sigma_lesser._data
        self.data.sigma_greater_prev._data[:] = self.data.sigma_greater._data
        self.data.sigma_retarded_prev._data[:] = self.data.sigma_retarded._data

        self.data.sigma_retarded._data[:] = 0.0
        self.data.sigma_lesser._data[:] = 0.0
        self.data.sigma_greater._data[:] = 0.0

    def _stash_sigma_x(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_x_prev._data[:] = self.data.sigma_x._data
        self.data.sigma_retarded_prev._data[:] = self.data.sigma_retarded._data

        self.data.sigma_retarded._data[:] = 0.0
        self.data.sigma_x._data[:] = 0.0

    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""
        mixing_factor = self.quatrex_config.scba.mixing_factor
        self.data.sigma_lesser.data = (
            (1 - mixing_factor) * self.data.sigma_lesser_prev.data
            + mixing_factor * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data = (
            (1 - mixing_factor) * self.data.sigma_greater_prev.data
            + mixing_factor * self.data.sigma_greater.data
        )
        self.data.sigma_retarded.data = (
            (1 - mixing_factor) * self.data.sigma_retarded_prev.data
            + mixing_factor * self.data.sigma_retarded.data
        )

        # Symmetrization.
        self.data.sigma_lesser.data = 0.5 * (
            self.data.sigma_lesser.data
            - self.data.sigma_lesser.ltranspose(copy=True).data.conj()
        )
        self.data.sigma_greater.data = 0.5 * (
            self.data.sigma_greater.data
            - self.data.sigma_greater.ltranspose(copy=True).data.conj()
        )
        self.data.sigma_lesser._data.real = 0
        self.data.sigma_greater._data.real = 0

        self.data.sigma_retarded._data.imag = 0.0

        self.data.sigma_retarded.data += 0.5 * (
            self.data.sigma_greater.data - self.data.sigma_lesser.data
        )

    def _update_sigma_x(self) -> None:
        """Updates the self-energy with a mixing factor."""
        mixing_factor = self.quatrex_config.scba.mixing_factor
        self.data.sigma_x.data = (
            1 - mixing_factor
        ) * self.data.sigma_x_prev.data + mixing_factor * self.data.sigma_x.data
        self.data.sigma_retarded.data = (
            (1 - mixing_factor) * self.data.sigma_retarded_prev.data
            + mixing_factor * self.data.sigma_retarded.data
        )

        # Symmetrization.
        self.data.sigma_x.data = 0.5 * (
            self.data.sigma_x.data - self.data.sigma_x.ltranspose(copy=True).data.conj()
        )
        self.data.sigma_x._data.real = 0

    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        max_diff = xp.max(xp.abs(diff))
        max_diff = comm.allreduce(max_diff, op=MPI.MAX)
        (
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            if comm.rank == 0
            else None
        )
        # if max_diff < self.quatrex_config.scba.convergence_tol:
        #     return True
        # return False
        i_left, i_right = contact_currents(self.electron_solver)
        change_left = xp.linalg.norm(
            i_left.real - self.observables.electron_current.get("left", 0.0)
        )
        change_right = xp.linalg.norm(
            i_right.real - self.observables.electron_current.get("right", 0.0)
        )
        ave_change = 0.5 * (change_left + change_right)
        (
            print(f"Average Current Change: {ave_change}", flush=True)
            if comm.rank == 0
            else None
        )
        rel_change_left = change_left / xp.linalg.norm(i_left.real)
        rel_change_right = change_right / xp.linalg.norm(i_right.real)
        rel_ave_change = 0.5 * (rel_change_left + rel_change_right)
        (
            print(f"Relative Average Current Change: {rel_ave_change}", flush=True)
            if comm.rank == 0
            else None
        )

        diff = i_left.real.sum() + i_right.real.sum()
        print(f"Current Difference: {diff}", flush=True) if comm.rank == 0 else None

        rel_diff = diff / (i_left.real.sum() - i_right.real.sum())
        (
            print(f"Relative Current Conservation: {rel_diff}", flush=True)
            if comm.rank == 0
            else None
        )

        # if ave_change < self.quatrex_config.scba.convergence_tol:
        #     return True

        return False  # TODO: :-)

    def _compute_phonon_interaction(self):
        """Computes the phonon interaction."""
        if self.quatrex_config.phonon.model == "negf":
            raise NotImplementedError

        elif self.quatrex_config.phonon.model == "pseudo-scattering":
            self.sigma_phonon.compute(
                self.data.g_lesser,
                self.data.g_greater,
                out=(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                ),
            )

    def _compute_photon_interaction(self):
        """Computes the photon interaction."""
        raise NotImplementedError

    def _compute_coulomb_screening_interaction(self):
        """Computes the Coulomb screening interaction."""
        times = []
        times.append(time.perf_counter())
        if self.quatrex_config.electron.use_energy_x:
            self.p_coulomb_screening.compute(
                self.data.g_x,
                out=(self.data.p_lesser, self.data.p_retarded),
            )
        else:
            self.p_coulomb_screening.compute(
                self.data.g_lesser,
                self.data.g_greater,
                out=(self.data.p_lesser, self.data.p_greater, self.data.p_retarded),
            )
        t_polarization = time.perf_counter() - times.pop()
        (
            print(f"Time for polarization: {t_polarization:.2f} s", flush=True)
            if comm.rank == 0
            else None
        )
        times.append(time.perf_counter())
        if self.quatrex_config.electron.use_energy_x:
            self.coulomb_screening_solver.solve(
                self.data.p_lesser,
                # TODO: This is a hack
                self.data.p_lesser,
                self.data.p_retarded,
                out=(self.data.w_lesser, self.data.w_retarded),
            )
        else:
            self.coulomb_screening_solver.solve(
                self.data.p_lesser,
                self.data.p_greater,
                self.data.p_retarded,
                out=(self.data.w_lesser, self.data.w_greater, self.data.w_retarded),
            )
        t_solve = time.perf_counter() - times.pop()
        (
            print(f"Time for Coulomb screening solver: {t_solve:.2f} s", flush=True)
            if comm.rank == 0
            else None
        )
        times.append(time.perf_counter())
        if self.quatrex_config.electron.use_energy_x:
            self.sigma_fock.compute(
                self.data.g_x,
                out=(self.data.sigma_retarded,),
            )
        else:
            self.sigma_fock.compute(
                self.data.g_lesser,
                out=(self.data.sigma_retarded,),
            )
        t_fock = time.perf_counter() - times.pop()
        (
            print(f"Time for Fock self-energy: {t_fock:.2f} s", flush=True)
            if comm.rank == 0
            else None
        )
        times.append(time.perf_counter())
        if self.quatrex_config.electron.use_energy_x:
            self.sigma_coulomb_screening.compute(
                self.data.g_x,
                self.data.w_lesser,
                out=(self.data.sigma_x, self.data.sigma_retarded),
            )
        else:
            self.sigma_coulomb_screening.compute(
                self.data.g_lesser,
                self.data.g_greater,
                self.data.w_lesser,
                self.data.w_greater,
                out=(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                ),
            )
        t_sigma = time.perf_counter() - times.pop()
        (
            print(
                f"Time for Coulomb screening self-energy: {t_sigma:.2f} s", flush=True
            )
            if comm.rank == 0
            else None
        )

    def _compute_observables(self) -> None:
        """Computes observables."""
        self.observables.electron_ldos = -density(
            self.data.g_retarded,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.electron_density = density(
            self.data.g_lesser,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.hole_density = -density(
            self.data.g_greater,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)

        self.observables.electron_current = dict(
            zip(("left", "right"), contact_currents(self.electron_solver))
        )

        if self.quatrex_config.scba.coulomb_screening:
            self.observables.p_retarded_density = -density(self.data.p_retarded) / (
                2 * xp.pi
            )
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )
            self.observables.p_greater_density = -density(self.data.p_greater) / (
                2 * xp.pi
            )

            self.observables.w_retarded_density = -density(self.data.w_retarded) / (
                2 * xp.pi
            )
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )
            self.observables.w_greater_density = -density(self.data.w_greater) / (
                2 * xp.pi
            )

        self.observables.sigma_retarded_density = -density(
            self.data.sigma_retarded,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.sigma_lesser_density = density(
            self.data.sigma_lesser,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.sigma_greater_density = -density(
            self.data.sigma_greater,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)

    def _compute_observables_x(self) -> None:
        """Computes observables."""
        self.observables.electron_ldos = -density(
            self.data.g_retarded,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        # NOTE: This is not the electron density, but the x density (combination of lesser and greater).
        # TODO: Compute observables correctly.
        self.observables.electron_density = density(
            self.data.g_x,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)

        self.observables.electron_current = dict(
            zip(("left", "right"), contact_currents(self.electron_solver))
        )

        if self.quatrex_config.scba.coulomb_screening:
            self.observables.p_retarded_density = -density(self.data.p_retarded) / (
                2 * xp.pi
            )
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )

            self.observables.w_retarded_density = -density(self.data.w_retarded) / (
                2 * xp.pi
            )
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )

        self.observables.sigma_retarded_density = -density(
            self.data.sigma_retarded,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        # NOTE: This is not the lesser density, but the x density (combination of lesser and greater).
        # TODO: Compute observables correctly.
        self.observables.sigma_lesser_density = density(
            self.data.sigma_x,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)

    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if comm.rank != 0:
            return

        print(f"Writing output for iteration {iteration}...", flush=True)

        output_dir = self.quatrex_config.simulation_dir / "outputs"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        xp.save(
            f"{output_dir}/electron_ldos_{iteration}.npy",
            self.observables.electron_ldos,
        )
        xp.save(
            f"{output_dir}/electron_density_{iteration}.npy",
            self.observables.electron_density,
        )
        xp.save(
            f"{output_dir}/hole_density_{iteration}.npy", self.observables.hole_density
        )
        xp.save(
            f"{output_dir}/i_left_{iteration}.npy",
            self.observables.electron_current["left"],
        )
        xp.save(
            f"{output_dir}/i_right_{iteration}.npy",
            self.observables.electron_current["right"],
        )

        if self.quatrex_config.scba.coulomb_screening:
            xp.save(
                f"{output_dir}/p_lesser_density_{iteration}.npy",
                self.observables.p_lesser_density,
            )
            xp.save(
                f"{output_dir}/p_greater_density_{iteration}.npy",
                self.observables.p_greater_density,
            )
            xp.save(
                f"{output_dir}/p_retarded_density_{iteration}.npy",
                self.observables.p_retarded_density,
            )

            xp.save(
                f"{output_dir}/w_lesser_density_{iteration}.npy",
                self.observables.w_lesser_density,
            )
            xp.save(
                f"{output_dir}/w_greater_density_{iteration}.npy",
                self.observables.w_greater_density,
            )
            xp.save(
                f"{output_dir}/w_retarded_density_{iteration}.npy",
                self.observables.w_retarded_density,
            )

        xp.save(
            f"{output_dir}/sigma_retarded_density_{iteration}.npy",
            self.observables.sigma_retarded_density,
        )
        xp.save(
            f"{output_dir}/sigma_lesser_density_{iteration}.npy",
            self.observables.sigma_lesser_density,
        )
        xp.save(
            f"{output_dir}/sigma_greater_density_{iteration}.npy",
            self.observables.sigma_greater_density,
        )

    def _write_iteration_outputs_x(self, iteration: int):
        """Writes output for the current iteration on rank zero."""
        # TODO: This is a hack. Ideally one write method should be enough.

        if comm.rank != 0:
            return

        print(f"Writing output for iteration {iteration}...", flush=True)

        output_dir = self.quatrex_config.simulation_dir / "outputs"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        xp.save(
            f"{output_dir}/electron_ldos_{iteration}.npy",
            self.observables.electron_ldos,
        )
        xp.save(
            f"{output_dir}/x_density_{iteration}.npy",
            self.observables.electron_density,
        )
        xp.save(
            f"{output_dir}/i_left_{iteration}.npy",
            self.observables.electron_current["left"],
        )
        xp.save(
            f"{output_dir}/i_right_{iteration}.npy",
            self.observables.electron_current["right"],
        )

        if self.quatrex_config.scba.coulomb_screening:
            xp.save(
                f"{output_dir}/p_lesser_density_{iteration}.npy",
                self.observables.p_lesser_density,
            )
            xp.save(
                f"{output_dir}/p_retarded_density_{iteration}.npy",
                self.observables.p_retarded_density,
            )

            xp.save(
                f"{output_dir}/w_lesser_density_{iteration}.npy",
                self.observables.w_lesser_density,
            )
            xp.save(
                f"{output_dir}/w_retarded_density_{iteration}.npy",
                self.observables.w_retarded_density,
            )

        xp.save(
            f"{output_dir}/sigma_retarded_density_{iteration}.npy",
            self.observables.sigma_retarded_density,
        )
        xp.save(
            f"{output_dir}/sigma_x_density_{iteration}.npy",
            self.observables.sigma_lesser_density,
        )

    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None
        times = []
        for i in range(self.quatrex_config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None
            # append for iteration time
            times.append(time.perf_counter())

            times.append(time.perf_counter())
            if self.quatrex_config.electron.use_energy_x:
                self.electron_solver.solve(
                    self.data.sigma_x,
                    # TODO: This is a hack
                    self.data.sigma_x,
                    self.data.sigma_retarded,
                    out=(self.data.g_x, self.data.g_retarded),
                )
            else:
                self.electron_solver.solve(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                    out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
                )
            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            # Stash current into previous self-energy buffer.
            times.append(time.perf_counter())
            if self.quatrex_config.electron.use_energy_x:
                self._stash_sigma_x()
            else:
                self._stash_sigma()
            t_swap = time.perf_counter() - times.pop()
            (
                print(f"Time for swapping: {t_swap:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            if self.quatrex_config.scba.coulomb_screening:
                times.append(time.perf_counter())
                self._compute_coulomb_screening_interaction()
                t_coulomb = time.perf_counter() - times.pop()
                (
                    print(
                        f"Time for Coulomb screening interaction: {t_coulomb:.2f} s",
                        flush=True,
                    )
                    if comm.rank == 0
                    else None
                )

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.scba.phonon:
                times.append(time.perf_counter())
                self._compute_phonon_interaction()
                t_phonon = time.perf_counter() - times.pop()
                (
                    print(f"Time for phonon interaction: {t_phonon:.2f} s", flush=True)
                    if comm.rank == 0
                    else None
                )

            self.observables.electron_current = dict(
                zip(("left", "right"), contact_currents(self.electron_solver))
            )

            times.append(time.perf_counter())
            if self._has_converged():
                (
                    print(f"SCBA converged after {i} iterations.")
                    if comm.rank == 0
                    else None
                )
                self._compute_observables()
                break
            t_convergence = time.perf_counter() - times.pop()
            (
                print(f"Time for convergence check: {t_convergence:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            # Update self-energy for next iteration with mixing factor.
            times.append(time.perf_counter())
            if self.quatrex_config.electron.use_energy_x:
                self._update_sigma_x()
            else:
                self._update_sigma()
            t_update = time.perf_counter() - times.pop()
            (
                print(f"Time for updating: {t_update:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )
            if self.quatrex_config.electron.use_energy_x:
                self._compute_observables_x()
            else:
                self._compute_observables()

            if i % self.quatrex_config.scba.output_interval == 0:
                if self.quatrex_config.electron.use_energy_x:
                    self._write_iteration_outputs_x(i)
                else:
                    self._write_iteration_outputs(i)

            t_iteration = time.perf_counter() - times.pop()
            (
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

        else:  # Did not break, i.e. max_iterations reached.
            (
                print(f"SCBA did not converge after {i} iterations.")
                if comm.rank == 0
                else None
            )
            if self.quatrex_config.electron.use_energy_x:
                self._compute_observables_x()
            else:
                self._compute_observables()
