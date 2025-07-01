# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass

import numpy as np
from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.kernels.linalg import inv
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host, synchronize_device
from qttools.utils.input_utils import cutoff_hr, get_hamiltonian_block
from qttools.utils.mpi_utils import distributed_load, get_local_slice
from qttools.utils.stack_utils import scale_stack

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import density
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.utils import assemble_kpoint_dsb
from quatrex.coulomb_screening import PCoulombScreening
from quatrex.electron import SigmaCoulombScreening, SigmaFock, SigmaPhonon, SigmaPhoton
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton

profiler = Profiler()


class BSCData:
    """Data container class for the BSC.

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
        electron_energies: NDArray,
    ) -> None:
        """Initializes the BSC data."""
        # Load orbital positions, energy vector and block-sizes.
        if quatrex_config.device.construct_from_unit_cell:
            wannier_centers = distributed_load(
                quatrex_config.input_dir / "wannier_centers.npy"
            )
            # lattice_vectors = distributed_load(
            #    quatrex_config.input_dir / "lattice_vectors.npy"
            # )

        else:
            # Not supported yet.
            raise NotImplementedError(
                "Constructing the BSC from a non-unit cell is not supported yet."
            )
        number_of_kpoints = quatrex_config.electron.number_of_kpoints
        # We only use dense matrices for the BSC
        self.sparsity_pattern = sparse.csr_matrix(
            np.ones(
                (len(wannier_centers), len(wannier_centers)),
                dtype=np.complex128,
            )
        )

        block_sizes = xp.array([len(wannier_centers)])

        dsdbsparse_type = compute_config.dsdbsparse_type

        self.g_retarded = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in number_of_kpoints if k > 1]),
        )
        self.g_retarded.data[:] = 0.0  # Initialize to zero.
        self.g_system_matrix = dsdbsparse_type.zeros_like(self.g_retarded)

        self.g_lesser = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in number_of_kpoints if k > 1]),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )
        self.g_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_lesser_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_retarded_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_retarded = dsdbsparse_type.zeros_like(self.g_lesser)
        if quatrex_config.scba.symmetric:
            self.sigma_retarded.symmetry_op = lambda a: a
            self.sigma_retarded_prev.symmetry_op = lambda a: a

        if quatrex_config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_retarded = dsdbsparse_type.zeros_like(self.g_retarded)
            self.p_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
            self.p_greater = dsdbsparse_type.zeros_like(self.g_lesser)

            self.w_retarded = dsdbsparse_type.zeros_like(self.g_retarded)
            self.w_system_matrix = dsdbsparse_type.zeros_like(self.g_retarded)
            self.w_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
            self.w_greater = dsdbsparse_type.zeros_like(self.g_lesser)

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

    # --- Phonons ------------------------------------------------------
    pi_phonon_retarded_density: NDArray = None
    pi_phonon_lesser_density: NDArray = None
    pi_phonon_greater_density: NDArray = None
    d_phonon_retarded_density: NDArray = None
    d_phonon_lesser_density: NDArray = None
    d_phonon_greater_density: NDArray = None


class BSC:
    """Bandstructure calculation (sorta).

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

        self.observables = Observables()
        electron_energies = xp.zeros((comm.size,))
        self.data = BSCData(
            quatrex_config, compute_config, electron_energies=electron_energies
        )  # dummy data
        self.mixing_factor = self.quatrex_config.scba.mixing_factor

        # ----- Electrons ----------------------------------------------
        if (self.quatrex_config.electron.energy_window_max is not None) and (
            self.quatrex_config.electron.energy_window_min is not None
        ):
            if self.quatrex_config.electron.energy_window_num is not None:
                if self.quatrex_config.electron.energy_window_num_per_rank is not None:
                    raise ValueError(
                        "Should **exclusively** set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
                    )
                self.electron_energies = xp.linspace(
                    self.quatrex_config.electron.energy_window_min,
                    self.quatrex_config.electron.energy_window_max,
                    self.quatrex_config.electron.energy_window_num,
                )
            elif self.quatrex_config.electron.energy_window_num_per_rank is not None:
                energy_window_num = (
                    self.quatrex_config.electron.energy_window_num_per_rank
                    * comm.stack.size
                )
                self.electron_energies = xp.linspace(
                    self.quatrex_config.electron.energy_window_min,
                    self.quatrex_config.electron.energy_window_max,
                    energy_window_num,
                )
            else:
                raise ValueError(
                    "Should set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
                )
        else:
            energies_path = self.quatrex_config.input_dir / "electron_energies.npy"
            if os.path.isfile(energies_path):
                self.electron_energies = distributed_load(energies_path)
            else:
                if comm.rank == 0:
                    message = f"""
                                {'-'*40}
                                # WARNING
                                # since no information about electron energy grid is provided,
                                # we decide to take an energy range enough to cover all the bands 
                                # in the contact bandstructure. This can lead to unexpected memory usage.
                                {'-'*40}
                                """
                    print(message)
                self.electron_energies = self._determine_electron_energy_window(
                    quatrex_config, compute_config
                )
        self.local_electron_energies = get_local_slice(
            self.electron_energies, comm.stack
        )
        self.occupancies = fermi_dirac(
            self.local_electron_energies - quatrex_config.electron.fermi_level,
            quatrex_config.electron.temperature,
        )
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
            print(f"Resolution is {energy_resolution} eV.", flush=True)
            print(
                f"Each comm.block has {num_energies_per_rank} grid points.", flush=True
            )

        # ----- Read the Hamiltonian -----------------------------------
        hamiltonian_unit_cells = distributed_load(
            quatrex_config.input_dir / "hamiltonian_unit_cells.npy"
        ).astype(xp.complex128)

        # Apply the cutoff.
        if quatrex_config.device.R_cutoff is not None:
            hamiltonian_unit_cells = cutoff_hr(
                hamiltonian_unit_cells,
                R_cutoff=quatrex_config.device.R_cutoff,
            )
        hamiltonian_dict = {}
        # Create the Hamiltonian for each periodic shift.
        for periodic_shift in xp.ndindex(
            tuple(
                2 * ps - 1 for ps in quatrex_config.device.cells_in_periodic_directions
            )
        ):
            periodic_shift = tuple(
                [
                    ps - quatrex_config.device.cells_in_periodic_directions[i] + 1
                    for i, ps in enumerate(periodic_shift)
                ]
            )
            hamiltonian_block = get_hamiltonian_block(
                hamiltonian_unit_cells, (1, 1, 1), periodic_shift
            )
            hamiltonian_dict[periodic_shift] = sparse.csr_matrix(hamiltonian_block)

        self.hamiltonian = compute_config.dsdbsparse_type.from_sparray(
            self.data.sparsity_pattern,
            block_sizes=xp.array([hamiltonian_block.shape[0]]),
            global_stack_shape=(comm.stack.size,)
            + tuple(
                [k for k in self.quatrex_config.electron.number_of_kpoints if k > 1]
            ),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=xp.conj,
        )
        self.hamiltonian.data = 0.0
        number_of_kpoints = xp.array(
            [1 if k <= 1 else k for k in self.quatrex_config.electron.number_of_kpoints]
        )
        assemble_kpoint_dsb(
            self.hamiltonian,
            hamiltonian_dict,
            number_of_kpoints,
            0,
        )
        del hamiltonian_block
        del hamiltonian_dict

        # ----- Coulomb screening --------------------------------------
        if self.quatrex_config.scba.coulomb_screening:
            # Load the Coulomb matrix.
            coulomb_matrix_unit_cells = distributed_load(
                quatrex_config.input_dir / "coulomb_matrix_unit_cells.npy"
            ).astype(xp.complex128)
            # Apply the cutoff to the Coulomb matrix.
            if quatrex_config.device.R_cutoff is not None:
                coulomb_matrix_unit_cells = cutoff_hr(
                    coulomb_matrix_unit_cells,
                    R_cutoff=quatrex_config.device.R_cutoff,
                )
            coulomb_matrix_dict = {}
            for periodic_shift in xp.ndindex(
                tuple(
                    2 * ps - 1
                    for ps in quatrex_config.device.cells_in_periodic_directions
                )
            ):
                periodic_shift = tuple(
                    [
                        ps - quatrex_config.device.cells_in_periodic_directions[i] + 1
                        for i, ps in enumerate(periodic_shift)
                    ]
                )
                coulomb_matrix_block = get_hamiltonian_block(
                    coulomb_matrix_unit_cells,
                    (1, 1, 1),
                    periodic_shift,
                )
                coulomb_matrix_dict[periodic_shift] = sparse.csr_matrix(
                    coulomb_matrix_block
                )

            self.coulomb_matrix = compute_config.dsdbsparse_type.from_sparray(
                self.data.sparsity_pattern.astype(xp.complex128),
                block_sizes=xp.array([coulomb_matrix_block.shape[0]]),
                global_stack_shape=(comm.size,)
                + tuple(
                    [k for k in quatrex_config.electron.number_of_kpoints if k > 1]
                ),
                symmetry=quatrex_config.scba.symmetric,
                symmetry_op=xp.conj,
            )
            self.coulomb_matrix._data[:] = 0.0  # Initialize to zero.
            number_of_kpoints = xp.array(
                [
                    1 if k <= 1 else k
                    for k in self.quatrex_config.electron.number_of_kpoints
                ]
            )
            assemble_kpoint_dsb(
                self.coulomb_matrix,
                coulomb_matrix_dict,
                number_of_kpoints,
                -(number_of_kpoints // 2),
            )
            # Explicitely try to free the memory
            del coulomb_matrix_block
            del coulomb_matrix_dict

            # Make sure the Coulomb matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if not self.coulomb_matrix.symmetry:
                self.coulomb_matrix.symmetrize()
            self.coulomb_matrix._data /= (
                quatrex_config.coulomb_screening.epsilon_r
                * self.quatrex_config.coulomb_screening.num_adiabatic_steps
            )

            self.coulomb_screening_energies = (
                self.electron_energies - self.electron_energies[0]
            )
            # Remove the zero energy to avoid division by zero.
            self.coulomb_screening_energies += 1e-6

            (
                self.coulomb_matrix.dtranspose()
                if self.coulomb_matrix.distribution_state != "nnz"
                else None
            )
            self.sigma_fock = SigmaFock(
                self.quatrex_config,
                self.coulomb_matrix,
                self.electron_energies,
            )
            # Have to transpose the coulomb matrix back to the original distribution.
            (
                self.coulomb_matrix.dtranspose()
                if self.coulomb_matrix.distribution_state == "nnz"
                else None
            )

            # NOTE: No sparsity information required here.
            self.p_coulomb_screening = PCoulombScreening(
                self.quatrex_config,
                self.compute_config,
                self.coulomb_screening_energies,
            )
            self.sigma_coulomb_screening = SigmaCoulombScreening(
                self.quatrex_config,
                self.compute_config,
                self.electron_energies,
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

        self.data = BSCData(
            quatrex_config, compute_config, electron_energies=self.electron_energies
        )  # real data

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_lesser_prev.data[:] = self.data.sigma_lesser.data
        self.data.sigma_greater_prev.data[:] = self.data.sigma_greater.data
        self.data.sigma_retarded_prev.data[:] = self.data.sigma_retarded.data

        self.data.sigma_retarded.data[:] = 0.0
        self.data.sigma_lesser.data[:] = 0.0
        self.data.sigma_greater.data[:] = 0.0

    @profiler.profile(level="api")
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
        self.data.sigma_retarded.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_retarded_prev.data
            + self.mixing_factor * self.data.sigma_retarded.data
        )

        # Symmetrization.
        synchronize_device()
        time_start = time.perf_counter()
        if not self.quatrex_config.scba.symmetric:
            self.data.sigma_lesser.symmetrize(xp.subtract)
            self.data.sigma_greater.symmetrize(xp.subtract)
            # Make the self-energy Hermitian (removing the skew-Hermitian part).
            self.data.sigma_retarded.symmetrize(xp.add)

        # self.data.sigma_lesser._data.real = 0
        # self.data.sigma_greater._data.real = 0

        # Now add the imaginary, skew-Hermitian part back.
        self.data.sigma_retarded.data += 0.5 * (
            self.data.sigma_greater.data - self.data.sigma_lesser.data
        )
        synchronize_device()
        time_end = time.perf_counter()
        if comm.rank == 0:
            print(f"Symmetrization time: {time_end-time_start}", flush=True)

    @profiler.profile(level="api")
    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        local_max_diff = get_host(xp.max(xp.abs(diff)))
        max_diff = np.empty_like(local_max_diff)
        global_comm.Allreduce(local_max_diff, max_diff, op=MPI.MAX)

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)

        return False  # TODO: :-)

    @profiler.profile(level="api")
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

    @profiler.profile(level="api")
    def _compute_photon_interaction(self):
        """Computes the photon interaction."""
        raise NotImplementedError

    @profiler.profile(level="api")
    def _compute_coulomb_screening_interaction(self):
        """Computes the Coulomb screening interaction."""

        self.data.p_greater.allocate_data()
        self.data.p_lesser.allocate_data()
        self.data.p_retarded.allocate_data()

        t_polarization_start = time.perf_counter()
        self.p_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            out=(self.data.p_lesser, self.data.p_greater, self.data.p_retarded),
        )
        synchronize_device()
        t_polarization_end = time.perf_counter()
        comm.barrier()
        t_polarization_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for polarization: {t_polarization_end - t_polarization_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for polarization all: {t_polarization_end_all - t_polarization_start:.3f} s",
                flush=True,
            )

        self.data.w_greater.allocate_data()
        self.data.w_lesser.allocate_data()
        self.data.w_retarded.allocate_data()

        # Coulomb screening interaction.
        t_coulomb_start = time.perf_counter()
        self.data.w_system_matrix.data = 0
        self.data.w_system_matrix += sparse.eye(
            self.data.w_system_matrix.shape[-1],
            dtype=xp.complex128,
        )
        self.data.w_system_matrix.blocks[0, 0] -= (
            self.coulomb_matrix.blocks[0, 0] @ self.data.p_retarded.blocks[0, 0]
        )
        self.data.w_retarded.blocks[0, 0] = (
            inv(self.data.w_system_matrix.blocks[0, 0])
            @ self.coulomb_matrix.blocks[0, 0]
        )
        self.data.w_lesser.blocks[0, 0] = (
            self.data.w_retarded.blocks[0, 0]
            @ self.data.p_lesser.blocks[0, 0]
            @ self.data.w_retarded.blocks[0, 0].conj().swapaxes(-1, -2)
        )
        self.data.w_greater.blocks[0, 0] = (
            self.data.w_retarded.blocks[0, 0]
            @ self.data.p_greater.blocks[0, 0]
            @ self.data.w_retarded.blocks[0, 0].conj().swapaxes(-1, -2)
        )
        synchronize_device()
        t_coulomb_end = time.perf_counter()
        comm.barrier()
        t_coulomb_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Coulomb screening: {t_coulomb_end - t_coulomb_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Coulomb screening all: {t_coulomb_end_all - t_coulomb_start:.3f} s",
                flush=True,
            )

        t_coulomb_observables = time.perf_counter()
        self._compute_coulomb_screening_observables()
        synchronize_device()
        t_coulomb_observables_end = time.perf_counter()
        comm.barrier()
        t_coulomb_observables_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Coulomb screening observables: {t_coulomb_observables_end - t_coulomb_observables:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Coulomb screening observables all: {t_coulomb_observables_end_all - t_coulomb_observables:.3f} s",
                flush=True,
            )

        self.data.p_lesser.free_data()
        self.data.p_greater.free_data()
        self.data.p_retarded.free_data()

        t_sigma_fock_start = time.perf_counter()
        self.sigma_fock.compute(
            self.data.g_lesser,
            out=(self.data.sigma_retarded,),
        )
        synchronize_device()
        t_sigma_fock_end = time.perf_counter()
        comm.barrier()
        t_sigma_fock_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Fock self-energy: {t_sigma_fock_end - t_sigma_fock_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Fock self-energy all: {t_sigma_fock_end_all - t_sigma_fock_start:.3f} s",
                flush=True,
            )

        t_sigma_start = time.perf_counter()
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
        synchronize_device()
        t_sigma_end = time.perf_counter()
        comm.barrier()
        t_sigma_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Coulomb screening self-energy: {t_sigma_end - t_sigma_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Coulomb screening self-energy all: {t_sigma_end_all - t_sigma_start:.3f} s",
                flush=True,
            )

        self.data.w_retarded.free_data()
        self.data.w_greater.free_data()
        self.data.w_lesser.free_data()

    @profiler.profile(level="debug")
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        if self.quatrex_config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                # self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)
        if self.quatrex_config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                # self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)
        if self.quatrex_config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                # self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)

        if self.quatrex_config.outputs.self_energy_density:
            self.observables.sigma_retarded_density = -density(
                self.data.sigma_retarded,
                # self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)
            self.observables.sigma_lesser_density = density(
                self.data.sigma_lesser,
                # self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)
            self.observables.sigma_greater_density = -density(
                self.data.sigma_greater,
                # self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)

    @profiler.profile(level="debug")
    def _compute_coulomb_screening_observables(self) -> None:

        if self.quatrex_config.outputs.polarization_density:
            self.observables.p_retarded_density = -density(self.data.p_retarded) / (
                2 * xp.pi
            )
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )
            self.observables.p_greater_density = -density(self.data.p_greater) / (
                2 * xp.pi
            )

        if self.quatrex_config.outputs.coulomb_screening_density:
            self.observables.w_retarded_density = -density(self.data.w_retarded) / (
                2 * xp.pi
            )
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )
            self.observables.w_greater_density = -density(self.data.w_greater) / (
                2 * xp.pi
            )

    @profiler.profile(level="debug")
    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if self.quatrex_config.outputs.profiling_stats:
            profiler.dump_stats(self.quatrex_config.output_dir / "profiling_stats")

        if comm.rank != 0:
            return

        outputs = {}

        if self.quatrex_config.outputs.electron_ldos:
            outputs[f"electron_ldos_{iteration}.npy"] = self.observables.electron_ldos
        if self.quatrex_config.outputs.electron_density:
            outputs[f"electron_density_{iteration}.npy"] = (
                self.observables.electron_density
            )
        if self.quatrex_config.outputs.hole_density:
            outputs[f"hole_density_{iteration}.npy"] = self.observables.hole_density

        if self.quatrex_config.scba.coulomb_screening:
            if self.quatrex_config.outputs.polarization_density:
                outputs.update(
                    {
                        f"p_lesser_density_{iteration}.npy": self.observables.p_lesser_density,
                        f"p_greater_density_{iteration}.npy": self.observables.p_greater_density,
                        f"p_retarded_density_{iteration}.npy": self.observables.p_retarded_density,
                    }
                )
            if self.quatrex_config.outputs.coulomb_screening_density:
                outputs.update(
                    {
                        f"w_retarded_density_{iteration}.npy": self.observables.w_retarded_density,
                        f"w_lesser_density_{iteration}.npy": self.observables.w_lesser_density,
                        f"w_greater_density_{iteration}.npy": self.observables.w_greater_density,
                    }
                )

        if self.quatrex_config.outputs.self_energy_density:
            outputs.update(
                {
                    f"sigma_retarded_density_{iteration}.npy": self.observables.sigma_retarded_density,
                    f"sigma_lesser_density_{iteration}.npy": self.observables.sigma_lesser_density,
                    f"sigma_greater_density_{iteration}.npy": self.observables.sigma_greater_density,
                }
            )

        print(f"Writing output for iteration {iteration}...", flush=True)

        if not os.path.exists(self.quatrex_config.output_dir):
            os.mkdir(self.quatrex_config.output_dir)

        for filename, data in outputs.items():
            xp.save(self.quatrex_config.output_dir / filename, data)

    @profiler.profile(level="basic")
    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None

        for i in range(self.quatrex_config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None
            # append for iteration time
            synchronize_device()
            comm.barrier()
            t_iteration_start = time.perf_counter()

            t_solve_start = time.perf_counter()
            self.data.g_system_matrix.data = 0
            # Assumes orthonormal basis, so the overlap matrix is the identity.
            self.data.g_system_matrix += sparse.eye(
                self.data.g_system_matrix.shape[-1],
                dtype=xp.complex128,
            )
            scale_stack(
                self.data.g_system_matrix.data,
                self.local_electron_energies + 1j * self.quatrex_config.electron.eta,
            )
            self.data.g_system_matrix.blocks[0, 0] -= (
                self.hamiltonian.blocks[0, 0] + self.data.sigma_retarded.blocks[0, 0]
            )
            self.data.g_retarded.blocks[0, 0] = inv(
                self.data.g_system_matrix.blocks[0, 0]
            )
            spectral_function = self.data.g_retarded.blocks[
                0, 0
            ] - self.data.g_retarded.blocks[0, 0].conj().swapaxes(-1, -2)
            self.data.g_lesser.blocks[0, 0] = scale_stack(
                -spectral_function, self.occupancies
            )
            self.data.g_greater.blocks[0, 0] = scale_stack(
                spectral_function, 1 - self.occupancies
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for electron solver: {t_solve_end - t_solve_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for electron solver all: {t_solve_end_all - t_solve_start:.3f} s",
                    flush=True,
                )

            t_oberservables_start = time.perf_counter()
            self._compute_electron_observables()
            synchronize_device()
            t_oberservables_end = time.perf_counter()
            comm.barrier()

            t_oberservables_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for computing observables: {t_oberservables_end - t_oberservables_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for computing observables all: {t_oberservables_end_all - t_oberservables_start:.3f} s",
                    flush=True,
                )

            # Stash current into previous self-energy buffer.
            t_stash_start = time.perf_counter()
            self._stash_sigma()
            synchronize_device()
            t_stash_end = time.perf_counter()
            comm.barrier()
            t_stash_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for swapping: {t_stash_end - t_stash_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for swapping all: {t_stash_end_all - t_stash_start:.3f} s",
                    flush=True,
                )

            # Transpose to nnz distribution.
            # NOTE: While computing all interactions, we only ever need
            # to access the Green's function and the self-energies in
            # their nnz-distributed state.
            t_start_transpose = time.perf_counter()
            for m in (self.data.g_lesser, self.data.g_greater):
                m.dtranspose(discard=False)  # This must not be discarded.
                assert m.distribution_state == "nnz"
            for m in (
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
            ):
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "nnz"
            synchronize_device()
            t_end_transpose = time.perf_counter()
            comm.barrier()
            t_end_transpose_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"scba: Time for transposing forth: {t_end_transpose - t_start_transpose:.3f} s",
                    flush=True,
                )
                print(
                    f"scba: Time for transposing forth all: {t_end_transpose_all - t_start_transpose:.3f} s",
                    flush=True,
                )

            if self.quatrex_config.scba.coulomb_screening:
                if (
                    i >= 1
                    and i < self.quatrex_config.coulomb_screening.num_adiabatic_steps
                ):
                    self.coulomb_matrix.data *= (i + 1) / i
                    self.sigma_fock.coulomb_matrix_data *= (i + 1) / i

            if self.quatrex_config.scba.coulomb_screening:
                t_start_coulomb = time.perf_counter()
                self._compute_coulomb_screening_interaction()
                synchronize_device()
                t_end_coulomb = time.perf_counter()
                comm.barrier()
                t_end_coulomb_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"Time for Coulomb screening interaction: {t_end_coulomb - t_start_coulomb:.3f} s",
                        flush=True,
                    )
                    print(
                        f"Time for Coulomb screening interaction all: {t_end_coulomb_all - t_start_coulomb:.3f} s",
                        flush=True,
                    )

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.scba.phonon:
                t_start_phonon = time.perf_counter()
                self._compute_phonon_interaction()
                synchronize_device()
                t_end_phonon = time.perf_counter()
                comm.barrier()
                t_end_phonon_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"Time for phonon interaction: {t_end_phonon - t_start_phonon:.3f} s",
                        flush=True,
                    )
                    print(
                        f"Time for phonon interaction all: {t_end_phonon_all - t_start_phonon:.3f} s",
                        flush=True,
                    )

            # Transpose back to stack distribution.
            t_transpose_sigma_start = time.perf_counter()
            for m in (self.data.g_lesser, self.data.g_greater):
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "stack"
            for m in (
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
            ):
                m.dtranspose(discard=False)  # This must not be discarded.
                assert m.distribution_state == "stack"
            synchronize_device()
            t_transpose_sigma_end = time.perf_counter()
            comm.barrier()
            t_transpose_sigma_end_all = time.perf_counter()

            if comm.rank == 0:
                print(
                    f"scba: Time for transposing back: {t_transpose_sigma_end - t_transpose_sigma_start:.3f} s",
                    flush=True,
                )
                print(
                    f"scba: Time for transposing back all: {t_transpose_sigma_end_all - t_transpose_sigma_start:.3f} s",
                    flush=True,
                )

            t_convergence_start = time.perf_counter()
            if self._has_converged():
                if comm.rank == 0:
                    print(f"BSC converged after {i} iterations.", flush=True)

                break
            synchronize_device()
            t_convergence_end = time.perf_counter()
            comm.barrier()
            t_convergence_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for convergence check: {t_convergence_end - t_convergence_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for convergence check all: {t_convergence_end_all - t_convergence_start:.3f} s",
                    flush=True,
                )

            # Update self-energy for next iteration with mixing factor.
            t_sigma_update_start = time.perf_counter()
            self._update_sigma()
            synchronize_device()
            t_sigma_update_end = time.perf_counter()
            comm.barrier()
            t_sigma_update_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for updating: {t_sigma_update_end - t_sigma_update_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for updating all: {t_sigma_update_end_all - t_sigma_update_start:.3f} s",
                    flush=True,
                )

            t_iteration = time.perf_counter() - t_iteration_start
            if comm.rank == 0:
                print(f"Time for iteration all: {t_iteration:.3f} s", flush=True)

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

            if i % self.quatrex_config.scba.output_interval == 0:
                synchronize_device()
                comm.barrier()
                t_write_start = time.perf_counter()
                self._write_iteration_outputs(i)
                synchronize_device()
                t_write_end = time.perf_counter()
                comm.barrier()
                t_write_end_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"Time for writing outputs: {t_write_end_all - t_write_start:.3f} s",
                        flush=True,
                    )
                    print(
                        f"Time for writing outputs all: {t_write_end - t_write_start:.3f} s",
                        flush=True,
                    )

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                print(f"SCBA did not converge after {i} iterations.")
