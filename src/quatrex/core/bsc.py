# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass

import numpy as np
from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm
from scipy.signal import find_peaks

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.routines import bd_matmul, bd_sandwich
from qttools.greens_function_solver import RGF
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host, synchronize_device
from qttools.utils.mpi_utils import distributed_load, get_local_slice
from qttools.utils.stack_utils import scale_stack
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import density
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import bose_einstein, fermi_dirac
from quatrex.coulomb_screening import PCoulombScreening
from quatrex.device.inputs import (
    _assemble_kpoint,
    _get_transport_block,
    trim_tight_binding_matrix,
)
from quatrex.electron import (
    SigmaCoulombScreening,
    SigmaFock,
    SigmaHartree,
    SigmaPhonon,
    SigmaPhoton,
)
from quatrex.grid import get_electron_energies
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton

profiler = Profiler()


def _spectral_function(
    retarded: DSDBSparse, out: DSDBSparse | None = None
) -> DSDBSparse:
    """Computes the spectral function `A-A.dagger` from the retarded Green's function.

    Parameters
    ----------
    retarded : DSDBSparse
        The retarded Green's function.
    out : DSDBSparse, optional
        The output matrix to store the result. If None, a new matrix is created.

    Returns
    -------
    DSDBSparse
        The spectral function.

    """
    return_out = False
    if out is None:
        return_out = True
        out = retarded.zeros_like(retarded)
    retarded_ = retarded.stack[...]
    for i in range(retarded.num_local_blocks):
        out.blocks[i, i] = retarded_.blocks[i, i] - retarded_.blocks[
            i, i
        ].conj().swapaxes(-1, -2)

        j = i + 1
        if j >= retarded.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        out.blocks[i, j] = retarded_.blocks[i, j] - retarded_.blocks[
            j, i
        ].conj().swapaxes(-1, -2)
        out.blocks[j, i] = out.blocks[i, j].conj().swapaxes(-1, -2)

    if return_out:
        return out


def _load_matrix(
    quatrex_config: QuatrexConfig, compute_config: ComputeConfig, matrix_name: str
) -> dict[tuple[int, ...], NDArray]:
    """
    Load a tight-binding matrix from file, and creates a dictionary of matrices
    from the unit cells (mainly a hack to use the _assemble_kpoint function), and
    assemble the k-point matrix.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    matrix_name : str
        The name of the matrix to load. Should be one of "hamiltonian", "overlap_matrix", or "coulomb_matrix".

    Returns
    -------
    matrix : DSDBSparse
        The loaded matrix.
    """
    unit_cells = distributed_load(
        quatrex_config.input_dir / f"{matrix_name}_unit_cells.npy"
    ).astype(xp.complex128)

    trimmed_unit_cells = trim_tight_binding_matrix(
        tight_binding_matrix=unit_cells,
        neighbor_cell_cutoff=quatrex_config.device.neighbor_cell_cutoff,
    )

    transverse_repetitions = trimmed_unit_cells.shape[:3]
    matrix_dict = {}
    # Create a matrix for each connecting layer along the transverse
    # directions. The number of periodic cells is determined by the
    # shape of the unit cell data.
    for periodic_shift in xp.ndindex(transverse_repetitions):
        # Center the periodic shift around zero.
        periodic_shift = tuple(
            [ps - (us // 2) for ps, us in zip(periodic_shift, transverse_repetitions)]
        )
        matrix_block = _get_transport_block(
            trimmed_unit_cells,
            (1, 1, 1),
            periodic_shift,
        )
        matrix_dict[periodic_shift] = matrix_block

    sparsity_pattern = sparse.csr_matrix(
        xp.ones(
            (trimmed_unit_cells.shape[-2], trimmed_unit_cells.shape[-1]),
            dtype=xp.complex128,
        )
    )
    matrix = compute_config.dsdbsparse_type.from_sparray(
        sparsity_pattern,
        block_sizes=np.array([sparsity_pattern.shape[0]]),
        global_stack_shape=(comm.size,)
        + tuple([k for k in quatrex_config.electron.num_kpoints if k > 1]),
        symmetry=quatrex_config.bsc.symmetric,
        symmetry_op=xp.conj,
    )
    matrix._data[:] = 0.0  # Initialize to zero.
    num_kpoints = xp.array(quatrex_config.electron.num_kpoints)
    if matrix_name == "coulomb_matrix":
        kshift = -num_kpoints // 2
    elif matrix_name == "hamiltonian" or matrix_name == "overlap_matrix":
        kshift = 0
    _assemble_kpoint(
        matrix,
        matrix_dict,
        num_kpoints,
        kshift=kshift,
    )
    return matrix


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
        a_.blocks[i, i] -= b_.blocks[i, i]

        j = i + 1
        if j >= a.num_local_blocks and comm.block.rank == comm.block.size - 1:
            # The last rank does not have these blocks.
            continue

        a_.blocks[i, j] -= b_.blocks[i, j]
        a_.blocks[j, i] -= b_.blocks[j, i]


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
        wannier_centers = distributed_load(
            quatrex_config.input_dir / "wannier_centers.npy"
        )
        # TODO: Should maybe not be loaded here, but in the BSC class (needed for charge density calculation).
        self.lattice_vectors = distributed_load(
            quatrex_config.input_dir / "lattice_vectors.npy"
        )

        num_kpoints = quatrex_config.electron.num_kpoints
        # We only use dense matrices for the BSC.
        # Sure, the Hamiltonian can be sparse, but the sparsity pattern is not
        # the usual block-tridiagonal one, there are also corner blocks that have
        # to be taken into account. If the Hamiltonian is very sparse, it is better
        # to use open boundary conditions (transport calculations with flat bands).
        # Note that then no bandstructure is obtained, but for large matrices only
        # the gamma point is usually needed.
        self.sparsity_pattern = sparse.csr_matrix(
            xp.ones(
                (len(wannier_centers), len(wannier_centers)),
                dtype=np.complex128,
            )
        )

        block_sizes = np.array([len(wannier_centers)])

        dsdbsparse_type = compute_config.dsdbsparse_type

        self.g_retarded = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in num_kpoints if k > 1]),
        )
        self.g_retarded.data[:] = 0.0  # Initialize to zero.
        self.g_system_matrix = dsdbsparse_type.zeros_like(self.g_retarded)

        self.g_lesser = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in num_kpoints if k > 1]),
            symmetry=quatrex_config.bsc.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )
        self.g_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_retarded_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_retarded = dsdbsparse_type.zeros_like(self.g_lesser)
        if quatrex_config.bsc.symmetric:
            self.sigma_retarded.symmetry_op = lambda a: a
            self.sigma_retarded_prev.symmetry_op = lambda a: a

        if quatrex_config.bsc.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_retarded = dsdbsparse_type.zeros_like(self.g_retarded)
            self.p_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
            self.p_greater = dsdbsparse_type.zeros_like(self.g_lesser)

            self.dielectric_inverse = dsdbsparse_type.zeros_like(self.g_retarded)
            self.w_retarded = dsdbsparse_type.zeros_like(self.g_retarded)
            self.w_system_matrix = dsdbsparse_type.zeros_like(self.g_retarded)
            self.w_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
            self.w_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        # TODO: The interactions with photons and phonons are not yet
        # implemented.
        if quatrex_config.bsc.photon:
            raise NotImplementedError

        if quatrex_config.bsc.phonon and quatrex_config.phonon.model == "negf":
            raise NotImplementedError


@dataclass
class Observables:
    """Observable quantities for the BSC."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None

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
        """Initializes an BSC instance."""
        self.quatrex_config = quatrex_config

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        self.observables = Observables()

        self.solver = RGF(
            max_batch_size=quatrex_config.electron.solver.max_batch_size,
        )

        self.conduction_band_edges = self.quatrex_config.electron.conduction_band_edge
        self.valence_band_edges = self.quatrex_config.electron.valence_band_edge
        self.fermi_level = self.quatrex_config.electron.fermi_level
        self.delta_conduction_band_edge = (
            self.quatrex_config.electron.conduction_band_edge
            - self.quatrex_config.electron.fermi_level
        )
        electron_energies = xp.zeros((comm.size,))
        self.data = BSCData(
            quatrex_config, compute_config, electron_energies=electron_energies
        )  # dummy data
        self.mixing_factor = self.quatrex_config.bsc.mixing_factor

        # ----- Electrons ----------------------------------------------
        self.electron_energies = get_electron_energies(quatrex_config)

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

        self.local_electron_energies = get_local_slice(
            self.electron_energies, comm.stack
        )
        self.occupancies = fermi_dirac(
            self.local_electron_energies - self.fermi_level,
            quatrex_config.electron.temperature,
        )

        # ----- Load the Hamiltonian -----------------------------------
        self.hamiltonian = _load_matrix(
            quatrex_config,
            compute_config,
            "hamiltonian",
        )

        # Create the overlap matrix.
        try:
            # Load the overlap matrix from file, if it exists...
            self.overlap = _load_matrix(
                quatrex_config,
                compute_config,
                "overlap_matrix",
            )
            self.orthogonal_basis = False
        except FileNotFoundError:
            # ... if it does not exist, assume orthogonal basis.
            self.overlap = None
            self.orthogonal_basis = True

        # ----- Coulomb screening --------------------------------------
        if self.quatrex_config.bsc.coulomb_screening or self.quatrex_config.bsc.hartree:
            # Load the Coulomb matrix.
            self.coulomb_matrix = _load_matrix(
                quatrex_config,
                compute_config,
                "coulomb_matrix",
            )

            # Make sure the Coulomb matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if not self.coulomb_matrix.symmetry:
                self.coulomb_matrix.symmetrize()
            self.coulomb_matrix._data /= quatrex_config.coulomb_screening.epsilon_r

        if self.quatrex_config.bsc.hartree:
            self.sigma_hartree = SigmaHartree(
                self.quatrex_config,
                self.coulomb_matrix,
                self.electron_energies,
                self.data.lattice_vectors,
            )

        if self.quatrex_config.bsc.coulomb_screening:
            self.coulomb_screening_energies = (
                self.electron_energies - self.electron_energies[0]
            )
            local_coulomb_screening_energies = get_local_slice(
                self.coulomb_screening_energies, comm.stack
            )
            self.occupancies_coulomb_screening = bose_einstein(
                local_coulomb_screening_energies,
                quatrex_config.coulomb_screening.temperature,
            )

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
        if self.quatrex_config.bsc.photon:
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
        if self.quatrex_config.bsc.phonon:
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

    def _assemble_greens_function_system_matrix(self, sse_retarded: DSDBSparse) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_retarded : DSDBSparse
            The retarded scattering self-energy.

        """
        self.data.g_system_matrix.data = 0.0
        if self.orthogonal_basis:
            self.data.g_system_matrix.fill_diagonal(1.0)
        else:
            self.data.g_system_matrix += self.overlap
        scale_stack(
            self.data.g_system_matrix.data,
            self.local_electron_energies + 1j * self.quatrex_config.electron.eta,
        )
        # self.data.g_system_matrix -= sparse.diags(self.potential, format="csr")
        _btd_subtract(self.data.g_system_matrix, self.hamiltonian)
        _btd_subtract(self.data.g_system_matrix, sse_retarded)

    def _assemble_screened_interaction_system_matrix(
        self, p_retarded: DSDBSparse
    ) -> None:
        """Assembles the system matrix."""
        self.data.w_system_matrix.data = 0.0

        bd_matmul(
            self.coulomb_matrix,
            p_retarded,
            out=self.data.w_system_matrix,
        )
        xp.negative(self.data.w_system_matrix.data, out=self.data.w_system_matrix.data)
        if self.orthogonal_basis:
            self.data.w_system_matrix += sparse.eye(self.data.w_system_matrix.shape[-1])
        else:
            # I believe it should be the overlap matrix here
            self.data.w_system_matrix += self.overlap

    def _find_band_edges(self) -> None:
        """Find the band edges (conduction band minima and valence band maxima)."""
        # Find peaks in the electron density.
        dos = xp.sum(
            -density(
                self.data.g_retarded,
                self.overlap,
            )
            / (2 * xp.pi),
            axis=-1,
        )
        energies = self.electron_energies
        kpoints = dos.shape[1:]
        mid_bandgap = (self.conduction_band_edges + self.valence_band_edges) / 2
        conduction_band_edge = np.zeros(kpoints)
        valence_band_edge = np.zeros(kpoints)
        for kp in np.ndindex(kpoints):
            # NOTE: Find peaks don't work with cupy arrays, so we have to use numpy.
            peaks, _ = find_peaks(get_host(dos[:, *kp]), height=0.01)
            bands = energies[peaks]
            # Find the conduction and valence band edges.
            conduction_band_edge[kp] = xp.min(bands[bands > mid_bandgap])
            valence_band_edge[kp] = xp.max(bands[bands < mid_bandgap])
        self.conduction_band_edges = xp.min(conduction_band_edge)
        self.valence_band_edges = xp.max(valence_band_edge)
        if comm.rank == 0:
            print(f"Mid bandgap: {mid_bandgap}", flush=True)
            print(
                f"Conduction Band Edge: {self.conduction_band_edges}, k-point: {np.argmin(conduction_band_edge)}",
                flush=True,
            )
            print(
                f"Valence Band Edge: {self.valence_band_edges}, k-point: {np.argmax(valence_band_edge)}",
                flush=True,
            )

    def _update_fermi_level(self) -> None:
        """Update the Fermi level based on the current band edges or charge neutrality."""
        # TODO: This is a naive implementation, should be self-consistent such that
        # total charge is conserved.
        # Should the potential also be updated?
        if self.quatrex_config.electron.fermi_level_mode == "track_band_edge":
            # NOTE: This option is not really converging well (should be dropped?).
            self.fermi_level = (
                self.conduction_band_edges - self.delta_conduction_band_edge
            )
            self.occupancies = fermi_dirac(
                self.local_electron_energies - self.fermi_level,
                self.quatrex_config.electron.temperature,
            )
        # TODO: Should clean this up a bit.
        elif self.quatrex_config.electron.fermi_level_mode == "charge_neutrality":
            from scipy.optimize import bisect

            doping = self.quatrex_config.electron.doping
            dos_density = self._compute_dos_density()
            mid_bandgap = (self.conduction_band_edges + self.valence_band_edges) / 2
            equilibrium_occupancies = fermi_dirac(
                self.electron_energies - mid_bandgap,
                self.quatrex_config.electron.temperature,
            )

            def func(fermi_level):
                occupancies = fermi_dirac(
                    self.electron_energies - fermi_level,
                    self.quatrex_config.electron.temperature,
                )
                charge_density = xp.sum(
                    dos_density * (occupancies - equilibrium_occupancies)
                )
                return charge_density - doping

            self.fermi_level = bisect(
                func, self.electron_energies[0], self.electron_energies[-1]
            )
            self.occupancies = fermi_dirac(
                self.local_electron_energies - self.fermi_level,
                self.quatrex_config.electron.temperature,
            )

    def _compute_dos_density(self) -> float:
        """Compute the dos density (stupid name)."""
        ldos = -density(
            self.data.g_retarded,
            self.overlap,
        ) / (xp.pi)
        # Sum all axis except the first one (which is the energy axis).
        dos_density = xp.sum(ldos, axis=tuple(i for i in range(1, ldos.ndim)))
        # Normalize the dos density to get it in 1/cm^2.
        # TODO: This is hard coded for 2D materials.
        de = self.electron_energies[1] - self.electron_energies[0]
        nk = np.prod(self.quatrex_config.electron.num_kpoints)
        uc_area = np.linalg.det(self.data.lattice_vectors[:2, :2])
        dos_density *= de / (nk * uc_area) * 1e16  # in 1/cm^2
        return dos_density

    def _compute_total_charge(self) -> float:
        """Compute the total charge in the conduction band."""
        ldos = -density(
            self.data.g_retarded,
            self.overlap,
        ) / (2 * xp.pi)
        # Sum all axis except the first one (which is the energy axis).
        dos = xp.sum(ldos, axis=tuple(i for i in range(1, ldos.ndim)))
        # Gather the occupancies across all ranks.
        occupancies = comm.stack.all_gather_v(
            self.occupancies,
            axis=0,
        )
        mid_bandgap = (self.conduction_band_edges + self.valence_band_edges) / 2
        equilibrium_occupancies = fermi_dirac(
            self.electron_energies - mid_bandgap,
            self.quatrex_config.electron.temperature,
        )
        # Compute the total charge in the system.
        total_charge = xp.sum(dos * (occupancies - equilibrium_occupancies))
        de = self.electron_energies[1] - self.electron_energies[0]
        total_charge *= de
        return total_charge

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_retarded_prev.data[:] = self.data.sigma_retarded.data

        self.data.sigma_retarded.data[:] = 0.0
        self.data.sigma_lesser.data[:] = 0.0
        self.data.sigma_greater.data[:] = 0.0

    @profiler.profile(level="api")
    def _symmetrize_sigma(self) -> None:
        """Symmetrize the self-energy."""

        if not self.quatrex_config.bsc.symmetric:
            self.data.sigma_lesser.symmetrize(xp.subtract)
            self.data.sigma_greater.symmetrize(xp.subtract)
            # Make the self-energy Hermitian (removing the skew-Hermitian part).
            self.data.sigma_retarded.symmetrize(xp.add)

        if self.quatrex_config.coulomb_screening.discard_real_parts:
            self.data.sigma_lesser._data.real = 0
            self.data.sigma_greater._data.real = 0
            # Make sure that the imaginary part comes only from
            # sigma_greater - sigma_lesser.
            self.data.sigma_retarded._data.imag = 0

        # Now add the imaginary, skew-Hermitian part back.
        if self.quatrex_config.electron.use_sigma_ah:
            self.data.sigma_retarded.data += 0.5 * (
                self.data.sigma_greater.data - self.data.sigma_lesser.data
            )

    @profiler.profile(level="api")
    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""

        self.data.sigma_retarded.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_retarded_prev.data
            + self.mixing_factor * self.data.sigma_retarded.data
        )

    @profiler.profile(level="api")
    def _has_converged(self) -> bool:
        """Checks if the BSC has converged."""
        # Infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        local_max_diff = get_host(xp.max(xp.abs(diff)))
        max_diff = np.empty_like(local_max_diff)
        global_comm.Allreduce(local_max_diff, max_diff, op=MPI.MAX)

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)

        return False  # TODO: :-)

    def _compute_hartree_interaction(self):
        """Computes the Hartree interaction."""
        t_sigma_hartree_start = time.perf_counter()
        intrinsic_occupancies = fermi_dirac(
            self.local_electron_energies
            - (self.conduction_band_edges + self.valence_band_edges) / 2,
            self.quatrex_config.electron.temperature,
        )
        self.sigma_hartree.compute(
            _spectral_function(self.data.g_retarded),
            self.occupancies,
            intrinsic_occupancies,
            out=(self.data.sigma_retarded,),
        )
        synchronize_device()
        t_sigma_hartree_end = time.perf_counter()
        comm.barrier()
        t_sigma_hartree_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Hartree self-energy: {t_sigma_hartree_end - t_sigma_hartree_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Hartree self-energy all: {t_sigma_hartree_end_all - t_sigma_hartree_start:.3f} s",
                flush=True,
            )

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

        self.data.dielectric_inverse.allocate_data()
        self.data.w_retarded.allocate_data()
        self.data.w_greater.allocate_data()
        self.data.w_lesser.allocate_data()

        # Coulomb screening interaction.

        t_coulomb_start = time.perf_counter()
        self._assemble_screened_interaction_system_matrix(
            self.data.p_retarded,
        )

        self.solver.selected_inv(
            self.data.w_system_matrix,
            out=self.data.dielectric_inverse,
        )
        bd_matmul(
            self.data.dielectric_inverse,
            self.coulomb_matrix,
            out=self.data.w_retarded,
        )

        # Omega = self.data.w_retarded.blocks[0, 0] - self.data.w_retarded.blocks[0, 0].conj().swapaxes(-1, -2)
        # self.data.w_lesser.blocks[0, 0] = scale_stack(
        #   Omega.copy(), self.occupancies_coulomb_screening
        # )
        # self.data.w_greater.blocks[0, 0] = scale_stack(
        #   Omega.copy(), (1 + self.occupancies_coulomb_screening)
        # )

        bd_sandwich(
            self.data.w_retarded,
            self.data.p_lesser,
            out=self.data.w_lesser,
            symmetric=False,
        )
        bd_sandwich(
            self.data.w_retarded,
            self.data.p_greater,
            out=self.data.w_greater,
            symmetric=False,
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

        self.data.dielectric_inverse.free_data()
        self.data.w_retarded.free_data()
        self.data.w_greater.free_data()
        self.data.w_lesser.free_data()

    @profiler.profile(level="debug")
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        if self.quatrex_config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                self.overlap,
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.electron_ldos = self.observables.electron_ldos.sum(
                    axis=-1
                )
        if self.quatrex_config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                self.overlap,
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.electron_density = (
                    self.observables.electron_density.sum(axis=-1)
                )
        if self.quatrex_config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                self.overlap,
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.hole_density = self.observables.hole_density.sum(
                    axis=-1
                )
        if self.quatrex_config.outputs.self_energy_density:
            self.observables.sigma_retarded_density = -density(
                self.data.sigma_retarded,
                self.overlap,
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.sigma_retarded_density = (
                    self.observables.sigma_retarded_density.sum(axis=-1)
                )
            self.observables.sigma_lesser_density = density(
                self.data.sigma_lesser,
                self.overlap,
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.sigma_lesser_density = (
                    self.observables.sigma_lesser_density.sum(axis=-1)
                )
            self.observables.sigma_greater_density = -density(
                self.data.sigma_greater,
                self.overlap,
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.sigma_greater_density = (
                    self.observables.sigma_greater_density.sum(axis=-1)
                )

    @profiler.profile(level="debug")
    def _compute_coulomb_screening_observables(self) -> None:

        if self.quatrex_config.outputs.polarization_density:
            self.observables.p_retarded_density = -density(
                self.data.p_retarded, self.overlap
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.p_retarded_density = (
                    self.observables.p_retarded_density.sum(axis=-1)
                )
            self.observables.p_lesser_density = -density(
                self.data.p_lesser, self.overlap
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.p_lesser_density = (
                    self.observables.p_lesser_density.sum(axis=-1)
                )
            self.observables.p_greater_density = -density(
                self.data.p_greater, self.overlap
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.p_greater_density = (
                    self.observables.p_greater_density.sum(axis=-1)
                )

        if self.quatrex_config.outputs.coulomb_screening_density:
            self.observables.w_retarded_density = -density(
                self.data.w_retarded, self.overlap
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.w_retarded_density = (
                    self.observables.w_retarded_density.sum(axis=-1)
                )
            self.observables.w_lesser_density = -density(
                self.data.w_lesser, self.overlap
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.w_lesser_density = (
                    self.observables.w_lesser_density.sum(axis=-1)
                )
            self.observables.w_greater_density = -density(
                self.data.w_greater, self.overlap
            )
            if not self.quatrex_config.outputs.spatially_resolved:
                self.observables.w_greater_density = (
                    self.observables.w_greater_density.sum(axis=-1)
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

        if self.quatrex_config.bsc.coulomb_screening:
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

    def _compute_valley_difference(self, sp1, sp2, band="conduction") -> float:
        """
        Compute the valley differences between the `sp1` and `sp2` symmetry points in the `band` band edge.

        Parameters
        ----------
        sp1 : array-like
            The first symmetry point.
        sp2 : array-like
            The second symmetry point.
        band : str
            The band edge to consider ('conduction' or 'valence').
        """
        dos = xp.sum(
            -density(self.data.g_retarded, self.overlap) / (2 * xp.pi),
            axis=-1,
        )
        num_kpoints = np.array(
            [kp for kp in self.quatrex_config.electron.num_kpoints if kp > 1]
        )
        grids = [(np.arange(n) - n // 2) / n for n in num_kpoints]
        kpoint_mesh = np.meshgrid(*grids, indexing="ij")
        # Find the corresponding k-point indices for the symmetry points. The closest k-point is used.
        # Make sure the symmetry points are numpy arrays.
        sp1 = get_host(xp.asarray(sp1))
        sp2 = get_host(xp.asarray(sp2))
        sp1_idx = divmod(
            np.argmin(
                np.linalg.norm(kpoint_mesh - sp1.reshape(-1, 1, 1), axis=0),
            ),
            num_kpoints[-1],
        )
        sp2_idx = divmod(
            np.argmin(
                np.linalg.norm(kpoint_mesh - sp2.reshape(-1, 1, 1), axis=0),
            ),
            num_kpoints[-1],
        )
        # Find the peaks in the density of states at the symmetry points.
        peaks_sp1, _ = find_peaks(get_host(dos[:, *sp1_idx]))
        peaks_sp2, _ = find_peaks(get_host(dos[:, *sp2_idx]))
        energies_sp1 = self.electron_energies[peaks_sp1]
        energies_sp2 = self.electron_energies[peaks_sp2]
        mid_bandgap = (self.conduction_band_edges + self.valence_band_edges) / 2
        if band == "conduction":
            edge_sp1 = np.min(energies_sp1[energies_sp1 > mid_bandgap])
            edge_sp2 = np.min(energies_sp2[energies_sp2 > mid_bandgap])
        elif band == "valence":
            edge_sp1 = np.max(energies_sp1[energies_sp1 < mid_bandgap])
            edge_sp2 = np.max(energies_sp2[energies_sp2 < mid_bandgap])
        else:
            raise ValueError("band must be either 'conduction' or 'valence'.")
        valley_difference = edge_sp1 - edge_sp2
        return valley_difference

    @profiler.profile(level="basic")
    def run(self) -> None:
        """Runs the BSC to convergence."""
        print("Entering BSC loop...", flush=True) if comm.rank == 0 else None

        for i in range(self.quatrex_config.bsc.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None
            # append for iteration time
            synchronize_device()
            comm.barrier()
            t_iteration_start = time.perf_counter()

            t_assemble_start = time.perf_counter()
            self._assemble_greens_function_system_matrix(
                self.data.sigma_retarded,
            )
            synchronize_device()
            t_assemble_end = time.perf_counter()
            comm.barrier()
            t_assemble_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for assembling system matrix: {t_assemble_end - t_assemble_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for assembling system matrix all: {t_assemble_end_all - t_assemble_start:.3f} s",
                    flush=True,
                )

            t_solve_start = time.perf_counter()
            self.solver.selected_inv(
                self.data.g_system_matrix,
                out=self.data.g_retarded,
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for inverting system matrix: {t_solve_end - t_solve_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for inverting system matrix all: {t_solve_end_all - t_solve_start:.3f} s",
                    flush=True,
                )

            t_band_edge_start = time.perf_counter()
            # Find the band edges and update the Fermi level (for charge neutrality).
            self._find_band_edges()
            self._update_fermi_level()
            if comm.rank == 0:
                print(f"Fermi level: {self.fermi_level}", flush=True)
            if i == 0:
                previous_charge = self._compute_total_charge()
            else:
                current_charge = self._compute_total_charge()
                if comm.rank == 0:
                    print(f"Current charge: {current_charge}", flush=True)
                    print(f"Previous charge: {previous_charge}", flush=True)
                    print(
                        f"Charge difference: {current_charge - previous_charge}",
                        flush=True,
                    )
                previous_charge = current_charge

            # Compute the valley difference between symmetry points.
            # The `K` and `Q` symmetry points are used for the conduction band edge.
            K_symmetry_point = np.array([1 / 3, 1 / 3])
            Q_symmetry_point = np.array([1 / 6, 1 / 6])
            valley_difference = self._compute_valley_difference(
                K_symmetry_point, Q_symmetry_point
            )
            if comm.rank == 0:
                print(
                    f"Valley difference between K and Q symmetry points: {valley_difference}",
                    flush=True,
                )
            # The `K` and `G` symmetry points are used for the valence band edge.
            G_symmetry_point = np.array([0, 0])
            valley_difference = self._compute_valley_difference(
                K_symmetry_point, G_symmetry_point, band="valence"
            )
            if comm.rank == 0:
                print(
                    f"Valley difference between K and G symmetry points: {valley_difference}",
                    flush=True,
                )

            synchronize_device()
            t_band_edge_end = time.perf_counter()
            comm.barrier()
            t_band_edge_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for band edge and fermi level: {t_band_edge_end - t_band_edge_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for band edge and fermi level all: {t_band_edge_end_all - t_band_edge_start:.3f} s",
                    flush=True,
                )

            t_lesser_greater_start = time.perf_counter()
            _spectral_function(self.data.g_retarded, out=self.data.g_lesser)
            self.data.g_greater.data[:] = self.data.g_lesser.data
            scale_stack(self.data.g_lesser.data, -self.occupancies)
            scale_stack(self.data.g_greater.data, 1 - self.occupancies)

            synchronize_device()
            t_lesser_greater_end = time.perf_counter()
            comm.barrier()
            t_lesser_greater_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for  lesser/greater: {t_lesser_greater_end - t_lesser_greater_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for  lesser/greater all: {t_lesser_greater_end_all - t_lesser_greater_start:.3f} s",
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

            # Stash current into previous self-energy buffer, also zero out current self-energy.
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

            # Hartree interaction.
            # Can (should) be computed in stack distribution.
            if self.quatrex_config.bsc.hartree:
                self._compute_hartree_interaction()

            # Transpose to nnz distribution.
            # NOTE: While computing all interactions, we only ever need
            # to access the Green's function and the self-energies in
            # their nnz-distributed state.
            t_start_transpose = time.perf_counter()
            for m in (
                self.data.g_lesser,
                self.data.g_greater,
                self.data.sigma_retarded,
            ):
                m.dtranspose(discard=False)  # This must not be discarded.
                assert m.distribution_state == "nnz"
            for m in (
                self.data.sigma_lesser,
                self.data.sigma_greater,
                # self.data.sigma_retarded,
            ):
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "nnz"
            synchronize_device()
            t_end_transpose = time.perf_counter()
            comm.barrier()
            t_end_transpose_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"bsc: Time for transposing forth: {t_end_transpose - t_start_transpose:.3f} s",
                    flush=True,
                )
                print(
                    f"bsc: Time for transposing forth all: {t_end_transpose_all - t_start_transpose:.3f} s",
                    flush=True,
                )

            if self.quatrex_config.bsc.coulomb_screening:
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

            if self.quatrex_config.bsc.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.bsc.phonon:
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
                    f"bsc: Time for transposing back: {t_transpose_sigma_end - t_transpose_sigma_start:.3f} s",
                    flush=True,
                )
                print(
                    f"bsc: Time for transposing back all: {t_transpose_sigma_end_all - t_transpose_sigma_start:.3f} s",
                    flush=True,
                )

            t_sigma_symmetrize_start = time.perf_counter()
            # Make sure the self-energies satify the required symmetries.
            self._symmetrize_sigma()
            synchronize_device()
            t_sigma_symmetrize_end = time.perf_counter()
            comm.barrier()
            t_sigma_symmetrize_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for symmetrizing: {t_sigma_symmetrize_end - t_sigma_symmetrize_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for symmetrizing all: {t_sigma_symmetrize_end_all - t_sigma_symmetrize_start:.3f} s",
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

            if i % self.quatrex_config.bsc.output_interval == 0:
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
                print(f"BSC did not converge after {i} iterations.")
