# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the core classes for the self-consistent Born approximation (SCBA) solver."""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.comm import comm, distributed_max
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.observables import current_conservation, density, scattering_rate
from quatrex.core.transport import TransportSolver
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.device import BaseDevice, SCBADevice
from quatrex.device.inputs import get_block_sizes
from quatrex.electron import (
    ElectronSolver,
    SigmaCoulombScreening,
    SigmaFock,
    SigmaPhononDeformationPotential,
    SigmaPhononPseudoScattering,
    SigmaPhoton,
)
from quatrex.grid import KGrid, get_electron_energies, get_equal_spacing
from quatrex.grid.orbit_packing import OrbitPacking
from quatrex.photon import PhotonSolver, PiPhoton

profiler = Profiler()


class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.
    device : SCBADevice
        The device object to be used in the simulation.
    electron_energies : NDArray
        The electron energies for the SCBA calculation.

    """

    def __init__(
        self, config: QuatrexConfig, device: SCBADevice, electron_energies: NDArray
    ) -> None:
        """Initializes the SCBA data."""
        # Load orbital positions, energy vector and block-sizes.

        __, grid, __, atomic_species, __ = BaseDevice._load_structure(config)
        self.orbitals_per_atom = [
            config.device.num_orbitals_per_atom.get(s, 1) for s in atomic_species
        ]

        block_sizes = get_block_sizes(config, grid)

        # Find the maximum interaction cutoff.

        dsdbsparse_type = config.compute.dsdbsparse_type

        self.g_retarded = dsdbsparse_type.from_sparray(
            sparray=device.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape + (device.kgrid.num_points,),
            allocate=False,
        )

        self.g_lesser = dsdbsparse_type.from_sparray(
            sparray=device.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape + (device.kgrid.num_points,),
            symmetry="skew-hermitian" if config.scba.symmetric else None,
            allocate=False,
        )
        self.g_greater = dsdbsparse_type.empty_like(self.g_lesser)

        self.sigma_lesser_prev = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_lesser = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_greater_prev = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_greater = dsdbsparse_type.empty_like(self.g_lesser)

        self.sigma_retarded_hermitian_prev = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_retarded_hermitian = dsdbsparse_type.empty_like(self.g_lesser)
        if config.scba.symmetric:
            self.sigma_retarded_hermitian.symmetry = "hermitian"
            self.sigma_retarded_hermitian_prev.symmetry = "hermitian"

        if config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_lesser = dsdbsparse_type.from_sparray(
                sparray=device.sparsity_pattern.astype(xp.complex128),
                block_sizes=block_sizes,
                global_stack_shape=electron_energies.shape + (device.qgrid.num_points,),
                symmetry="skew-hermitian" if config.scba.symmetric else None,
                allocate=False,
            )
            self.p_greater = dsdbsparse_type.empty_like(self.p_lesser)
            self.p_retarded_hermitian = dsdbsparse_type.empty_like(self.p_lesser)

            if config.scba.symmetric:
                self.p_retarded_hermitian.symmetry = "hermitian"

            self.w_lesser = dsdbsparse_type.from_sparray(
                sparray=device.sparsity_pattern.astype(xp.complex128),
                block_sizes=device.coulomb_block_sizes,
                global_stack_shape=electron_energies.shape + (device.qgrid.num_points,),
                symmetry="skew-hermitian" if config.scba.symmetric else None,
                allocate=False,
            )
            self.w_greater = dsdbsparse_type.empty_like(self.w_lesser)

        # TODO: The interactions with photons are not yet
        # implemented.
        if config.scba.photon:
            raise NotImplementedError

        # Allocate the data for the Green's functions and self-energies.
        self.g_lesser.allocate_data()
        self.g_greater.allocate_data()
        # NOTE: The solvers do not set the elements that are outside of
        # the block-tridiagonal part.
        self.g_lesser.data = 0.0
        self.g_greater.data = 0.0
        self.sigma_lesser.allocate_data()
        self.sigma_greater.allocate_data()
        self.sigma_retarded_hermitian.allocate_data()
        self.sigma_lesser.data = 0.0
        self.sigma_greater.data = 0.0
        self.sigma_retarded_hermitian.data = 0.0


class PackedSCBAData:
    """Data container class for the symmetry-unique elements.

    The packed matrices are elements in real space and do not have any
    k-points. The interactions are thus the ones of a system with a
    single k-point, which is described by the `device` attribute.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.
    device : SCBADevice
        The device object to be used in the simulation.
    data : SCBAData
        The data on the irreducible k-points. Only the sparsity pattern
        is used.
    num_energies : int
        The number of electron energies.

    """

    def __init__(
        self,
        config: QuatrexConfig,
        device: SCBADevice,
        data: SCBAData,
        num_energies: int,
    ) -> None:
        """Initializes the packed SCBA data."""
        self.packing = OrbitPacking(
            data.g_lesser.rows,
            data.g_lesser.cols,
            device.orbital_operations,
            device.kgrid,
            device.qgrid,
        )
        self.dsdbsparse_type = config.compute.dsdbsparse_type
        # The number of sections in which to unpack.
        self.num_batches = config.compute.convolve.num_batches

        names = ["g_lesser", "g_greater", "sigma_lesser", "sigma_greater"]
        hermitian_names = ["sigma_retarded_hermitian"]
        if config.scba.coulomb_screening:
            names += ["p_lesser", "p_greater", "w_lesser", "w_greater"]
            hermitian_names += ["p_retarded_hermitian"]

        for name in names:
            setattr(self, name, self._create_matrix(num_energies, "skew-hermitian"))
        for name in hermitian_names:
            setattr(self, name, self._create_matrix(num_energies, "hermitian"))

        self.device = SimpleNamespace(
            kgrid=KGrid((1, 1, 1)), qgrid=KGrid((1, 1, 1), gamma_centered=True)
        )
        if config.scba.coulomb_screening:
            self.device.coulomb_matrix = self._create_matrix(
                comm.stack.size, "hermitian"
            )
            self.packing.accumulate_matrix(
                self.device.coulomb_matrix.data[:, 0],
                device.coulomb_matrix,
                signed=False,
                mask=xp.ones(1, dtype=bool),
            )

        if comm.rank == 0:
            num_full = len(data.g_lesser.rows) * device.kgrid.num_full_points
            print(
                f"Using {self.packing.num_representatives} symmetry-unique elements "
                f"of {num_full} on the full k-point grid.",
                flush=True,
            )

    def _create_matrix(self, num_energies: int, symmetry: str) -> DSDBSparse:
        """Creates a packed matrix."""
        return self.packing.create_matrix(num_energies, self.dsdbsparse_type, symmetry)

    def get_outputs(self, *names: str) -> tuple[NDArray, ...]:
        """Gets the arrays in which the solvers have to pack.

        The current data is discarded.

        """
        outputs = []
        for name in names:
            matrix = getattr(self, name)
            if matrix.distribution_state != "stack":
                matrix.dtranspose(discard=True)
            outputs.append(matrix.data[:, 0])
        return tuple(outputs)

    def unpack(self, name: str, out: DSDBSparse, signed: bool) -> None:
        """Unpacks a matrix to the irreducible points.

        Parameters
        ----------
        name : str
            The name of the packed matrix.
        out : DSDBSparse
            The matrix on the irreducible points, in stack distribution.
        signed : bool
            Whether this is an electronic quantity (Sigma) or an
            interaction (P).

        """
        matrix = getattr(self, name)
        if matrix.distribution_state != "stack":
            matrix.dtranspose()

        num_energies = out.data.shape[0]
        if num_energies == 0:
            return
        sizes, __ = get_section_sizes(num_energies, min(self.num_batches, num_energies))
        offsets = np.hstack(([0], np.cumsum(sizes)))
        for start, end in zip(offsets, offsets[1:]):
            batch = slice(int(start), int(end))
            out.data[batch] = self.packing.unpack(
                matrix.data[batch, 0], signed, out.symmetry
            )


class PackedDiagonal:
    """The diagonal elements of a packed matrix.

    This provides what is needed for interactions that only act on the
    diagonal elements, irrespective of the k-point.

    Parameters
    ----------
    matrix : DSDBSparse
        The packed matrix.
    packing : OrbitPacking
        The packing of the matrix.

    """

    def __init__(self, matrix: DSDBSparse, packing: OrbitPacking) -> None:
        """Initializes the view on the diagonal elements."""
        self._matrix = matrix
        if matrix.distribution_state != "nnz":
            matrix.dtranspose()
        start = int(matrix.nnz_section_offsets[comm.stack.rank])
        size = int(matrix.nnz_section_sizes[comm.stack.rank])
        is_diagonal = (packing.rep_rows == packing.rep_cols)[start : start + size]
        self._inds = xp.asarray(np.flatnonzero(is_diagonal))

    @property
    def data(self) -> NDArray:
        """The data of the packed matrix."""
        return self._matrix.data

    @property
    def distribution_state(self) -> str:
        """The distribution state of the packed matrix."""
        return self._matrix.distribution_state

    def diagonal(self) -> NDArray:
        """Returns the elements (i, i, R)."""
        return self._matrix.data[..., self._inds]

    def fill_diagonal(self, values: NDArray) -> None:
        """Sets the elements (i, i, R)."""
        self._matrix.data[..., self._inds] = values


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

    sigma_lesser_density: NDArray = None
    sigma_greater_density: NDArray = None

    scattering_rate: NDArray = None

    current_conservation: NDArray = None

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

    device : SCBADevice
        The device object to be used in the simulation.

    """

    def __init__(self, config: QuatrexConfig, device: SCBADevice) -> None:
        """Initializes an SCBA instance."""
        self.config = config

        self.device = device
        self.observables = Observables()
        electron_energies = xp.zeros((comm.size,))
        self.data = SCBAData(
            config=config,
            device=self.device,
            electron_energies=electron_energies,
        )  # dummy data
        self.mixing_factor = self.config.scba.mixing_factor

        # ----- Electrons ----------------------------------------------
        self.electron_energies = get_electron_energies(config)

        # If the symmetry of the device is known, the interactions are
        # computed from the symmetry-unique elements.
        self.packed = None
        interaction_device = self.device
        if self.device.uses_packing:
            self.packed = PackedSCBAData(
                config=config,
                device=self.device,
                data=self.data,
                num_energies=len(self.electron_energies),
            )
            interaction_device = self.packed.device

        min_energy = self.electron_energies[0]
        max_energy = self.electron_energies[-1]
        num_energies = len(self.electron_energies)
        energy_resolution = get_equal_spacing(self.electron_energies)
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
            config=self.config,
            device=self.device,
            energies=self.electron_energies,
        )

        # ----- Coulomb screening --------------------------------------
        if self.config.scba.coulomb_screening:

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
                interaction_device.coulomb_matrix.dtranspose()
                if interaction_device.coulomb_matrix.distribution_state != "nnz"
                else None
            )
            if "exchange" in self.config.coulomb_screening.energy_renormalization_terms:
                self.sigma_fock = SigmaFock(
                    self.config,
                    interaction_device,
                    self.electron_energies,
                )
            # Have to transpose the coulomb matrix back to the original distribution.
            (
                interaction_device.coulomb_matrix.dtranspose()
                if interaction_device.coulomb_matrix.distribution_state == "nnz"
                else None
            )

            # NOTE: No sparsity information required here.
            self.p_coulomb_screening = PCoulombScreening(
                self.config,
                interaction_device,
                self.coulomb_screening_energies,
            )
            self.coulomb_screening_solver = CoulombScreeningSolver(
                self.config,
                self.device,
                self.coulomb_screening_energies,
            )
            self.sigma_coulomb_screening = SigmaCoulombScreening(
                self.config,
                interaction_device,
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
            if self.config.phonon.model == "pseudo-scattering":
                self.sigma_phonon = SigmaPhononPseudoScattering(
                    config, self.electron_energies
                )

            elif self.config.phonon.model == "deformation-potential":
                if self.device.overlap_matrices is not None:
                    raise ValueError(
                        'The "deformation-potential" model is only implemented for an orthonormal basis.'
                    )
                self.sigma_phonon = SigmaPhononDeformationPotential(
                    config, self.electron_energies
                )

            else:
                raise ValueError(f"Unknown phonon model: {self.config.phonon.model}")

        self.data = SCBAData(
            config=config,
            device=self.device,
            electron_energies=self.electron_energies,
        )  # real data

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""

        self.data.sigma_lesser_prev.allocate_data()
        self.data.sigma_greater_prev.allocate_data()
        self.data.sigma_retarded_hermitian_prev.allocate_data()

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
            self.data.sigma_lesser.symmetrize("skew-hermitian")
            self.data.sigma_greater.symmetrize("skew-hermitian")
            # Make the self-energy Hermitian
            # This is done before adding the skew hermitian part coming
            # from the lesser and greater self-energies
            self.data.sigma_retarded_hermitian.symmetrize("hermitian")

        # NOTE: The packed self-energies are already aligned.
        if (
            self.config.scba.align_self_energy_to_complex_axes
            and not self.device.uses_packing
        ):
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

        self.data.sigma_lesser_prev.free_data()
        self.data.sigma_greater_prev.free_data()
        self.data.sigma_retarded_hermitian_prev.free_data()

    @profiler.profile(label="SCBA: Convergence test", level="default", comm=comm)
    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        max_diff_sigma_lesser = distributed_max(
            xp.abs(self.data.sigma_lesser.data - self.data.sigma_lesser_prev.data),
            comm.global_,
        )
        max_diff_sigma_greater = distributed_max(
            xp.abs(self.data.sigma_greater.data - self.data.sigma_greater_prev.data),
            comm.global_,
        )
        max_diff_sigma_retarded_hermitian = distributed_max(
            xp.abs(
                self.data.sigma_retarded_hermitian.data
                - self.data.sigma_retarded_hermitian_prev.data
            ),
            comm.global_,
        )

        meir_wingreen_current = self.observables.electron_current.get(
            "meir-wingreen", xp.array([0, 0])
        )
        i_left = self.device.kgrid.integrate(xp.real(meir_wingreen_current[..., 0]))
        i_right = self.device.kgrid.integrate(xp.real(meir_wingreen_current[..., -1]))

        dE = get_equal_spacing(self.electron_energies)
        left_current = xp.sum(i_left) * dE
        right_current = xp.sum(i_right) * dE
        avg_current = (left_current + right_current) / 2
        current_diff = xp.abs(left_current - right_current)
        rel_current_diff = current_diff / xp.abs(avg_current)

        current_conservation = self.observables.current_conservation

        if comm.rank == 0:
            print(
                f"Maximum Lesser Self-Energy Update: {max_diff_sigma_lesser}",
                flush=True,
            )
            print(
                f"Maximum Greater Self-Energy Update: {max_diff_sigma_greater}",
                flush=True,
            )
            print(
                f"Maximum Hermitian Retarded Self-Energy Update: {max_diff_sigma_retarded_hermitian}",
                flush=True,
            )
            print(f"Left Contact Current: {left_current}", flush=True)
            print(f"Right Contact Current: {right_current}", flush=True)
            print(f"Contact Current Difference: {current_diff}", flush=True)
            print(
                f"Relative Contact Current Difference: {rel_current_diff}", flush=True
            )
            print(
                f"Relative Scattering Current Conservation: {current_conservation}",
                flush=True,
            )

        return False  # TODO: :-)

    @profiler.profile(label="SCBA: Phonon interactions", level="default", comm=comm)
    def _compute_phonon_interaction(self):
        """Computes the phonon interaction."""
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

        if "exchange" in self.config.coulomb_screening.energy_renormalization_terms:
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

    @profiler.profile(label="SCBA: Packed interactions", level="default", comm=comm)
    def _compute_packed_interactions(self) -> None:
        """Computes the interactions from the symmetry-unique elements.

        The polarization and the self-energy are only unpacked to the
        irreducible points once they are needed by the solvers.

        """
        packed = self.packed
        packed_sigma = (
            packed.sigma_lesser,
            packed.sigma_greater,
            packed.sigma_retarded_hermitian,
        )
        sigma = (
            self.data.sigma_lesser,
            self.data.sigma_greater,
            self.data.sigma_retarded_hermitian,
        )
        with profiler.profile_range(
            label="SCBA: stack->nnz transpose", level="default", comm=comm
        ):
            for m in (packed.g_lesser, packed.g_greater):
                m.dtranspose(discard=False)  # This must not be discarded.
                assert m.distribution_state == "nnz"
            for m in packed_sigma:
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "nnz"
                m.data[:] = 0.0

        # The Green's functions on the irreducible k-points are only
        # needed for the observables.
        self.data.g_lesser.free_data()
        self.data.g_greater.free_data()

        if self.config.scba.coulomb_screening:
            self._compute_packed_coulomb_screening_interaction(packed_sigma)

        if self.config.scba.photon:
            self._compute_photon_interaction()

        if self.config.scba.phonon:
            self._compute_packed_phonon_interaction(packed_sigma)

        with profiler.profile_range(
            label="SCBA: Unpack Sigma", level="default", comm=comm
        ):
            if self.config.scba.align_self_energy_to_complex_axes:
                # NOTE: This is done in real space. In k-space, the
                # elements also have phases due to the neighboring cells.
                packed.sigma_lesser._data.real = 0
                packed.sigma_greater._data.real = 0
                packed.sigma_retarded_hermitian._data.imag = 0

            for name, m in zip(
                ("sigma_lesser", "sigma_greater", "sigma_retarded_hermitian"), sigma
            ):
                packed.unpack(name, out=m, signed=True)

        for m in (self.data.g_lesser, self.data.g_greater):
            m.allocate_data()
            m.data = 0.0

    def _compute_packed_coulomb_screening_interaction(
        self, packed_sigma: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the Coulomb screening interaction.

        Parameters
        ----------
        packed_sigma : tuple[DSDBSparse, ...]
            The packed self-energies.

        """
        packed = self.packed
        self.p_coulomb_screening.compute(
            packed.g_lesser,
            packed.g_greater,
            out=(packed.p_lesser, packed.p_greater, packed.p_retarded_hermitian),
        )

        p_names = ("p_lesser", "p_greater", "p_retarded_hermitian")
        for name in p_names:
            m = getattr(self.data, name)
            m.allocate_data()
            packed.unpack(name, out=m, signed=False)

        self.data.w_greater.allocate_data()
        self.data.w_lesser.allocate_data()

        self.coulomb_screening_solver.solve(
            self.data.p_lesser,
            self.data.p_greater,
            self.data.p_retarded_hermitian,
            out=(self.data.w_lesser, self.data.w_greater),
            packing=packed.packing,
            packed_out=packed.get_outputs("w_lesser", "w_greater"),
        )

        self._compute_coulomb_screening_observables()

        for name in (*p_names, "w_lesser", "w_greater"):
            getattr(self.data, name).free_data()

        if "exchange" in self.config.coulomb_screening.energy_renormalization_terms:
            self.sigma_fock.compute(
                packed.g_lesser, out=(packed.sigma_retarded_hermitian,)
            )

        self.sigma_coulomb_screening.compute(
            packed.g_lesser,
            packed.g_greater,
            packed.w_lesser,
            packed.w_greater,
            out=packed_sigma,
        )

    def _compute_packed_phonon_interaction(
        self, packed_sigma: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the phonon interaction.

        This does not mix different k-points, so that it applies to the
        elements in real space in the same way.

        Parameters
        ----------
        packed_sigma : tuple[DSDBSparse, ...]
            The packed self-energies.

        """
        packed_g = (self.packed.g_lesser, self.packed.g_greater)
        if self.config.phonon.model == "pseudo-scattering":
            # Only acts on the diagonal elements.
            packed_g, packed_sigma = (
                tuple(PackedDiagonal(m, self.packed.packing) for m in matrices)
                for matrices in (packed_g, packed_sigma)
            )
        self.sigma_phonon.compute(*packed_g, out=packed_sigma)

    @profiler.profile(label="SCBA: Full interactions", level="default", comm=comm)
    def _compute_interactions(self) -> None:
        """Computes the interactions on the k-point grid."""
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

    @profiler.profile(label="SCBA: G observables", level="default", comm=comm)
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""

        self.observables.current_conservation = current_conservation(
            self.data.g_lesser,
            self.data.g_greater,
            self.data.sigma_lesser,
            self.data.sigma_greater,
            kpoint_weights=self.device.kgrid._multiplicities,
        )
        if self.config.outputs.scattering_rate:
            self.observables.scattering_rate = scattering_rate(
                self.data.g_lesser,
                self.data.g_greater,
                self.data.sigma_lesser,
                self.data.sigma_greater,
            )

        if self.config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                self.device.overlap_matrices,
            ) / (2 * xp.pi)
            self.observables.electron_ldos *= 2  # Spin
        if self.config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                self.device.overlap_matrices,
            ) / (2 * xp.pi)
            self.observables.electron_density *= 2  # Spin
        if self.config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                self.device.overlap_matrices,
            ) / (2 * xp.pi)
            self.observables.hole_density *= 2  # Spin

        if self.config.outputs.device_currents:
            self.observables.electron_current["device"] = comm.stack.all_gather_v(
                self.electron_solver.device_current,
                axis=0,
                mask=self.data.g_lesser._stack_padding_mask,
            )

        if self.config.outputs.meir_wingreen_currents:
            local_current = self.electron_solver.meir_wingreen_current
            meir_wingreen_current = comm.stack.all_gather_v(
                local_current,
                axis=0,
                mask=self.data.g_lesser._stack_padding_mask,
            )

            self.observables.electron_current["meir-wingreen"] = meir_wingreen_current

        if self.config.outputs.self_energy_density:
            self.observables.sigma_lesser_density = density(
                self.data.sigma_lesser,
                self.device.overlap_matrices,
            ) / (2 * xp.pi)
            self.observables.sigma_greater_density = -density(
                self.data.sigma_greater,
                self.device.overlap_matrices,
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

    def _expand_orbitals(self, values: NDArray, grid: KGrid) -> NDArray:
        """Expands orbital-resolved values to the full k-point grid."""
        if self.device.uses_packing:
            return self.packed.packing.expand_orbitals(values, grid)
        return grid.expand(values)

    @profiler.profile(label="SCBA: Write outputs", level="default", comm=comm)
    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if comm.rank != 0:
            return

        outputs = {}

        kpoint_shape = tuple(s for s in self.device.kgrid.size if s > 1)
        newshape = (*self.electron_energies.shape, *kpoint_shape, -1)

        if self.config.outputs.scattering_rate:
            outputs[f"scattering_rate_{iteration}.npy"] = self._expand_orbitals(
                self.observables.scattering_rate, self.device.kgrid
            ).reshape(newshape)

        if self.config.outputs.electron_ldos:
            outputs[f"electron_ldos_{iteration}.npy"] = self._expand_orbitals(
                self.observables.electron_ldos, self.device.kgrid
            ).reshape(newshape)
        if self.config.outputs.electron_density:
            outputs[f"electron_density_{iteration}.npy"] = self._expand_orbitals(
                self.observables.electron_density, self.device.kgrid
            ).reshape(newshape)
        if self.config.outputs.hole_density:
            outputs[f"hole_density_{iteration}.npy"] = self._expand_orbitals(
                self.observables.hole_density, self.device.kgrid
            ).reshape(newshape)

        if self.config.outputs.device_currents:
            outputs[f"device_current_{iteration}.npy"] = self.device.kgrid.expand(
                self.observables.electron_current["device"]
            ).reshape(newshape)
        if self.config.outputs.meir_wingreen_currents:
            outputs[f"meir_wingreen_current_{iteration}.npy"] = (
                self.device.kgrid.expand(
                    self.observables.electron_current["meir-wingreen"]
                ).reshape(newshape)
            )

        if self.config.scba.coulomb_screening:
            if self.config.outputs.polarization_density:
                outputs.update(
                    {
                        f"p_lesser_density_{iteration}.npy": self._expand_orbitals(
                            self.observables.p_lesser_density, self.device.qgrid
                        ).reshape(newshape),
                        f"p_greater_density_{iteration}.npy": self._expand_orbitals(
                            self.observables.p_greater_density, self.device.qgrid
                        ).reshape(newshape),
                    }
                )
            if self.config.outputs.coulomb_screening_density:
                outputs.update(
                    {
                        f"w_lesser_density_{iteration}.npy": self._expand_orbitals(
                            self.observables.w_lesser_density, self.device.qgrid
                        ).reshape(newshape),
                        f"w_greater_density_{iteration}.npy": self._expand_orbitals(
                            self.observables.w_greater_density, self.device.qgrid
                        ).reshape(newshape),
                    }
                )

        if self.config.outputs.self_energy_density:
            outputs.update(
                {
                    f"sigma_lesser_density_{iteration}.npy": self._expand_orbitals(
                        self.observables.sigma_lesser_density, self.device.kgrid
                    ).reshape(newshape),
                    f"sigma_greater_density_{iteration}.npy": self._expand_orbitals(
                        self.observables.sigma_greater_density, self.device.kgrid
                    ).reshape(newshape),
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

        # NOTE: Use the mid-gap-energy of a reference contact.
        if self.electron_solver.left_voltage == 0.0:
            mid_gap_energy = self.electron_solver.left_mid_gap_energy
        elif self.electron_solver.right_voltage == 0.0:
            mid_gap_energy = self.electron_solver.right_mid_gap_energy
        else:
            raise NotImplementedError(
                "Cannot determine mid-gap energy for excess charge density calculation. "
                "At least one contact must be grounded (zero voltage)."
            )

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

        This computes the excess charge density from the spectral
        electron and hole densities.

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

        if self.device.uses_packing:
            self.device.check_potential(self.device.potential)

        for i in range(self.config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            with profiler.profile_range(
                label="SCBA: Iteration", level="default", comm=comm
            ):
                self.data.g_retarded.allocate_data()
                self.data.g_retarded.data = 0.0
                packing_kwargs = {}
                if self.device.uses_packing:
                    packing_kwargs = {
                        "packing": self.packed.packing,
                        "packed_out": self.packed.get_outputs("g_lesser", "g_greater"),
                    }
                self.electron_solver.solve(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded_hermitian,
                    out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
                    **packing_kwargs,
                )
                self._compute_electron_observables()

                self.data.g_retarded.free_data()

                # Stash current into previous self-energy buffer.
                self._stash_sigma()

                if self.device.uses_packing:
                    self._compute_packed_interactions()
                else:
                    self._compute_interactions()

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
