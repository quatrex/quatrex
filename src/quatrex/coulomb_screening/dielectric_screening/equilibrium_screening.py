from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .rpa_compute import (
    BrillouinZoneMesh,
    PolarizationResult,
    RPAPolarization,
    ScreeningChannels,
    build_bloch_hamiltonian,
    load_translation_blocks_from_config,
    load_translation_blocks,
)

if TYPE_CHECKING:
    from quatrex.core.config import QuatrexConfig


@dataclass(frozen=True)
class ScreenedCoulombResult:
    """Screened Coulomb quantities for a selected momentum and frequency point."""

    polarization_result: PolarizationResult
    coulomb_matrix: NDArray[np.complex128]
    polarization: np.complex128
    dielectric_matrix: NDArray[np.complex128]
    screened_interaction: NDArray[np.complex128]
    q_index: int
    frequency_index: int
    q_value: float
    frequency_value: float


@dataclass(frozen=True)
class ScreenedCoulombGridResult:
    """Screened Coulomb quantities on the full ``(q, omega)`` response grid."""

    polarization_result: PolarizationResult
    coulomb_matrices: NDArray[np.complex128]
    dielectric_matrices: NDArray[np.complex128]
    screened_interactions: NDArray[np.complex128]
    q_points: NDArray[np.float64]
    frequencies: NDArray[np.float64]


@dataclass(frozen=True)
class EquilibriumScreeningInputs:
    """Preloaded unit-cell inputs for a runtime-managed equilibrium screening solve."""

    hamiltonian_blocks: dict[tuple[int, int, int], NDArray[np.complex128]]
    coulomb_blocks: dict[tuple[int, int, int], NDArray[np.complex128]]


def load_coulomb_matrix(
    mat_file: str | Path,
    *,
    periodic_axis: int | None = None,
    lattice_constant: float = 1.0,
) -> dict[tuple[int, int, int], NDArray[np.complex128]]:
    """Load translation-resolved Coulomb blocks from a MATLAB file."""

    return load_translation_blocks(mat_file)


def load_coulomb_matrix_from_config(
    config: QuatrexConfig,
    *,
    matrix_name: str = "coulomb_matrix",
) -> dict[tuple[int, int, int], NDArray[np.complex128]]:
    """Load Coulomb translation blocks from ``config.input_dir``."""

    return load_translation_blocks_from_config(config, matrix_name=matrix_name)


def build_coulomb_matrices(
    coulomb_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
    q_points: NDArray[np.float64],
    *,
    periodic_axis: int | None = None,
    lattice_constant: float = 1.0,
) -> NDArray[np.complex128]:
    """Build bare Coulomb matrices ``V(q)`` on the q grid from translation blocks."""

    return build_bloch_hamiltonian(
        coulomb_blocks,
        q_points,
        periodic_axis=periodic_axis,
        lattice_constant=lattice_constant,
    )


def compute_dielectric_matrix(
    coulomb_matrix: NDArray[np.complex128],
    polarization: complex,
) -> NDArray[np.complex128]:
    """Compute the matrix dielectric function ``epsilon = I - V * Pi``.

    This is the simplest equilibrium matrix generalization where ``Pi`` is treated as
    a scalar response value at a chosen ``(q, omega)`` and ``V`` is the bare Coulomb
    matrix in the orbital or block basis.
    """

    matrix = np.asarray(coulomb_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coulomb_matrix must be a square matrix.")
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return identity - np.complex128(polarization) * matrix


def compute_screened_coulomb_matrix(
    coulomb_matrix: NDArray[np.complex128],
    polarization: complex,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compute ``epsilon`` and ``W = epsilon^{-1} V`` for a scalar polarization."""

    dielectric_matrix = compute_dielectric_matrix(coulomb_matrix, polarization)
    screened_interaction = np.linalg.solve(dielectric_matrix, coulomb_matrix)
    return dielectric_matrix, screened_interaction


def compute_screened_coulomb_matrices(
    coulomb_matrices: NDArray[np.complex128],
    polarization: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compute dielectric and screened interaction matrices on a full ``(q, omega)`` grid."""

    matrices = np.asarray(coulomb_matrices, dtype=np.complex128)
    polarization_grid = np.asarray(polarization, dtype=np.complex128)

    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "coulomb_matrices must have shape (nq, norb, norb) with square matrices."
        )
    if polarization_grid.ndim != 2:
        raise ValueError("polarization must have shape (nq, n_omega).")
    if matrices.shape[0] != polarization_grid.shape[0]:
        raise ValueError("The q dimension of coulomb_matrices and polarization must match.")

    nq, norb, _ = matrices.shape
    n_omega = polarization_grid.shape[1]
    identity = np.eye(norb, dtype=np.complex128)

    dielectric_matrices = np.empty((nq, n_omega, norb, norb), dtype=np.complex128)
    screened_interactions = np.empty_like(dielectric_matrices)

    for q_index in range(nq):
        coulomb_matrix = matrices[q_index]
        for frequency_index in range(n_omega):
            dielectric_matrix = (
                identity - polarization_grid[q_index, frequency_index] * coulomb_matrix
            )
            dielectric_matrices[q_index, frequency_index] = dielectric_matrix
            screened_interactions[q_index, frequency_index] = np.linalg.solve(
                dielectric_matrix,
                coulomb_matrix,
            )

    return dielectric_matrices, screened_interactions


class EquilibriumScreening:
    """Lightweight equilibrium screening bridge using RPA polarization and a Coulomb matrix."""

    def __init__(self, *, channels: ScreeningChannels | None = None) -> None:
        self.polarization_solver = RPAPolarization(channels=channels)

    def load_inputs_from_config(
        self,
        config: QuatrexConfig,
        *,
        hamiltonian_matrix_name: str = "hamiltonian",
        coulomb_matrix_name: str = "coulomb_matrix",
    ) -> EquilibriumScreeningInputs:
        """Load and bundle dielectric-screening inputs from ``config.input_dir``."""

        return EquilibriumScreeningInputs(
            hamiltonian_blocks=load_translation_blocks_from_config(
                config,
                matrix_name=hamiltonian_matrix_name,
            ),
            coulomb_blocks=load_coulomb_matrix_from_config(
                config,
                matrix_name=coulomb_matrix_name,
            ),
        )

    def solve_at_indices(
        self,
        *,
        hamiltonian_file: str | Path,
        coulomb_file: str | Path,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        q_index: int,
        frequency_index: int,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombResult:
        """Solve for screening at one selected ``(q, omega)`` point.

        The selected scalar polarization value is combined with the bare Coulomb
        matrix through ``epsilon = I - V * Pi`` and ``W = epsilon^{-1} V``.
        """

        hamiltonian_blocks = load_translation_blocks(hamiltonian_file)
        coulomb_blocks = load_coulomb_matrix(
            coulomb_file,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
        )
        return self.solve_from_translation_blocks(
            hamiltonian_blocks=hamiltonian_blocks,
            coulomb_blocks=coulomb_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
            q_index=q_index,
            frequency_index=frequency_index,
        )

    def solve_grid_from_inputs(
        self,
        *,
        inputs: EquilibriumScreeningInputs,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombGridResult:
        """Solve for screening on the full ``(q, omega)`` grid from preloaded inputs."""

        return self.solve_grid_from_translation_blocks(
            hamiltonian_blocks=inputs.hamiltonian_blocks,
            coulomb_blocks=inputs.coulomb_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )

    def solve_grid_from_translation_blocks(
        self,
        *,
        hamiltonian_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        coulomb_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombGridResult:
        """Solve for screening on the full ``(q, omega)`` grid."""

        polarization_result = self.polarization_solver.solve_from_translation_blocks(
            translation_blocks=hamiltonian_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )
        coulomb_matrices = np.asarray(
            build_coulomb_matrices(
                coulomb_blocks,
                mesh.q_points,
                periodic_axis=periodic_axis,
                lattice_constant=lattice_constant,
            ),
            dtype=np.complex128,
        )
        dielectric_matrices, screened_interactions = compute_screened_coulomb_matrices(
            coulomb_matrices,
            polarization_result.polarization,
        )
        return ScreenedCoulombGridResult(
            polarization_result=polarization_result,
            coulomb_matrices=coulomb_matrices,
            dielectric_matrices=dielectric_matrices,
            screened_interactions=screened_interactions,
            q_points=np.asarray(mesh.q_points, dtype=np.float64),
            frequencies=np.asarray(mesh.frequencies, dtype=np.float64),
        )

    def solve_from_inputs(
        self,
        *,
        inputs: EquilibriumScreeningInputs,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        q_index: int,
        frequency_index: int,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombResult:
        """Solve for screening from a preloaded runtime input bundle."""

        return self.solve_from_translation_blocks(
            hamiltonian_blocks=inputs.hamiltonian_blocks,
            coulomb_blocks=inputs.coulomb_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            q_index=q_index,
            frequency_index=frequency_index,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )

    def solve_from_translation_blocks(
        self,
        *,
        hamiltonian_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        coulomb_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        q_index: int,
        frequency_index: int,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombResult:
        """Solve for screening from preloaded Hamiltonian and Coulomb blocks."""

        grid_result = self.solve_grid_from_translation_blocks(
            hamiltonian_blocks=hamiltonian_blocks,
            coulomb_blocks=coulomb_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )
        q_points = grid_result.q_points
        frequencies = grid_result.frequencies
        if not (0 <= q_index < q_points.size):
            raise IndexError("q_index is out of bounds for mesh.q_points.")
        if not (0 <= frequency_index < frequencies.size):
            raise IndexError("frequency_index is out of bounds for mesh.frequencies.")

        coulomb_matrix = np.asarray(
            grid_result.coulomb_matrices[q_index], dtype=np.complex128
        )
        polarization = np.complex128(
            grid_result.polarization_result.polarization[q_index, frequency_index]
        )
        dielectric_matrix = np.asarray(
            grid_result.dielectric_matrices[q_index, frequency_index],
            dtype=np.complex128,
        )
        screened_interaction = np.asarray(
            grid_result.screened_interactions[q_index, frequency_index],
            dtype=np.complex128,
        )
        return ScreenedCoulombResult(
            polarization_result=grid_result.polarization_result,
            coulomb_matrix=coulomb_matrix,
            polarization=polarization,
            dielectric_matrix=dielectric_matrix,
            screened_interaction=screened_interaction,
            q_index=q_index,
            frequency_index=frequency_index,
            q_value=float(q_points[q_index]),
            frequency_value=float(frequencies[frequency_index]),
        )

    def solve_from_config(
        self,
        config: QuatrexConfig,
        *,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        q_index: int,
        frequency_index: int,
        hamiltonian_matrix_name: str = "hamiltonian",
        coulomb_matrix_name: str = "coulomb_matrix",
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombResult:
        """Solve for screening using inputs loaded through ``config.input_dir``."""

        inputs = self.load_inputs_from_config(
            config,
            hamiltonian_matrix_name=hamiltonian_matrix_name,
            coulomb_matrix_name=coulomb_matrix_name,
        )
        return self.solve_from_inputs(
            inputs=inputs,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            q_index=q_index,
            frequency_index=frequency_index,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )

    def solve_grid_from_config(
        self,
        config: QuatrexConfig,
        *,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        hamiltonian_matrix_name: str = "hamiltonian",
        coulomb_matrix_name: str = "coulomb_matrix",
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> ScreenedCoulombGridResult:
        """Solve for screening on the full ``(q, omega)`` grid using ``config.input_dir`` inputs."""

        inputs = self.load_inputs_from_config(
            config,
            hamiltonian_matrix_name=hamiltonian_matrix_name,
            coulomb_matrix_name=coulomb_matrix_name,
        )
        return self.solve_grid_from_inputs(
            inputs=inputs,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )


__all__ = [
    "EquilibriumScreening",
    "EquilibriumScreeningInputs",
    "ScreenedCoulombResult",
    "ScreenedCoulombGridResult",
    "build_coulomb_matrices",
    "compute_dielectric_matrix",
    "compute_screened_coulomb_matrix",
    "compute_screened_coulomb_matrices",
    "load_coulomb_matrix",
    "load_coulomb_matrix_from_config",
]