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
    load_translation_blocks,
    load_translation_blocks_from_config,
)

if TYPE_CHECKING:
    from quatrex.core.config import QuatrexConfig


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


def compute_screened_coulomb_matrices(
    coulomb_matrices: NDArray[np.complex128],
    polarization: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compute dielectric and screened interaction matrices on a full grid.

    ``polarization`` may be either scalar-valued with shape ``(nq, n_omega)`` or
    matrix-valued with shape ``(nq, n_omega, norb, norb)``.
    """

    matrices = np.asarray(coulomb_matrices, dtype=np.complex128)
    polarization_grid = np.asarray(polarization, dtype=np.complex128)

    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "coulomb_matrices must have shape (nq, norb, norb) with square matrices."
        )
    if polarization_grid.ndim not in (2, 4):
        raise ValueError(
            "polarization must have shape (nq, n_omega) or (nq, n_omega, norb, norb)."
        )
    if matrices.shape[0] != polarization_grid.shape[0]:
        raise ValueError(
            "The q dimension of coulomb_matrices and polarization must match."
        )
    if (
        polarization_grid.ndim == 4
        and polarization_grid.shape[-2:] != matrices.shape[-2:]
    ):
        raise ValueError("matrix-valued polarization must match coulomb matrix shape.")

    nq, norb, _ = matrices.shape
    n_omega = polarization_grid.shape[1]
    identity = np.eye(norb, dtype=np.complex128)

    dielectric_matrices = np.empty((nq, n_omega, norb, norb), dtype=np.complex128)
    screened_interactions = np.empty_like(dielectric_matrices)

    for q_index in range(nq):
        coulomb_matrix = matrices[q_index]
        for frequency_index in range(n_omega):
            polarization_value = polarization_grid[q_index, frequency_index]
            if polarization_grid.ndim == 2:
                dielectric_matrix = identity - polarization_value * coulomb_matrix
            else:
                dielectric_matrix = identity - coulomb_matrix @ polarization_value
            dielectric_matrices[q_index, frequency_index] = dielectric_matrix
            screened_interactions[q_index, frequency_index] = np.linalg.solve(
                dielectric_matrix,
                coulomb_matrix,
            )

    return dielectric_matrices, screened_interactions


class EquilibriumScreening:
    """Lightweight equilibrium screening bridge using RPA polarization and a Coulomb matrix."""

    def __init__(
        self,
        *,
        channels: ScreeningChannels | None = None,
        matrix_polarization: bool = False,
        frequency_axis: str = "imaginary",
    ) -> None:
        self.polarization_solver = RPAPolarization(
            channels=channels,
            frequency_axis=frequency_axis,
        )
        self.matrix_polarization = matrix_polarization

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

        if self.matrix_polarization:
            polarization_result = (
                self.polarization_solver.solve_matrix_from_translation_blocks(
                    translation_blocks=hamiltonian_blocks,
                    mesh=mesh,
                    chemical_potential=chemical_potential,
                    temperature=temperature,
                    periodic_axis=periodic_axis,
                    lattice_constant=lattice_constant,
                    broadening=broadening,
                )
            )
        else:
            polarization_result = (
                self.polarization_solver.solve_from_translation_blocks(
                    translation_blocks=hamiltonian_blocks,
                    mesh=mesh,
                    chemical_potential=chemical_potential,
                    temperature=temperature,
                    periodic_axis=periodic_axis,
                    lattice_constant=lattice_constant,
                    broadening=broadening,
                )
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
    "ScreenedCoulombGridResult",
    "build_coulomb_matrices",
    "compute_screened_coulomb_matrices",
    "load_coulomb_matrix",
    "load_coulomb_matrix_from_config",
]
