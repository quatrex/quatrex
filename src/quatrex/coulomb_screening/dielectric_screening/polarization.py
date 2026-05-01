# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .rpa_compute import (
    PolarizationResult,
    RPAPolarization,
    ScreeningChannels,
    resolve_unit_cell_matrix_path,
)

if TYPE_CHECKING:
    from quatrex.core.config import QuatrexConfig

    from .rpa_compute import BrillouinZoneMesh


@dataclass(frozen=True)
class DielectricPolarizationInputs:
    """Runtime-managed inputs for dielectric-screening polarization."""

    hamiltonian_file: Path


class DielectricPolarization:
    """Config-driven RPA polarization loader for dielectric screening.

    This mirrors the role of ``quatrex.coulomb_screening.polarization`` at the
    package level: a higher-level runtime provides simulation state and this class
    loads the persistent Hamiltonian input from ``config.input_dir``.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        *,
        matrix_name: str = "hamiltonian",
        channels: ScreeningChannels | None = None,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
    ) -> None:
        self.config = config
        self.matrix_name = matrix_name
        self.periodic_axis = periodic_axis
        self.lattice_constant = lattice_constant
        self._solver = RPAPolarization(channels=channels)

    def load_inputs(self) -> DielectricPolarizationInputs:
        """Resolve the Hamiltonian file from ``config.input_dir``."""

        return DielectricPolarizationInputs(
            hamiltonian_file=resolve_unit_cell_matrix_path(
                self.config, self.matrix_name
            )
        )

    def compute(
        self,
        *,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute RPA polarization from the Hamiltonian in ``config.input_dir``."""

        return self._solver.solve_from_config(
            self.config,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            matrix_name=self.matrix_name,
            periodic_axis=self.periodic_axis,
            lattice_constant=self.lattice_constant,
            broadening=broadening,
        )


class PCoulombScreening(DielectricPolarization):
    """Backward-compatible alias for dielectric-screening polarization."""


__all__ = [
    "DielectricPolarization",
    "DielectricPolarizationInputs",
    "PCoulombScreening",
]
