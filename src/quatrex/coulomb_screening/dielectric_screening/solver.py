# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from __future__ import annotations

from typing import TYPE_CHECKING

from .equilibrium_screening import (
    EquilibriumScreening,
    EquilibriumScreeningInputs,
    ScreenedCoulombResult,
)

if TYPE_CHECKING:
    from quatrex.core.config import QuatrexConfig

    from .rpa_compute import BrillouinZoneMesh, ScreeningChannels


class DielectricScreeningSolver:
    """Config-driven equilibrium dielectric-screening solver.

    This mirrors the role of ``quatrex.coulomb_screening.solver`` at the package
    level: the higher-level driver supplies runtime parameters, while this class
    loads persistent Hamiltonian and Coulomb inputs from ``config.input_dir``.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        *,
        hamiltonian_matrix_name: str = "hamiltonian",
        coulomb_matrix_name: str = "coulomb_matrix",
        channels: ScreeningChannels | None = None,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
    ) -> None:
        self.config = config
        self.hamiltonian_matrix_name = hamiltonian_matrix_name
        self.coulomb_matrix_name = coulomb_matrix_name
        self.periodic_axis = periodic_axis
        self.lattice_constant = lattice_constant
        self._solver = EquilibriumScreening(channels=channels)

    def load_inputs(self) -> EquilibriumScreeningInputs:
        """Load runtime-managed matrix inputs from ``config.input_dir``."""

        return self._solver.load_inputs_from_config(
            self.config,
            hamiltonian_matrix_name=self.hamiltonian_matrix_name,
            coulomb_matrix_name=self.coulomb_matrix_name,
        )

    def solve(
        self,
        *,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        q_index: int,
        frequency_index: int,
        broadening: float = 0.0,
    ) -> ScreenedCoulombResult:
        """Solve dielectric screening from inputs stored in ``config.input_dir``."""

        return self._solver.solve_from_inputs(
            inputs=self.load_inputs(),
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            q_index=q_index,
            frequency_index=frequency_index,
            periodic_axis=self.periodic_axis,
            lattice_constant=self.lattice_constant,
            broadening=broadening,
        )


class CoulombScreeningSolver(DielectricScreeningSolver):
    """Backward-compatible alias for dielectric-screening solves."""


__all__ = [
    "CoulombScreeningSolver",
    "DielectricScreeningSolver",
]
