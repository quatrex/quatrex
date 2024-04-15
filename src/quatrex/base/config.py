# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.

from pathlib import Path

from pydantic import BaseModel


class StructureConfig(BaseModel):
    temperature: float = 300.0  # K
    num_cells: int


class OpenBoundaryConditionConfig(BaseModel):
    algorithm: str = "sancho-rubio"
    interaction_range: int = 1

    # Parameters for iterative solvers.
    max_iterations: int = 5000
    max_delta: float = 1e-8


class ElectronConfig(BaseModel):
    energy_range: list[float]
    energy_step: float = 1e-3  # eV
    eta: float = 1e-6  # eV

    fermi_level: float = 0.0  # eV
    fermi_level_splitting = 0.0  # eV

    solver: str = "dense"

    obc: OpenBoundaryConditionConfig = OpenBoundaryConditionConfig()


class QuatrexConfig(BaseModel):
    structure: StructureConfig

    electron: ElectronConfig

    # --- Directory paths ----------------------------------------------
    simulation_dir: Path = Path("./quatrex/")
