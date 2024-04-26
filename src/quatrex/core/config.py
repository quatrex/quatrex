# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.


import tomllib


from pathlib import Path

from pydantic import BaseModel


class SCBAConfig(BaseModel):
    max_iterations: int = 1000
    tolerance: float = 1e-8

    mixing_factor: float = 0.1

    electron: bool = True
    phonon: bool = False


class StructureConfig(BaseModel):
    temperature: float = 300.0  # K
    num_cells: int


class OBCConfig(BaseModel):
    algorithm: str = "sancho-rubio"

    # Parameters for iterative solvers.
    max_iterations: int = 5000
    max_delta: float = 1e-8


class ElectronConfig(BaseModel):
    eta: float = 1e-6  # eV

    fermi_level: float = 0.0  # eV
    fermi_level_splitting = 0.0  # eV

    solver: str = "dense_inv"

    obc: OBCConfig = OBCConfig()


class PhononConfig(BaseModel):
    energy_range: list[float]
    energy_step: float = 1e-3  # eV

    solver: str = "dense_inv"

    # "greens_function" or "deformation_potential"
    model: str = "deformation_potential"

    obc: OBCConfig = OBCConfig()


class QuatrexConfig(BaseModel):
    structure: StructureConfig

    electron: ElectronConfig
    phonon: PhononConfig

    # --- Directory paths ----------------------------------------------
    simulation_dir: Path = Path("./quatrex/")

    @property
    def input_dir(self) -> Path:
        return self.simulation_dir / "inputs/"


def parse_config(config_file: Path) -> QuatrexConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    QuatrexConfig.validate(config)
    return QuatrexConfig.parse_obj(config)
