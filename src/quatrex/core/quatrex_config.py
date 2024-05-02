# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.


import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self


class SCSPConfig(BaseModel):
    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, gt=0.0, le=1.0)


class SCBAConfig(BaseModel):
    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, gt=0.0, le=1.0)

    observables_interval: PositiveInt | None = None

    coulomb_screening: bool = False
    photon: bool = False
    phonon: bool = False


class PoissonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["point-charge", "orbital"] = "point-charge"
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5
    mixing_factor: PositiveFloat = Field(default=0.1, gt=0.0, le=1.0)


class OBCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "beyn"] = "sancho-rubio"

    max_iterations: PositiveInt = 1000
    convergence_tol: PositiveFloat = 1e-7

    lyapunov_method: Literal["spectral", "scipy"] = "spectral"


class ElectronConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver: Literal["rgf", "inv"] = "rgf"

    obc: OBCConfig = OBCConfig()

    eta: PositiveFloat = 1e-6  # eV

    fermi_level: float

    left_fermi_level: float | None = None
    right_fermi_level: float | None = None

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    @model_validator(mode="after")
    def set_left_right_fermi_levels(self) -> Self:
        if self.left_fermi_level is None and self.right_fermi_level is not None:
            self.left_fermi_level = self.fermi_level
            self.right_fermi_level = self.fermi_level
        if (
            self.left_fermi_level is None
            and self.right_fermi_level is not None
            or self.left_fermi_level is not None
            and self.right_fermi_level is None
        ):
            raise ValueError(
                "Either both left and right Fermi levels must be set or none."
            )
        return self

    @model_validator(mode="after")
    def set_left_right_temperatures(self) -> Self:
        if self.left_temperature is None and self.right_temperature is not None:
            self.left_temperature = self.temperature
            self.right_temperature = self.temperature
        if (
            self.left_temperature is None
            and self.right_temperature is not None
            or self.left_temperature is not None
            and self.right_temperature is None
        ):
            raise ValueError(
                "Either both left and right temperatures must be set or none."
            )
        return self


class CoulombScreeningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver: Literal["rgf", "inv"] = "rgf"
    obc: OBCConfig = Field(default_factory=OBCConfig)


class PhotonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver: Literal["rgf", "inv"] = "rgf"
    obc: OBCConfig = Field(default_factory=OBCConfig)


class PhononConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solver: Literal["rgf", "inv"] = "rgf"
    obc: OBCConfig = Field(default_factory=OBCConfig)

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    phonon_energy: PositiveFloat | None
    deformation_potential: PositiveFloat | None

    @model_validator(mode="after")
    def check_phonon_energy_or_deformation_potential(self):
        if self.model == "pseudo-scattering":
            if self.phonon_energy is None or self.deformation_potential is None:
                raise ValueError("Phonon energy and deformation potential must be set.")


class QuatrexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    scsp: SCSPConfig = SCSPConfig()
    scba: SCBAConfig = SCBAConfig()
    poisson: PoissonConfig = PoissonConfig()

    electron: ElectronConfig

    phonon: PhononConfig | None = None
    coulomb_screening: CoulombScreeningConfig | None = None
    photon: PhotonConfig | None = None

    # --- Directory paths ----------------------------------------------
    simulation_dir: Path = Path("./quatrex/")

    @property
    def input_dir(self) -> Path:
        return self.simulation_dir / "inputs/"


def parse_config(config_file: Path) -> QuatrexConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return QuatrexConfig(**config)
