# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import tomllib
from math import isclose
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self


class SCSPConfig(BaseModel):
    """Options for the self-consistent Schrödinger-Poisson loop."""

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)


class SCBAConfig(BaseModel):
    """Options for the self-consistent Born approximation."""

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)

    output_interval: PositiveInt = 1

    coulomb_screening: bool = False
    photon: bool = False
    phonon: bool = False


class PoissonConfig(BaseModel):
    """Options for the Poisson solver."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["point-charge", "orbital"] = "point-charge"
    max_iterations: PositiveInt = 100
    convergence_tol: PositiveFloat = 1e-5
    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)

    rho_shift: NonNegativeFloat = 1e-8
    cg_tol: PositiveFloat = 1e-5
    cg_max_iter: PositiveInt = 100

    num_orbitals_per_atom: dict[str, int] = Field(default_factory=dict)


class MemoizerConfig(BaseModel):
    """Options for memoizing wrappers."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    num_ref_iterations: PositiveInt = 2
    convergence_tol: PositiveFloat = 1e-5


class SolverConfig(BaseModel):
    """Options for the system solver."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["rgf", "inv"] = "rgf"

    # The maximum number of energies per batch.
    max_batch_size: PositiveInt = 1


class OBCConfig(BaseModel):
    """Options for open-boundary condition (OBC) solvers."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "spectral"] = "spectral"
    nevp_solver: Literal["beyn", "full"] = "beyn"

    # Parameters for spectral OBC algorithms.
    block_sections: PositiveInt = 1
    min_decay: PositiveFloat = 1e-6
    max_decay: PositiveFloat | None = None
    num_ref_iterations: PositiveInt = 2
    x_ii_formula: Literal["self-energy", "direct"] = "self-energy"
    two_sided: bool = False
    treat_pairwise: bool = True
    pairing_threshold: PositiveFloat = 0.25
    min_propagation: PositiveFloat = 1e-2

    # Parameters for iterative OBC algorithms.
    max_iterations: PositiveInt = 1000
    convergence_tol: PositiveFloat = 1e-7

    # Parameters for subspace NEVP solvers.
    r_o: PositiveFloat = 10.0
    r_i: PositiveFloat = 0.9
    m_0: PositiveInt = 10
    num_quad_points: PositiveInt = 20

    # Parameters for reusing surface Green's functions from previous
    # SCBA iterations.
    memoizer: MemoizerConfig = MemoizerConfig()


class LyapunovConfig(BaseModel):
    """Options for solving the Lyapunov equation."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["spectral", "doubling"] = "spectral"

    # Parameters for iterative Lyapunov algorithms.
    max_iterations: PositiveInt = 1000
    convergence_tol: PositiveFloat = 1e-7

    # Parameter for spectral Lyapunov solver.
    num_ref_iterations: PositiveInt = 3

    memoizer: MemoizerConfig = MemoizerConfig()


class ElectronConfig(BaseModel):
    """Options for the electronic subsystem solver."""

    model_config = ConfigDict(extra="forbid")

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    eta_obc: NonNegativeFloat = 0  # eV
    eta: NonNegativeFloat = 1e-12  # eV

    use_energy_x: bool = False

    fermi_level: float | None = None
    conduction_band_edge: float | None = None
    valence_band_edge: float | None = None

    left_fermi_level: float | None = None
    right_fermi_level: float | None = None

    band_edge_tracking: Literal["dos-peaks", "eigenvalues"] | None = None

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    flatband: bool | None = None

    @model_validator(mode="after")
    def set_left_right_fermi_levels(self) -> Self:
        """Sets the left and right Fermi levels if not already set."""
        if (self.left_fermi_level is None) != (self.right_fermi_level is None):
            raise ValueError(
                "Either both left and right Fermi levels must be set or neither."
            )

        if self.left_fermi_level is None and self.right_fermi_level is None:
            if self.fermi_level is None:
                raise ValueError("Fermi level must be set.")

            self.left_fermi_level = self.fermi_level
            self.right_fermi_level = self.fermi_level

        return self

    @model_validator(mode="after")
    def set_left_right_temperatures(self) -> Self:
        """Sets the left and right temperatures if not already set."""
        if (self.left_temperature is None) != (self.right_temperature is None):
            raise ValueError(
                "Either both left and right temperatures must be set or neither."
            )

        if self.left_temperature is None and self.right_temperature is None:
            self.left_temperature = self.temperature
            self.right_temperature = self.temperature

        return self

    @model_validator(mode="after")
    def set_flatband(self) -> Self:
        """Sets the flatband flags if not already set."""
        if self.flatband is None:
            if isclose(self.left_fermi_level, self.right_fermi_level):
                self.flatband = True
            else:
                self.flatband = False

        return self


class CoulombScreeningConfig(BaseModel):
    """Options for the Coulomb screening solver."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None


class PhotonConfig(BaseModel):
    """Options for the optical degrees of freedom."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()


class PhononConfig(BaseModel):
    """Options for the thermal degrees of freedom."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    phonon_energy: NonNegativeFloat | None = None
    deformation_potential: NonNegativeFloat | None = None
    temperature: PositiveFloat = 300.0  # K

    @model_validator(mode="after")
    def check_phonon_energy_or_deformation_potential(self):
        """Check if 'phonon_energy' and 'deformation_potential' are set."""
        if self.model == "pseudo-scattering" and (
            self.phonon_energy is None or self.deformation_potential is None
        ):
            raise ValueError("'phonon_energy' and 'deformation_potential' must be set.")

        return self


class QuatrexConfig(BaseModel):
    """Top-level simulation configuration."""

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
    config_dir: Path
    simulation_dir: Path = Path("./quatrex/")

    @model_validator(mode="after")
    def resolve_config_path(self) -> Self:
        """Resolves the config directory path."""
        self.config_dir = Path(self.config_dir).resolve()
        return self

    @model_validator(mode="after")
    def resolve_simulation_dir(self):
        """Resolves the simulation directory path."""
        self.simulation_dir = (self.config_dir / self.simulation_dir).resolve()
        return self

    @property
    def input_dir(self) -> Path:
        """Returns the input directory path."""
        return self.simulation_dir / "inputs/"


def parse_config(config_file: Path) -> QuatrexConfig:
    """Reads the TOML config file.

    Parameters
    ----------
    config_file : Path
        Path to the TOML configuration file.

    Returns
    -------
    QuatrexConfig
        The parsed configuration object.

    """

    config_file = Path(config_file).resolve()

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    if "simulation_dir" in config:
        simulation_dir = config["simulation_dir"]
        if not os.path.isabs(simulation_dir):
            parent_dir = os.path.dirname(os.path.abspath(config_file))
            simulation_dir = Path(os.path.join(parent_dir, simulation_dir))
            config["simulation_dir"] = simulation_dir

    config["config_dir"] = config_file.parent

    return QuatrexConfig(**config)
