# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import tomllib
from math import isclose
from pathlib import Path
from typing import Literal

import numpy as np
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

    symmetric: bool = False


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
    """Options for memoizing wrappers.

    The memoizers store and reuse previously computed results
    to speed up the fixed-point iterations in OBC and Lyapunov solvers.

    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "force", "force-after-first", "off"] = "auto"
    """The memoization mode to determine when to do fixed-point iterations.
    
    - "auto": Automatically decides whether to use memoization based on the
        specified tolerances. Only useful if all ranks memoize.
    - "force": Always use memoization.
    - "force-after-first": Use memoization after the first SCBA iteration.
    - "off": Never use memoization.
    """

    num_ref_iterations: PositiveInt = Field(default=2, ge=2)
    """The number of fixed-point iterations to perform."""

    relative_tol: PositiveFloat = 2e-1
    """The relative tolerance for the fixed-point iterations.
    
    Only used if `mode` is set to "auto".
    """

    absolute_tol: PositiveFloat = 1e-6
    """The absolute tolerance for the fixed-point iterations.

    Only used if `mode` is set to "auto".
    """

    warning_threshold: PositiveFloat = 1e-1
    """The threshold for issuing a warning if the surface Green's functions
        residual is above this value after the fixed-point iterations.
    """


class SolverConfig(BaseModel):
    """Options for the system solver."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["rgf", "inv"] = "rgf"

    # The maximum number of energies per batch.
    max_batch_size: PositiveInt = 100

    # Whether to compute the current via the Meir-Wingreen formula.
    compute_current: bool = False


class OBCConfig(BaseModel):
    r"""Options for open-boundary condition (OBC) solvers.

    The OBC solvers compute the surface Green's functions of the contacts.
    The surface Green's functions is the solution of the non-linear equation:

    $$ \mathbf{g} = [\mathbf{M}_{0} - \mathbf{M}_{-1} g \mathbf{M}_{1} ]^{-1} $$
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "spectral"] = "spectral"
    """The OBC algorithm to use.
    
    - "sancho-rubio": Uses the Sancho-Rubio iterative scheme to compute the
        surface Green's functions. This method achieves exponential convergence
        compared to the linear convergence of fixed-point iterations.
    - "spectral": Uses a spectral NEVP solver to compute eigenpair and uses
        them to construct the surface Green's functions. This is generally more
        efficient method when combined with a contour integral NEVP solver,
        but requires more parameter tuning.
    """

    nevp_solver: Literal["beyn", "full"] = "beyn"
    r"""The NEVP solver to use for the spectral OBC algorithm.

    - "beyn": Uses the Beyn's contour integral method to solve the NEVP to
        find the eigenpairs within a specified contour in the complex plane.

    - "full": Uses a full dense eigensolver to solve for all eigenvalues by linearizing
        the problem. This results in a doubled problem size which is also not reduced by
        block sectioning / periodicity.

    The following NEVP problem is solved:

    $$ \sum \limits_{n=-b}^{b} \lambda^{n} \hat{\mathbf{M}}_{n} \vec{v} = 0 $$

    where b goes from -block_sections to +block_sections and
    $\hat{\mathbf{M}}_{n}$ are potentially reduced coupling matrices.

    Only used if `algorithm` is set to "spectral".
    """

    # Parameters for spectral OBC algorithms.
    block_sections: PositiveInt = 1
    """The periodicity of the blocks along the transport direction.

    Used in the spectral method with beyn to reduce the size of the NEVP.
    For example, if the supercell is constructed from 2 unit cells along the
    transport direction, setting this parameter to 2 will halve the size of the NEVP.

    Contact blocks need to be sorted accordingly.
    """

    min_decay: PositiveFloat = 1e-3
    """The minimum decay rate where to differentiate between propagating and evanescent modes."""

    max_decay: PositiveFloat | None = None
    """The maximum decay rate for evanescent modes.

    Very large modes do not contribute to the surface Green's functions and
    can be neglected. Very large modes can also lead to numerical instabilities.

    If not set, it is computed as 1.5 * log(r_o).
    """

    num_ref_iterations: PositiveInt = 2
    r"""The number of fixed-point iterations used to refine the surface Green's functions.

        $$ \mathbf{g}_{n+1} = [\mathbf{M}_{0} - \mathbf{M}_{-1} \mathbf{g}_{n} \mathbf{M}_{1} ]^{-1} $$

    This is needed to improve the accuracy of the surface Green's functions
    if not enough eigenpairs are considered. 

    Only used if `algorithm` is set to "spectral".
    """

    min_propagation: PositiveFloat = 1e-2
    r"""The minimum propagation speed for propagating modes.
    
    The propagation speed is computed as:
    $$ abs(real(\frac{dE}{dk})) / abs(imag(\frac{dE}{dk})) $$

    """

    residual_tolerance: PositiveFloat = 1e-3
    r"""The tolerance for the residual of the eigenpairs.
    
    The residuals are computed as:
    $$ \lvert \sum \limits_{n=-b}^{b} \lambda^{b} \mathbf{M}_{n} \vec{v} \rvert $$

    Modes above this tolerance are considered wrong and are not used.

    Only used if `algorithm` is set to "spectral".
    """

    residual_normalization: bool = True
    """Whether to normalize the residuals by the norm of the eigenvalue.
    
    This is useful to avoid that large eigenvalues have large residuals
    and small eigenvalues have small residuals.
    """

    warning_threshold: PositiveFloat = 1e-1
    r"""The threshold for issuing a warning if the surface Green's functions
    residual is above this value.

    The residual is computed as:
    $$ \lvert \mathbf{g} - [\mathbf{M}_{0} - \mathbf{M}_{-1} \mathbf{g} \mathbf{M}_{1} ]^{-1} \rvert / \lvert \mathbf{g} \rvert $$
    """

    eta_decay: PositiveFloat = 1e-12
    """Small value to separate very slow decaying modes from
        non-decaying ones in the spectral OBC solver.

    Modes that are very close to the unit contour could be misclassified
    with 'min_decay' and 'min_propagation' conditions i.e. 
    when their decay is smaller than 'min_decay' but they are not propagating fast enough.
    The not fast enough propagating ones with decay smaller than 'eta_decay' are 
    considered as well decaying modes.
    """

    # Parameters for iterative OBC algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the Sancho-Rubio method."""

    convergence_tol: PositiveFloat = 1e-6
    """The convergence tolerance for the Sancho-Rubio method."""

    # Parameters for subspace NEVP solvers.
    r_o: PositiveFloat = 10.0
    """The outer radius of the contour in the complex plane for the contour methods.
    
    This parameter should not be too large to avoid having too many eigenpairs
    inside the contour. It should also not be too small to avoid missing important
    eigenpairs. If a eigenpair is too close to the contour,
    it can lead to numerical instabilities.
    """

    r_i: PositiveFloat = 0.8
    """The inner radius of the contour in the complex plane for the contour methods.

    This parameter should be chosen to be <1 to capture propagating modes, but
    not too small to avoid including too many modes.
    """

    m_0: PositiveInt = 10
    """The subspace guess in the contour methods.
    
    The guess has to be larger than the expected number of eigenvalues
    inside the contour. If too small, the method will fail. If too large, the method
    will be not/less efficient.
    """

    num_quad_points: PositiveInt = 20
    """The number of quadrature points for the contour integrals."""

    # Parameters for reusing surface Green's functions from previous
    # SCBA iterations.
    memoizer: MemoizerConfig = MemoizerConfig()

    @model_validator(mode="after")
    def set_max_decay(self) -> Self:
        """Sets the max decay if not already set."""
        if self.max_decay is None:
            self.max_decay = 1.5 * np.log(self.r_o)

        return self

    @model_validator(mode="after")
    def scale_contour_radii(self) -> Self:
        """Scales the contour radii based on block_sections."""
        self.r_o **= 1 / self.block_sections
        self.r_i **= 1 / self.block_sections

        return self


class LyapunovConfig(BaseModel):
    """Options for solving the Lyapunov equation."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["spectral", "doubling"] = "spectral"
    reduce_sparsity: bool = True

    # Parameters for iterative Lyapunov algorithms.
    max_iterations: PositiveInt = 100
    relative_tol: PositiveFloat = 1e-4
    absolute_tol: PositiveFloat = 1e-8

    # Parameter for spectral Lyapunov solver.
    num_ref_iterations: PositiveInt = Field(default=2, ge=1)
    warning_threshold: PositiveFloat = 1e-1

    memoizer: MemoizerConfig = MemoizerConfig()


class ElectronConfig(BaseModel):
    """Options for the electronic subsystem solver."""

    model_config = ConfigDict(extra="forbid")

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()

    eta_obc: NonNegativeFloat = 0  # eV
    eta: NonNegativeFloat = 1e-12  # eV

    fermi_level: float | None = None
    conduction_band_edge: float | None = None
    valence_band_edge: float | None = None

    left_fermi_level: float | None = None
    right_fermi_level: float | None = None

    band_edge_tracking: Literal["dos-peaks", "eigenvalues"] | None = None

    temperature: PositiveFloat = 300.0  # K

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    energy_window_min: float | None = None
    energy_window_max: float | None = None
    energy_window_num: PositiveInt | None = None
    energy_window_num_per_rank: PositiveInt | None = None

    flatband: bool | None = None

    dos_peak_limit: PositiveFloat = 100.0

    filtering_iteration_limit: PositiveInt = 1

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

    epsilon_r: PositiveFloat = 1.0

    left_temperature: PositiveFloat | None = None
    right_temperature: PositiveFloat | None = None

    # How many blocks should be merged into a single block.
    num_connected_blocks: Literal["auto"] | PositiveInt = "auto"

    dos_peak_limit: PositiveFloat = 100.0

    filtering_iteration_limit: PositiveInt = 1


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


class OutputConfig(BaseModel):
    """Options for the output."""

    model_config = ConfigDict(extra="forbid")

    # Only the spectral currents are saved by default.
    contact_currents: bool = True
    device_currents: bool = True

    potential: bool = False

    electron_ldos: bool = False
    electron_density: bool = False
    hole_density: bool = False

    polarization_density: bool = False
    coulomb_screening_density: bool = False

    self_energy_density: bool = False

    profiling_stats: bool = False


class DeviceConfig(BaseModel):

    construct_from_unit_cell: bool = False

    # --- Device geometry ---------------------------------------------
    unit_cell_per_supercell: tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 1)
    number_of_supercells: PositiveInt = 1
    transport_direction: Literal["x", "y", "z"]

    @model_validator(mode="after")
    def to_tuple(self) -> Self:
        """Transforms list to tuple."""
        self.unit_cell_per_supercell = tuple(self.unit_cell_per_supercell)
        return self


class QuatrexConfig(BaseModel):
    """Top-level simulation configuration."""

    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    device: DeviceConfig
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
    input_dir: Path | None = None
    output_dir: Path | None = None

    # --- Output options -----------------------------------------------
    outputs: OutputConfig = OutputConfig()

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

    @model_validator(mode="after")
    def set_output_dir(self):
        """Resolves the simulation directory path."""
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir).resolve()
            return self

        self.output_dir = self.simulation_dir / "outputs/"
        return self

    @model_validator(mode="after")
    def set_input_dir(self) -> Path:
        """Returns the input directory path."""
        if self.input_dir is not None:
            self.input_dir = Path(self.input_dir).resolve()
            return self
        self.input_dir = self.simulation_dir / "inputs/"
        return self


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
