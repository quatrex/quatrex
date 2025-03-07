# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator
from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse


class LyapunovConfig(BaseModel):
    """Configuration concerning the Lyapunov solvers."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy"] = "numpy"


class NEVPConfig(BaseModel):
    """All configurations concerning the solution of NEVPs."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy"] = "numpy"
    project_compute_location: Literal["numpy", "cupy"] = "numpy"

    use_qr: bool = False
    contour_batch_size: PositiveInt | None = None
    num_threads_contour: PositiveInt = 1024


class BandEdgeConfig(BaseModel):
    """Parameters concerning the eigenvalue-based band-edge tracking."""

    model_config = ConfigDict(extra="forbid")

    use_eigvalsh: bool = False
    eigvalsh_compute_location: Literal["numpy", "cupy"] = "numpy"


class ConvolveConfig(BaseModel):
    """All configurations concerning the fft convolution."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: should be calculate from the number of energy points, ranks,
    # and nnz.
    batch_size: PositiveInt | None = None


class ComputeConfig(BaseModel):
    """All configurations concerning computational details."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsbsparse_type: DSBSparse = DSBCOO

    convolve: ConvolveConfig = ConvolveConfig()
    nevp: NEVPConfig = NEVPConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()
    band_edge: BandEdgeConfig = BandEdgeConfig()

    @field_validator("dsbsparse_type", mode="before")
    def set_dsbsparse(cls, value) -> DSBSparse:
        """Converts the string value to the corresponding DSBSparse object."""
        if value == "DSBCSR":
            return DSBCSR
        elif value == "DSBCOO":
            return DSBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")


def parse_config(config_file: Path) -> ComputeConfig:
    """Reads the TOML config file.

    Parameters
    ----------
    config_file : Path
        Path to the TOML config file.

    Returns
    -------
    ComputeConfig
        The parsed compute config.

    """
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return ComputeConfig(**config)
