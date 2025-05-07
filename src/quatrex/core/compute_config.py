# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator
from qttools.datastructures import DSBCOO, DSBCSR, DSDBCOO, DSBSparse, DSDBSparse
from qttools.comm import comm
from typing_extensions import Self
from qttools import xp

class LyapunovConfig(BaseModel):
    """Configuration concerning the Lyapunov solvers."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy"] = "numpy"
    use_pinned_memory: bool = True


class NEVPConfig(BaseModel):
    """All configurations concerning the solution of NEVPs."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy"] = "numpy"
    project_compute_location: Literal["numpy", "cupy"] = "numpy"
    use_pinned_memory: bool = True

    use_qr: bool = False
    contour_batch_size: PositiveInt | None = None
    num_threads_contour: PositiveInt = 1024


class BandEdgeConfig(BaseModel):
    """Parameters concerning the eigenvalue-based band-edge tracking."""

    model_config = ConfigDict(extra="forbid")

    use_eigvalsh: bool = False
    eigvalsh_compute_location: Literal["numpy", "cupy"] = "cupy"
    use_pinned_memory: bool = True
    block_sections: PositiveInt = 1


class ConvolveConfig(BaseModel):
    """All configurations concerning the fft convolution."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: should be calculate from the number of energy points, ranks,
    # and nnz.
    batch_size: PositiveInt | None = None

class CommConfig(BaseModel):
    """All configurations concerning the communication."""

    model_config = ConfigDict(extra="forbid")

    block_comm_size: PositiveInt = 1

    block_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    block_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None

    stack_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    stack_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None

    @model_validator(mode="after")
    def configure(self) -> Self:
        if xp.__name__ == "cupy":
            block_comm_config = {
                "all_to_all": self.block_all_to_all or "host_mpi",
                "all_gather": self.block_all_gather or "host_mpi",
                "all_reduce": self.block_all_reduce or "host_mpi",
                "bcast": self.block_bcast or "host_mpi",
            }

            stack_comm_config = {
                "all_to_all": self.stack_all_to_all or "host_mpi",
                "all_gather": self.stack_all_gather or "host_mpi",
                "all_reduce": self.stack_all_reduce or "host_mpi",
                "bcast": self.stack_bcast or "host_mpi",
        }
        else:
            block_comm_config = {
                "all_to_all": self.block_all_to_all or "device_mpi",
                "all_gather": self.block_all_gather or "device_mpi",
                "all_reduce": self.block_all_reduce or "device_mpi",
                "bcast": self.block_bcast or "device_mpi",
            }

            stack_comm_config = {
                "all_to_all": self.stack_all_to_all or "device_mpi",
                "all_gather": self.stack_all_gather or "device_mpi",
                "all_reduce": self.stack_all_reduce or "device_mpi",
                "bcast": self.stack_bcast or "device_mpi", 
            }

        comm.configure(
            block_comm_size=self.block_comm_size,
            block_comm_config=block_comm_config,
            stack_comm_config=stack_comm_config,
        )
        return self


class ComputeConfig(BaseModel):
    """All configurations concerning computational details."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsbsparse_type: DSBSparse = DSBCOO
    dsdbsparse_type: DSDBSparse = DSDBCOO

    convolve: ConvolveConfig = ConvolveConfig()
    nevp: NEVPConfig = NEVPConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()
    band_edge: BandEdgeConfig = BandEdgeConfig()
    comm: CommConfig = CommConfig()

    @field_validator("dsbsparse_type", mode="before")
    def set_dsbsparse(cls, value) -> DSBSparse:
        """Converts the string value to the corresponding DSBSparse object."""
        if value == "DSBCSR":
            return DSBCSR
        elif value == "DSBCOO":
            return DSBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")

    @field_validator("dsdbsparse_type", mode="before")
    def set_dsdbsparse(cls, value) -> DSDBSparse:
        """Converts the string value to the corresponding DSDBSparse object."""
        if value == "DSDBCOO":
            return DSDBCOO
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
