# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import tomllib
import warnings
from pathlib import Path
from typing import Literal

import numba as nb
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from qttools import xp
from qttools.comm import comm as qtx_comm
from qttools.datastructures import DSDBCOO, DSDBSparse


class LyapunovConfig(BaseModel):
    """Configuration concerning the Lyapunov solvers."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"
    use_pinned_memory: bool = True


class NEVPConfig(BaseModel):
    """All configurations concerning the solution of NEVPs."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"

    # Parameters for contour NEVP solvers.
    project_compute_location: Literal["numpy", "cupy"] = "numpy"
    use_pinned_memory: bool = True

    use_qr: bool = False
    contour_batch_size: PositiveInt | None = None
    num_threads_contour: PositiveInt = 1024

    # Parameters for full NEVP solvers.
    reduce_sparsity: bool = False


class BandEdgeConfig(BaseModel):
    """Parameters concerning the eigenvalue-based band-edge tracking."""

    model_config = ConfigDict(extra="forbid")

    use_eigvalsh: bool = True
    """Whether to use eigvalsh or eig to compute the eigenvalues to
    determine the band edges. The eigvalsh function is more efficient,
    but is an approximation if scattering is included.

    Only used if the band edge tracking is set to "eigenvalues".
    """

    eigvalsh_compute_location: Literal["numpy", "cupy"] = "numpy"
    """Location where to compute the eigenvalues.

    Only used if the band edge tracking is set to "eigenvalues".
    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory for eigenvalue computations.

    Only used if the band edge tracking is set to "eigenvalues".
    """

    block_sections: PositiveInt = 1

    @field_validator("use_eigvalsh", mode="after")
    def check_use_eigvalsh(cls, value, info) -> bool:
        if not value:
            raise NotImplementedError(
                "Only use_eigvalsh=True is supported at the moment."
            )
        return value

    @field_validator("eigvalsh_compute_location", mode="after")
    def check_eigvalsh_location(cls, value) -> Literal["numpy", "cupy"]:
        if value == "cupy" and xp.__name__ != "cupy":
            warnings.warn(
                "eigvalsh_compute_location is set to 'cupy' but cupy is not available. Falling back to 'numpy'.",
                UserWarning,
            )
            return "numpy"
        elif value == "numpy" and xp.__name__ == "cupy":
            warnings.warn(
                "eigvalsh_compute_location is set to 'numpy' but cupy is available. Consider setting it to 'cupy' for better performance.",
                UserWarning,
            )

        return value


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

    block_comm_config: dict[str, str] = {}
    stack_comm_config: dict[str, str] = {}

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        if xp.__name__ == "cupy":
            self.block_comm_config = {
                "all_to_all": self.block_all_to_all or "host_mpi",
                "all_gather": self.block_all_gather or "host_mpi",
                "all_reduce": self.block_all_reduce or "host_mpi",
                "bcast": self.block_bcast or "host_mpi",
            }

            self.stack_comm_config = {
                "all_to_all": self.stack_all_to_all or "host_mpi",
                "all_gather": self.stack_all_gather or "host_mpi",
                "all_reduce": self.stack_all_reduce or "host_mpi",
                "bcast": self.stack_bcast or "host_mpi",
            }
        else:
            self.block_comm_config = {
                "all_to_all": self.block_all_to_all or "device_mpi",
                "all_gather": self.block_all_gather or "device_mpi",
                "all_reduce": self.block_all_reduce or "device_mpi",
                "bcast": self.block_bcast or "device_mpi",
            }

            self.stack_comm_config = {
                "all_to_all": self.stack_all_to_all or "device_mpi",
                "all_gather": self.stack_all_gather or "device_mpi",
                "all_reduce": self.stack_all_reduce or "device_mpi",
                "bcast": self.stack_bcast or "device_mpi",
            }

        # configure the comm
        qtx_comm.configure(
            block_comm_size=self.block_comm_size,
            block_comm_config=self.block_comm_config,
            stack_comm_config=self.stack_comm_config,
            override=True,
        )

        return self


class ComputeConfig(BaseModel):
    """All configurations concerning computational details."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsdbsparse_type: DSDBSparse = DSDBCOO
    numba_threading_layer: Literal["workqueue", "omp", "tbb"] = "workqueue"
    threadpool_api: Literal["blas", "openmp", "tbb"] | None = None
    numba_num_threads: PositiveInt | None = None
    blas_num_threads: PositiveInt | Literal["sequential_blas_under_openmp"] | None = (
        None
    )

    convolve: ConvolveConfig = ConvolveConfig()
    nevp: NEVPConfig = NEVPConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()
    band_edge: BandEdgeConfig = BandEdgeConfig()
    comm: CommConfig = CommConfig()

    @field_validator("dsdbsparse_type", mode="before")
    def set_dsdbsparse(cls, value) -> DSDBSparse:
        """Converts the string value to the corresponding DSDBSparse object."""
        if value == "DSDBCOO":
            return DSDBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")

    @model_validator(mode="after")
    def set_threading(self) -> Self:

        # TODO: set the number of threads automatically based on the available cores
        # problems is that we do not know yet how many energy points there will be
        # has to be after unifying the configs
        if self.numba_num_threads is None:
            self.numba_num_threads = 1
        if self.blas_num_threads is None:
            self.blas_num_threads = 1

        nb.set_num_threads(self.numba_num_threads)
        nb.config.THREADING_LAYER = self.numba_threading_layer

        if self.numba_num_threads == 1 and self.blas_num_threads in [
            "sequential_blas_under_openmp",
            1,
        ]:
            if qtx_comm.rank == 0:
                warnings.warn(
                    "The CPU code will run sequentially which may impact performance.",
                    UserWarning,
                )

        return self


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
