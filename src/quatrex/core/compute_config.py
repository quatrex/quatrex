# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from pathlib import Path

import tomllib
from pydantic import BaseModel, ConfigDict, field_validator
from qttools.datastructures import DSBCSR, DSBSparse


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- Memory config ------------------------------------------------
    dbsparse: DSBSparse = DSBCSR

    @field_validator("dbsparse", mode="before")
    def set_dbsparse(cls, value) -> DSBSparse:
        if value == "DSBCSR":
            return DSBCSR
        raise ValueError(f"Invalid value '{value}' for dbsparse")


def parse_config(config_file: Path) -> ComputeConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return ComputeConfig(**config)
