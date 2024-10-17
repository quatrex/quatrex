# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dbsparse_type: DSBSparse = DSBCOO

    @field_validator("dbsparse_type", mode="before")
    def set_dbsparse(cls, value) -> DSBSparse:
        if value == "DSBCSR":
            return DSBCSR
        elif value == "DSBCOO":
            return DSBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")


def parse_config(config_file: Path) -> ComputeConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return ComputeConfig(**config)
