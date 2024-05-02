# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.


import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Memory config ---------------------------------------
    datastructure: Literal["dbcsr"] = "dbcsr"


def parse_config(config_file: Path) -> ComputeConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return ComputeConfig(**config)
