import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from qttools.datastructures import DBCSR, DBSparse


class ComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Memory config ------------------------------------------------
    dbsparse: DBSparse = DBCSR

    @field_validator("dbsparse", mode="before")
    def set_dbsparse(cls, value) -> DBSparse:
        if value == "DBCSR":
            return DBCSR
        raise ValueError(f"Invalid value '{value}' for dbsparse")


def parse_config(config_file: Path) -> ComputeConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return ComputeConfig(**config)
