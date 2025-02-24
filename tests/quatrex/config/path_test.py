# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
from pathlib import Path

import pytest

from quatrex.core.quatrex_config import parse_config

quatrex_folder = Path(os.path.dirname(__file__)).parent.parent


@pytest.mark.parametrize(
    "toml_filename", ["examples/w90/cnt/config.toml", "examples/cp2k/cnt/config.toml"]
)
def test_simulation_path(toml_filename: str) -> None:
    toml_file = quatrex_folder.joinpath(toml_filename)
    print(toml_file)
    config = parse_config(toml_file)
    print(config.simulation_dir)
    assert os.path.exists(config.simulation_dir)
    assert config.simulation_dir == toml_file.parent


if __name__ == "__main__":
    pytest.main(["-v", __file__])
