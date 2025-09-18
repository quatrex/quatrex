# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import pytest

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.compute_config import parse_config as parse_compute_config
from quatrex.core.quatrex_config import parse_config as parse_quatrex_config
from quatrex.examples import get_example_dir


@pytest.mark.usefixtures("example")
def test_parse_quatrex_config(example: str):
    _, _, example_path = get_example_dir(example)
    quatrex_config_path = example_path / "quatrex_config.toml"

    parse_quatrex_config(quatrex_config_path)


@pytest.mark.usefixtures("non_distributed_example")
def test_parse_compute_config(non_distributed_example: str):
    _, _, example_path = get_example_dir(non_distributed_example)
    compute_config_path = example_path / "compute_config.toml"
    if not compute_config_path.exists():
        ComputeConfig()
    else:
        parse_compute_config(compute_config_path)


@pytest.mark.usefixtures("non_distributed_example")
def test_parse_config(non_distributed_example: str):
    _, _, example_path = get_example_dir(non_distributed_example)
    quatrex_config_path = example_path / "quatrex_config.toml"
    compute_config_path = example_path / "compute_config.toml"
    if not compute_config_path.exists():
        ComputeConfig()
    else:
        parse_compute_config(compute_config_path)

    parse_quatrex_config(quatrex_config_path)
