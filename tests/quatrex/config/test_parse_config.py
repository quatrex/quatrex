# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import pytest

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.compute_config import parse_config as parse_compute_config
from quatrex.core.quatrex_config import parse_config as parse_quatrex_config
from quatrex.examples import EXAMPLES_DIR


@pytest.mark.usefixtures("example_name")
def test_parse_quatrex_config(example_name: str):
    quatrex_config_path = EXAMPLES_DIR / example_name / "quatrex_config.toml"

    parse_quatrex_config(quatrex_config_path)


@pytest.mark.usefixtures("example_name")
def test_parse_compute_config(example_name: str):
    compute_config_path = EXAMPLES_DIR / example_name / "compute_config.toml"
    if not compute_config_path.exists():
        ComputeConfig()
    else:
        parse_compute_config(compute_config_path)


@pytest.mark.usefixtures("example_name")
def test_parse_config(example_name: str):
    quatrex_config_path = EXAMPLES_DIR / example_name / "quatrex_config.toml"
    compute_config_path = EXAMPLES_DIR / example_name / "compute_config.toml"
    if not compute_config_path.exists():
        ComputeConfig()
    else:
        parse_compute_config(compute_config_path)

    parse_quatrex_config(quatrex_config_path)
