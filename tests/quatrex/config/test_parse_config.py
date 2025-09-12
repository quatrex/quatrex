# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import pytest

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.compute_config import parse_config as parse_compute_config
from quatrex.core.quatrex_config import parse_config as parse_quatrex_config
from quatrex.examples import load as load_example


@pytest.mark.usefixtures("example_name")
def test_parse_quatrex_config(example_name: str):
    try:
        quatrex_config_path = (
            load_example(
                example_name + "-quatrex-config", folder=example_name, force=True
            )
            / "quatrex_config.toml"
        )
    except Exception as e:
        pytest.fail(f"Failed to load example config: {e}")

    try:
        parse_quatrex_config(quatrex_config_path)
    except Exception as e:
        pytest.fail(f"Failed to parse config file: {e}")


@pytest.mark.usefixtures("example_name")
def test_parse_compute_config(example_name: str):
    try:
        try:
            compute_config_path = (
                load_example(
                    example_name + "-compute-config", folder=example_name, force=True
                )
                / "compute_config.toml"
            )
        except (FileNotFoundError, ValueError):
            compute_config_path = None
    except Exception as e:
        pytest.fail(f"Failed to load example config: {e}")

    try:
        if compute_config_path is not None:
            parse_compute_config(compute_config_path)
        else:
            ComputeConfig()
    except Exception as e:
        pytest.fail(f"Failed to parse config file: {e}")


@pytest.mark.usefixtures("example_name")
def test_parse_config(example_name: str):
    try:
        quatrex_config_path = (
            load_example(
                example_name + "-quatrex-config", folder=example_name, force=True
            )
            / "quatrex_config.toml"
        )
        try:
            compute_config_path = (
                load_example(
                    example_name + "-compute-config", folder=example_name, force=True
                )
                / "compute_config.toml"
            )
        except (FileNotFoundError, ValueError):
            compute_config_path = None
    except Exception as e:
        pytest.fail(f"Failed to load example configs: {e}")

    try:
        if compute_config_path is not None:
            parse_compute_config(compute_config_path)
        else:
            ComputeConfig()

        parse_quatrex_config(quatrex_config_path)
    except Exception as e:
        pytest.fail(f"Failed to parse config files: {e}")
