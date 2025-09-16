# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import subprocess

import pytest

from quatrex.cli.main import fetch_example, run_quatrex
from quatrex.examples import EXAMPLES_DIR


@pytest.mark.usefixtures("example_name")
def test_fetch_example(example_name: str):
    try:
        fetch_example(example_name, force=True)
    except Exception as e:
        pytest.fail(f"fetch_example failed: {e}")


@pytest.mark.usefixtures("example_name")
def test_fetch_example_cli(example_name: str):
    try:
        subprocess.run(
            ["quatrex", "fetch-example", "--name", example_name, "--force"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"fetch-example CLI failed: {e}")


@pytest.mark.usefixtures("example_name")
def test_main(example_name: str):

    quatrex_config = (
        EXAMPLES_DIR / example_name / "quatrex_config.toml"
    )
    compute_config = None

    run_quatrex(
        quatrex_config,
        compute_config,
    )


@pytest.mark.usefixtures("example_name")
def test_main_cli(example_name: str):

    quatrex_config_path = (
        EXAMPLES_DIR / example_name / "quatrex_config.toml"
    )
    try:
        compute_config_path = (
            EXAMPLES_DIR / example_name / "compute_config.toml"
        )
    except (FileNotFoundError, ValueError):
        compute_config_path = None

    try:
        if compute_config_path is None:
            subprocess.run(
                ["quatrex", "--quatrex-config", str(quatrex_config_path)],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "quatrex",
                    "--quatrex-config",
                    str(quatrex_config_path),
                    "--compute-config",
                    str(compute_config_path),
                ],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"main CLI failed: {e}")


def test_help_cli():
    try:
        subprocess.run(
            ["quatrex", "--help"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"help CLI failed: {e}")


def test_version_cli():
    try:
        subprocess.run(
            ["quatrex", "--version"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"version CLI failed: {e}")
