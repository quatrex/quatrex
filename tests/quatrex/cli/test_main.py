# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import subprocess

import pytest

from quatrex.cli.main import fetch_example, run_example
from quatrex.examples import load as load_example


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
            ["quatrex", "fetch_example", "--name", example_name, "--force"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"fetch_example CLI failed: {e}")


@pytest.mark.usefixtures("example_name")
def test_run_example(example_name: str):
    try:
        run_example(example_name, force=True)
    except Exception as e:
        pytest.fail(f"run_example failed: {e}")


@pytest.mark.usefixtures("example_name")
def test_run_example_cli(example_name: str):
    try:
        subprocess.run(
            ["quatrex", "run_example", "--name", example_name, "--force"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"run_example CLI failed: {e}")


@pytest.mark.usefixtures("example_name")
def test_main_cli(example_name: str):

    try:
        quatrex_config = (
            load_example(
                example_name + "-quatrex-config", folder=example_name, force=True
            )
            / "quatrex_config.toml"
        )
        try:
            compute_config = (
                load_example(
                    example_name + "-compute-config", folder=example_name, force=True
                )
                / "compute_config.toml"
            )
        except (FileNotFoundError, ValueError):
            compute_config = None
    except Exception as e:
        pytest.fail(f"Failed to load example configs: {e}")

    try:
        if compute_config is None:
            subprocess.run(
                ["quatrex", "--quatrex-config", str(quatrex_config)],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "quatrex",
                    "--quatrex-config",
                    str(quatrex_config),
                    "--compute-config",
                    str(compute_config),
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
