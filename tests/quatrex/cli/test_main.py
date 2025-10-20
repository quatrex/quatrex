# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import subprocess

import pytest

from quatrex.cli.main import fetch_example, run_quatrex
from quatrex.examples import get_example_dir


@pytest.mark.usefixtures("example")
def test_fetch_example(example: str):
    try:
        fetch_example(example)
    except Exception as e:
        pytest.fail(f"fetch_example failed: {e}")


@pytest.mark.usefixtures("example")
def test_fetch_example_cli(example: str):
    try:
        subprocess.run(
            ["quatrex", "fetch-example", example],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"fetch-example CLI failed: {e}")


@pytest.mark.usefixtures("non_distributed_example")
def test_main(non_distributed_example: str):

    try:
        fetch_example(non_distributed_example)
    except Exception as e:
        pytest.fail(f"fetch-example failed: {e}")

    _, _, example_path = get_example_dir(non_distributed_example)

    quatrex_config_path = example_path / "quatrex_config.toml"
    compute_config_path = example_path / "compute_config.toml"

    if not compute_config_path.exists():
        compute_config_path = None

    run_quatrex(
        quatrex_config_path,
        compute_config_path,
    )


@pytest.mark.usefixtures("example")
def test_main_cli(example: str):

    try:
        fetch_example(example)
    except Exception as e:
        pytest.fail(f"fetch-example failed: {e}")

    _, _, example_path = get_example_dir(example)

    quatrex_config_path = example_path / "quatrex_config.toml"
    compute_config_path = example_path / "compute_config.toml"

    try:
        if not compute_config_path.exists():
            subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "6",
                    "quatrex",
                    "run",
                    str(quatrex_config_path),
                ],
                check=True,
                stdout=None,
                stderr=None,
            )
        else:
            subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "6",
                    "quatrex",
                    "run",
                    str(quatrex_config_path),
                    str(compute_config_path),
                ],
                check=True,
                stdout=None,
                stderr=None,
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
