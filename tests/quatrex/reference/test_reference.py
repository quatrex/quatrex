# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import subprocess
import tomllib
from importlib.resources import files

import numpy as np
import pytest

from quatrex.cli.main import fetch_example, run
from quatrex.examples import get_example_dir, load

REFERENCE_OBSERVABLES = {
    "carbon-nanotube:": [
        "i_meir-wingreen_1",
        "electron_ldos_1",
    ],
    "carbon-nanotube:dist": [
        "electron_density_1",
        "i_device_1",
    ],
    "cp2k-atomic-chain:qtbm": [
        "dos_l",
        "dos_r",
        "transmission_lr",
    ],
}

assert len(REFERENCE_OBSERVABLES) == len(set(REFERENCE_OBSERVABLES.keys()))

# Load manifest containing example dataset info.
with open(files("quatrex.examples") / "_manifest.toml", "rb") as f:
    MANIFEST = tomllib.load(f)


for key, subnames in REFERENCE_OBSERVABLES.items():
    key, config = key.split(":")
    for subname in subnames:
        name = f"{key}-{subname}-{config}"
        # check exists in manifest
        if name not in MANIFEST:
            raise ValueError(f"Example '{name}' not found in manifest.")


@pytest.mark.usefixtures("non_distributed_example")
def test_non_distributed(non_distributed_example: str):

    if len(REFERENCE_OBSERVABLES[non_distributed_example]) == 0:
        pytest.skip(
            f"No reference observables defined for example '{non_distributed_example}'"
        )

    try:
        fetch_example(non_distributed_example)
    except Exception as e:
        pytest.fail(f"fetch-example failed: {e}")

    device_key, config_key, example_path = get_example_dir(non_distributed_example)

    quatrex_config_path = example_path / "quatrex_config.toml"
    compute_config_path = example_path / "compute_config.toml"

    if not compute_config_path.exists():
        compute_config_path = None

    run(
        quatrex_config_path,
        compute_config_path,
    )

    for observable in REFERENCE_OBSERVABLES[non_distributed_example]:

        # fetch reference solution
        load(
            device_key + "-" + observable + "-" + config_key,
            target_dir=example_path / "outputs",
        )

        reference_path = (
            example_path / "outputs" / (observable + "_" + config_key + ".npy")
        )
        test_path = example_path / "outputs" / (observable + ".npy")

        if not reference_path.exists():
            pytest.fail(f"Reference solution '{reference_path}' not found.")
        if not test_path.exists():
            pytest.fail(f"Test solution '{test_path}' not found.")

        reference = np.load(reference_path)
        test = np.load(test_path)

        assert (
            reference.shape == test.shape
        ), f"Shape mismatch for '{observable}': {reference.shape} vs {test.shape}"
        assert np.allclose(
            reference, test, rtol=1e-4, atol=1e-6
        ), f"Value mismatch for '{observable}'"


@pytest.mark.usefixtures("domain_distributed_example")
def test_distributed(domain_distributed_example: str):

    if len(REFERENCE_OBSERVABLES[domain_distributed_example]) == 0:
        pytest.skip(
            f"No reference observables defined for example '{domain_distributed_example}'"
        )

    try:
        fetch_example(domain_distributed_example)
    except Exception as e:
        pytest.fail(f"fetch-example failed: {e}")

    device_key, config_key, example_path = get_example_dir(domain_distributed_example)

    quatrex_config_path = example_path / "quatrex_config.toml"
    compute_config_path = example_path / "compute_config.toml"

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

    for observable in REFERENCE_OBSERVABLES[domain_distributed_example]:

        # fetch reference solution
        load(
            device_key + "-" + observable + "-" + config_key,
            target_dir=example_path / "outputs",
        )

        reference_path = (
            example_path / "outputs" / (observable + "_" + config_key + ".npy")
        )
        test_path = example_path / "outputs" / (observable + ".npy")

        if not reference_path.exists():
            pytest.fail(f"Reference solution '{reference_path}' not found.")
        if not test_path.exists():
            pytest.fail(f"Test solution '{test_path}' not found.")

        reference = np.load(reference_path)
        test = np.load(test_path)

        assert (
            reference.shape == test.shape
        ), f"Shape mismatch for '{observable}': {reference.shape} vs {test.shape}"
        assert np.allclose(
            reference, test, rtol=1e-4, atol=1e-6
        ), f"Value mismatch for '{observable}'"
