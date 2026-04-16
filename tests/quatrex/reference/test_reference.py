# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pytest


# NOTE: Skip this if running in an MPI environment. These should be run
# in a single process only.
@pytest.mark.mpi_skip()
def test_single_rank(
    example: tuple[Path, bool], tmp_path: Path, adjust_config_paths: Callable
):
    """Tests that the example runs and matches reference observables."""

    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)

    # Run the example using the CLI.
    from quatrex.cli.main import run

    run(tmp_config_path)

    output_dir = tmp_path / "outputs"
    reference_output_dir = example_path / "reference-outputs"

    for output_file in output_dir.glob("*.npy"):

        reference = np.load(reference_output_dir / output_file.name)
        test = np.load(output_file)
        assert (
            reference.shape == test.shape
        ), f"Shape mismatch for '{output_file.name}': {reference.shape} vs {test.shape}"
        assert np.allclose(
            reference, test, rtol=1e-5, atol=1e-6
        ), f"Value mismatch for '{output_file.name}'"


# NOTE: Skip this if already running in an MPI environment, to avoid
# nested MPI runs.
@pytest.mark.mpi_skip()
def test_distributed(
    example: tuple[Path, bool], tmp_path: Path, adjust_config_paths: Callable
):
    """Tests that the distributed example runs and matches reference observables."""

    example_path, distributed = example

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)

    # TODO: This is not compatible with SLURM yet.
    # This needs to be adapted when running on a cluster.
    args = ["mpiexec", "-n", "6", "quatrex", "run", str(tmp_config_path)]

    subprocess.run(args, check=True, stdout=None, stderr=None)

    output_dir = tmp_path / "outputs"
    reference_output_dir = example_path / "reference-outputs"

    for output_file in output_dir.glob("*.npy"):

        reference = np.load(reference_output_dir / output_file.name)
        test = np.load(output_file)
        assert (
            reference.shape == test.shape
        ), f"Shape mismatch for '{output_file.name}': {reference.shape} vs {test.shape}"
        assert np.allclose(
            reference, test, rtol=1e-3, atol=1e-4, equal_nan=True
        ), f"Value mismatch for '{output_file.name}'"
