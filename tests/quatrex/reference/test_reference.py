# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import subprocess
from pathlib import Path

import numpy as np
import pytest


def test_single_rank(example: tuple[Path, bool]):
    """Tests that the example runs and matches reference observables."""

    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    quatrex_config_path = example_path / "quatrex_config.toml"

    from quatrex.cli.main import run

    run(quatrex_config_path)

    output_dir = example_path / "outputs"
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


def test_distributed(example: tuple[Path, bool]):
    """Tests that the distributed example runs and matches reference observables."""

    example_path, distributed = example

    quatrex_config_path = example_path / "quatrex_config.toml"

    # TODO: This is not compatible with SLURM yet.
    # This needs to be adapted when running on a cluster.
    args = ["mpiexec", "-n", "6", "quatrex", "run", str(quatrex_config_path)]

    subprocess.run(args, check=True, stdout=None, stderr=None)

    output_dir = example_path / "outputs"
    reference_output_dir = example_path / "reference-outputs"

    for output_file in output_dir.glob("*.npy"):

        reference = np.load(reference_output_dir / output_file.name)
        test = np.load(output_file)
        assert (
            reference.shape == test.shape
        ), f"Shape mismatch for '{output_file.name}': {reference.shape} vs {test.shape}"
        assert np.allclose(
            reference, test, rtol=1e-3, atol=1e-4
        ), f"Value mismatch for '{output_file.name}'"
