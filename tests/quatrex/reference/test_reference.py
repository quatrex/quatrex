# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest


def _adjust_config_paths(quatrex_config_path: Path, tmp_config_path: Path):
    """Adjusts the input directory path in the temporary config to point
    to the example's input directory.

    Parameters
    ----------
    quatrex_config_path : Path
        The path to the original config file in the example directory.
    tmp_config_path : Path
        The path to the temporary config file that will be used for
        testing.

    """
    # Read the original config to find the input directory.
    with open(quatrex_config_path, "rb") as f:
        config = tomllib.load(f)

    config_text = quatrex_config_path.read_text()

    input_dir = config.get("input_dir")
    if input_dir is None:
        # If the input directory is not specified, we assume it is
        # "inputs" relative to the config file.
        abs_input_dir = str((quatrex_config_path.parent / "inputs").resolve())
        config_text = f'input_dir = "{abs_input_dir}"\n' + config_text

    elif not Path(input_dir).is_absolute():
        abs_input_dir = str((quatrex_config_path.parent / input_dir).resolve())
        config_text = config_text.replace(input_dir, abs_input_dir)

    # Copy the config and replace the input directory with the absolute path.
    tmp_config_path.write_text(config_text)


# NOTE: Skip this if running in an MPI environment. These should be run
# in a single process only.
@pytest.mark.mpi_skip()
def test_single_rank(example: tuple[Path, bool], tmp_path: Path):
    """Tests that the example runs and matches reference observables."""

    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    _adjust_config_paths(quatrex_config_path, tmp_config_path)

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
def test_distributed(example: tuple[Path, bool], tmp_path: Path):
    """Tests that the distributed example runs and matches reference observables."""

    example_path, distributed = example

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    _adjust_config_paths(quatrex_config_path, tmp_config_path)

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
            reference, test, rtol=1e-3, atol=1e-4
        ), f"Value mismatch for '{output_file.name}'"
