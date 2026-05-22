# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as comm

from quatrex.cli.main import run as cli_run


def _verify_outputs(
    output_dir: Path,
    reference_output_dir: Path,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Helper function to verify that the outputs in `output_dir` match the
    reference outputs in `reference_output_dir`."""
    test_failed = False

    # NOTE: We loop through all output files without asserting immediately. This
    # allows us to report all mismatches at once, rather than stopping at the
    # first failure.
    for output_file in output_dir.glob("*.npy"):

        reference = np.load(reference_output_dir / output_file.name)
        test = np.load(output_file)
        shape_match = reference.shape == test.shape
        if not shape_match:
            print(
                f"Shape mismatch for '{output_file.name}': {reference.shape} vs {test.shape}"
            )

        value_match = np.allclose(reference, test, rtol=rtol, atol=atol, equal_nan=True)
        if not value_match:
            print(f"Value mismatch for '{output_file.name}':")
            print(
                f"    Relative error: {np.linalg.norm(reference - test) / np.linalg.norm(reference)}"
            )
            print(f"    Absolute error: {np.linalg.norm(reference - test)}")
            print(f"    Reference norm: {np.linalg.norm(reference)}")

        test_failed |= not shape_match or not value_match

    assert not test_failed, "One or more output files did not match the reference."


# NOTE: Skip this if running in an MPI environment. These should be run
# in a single process only.
@pytest.mark.mpi_skip()
def test_single_rank(
    example: tuple[Path, bool],
    tmp_path: Path,
    adjust_config_paths: Callable,
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
    cli_run(tmp_config_path)

    output_dir = tmp_path / "outputs"
    reference_output_dir = example_path / "reference-outputs"

    _verify_outputs(output_dir, reference_output_dir)


# NOTE: The distributed test will fail if the number of ranks is not a
# multiple of three.
@pytest.mark.mpi(min_size=3)
def test_distributed(
    example: tuple[Path, bool],
    mpi_tmp_path: Path,
    adjust_config_paths: Callable,
):
    """Tests that the distributed example runs and matches reference observables."""

    example_path, distributed = example

    # Set up reference and temporary configs.
    tmp_config_path = mpi_tmp_path / "quatrex_config.toml"
    if comm.rank == 0:
        quatrex_config_path = example_path / "quatrex_config.toml"
        adjust_config_paths(quatrex_config_path, tmp_config_path)

    comm.barrier()  # Ensure all ranks wait until the config is set up.

    # Run the example using the CLI.
    cli_run(tmp_config_path)

    comm.barrier()  # Ensure all ranks wait until the run is complete.

    if comm.rank != 0:
        return  # Only rank 0 will check the outputs.

    output_dir = mpi_tmp_path / "outputs"
    reference_output_dir = example_path / "reference-outputs"

    _verify_outputs(output_dir, reference_output_dir)
