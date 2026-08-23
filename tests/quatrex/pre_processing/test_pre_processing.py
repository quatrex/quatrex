# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Tests for the pre-processing step of the quatrex package."""

from pathlib import Path
from typing import Callable

import pytest

from quatrex.cli.main import pre_process


@pytest.mark.mpi_skip()
def test_pre_process(
    example: tuple[Path, bool],
    tmp_path: Path,
    adjust_config_paths: Callable,
):
    """Tests the pre-processing step."""
    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)

    # Run the example using the CLI.
    pre_process(tmp_config_path, abort_on_exception=False)
