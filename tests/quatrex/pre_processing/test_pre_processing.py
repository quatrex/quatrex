# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Tests for the pre-processing step of the quatrex package."""

from contextlib import nullcontext
from pathlib import Path
from textwrap import dedent
from typing import Callable

import pytest
import tomlkit

from quatrex.cli.main import pre_process
from quatrex.config import merge_toml
from quatrex.core.config import parse_config


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


@pytest.mark.mpi_skip()
def test_fermi_level(
    example: tuple[Path, bool],
    tmp_path: Path,
    adjust_config_paths: Callable,
):
    """Tests the Fermi level computation."""
    # NOTE: We still call `pre_process` to also test the interaction
    # with the other pre-processing steps.

    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)

    config = parse_config(tmp_config_path)

    # A midgap energy must be defined for each contact in order to
    # compute the Fermi level.
    compute_fermi_level = True
    for contact in config.device.contacts:
        if contact.mid_gap_energy is None:
            compute_fermi_level = False
            break

    # Default for the examples should not be to compute the Fermi
    # level but we want to test that functionality, so we enable it
    # here.
    config = tomlkit.parse(tmp_config_path.read_text())

    patch = dedent(
        """
        [pre_process]
        compute_fermi_level = true 
        """
    )

    merge_toml(
        config,
        tomlkit.parse(patch),
    )
    config.pop("config_file", None)
    tmp_config_path.write_text(tomlkit.dumps(config))

    # Run the example using the CLI.
    with pytest.raises(ValueError) if not compute_fermi_level else nullcontext():
        pre_process(tmp_config_path, abort_on_exception=False)
