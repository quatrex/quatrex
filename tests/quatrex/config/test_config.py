# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import pytest

from quatrex.core.config import parse_config, setup_context


def test_parse_config(example: tuple[Path, bool]):
    """Tests that the quatrex configuration can be parsed."""
    example_path, _ = example
    quatrex_config_path = example_path / "quatrex_config.toml"
    parse_config(quatrex_config_path)


def test_setup_context(
    example: tuple[Path, bool], tmp_path: Path, adjust_config_paths: Callable
):
    """Tests that the simulation context can be set up from a config."""
    example_path, dist = example

    # Distributed examples should not validate the config parsing, as
    # they may contain distributed-specific options that are not
    # supported by the single-rank parser.
    with pytest.raises(ValueError) if dist else nullcontext():
        quatrex_config_path = example_path / "quatrex_config.toml"
        tmp_config_path = tmp_path / "quatrex_config.toml"
        adjust_config_paths(quatrex_config_path, tmp_config_path)
        config = parse_config(tmp_config_path)
        setup_context(config)


@pytest.mark.mpi(min_size=3)
def test_setup_context_dist(
    example: tuple[Path, bool], tmp_path: Path, adjust_config_paths: Callable
):
    """Tests that the simulation context can be set up from a config."""
    example_path, dist = example

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)
    config = parse_config(tmp_config_path)

    # NOTE: this is expected to fail if the comm size is not a multiple
    # of the comm block size
    setup_context(config)
