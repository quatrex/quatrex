# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from contextlib import nullcontext
from pathlib import Path

import pytest

from quatrex.core.config import parse_config


def test_parse_config(example: tuple[Path, bool]):
    """Tests that the quatrex configuration can be parsed."""
    example_path, dist = example

    # Distributed examples should not validate the config parsing,
    # as they may contain distributed-specific options that are not
    # supported by the single-rank parser.
    with pytest.raises(ValueError) if dist else nullcontext():
        quatrex_config = example_path / "quatrex_config.toml"
        parse_config(quatrex_config)
