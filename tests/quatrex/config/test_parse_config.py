# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from pathlib import Path

from quatrex.core.config import parse_config


def test_parse_config(example: tuple[Path, bool]):
    """Tests that the quatrex configuration can be parsed."""
    example_path, __ = example
    quatrex_config = example_path / "quatrex_config.toml"
    parse_config(quatrex_config)
