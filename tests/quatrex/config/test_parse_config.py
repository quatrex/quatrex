# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.
from contextlib import nullcontext
from pathlib import Path

import pytest

from quatrex.core.compute_config import parse_config as parse_compute_config
from quatrex.core.quatrex_config import parse_config as parse_quatrex_config


def test_parse_quatrex_config(example: tuple[Path, bool]):
    """Tests that the quatrex configuration can be parsed."""
    example_path, __ = example
    quatrex_config = example_path / "quatrex_config.toml"
    parse_quatrex_config(quatrex_config)


def test_parse_compute_config(example: tuple[Path, bool]):
    """Tests that the compute configuration can be parsed."""
    example_path, distributed = example
    compute_config_path = example_path / "compute_config.toml"
    if not compute_config_path.exists():
        pytest.skip("No compute config to parse.")
    with nullcontext() if not distributed else pytest.raises(ValueError):
        # NOTE: We expect that the distributed configs will raise a
        # ValueError in the QuatrexCommunicator when parsed without
        # initializing MPI.
        parse_compute_config(compute_config_path)
