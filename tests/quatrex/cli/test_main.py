# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
import subprocess

import pytest


@pytest.mark.skipif(
    "SLURM_JOB_ID" in os.environ,
    reason="This test is not intended to be run in a Slurm environment.",
)
def test_help_cli():
    """Tests the help CLI."""
    subprocess.run(["quatrex", "--help"], check=True)


@pytest.mark.skipif(
    "SLURM_JOB_ID" in os.environ,
    reason="This test is not intended to be run in a Slurm environment.",
)
def test_version_cli():
    """Tests the version CLI."""
    subprocess.run(["quatrex", "--version"], check=True)
