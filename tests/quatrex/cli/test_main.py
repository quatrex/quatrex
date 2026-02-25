# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import subprocess


def test_help_cli():
    """Tests the help CLI."""
    subprocess.run(["quatrex", "--help"], check=True)


def test_version_cli():
    """Tests the version CLI."""
    subprocess.run(["quatrex", "--version"], check=True)

# add if you want to run the tests directly as python file and debug it
# if __name__ == "__main__":
#     pytest.main([__file__])