# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import tomllib
from importlib.resources import files
from pathlib import Path

from quatrex.examples._downloader import download_and_extract

# Find repo root relative to this file.
# quatrex/src/quatrex/examples -> quatrex/, so go up three levels.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Load manifest containing example dataset info.
with open(files("quatrex.examples") / "_manifest.toml", "rb") as f:
    MANIFEST = tomllib.load(f)


def load(name: str) -> Path:
    """Loads an example dataset by name.

    Downloads and extracts the dataset if not already present.

    Parameters
    ----------
    name : str
        Name of the example dataset to load.

    Returns
    -------
    Path
        Path to the directory containing the example dataset.

    """
    if name not in MANIFEST:
        raise ValueError(f"Unknown example: {name}")

    info = MANIFEST[name]
    target_dir = EXAMPLES_DIR / name
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract(info["url"], target_dir, info.get("sha256"))

    return target_dir
