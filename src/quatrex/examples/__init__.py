# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Tuple

from quatrex.examples._downloader import download_and_extract

# Find repo root relative to this file.
# quatrex/src/quatrex/examples -> quatrex/, so go up three levels.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"

ALLOWED_EXAMPLES = {
    "carbon-nanotube:": [
        "hamiltonian",
        "coulomb-matrix",
        "potential",
        "grid",
        "block-sizes",
    ],
    "carbon-nanotube:dist": [
        "hamiltonian",
        "coulomb-matrix",
        "potential",
        "grid",
        "block-sizes",
    ],
    "cp2k-atomic-chain:qtbm": [
        "hamiltonian_0_0_0",
        "potential",
        "lattice",
        "overlap_0_0_0",
    ],
    "wann-si-bulk:qtbm": [
        "hamiltonian_0_-1_-1",
        "hamiltonian_0_-1_0",
        "hamiltonian_0_-1_1",
        "hamiltonian_0_0_-1",
        "hamiltonian_0_0_0",
        "hamiltonian_0_0_1",
        "hamiltonian_0_1_-1",
        "hamiltonian_0_1_0",
        "hamiltonian_0_1_1",
        "potential",
        "lattice",
    ],
}

assert len(ALLOWED_EXAMPLES) == len(set(ALLOWED_EXAMPLES.keys()))


# Load manifest containing example dataset info.
with open(files("quatrex.examples") / "_manifest.toml", "rb") as f:
    MANIFEST = tomllib.load(f)


for key, subnames in ALLOWED_EXAMPLES.items():
    key, _ = key.split(":")
    for subname in subnames:
        name = f"{key}-{subname}"
        # check exists in manifest
        if name not in MANIFEST:
            raise ValueError(f"Example '{name}' not found in manifest.")


def get_example_dir(name: str) -> Tuple[str, str, Path]:
    """Returns the folder path for a given example name."""
    device_key, config_key = name.split(":")
    folder = device_key if config_key == "" else f"{device_key}-{config_key}"
    return device_key, config_key, EXAMPLES_DIR / folder


def load(name: str, target_dir: Path, force: bool = False) -> Path:
    """Loads an example dataset by name.

    Downloads and extracts the dataset if not already present.

    Parameters
    ----------
    name : str
        Name of the example dataset to load.
    target_dir : Path
        Folder to store the dataset in.
    force : bool
        If True, forces re-download even if the dataset already exists.

    Returns
    -------
    Path
        Path to the directory containing the example dataset.

    """
    if name not in MANIFEST:
        raise ValueError(f"Unknown example: {name}")

    info = MANIFEST[name]

    target_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract(info["url"], target_dir, info.get("sha256"), force=force)

    return target_dir
