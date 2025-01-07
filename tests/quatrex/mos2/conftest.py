import os
from pathlib import Path
from shutil import copytree

import pytest
from qttools.utils.mpi_utils import distributed_load

input_dir = Path(__file__).parent / "inputs"
config_path = Path(__file__).parent / "config.toml"


@pytest.fixture
def datadir(tmp_path, request):
    """Copy the contents of the data directory to a temporary directory.

    Based on https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    """
    data_dir, _ = os.path.splitext(request.module.__file__)
    if os.path.isdir(data_dir):
        copytree(data_dir, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
def quatrex_config():
    from quatrex.core.quatrex_config import parse_config

    return parse_config(config_path)


@pytest.fixture
def compute_config():
    from quatrex.core.compute_config import ComputeConfig

    return ComputeConfig()


@pytest.fixture
def block_sizes():
    return distributed_load(input_dir / "block_sizes.npy")


@pytest.fixture
def electron_energies():
    return distributed_load(input_dir / "electron_energies.npy")


@pytest.fixture
def coulomb_screening_energies():
    electron_energies = distributed_load(input_dir / "electron_energies.npy")
    return electron_energies - electron_energies[0]


@pytest.fixture
def number_of_kpoints(quatrex_config):
    return quatrex_config.electron.number_of_kpoints


@pytest.fixture(params=[0, 1, 2])
def iteration(request):
    return request.param
