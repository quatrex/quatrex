# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import numpy as np
import pytest

from qttools import NDArray, sparse, xp
from qttools.nevp import NEVP, Beyn, Full
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.config import parse_config

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"
CARBON_NANOTUBE_EXAMPLE = EXAMPLES_DIR / "w90" / "carbon-nanotube" / "gw"

BLOCK_SIZE = [
    pytest.param(21, id="21x21"),
    pytest.param(18, id="18x18"),
]

BLOCK_SECTIONS = [
    pytest.param(1, id="no-subblocks"),
    pytest.param(3, id="three-subblocks"),
]

BATCH_SIZE = [
    pytest.param(1, id="single-batch"),
    pytest.param(3, id="three-batches"),
]

CONTACTS = ["left", "right"]

NEVP_SOLVERS = [
    pytest.param(
        Beyn(r_o=10, r_i=0.99, m_0=32, num_quad_points=15, use_qr=False), id="Beyn"
    ),
    pytest.param(
        Beyn(r_o=10, r_i=0.99, m_0=32, num_quad_points=15, use_qr=True), id="Beyn"
    ),
    pytest.param(Full(), id="Full"),
]

ENERGIES = [[-10, -5, 0]]


@pytest.fixture(params=BLOCK_SIZE)
def block_size(request: pytest.FixtureRequest) -> int:
    """Returns the block size."""
    return request.param


@pytest.fixture(params=BLOCK_SECTIONS)
def block_sections(request: pytest.FixtureRequest) -> int:
    """Returns the number of block sections."""
    return request.param


@pytest.fixture(params=BATCH_SIZE)
def batch_size(request: pytest.FixtureRequest) -> int:
    """Returns the block size."""
    return request.param


@pytest.fixture(params=CONTACTS, autouse=True)
def contact(request: pytest.FixtureRequest) -> str:
    """Returns a contact."""
    return request.param


@pytest.fixture(params=ENERGIES, autouse=True, scope="session")
def a_xx(request: pytest.FixtureRequest) -> tuple[NDArray, NDArray, NDArray]:
    """Returns some boundary blocks for the carbon nanotube example."""

    energies = xp.array(request.param)

    quatrex_config_path = CARBON_NANOTUBE_EXAMPLE / "quatrex_config.toml"
    config = parse_config(quatrex_config_path)

    hamiltonian_sparray = distributed_load(config.input_dir / "hamiltonian.mat")
    if (0, 0, 0) not in hamiltonian_sparray.keys():
        raise ValueError(
            f"Expected to find a key [0,0,0] in the matrix file, but it was not found. "
            f"Available keys: {list(hamiltonian_sparray.keys())}"
        )
    hamiltonian_sparray = hamiltonian_sparray[(0, 0, 0)]
    hamiltonian_sparray = sparse.coo_matrix(hamiltonian_sparray).astype(xp.complex128)

    block_sizes = get_host(np.loadtxt(config.input_dir / "block_sizes.txt", dtype=int))

    hamiltonian_sparray = xp.asarray(hamiltonian_sparray.toarray())

    block_size = block_sizes[0]

    H00 = hamiltonian_sparray[:block_size, :block_size]
    H01 = hamiltonian_sparray[:block_size, block_size : 2 * block_size]
    H10 = H01.conj().T

    # this examples does not have a overlap matrix
    M00 = xp.eye(block_size, dtype=xp.complex128)[xp.newaxis, :, :]
    M00 = M00 * (energies[:, xp.newaxis, xp.newaxis] + 1j * 1e-7)

    M00 -= H00
    M01 = xp.repeat(-H01[xp.newaxis, :, :], len(energies), axis=0)
    M10 = xp.repeat(-H10[xp.newaxis, :, :], len(energies), axis=0)

    return M10, M00, M01


@pytest.fixture(params=NEVP_SOLVERS)
def nevp(request: pytest.FixtureRequest) -> NEVP:
    """Returns a NEVP solver."""
    return request.param
