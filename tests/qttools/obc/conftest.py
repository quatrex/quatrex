# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.nevp import NEVP, Beyn, Full
from quatrex.cli.main import fetch_example
from quatrex.core.compute_config import CommConfig
from quatrex.core.quatrex_config import parse_config as parse_quatrex_config
from quatrex.electron.solver import ElectronSolver
from quatrex.examples import get_example_dir

NEVP_SOLVERS = [
    pytest.param(
        Beyn(r_o=10, r_i=0.99, m_0=32, num_quad_points=15, use_qr=False), id="Beyn"
    ),
    pytest.param(
        Beyn(r_o=10, r_i=0.99, m_0=32, num_quad_points=15, use_qr=True), id="Beyn"
    ),
    pytest.param(Full(), id="Full"),
]

X_II_FORMULAS = ["self-energy", "direct"]

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


@pytest.fixture(params=X_II_FORMULAS)
def x_ii_formula(request: pytest.FixtureRequest) -> str:
    """Returns a NEVP solver."""
    return request.param


@pytest.fixture(params=NEVP_SOLVERS)
def nevp(request: pytest.FixtureRequest) -> NEVP:
    """Returns a NEVP solver."""
    return request.param


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


ENERGIES = [[-10, -5, 0]]


@pytest.fixture(params=ENERGIES, autouse=True, scope="session")
def a_xx(request: pytest.FixtureRequest) -> tuple[NDArray, NDArray, NDArray]:
    """Returns some CNT boundary blocks."""

    energies = xp.array(request.param)

    try:
        fetch_example("carbon-nanotube:")
    except Exception as e:
        pytest.fail(f"fetch-example failed: {e}")

    _, _, example_path = get_example_dir("carbon-nanotube:")

    quatrex_config_path = example_path / "quatrex_config.toml"
    quatrex_config = parse_quatrex_config(quatrex_config_path)

    # hack to configure the communicator
    CommConfig()

    hamiltonian_sparray, block_sizes = ElectronSolver.load_hamiltonian(quatrex_config)
    hamiltonian_sparray = hamiltonian_sparray.toarray()

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


@pytest.fixture(params=CONTACTS, autouse=True)
def contact(request: pytest.FixtureRequest) -> str:
    """Returns a contact."""
    return request.param
