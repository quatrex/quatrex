# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests the electrostatic potential in the leads."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from qttools import xp
from qttools.comm import comm
from qttools.comm.comm import _default_config
from qttools.datastructures import DSDBCOO
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.utils.gpu_utils import get_host
from quatrex.electron.solver import (
    ElectronSolver,
    SystemMatrix,
    get_lead_potential,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
# Two unit cells in the transverse direction per layer.
NUM_LAYERS = 2
NY = 2
CELL_SIZE = BLOCK_SIZE // (NUM_LAYERS * NY)
LAYER_SIZE = NY * CELL_SIZE
SIZE = NUM_BLOCKS * BLOCK_SIZE
ENERGIES = np.array([-0.5, 0.7])


def _unit_cells(right: bool) -> dict[tuple, np.ndarray]:
    """Creates the orbital indices of the unit cells of a contact.

    The orbitals in the second transverse cell are stored in reverse
    order, i.e. the equivalent orbitals do not have the same order in
    the device.

    """
    cells = {}
    for layer, k in np.ndindex(NUM_LAYERS + 1, NY):
        inds = (layer * NY + k) * CELL_SIZE + np.arange(CELL_SIZE)
        if k == 1:
            inds = inds[::-1]
        cells[layer, k, 0] = SIZE - 1 - inds if right else inds
    return cells


def _make_contact(right: bool) -> SimpleNamespace:
    """Creates a contact with only what the OBC computation needs."""
    last = NUM_BLOCKS - 1
    return SimpleNamespace(
        name="right" if right else "left",
        diagonal_inds=(last, last) if right else (0, 0),
        upper_inds=(last, last - 1) if right else (0, 1),
        order="reverse" if right else None,
        transport_repetitions=NUM_LAYERS,
        transverse_repetition_grid=(NY, 1),
        unit_cell_orbital_indices=_unit_cells(right),
    )


LEFT, RIGHT = _make_contact(False), _make_contact(True)


def _reference_potential(
    potential: np.ndarray, contact: SimpleNamespace, mode: str
) -> np.ndarray:
    """Computes the potential in the lead in an independent way."""
    if mode == "device":
        return potential

    # View the right contact from the right.
    reference = (potential[::-1] if contact is RIGHT else potential).copy()
    layer = reference[:LAYER_SIZE]
    if mode == "average":
        new_layer = np.full(LAYER_SIZE, layer.mean())
    else:
        new_layer = np.concatenate([layer[:CELL_SIZE], layer[:CELL_SIZE][::-1]])
    reference[: 2 * BLOCK_SIZE] = np.tile(new_layer, 2 * NUM_LAYERS)
    return reference[::-1] if contact is RIGHT else reference


@pytest.fixture(autouse=True, scope="module")
def configure_comm():
    """Configures the communicator for the tests."""
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config.copy(),
        stack_comm_config=_default_config.copy(),
        global_comm_config=_default_config.copy(),
        override=True,
    )


def _to_dsdbcoo(dense: np.ndarray) -> DSDBCOO:
    """Converts a dense Hermitian matrix to a DSDBSparse."""
    return DSDBCOO.from_sparray(
        sparse.coo_matrix(dense.astype(np.complex128)),
        block_sizes=np.full(NUM_BLOCKS, BLOCK_SIZE),
        global_stack_shape=(comm.stack.size,),
    )


def _periodic_matrix(onsite: float, hopping: float) -> np.ndarray:
    """Creates a matrix that is periodic with the contact cell."""
    matrix = onsite * np.eye(SIZE)
    for i in range(1, LAYER_SIZE + 1):
        matrix += hopping / i * (np.eye(SIZE, k=i) + np.eye(SIZE, k=-i))
    return matrix


@pytest.fixture
def potential() -> np.ndarray:
    """A potential that varies everywhere."""
    return np.random.default_rng(0).uniform(-0.2, 0.2, SIZE)


@pytest.fixture
def hamiltonian() -> DSDBCOO:
    return _to_dsdbcoo(_periodic_matrix(0.0, -1.0))


@pytest.fixture
def overlap() -> DSDBCOO:
    return _to_dsdbcoo(_periodic_matrix(1.0, 0.1))


def _make_solver(hamiltonian, overlap, potential, mode: str) -> ElectronSolver:
    """Creates a solver with only what the OBC computation needs."""
    solver = ElectronSolver.__new__(ElectronSolver)
    solver.lead_potential = mode
    solver.eta_obc = 0.0
    solver.compute_overlap_density = False
    solver.obc_blocks = OBCBlocks(num_blocks=NUM_BLOCKS)
    solver.device = SimpleNamespace(hamiltonians=hamiltonian, overlap_matrices=overlap)
    solver.system_matrix = SystemMatrix(
        stack_shape=(ENERGIES.size,),
        stack_index=(...,),
        energies=xp.asarray(ENERGIES) + 1e-3j,
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=xp.asarray(potential),
    )
    return solver


def _lead_blocks(solver: ElectronSolver, contact) -> list[np.ndarray]:
    """Returns the blocks that the OBC solver receives for a contact."""
    captured = []

    def obc_solver(blocks, contact):
        captured.extend(get_host(block).copy() for block in blocks)
        return (xp.zeros_like(blocks[1]),)

    contact = SimpleNamespace(**vars(contact), obc_solvers={"electron": obc_solver})
    solver._compute_contact_obc(contact, "test", xp.ones(ENERGIES.size))
    return captured


@pytest.mark.parametrize("contact", [LEFT, RIGHT], ids=["left", "right"])
@pytest.mark.parametrize("mode", ["device", "unit-cell", "average"])
@pytest.mark.parametrize("global_offset", [0, BLOCK_SIZE])
def test_get_lead_potential(
    contact: SimpleNamespace, mode: str, global_offset: int, potential: np.ndarray
):
    """Tests the potential, also with a domain decomposition."""
    if contact is LEFT and global_offset > 0:
        pytest.skip("The left contact is always on the first rank.")

    local_potential = xp.asarray(potential[global_offset:].copy())
    first_block = global_offset // BLOCK_SIZE
    local_contact = SimpleNamespace(**vars(contact))
    local_contact.diagonal_inds = tuple(i - first_block for i in contact.diagonal_inds)
    local_contact.upper_inds = tuple(i - first_block for i in contact.upper_inds)

    lead_potential = get_lead_potential(
        local_potential,
        local_contact,
        block_offsets=np.arange(NUM_BLOCKS - first_block + 1) * BLOCK_SIZE,
        global_offset=global_offset,
        mode=mode,
    )

    reference = _reference_potential(potential, contact, mode)
    assert np.allclose(get_host(lead_potential), reference[global_offset:])
    # The original potential must not be changed.
    assert np.array_equal(get_host(local_potential), potential[global_offset:])


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("contact", [LEFT, RIGHT], ids=["left", "right"])
@pytest.mark.parametrize("mode", ["unit-cell", "average"])
def test_obc_with_lead_potential(
    contact: SimpleNamespace,
    mode: str,
    potential: np.ndarray,
    hamiltonian: DSDBCOO,
    overlap: DSDBCOO,
):
    """Tests that only the leads see the potential of the lead."""
    solver = _make_solver(hamiltonian, overlap, potential, mode)
    lead_blocks = _lead_blocks(solver, contact)

    # The lead is the same as that of a device at the lead potential.
    references = _lead_blocks(
        _make_solver(
            hamiltonian,
            overlap,
            _reference_potential(potential, contact, mode).copy(),
            "device",
        ),
        contact,
    )
    for block, reference in zip(lead_blocks, references, strict=True):
        assert np.allclose(block, reference)

    # By default, the lead is different.
    default_solver = _make_solver(hamiltonian, overlap, potential, "device")
    assert not np.allclose(_lead_blocks(default_solver, contact)[1], references[1])

    # The system matrix of the device is not changed, also after the
    # blocks have been accessed for the OBC.
    for i in range(NUM_BLOCKS):
        for j in range(max(0, i - 1), min(NUM_BLOCKS, i + 2)):
            assert np.array_equal(
                get_host(solver.system_matrix.blocks[i, j]),
                get_host(default_solver.system_matrix.blocks[i, j]),
            )


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("mode", ["unit-cell", "average"])
def test_constant_potential_is_unchanged(
    mode: str, hamiltonian: DSDBCOO, overlap: DSDBCOO
):
    """Tests that all modes agree for a constant potential."""
    potential = np.full(SIZE, 0.13)
    for contact in (LEFT, RIGHT):
        blocks, references = (
            _lead_blocks(_make_solver(hamiltonian, overlap, potential, m), contact)
            for m in (mode, "device")
        )
        for block, reference in zip(blocks, references, strict=True):
            assert np.allclose(block, reference, rtol=0, atol=1e-14)
