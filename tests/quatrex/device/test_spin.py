# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Tests for the explicit spin treatment of the device inputs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sps

from qttools import xp
from qttools.utils.gpu_utils import get_host
from qttools.utils.hdf5_utils import save_hdf5_dict
from quatrex.core.config import DeviceConfig
from quatrex.device.base import map_potential_to_orbitals
from quatrex.device.inputs import _sum_operator, get_block_sizes, load_matrices

NUM_ORBITALS = 6

SPIN_BLOCKS = {"identity": np.eye(2), "ones": np.ones((2, 2))}


def _device_config(spin: bool, expand: bool, **kwargs) -> SimpleNamespace:
    """Creates a minimal stand-in for the device configuration."""
    num_spin_channels = 2 if spin else 1
    defaults = dict(
        spin=spin,
        expand_spinless_inputs=expand,
        num_spin_channels=num_spin_channels,
        spin_degeneracy=1 if spin else 2,
        construct_from_unit_cell=False,
        neighbor_cell_cutoff=None,
        transport_direction="c",
        block_size=2,
        num_transport_cells=1,
        num_orbitals_per_atom={"A": 2, "B": 1},
    )
    defaults.update(kwargs)
    device = SimpleNamespace(**defaults)
    device.orbitals_per_atom = (
        lambda s: device.num_orbitals_per_atom.get(s, 1) * num_spin_channels
    )
    return device


def _spinless_matrices(seed: int = 0) -> dict[tuple, sps.csr_matrix]:
    """Creates a random set of periodic images with M(-R) = M(R)^H."""
    rng = np.random.default_rng(seed)

    def random_matrix():
        return rng.random((NUM_ORBITALS, NUM_ORBITALS)) + 1j * rng.random(
            (NUM_ORBITALS, NUM_ORBITALS)
        )

    onsite = random_matrix()
    onsite = onsite + onsite.conj().T
    hopping = random_matrix()
    # NOTE: Images along 'a' are kept since transport is along 'c'.
    return {
        (0, 0, 0): sps.csr_matrix(onsite),
        (1, 0, 0): sps.csr_matrix(hopping),
        (-1, 0, 0): sps.csr_matrix(hopping.conj().T),
    }


def _write_inputs(directory: Path, matrices: dict, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_hdf5_dict(
        str(directory / f"{name}.h5"),
        {f"[{x},{y},{z}]": m for (x, y, z), m in matrices.items()},
    )


def _full(matrix_dict: dict) -> np.ndarray:
    """Hermitian sum over the loaded (upper-triangular) images."""
    return get_host(_sum_operator(matrix_dict).toarray())


@pytest.mark.parametrize("spin_block", ["identity", "ones"])
def test_expand_spinless_inputs(tmp_path: Path, spin_block: str):
    """Expanded inputs equal kron(M, spin_block) of the full operator.

    For the Coulomb matrix (all-ones spin block), this checks that the
    expansion happens before the upper triangular part is taken.
    """
    matrices = _spinless_matrices()
    _write_inputs(tmp_path, matrices, "matrix")

    config = SimpleNamespace(
        input_dir=tmp_path, device=_device_config(spin=True, expand=True)
    )
    loaded = load_matrices(
        config,
        "matrix",
        num_spatial_orbitals=NUM_ORBITALS,
        spin_block=spin_block,
    )

    reference = sum(m.toarray() for m in matrices.values())
    expected = np.kron(reference, SPIN_BLOCKS[spin_block])
    assert loaded[(0, 0, 0)].shape == (2 * NUM_ORBITALS, 2 * NUM_ORBITALS)
    assert np.allclose(_full(loaded), expected)


@pytest.mark.parametrize("spin_block", ["identity", "ones"])
def test_native_spinful_inputs_match_expanded(tmp_path: Path, spin_block: str):
    """Spinful inputs without expansion give the same as expanded ones."""
    matrices = _spinless_matrices()
    spinful = {
        coord: sps.csr_matrix(np.kron(m.toarray(), SPIN_BLOCKS[spin_block]))
        for coord, m in matrices.items()
    }
    _write_inputs(tmp_path / "spinless", matrices, "matrix")
    _write_inputs(tmp_path / "spinful", spinful, "matrix")

    expanded = load_matrices(
        SimpleNamespace(
            input_dir=tmp_path / "spinless",
            device=_device_config(spin=True, expand=True),
        ),
        "matrix",
        num_spatial_orbitals=NUM_ORBITALS,
        spin_block=spin_block,
    )
    native = load_matrices(
        SimpleNamespace(
            input_dir=tmp_path / "spinful",
            device=_device_config(spin=True, expand=False),
        ),
        "matrix",
        num_spatial_orbitals=NUM_ORBITALS,
        spin_block=spin_block,
    )

    assert expanded.keys() == native.keys()
    assert np.allclose(_full(expanded), _full(native))


@pytest.mark.parametrize(
    "spin, expand, stored_spinful, match",
    [
        (True, False, False, "expand_spinless_inputs = true"),
        (True, True, True, "expand_spinless_inputs = false"),
        (False, False, True, "spin = false"),
    ],
)
def test_input_size_mismatch(
    tmp_path: Path, spin: bool, expand: bool, stored_spinful: bool, match: str
):
    """The code stops if the input size does not match the spin mode."""
    matrices = _spinless_matrices()
    if stored_spinful:
        matrices = {
            coord: sps.csr_matrix(sps.kron(m, np.eye(2)))
            for coord, m in matrices.items()
        }
    _write_inputs(tmp_path, matrices, "matrix")

    config = SimpleNamespace(
        input_dir=tmp_path, device=_device_config(spin=spin, expand=expand)
    )
    with pytest.raises(ValueError, match=match):
        load_matrices(config, "matrix", num_spatial_orbitals=NUM_ORBITALS)


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("block_size", [2, [2, 4]])
def test_block_sizes(spin: bool, block_size: int | list[int]):
    """Block sizes are given in spatial orbitals and doubled with spin."""
    device = _device_config(spin=spin, expand=False, block_size=block_size)
    num_spin_channels = device.num_spin_channels
    block_sizes = get_block_sizes(
        SimpleNamespace(device=device), NUM_ORBITALS * num_spin_channels
    )
    expected = ([2, 2, 2] if isinstance(block_size, int) else block_size) * np.array(
        num_spin_channels
    )
    assert np.array_equal(block_sizes, expected)


def test_block_sizes_mismatch():
    device = _device_config(spin=True, expand=False, block_size=[2, 2])
    with pytest.raises(ValueError, match="spatial orbitals"):
        get_block_sizes(SimpleNamespace(device=device), 2 * NUM_ORBITALS)


@pytest.mark.parametrize("spin", [False, True])
def test_map_potential_to_orbitals(spin: bool):
    """Atom- and orbital-resolved potentials map to (spin-)orbitals."""
    device = _device_config(spin=spin, expand=False)
    species = np.array(["A", "B", "A"])
    ns = device.num_spin_channels

    atom_potential = np.array([1.0, 2.0, 3.0])
    expected = np.repeat([1.0, 1.0, 2.0, 3.0, 3.0], ns)
    result = map_potential_to_orbitals(atom_potential, species, device)
    assert np.allclose(get_host(result), expected)

    spatial_potential = np.array([1.0, 1.5, 2.0, 3.0, 3.5])
    result = map_potential_to_orbitals(spatial_potential, species, device)
    assert np.allclose(get_host(result), np.repeat(spatial_potential, ns))

    full_potential = xp.arange(5 * ns, dtype=float)
    result = map_potential_to_orbitals(full_potential, species, device)
    assert np.allclose(get_host(result), get_host(full_potential))

    with pytest.raises(ValueError):
        map_potential_to_orbitals(np.zeros(4), species, device)


def test_config_expand_requires_spin():
    """`expand_spinless_inputs` without `spin` is rejected."""
    fields = dict(
        transport_direction="c",
        geometry={"regions": []},
        construct_from_unit_cell=True,
        contacts=[{"name": "left"}, {"name": "right"}],
    )
    with pytest.raises(ValueError, match="requires `spin = true`"):
        DeviceConfig(**fields, expand_spinless_inputs=True)

    config = DeviceConfig(**fields, spin=True, num_orbitals_per_atom={"A": 3})
    assert config.num_spin_channels == 2
    assert config.spin_degeneracy == 1
    assert config.orbitals_per_atom("A") == 6
    assert config.orbitals_per_atom("X") == 2
