# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Tests the orbit packing with tight-binding bulk silicon.

The transport direction is [100] with conventional (cubic) cells as
transport cells. Both transverse directions are periodic.

"""

from dataclasses import dataclass
from functools import cache

import ase
import ase.build
import numpy as np
import pytest

pytest.importorskip("spglib")

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_host
from quatrex.core.utils import compute_sparsity_pattern
from quatrex.grid.kpoints import (
    DeviceSymmetry,
    KGrid,
    SymmetryOperation,
    collect_symmetry_operations,
)
from quatrex.grid.orbit_packing import (
    OrbitPacking,
    create_orbital_operation,
    filter_packable_operations,
)

LATTICE_CONSTANT = 5.431  # Angstrom
ORBITAL_LABELS = ["s", "px", "py", "pz", "s*"]
NUM_ORBITALS = len(ORBITAL_LABELS)

# sp3s* parameters (eV) of P. Vogl et al., J. Phys. Chem. Solids 44, 365 (1983).
E_S, E_P, E_S_STAR = -4.2, 1.715, 6.685
V_SS, V_XX, V_XY, V_SP, V_S_STAR_P = -8.3, 1.715, 4.575, 5.7292, 5.3749

NUM_BLOCKS = 2
ATOMS_PER_BLOCK = 8
BLOCK_SIZE = ATOMS_PER_BLOCK * NUM_ORBITALS
CUTOFF = 4.0  # Angstrom

# The number of points and whether the k-point grid is gamma-centered.
GRIDS = {
    "2x2-mp": ((1, 2, 2), False),
    "3x3-mp": ((1, 3, 3), False),
    "4x4-mp": ((1, 4, 4), False),
    "4x4-gamma": ((1, 4, 4), True),
    "4x2-mp": ((1, 4, 2), False),
}


def _hopping(direction: NDArray) -> NDArray:
    """Nearest-neighbor hopping <0|H|d>, in the order of `ORBITAL_LABELS`."""
    g = np.sign(direction)
    h = np.zeros((NUM_ORBITALS, NUM_ORBITALS))
    h[0, 0] = V_SS
    h[0, 1:4] = V_SP * g
    h[1:4, 0] = -V_SP * g
    h[4, 1:4] = V_S_STAR_P * g
    h[1:4, 4] = -V_S_STAR_P * g
    h[1:4, 1:4] = V_XY * np.outer(g, g)
    h[(1, 2, 3), (1, 2, 3)] = V_XX
    return h / 4


def _silicon(num_cells: int) -> ase.Atoms:
    """Silicon with conventional cells repeated along the transport axis."""
    unit_cell = ase.build.bulk("Si", "diamond", a=LATTICE_CONSTANT, cubic=True)
    return unit_cell.repeat((num_cells, 1, 1))


def _images(distances_to: NDArray, length: float) -> dict[tuple, NDArray]:
    """Displacements r_j + R - r_i to the surrounding transverse cells."""
    displacements = distances_to[None, :] - distances_to[:, None]
    return {
        (ry, rz): displacements + np.array([0.0, ry, rz]) * length
        for ry in (-1, 0, 1)
        for rz in (-1, 0, 1)
    }


@cache
def _hamiltonian_cells(num_cells: int) -> dict[tuple, NDArray]:
    """Hamiltonian H_ij(R) of an open device for all transverse cells R."""
    positions = _silicon(num_cells).positions
    num_atoms = len(positions)
    bond_length = np.sqrt(3) / 4 * LATTICE_CONSTANT

    hamiltonian = {}
    for cell, displacements in _images(positions, LATTICE_CONSTANT).items():
        h = np.zeros((num_atoms, NUM_ORBITALS, num_atoms, NUM_ORBITALS))
        distances = np.linalg.norm(displacements, axis=-1)
        for a, b in np.argwhere(np.abs(distances - bond_length) < 1e-3):
            h[a, :, b, :] = _hopping(displacements[a, b])
        if cell == (0, 0):
            for a in range(num_atoms):
                h[a, :, a, :] = np.diag([E_S, E_P, E_P, E_P, E_S_STAR])
        hamiltonian[cell] = h.reshape(num_atoms * NUM_ORBITALS, -1)
    return hamiltonian


@cache
def _coulomb_cells() -> dict[tuple, NDArray]:
    """Softened, short-ranged interaction V_ij(R), different for s and p."""
    strength = np.array([1.0, 0.8, 0.8, 0.8, 0.9])
    coulomb = {}
    for cell, displacements in _images(
        _silicon(NUM_BLOCKS).positions, LATTICE_CONSTANT
    ).items():
        distances = np.linalg.norm(displacements, axis=-1)
        # NOTE: Smooth and negligible beyond the images that are summed.
        v = np.exp(-((distances / 1.5) ** 2)) / (1.0 + distances)
        v[distances > LATTICE_CONSTANT] = 0.0
        coulomb[cell] = np.kron(v, np.outer(strength, strength))
    return coulomb


def _bloch_sum(cells: dict[tuple, NDArray], kpoint: NDArray) -> NDArray:
    """Sums up the transverse cells in cell gauge."""
    return sum(
        np.exp(2j * np.pi * (ry * kpoint[1] + rz * kpoint[2])) * matrix
        for (ry, rz), matrix in cells.items()
    )


def _device_greens_functions(
    kpoints: NDArray, energies: NDArray, fermi_levels: tuple[float, float]
) -> tuple[NDArray, NDArray]:
    """Lesser/greater Green's functions of the biased device."""
    size = NUM_BLOCKS * BLOCK_SIZE
    cells = _hamiltonian_cells(NUM_BLOCKS + 2)
    lead, device = slice(0, BLOCK_SIZE), slice(BLOCK_SIZE, BLOCK_SIZE + size)

    g_lesser = np.zeros((len(energies), len(kpoints), size, size), dtype=complex)
    g_greater = np.zeros_like(g_lesser)
    for k, kpoint in enumerate(kpoints):
        h = _bloch_sum(cells, kpoint)
        h_00 = h[lead, lead]
        h_01 = h[lead, BLOCK_SIZE : 2 * BLOCK_SIZE]
        h_10 = h_01.conj().T
        h_device = h[device, device]
        for e, energy in enumerate(energies):
            identity = np.eye(BLOCK_SIZE)
            z = (energy + 0.2j) * identity
            # Sancho-Rubio for both surfaces.
            alpha, beta, eps = h_01, h_10, h_00
            eps_left, eps_right = h_00, h_00
            for __ in range(50):
                g = np.linalg.inv(z - eps)
                eps_left = eps_left + beta @ g @ alpha
                eps_right = eps_right + alpha @ g @ beta
                eps = eps + alpha @ g @ beta + beta @ g @ alpha
                alpha, beta = alpha @ g @ alpha, beta @ g @ beta
                if np.abs(alpha).max() < 1e-14:
                    break
            sigmas = np.zeros((2, size, size), dtype=complex)
            sigmas[0, :BLOCK_SIZE, :BLOCK_SIZE] = (
                h_10 @ np.linalg.inv(z - eps_left) @ h_01
            )
            sigmas[1, -BLOCK_SIZE:, -BLOCK_SIZE:] = (
                h_01 @ np.linalg.inv(z - eps_right) @ h_10
            )
            g_retarded = np.linalg.inv(
                energy * np.eye(size) - h_device - sigmas.sum(axis=0)
            )
            gammas = 1j * (sigmas - sigmas.conj().swapaxes(-1, -2))
            for gamma, fermi_level in zip(gammas, fermi_levels):
                f = 1 / (1 + np.exp((energy - fermi_level) / 0.1))
                spectral = g_retarded @ gamma @ g_retarded.conj().T
                g_lesser[e, k] += 1j * f * spectral
                g_greater[e, k] -= 1j * (1 - f) * spectral

    return g_lesser, g_greater


def _correlate(a: NDArray, b: NDArray) -> NDArray:
    """sum_E a(E) b(E - w) for w = -(ne - 1), ..., ne - 1."""
    n = 2 * len(a) - 1
    return np.fft.ifft(
        np.fft.fft(a, n, axis=0) * np.fft.fft(b[::-1], n, axis=0), axis=0
    )


def _convolve(a: NDArray, b: NDArray) -> NDArray:
    """sum_w a(E - w) b(w)."""
    n = 2 * len(a) - 1
    return np.fft.ifft(np.fft.fft(a, n, axis=0) * np.fft.fft(b, n, axis=0), axis=0)


def _relative_error(a: NDArray, b: NDArray) -> float:
    """Largest deviation of a from b."""
    return float(np.abs(get_host(a) - b).max() / np.abs(b).max())


@dataclass
class System:
    """Everything needed to compare the IBZ to the full BZ."""

    symmetry: DeviceSymmetry
    kgrid: KGrid
    qgrid: KGrid
    operations: list
    rows: NDArray
    cols: NDArray
    energies: NDArray
    g_full: tuple[NDArray, NDArray]  # Directly solved on the full grid.

    def packing(self, storage: str = "upper") -> OrbitPacking:
        rows, cols = self.pattern(storage)
        return OrbitPacking(rows, cols, self.operations, self.kgrid, self.qgrid)

    def pattern(self, storage: str = "upper") -> tuple[NDArray, NDArray]:
        mask = self.rows <= self.cols if storage == "upper" else slice(None)
        return self.rows[mask], self.cols[mask]

    @property
    def g_irreducible(self) -> tuple[NDArray, NDArray]:
        return tuple(g[:, self.kgrid._reduce_mapping] for g in self.g_full)

    def q_difference(self, k: int) -> NDArray:
        """Flat index of k - q for all q."""
        cells = np.moveaxis(np.indices(self.kgrid.size), 0, -1).reshape(-1, 3)
        return np.ravel_multi_index(
            tuple(((cells[k] - cells) % self.kgrid.size).T), self.kgrid.size
        )

    def polarization(self) -> tuple[NDArray, NDArray]:
        """Full-BZ reference for P lesser/greater, for w >= 0."""
        g_lesser, g_greater = self.g_full
        ne, nk = g_lesser.shape[:2]
        p_lesser, p_greater = np.zeros_like(g_lesser), np.zeros_like(g_lesser)
        for k in range(nk):
            # This runs over k' = k - q, i.e. over q for fixed k.
            kmq = self.q_difference(k)
            for p, g_a, g_b in (
                (p_lesser, g_lesser, g_greater),
                (p_greater, g_greater, g_lesser),
            ):
                p += _correlate(g_a[:, k, None], g_b[:, kmq].swapaxes(-1, -2))[ne - 1 :]
        return -1j * p_lesser / nk, -1j * p_greater / nk

    def screened_interaction(
        self, grid_points: NDArray, p_lesser: NDArray, p_greater: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Solves for W lesser/greater."""
        mask = np.zeros(p_lesser.shape[-2:])
        mask[self.rows, self.cols] = 1.0
        cells = _coulomb_cells()
        v = np.stack([_bloch_sum(cells, q) for q in grid_points]) * mask
        p_lesser, p_greater = p_lesser * mask, p_greater * mask
        p_retarded = (p_greater - p_lesser) / 2
        w_retarded = np.linalg.solve(np.eye(v.shape[-1]) - v @ p_retarded, v)
        w_advanced = w_retarded.conj().swapaxes(-1, -2)
        return w_retarded @ p_lesser @ w_advanced, w_retarded @ p_greater @ w_advanced

    def self_energy(self, g: NDArray, w: NDArray) -> NDArray:
        """Full-BZ reference for sum_q,w G(k - q, E - w) W(q, w)."""
        ne, nk = g.shape[:2]
        sigma = np.zeros((2 * ne - 1,) + g.shape[1:], dtype=complex)
        for k in range(nk):
            sigma[:, k] = _convolve(g[:, self.q_difference(k)], w).sum(axis=1)
        return sigma / nk


@cache
def _create_system(size: tuple, gamma_centered: bool) -> System:
    """Creates the system. This is cached as the reference is expensive."""
    structure = _silicon(NUM_BLOCKS)
    symmetry = DeviceSymmetry(
        structure.get_scaled_positions(wrap=False),
        structure.cell.array,
        ORBITAL_LABELS,
    )
    operations = filter_packable_operations(
        symmetry, collect_symmetry_operations(structure, 0, time_reversal=False)
    )
    # The two grids have to share the operations.
    kwargs = {"size": size, "symmetry_reduction": True, "time_reversal": False}
    operations = KGrid(
        gamma_centered=gamma_centered, operations=operations, **kwargs
    ).operations
    qgrid = KGrid(gamma_centered=True, operations=operations, **kwargs)
    operations = qgrid.operations
    kgrid = KGrid(gamma_centered=gamma_centered, operations=operations, **kwargs)

    positions = np.repeat(structure.positions, NUM_ORBITALS, axis=0)
    pattern = compute_sparsity_pattern(xp.asarray(positions), CUTOFF)

    energies = np.linspace(-3.0, 4.0, 6)
    return System(
        symmetry=symmetry,
        kgrid=kgrid,
        qgrid=qgrid,
        operations=[create_orbital_operation(symmetry, op) for op in operations],
        rows=get_host(pattern.row),
        cols=get_host(pattern.col),
        energies=energies,
        g_full=_device_greens_functions(kgrid.full_points, energies, (1.5, -0.5)),
    )


@pytest.fixture(params=GRIDS.values(), ids=GRIDS.keys())
def system(request: pytest.FixtureRequest) -> System:
    return _create_system(*request.param)


# ----------------------------------------------------------------------


def test_silicon_band_gap():
    """The model gives the published gap of silicon."""
    neighbors = (
        LATTICE_CONSTANT
        / 4
        * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    )
    onsite = np.diag([E_S, E_P, E_P, E_P, E_S_STAR])

    def bands(k):
        h_ac = sum(np.exp(1j * d @ k) * _hopping(d) for d in neighbors)
        return np.linalg.eigvalsh(np.block([[onsite, h_ac], [h_ac.conj().T, onsite]]))

    path = np.linspace(0, 1, 101)[:, None] * [2 * np.pi / LATTICE_CONSTANT, 0, 0]
    energies = np.array([bands(k) for k in path])
    assert energies[0, 3] == pytest.approx(0.0, abs=1e-2)
    assert energies[:, 4].min() - energies[:, 3].max() == pytest.approx(1.17, abs=0.02)


def test_operations():
    """There are four point operations, with and without centering."""
    system = _create_system((1, 4, 4), False)
    assert len(system.operations) == 8
    identity = np.eye(3)
    centering = [
        op
        for op in system.operations
        if np.array_equal(op.rotation, identity) and np.any(op.translation)
    ]
    # (0, 1/2, 1/2) keeps every k-point and only reduces the (i, j) pairs.
    assert len(centering) == 1
    assert any(np.any(op.shifts) for op in system.operations)


@pytest.mark.parametrize("cells", ["hamiltonian", "coulomb"])
def test_matrices_are_covariant(cells: str):
    """H transforms with the signs of the orbitals, V without."""
    system = _create_system((1, 4, 4), False)
    matrices = (
        _hamiltonian_cells(NUM_BLOCKS) if cells == "hamiltonian" else _coulomb_cells()
    )
    kpoint = np.array([0.0, 0.13, -0.29])
    for op in system.operations:
        rotated = np.linalg.inv(op.rotation).T @ kpoint
        phases = np.exp(2j * np.pi * (op.shifts @ rotated))
        signs = op.signs if cells == "hamiltonian" else 1
        factors = signs * phases.conj()
        matrix = _bloch_sum(matrices, kpoint)
        expected = _bloch_sum(matrices, rotated)[np.ix_(op.targets, op.targets)]
        transformed = factors[:, None] * matrix * factors.conj()[None, :]
        np.testing.assert_allclose(transformed, expected, atol=1e-12)


def test_greens_function_is_covariant(system: System):
    """The irreducible points give the directly solved full grid."""
    for g, g_full in zip(system.g_irreducible, system.g_full):
        expanded = system.kgrid.expand_matrices(g, system.symmetry)
        assert _relative_error(expanded, g_full) < 1e-10


def test_bias_breaks_time_reversal():
    """G(-k) = G(k)^T does not hold, hence time reversal is rejected."""
    kpoint = np.array([[0.0, 0.1, 0.2]])
    g_lesser, __ = _device_greens_functions(
        np.vstack([kpoint, -kpoint]), np.array([1.0]), (1.5, -0.5)
    )
    assert _relative_error(g_lesser[0, 0].T, g_lesser[0, 1]) > 1e-2


@pytest.mark.parametrize("storage", ["upper", "full"])
def test_pack_unpack(system: System, storage: str):
    """Packing gives G(R), unpacking gives G(k)."""
    packing = system.packing(storage)
    rows, cols = system.pattern(storage)
    for g, g_full in zip(system.g_irreducible, system.g_full):
        data = g[..., rows, cols]
        packed = packing.pack(xp.asarray(data), True, "skew-hermitian")
        phases = np.exp(-2j * np.pi * (system.kgrid.full_points @ packing.cells.T))
        reference = np.einsum(
            "kr,ekr->er",
            phases[:, packing.rep_cells],
            g_full[..., packing.rep_rows, packing.rep_cols],
        )
        assert _relative_error(packed, reference / packing.num_cells) < 1e-10
        unpacked = packing.unpack(packed, True, "skew-hermitian")
        assert _relative_error(unpacked, data) < 1e-10


@pytest.mark.parametrize("hermiticity", ["hermitian", "skew-hermitian"])
def test_pack_unpack_hermiticity(system: System, hermiticity: str):
    """Hermitian matrices use the opposite sign."""
    packing = system.packing()
    rows, cols = system.pattern()
    factor = 1j if hermiticity == "hermitian" else 1
    data = factor * system.g_irreducible[0][..., rows, cols]
    packed = packing.pack(xp.asarray(data), True, hermiticity)
    unpacked = packing.unpack(packed, True, hermiticity)
    assert _relative_error(unpacked, data) < 1e-10


def test_accumulate_blocks(system: System):
    """Dense blocks on and above the diagonal are sufficient."""
    packing = system.packing()
    rows, cols = system.pattern()
    g, __ = system.g_irreducible
    reference = get_host(
        packing.pack(xp.asarray(g[..., rows, cols]), True, "skew-hermitian")
    )
    packed = xp.zeros_like(xp.asarray(reference))
    for i in range(NUM_BLOCKS):
        for j in range(i, NUM_BLOCKS):
            block = g[
                ...,
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]
            packing.accumulate_block(
                packed, xp.asarray(block), i * BLOCK_SIZE, j * BLOCK_SIZE, True
            )
    assert _relative_error(packed, reference) < 1e-12


def test_gw_iteration(system: System):
    """P and Sigma from the packed matrices match the full BZ."""
    packing = system.packing()
    rows, cols = system.pattern()
    ne = len(system.energies)
    g_packed = [
        get_host(packing.pack(xp.asarray(g[..., rows, cols]), True, "skew-hermitian"))
        for g in system.g_irreducible
    ]

    # Polarization, G_ji(-R) = -conj(G_ij(R)).
    p_full = system.polarization()
    p_greater_all = -1j * _correlate(g_packed[1], -g_packed[0].conj())
    p_packed = {
        # P^<(w) = -conj(P^>(-w)) holds in real space.
        "lesser": -p_greater_all[::-1].conj()[ne - 1 :],
        "greater": p_greater_all[ne - 1 :],
    }
    p_irreducible = []
    for p, reference in zip((p_packed["lesser"], p_packed["greater"]), p_full):
        reference = reference[:, system.qgrid._reduce_mapping]
        unpacked = packing.unpack(xp.asarray(p), False, "skew-hermitian")
        assert _relative_error(unpacked, reference[..., rows, cols]) < 1e-10
        dense = np.zeros_like(reference)
        dense[..., cols, rows] = -get_host(unpacked).conj()
        dense[..., rows, cols] = get_host(unpacked)
        p_irreducible.append(dense)

    # Screened interaction on the irreducible and the full q-points.
    w_irreducible = system.screened_interaction(
        system.qgrid.irreducible_points, *p_irreducible
    )
    w_full = system.screened_interaction(system.qgrid.full_points, *p_full)

    # Self-energy.
    for g, w, g_ref, w_ref in zip(g_packed, w_irreducible, system.g_full, w_full):
        w_packed = get_host(
            packing.pack(xp.asarray(w[..., rows, cols]), False, "skew-hermitian")
        )
        sigma = packing.unpack(
            xp.asarray(1j * _convolve(g, w_packed)), True, "skew-hermitian"
        )
        reference = 1j * system.self_energy(g_ref, w_ref)
        reference = reference[:, system.kgrid._reduce_mapping][..., rows, cols]
        assert _relative_error(sigma, reference) < 1e-10


def test_identity_group(system: System):
    """Without symmetry, only the Hermiticity is exploited."""
    identity = [
        op
        for op in system.operations
        if np.array_equal(op.rotation, np.eye(3)) and not np.any(op.translation)
    ]
    grids = [
        KGrid(system.kgrid.size, gamma_centered=gamma_centered)
        for gamma_centered in (system.kgrid.gamma_centered, True)
    ]
    rows, cols = system.pattern()
    packing = OrbitPacking(rows, cols, identity, *grids)
    data = system.g_full[0][..., rows, cols]
    packed = packing.pack(xp.asarray(data), True, "skew-hermitian")
    unpacked = packing.unpack(packed, True, "skew-hermitian")
    assert _relative_error(unpacked, data) < 1e-10


def _little_group(kgrid: KGrid, kpoint: NDArray) -> list[SymmetryOperation]:
    """Operations that map the k-point onto itself."""
    differences = [op.map_kpoints(kpoint[None])[0] - kpoint for op in kgrid.operations]
    return [
        op
        for op, difference in zip(kgrid.operations, differences)
        if np.allclose(difference, np.round(difference))
    ]


def _random_matrices(
    kgrid: KGrid, symmetry: DeviceSymmetry, num_stack: int = 2
) -> NDArray:
    """Random skew-Hermitian matrices, invariant under the little groups."""
    rng = np.random.default_rng(0)
    n = symmetry.num_atoms * symmetry.num_orbitals
    matrices = []
    for kpoint in kgrid.points:
        x = rng.normal(size=(num_stack, n, n)) + 1j * rng.normal(size=(num_stack, n, n))
        x = sum(
            symmetry.transform(x, op, kpoint) for op in _little_group(kgrid, kpoint)
        )
        matrices.append(x - x.conj().swapaxes(-1, -2))
    return np.stack(matrices, axis=1)


def test_volume(system: System):
    """There is exactly one representative per independent element."""
    packing = system.packing()
    rows, cols = system.pattern()

    # The number of real parameters of a skew-Hermitian matrix in the
    # pattern that is invariant under the little groups.
    size = NUM_BLOCKS * BLOCK_SIZE
    mask = np.zeros((size, size))
    mask[system.rows, system.cols] = 1.0
    num_parameters = 0.0
    for kpoint in system.kgrid.points:
        little_group = _little_group(system.kgrid, kpoint)
        for operation in little_group:
            op = create_orbital_operation(system.symmetry, operation)
            rotated = np.linalg.inv(op.rotation).T @ kpoint
            characters = (
                (op.targets == np.arange(size))
                * op.signs
                * np.exp(2j * np.pi * (op.shifts @ rotated))
            )
            trace = (characters @ mask @ characters.conj()).real
            num_parameters += trace / len(little_group)

    # Some of the representatives are purely real or imaginary.
    matrices = _random_matrices(system.kgrid, system.symmetry)
    packed = get_host(
        packing.pack(xp.asarray(matrices[..., rows, cols]), True, "skew-hermitian")
    )
    is_constrained = np.logical_or(
        np.all(np.abs(packed.real) < 1e-10, axis=0),
        np.all(np.abs(packed.imag) < 1e-10, axis=0),
    )
    assert np.all(np.abs(packed).max(axis=0) > 1e-10)
    assert 2 * len(is_constrained) - is_constrained.sum() == round(num_parameters)


# ----------------------------------------------------------------------


def _c4_structure() -> ase.Atoms:
    """Two planes of a chiral structure, such that C4 has no mirrors."""
    u, v = 0.17, 0.29
    ring = [(u, v), (-v, u), (-u, -v), (v, -u)]
    return ase.Atoms(
        ["Si"] + ["C"] * 4 + ["Ge"],
        scaled_positions=[(0.1, 0.0, 0.0)]
        + [(0.1, y % 1, z % 1) for y, z in ring]
        + [(0.6, 0.5, 0.5)],
        cell=[7.0, 5.0, 5.0],
        pbc=True,
    )


def _triclinic_structure() -> ase.Atoms:
    """A structure without any symmetry."""
    return ase.Atoms(
        ["Si", "C", "Ge"],
        scaled_positions=[(0.1, 0.07, 0.11), (0.4, 0.33, 0.52), (0.7, 0.81, 0.23)],
        cell=[[7.0, 0.0, 0.0], [0.3, 5.0, 0.0], [0.1, 0.4, 6.0]],
        pbc=True,
    )


def _slab_structure() -> ase.Atoms:
    """A layer with sigma_z, which keeps all k-points along y."""
    return ase.Atoms(
        ["Si", "C", "C", "Ge", "Ge"],
        scaled_positions=[
            (0.1, 0.0, 0.5),
            (0.2, 0.3, 0.58),
            (0.2, 0.3, 0.42),
            (0.45, 0.6, 0.56),
            (0.45, 0.6, 0.44),
        ],
        cell=[7.0, 5.0, 20.0],
        pbc=True,
    )


@pytest.mark.parametrize("gamma_centered", [False, True], ids=["mp", "gamma"])
@pytest.mark.parametrize(
    "structure, size, num_operations, reduces_points",
    [
        pytest.param(_c4_structure(), (1, 4, 4), 4, True, id="c4-4x4"),
        pytest.param(_c4_structure(), (1, 3, 3), 4, True, id="c4-3x3"),
        pytest.param(_triclinic_structure(), (1, 4, 3), 1, False, id="triclinic"),
        pytest.param(_slab_structure(), (1, 4, 1), 2, False, id="sigma-z"),
    ],
)
def test_random_covariant_matrices(
    structure: ase.Atoms,
    size: tuple,
    num_operations: int,
    reduces_points: bool,
    gamma_centered: bool,
):
    """Groups that silicon does not cover, with the weakest assumptions.

    The C4 operations are not their own inverse. Without symmetry, only
    the Hermiticity relates the matrix elements.

    """
    symmetry = DeviceSymmetry(
        structure.get_scaled_positions(wrap=False),
        structure.cell.array,
        ORBITAL_LABELS,
    )
    operations = filter_packable_operations(
        symmetry, collect_symmetry_operations(structure, 0, time_reversal=False)
    )
    assert len(operations) == num_operations
    kgrid, qgrid = (
        KGrid(
            size,
            gamma_centered=centered,
            symmetry_reduction=True,
            operations=operations,
        )
        for centered in (gamma_centered, True)
    )
    assert (kgrid.num_points < kgrid.num_full_points) == reduces_points

    matrices = _random_matrices(kgrid, symmetry)
    g_full = kgrid.expand_matrices(matrices, symmetry)

    rows, cols = np.triu_indices(len(structure) * NUM_ORBITALS)
    packing = OrbitPacking(
        rows,
        cols,
        [create_orbital_operation(symmetry, op) for op in operations],
        kgrid,
        qgrid,
    )
    packed = packing.pack(xp.asarray(matrices[..., rows, cols]), True, "skew-hermitian")
    phases = np.exp(-2j * np.pi * (kgrid.full_points @ packing.cells.T))
    reference = np.einsum(
        "kr,ekr->er",
        phases[:, packing.rep_cells],
        g_full[..., packing.rep_rows, packing.rep_cols],
    )
    assert _relative_error(packed, reference / packing.num_cells) < 1e-10
    unpacked = packing.unpack(packed, True, "skew-hermitian")
    assert _relative_error(unpacked, matrices[..., rows, cols]) < 1e-10

    # The symmetry reduces the data even if it does not reduce the grid.
    full_size = len(rows) * kgrid.num_full_points
    assert packing.num_representatives < 1.15 * full_size / num_operations


# ----------------------------------------------------------------------


def test_open_pattern_raises():
    """The sparsity pattern needs to be closed."""
    system = _create_system((1, 2, 2), False)
    rows, cols = system.pattern()
    with pytest.raises(ValueError, match="not closed"):
        OrbitPacking(
            rows[rows != 3],
            cols[rows != 3],
            system.operations,
            system.kgrid,
            system.qgrid,
        )


def test_missing_operation_raises():
    """The operations need to form a group containing those of the grid."""
    system = _create_system((1, 4, 4), False)
    rows, cols = system.pattern()
    with pytest.raises(ValueError, match="group"):
        OrbitPacking(rows, cols, system.operations[:3], system.kgrid, system.qgrid)


def test_time_reversal_raises():
    """Time reversal does not hold for lesser/greater under bias."""
    system = _create_system((1, 2, 2), False)
    operation = SymmetryOperation(np.eye(3, dtype=int), np.zeros(3), True)
    with pytest.raises(ValueError, match="Time-reversed"):
        create_orbital_operation(system.symmetry, operation)
    grid = KGrid((1, 2, 2), symmetry_reduction=True, structure=_silicon(NUM_BLOCKS))
    with pytest.raises(ValueError, match="outside of the group"):
        OrbitPacking(*system.pattern(), system.operations, grid, system.qgrid)


def test_mixed_orbitals_raise():
    """A diagonal mirror mixes dx2-y2 and dz2, which is not supported."""
    structure = _silicon(1)
    symmetry = DeviceSymmetry(
        structure.get_scaled_positions(wrap=False),
        structure.cell.array,
        ["s", "dxy", "dyz", "dzx", "dx2y2", "dz2"],
    )
    operations = collect_symmetry_operations(structure, 0, time_reversal=False)
    packable = filter_packable_operations(symmetry, operations)
    assert len(packable) == 4  # E and C2x, with and without centering.
    with pytest.raises(ValueError, match="mixes orbitals"):
        for operation in operations:
            create_orbital_operation(symmetry, operation)


def test_mismatched_blocks_raise():
    """Operations must not connect different blocks."""
    system = _create_system((1, 2, 2), False)
    packing = system.packing()
    g, __ = system.g_irreducible
    packed = xp.zeros((len(system.energies), packing.num_representatives), complex)
    # The first two atoms are on different planes. C2x moves the second.
    size = 2 * NUM_ORBITALS
    with pytest.raises(ValueError, match="different blocks"):
        packing.accumulate_block(packed, xp.asarray(g[..., :size, :size]), 0, 0, True)
