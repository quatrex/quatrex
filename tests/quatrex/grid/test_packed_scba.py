# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Compares an SCBA iteration with orbit packing to the full BZ.

This uses the actual device, solvers, and scattering self-energies for
tight-binding bulk silicon under bias. The reference runs on the full
grid and does not use any crystal symmetry.

"""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import ase.build
import ase.io
import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as mpi_comm

pytest.importorskip("spglib")

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.utils.hdf5_utils import load_hdf5_dict, save_hdf5_dict
from quatrex.core.config import QuatrexConfig, parse_config, setup_context
from quatrex.core.scba import SCBA, SCBAData
from quatrex.coulomb_screening.polarization import PCoulombScreening
from quatrex.coulomb_screening.solver import CoulombScreeningSolver
from quatrex.device import SCBADevice, create_device
from quatrex.electron.solver import ElectronSolver
from quatrex.electron.sse_coulomb_screening import SigmaCoulombScreening
from quatrex.electron.sse_fock import SigmaFock
from quatrex.grid import KGrid, get_electron_energies
from quatrex.grid.kpoints import DeviceSymmetry
from quatrex.grid.orbit_packing import OrbitPacking, create_orbital_operation

A = 5.431  # Angstrom
ORBITAL_LABELS = ["s", "px", "py", "pz", "s*"]
NUM_ORBITALS = len(ORBITAL_LABELS)

# sp3s* parameters (eV) of P. Vogl et al., J. Phys. Chem. Solids 44, 365 (1983).
ONSITE = np.diag([-4.2, 1.715, 1.715, 1.715, 6.685])
V_SS, V_XX, V_XY, V_SP, V_S_STAR_P = -8.3, 1.715, 4.575, 5.7292, 5.3749

CONFIG = """
simulation_dir = "."
input_dir = "."
formalism = "negf"

[device]
kpoint_grid = {kpoint_grid}
kpoint_symmetry_reduction = {symmetry_reduction}
{device_options}
construct_from_unit_cell = true
transport_direction = "a"
neighbor_cell_cutoff = [1, 1, 1]
num_transport_cells = 4
num_orbitals_per_atom = {{ "Si" = 5 }}

    [[device.contacts]]
    name = "left"
    fermi_level = 1.5
    mid_gap_energy = 0.6

    [[device.contacts]]
    name = "right"
    fermi_level = 1.5
    mid_gap_energy = 0.6
    voltage = 1.0

[scba]
symmetric = true
coulomb_screening = true
align_self_energy_to_complex_axes = {align}
{scba_options}

[electron]
flatband = false
energy_window_min = -4.0
energy_window_max = 6.0
energy_window_num = 24
eta = 1e-2
dos_peak_limit = {dos_peak_limit}
solver.algorithm = "rgf"
solver.max_batch_size = 4
max_batch_size = 10
obc.algorithm = "sancho-rubio"
obc.memoizer.mode = "off"

[coulomb_screening]
interaction_cutoff = 4.0
epsilon_r = 4.0
dos_peak_limit = {dos_peak_limit}
num_connected_blocks = 2
align_polarization_to_complex_axes = {align}
{coulomb_screening_options}
apply_hilbert_correction = {hilbert_correction}
max_batch_size = 10
solver.algorithm = "rgf"
solver.max_batch_size = 4
obc.algorithm = "sancho-rubio"
obc.memoizer.mode = "off"
lyapunov.algorithm = "doubling"
lyapunov.memoizer.mode = "off"

[outputs]
device_currents = true
meir_wingreen_currents = true
electron_ldos = true
electron_density = true
hole_density = true
scattering_rate = true
self_energy_density = true
polarization_density = true
coulomb_screening_density = true

{sections}
"""

LABELS = 'orbital_labels = { "Si" = ["s", "px", "py", "pz", "s*"] }'
PHONON = """
[phonon]
model = "pseudo-scattering"
phonon_energy = 0.9
deformation_potential = 0.3
"""


def _hopping(direction: NDArray) -> NDArray:
    """Nearest-neighbor hopping <0|H|d>, in the order of `ORBITAL_LABELS`."""
    g = np.sign(direction)
    h = np.zeros((NUM_ORBITALS, NUM_ORBITALS))
    h[0, 0] = V_SS
    h[0, 1:4], h[1:4, 0] = V_SP * g, -V_SP * g
    h[4, 1:4], h[1:4, 4] = V_S_STAR_P * g, -V_S_STAR_P * g
    h[1:4, 1:4] = V_XY * np.outer(g, g)
    h[(1, 2, 3), (1, 2, 3)] = V_XX
    return h / 4


def _write_inputs(path: Path) -> None:
    """Writes the structure, Hamiltonian and Coulomb matrix."""
    unit_cell = ase.build.bulk("Si", "diamond", a=A, cubic=True)
    ase.io.write(path / "structure.xyz", unit_cell, format="extxyz", parallel=False)
    positions = unit_cell.positions
    strength = np.array([1.0, 0.8, 0.8, 0.8, 0.9])

    hamiltonian, coulomb = {}, {}
    for cell in np.moveaxis(np.indices((3, 3, 3)), 0, -1).reshape(-1, 3) - 1:
        displacements = positions[None, :] + cell * A - positions[:, None]
        distances = np.linalg.norm(displacements, axis=-1)

        h = np.zeros((8, NUM_ORBITALS, 8, NUM_ORBITALS))
        for a, b in np.argwhere(np.abs(distances - np.sqrt(3) / 4 * A) < 1e-3):
            h[a, :, b, :] = _hopping(displacements[a, b])
        if not np.any(cell):
            for a in range(8):
                h[a, :, a, :] = ONSITE
        v = np.exp(-((distances / 1.5) ** 2)) / (0.5 + distances)
        # NOTE: Only keep what is included for all the neighbors.
        v[distances > 0.99 * A] = 0.0

        key = str(cell.tolist())
        hamiltonian[key] = h.reshape(8 * NUM_ORBITALS, -1)
        coulomb[key] = np.kron(v, np.outer(strength, strength)) * 5.0

    save_hdf5_dict(path / "hamiltonian.h5", hamiltonian)
    save_hdf5_dict(path / "coulomb_matrix.h5", coulomb)


def _create(
    path: Path,
    size: list,
    symmetry_reduction: bool,
    dos_peak_limit: float = 100.0,
    hilbert_correction: bool = False,
    align: bool = False,
    device_options: str = "",
    scba_options: str = "",
    coulomb_screening_options: str = "",
    sections: str = "",
) -> tuple[QuatrexConfig, SCBADevice]:
    """Creates the config and the device."""
    config_path = path / f"quatrex_config_{symmetry_reduction}.toml"
    if mpi_comm.rank == 0:
        config_path.write_text(
            CONFIG.format(
                kpoint_grid=size,
                symmetry_reduction=str(symmetry_reduction).lower(),
                dos_peak_limit=dos_peak_limit,
                hilbert_correction=str(hilbert_correction).lower(),
                align=str(align).lower(),
                device_options=device_options,
                scba_options=scba_options,
                coulomb_screening_options=coulomb_screening_options,
                sections=sections,
            )
        )
    mpi_comm.barrier()
    config = parse_config(config_path)
    setup_context(config)
    return config, create_device(config)


def _transpose(state: str, *matrices: DSDBSparse) -> None:
    """Transposes the matrices into the given distribution."""
    for m in matrices:
        m.dtranspose() if m.distribution_state != state else None


def _prepare(config: QuatrexConfig, device: SCBADevice) -> tuple:
    """Allocates everything that is needed for one iteration."""
    energies = get_electron_energies(config)
    data = SCBAData(config, device, energies)
    for name in ("g_retarded", "p_lesser", "p_greater", "p_retarded_hermitian"):
        getattr(data, name).allocate_data()
    for name in ("w_lesser", "w_greater"):
        getattr(data, name).allocate_data()
    electron_solver = ElectronSolver(config, device, energies)
    coulomb_energies = energies - energies[0] + 1e-6
    coulomb_solver = CoulombScreeningSolver(config, device, coulomb_energies)
    return energies, coulomb_energies, data, electron_solver, coulomb_solver


def _snapshot(*matrices: DSDBSparse) -> list[NDArray]:
    """Copies the data, sorted by rows and columns to be comparable."""
    out = []
    for m in matrices:
        _transpose("stack", m)
        order = xp.lexsort(xp.stack([m.cols, m.rows]))
        out.append(m.data[..., order].copy())
    return out


def _solve_electrons(solver: ElectronSolver, data: SCBAData, **kwargs) -> None:
    """Solves for the Green's functions."""
    solver.solve(
        data.sigma_lesser,
        data.sigma_greater,
        data.sigma_retarded_hermitian,
        out=(data.g_lesser, data.g_greater, data.g_retarded),
        **kwargs,
    )


def _reference_iteration(config: QuatrexConfig, device: SCBADevice) -> dict:
    """One iteration on the full grid."""
    energies, coulomb_energies, data, electron_solver, coulomb_solver = _prepare(
        config, device
    )
    sigma = (data.sigma_lesser, data.sigma_greater, data.sigma_retarded_hermitian)
    p = (data.p_lesser, data.p_greater, data.p_retarded_hermitian)
    _transpose("nnz", device.coulomb_matrix)
    sigma_fock = SigmaFock(config, device, energies)
    _transpose("stack", device.coulomb_matrix)
    results = {}

    data.g_retarded.data[:] = 0.0
    _solve_electrons(electron_solver, data)
    results["g"] = _snapshot(data.g_lesser, data.g_greater)

    # NOTE: The lesser polarization is taken from its definition, by
    # exchanging the Green's functions, instead of from the symmetry
    # relation in PCoulombScreening. They have to be the same.
    p_coulomb_screening = PCoulombScreening(config, device, coulomb_energies)
    p_exchanged = [config.compute.dsdbsparse_type.empty_like(m) for m in p]
    for m in p_exchanged:
        m.allocate_data()
    p_coulomb_screening.compute(data.g_greater, data.g_lesser, out=p_exchanged)
    p_coulomb_screening.compute(data.g_lesser, data.g_greater, out=p)
    results["p_lesser_from_code"] = _snapshot(data.p_lesser)
    data.p_lesser.data[:] = p_exchanged[1].data
    results["p"] = _snapshot(data.p_lesser, data.p_greater)

    coulomb_solver.solve(*p, out=(data.w_lesser, data.w_greater))
    results["w"] = _snapshot(data.w_lesser, data.w_greater)

    for m in sigma:
        m.data[:] = 0.0
    sigma_fock.compute(data.g_lesser, out=(data.sigma_retarded_hermitian,))
    SigmaCoulombScreening(config, device, energies).compute(
        data.g_lesser, data.g_greater, data.w_lesser, data.w_greater, out=sigma
    )
    results["sigma"] = _snapshot(*sigma)

    _transpose("stack", data.g_lesser, data.g_greater)
    _solve_electrons(electron_solver, data)
    results["g_next"] = _snapshot(data.g_lesser, data.g_greater)
    results["current"] = [electron_solver.meir_wingreen_current.copy()]
    return results


def _create_packing(device: SCBADevice, data: SCBAData) -> OrbitPacking:
    """Creates the packing from the geometry of the device."""
    num_cells = device.device_config.num_transport_cells
    structure = device.structure.repeat((num_cells, 1, 1))
    np.testing.assert_allclose(structure.positions, device.atom_coordinates)
    # NOTE: The operations are in the basis of the unit cell. They leave
    # the transport axis untouched, and thus also apply to the device.
    symmetry = DeviceSymmetry(
        structure.get_scaled_positions(wrap=False),
        structure.cell.array,
        ORBITAL_LABELS,
    )
    assert len(device.kgrid.operations) == len(device.qgrid.operations)
    operations = [
        create_orbital_operation(symmetry, op) for op in device.kgrid.operations
    ]
    return OrbitPacking(
        data.g_lesser.rows, data.g_lesser.cols, operations, device.kgrid, device.qgrid
    )


def _packed_iteration(config: QuatrexConfig, device: SCBADevice) -> dict:
    """One iteration on the irreducible points with packed matrices."""
    energies, coulomb_energies, data, electron_solver, coulomb_solver = _prepare(
        config, device
    )
    sigma = (data.sigma_lesser, data.sigma_greater, data.sigma_retarded_hermitian)
    p = (data.p_lesser, data.p_greater, data.p_retarded_hermitian)
    packing = _create_packing(device, data)
    create = partial(
        packing.create_matrix,
        num_energies=len(energies),
        dsdbsparse_type=config.compute.dsdbsparse_type,
    )
    names = ("g_lesser", "g_greater", "p_lesser", "p_greater", "w_lesser", "w_greater")
    packed = {name: create() for name in (*names, "sigma_lesser", "sigma_greater")}
    for name in ("p_retarded_hermitian", "sigma_retarded_hermitian"):
        packed[name] = create(symmetry="hermitian")
    packed_sigma = [
        packed[f"sigma_{name}"] for name in ("lesser", "greater", "retarded_hermitian")
    ]
    packed_p = [
        packed[f"p_{name}"] for name in ("lesser", "greater", "retarded_hermitian")
    ]

    # The elementwise products do not have any k-points in real space.
    coulomb_matrix = packing.create_matrix(
        comm.stack.size, config.compute.dsdbsparse_type, "hermitian"
    )
    packing.accumulate_matrix(
        coulomb_matrix.data[:, 0],
        device.coulomb_matrix,
        signed=False,
        mask=xp.ones(1, dtype=bool),
    )
    packed_device = SimpleNamespace(
        kgrid=KGrid((1, 1, 1)),
        qgrid=KGrid((1, 1, 1), gamma_centered=True),
        coulomb_matrix=coulomb_matrix,
    )
    results = {"packing": packing}

    # The solvers have to reset what they pack.
    for name in ("g_lesser", "g_greater", "w_lesser", "w_greater"):
        packed[name].data[:] = 1.0

    data.g_retarded.data[:] = 0.0
    _solve_electrons(
        electron_solver,
        data,
        packing=packing,
        packed_out=(packed["g_lesser"].data[:, 0], packed["g_greater"].data[:, 0]),
    )
    results["g"] = _snapshot(data.g_lesser, data.g_greater)
    # What was packed in the backward sweep is what ended up in the output.
    results["g_unpacked"] = []
    for m, m_packed in (
        (data.g_lesser, packed["g_lesser"]),
        (data.g_greater, packed["g_greater"]),
    ):
        unpacked = packing.unpack(m_packed.data[:, 0], True, "skew-hermitian")
        order = xp.lexsort(xp.stack([m.cols, m.rows]))
        results["g_unpacked"].append(unpacked[..., order])

    PCoulombScreening(config, packed_device, coulomb_energies).compute(
        packed["g_lesser"], packed["g_greater"], out=packed_p
    )
    for m, m_packed in zip(p, packed_p):
        assert xp.array_equal(m.rows, data.g_lesser.rows)
        assert xp.array_equal(m.cols, data.g_lesser.cols)
        m.data[:] = packing.unpack(m_packed.data[:, 0], False, m.symmetry)
    results["p"] = _snapshot(data.p_lesser, data.p_greater)

    coulomb_solver.solve(
        *p,
        out=(data.w_lesser, data.w_greater),
        packing=packing,
        packed_out=(packed["w_lesser"].data[:, 0], packed["w_greater"].data[:, 0]),
    )
    results["w"] = _snapshot(data.w_lesser, data.w_greater)
    # NOTE: Compared in packed form, as orbits that vanish for G do not
    # have to vanish for W, and are not represented.
    results["w_packed"], results["w_repacked"] = [], []
    for name in ("w_lesser", "w_greater"):
        w, w_packed = getattr(data, name), packed[name].data[:, 0]
        repacked = xp.zeros_like(w_packed)
        mask = xp.ones(len(repacked), dtype=bool)
        packing.accumulate_matrix(repacked, w, False, mask)
        results["w_packed"].append(w_packed.copy())
        results["w_repacked"].append(repacked)

    SigmaFock(config, packed_device, energies).compute(
        packed["g_lesser"], out=(packed["sigma_retarded_hermitian"],)
    )
    SigmaCoulombScreening(config, packed_device, energies).compute(
        packed["g_lesser"],
        packed["g_greater"],
        packed["w_lesser"],
        packed["w_greater"],
        out=packed_sigma,
    )
    _transpose("stack", *packed_sigma)
    for m, m_packed in zip(sigma, packed_sigma):
        m.data[:] = packing.unpack(m_packed.data[:, 0], True, m.symmetry)
    results["sigma"] = _snapshot(*sigma)

    _solve_electrons(electron_solver, data)
    results["g_next"] = _snapshot(data.g_lesser, data.g_greater)
    results["current"] = [electron_solver.meir_wingreen_current.copy()]
    return results


def _relative_error(a: NDArray, b: NDArray) -> float:
    """Largest deviation of a from b on all ranks."""
    errors = np.array([float(xp.abs(a - b).max()), float(xp.abs(b).max())])
    errors = np.max(mpi_comm.allgather(errors), axis=0)
    return errors[0] / errors[1]


def run_comparison(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: list,
    dos_peak_limit: float = 100.0,
    hilbert_correction: bool = False,
) -> tuple[dict, dict, SCBADevice]:
    """Runs both iterations."""
    # Time reversal does not hold for the lesser/greater quantities.
    for module in ("base", "scba"):
        monkeypatch.setattr(
            f"quatrex.device.{module}.KGrid", partial(KGrid, time_reversal=False)
        )
    path = Path(mpi_comm.bcast(str(path)))
    if mpi_comm.rank == 0:
        _write_inputs(path)
    mpi_comm.barrier()
    args = (size, dos_peak_limit, hilbert_correction)
    full_config, full_device = _create(path, args[0], False, *args[1:])
    reference = _reference_iteration(full_config, full_device)
    config, device = _create(path, args[0], True, *args[1:])
    return reference, _packed_iteration(config, device), device


@pytest.mark.parametrize(
    "size, dos_peak_limit, hilbert_correction",
    [
        pytest.param([1, 2, 2], 100.0, False, id="2x2"),
        pytest.param([1, 3, 3], 100.0, False, id="3x3"),
        pytest.param([1, 3, 3], 100.0, True, id="3x3-hilbert-correction"),
        pytest.param([1, 4, 4], 100.0, False, id="4x4"),
        pytest.param([1, 3, 3], 0.5, False, id="3x3-filtered"),
    ],
)
def test_scba_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: list,
    dos_peak_limit: float,
    hilbert_correction: bool,
):
    """All quantities match the ones on the full grid."""
    reference, result, device = run_comparison(
        tmp_path, monkeypatch, size, dos_peak_limit, hilbert_correction
    )
    assert device.kgrid.num_points < device.kgrid.num_full_points

    for g, g_unpacked in zip(result["g"], result["g_unpacked"]):
        assert _relative_error(g_unpacked, g) < 1e-8
    for w_packed, w_repacked in zip(result["w_packed"], result["w_repacked"]):
        assert _relative_error(w_packed, w_repacked) < 1e-8

    if dos_peak_limit < 1.0:
        # Make sure that the filter actually does something.
        # The screened interaction at zero energy is always removed.
        for x in (result["g"][1], result["w"][1][1:]):
            is_zero = xp.all(x == 0.0, axis=-1)
            assert xp.any(is_zero) and not xp.all(is_zero)

    assert _relative_error(reference["p_lesser_from_code"][0], reference["p"][0]) < 1e-8

    for name in ("g", "p", "w", "sigma", "g_next", "current"):
        grid = device.qgrid if name in ("p", "w") else device.kgrid
        for a, b in zip(result[name], reference[name]):
            b = b[:, grid._reduce_mapping]
            assert _relative_error(a, b) < 1e-8, name


def _run_scba(path: Path, name: str, modify=None, **kwargs) -> tuple[SCBA, dict]:
    """Runs the SCBA and gets its outputs of the last iteration."""
    path = Path(mpi_comm.bcast(str(path))) / name
    if mpi_comm.rank == 0:
        path.mkdir()
        _write_inputs(path)
        if modify is not None:
            modify(path)
    mpi_comm.barrier()
    config, device = _create(path, **kwargs)
    scba = SCBA(config, device)
    scba.run()

    mpi_comm.barrier()
    iteration = config.scba.max_iterations - 1
    files = sorted(Path(config.output_dir).glob(f"*_{iteration}.npy"))
    return scba, {file.stem: xp.load(file) for file in files}


def _compare_scba(result: tuple, reference: tuple, tolerance: float = 1e-8) -> None:
    """Compares the self-energies, the observables and the outputs."""
    (scba, outputs), (reference_scba, reference_outputs) = result, reference
    grid = scba.device.kgrid
    for name in ("sigma_lesser", "sigma_greater", "sigma_retarded_hermitian"):
        (a,), (b,) = (_snapshot(getattr(s.data, name)) for s in (scba, reference_scba))
        assert float(xp.abs(b).max()) > 1e-4
        assert _relative_error(a, b[:, grid._reduce_mapping]) < tolerance, name

    a, b = (float(s.observables.current_conservation) for s in (scba, reference_scba))
    assert abs(a - b) < 1e-6 * abs(b)

    assert len(outputs) == 12
    assert outputs.keys() == reference_outputs.keys()
    for name, a in outputs.items():
        b = reference_outputs[name]
        assert a.shape == b.shape
        assert _relative_error(a, b) < tolerance, name


@pytest.mark.parametrize(
    "size, kwargs",
    [
        pytest.param([1, 2, 2], {}, id="2x2"),
        pytest.param([1, 3, 3], {}, id="3x3"),
        pytest.param([1, 4, 2], {}, id="4x2"),
        pytest.param([1, 4, 4], {}, id="4x4"),
        pytest.param([1, 3, 3], {"sections": PHONON}, id="3x3-phonon"),
        pytest.param(
            [1, 3, 3],
            {
                "coulomb_screening_options": "energy_renormalization_terms = "
                '["exchange", "correlation", "polarization"]'
            },
            id="3x3-polarization",
        ),
        pytest.param([1, 3, 3], {"dos_peak_limit": 0.5}, id="3x3-filtered"),
    ],
)
def test_scba_run(tmp_path: Path, size: list, kwargs: dict):
    """The SCBA is the same as on the full grid."""
    kwargs = {
        "size": size,
        "scba_options": "max_iterations = 3\nmixing_factor = 0.6",
        **kwargs,
    }
    if "sections" in kwargs:
        kwargs["scba_options"] += "\nphonon = true"
    reference = _run_scba(tmp_path, "full", symmetry_reduction=False, **kwargs)
    result = _run_scba(
        tmp_path, "packed", symmetry_reduction=True, device_options=LABELS, **kwargs
    )
    assert reference[0].packed is None
    assert result[0].packed is not None
    assert result[0].device.kgrid.num_points < result[0].device.kgrid.num_full_points
    _compare_scba(result, reference)


def test_scba_run_aligned(tmp_path: Path):
    """The alignment to the complex axes is compatible with symmetry."""
    kwargs = {
        "size": [1, 3, 3],
        "align": True,
        "symmetry_reduction": True,
        "device_options": LABELS,
        "scba_options": "max_iterations = 3",
    }
    result = _run_scba(tmp_path, "packed", **kwargs)

    def break_symmetry(path: Path) -> None:
        """Leaves only the identity, without changing the matrices."""
        atoms = ase.io.read(path / "structure.xyz", format="extxyz", parallel=False)
        rng = np.random.default_rng(0)
        atoms.positions += rng.uniform(-1e-3, 1e-3, atoms.positions.shape)
        ase.io.write(path / "structure.xyz", atoms, format="extxyz", parallel=False)

    reference = _run_scba(tmp_path, "identity", modify=break_symmetry, **kwargs)
    device = reference[0].device
    assert len(device.orbital_operations) == 1
    assert device.kgrid.num_points == device.kgrid.num_full_points
    _compare_scba(result, reference)

    # Both runs are packed, so this has to be checked independently. The
    # elements are on the imaginary (real) axis in real space if those at
    # opposite k-points are related by complex conjugation.
    points = device.kgrid.points
    differences = points[:, None] + points[None]
    distances = np.abs(differences - np.round(differences)).max(axis=-1)
    opposite = distances.argmin(axis=1)
    assert distances.min(axis=1).max() < 1e-12
    assert (opposite != np.arange(len(points))).any()
    for name, sign in (
        ("sigma_lesser", -1),
        ("sigma_greater", -1),
        ("sigma_retarded_hermitian", 1),
    ):
        (a,) = _snapshot(getattr(reference[0].data, name))
        assert _relative_error(a[:, opposite], sign * a.conj()) < 1e-10, name


@pytest.mark.parametrize("name", ["hamiltonian", "coulomb_matrix", "potential"])
def test_broken_symmetry(tmp_path: Path, name: str):
    """Inputs that break the symmetry are refused."""

    def modify(path: Path) -> None:
        """Breaks the symmetry of the input."""
        if name == "potential":
            np.save(path / "potential.npy", np.linspace(0.0, 1.0, 4 * 8 * NUM_ORBITALS))
            return
        matrices = load_hdf5_dict(path / f"{name}.h5")
        matrices["[0, 0, 0]"][0, 0] += 0.1
        save_hdf5_dict(path / f"{name}.h5", matrices)

    with pytest.raises(ValueError, match=f"{name} breaks the symmetry"):
        _run_scba(
            tmp_path,
            "packed",
            modify=modify,
            size=[1, 2, 2],
            symmetry_reduction=True,
            device_options=LABELS,
            scba_options="max_iterations = 1",
        )


def test_wrong_labels(tmp_path: Path):
    """Orbitals in the wrong order are noticed."""
    labels = LABELS.replace('"px", "py"', '"py", "px"')
    with pytest.raises(ValueError, match="breaks the symmetry"):
        _run_scba(
            tmp_path,
            "packed",
            size=[1, 2, 2],
            symmetry_reduction=True,
            device_options=labels,
        )


def test_missing_labels(tmp_path: Path):
    """Without the orbitals there is a warning."""
    path = Path(mpi_comm.bcast(str(tmp_path)))
    if mpi_comm.rank == 0:
        _write_inputs(path)
    mpi_comm.barrier()
    with pytest.warns(UserWarning, match="orbital_labels"):
        __, device = _create(path, [1, 2, 2], True)
    assert not device.uses_packing
