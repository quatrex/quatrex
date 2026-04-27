# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from qttools.comm import comm
from qttools.utils.mpi_utils import distributed_load
from qttools.utils.gpu_utils import get_host
from quatrex.core.config import DeviceConfig, QuatrexConfig
from quatrex.core.scba import SCBAData
from quatrex.coulomb_screening import PCoulombScreening
from quatrex.device.inputs import load_matrix
from quatrex.electron import ElectronSolver
from quatrex.grid import get_electron_energies


@dataclass(frozen=True)
class EnvironmentScreeningExport:
    """Paths written by the environment screening export."""

    output_dir: Path
    p_retarded: Path
    p_lesser: Path
    p_greater: Path
    v_ee: Path
    epsilon_inverse_retarded: Path


def build_environment_config(config: QuatrexConfig) -> QuatrexConfig:
    """Build a Quatrex config that treats ``config.environment`` as the device."""

    if config.environment is None or not config.environment.enabled:
        raise ValueError("The configuration does not enable an environment subsystem.")

    environment = config.environment
    device = DeviceConfig(
        construct_from_unit_cell=environment.construct_from_unit_cell,
        structure_file=environment.structure_file or "structure.xyz",
        neighbor_cell_cutoff=environment.neighbor_cell_cutoff,
        num_transport_cells=environment.num_transport_cells,
        transport_direction=environment.transport_direction,
        block_size=environment.block_size,
        kpoint_grid=environment.kpoint_grid,
        kpoint_shift=environment.kpoint_shift,
        orthogonal_basis=environment.orthogonal_basis,
        num_orbitals_per_atom=config.device.num_orbitals_per_atom,
    )

    electron_solver = config.electron.solver.model_copy(
        update={"compute_current": False}
    )
    electron = config.electron.model_copy(
        update={
            "solver": electron_solver,
            # Keep the environment solve at equilibrium unless explicitly split later.
            "left_fermi_level": config.electron.fermi_level,
            "right_fermi_level": config.electron.fermi_level,
            # The flatband shortcut calls homogenize(), which is not implemented.
            "flatband": False,
            "band_edge_tracking": None,
        }
    )
    # Keep Coulomb buffers allocated for P_ee, but do not instantiate the
    # full CoulombScreeningSolver. The export path only needs H_ee -> G_ee -> P_ee.
    scba = config.scba.model_copy(
        update={
            "coulomb_screening": True,
            "phonon": False,
            "photon": False,
        }
    )

    return config.model_copy(
        update={
            "device": device,
            "electron": electron,
            "scba": scba,
            "input_dir": environment.input_dir,
            "output_dir": config.output_dir / "environment",
            "environment": None,
        },
        deep=True,
    )


def _gather_dense_stack(matrix) -> np.ndarray:
    dense_local = matrix.to_dense()
    dense_full = comm.stack.all_gather_v(
        dense_local,
        axis=0,
        mask=matrix._stack_padding_mask,
    )
    return get_host(dense_full)


def _compute_epsilon_inverse_retarded(
    *,
    v_ee: np.ndarray,
    p_retarded: np.ndarray,
) -> np.ndarray:
    identity = np.eye(v_ee.shape[-1], dtype=np.complex128)
    system = identity[np.newaxis, :, :] - v_ee[np.newaxis, :, :] @ p_retarded
    rhs = np.broadcast_to(identity, system.shape)
    return np.linalg.solve(system, rhs)


def export_environment_screening(
    config: QuatrexConfig,
    *,
    output_dir: Path | None = None,
) -> EnvironmentScreeningExport:
    """Compute and write environment polarization and dielectric inputs.

    The existing electron and Coulomb-polarization machinery is reused on the
    environment subsystem. This is intentionally a one-shot export, not a
    replacement for the main coupled-region runtime.
    """

    environment_config = build_environment_config(config)
    if output_dir is not None:
        environment_config.output_dir = Path(output_dir).resolve()

    def log(message: str) -> None:
        if comm.rank == 0:
            print(f"[environment-export] {message}", flush=True)

    tic = time.perf_counter()
    log("Preparing environment data structures.")
    electron_energies = get_electron_energies(environment_config)
    data = SCBAData(environment_config, electron_energies=electron_energies)
    log(
        "Energy grid: "
        f"{electron_energies[0]} to {electron_energies[-1]} eV "
        f"with {electron_energies.size} points."
    )

    log("Initializing environment electron solver.")
    electron_solver = ElectronSolver(
        environment_config,
        electron_energies,
        sparsity_pattern=data.sparsity_pattern,
    )

    energies_path = environment_config.input_dir / "coulomb_screening_energies.npy"
    if energies_path.is_file():
        coulomb_screening_energies = distributed_load(energies_path)
    else:
        coulomb_screening_energies = electron_energies - electron_energies[0]
        coulomb_screening_energies += 1e-6

    polarization_solver = PCoulombScreening(
        environment_config,
        coulomb_screening_energies,
    )

    log("Solving environment Green's functions.")
    stage_tic = time.perf_counter()
    electron_solver.solve(
        data.sigma_lesser,
        data.sigma_greater,
        data.sigma_retarded,
        out=(data.g_lesser, data.g_greater, data.g_retarded),
    )
    log(f"Environment Green's functions solved in {time.perf_counter() - stage_tic:.2f} s.")

    log("Transposing Green's functions for polarization.")
    for matrix in (data.g_lesser, data.g_greater):
        matrix.dtranspose(discard=False)

    log("Computing environment polarization P_ee.")
    stage_tic = time.perf_counter()
    data.p_lesser.allocate_data()
    data.p_greater.allocate_data()
    data.p_retarded.allocate_data()
    polarization_solver.compute(
        data.g_lesser,
        data.g_greater,
        out=(data.p_lesser, data.p_greater, data.p_retarded),
    )
    log(f"Environment polarization computed in {time.perf_counter() - stage_tic:.2f} s.")

    log("Gathering dense P_ee stacks.")
    p_retarded = _gather_dense_stack(data.p_retarded)
    p_lesser = _gather_dense_stack(data.p_lesser)
    p_greater = _gather_dense_stack(data.p_greater)

    log("Loading environment bare Coulomb matrix v_ee.")
    coulomb_matrix, __ = load_matrix(
        config=environment_config,
        matrix_name=environment_config.coulomb_screening.coulomb_matrix_name,
        sparsity_pattern=data.sparsity_pattern,
        shift_kpoints=True,
    )
    if not coulomb_matrix.symmetry:
        coulomb_matrix.symmetrize()
    coulomb_matrix._data /= environment_config.coulomb_screening.epsilon_r
    v_ee = get_host(coulomb_matrix.to_dense()[0])

    log("Computing epsilon_E^-1 from v_ee and P_ee^R.")
    epsilon_inverse_retarded = _compute_epsilon_inverse_retarded(
        v_ee=v_ee,
        p_retarded=p_retarded,
    )

    export_dir = environment_config.output_dir
    if comm.rank == 0:
        log(f"Writing environment export arrays to {export_dir}.")
        export_dir.mkdir(parents=True, exist_ok=True)
        np.save(export_dir / "p_ee_retarded.npy", p_retarded)
        np.save(export_dir / "p_ee_lesser.npy", p_lesser)
        np.save(export_dir / "p_ee_greater.npy", p_greater)
        np.save(export_dir / "v_ee.npy", v_ee)
        np.save(
            export_dir / "epsilon_environment_inverse_retarded.npy",
            epsilon_inverse_retarded,
        )
        log(f"Done in {time.perf_counter() - tic:.2f} s.")

    return EnvironmentScreeningExport(
        output_dir=export_dir,
        p_retarded=export_dir / "p_ee_retarded.npy",
        p_lesser=export_dir / "p_ee_lesser.npy",
        p_greater=export_dir / "p_ee_greater.npy",
        v_ee=export_dir / "v_ee.npy",
        epsilon_inverse_retarded=export_dir / "epsilon_environment_inverse_retarded.npy",
    )


__all__ = [
    "EnvironmentScreeningExport",
    "build_environment_config",
    "export_environment_screening",
]
