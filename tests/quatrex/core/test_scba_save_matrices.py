# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests saving the matrices of the last SCBA iteration."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qttools.datastructures import DSDBSparse
from qttools.utils.gpu_utils import get_host
from quatrex.core.config import (
    SCBA_MATRICES,
    OutputConfig,
    parse_config,
    setup_context,
)
from quatrex.core.scba import SCBA
from quatrex.core.scsp import SCSP
from quatrex.device import create_device

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"
GW_EXAMPLES = EXAMPLES_DIR / "w90" / "carbon-nanotube"

ALL_MATRICES = [name for group in SCBA_MATRICES.values() for name in group]


def _elements(matrix: DSDBSparse) -> tuple[np.ndarray, np.ndarray]:
    """Returns the sorted positions and the corresponding elements."""
    rows, cols = (get_host(ind).astype(np.int64) for ind in matrix.spy())
    keys = rows * matrix.shape[-1] + cols
    order = np.argsort(keys)
    return keys[order], get_host(matrix.data)[..., order].copy()


class _CapturingSCBA(SCBA):
    """Captures the matrices as soon as they have been computed."""

    def _capture(self, *names: str) -> None:
        self.captured = getattr(self, "captured", {})
        for name in names:
            self.captured[name] = _elements(getattr(self.data, name))

    def _compute_electron_observables(self) -> None:
        self._capture("g_lesser", "g_greater", "g_retarded")
        super()._compute_electron_observables()

    def _compute_coulomb_screening_observables(self) -> None:
        self._capture(*SCBA_MATRICES["p"])
        self._capture(*SCBA_MATRICES["w"])
        super()._compute_coulomb_screening_observables()


def _make_config(
    tmp_path: Path,
    adjust_config_paths: Callable,
    matrices: list[str],
    symmetric: bool = True,
    example: str = "gw-unit-cell",
):
    """Creates the config of a small GW simulation (two iterations)."""
    tmp_path.mkdir(exist_ok=True, parents=True)
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(GW_EXAMPLES / example / "quatrex_config.toml", tmp_config_path)

    text = tmp_config_path.read_text()
    for old, new in (
        ("energy_window_num = 100", "energy_window_num = 20"),
        ("symmetric = true", f"symmetric = {str(symmetric).lower()}"),
        ("[outputs]", f"[outputs]\nscba_matrices = {matrices}\n"),
        # NOTE: Required for a non-zero `p_retarded_hermitian`.
        (
            "[coulomb_screening]",
            '[coulomb_screening]\nenergy_renormalization_terms = ["polarization", "correlation"]\n',
        ),
    ):
        assert text.count(old) == 1
        text = text.replace(old, new)
    tmp_config_path.write_text(text)

    config = parse_config(tmp_config_path)
    setup_context(config)
    return config


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("symmetric", [True, False])
def test_saved_matrices(tmp_path: Path, adjust_config_paths: Callable, symmetric: bool):
    """Tests that the files contain the matrices of the last iteration."""
    config = _make_config(
        tmp_path, adjust_config_paths, ["g", "p", "w", "sigma"], symmetric
    )
    scba = _CapturingSCBA(config, create_device(config))
    scba.run()
    assert scba.iteration == 2
    # The files must only be written at the end.
    assert not list(config.output_dir.glob("*.h5"))

    scba.save_matrices()
    files = {file.stem for file in config.output_dir.glob("*.h5")}
    assert files == set(ALL_MATRICES)

    for name in SCBA_MATRICES["sigma"]:
        scba._capture(name)

    for name in ALL_MATRICES:
        reference_keys, reference = scba.captured[name]
        loaded = DSDBSparse.load(config.output_dir / f"{name}.h5")
        keys, elements = _elements(loaded)

        assert np.array_equal(keys, reference_keys), name
        assert np.array_equal(elements, reference), name
        assert loaded.symmetry == getattr(scba.data, name).symmetry, name
        assert np.any(elements), name


@pytest.mark.mpi_skip()
def test_saving_does_not_change_results(
    tmp_path: Path, adjust_config_paths: Callable, monkeypatch: pytest.MonkeyPatch
):
    """Tests that only the selected matrices are saved.

    The sparsity pattern of this example contains blocks that are not
    written by the solver. All newly allocated matrices are therefore
    filled with non-zero values, which must not affect the results.

    """
    allocate_data = DSDBSparse.allocate_data

    def _run(matrices: list[str], poison: bool) -> tuple[dict, dict]:
        def _allocate_data(self, *args, **kwargs):
            is_new = self._data is None
            allocate_data(self, *args, **kwargs)
            if is_new and poison:
                self._data[:] = 1.0 + 1.0j

        monkeypatch.setattr(DSDBSparse, "allocate_data", _allocate_data)

        config = _make_config(
            tmp_path / f"{len(matrices)}-{poison}",
            adjust_config_paths,
            matrices,
            example="gw",
        )
        scba = SCBA(config, create_device(config))
        scba.run()
        scba.save_matrices()

        files = {file.stem for file in config.output_dir.glob("*.h5")}
        assert files == set(matrices)

        observables = {
            f.name: np.load(f) for f in config.output_dir.glob("*_[0-9].npy")
        }
        sigma = {
            name: _elements(getattr(scba.data, name))[1]
            for name in SCBA_MATRICES["sigma"]
        }
        return observables | sigma, scba

    reference, __ = _run([], poison=False)
    assert len(reference) > 3
    for matrices in ([], ["g_lesser", "g_greater", "w_lesser"]):
        result, scba = _run(matrices, poison=True)
        assert reference.keys() == result.keys()
        for name in reference:
            assert np.array_equal(reference[name], result[name], equal_nan=True), name

    # The elements that are not computed are zero.
    rows, cols = (get_host(ind) // 32 for ind in scba.data.g_lesser.spy())
    outside = np.abs(rows - cols) > 1
    assert outside.any()
    assert not np.any(get_host(scba.data.g_lesser.data)[..., outside])


def test_expand_groups():
    """Tests that groups are expanded without duplicates."""
    outputs = OutputConfig(scba_matrices=["w_lesser", "w", "sigma_greater"])
    assert outputs.scba_matrices == ["w_lesser", "w_greater", "sigma_greater"]
    assert OutputConfig().scba_matrices == []

    with pytest.raises(ValueError):
        OutputConfig(scba_matrices=["not_a_matrix"])


def test_requires_coulomb_screening(tmp_path: Path, adjust_config_paths: Callable):
    """Tests that the matrices to save have to be computed."""
    config = _make_config(tmp_path, adjust_config_paths, ["w"])
    config_path = config.config_file
    text = config_path.read_text()
    assert text.count("coulomb_screening = true") == 1
    config_path.write_text(
        text.replace("coulomb_screening = true", "coulomb_screening = false")
    )
    with pytest.raises(ValueError, match="coulomb_screening"):
        parse_config(config_path)


@pytest.mark.parametrize("converges", [True, False])
def test_scsp_saves_once(tmp_path: Path, converges: bool):
    """Tests that SCSP runs save the matrices once, after the last step."""
    calls = []
    transport_solver = SimpleNamespace(
        set_potential=lambda potential: None,
        run=lambda: calls.append("run"),
        get_charge_density=lambda: None,
        save_matrices=lambda: calls.append("save"),
    )

    scsp = SCSP.__new__(SCSP)
    scsp.config = SimpleNamespace(
        output_dir=tmp_path, scsp=SimpleNamespace(max_iterations=3)
    )
    scsp.transport_solver = transport_solver
    scsp.electrostatic_solver = SimpleNamespace(
        generate_initial_guess=lambda: np.zeros(2),
        solve=lambda density, potential, spectral_dos: potential
        + (0.0 if converges else 1.0),
    )
    scsp.mixer = SimpleNamespace(mix=lambda old, new: new)
    scsp.use_spectral_dos = False
    scsp.convergence_tol = 1e-3
    scsp.scba_iterations_per_step = None
    scsp.run()

    assert calls == ["run"] * (1 if converges else 3) + ["save"]
