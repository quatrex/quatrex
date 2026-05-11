from pathlib import Path
from types import SimpleNamespace

from quatrex.coulomb_screening.dielectric_screening.rpa_compute import (
    resolve_unit_cell_matrix_path,
)


def test_resolve_unit_cell_matrix_path_uses_environment_input_dir_when_enabled():
    config = SimpleNamespace(
        input_dir=Path("/tmp/device_inputs"),
        environment=SimpleNamespace(
            enabled=True,
            input_dir=Path("/tmp/environment_inputs"),
        ),
    )

    resolved = resolve_unit_cell_matrix_path(config, "hamiltonian")

    assert resolved == Path("/tmp/environment_inputs/hamiltonian.mat")


def test_resolve_unit_cell_matrix_path_falls_back_to_device_input_dir():
    config = SimpleNamespace(
        input_dir=Path("/tmp/device_inputs"),
        environment=SimpleNamespace(
            enabled=False,
            input_dir=Path("/tmp/environment_inputs"),
        ),
    )

    resolved = resolve_unit_cell_matrix_path(config, "coulomb_matrix")

    assert resolved == Path("/tmp/device_inputs/coulomb_matrix.mat")
