from pathlib import Path

import numpy as np
import pytest
import scipy.io

from quatrex.core.config import parse_config


def _write_rpa_screening_inputs(input_dir: Path) -> None:
    scipy.io.savemat(
        input_dir / "hamiltonian.mat",
        {
            "[0, 0, 0]": [[0.0]],
            "[0, 0, 1]": [[-1.0]],
            "[0, 0, -1]": [[-1.0]],
        },
    )
    scipy.io.savemat(
        input_dir / "coulomb_matrix.mat",
        {
            "[0, 0, 0]": [[2.0]],
            "[0, 0, 1]": [[0.25]],
            "[0, 0, -1]": [[0.25]],
        },
    )


def test_parse_negf_config_with_rpa_screening(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_rpa_screening_inputs(input_dir)

    config_path = tmp_path / "quatrex_config.toml"
    config_path.write_text(
        """
simulation_dir = "."
formalism = "negf"
input_dir = "inputs"

[device]
transport_direction = "z"
construct_from_unit_cell = true
neighbor_cell_cutoff = [0, 0, 1]

[scba]
coulomb_screening = true

[electron]
fermi_level = 0.0
temperature = 300.0
energy_window_min = -1.0
energy_window_max = 1.0
energy_window_num = 8

[coulomb_screening]
polarization_method = "rpa"
interaction_cutoff = 8.0
num_k_points = 8
num_frequencies = 3
max_frequency = 0.5
""".strip()
    )

    config = parse_config(config_path)

    assert config.formalism == "negf"
    assert config.coulomb_screening is not None
    assert config.coulomb_screening.polarization_method == "rpa"
    assert config.coulomb_screening.num_k_points == 8
    assert config.input_dir == input_dir.resolve()


def test_rpa_screening_requires_mesh_fields(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_rpa_screening_inputs(input_dir)

    config_path = tmp_path / "quatrex_config.toml"
    config_path.write_text(
        """
simulation_dir = "."
formalism = "negf"
input_dir = "inputs"

[device]
transport_direction = "z"
construct_from_unit_cell = true
neighbor_cell_cutoff = [0, 0, 1]

[scba]
coulomb_screening = true

[electron]
fermi_level = 0.0
temperature = 300.0
energy_window_min = -1.0
energy_window_max = 1.0
energy_window_num = 8

[coulomb_screening]
polarization_method = "rpa"
interaction_cutoff = 8.0
""".strip()
    )

    with pytest.raises(
        ValueError,
        match="required for dielectric RPA",
    ):
        parse_config(config_path)


def test_parse_negf_config_with_environment_screening_block(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    environment_input_dir = tmp_path / "environment_inputs"
    environment_input_dir.mkdir()
    _write_rpa_screening_inputs(environment_input_dir)

    config_path = tmp_path / "quatrex_config.toml"
    config_path.write_text(
        """
simulation_dir = "."
formalism = "negf"
input_dir = "inputs"

[device]
transport_direction = "z"
construct_from_unit_cell = true
neighbor_cell_cutoff = [0, 0, 1]

[scba]
coulomb_screening = true

[electron]
fermi_level = 0.0
temperature = 300.0
energy_window_min = -1.0
energy_window_max = 1.0
energy_window_num = 8

[coulomb_screening]
interaction_cutoff = 8.0

[environment]
enabled = true
input_dir = "environment_inputs"
transport_direction = "z"
construct_from_unit_cell = true
neighbor_cell_cutoff = [0, 0, 1]

[environment.screening]
method = "rpa"
num_k_points = 8
num_frequencies = 3
max_frequency = 0.5
""".strip()
    )

    config = parse_config(config_path)

    assert config.environment is not None
    assert config.environment.enabled is True
    assert config.environment.input_dir == environment_input_dir.resolve()
    assert config.environment.screening.method == "rpa"
    assert config.environment_screening.method == "rpa"
    assert config.environment_screening.num_k_points == 8
    assert config.coulomb_screening.dielectric_environment is True
    assert config.coulomb_screening.dielectric_method == "rpa"


def test_parse_negf_config_with_saved_environment_screening(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    environment_export_dir = tmp_path / "environment_export"
    environment_export_dir.mkdir()

    np.save(environment_export_dir / "p_ee_retarded.npy", np.zeros((8, 1, 1)))
    np.save(environment_export_dir / "p_ee_lesser.npy", np.zeros((8, 1, 1)))
    np.save(environment_export_dir / "v_ee.npy", np.eye(1))

    config_path = tmp_path / "quatrex_config.toml"
    config_path.write_text(
        """
simulation_dir = "."
formalism = "negf"
input_dir = "inputs"

[device]
transport_direction = "z"
construct_from_unit_cell = true
neighbor_cell_cutoff = [0, 0, 1]

[scba]
coulomb_screening = true

[electron]
fermi_level = 0.0
temperature = 300.0
energy_window_min = -1.0
energy_window_max = 1.0
energy_window_num = 8

[coulomb_screening]
interaction_cutoff = 8.0

[environment.screening]
method = "negf"
source = "file"
input_dir = "environment_export"
""".strip()
    )

    config = parse_config(config_path)

    assert config.environment is not None
    assert config.environment.enabled is False
    assert config.environment.input_dir is None
    assert config.environment_screening.method == "negf"
    assert config.environment_screening.source == "file"
    assert config.environment_screening.input_dir == environment_export_dir.resolve()
    assert config.coulomb_screening.dielectric_environment is True
    assert config.coulomb_screening.dielectric_method == "negf"


def test_negf_environment_screening_requires_saved_export(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    config_path = tmp_path / "quatrex_config.toml"
    config_path.write_text(
        """
simulation_dir = "."
formalism = "negf"
input_dir = "inputs"

[device]
transport_direction = "z"
construct_from_unit_cell = true
neighbor_cell_cutoff = [0, 0, 1]

[scba]
coulomb_screening = true

[electron]
fermi_level = 0.0
temperature = 300.0
energy_window_min = -1.0
energy_window_max = 1.0
energy_window_num = 8

[coulomb_screening]
interaction_cutoff = 8.0

[environment.screening]
method = "negf"
source = "compute"
""".strip()
    )

    with pytest.raises(ValueError, match="currently requires source='file'"):
        parse_config(config_path)
