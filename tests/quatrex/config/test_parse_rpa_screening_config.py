from pathlib import Path

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

    with pytest.raises(ValueError, match="required for polarization_method='rpa'"):
        parse_config(config_path)
