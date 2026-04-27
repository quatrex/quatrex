from types import SimpleNamespace

import numpy as np
import scipy.io
from numpy.testing import assert_allclose

from quatrex.dielectric_screening.negf_bridge import EquilibriumRPAScreeningBridge


def _write_input_matrices(tmp_path):
    scipy.io.savemat(
        tmp_path / "hamiltonian.mat",
        {
            "[0, 0, 0]": np.array([[0.0]], dtype=np.complex128),
            "[0, 0, 1]": np.array([[-1.0]], dtype=np.complex128),
            "[0, 0, -1]": np.array([[-1.0]], dtype=np.complex128),
        },
    )
    scipy.io.savemat(
        tmp_path / "coulomb_matrix.mat",
        {
            "[0, 0, 0]": np.array([[2.0]], dtype=np.complex128),
            "[0, 0, 1]": np.array([[0.25]], dtype=np.complex128),
            "[0, 0, -1]": np.array([[0.25]], dtype=np.complex128),
        },
    )


def test_negf_rpa_bridge_rejects_transverse_k_sampling(tmp_path):
    _write_input_matrices(tmp_path)
    config = SimpleNamespace(
        input_dir=tmp_path,
        device=SimpleNamespace(
            construct_from_unit_cell=True,
            transport_direction="z",
            kpoint_grid=(2, 1, 1),
            neighbor_cell_cutoff=(0, 0, 1),
            num_transport_cells=1,
        ),
        coulomb_screening=SimpleNamespace(
            num_k_points=8,
            num_q_points=3,
            interaction_cutoff=8.0,
            temperature=300.0,
            epsilon_r=1.0,
            hamiltonian_matrix_name="hamiltonian",
            coulomb_matrix_name="coulomb_matrix",
            periodic_axis=2,
            lattice_constant=1.0,
            broadening=0.0,
            include_zero_q=True,
            chemical_potential=0.0,
        ),
        electron=SimpleNamespace(fermi_level=0.0, left_fermi_level=0.0),
    )
    template = SimpleNamespace(stack_section_sizes=np.array([4]))
    bridge = EquilibriumRPAScreeningBridge(
        config,
        screening_energies=np.linspace(1e-6, 0.5, 4),
        template=template,
    )

    try:
        bridge._validate_supported_configuration()
    except NotImplementedError as exc:
        assert "transverse k-point sampling" in str(exc)
    else:
        raise AssertionError("Expected transverse k-grid validation to fail.")


def test_negf_rpa_bridge_builds_transport_matrices(tmp_path):
    _write_input_matrices(tmp_path)
    config = SimpleNamespace(
        input_dir=tmp_path,
        device=SimpleNamespace(
            construct_from_unit_cell=True,
            transport_direction="z",
            kpoint_grid=(1, 1, 1),
            neighbor_cell_cutoff=(0, 0, 1),
            num_transport_cells=1,
        ),
        coulomb_screening=SimpleNamespace(
            num_k_points=8,
            num_q_points=3,
            interaction_cutoff=8.0,
            temperature=300.0,
            epsilon_r=1.0,
            hamiltonian_matrix_name="hamiltonian",
            coulomb_matrix_name="coulomb_matrix",
            periodic_axis=2,
            lattice_constant=1.0,
            broadening=0.0,
            include_zero_q=True,
            chemical_potential=0.0,
        ),
        electron=SimpleNamespace(fermi_level=0.0, left_fermi_level=0.0),
    )
    template = SimpleNamespace(stack_section_sizes=np.array([2]))
    bridge = EquilibriumRPAScreeningBridge(
        config,
        screening_energies=np.array([1e-6, 0.25]),
        template=template,
    )
    mesh = bridge._build_mesh()
    q_values = np.ones((mesh.q_points.size, mesh.frequencies.size, 1, 1), dtype=np.complex128)

    matrices = bridge._build_transport_matrices(mesh, q_values)

    assert len(matrices) == mesh.frequencies.size
    assert matrices[0].shape == (1, 1)
    assert_allclose(matrices[0].toarray(), matrices[1].toarray())
    assert np.isfinite(matrices[0].toarray()).all()
    assert abs(matrices[0].toarray()[0, 0]) > 0.0