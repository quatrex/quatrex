from types import SimpleNamespace

import numpy as np
import scipy.io
from numpy.testing import assert_allclose

from quatrex.coulomb_screening.dielectric_screening.negf_bridge import (
    EquilibriumRPAScreeningBridge,
    _dress_environment_interactions,
)


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
    environment_screening = SimpleNamespace(
        num_k_points=8,
        num_q_points=3,
        periodic_axis=2,
        lattice_constant=1.0,
        include_zero_q=True,
        chemical_potential=0.0,
    )
    config = SimpleNamespace(
        input_dir=tmp_path,
        device=SimpleNamespace(
            construct_from_unit_cell=True,
            transport_direction="z",
            kpoint_grid=(2, 1, 1),
            neighbor_cell_cutoff=(0, 0, 1),
            num_transport_cells=1,
        ),
        environment_screening=environment_screening,
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
    environment_screening = SimpleNamespace(
        num_k_points=8,
        num_q_points=3,
        periodic_axis=2,
        lattice_constant=1.0,
        include_zero_q=True,
        chemical_potential=0.0,
    )
    config = SimpleNamespace(
        input_dir=tmp_path,
        device=SimpleNamespace(
            construct_from_unit_cell=True,
            transport_direction="z",
            kpoint_grid=(1, 1, 1),
            neighbor_cell_cutoff=(0, 0, 1),
            num_transport_cells=1,
        ),
        environment_screening=environment_screening,
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
    q_values = np.ones(
        (mesh.q_points.size, mesh.frequencies.size, 1, 1), dtype=np.complex128
    )

    matrices = bridge._build_transport_matrices(config, mesh, q_values)

    assert len(matrices) == mesh.frequencies.size
    assert matrices[0].shape == (1, 1)
    assert_allclose(matrices[0].toarray(), matrices[1].toarray())
    assert np.isfinite(matrices[0].toarray()).all()
    assert abs(matrices[0].toarray()[0, 0]) > 0.0


def test_dress_environment_interactions_uses_block_screening_equations():
    v_c = np.array([[2.0]], dtype=np.complex128)
    v_ee = np.array([[1.5]], dtype=np.complex128)
    v_ce = np.array([[0.25]], dtype=np.complex128)
    v_ec = np.array([[0.25]], dtype=np.complex128)
    p_ee_retarded = scipy.sparse.coo_matrix(np.array([[0.1 + 0.05j]]))
    p_ee_lesser = scipy.sparse.coo_matrix(np.array([[0.02j]]))

    result = _dress_environment_interactions(
        v_c=v_c,
        v_ee=v_ee,
        v_ce=v_ce,
        v_ec=v_ec,
        p_ee_retarded_matrices=[p_ee_retarded],
        p_ee_lesser_matrices=[p_ee_lesser],
    )

    assert len(result.w_retarded_matrices) == 1
    assert len(result.w_lesser_matrices) == 1
    assert len(result.w_greater_matrices) == 1

    w_retarded = result.w_retarded_matrices[0].toarray()
    w_lesser = result.w_lesser_matrices[0].toarray()
    w_greater = result.w_greater_matrices[0].toarray()

    assert w_retarded.shape == (1, 1)
    assert w_lesser.shape == (1, 1)
    assert w_greater.shape == (1, 1)
    assert abs(w_retarded[0, 0]) > abs(v_c[0, 0])
    assert_allclose(w_greater - w_lesser, w_retarded - w_retarded.conj().T)


def test_negf_rpa_bridge_loads_saved_environment_screening(tmp_path):
    np.save(tmp_path / "p_ee_retarded.npy", np.zeros((2, 1, 1), dtype=np.complex128))
    np.save(tmp_path / "p_ee_lesser.npy", np.zeros((2, 1, 1), dtype=np.complex128))
    np.save(tmp_path / "v_ee.npy", np.array([[1.5]], dtype=np.complex128))

    config = SimpleNamespace(
        environment_screening=SimpleNamespace(input_dir=tmp_path),
        coulomb_screening=SimpleNamespace(),
    )
    bridge = EquilibriumRPAScreeningBridge(
        config,
        screening_energies=np.array([1e-6, 0.25]),
        template=SimpleNamespace(stack_section_sizes=np.array([2])),
    )

    v_ee, p_retarded_matrices, p_lesser_matrices = (
        bridge._load_saved_environment_screening()
    )

    assert_allclose(v_ee, np.array([[1.5]], dtype=np.complex128))
    assert len(p_retarded_matrices) == 2
    assert len(p_lesser_matrices) == 2
    assert p_retarded_matrices[0].shape == (1, 1)
