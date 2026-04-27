from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import scipy.io

from quatrex.dielectric_screening import (
	DielectricPolarization,
	DielectricScreeningSolver,
	EquilibriumScreening,
	RPAPolarization,
	build_uniform_brillouin_zone_mesh,
	compute_screened_coulomb_matrices,
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
	return SimpleNamespace(input_dir=tmp_path)


def test_dielectric_polarization_uses_config_inputs(tmp_path):
	config = _write_input_matrices(tmp_path)
	mesh = build_uniform_brillouin_zone_mesh(
		num_k_points=8,
		num_q_points=3,
		num_frequencies=3,
		max_frequency=0.5,
	)

	polarization = DielectricPolarization(config)
	inputs = polarization.load_inputs()
	result = polarization.compute(
		mesh=mesh,
		chemical_potential=0.0,
		temperature=300.0,
	)
	reference = RPAPolarization().solve_from_config(
		config,
		mesh=mesh,
		chemical_potential=0.0,
		temperature=300.0,
	)

	assert inputs.hamiltonian_file == tmp_path / "hamiltonian.mat"
	assert_allclose(result.band_structure.eigenvalues, reference.band_structure.eigenvalues)
	assert_allclose(result.polarization, reference.polarization)


def test_dielectric_screening_solver_uses_config_inputs(tmp_path):
	config = _write_input_matrices(tmp_path)
	mesh = build_uniform_brillouin_zone_mesh(
		num_k_points=8,
		num_q_points=3,
		num_frequencies=3,
		max_frequency=0.5,
	)

	solver = DielectricScreeningSolver(config)
	inputs = solver.load_inputs()
	result = solver.solve(
		mesh=mesh,
		chemical_potential=0.0,
		temperature=300.0,
		q_index=1,
		frequency_index=1,
	)
	reference = EquilibriumScreening().solve_from_inputs(
		inputs=inputs,
		mesh=mesh,
		chemical_potential=0.0,
		temperature=300.0,
		q_index=1,
		frequency_index=1,
	)

	assert inputs.hamiltonian_blocks
	assert inputs.coulomb_blocks
	assert_allclose(result.coulomb_matrix, reference.coulomb_matrix)
	assert_allclose(result.dielectric_matrix, reference.dielectric_matrix)
	assert_allclose(result.screened_interaction, reference.screened_interaction)


def test_equilibrium_screening_solve_grid_from_inputs(tmp_path):
	config = _write_input_matrices(tmp_path)
	mesh = build_uniform_brillouin_zone_mesh(
		num_k_points=8,
		num_q_points=3,
		num_frequencies=4,
		max_frequency=0.5,
	)

	solver = EquilibriumScreening()
	inputs = solver.load_inputs_from_config(config)
	grid_result = solver.solve_grid_from_inputs(
		inputs=inputs,
		mesh=mesh,
		chemical_potential=0.0,
		temperature=300.0,
	)
	reference_dielectric, reference_screened = compute_screened_coulomb_matrices(
		grid_result.coulomb_matrices,
		grid_result.polarization_result.polarization,
	)

	assert grid_result.coulomb_matrices.shape[:1] == (mesh.q_points.size,)
	assert grid_result.dielectric_matrices.shape[:2] == (
		mesh.q_points.size,
		mesh.frequencies.size,
	)
	assert grid_result.screened_interactions.shape == grid_result.dielectric_matrices.shape
	assert_allclose(grid_result.dielectric_matrices, reference_dielectric)
	assert_allclose(grid_result.screened_interactions, reference_screened)