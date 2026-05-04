# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from pathlib import Path
from typing import Callable

import pytest

from qttools import xp
from quatrex.bandstructure.band_edges import _compute_eigenvalues
from quatrex.core.config import parse_config, setup_context
from quatrex.device.inputs import load_matrix, load_structure
from quatrex.grid import get_electron_energies


def test_subsectioning(
    example: tuple[Path, bool],
    tmp_path: Path,
    adjust_config_paths: Callable,
):
    """Test that the eigenvalues computed with subsectioning are consistent with those computed without subsectioning."""

    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)

    config = parse_config(tmp_config_path)

    # test only meaningful if block_sections > 1
    if config.compute.band_edge.block_sections == 1:
        pytest.skip("Skipping test for block_sections=1.")
    # skip if `wf` formalism
    if config.formalism == "wf":
        pytest.skip("Skipping test for 'wf' formalism.")

    setup_context(config)

    # Load the device Hamiltonian.
    hamiltonian, sparsity_pattern = load_matrix(
        config=config,
        matrix_name="hamiltonian",
        sparsity_pattern=None,
        shift_kpoints=False,
    )
    try:
        overlap, __ = load_matrix(
            config=config,
            matrix_name="overlap",
            sparsity_pattern=None,
            shift_kpoints=False,
        )
    except FileNotFoundError:
        overlap = None

    potential = xp.zeros(hamiltonian.shape[-1], dtype=hamiltonian.dtype)

    block_sizes, __ = load_structure(config)
    energies = get_electron_energies(config)
    kpoint_grid = config.device.kpoint_grid

    dsdbsparse_type = config.compute.dsdbsparse_type

    # dummy self-energy
    sigma_retarded = dsdbsparse_type.from_sparray(
        sparsity_pattern.astype(xp.complex128),
        block_sizes=block_sizes,
        global_stack_shape=energies.shape + tuple([k for k in kpoint_grid if k > 1]),
    )
    sigma_retarded.data[:] = 0

    section_offsets = xp.array([0, len(energies)])

    ind_left = xp.argmin(xp.abs(energies - config.electron.conduction_band_edge))
    rank_left = xp.digitize(ind_left, section_offsets) - 1
    local_ind = (ind_left - section_offsets[rank_left],) + tuple(
        # Only take the k point indices
        [s // 2 for s in sigma_retarded.shape[1:-2]]
    )

    e_0_test = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_retarded=sigma_retarded,
        ind=local_ind,
        diagonal_inds=(0, 0),
        upper_inds=(0, 1),
        block_sections=config.compute.band_edge.block_sections,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    e_0_ref = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_retarded=sigma_retarded,
        ind=local_ind,
        diagonal_inds=(0, 0),
        upper_inds=(0, 1),
        block_sections=1,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    # check that all test eigenvalues are included in the reference eigenvalues
    for e in e_0_test:
        assert xp.any(xp.isclose(e, e_0_ref))


def test_left_right(
    example: tuple[Path, bool],
    tmp_path: Path,
    adjust_config_paths: Callable,
):
    """Test that the eigenvalues computed for the left and right band edges are consistent."""

    example_path, distributed = example

    if distributed:
        pytest.skip("Skipping single-rank test for distributed example.")

    # Set up reference and temporary configs.
    quatrex_config_path = example_path / "quatrex_config.toml"
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(quatrex_config_path, tmp_config_path)

    config = parse_config(tmp_config_path)

    # NOTE: This test is only meaningful
    # if the contacts are the same.
    # All current examples have the same contacts.
    # TODO: Skip test is contacts are different.

    # skip if `wf` formalism
    if config.formalism == "wf":
        pytest.skip("Skipping test for 'wf' formalism.")

    setup_context(config)

    # Load the device Hamiltonian.
    hamiltonian, sparsity_pattern = load_matrix(
        config=config,
        matrix_name="hamiltonian",
        sparsity_pattern=None,
        shift_kpoints=False,
    )
    try:
        overlap, __ = load_matrix(
            config=config,
            matrix_name="overlap",
            sparsity_pattern=None,
            shift_kpoints=False,
        )
    except FileNotFoundError:
        overlap = None

    potential = xp.zeros(hamiltonian.shape[-1], dtype=hamiltonian.dtype)

    block_sizes, __ = load_structure(config)
    energies = get_electron_energies(config)
    kpoint_grid = config.device.kpoint_grid

    dsdbsparse_type = config.compute.dsdbsparse_type

    # dummy self-energy
    sigma_retarded = dsdbsparse_type.from_sparray(
        sparsity_pattern.astype(xp.complex128),
        block_sizes=block_sizes,
        global_stack_shape=energies.shape + tuple([k for k in kpoint_grid if k > 1]),
    )
    sigma_retarded.data[:] = 0

    section_offsets = xp.array([0, len(energies)])

    ind_left = xp.argmin(xp.abs(energies - config.electron.conduction_band_edge))
    rank_left = xp.digitize(ind_left, section_offsets) - 1
    local_ind = (ind_left - section_offsets[rank_left],) + tuple(
        # Only take the k point indices
        [s // 2 for s in sigma_retarded.shape[1:-2]]
    )

    e_0_left = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_retarded=sigma_retarded,
        ind=local_ind,
        diagonal_inds=(0, 0),
        upper_inds=(0, 1),
        block_sections=config.compute.band_edge.block_sections,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    n = hamiltonian.num_local_blocks - 1
    m = n - 1
    e_0_right = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_retarded=sigma_retarded,
        ind=local_ind,
        diagonal_inds=(n, n),
        upper_inds=(n, m),
        order="reverse",
        block_sections=config.compute.band_edge.block_sections,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    assert xp.allclose(e_0_left, e_0_right)
