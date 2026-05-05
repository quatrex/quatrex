# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from qttools import xp
from quatrex.bandstructure.band_edges import _compute_eigenvalues
from quatrex.core.config import QuatrexConfig, parse_config, setup_context
from quatrex.device.inputs import assemble_matrix, load_structure
from quatrex.grid import get_electron_energies


def _intialize(config: QuatrexConfig):
    """Helper function to load the device Hamiltonian
    and initialize the self-energy for the band edge tests."""

    # Load the device Hamiltonian.
    hamiltonian, sparsity_pattern = assemble_matrix(
        config=config,
        matrix_name="hamiltonian",
        sparsity_pattern=None,
        shift_kpoints=False,
    )
    try:
        overlap, __ = assemble_matrix(
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
    sigma_dummy = dsdbsparse_type.from_sparray(
        sparsity_pattern.astype(xp.complex128),
        block_sizes=block_sizes,
        global_stack_shape=energies.shape + tuple([k for k in kpoint_grid if k > 1]),
    )
    sigma_dummy.data[:] = 0

    section_offsets = xp.array([0, len(energies)])

    ind_left = xp.argmin(xp.abs(energies - config.electron.conduction_band_edge))
    rank_left = xp.digitize(ind_left, section_offsets) - 1
    local_ind = (ind_left - section_offsets[rank_left],) + tuple(
        # Only take the k point indices
        [s // 2 for s in sigma_dummy.shape[1:-2]]
    )

    return hamiltonian, overlap, potential, sigma_dummy, local_ind


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

    hamiltonian, overlap, potential, sigma_dummy, local_ind = _intialize(config)

    e_0_test = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_lesser=sigma_dummy,
        sigma_greater=sigma_dummy,
        sigma_retarded_hermitian=sigma_dummy,
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
        sigma_lesser=sigma_dummy,
        sigma_greater=sigma_dummy,
        sigma_retarded_hermitian=sigma_dummy,
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

    hamiltonian, overlap, potential, sigma_dummy, local_ind = _intialize(config)

    e_0_left = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_lesser=sigma_dummy,
        sigma_greater=sigma_dummy,
        sigma_retarded_hermitian=sigma_dummy,
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
        sigma_lesser=sigma_dummy,
        sigma_greater=sigma_dummy,
        sigma_retarded_hermitian=sigma_dummy,
        ind=local_ind,
        diagonal_inds=(n, n),
        upper_inds=(n, m),
        order="reverse",
        block_sections=config.compute.band_edge.block_sections,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    assert xp.allclose(e_0_left, e_0_right)


class MockDSDBSparse:

    def __init__(self, data, block_sizes):
        self.data = data
        self.block_sizes = block_sizes

        self.blocks = {}

        # assumes constant block size for simplicity
        block_size = block_sizes[0]
        for i in range(len(block_sizes)):
            for j in range(len(block_sizes)):
                self.blocks[i, j] = data[
                    ...,
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ]


def test_overlap(
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

    # skip if `wf` formalism
    if config.formalism == "wf":
        pytest.skip("Skipping test for 'wf' formalism.")

    setup_context(config)

    hamiltonian, overlap, potential, sigma_dummy, local_ind = _intialize(config)

    if overlap is None:
        pytest.skip("Skipping test for missing overlap matrix.")

    if not np.all(hamiltonian.block_sizes == hamiltonian.block_sizes[0]):
        pytest.skip("Skipping test for non-uniform block sizes.")

    # test with a bit of potential
    # to check that it is correctly included
    potential += 0.1

    e_0_ref = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=overlap,
        potential=potential,
        sigma_lesser=sigma_dummy,
        sigma_greater=sigma_dummy,
        sigma_retarded_hermitian=sigma_dummy,
        ind=local_ind,
        diagonal_inds=(0, 0),
        upper_inds=(0, 1),
        block_sections=config.compute.band_edge.block_sections,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    overlap_gamma = (
        overlap.blocks[0, 0][:, *local_ind[1:]]
        + overlap.blocks[0, 1][:, *local_ind[1:]]
        + (overlap.blocks[0, 1][:, *local_ind[1:]]).conj().swapaxes(-1, -2)
    )

    L = xp.linalg.cholesky(overlap_gamma)
    L_inv = xp.linalg.inv(L)

    block_sizes = hamiltonian.block_sizes
    hamiltonian = hamiltonian.to_dense()
    sigma_dummy = sigma_dummy.to_dense()

    L_inv_full = xp.zeros_like(hamiltonian)
    for i in range(len(block_sizes)):
        L_inv_full[
            ...,
            i * block_sizes[0] : (i + 1) * block_sizes[0],
            i * block_sizes[0] : (i + 1) * block_sizes[0],
        ] = L_inv

    hamiltonian_hat = L_inv_full @ hamiltonian @ L_inv_full.swapaxes(-2, -1).conj()

    # NOTE: a mock class is used since it is not so straightforward
    # to do the correct DSDBSparse -> dense -> DSDBSparse conversion with the current API.
    hamiltonian = MockDSDBSparse(hamiltonian_hat, block_sizes)
    sigma_dummy = MockDSDBSparse(sigma_dummy, block_sizes)

    e_0_test = _compute_eigenvalues(
        hamiltonian=hamiltonian,
        overlap=None,
        potential=xp.zeros_like(potential),
        sigma_lesser=sigma_dummy,
        sigma_greater=sigma_dummy,
        sigma_retarded_hermitian=sigma_dummy,
        ind=local_ind,
        diagonal_inds=(0, 0),
        upper_inds=(0, 1),
        block_sections=config.compute.band_edge.block_sections,
        eigvalsh_compute_location=config.compute.band_edge.eigvalsh_compute_location,
    )

    assert len(e_0_test) == len(e_0_ref)
    # Shift to account for the potential
    assert xp.allclose(e_0_test + 0.1, e_0_ref)
