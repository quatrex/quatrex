# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

MODULE_PATH = Path(__file__).parents[3] / "src" / "quatrex" / "coulomb_screening"
MODULE_PATH = MODULE_PATH / "environment.py"
SPEC = importlib.util.spec_from_file_location("environment", MODULE_PATH)
environment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = environment
SPEC.loader.exec_module(environment)

FiveRegionDielectricStack = environment.FiveRegionDielectricStack
apply_layer_interaction_scaling = environment.apply_layer_interaction_scaling
build_layer_interaction_matrix = environment.build_layer_interaction_matrix
two_layer_effective_interaction = environment.two_layer_effective_interaction


def test_two_layer_effective_interaction_is_symmetric_for_symmetric_stack():
    stack = FiveRegionDielectricStack(
        kappa_1=1.0,
        kappa_2=4.0,
        kappa_3=2.0,
        kappa_4=4.0,
        kappa_5=1.0,
        d1=0.5,
        d2=0.5,
        layer_separation=1.0,
    )

    interaction = two_layer_effective_interaction(
        q=0.25,
        stack=stack,
        source_strength=1.0,
    )

    assert interaction.shape == (2, 2)
    assert np.isfinite(interaction).all()
    assert_allclose(interaction, interaction.T)
    assert_allclose(interaction[0, 0], interaction[1, 1])


def test_two_layer_effective_interaction_matches_layer_swap_relation():
    stack = FiveRegionDielectricStack(
        kappa_1=1.0,
        kappa_2=3.0,
        kappa_3=5.0,
        kappa_4=7.0,
        kappa_5=11.0,
        d1=0.4,
        d2=0.8,
        layer_separation=1.2,
    )

    interaction = two_layer_effective_interaction(
        q=0.3,
        stack=stack,
        source_strength=2.0,
    )
    swapped_interaction = two_layer_effective_interaction(
        q=0.3,
        stack=stack.swapped_layers(),
        source_strength=2.0,
    )

    assert_allclose(interaction[0, 0], swapped_interaction[1, 1])
    assert_allclose(interaction[1, 1], swapped_interaction[0, 0])


def test_build_layer_interaction_matrix_expands_to_orbital_space():
    layer_interaction = np.array([[10.0, 2.0], [2.0, 7.0]])
    layer_labels = np.array([0, 0, 1, 1])

    matrix = build_layer_interaction_matrix(layer_interaction, layer_labels)

    assert_allclose(
        matrix,
        np.array(
            [
                [10.0, 10.0, 2.0, 2.0],
                [10.0, 10.0, 2.0, 2.0],
                [2.0, 2.0, 7.0, 7.0],
                [2.0, 2.0, 7.0, 7.0],
            ],
            dtype=np.complex128,
        ),
    )


def test_apply_layer_interaction_scaling_preserves_matrix_sparsity_values():
    matrix = np.array(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    layer_labels = np.array([0, 0, 1])
    reference = np.array([[2.0, 4.0], [4.0, 8.0]])
    target = np.array([[1.0, 8.0], [8.0, 4.0]])

    scaled = apply_layer_interaction_scaling(
        matrix,
        layer_labels,
        target_layer_interaction=target,
        reference_layer_interaction=reference,
    )

    assert_allclose(
        scaled,
        np.array(
            [
                [0.5, 0.0, 6.0],
                [0.0, 1.0, 8.0],
                [6.0, 8.0, 2.5],
            ],
            dtype=np.complex128,
        ),
    )


@pytest.mark.parametrize("q", [0.0, -0.1])
def test_two_layer_effective_interaction_rejects_nonpositive_q(q):
    stack = FiveRegionDielectricStack(
        kappa_1=1.0,
        kappa_2=1.0,
        kappa_3=1.0,
        kappa_4=1.0,
        kappa_5=1.0,
        d1=1.0,
        d2=1.0,
        layer_separation=1.0,
    )

    with pytest.raises(ValueError, match="q must be positive"):
        two_layer_effective_interaction(q=q, stack=stack)
