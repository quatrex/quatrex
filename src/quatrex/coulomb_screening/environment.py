# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Environment-renormalized Coulomb interactions for non-RPA screening.

This module contains small preprocessing helpers for the non-RPA Coulomb
screening path.  The existing solver expects an interaction matrix and then
computes active-region screening from the NEGF polarization.  The routines here
build an effective interaction where a dielectric environment has already been
integrated out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FiveRegionDielectricStack:
    """Five-region dielectric geometry for two active layers.

    The regions are ordered as in the Appendix-F slab setup:

    ``kappa_1`` for ``z > d1``,
    ``kappa_2`` for ``0 < z < d1``,
    ``kappa_3`` for ``-l < z < 0``,
    ``kappa_4`` for ``-l - d2 < z < -l``, and
    ``kappa_5`` for ``z < -l - d2``.

    The active layers are at ``z = 0`` and ``z = -l``.
    """

    kappa_1: float
    kappa_2: float
    kappa_3: float
    kappa_4: float
    kappa_5: float
    d1: float
    d2: float
    layer_separation: float

    @property
    def kappas(self) -> tuple[float, float, float, float, float]:
        """Return dielectric constants in top-to-bottom order."""

        return (
            self.kappa_1,
            self.kappa_2,
            self.kappa_3,
            self.kappa_4,
            self.kappa_5,
        )

    def swapped_layers(self) -> "FiveRegionDielectricStack":
        """Return the stack seen by the second layer as the source layer."""

        return FiveRegionDielectricStack(
            kappa_1=self.kappa_5,
            kappa_2=self.kappa_4,
            kappa_3=self.kappa_3,
            kappa_4=self.kappa_2,
            kappa_5=self.kappa_1,
            d1=self.d2,
            d2=self.d1,
            layer_separation=self.layer_separation,
        )


def _validate_stack(stack: FiveRegionDielectricStack) -> None:
    for index, kappa in enumerate(stack.kappas, start=1):
        if kappa <= 0.0:
            raise ValueError(f"kappa_{index} must be positive.")
    if stack.d1 < 0.0:
        raise ValueError("d1 must be non-negative.")
    if stack.d2 < 0.0:
        raise ValueError("d2 must be non-negative.")
    if stack.layer_separation <= 0.0:
        raise ValueError("layer_separation must be positive.")


def _solve_source_at_top_layer(
    q: float,
    stack: FiveRegionDielectricStack,
    *,
    source_strength: complex,
) -> tuple[complex, complex]:
    """Solve the slab boundary equations for a source at ``z = 0``.

    Returns the potential at the source layer and at the second layer.  The
    normalization of the source discontinuity is supplied by ``source_strength``;
    use the convention needed by the caller's Coulomb units.
    """

    if q <= 0.0:
        raise ValueError("q must be positive.")
    _validate_stack(stack)

    k1, k2, k3, k4, k5 = stack.kappas
    d1 = stack.d1
    d2 = stack.d2
    ell = stack.layer_separation

    exp_d1 = np.exp(q * d1)
    exp_minus_d1 = np.exp(-q * d1)
    exp_l = np.exp(q * ell)
    exp_minus_l = np.exp(-q * ell)
    exp_ld2 = np.exp(q * (ell + d2))
    exp_minus_ld2 = np.exp(-q * (ell + d2))

    matrix = np.array(
        [
            [exp_minus_d1, -exp_d1, -exp_minus_d1, 0, 0, 0, 0, 0],
            [0, 1, 1, -1, -1, 0, 0, 0],
            [0, 0, 0, exp_minus_l, exp_l, -exp_minus_l, -exp_l, 0],
            [0, 0, 0, 0, 0, exp_minus_ld2, exp_ld2, -exp_minus_ld2],
            [-k1 * exp_minus_d1, -k2 * exp_d1, k2 * exp_minus_d1, 0, 0, 0, 0, 0],
            [0, k2, -k2, -k3, k3, 0, 0, 0],
            [0, 0, 0, k3 * exp_minus_l, -k3 * exp_l, -k4 * exp_minus_l, k4 * exp_l, 0],
            [0, 0, 0, 0, 0, k4 * exp_minus_ld2, -k4 * exp_ld2, -k5 * exp_minus_ld2],
        ],
        dtype=np.complex128,
    )
    rhs = np.zeros(8, dtype=np.complex128)
    rhs[5] = source_strength

    coefficients = np.linalg.solve(matrix, rhs)
    _, _, _, d_coeff, e_coeff, _, _, _ = coefficients

    source_layer = -(d_coeff + e_coeff)
    second_layer = -(d_coeff * exp_minus_l + e_coeff * exp_l)
    return source_layer, second_layer


def two_layer_effective_interaction(
    q: float,
    stack: FiveRegionDielectricStack,
    *,
    source_strength: complex = 1.0,
) -> NDArray[np.complex128]:
    """Return the 2x2 environment-renormalized interaction for two layers.

    The returned matrix is ordered as ``[[V11, V12], [V21, V22]]``.  ``V11`` and
    ``V12`` are obtained from a source at the top layer.  ``V22`` is obtained from
    the layer-swapped stack, matching the symmetry relation in the slab formula.
    """

    v11, v12 = _solve_source_at_top_layer(
        q,
        stack,
        source_strength=source_strength,
    )
    v22, v21 = _solve_source_at_top_layer(
        q,
        stack.swapped_layers(),
        source_strength=source_strength,
    )
    return np.array([[v11, v12], [v21, v22]], dtype=np.complex128)


def build_layer_interaction_matrix(
    layer_interaction: ArrayLike,
    layer_labels: ArrayLike,
) -> NDArray[np.complex128]:
    """Expand a layer-space interaction matrix into an orbital-space matrix.

    ``layer_labels`` assigns each orbital/site to an integer layer index.  For a
    two-layer interaction, valid labels are ``0`` and ``1``.
    """

    layer_values = np.asarray(layer_interaction, dtype=np.complex128)
    labels = np.asarray(layer_labels, dtype=np.int64)

    if layer_values.ndim != 2 or layer_values.shape[0] != layer_values.shape[1]:
        raise ValueError("layer_interaction must be a square matrix.")
    if labels.ndim != 1:
        raise ValueError("layer_labels must be a one-dimensional array.")
    if labels.size == 0:
        raise ValueError("layer_labels must not be empty.")
    if labels.min() < 0 or labels.max() >= layer_values.shape[0]:
        raise ValueError("layer_labels contain indices outside layer_interaction.")

    return layer_values[labels[:, np.newaxis], labels[np.newaxis, :]]


def apply_layer_interaction_scaling(
    matrix: ArrayLike,
    layer_labels: ArrayLike,
    target_layer_interaction: ArrayLike,
    reference_layer_interaction: ArrayLike,
) -> NDArray[np.complex128]:
    """Scale an existing orbital interaction matrix by layer-pair ratios.

    This is useful when an existing ``coulomb_matrix.mat`` already contains the
    orbital ordering and sparsity one wants to preserve.  The element ``(i, j)``
    is multiplied by
    ``target_layer_interaction[layer_i, layer_j]`` divided by
    ``reference_layer_interaction[layer_i, layer_j]``.
    """

    values = np.asarray(matrix, dtype=np.complex128)
    target = np.asarray(target_layer_interaction, dtype=np.complex128)
    reference = np.asarray(reference_layer_interaction, dtype=np.complex128)

    if target.shape != reference.shape:
        raise ValueError(
            "target and reference layer interactions must have the same shape."
        )
    if np.any(reference == 0):
        raise ValueError("reference_layer_interaction must not contain zeros.")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be a square matrix.")

    scale = build_layer_interaction_matrix(target / reference, layer_labels)
    if scale.shape != values.shape:
        raise ValueError("layer_labels length must match matrix dimensions.")
    return values * scale


__all__ = [
    "FiveRegionDielectricStack",
    "apply_layer_interaction_scaling",
    "build_layer_interaction_matrix",
    "two_layer_effective_interaction",
]
