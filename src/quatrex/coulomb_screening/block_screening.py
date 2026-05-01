# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexMatrix = NDArray[np.complex128]

__all__ = [
    "CentralScreenedInteraction",
    "EnvironmentDressedInteraction",
    "solve_central_screened_interaction",
    "solve_environment_dressed_interaction",
]


@dataclass(frozen=True)
class EnvironmentDressedInteraction:
    """Central-region interaction after dressing by environment blocks."""

    retarded: ComplexMatrix
    lesser: ComplexMatrix
    advanced: ComplexMatrix


@dataclass(frozen=True)
class CentralScreenedInteraction:
    """Central-region interaction after central polarization screening."""

    retarded: ComplexMatrix
    lesser: ComplexMatrix
    advanced: ComplexMatrix


def _as_square_matrix(matrix: ArrayLike, *, name: str) -> ComplexMatrix:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return array


def _as_rectangular_matrix(
    matrix: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int],
) -> ComplexMatrix:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return array


def _advanced(matrix: ComplexMatrix) -> ComplexMatrix:
    return matrix.conj().T


def _solve_right(matrix: ComplexMatrix, system: ComplexMatrix) -> ComplexMatrix:
    """Return ``matrix @ inv(system)`` without forming the inverse explicitly."""

    return np.linalg.solve(system.T, matrix.T).T


def solve_environment_dressed_interaction(
    *,
    v_c: ArrayLike,
    v_ee: ArrayLike,
    v_ce: ArrayLike,
    v_ec: ArrayLike,
    p_ee_retarded: ArrayLike,
    p_ee_lesser: ArrayLike,
    p_ee_advanced: ArrayLike | None = None,
) -> EnvironmentDressedInteraction:
    r"""Combine central and environment blocks into an environment-dressed interaction.

    This implements the reduced block-screening equations for ``P_ec = P_ce = 0``:

    ``epsilon_E^{-1,R} = (1 - v_ee P_ee^R)^{-1}``

    ``W_E^R = v_c + v_ce P_ee^R epsilon_E^{-1,R} v_ec``

    ``W_E^< = v_ce (P_ee^R epsilon_E^{-1,<} + P_ee^< epsilon_E^{-1,A}) v_ec``

    with ``epsilon_E^{-1,<} = epsilon_E^{-1,R} v_ee P_ee^< epsilon_E^{-1,A}``.
    The advanced polarization defaults to the Hermitian conjugate of the retarded
    polarization.
    """

    v_c_array = _as_square_matrix(v_c, name="v_c")
    v_ee_array = _as_square_matrix(v_ee, name="v_ee")
    central_size = v_c_array.shape[0]
    environment_size = v_ee_array.shape[0]

    v_ce_array = _as_rectangular_matrix(
        v_ce,
        name="v_ce",
        shape=(central_size, environment_size),
    )
    v_ec_array = _as_rectangular_matrix(
        v_ec,
        name="v_ec",
        shape=(environment_size, central_size),
    )
    p_ee_r = _as_rectangular_matrix(
        p_ee_retarded,
        name="p_ee_retarded",
        shape=v_ee_array.shape,
    )
    p_ee_l = _as_rectangular_matrix(
        p_ee_lesser,
        name="p_ee_lesser",
        shape=v_ee_array.shape,
    )
    p_ee_a = (
        _advanced(p_ee_r)
        if p_ee_advanced is None
        else _as_rectangular_matrix(
            p_ee_advanced,
            name="p_ee_advanced",
            shape=v_ee_array.shape,
        )
    )

    identity_e = np.eye(environment_size, dtype=np.complex128)
    epsilon_system_r = identity_e - v_ee_array @ p_ee_r
    epsilon_system_a = identity_e - v_ee_array @ p_ee_a

    epsilon_inv_r = np.linalg.solve(epsilon_system_r, identity_e)
    epsilon_inv_a = np.linalg.solve(epsilon_system_a, identity_e)
    epsilon_inv_l = epsilon_inv_r @ v_ee_array @ p_ee_l @ epsilon_inv_a

    w_e_r = v_c_array + v_ce_array @ p_ee_r @ epsilon_inv_r @ v_ec_array
    w_e_l = v_ce_array @ (p_ee_r @ epsilon_inv_l + p_ee_l @ epsilon_inv_a) @ v_ec_array
    w_e_a = _advanced(w_e_r)

    return EnvironmentDressedInteraction(
        retarded=w_e_r,
        lesser=w_e_l,
        advanced=w_e_a,
    )


def solve_central_screened_interaction(
    *,
    w_environment_retarded: ArrayLike,
    w_environment_lesser: ArrayLike,
    p_c_retarded: ArrayLike,
    p_c_lesser: ArrayLike,
    w_environment_advanced: ArrayLike | None = None,
    p_c_advanced: ArrayLike | None = None,
) -> CentralScreenedInteraction:
    r"""Screen the environment-dressed interaction by central-region polarization.

    The retarded component is solved from
    ``W_c^R = (1 - W_E^R P_c^R)^{-1} W_E^R``.

    The lesser component is
    ``W_c^< = L W_E^< R + W_c^R P_c^< W_c^A``, where
    ``L = (1 - W_E^R P_c^R)^{-1}`` and
    ``R = (1 - P_c^A W_E^A)^{-1}``.
    """

    w_e_r = _as_square_matrix(
        w_environment_retarded,
        name="w_environment_retarded",
    )
    central_size = w_e_r.shape[0]
    w_e_l = _as_rectangular_matrix(
        w_environment_lesser,
        name="w_environment_lesser",
        shape=w_e_r.shape,
    )
    p_c_r = _as_rectangular_matrix(
        p_c_retarded,
        name="p_c_retarded",
        shape=w_e_r.shape,
    )
    p_c_l = _as_rectangular_matrix(
        p_c_lesser,
        name="p_c_lesser",
        shape=w_e_r.shape,
    )
    w_e_a = (
        _advanced(w_e_r)
        if w_environment_advanced is None
        else _as_rectangular_matrix(
            w_environment_advanced,
            name="w_environment_advanced",
            shape=w_e_r.shape,
        )
    )
    p_c_a = (
        _advanced(p_c_r)
        if p_c_advanced is None
        else _as_rectangular_matrix(
            p_c_advanced,
            name="p_c_advanced",
            shape=w_e_r.shape,
        )
    )

    identity_c = np.eye(central_size, dtype=np.complex128)
    left_system = identity_c - w_e_r @ p_c_r
    right_system = identity_c - p_c_a @ w_e_a

    left_factor = np.linalg.solve(left_system, identity_c)
    right_factor = np.linalg.solve(right_system, identity_c)

    w_c_r = left_factor @ w_e_r
    w_c_a = _solve_right(w_e_a, right_system)
    w_c_l = left_factor @ w_e_l @ right_factor + w_c_r @ p_c_l @ w_c_a

    return CentralScreenedInteraction(
        retarded=w_c_r,
        lesser=w_c_l,
        advanced=w_c_a,
    )
