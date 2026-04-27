# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import importlib.util
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose

MODULE_PATH = Path(__file__).parents[3] / "src" / "quatrex" / "coulomb_screening"
MODULE_PATH = MODULE_PATH / "block_screening.py"
SPEC = importlib.util.spec_from_file_location("block_screening", MODULE_PATH)
block_screening = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = block_screening
SPEC.loader.exec_module(block_screening)

solve_central_screened_interaction = block_screening.solve_central_screened_interaction
solve_environment_dressed_interaction = (
    block_screening.solve_environment_dressed_interaction
)


def test_environment_dressed_interaction_reduces_to_central_coulomb_without_coupling():
    v_c = np.array([[2.0, 0.2], [0.2, 1.5]], dtype=np.complex128)
    v_ee = np.array([[1.0, 0.1], [0.1, 0.8]], dtype=np.complex128)
    p_ee_retarded = np.array(
        [[0.05 + 0.01j, 0.02], [0.02, 0.04 + 0.02j]],
        dtype=np.complex128,
    )
    p_ee_lesser = 1j * np.array([[0.01, 0.002], [0.002, 0.015]])

    result = solve_environment_dressed_interaction(
        v_c=v_c,
        v_ee=v_ee,
        v_ce=np.zeros((2, 2), dtype=np.complex128),
        v_ec=np.zeros((2, 2), dtype=np.complex128),
        p_ee_retarded=p_ee_retarded,
        p_ee_lesser=p_ee_lesser,
    )

    assert_allclose(result.retarded, v_c)
    assert_allclose(result.lesser, np.zeros_like(v_c))
    assert_allclose(result.advanced, v_c.conj().T)


def test_environment_dressed_interaction_reduces_to_central_coulomb_without_environment_polarization():
    v_c = np.array([[2.0, 0.2], [0.2, 1.5]], dtype=np.complex128)
    v_ee = np.array([[1.0, 0.1], [0.1, 0.8]], dtype=np.complex128)
    v_ce = np.array([[0.4, 0.05], [0.03, 0.2]], dtype=np.complex128)
    v_ec = v_ce.conj().T

    result = solve_environment_dressed_interaction(
        v_c=v_c,
        v_ee=v_ee,
        v_ce=v_ce,
        v_ec=v_ec,
        p_ee_retarded=np.zeros_like(v_ee),
        p_ee_lesser=np.zeros_like(v_ee),
    )

    assert_allclose(result.retarded, v_c)
    assert_allclose(result.lesser, np.zeros_like(v_c))
    assert_allclose(result.advanced, v_c.conj().T)


def test_central_screened_interaction_reduces_to_environment_input_without_central_polarization():
    w_e_retarded = np.array([[2.0, 0.1], [0.1, 1.3]], dtype=np.complex128)
    w_e_lesser = 1j * np.array([[0.05, 0.01], [0.01, 0.03]])
    zero = np.zeros_like(w_e_retarded)

    result = solve_central_screened_interaction(
        w_environment_retarded=w_e_retarded,
        w_environment_lesser=w_e_lesser,
        p_c_retarded=zero,
        p_c_lesser=zero,
    )

    assert_allclose(result.retarded, w_e_retarded)
    assert_allclose(result.lesser, w_e_lesser)
    assert_allclose(result.advanced, w_e_retarded.conj().T)


def test_central_screened_interaction_satisfies_retarded_dyson_equation():
    w_e_retarded = np.array(
        [[1.0 + 0.02j, 0.1], [0.1, 0.8 + 0.01j]],
        dtype=np.complex128,
    )
    p_c_retarded = np.array(
        [[0.08 + 0.01j, 0.01], [0.02, 0.06 + 0.02j]],
        dtype=np.complex128,
    )
    w_e_lesser = 1j * np.array([[0.02, 0.004], [0.004, 0.01]])
    p_c_lesser = 1j * np.array([[0.01, 0.002], [0.002, 0.015]])

    result = solve_central_screened_interaction(
        w_environment_retarded=w_e_retarded,
        w_environment_lesser=w_e_lesser,
        p_c_retarded=p_c_retarded,
        p_c_lesser=p_c_lesser,
    )

    assert_allclose(
        result.retarded,
        w_e_retarded + w_e_retarded @ p_c_retarded @ result.retarded,
    )
