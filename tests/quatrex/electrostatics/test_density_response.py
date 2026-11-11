# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np
import pytest
from scipy.integrate import quad

from quatrex.core.constants import hbar, k_B
from quatrex.electrostatics.density_response import (
    GrapheneDensityModel,
    dirac_net_density,
    dirac_net_density_derivative,
    inverse_dirac_net_density,
)
from quatrex.electrostatics.fermi_integrals import fermi_integral

AREA_PER_ATOM = 3 * np.sqrt(3) / 4 * 1.42**2  # Å^2
ETAS = np.array([-40.0, -10.0, -2.5, -0.3, 0.0, 0.3, 2.5, 10.0, 40.0])


def _fermi_integral_one_quad(eta: float) -> float:
    """Reference Fermi integral of order 1 by adaptive quadrature."""
    return quad(lambda x: x / (1 + np.exp(x - eta)), 0, max(eta, 0) + 60)[0]


def test_fermi_integral_one():
    """Tests the analytic order-1 Fermi integral against quadrature."""
    reference = np.array([_fermi_integral_one_quad(eta) for eta in ETAS])
    assert np.allclose(fermi_integral(1, ETAS), reference, rtol=1e-10, atol=1e-14)


def test_dirac_net_density():
    """Tests the net density against quadrature and checks its parity."""
    reference = np.array(
        [_fermi_integral_one_quad(eta) - _fermi_integral_one_quad(-eta) for eta in ETAS]
    )
    g = dirac_net_density(ETAS)
    assert np.allclose(g, reference, rtol=1e-10, atol=1e-12)
    assert np.allclose(g, -dirac_net_density(-ETAS))
    assert dirac_net_density(0.0) == 0.0


def test_dirac_net_density_derivative():
    """Tests the derivative against a central finite difference."""
    h = 1e-5
    finite_difference = (dirac_net_density(ETAS + h) - dirac_net_density(ETAS - h)) / (
        2 * h
    )
    assert np.allclose(dirac_net_density_derivative(ETAS), finite_difference, rtol=1e-7)
    assert np.isclose(dirac_net_density_derivative(0.0), 2 * np.log(2))


def test_inverse_dirac_net_density():
    """Tests that the inverse round-trips over many orders of magnitude."""
    u = np.concatenate([-np.logspace(-8, 4, 50), [0.0], np.logspace(-8, 4, 50)])
    assert np.allclose(dirac_net_density(inverse_dirac_net_density(u)), u, rtol=1e-10)


@pytest.mark.parametrize(
    "sheet_density, expected_fermi_level",
    [
        pytest.param(1e11, 0.0184, id="1e11"),
        pytest.param(1e12, 0.1070, id="1e12"),
        pytest.param(1e13, 0.3659, id="1e13"),
        pytest.param(-1e12, -0.1070, id="-1e12"),
    ],
)
def test_graphene_fermi_level(sheet_density: float, expected_fermi_level: float):
    """Tests E_F - E_D for a given sheet density in cm^-2 at 300 K."""
    density = np.array([sheet_density * 1e-16 * AREA_PER_ATOM])
    model = GrapheneDensityModel(density, potential=np.zeros(1))
    assert np.allclose(model.charge_neutrality_level, expected_fermi_level, atol=1e-3)


def test_graphene_effective_dos():
    """Tests the effective DOS against N_G = 2/pi (kT / hbar v_F)^2."""
    model = GrapheneDensityModel(np.zeros(1), np.zeros(1))
    hbar_v_F = hbar * 1e6 * 1e10  # eV Å
    expected = 2 / np.pi * (k_B * 300 / hbar_v_F) ** 2 * AREA_PER_ATOM
    assert np.isclose(model.effective_dos, expected)
    # Roughly 9.8e10 cm^-2 at 300 K.
    assert np.isclose(model.effective_dos / AREA_PER_ATOM * 1e16, 9.8e10, rtol=1e-2)


def test_graphene_density_model():
    """Tests consistency, ambipolarity and the derivative of the model."""
    rng = np.random.default_rng(0)
    density = rng.uniform(-1e-3, 1e-3, 20)
    density[:3] = 0.0  # Charge-neutral sites must be handled as well.
    potential = rng.uniform(-0.2, 0.2, 20)

    model = GrapheneDensityModel(density, potential)

    # Reproduces the input density at the input potential.
    assert np.allclose(model.density(potential), density, rtol=1e-9, atol=1e-15)

    # Density can change sign: a large shift flips all sites.
    assert np.all(model.density(potential + 1.0) < 0)
    assert np.all(model.density(potential - 1.0) > 0)

    # Derivative is negative everywhere (also at neutrality) and matches
    # a finite difference.
    h = 1e-6
    finite_difference = (
        model.density(potential + h) - model.density(potential - h)
    ) / (2 * h)
    derivative = model.density_derivative(potential)
    assert np.all(derivative < 0)
    assert np.allclose(derivative, finite_difference, rtol=1e-6)
