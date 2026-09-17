# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np
import pytest

from quatrex.electrostatics.fermi_integrals import (
    fermi_integral,
    inverse_fermi_integral,
)

ETAS = [
    pytest.param(-3, id="-3"),
    pytest.param(2, id="2"),
    pytest.param(10, id="10"),
]

ORDERS = [
    pytest.param(-1, id="-1"),
    pytest.param(-1 / 2, id="-1/2"),
    pytest.param(0, id="0"),
    pytest.param(1 / 2, id="1/2"),
]


@pytest.fixture(params=ETAS)
def eta(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=ORDERS)
def order(request: pytest.FixtureRequest) -> int:
    return request.param


def test_inverse_fermi_integral_numerical(order, eta):
    """Tests the numerical inverse Fermi integral."""
    # The inverse Fermi integral should return a value close to
    # the original eta when applied to the Fermi integral.
    u = fermi_integral(order, eta)
    eta_computed = inverse_fermi_integral(order, u, method="numerical")
    assert np.isclose(
        eta, eta_computed, rtol=1e-3
    ), f"Inverse Fermi integral did not return the original eta for order {order} and eta {eta}"


def test_inverse_fermi_integral_approximate(order, eta):
    """Tests the approximate inverse Fermi integral."""
    u = fermi_integral(order, eta)
    eta_computed = inverse_fermi_integral(order, u, method="approximate")
    assert np.isclose(
        eta, eta_computed, rtol=1e-3
    ), f"Inverse Fermi integral did not return the original eta for order {order} and eta {eta}"
