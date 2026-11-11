# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import warnings

import numpy as np

from qttools import xp
from quatrex.core.constants import k_B
from quatrex.core.statistics import fermi_dirac


def test_fermi_dirac_matches_logistic():
    """Agrees with 1 / (1 + exp(x)) where the latter does not overflow."""
    temperature = 300.0
    energy = xp.linspace(-1.0, 1.0, 2001)
    reference = 1 / (1 + xp.exp(energy / (k_B * temperature)))
    assert xp.allclose(fermi_dirac(energy, temperature), reference, rtol=1e-12)
    assert np.isclose(float(fermi_dirac(0.0, temperature)), 0.5)


def test_fermi_dirac_no_overflow():
    """Far from the chemical potential the occupation saturates silently."""
    energy = xp.array([-100.0, -20.0, 20.0, 100.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        occupation = fermi_dirac(energy, 300.0)
    assert xp.all(xp.isfinite(occupation))
    assert xp.allclose(occupation, [1.0, 1.0, 0.0, 0.0])
