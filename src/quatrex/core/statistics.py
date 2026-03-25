# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import math

from qttools import NDArray, xp
from quatrex.core.constants import k_B


def fermi_dirac(energy: float | NDArray, temperature: float) -> float | NDArray:
    """Fermi-Dirac distribution for given energy and temperature.

    Parameters
    ----------
    energy : float or NDArray
        Energy in eV.
    temperature : float
        Temperature in K.

    Returns
    -------
    float or NDArray
        Fermi-Dirac occupancy.

    """
    return 1 / (1 + xp.exp(energy / (k_B * temperature)))


def bose_einstein(energy: float | NDArray, temperature: float) -> float | NDArray:
    """Bose-Einstein distribution for given energy and temperature.

    Parameters
    ----------
    energy : float or NDArray
        Energy in eV.
    temperature : float
        Temperature in K.

    Returns
    -------
    float or NDArray
        Bose-Einstein occupancy.

    """
    if xp.isscalar(energy):
        if energy == 0:
            return 0.0
        return 1.0 / math.expm1(energy / (k_B * temperature))

    # If a zero energy is encountered, return zero occupancy to avoid division by zero.
    return xp.where(energy == 0, 0, 1 / xp.expm1(energy / (k_B * temperature)))
