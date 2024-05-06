# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from quatrex.core.constants import k_B


def fermi_dirac(energy: Union[float, ArrayLike], temperature: float) -> float:
    """Fermi-Dirac distribution for given energy and temperature.

    Parameters
    ----------
    energy : float or array_like
        Energy in eV.
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        Fermi-Dirac occupancy.

    """
    return 1 / (1 + np.exp(energy / (k_B * temperature)))


def bose_einstein(energy: Union[float, ArrayLike], temperature: float) -> float:
    """Bose-Einstein distribution for given energy and temperature.

    Parameters
    ----------
    energy : float or array_like
        Energy in eV.
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        Bose-Einstein occupancy.

    """
    return 1 / (np.exp(energy / (k_B * temperature)) - 1)
