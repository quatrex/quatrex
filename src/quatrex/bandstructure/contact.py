# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import numpy as np
from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_device, get_host
from scipy.optimize import minimize_scalar

from quatrex.core.statistics import fermi_dirac


def contact_band_structure(
    h_10: NDArray,
    h_00: NDArray,
    h_01: NDArray,
    num_k_points: int | None = None,
) -> NDArray:
    """Computes the band structure of a device contact.

    Parameters
    ----------
    h_10 : NDArray
        The off-diagonal element of the Hamiltonian.
    h_00 : NDArray
        The diagonal element of the Hamiltonian.
    h_01 : NDArray
        The off-diagonal element of the Hamiltonian.
    num_k_points : int, optional
        The number of k points. If not given, only the Gamma point is
        considered.

    Returns
    -------
    e_k : NDArray
        The sorted eigenvalues in energy and k.

    """
    k = (
        xp.linspace(-xp.pi, xp.pi, num_k_points)
        if num_k_points is not None
        else xp.array([0])
    )

    h_k = (
        h_01 * xp.exp(-1j * k)[:, xp.newaxis, xp.newaxis]
        + h_00
        + h_10 * xp.exp(1j * k)[:, xp.newaxis, xp.newaxis]
    )

    e_k = get_device(np.linalg.eigvals(get_host(h_k)))
    return xp.sort(e_k.real, axis=1)


def contact_dos(e_k: NDArray, energies: NDArray) -> NDArray:
    """Computes the density of states of a device contact.

    Parameters
    ----------
    e_k : NDArray
        The sorted eigenvalues in energy and k.
    energies : NDArray
        The energies.

    Returns
    -------
    dos : NDArray
        The density of states.

    """
    dos = np.zeros_like(energies)
    dos[:-1] = np.histogram(e_k, energies)[0]

    # Normalize the density of states.
    dos /= e_k.shape[0] * (energies[1] - energies[0])

    return dos


def contact_fermi_level(
    temperature: float,
    dos: NDArray,
    energies: NDArray,
    doping_density: float,
    midgap_energy: float,
) -> float:
    """Computes the Fermi level of a device contact.

    This is done by minimizing the excess charge difference, while
    taking doping into account.

    Parameters
    ----------
    temperature : float
        The temperature.
    dos : NDArray
        The density of states.
    energies : NDArray
        The energies.
    doping_density : float
        The doping density.
    midgap_energy : float
        The energy at the middle of the band gap. This is used to
        separate conduction from valence bands.

    Returns
    -------
    float
        The Fermi level.

    """
    dE = energies[1] - energies[0]

    def objective_function(fermi_level):
        f = fermi_dirac(energies - fermi_level, temperature)
        n = (f * dos)[energies >= midgap_energy].sum() * dE
        p = ((1 - f) * dos)[energies < midgap_energy].sum() * dE
        return ((n - p) - doping_density) ** 2

    result = minimize_scalar(
        objective_function,
        bounds=(energies.min(), energies.max()),
        method="bounded",
    )

    return result.x
