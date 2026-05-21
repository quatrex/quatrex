# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np
from scipy.optimize import minimize_scalar

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_array_module_name, get_device, get_host
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
    if get_array_module_name(h_k) == "cupy":
        e_k = get_device(np.linalg.eigvals(get_host(h_k)))
    else:
        e_k = np.linalg.eigvals(h_k)
    return xp.sort(e_k.real, axis=1)


def contact_fermi_level(
    e_k: NDArray,
    kpoints: NDArray,
    mid_gap_energy: float,
    cell_volume: float,
    doping_density: float,
    temperature: float = 300,
) -> float:
    """Computes the Fermi level of a device contact.

    The cell volume and the doping density must be given in the same
    units, so both should be in m^3/m^-3 or in nm^3/nm^-3, etc.

    Parameters
    ----------
    e_k : NDArray
        The sorted eigenvalues in energy and k. This should have shape
        (num_kpoints, num_bands).
    kpoints : NDArray
        The corresponding k-points. This should have shape
        (num_kpoints,).
    mid_gap_energy : float
        A guess for the mid-gap energy, which is used to determine the
        number of valence bands.
    cell_volume : float
        The volume of the unit cell. This needs to have the same units
        as the doping density.
    doping_density : float
        The doping density. This needs to have the same units as the
        cell volume.
    temperature : float, optional
        The temperature in K. Default is 300 K.

    Returns
    -------
    fermi_level : float
        The Fermi level in eV.
    mid_gap_energy : float
        The computed mid-gap energy in eV.

    """

    num_valence_bands = (e_k < mid_gap_energy).sum(axis=1).max()
    e_k_valence, e_k_conduction = np.split(e_k, [num_valence_bands], axis=1)

    mid_gap_energy = 0.5 * (e_k_valence.max() + e_k_conduction.min())

    def objective_function(fermi_level):
        """Charge neutrality objective function."""
        n_k = fermi_dirac(e_k_conduction - fermi_level, temperature).sum(axis=1)
        p_k = fermi_dirac(fermi_level - e_k_valence, temperature).sum(axis=1)
        n = np.trapezoid(n_k, kpoints)
        p = np.trapezoid(p_k, kpoints)

        rho = (n - p) / (2 * np.pi * cell_volume)
        rho *= 2  # Spin

        return (rho - doping_density) ** 2

    result = minimize_scalar(
        objective_function,
        bounds=(e_k.min(), e_k.max()),
        method="bounded",
    )

    return result.x, mid_gap_energy
