# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
from quatrex.core.constants import hbar, k_B
from quatrex.core.statistics import fermi_dirac


def approximate_fermi_levels(
    potential: NDArray,
    rho: NDArray,
    rho_sign: NDArray,
    max_rho: float,
    rho_shift: float,
    temperature: float,
):
    """Approximates local quasi-Fermi levels from charges.

    Parameters
    ----------
    potential : NDArray
        The potential.
    rho : NDArray
        The charge density.
    rho_sign : NDArray
        The sign of the charge density.
    max_rho : float
        The maximum charge density.
    rho_shift : float
        The shift in charge density.
    temperature : float
        The temperature.

    Returns
    -------
    fermi_levels : NDArray
        Local quasi-Fermi levels.

    """
    fermi_levels = potential + rho_sign * k_B * temperature * xp.log(
        xp.exp((rho + rho_sign * rho_shift) / (max_rho * rho_sign)) - 1
    )

    return fermi_levels


def approximate_rho(
    potential: NDArray,
    fermi_levels: NDArray,
    rho_sign: NDArray,
    max_rho: float,
    temperature: float,
):
    """Approximates the local quasi-equilibrium charge density.

    Parameters
    ----------
    potential : NDArray
        The potential.
    fermi_levels : NDArray
        The local quasi-Fermi levels.
    rho_sign : NDArray
        The sign of the charge density.
    max_rho : float
        The maximum charge density.
    temperature : float
        The temperature.

    Returns
    -------
    rho : NDArray
        The charge density.

    """
    rho = (
        -rho_sign
        * max_rho
        * xp.log(
            fermi_dirac(
                rho_sign * (fermi_levels - potential),
                temperature,
            )
        )
    )
    return rho


def approximate_drho_dV(
    potential: NDArray,
    fermi_levels: NDArray,
    rho_sign: NDArray,
    max_rho: float,
    temperature: float,
):
    """Approximates the derivative of the charge density w.r.t. potential.

    Parameters
    ----------
    potential : NDArray
        The potential.
    fermi_levels : NDArray
        The local quasi-Fermi levels.
    rho_sign : NDArray
        The sign of the charge density.
    max_rho : float
        The maximum charge density.
    temperature : float
        The temperature.

    Returns
    -------
    drho_dV : NDArray
        The derivative of the charge density w.r.t. potential.

    """
    drho_dV = (
        -max_rho
        / (k_B * temperature)
        * fermi_dirac(rho_sign * (potential - fermi_levels), temperature)
    )
    return drho_dV


def effective_mass(energies: NDArray, ldos: NDArray, dim: int):
    """Computes the approximate spectral effective mass from the LDOS.

    Parameters
    ----------
    energies : NDArray
        The energies.
    ldos : NDArray
        The LDOS.
    dim : int
        The dimensionality. Must be 1, 2, or 3.

    Returns
    -------
    m_star : NDArray
        The approximate spectral effective mass.

    """
    if dim == 1:
        return hbar**2 * xp.pi**2 / 2 * (energies[:, xp.newaxis]) * ldos**2
    if dim == 2:
        return xp.pi * hbar**2 * ldos
    if dim == 3:
        return (
            2 * xp.pi**2 * hbar**3 * ldos / xp.sqrt(energies[:, xp.newaxis])
        ) ** (2 / 3) / 2

    raise ValueError("Invalid dimensionality. Must be 1, 2, or 3.")


def ldos(energies: NDArray, m_star: NDArray, dim: int):
    """Computes the local density of states from the effective mass.

    Parameters
    ----------
    energies : NDArray
        The energies.
    m_star : NDArray
        The effective mass.
    dim : int
        The dimensionality. Must be 1, 2, or 3.

    Returns
    -------
    ldos : NDArray
        The local density of states.

    """
    if dim == 1:
        return xp.sqrt(2 * m_star) / (hbar * xp.pi * xp.sqrt(energies[:, xp.newaxis]))
    if dim == 2:
        return m_star / (xp.pi * hbar**2)
    if dim == 3:
        return (
            (2 * m_star) ** (3 / 2)
            / (2 * xp.pi**2 * hbar**3)
            * xp.sqrt(energies[:, xp.newaxis])
        )

    raise ValueError("Invalid dimensionality. Must be 1, 2, or 3.")
