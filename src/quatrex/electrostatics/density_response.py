# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Density Response Models."""

from abc import ABC, abstractmethod

import numpy as np

from qttools import NDArray
from quatrex.core.constants import hbar, k_B
from quatrex.electrostatics.fermi_integrals import (
    fermi_integral,
    inverse_fermi_integral,
)


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
        return hbar**2 * np.pi**2 / 2 * (energies[:, np.newaxis]) * ldos**2
    if dim == 2:
        return np.pi * hbar**2 * ldos
    if dim == 3:
        return (
            2 * np.pi**2 * hbar**3 * ldos / np.sqrt(energies[:, np.newaxis])
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
        return np.sqrt(2 * m_star) / (hbar * np.pi * np.sqrt(energies[:, np.newaxis]))
    if dim == 2:
        return m_star / (np.pi * hbar**2)
    if dim == 3:
        return (
            (2 * m_star) ** (3 / 2)
            / (2 * np.pi**2 * hbar**3)
            * np.sqrt(energies[:, np.newaxis])
        )

    raise ValueError("Invalid dimensionality. Must be 1, 2, or 3.")


class DensityModel(ABC):
    """Abstract base class for density response models."""

    @abstractmethod
    def density(self, potential: NDArray) -> NDArray:
        """Computes the density for a given potential."""
        ...

    @abstractmethod
    def density_derivative(self, potential: NDArray) -> NDArray:
        """Computes the derivative of the density with respect to the potential."""
        ...


class OMENDensityModel(DensityModel):
    """Density response model based on the approximations made in OMEN.

    Parameters
    ----------
    charge_density : NDArray
        The charge density.
    potential : NDArray
        The potential.
    temperature : float
        The temperature in Kelvin.
    rho_shift : float
        A small shift to avoid issues with very small charge densities.

    """

    def __init__(
        self,
        density: NDArray,
        potential: NDArray,
        temperature: float = 300,
        rho_shift: float = 1e-8,
    ):
        """Initializes the OMEN density model."""
        self.temperature = temperature
        self.effective_dos = max(np.max(np.abs(density)), 1e-6)
        self.rho_shift = rho_shift
        self.rho_sign = np.sign(density)

        # NOTE: OMEN does not use CODATA values. Instead, it hardcodes
        # the value of k_B in eV/K.
        self._k_B = 1.38e-23 / 1.6022e-19

        self.fermi_level = self._estimate_fermi_level(density, potential)

    def _estimate_fermi_level(
        self,
        density: NDArray,
        potential: NDArray,
    ) -> float:
        """Estimates the Fermi level from the given density.

        Parameters
        ----------
        density : NDArray
            The density.
        potential : NDArray
            The potential.

        Returns
        -------
        fermi_level : float
            The estimated Fermi level.

        """
        fermi_level = potential + self.rho_sign * self._k_B * self.temperature * np.log(
            np.exp(
                (density + self.rho_sign * self.rho_shift)
                / (self.effective_dos * self.rho_sign)
            )
            - 1
        )

        return fermi_level

    def density(self, potential: NDArray) -> NDArray:
        """Computes the density for a given potential.

        Parameters
        ----------
        potential : NDArray
            The potential.

        Returns
        -------
        density : NDArray
            The computed density.

        """
        density = (
            self.rho_sign
            * self.effective_dos
            * np.log(
                1
                + np.exp(
                    self.rho_sign
                    * (self.fermi_level - potential)
                    / (self._k_B * self.temperature)
                )
            )
        )
        return density

    def density_derivative(self, potential: NDArray) -> NDArray:
        """Computes the derivative of the density with respect to the potential.

        Parameters
        ----------
        potential : NDArray
            The potential.


        Returns
        -------
        density_derivative : NDArray
            The computed derivative of the density.

        """
        density_derivative = (
            -1
            / (self._k_B * self.temperature)
            * self.effective_dos
            / (
                np.exp(
                    self.rho_sign
                    * (potential - self.fermi_level)
                    / (self._k_B * self.temperature)
                )
                + 1
            )
        )
        return density_derivative


class SingleBandDensityModel(DensityModel):
    """Density response model for a single band in a certain dimension.

    Parameters
    ----------
    density : NDArray
        The charge density.
    potential : NDArray
        The potential.
    dim : int
        The dimensionality of the system (1, 2, or 3).
    temperature : float
        The temperature in Kelvin.

    """

    def __init__(
        self,
        density: NDArray,
        potential: NDArray,
        dim: int,
        temperature: float = 300,
    ):
        """Initializes the single-band density model."""
        self.temperature = temperature
        self.dim = dim

        self.rho_sign = np.sign(density)

        self.effective_dos = self._estimate_effective_dos(density)
        self.charge_neutrality_level = self._estimate_charge_neutrality_level(
            density, potential
        )

    def _estimate_effective_dos(self, density: NDArray) -> float:
        """Estimates the effective density of states from the given density.

        Parameters
        ----------
        density : NDArray
            The density.

        Returns
        -------
        effective_dos : float
            The estimated effective density of states.

        """
        return max(np.max(np.abs(density)), 1e-6)

    def _estimate_charge_neutrality_level(
        self, density: NDArray, potential: NDArray
    ) -> float:
        """Estimates the charge neutrality level from the given density.

        Parameters
        ----------
        density : NDArray
            The density.
        potential : NDArray
            The potential.

        Returns
        -------
        charge_neutrality_level : float
            The estimated charge neutrality level.

        """
        charge_neutrality_level = (
            self.rho_sign
            * (
                inverse_fermi_integral(
                    self.dim / 2 - 1,
                    np.abs(density) / self.effective_dos,
                    method="approximate",
                )
                * k_B
                * self.temperature
            )
            + potential
        )
        return charge_neutrality_level

    def density(self, potential: NDArray) -> NDArray:
        """Computes the density for a given potential.

        Parameters
        ----------
        potential : NDArray
            The potential.

        Returns
        -------
        density : NDArray
            The computed density.

        """
        density = (
            self.rho_sign
            * self.effective_dos
            * fermi_integral(
                self.dim / 2 - 1,
                self.rho_sign
                * (self.charge_neutrality_level - potential)
                / (k_B * self.temperature),
            )
        )
        return density

    def density_derivative(self, potential: NDArray) -> NDArray:
        """Computes the derivative of the density with respect to the potential.

        Parameters
        ----------
        potential : NDArray
            The potential.

        Returns
        -------
        density_derivative : NDArray
            The computed derivative of the density.

        """
        density_derivative = (
            -self.effective_dos
            / (k_B * self.temperature)
            * fermi_integral(
                # NOTE: The derivative of the Fermi integral of order n
                # is the Fermi integral of order n-1.
                self.dim / 2 - 2,
                self.rho_sign
                * (self.charge_neutrality_level - potential)
                / (k_B * self.temperature),
            )
        )
        return density_derivative
