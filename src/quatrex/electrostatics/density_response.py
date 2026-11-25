# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Density Response Models."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from qttools import NDArray
from quatrex.core.constants import hbar, k_B
from quatrex.core.utils import split_trapezoid
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
    density : NDArray
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


def dirac_net_density(eta: NDArray) -> NDArray:
    r"""Reduced net carrier density of a Dirac cone.

    Computes $g(\eta) = \mathcal{F}_1(\eta) - \mathcal{F}_1(-\eta)$,
    i.e. electrons minus holes in units of the graphene effective
    density of states. Using the reflection formula
    $\mathcal{F}_1(\eta) + \mathcal{F}_1(-\eta) = \eta^2/2 + \pi^2/6$,
    this is written such that it is odd in $\eta$ and does not
    overflow.

    Parameters
    ----------
    eta : NDArray
        The reduced chemical potential relative to the Dirac point.

    Returns
    -------
    g : NDArray
        The reduced net carrier density.

    """
    eta = np.asarray(eta, dtype=float)
    abs_eta = np.abs(eta)
    return np.sign(eta) * (
        abs_eta**2 / 2 + np.pi**2 / 6 - 2 * fermi_integral(1, -abs_eta)
    )


def dirac_net_density_derivative(eta: NDArray) -> NDArray:
    r"""Derivative of the reduced net carrier density of a Dirac cone.

    Computes $g'(\eta) = \mathcal{F}_0(\eta) + \mathcal{F}_0(-\eta) =
    |\eta| + 2 \ln(1 + e^{-|\eta|})$, which is strictly positive
    (it equals $2 \ln 2$ at the Dirac point).

    Parameters
    ----------
    eta : NDArray
        The reduced chemical potential relative to the Dirac point.

    Returns
    -------
    g_prime : NDArray
        The derivative of the reduced net carrier density.

    """
    abs_eta = np.abs(np.asarray(eta, dtype=float))
    return abs_eta + 2 * np.log1p(np.exp(-abs_eta))


def inverse_dirac_net_density(
    u: NDArray, tol: float = 1e-12, max_iterations: int = 100
) -> NDArray:
    r"""Inverts the reduced net carrier density of a Dirac cone.

    Solves $g(\eta) = u$ for $\eta$ with Newton's method. Since $g$ is
    odd, strictly increasing and convex for $\eta > 0$, Newton's method
    converges for all $u$ from the initial guess
    $\eta_0 = u / \sqrt{(2 \ln 2)^2 + |u| / 2}$, which matches both the
    linear ($|u| \ll 1$) and the degenerate ($|u| \gg 1$) limit.

    Parameters
    ----------
    u : NDArray
        The reduced net carrier density (can be of either sign).
    tol : float, optional
        Absolute tolerance on the Newton update. Default is 1e-12.
    max_iterations : int, optional
        Maximum number of Newton iterations. Default is 100.

    Returns
    -------
    eta : NDArray
        The reduced chemical potential relative to the Dirac point.

    """
    u = np.asarray(u, dtype=float)
    eta = u / np.sqrt((2 * np.log(2)) ** 2 + np.abs(u) / 2)
    for __ in range(max_iterations):
        delta = (dirac_net_density(eta) - u) / dirac_net_density_derivative(eta)
        eta = eta - delta
        if np.all(np.abs(delta) < tol):
            break
    return eta


class GrapheneDensityModel(DensityModel):
    r"""Ambipolar density response model for graphene.

    Graphene is a semimetal with a linear (Dirac cone) density of
    states around the charge neutrality level, so the single parabolic
    band picture of `SingleBandDensityModel` does not apply. Here,
    electrons and holes are treated simultaneously and the net excess
    charge is

    $$
    \rho(\phi) = N_G \left[\mathcal{F}_1(\eta) -
    \mathcal{F}_1(-\eta)\right], \quad \eta = \frac{\mu - \phi}{k_B T},
    $$

    with the effective density of states of graphene
    $N_G = \frac{2}{\pi} \left(\frac{k_B T}{\hbar v_F}\right)^2$
    (spin and valley degeneracy included), here expressed per atom. The
    local electrochemical potential $\mu$ is determined by inverting
    this relation for the given density and potential. Unlike the
    single-band model, the density can change sign during the Newton
    iterations (e.g. for p-n junctions) and the derivative (quantum
    capacitance) stays finite at the Dirac point.

    Parameters
    ----------
    density : NDArray
        The excess charge density (electrons per atom, electrons
        positive).
    potential : NDArray
        The potential in eV.
    fermi_velocity : float, optional
        The Fermi velocity of graphene in m/s. Default is 1e6 m/s.
    area_per_atom : float, optional
        The area per atom of the graphene sheet in Å^2. Default is the
        value for a C-C bond length of 1.42 Å.
    temperature : float, optional
        The temperature in Kelvin. Default is 300 K.

    """

    def __init__(
        self,
        density: NDArray,
        potential: NDArray,
        fermi_velocity: float = 1e6,
        area_per_atom: float = 3 * np.sqrt(3) / 4 * 1.42**2,
        temperature: float = 300,
    ):
        """Initializes the graphene density model."""
        self.temperature = temperature
        self.thermal_energy = k_B * temperature

        hbar_v_F = hbar * fermi_velocity * 1e10  # eV Å
        self.effective_dos = (
            2 / np.pi * (self.thermal_energy / hbar_v_F) ** 2 * area_per_atom
        )

        self.charge_neutrality_level = (
            potential
            + inverse_dirac_net_density(np.asarray(density) / self.effective_dos)
            * self.thermal_energy
        )

    def _eta(self, potential: NDArray) -> NDArray:
        """Computes the reduced chemical potential."""
        return (self.charge_neutrality_level - potential) / self.thermal_energy

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
        return self.effective_dos * dirac_net_density(self._eta(potential))

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
        return (
            -self.effective_dos
            / self.thermal_energy
            * dirac_net_density_derivative(self._eta(potential))
        )


def _excess_occupation(
    spectral_dos: NDArray,
    energies: NDArray,
    chemical_potentials: NDArray,
    neutrality_levels: NDArray,
    thermal_energy: float,
) -> NDArray:
    """Net excess occupation per atom for given chemical potentials.

    States above the neutrality level count as electrons (occupation
    f), states below it as holes (occupation 1 - f). Evaluating 1 - f
    directly avoids cancellation against the filled valence states. The
    integration is split exactly at the neutrality level, consistent
    with the excess charge of the transport solvers.

    """
    x = (energies[np.newaxis, :] - chemical_potentials[:, np.newaxis]) / thermal_energy
    __, electrons = split_trapezoid(
        spectral_dos * expit(-x), energies, neutrality_levels
    )
    holes, __ = split_trapezoid(spectral_dos * expit(x), energies, neutrality_levels)
    return electrons - holes


def local_quasi_fermi_levels(
    spectral_dos: NDArray,
    net_density: NDArray,
    energies: NDArray,
    neutrality_levels: NDArray,
    temperature: float,
    num_bisections: int = 64,
) -> NDArray:
    r"""Fits a local quasi-Fermi level to the occupation on each atom.

    Finds $\mu_i$ such that filling the local spectral function
    $A_i(E)$ with a Fermi-Dirac distribution reproduces the net excess
    density $n_i$ (electrons above, holes below the local neutrality
    level $E_{\mathrm{CNL}, i}$):

    $$
    \int_{E > E_{\mathrm{CNL}, i}} A_i(E) f(E - \mu_i) \mathrm{d}E -
    \int_{E < E_{\mathrm{CNL}, i}} A_i(E) f(\mu_i - E) \mathrm{d}E =
    n_i.
    $$

    The left-hand side is monotonic in $\mu_i$, so the levels are found
    by (vectorized) bisection. In equilibrium, where the occupation is
    Fermi-Dirac, this recovers the chemical potential. Atoms whose
    density cannot be reproduced within the energy window are clamped
    to the bracket bounds.

    Parameters
    ----------
    spectral_dos : NDArray
        The local spectral function per atom with shape (num_atoms,
        num_energies), in states per eV.
    net_density : NDArray
        The net excess density per atom (electrons positive).
    energies : NDArray
        The energy grid in eV.
    neutrality_levels : NDArray
        The local charge neutrality (mid-gap) level per atom in eV.
    temperature : float
        The temperature in K.
    num_bisections : int, optional
        The number of bisection steps.

    Returns
    -------
    quasi_fermi_levels : NDArray
        The local quasi-Fermi level per atom in eV.

    """
    spectral_dos = np.clip(np.asarray(spectral_dos, dtype=float), 0.0, None)
    net_density = np.asarray(net_density, dtype=float)
    energies = np.asarray(energies, dtype=float)
    neutrality_levels = np.asarray(neutrality_levels, dtype=float)
    thermal_energy = k_B * temperature

    margin = 40 * thermal_energy
    lower = np.full(net_density.shape, energies[0] - margin)
    upper = np.full(net_density.shape, energies[-1] + margin)
    for __ in range(num_bisections):
        middle = 0.5 * (lower + upper)
        too_low = (
            _excess_occupation(
                spectral_dos, energies, middle, neutrality_levels, thermal_energy
            )
            < net_density
        )
        lower = np.where(too_low, middle, lower)
        upper = np.where(too_low, upper, middle)

    return 0.5 * (lower + upper)


@dataclass
class SpectralDensityOfStates:
    """Atom-resolved local density of states from a transport solver.

    The LDOS is split into components that are each filled with a
    Fermi-Dirac distribution. For the wave-function formalism, the
    components are the states injected by each contact, filled up to
    the contact's chemical potential. For NEGF, where the occupation
    also contains scattering-in contributions, there is a single
    component (the local spectral function) filled up to a local
    quasi-Fermi level per atom.

    Attributes
    ----------
    spectral_dos : NDArray
        The LDOS per atom (summed over orbitals, spin included,
        averaged over transverse k-points) with shape (num_components,
        num_atoms, num_energies), in states per eV.
    energies : NDArray
        The energy grid in eV, with shape (num_energies,).
    chemical_potentials : NDArray
        The chemical potentials in eV, either one per component with
        shape (num_components,) or one per component and atom with shape
        (num_components, num_atoms).
    temperatures : NDArray
        The temperature of each component in K, with shape
        (num_components,).

    """

    spectral_dos: NDArray
    energies: NDArray
    chemical_potentials: NDArray
    temperatures: NDArray


class LDOSDensityModel(DensityModel):
    r"""Density response model from the transport LDOS (rigid shift).

    The local density of states $D_{i,c}(E)$ on atom $i$ of component
    $c$ (e.g. the states injected from contact $c$) is taken from the
    last transport calculation, which was done at the potential
    $\phi^0$. A change of the local potential $\delta_i = \phi_i -
    \phi_i^0$ is assumed to shift the local states rigidly, so that

    $$
    n_i(\phi) = n_i^0 + \sum_c \int D_{i,c}(E) \left[f_{c,i}(E +
    \delta_i) - f_{c,i}(E)\right] dE, \quad f_{c,i}(E) = \frac{1}{1 +
    e^{(E - \mu_{c,i}) / k_B T_c}}.
    $$

    The states below the charge neutrality level shift along with the
    potential, so they do not contribute to the change of the excess
    charge. The model reproduces the transport density at $\phi^0$
    exactly, and its derivative there is the local quantum capacitance
    of the actual (e.g. k-point sampled, non-parabolic) band structure.
    Electrons and holes are treated on the same footing.

    Parameters
    ----------
    density : NDArray
        The excess charge density per atom from the transport
        calculation (electrons positive).
    potential : NDArray
        The potential per atom at which the transport calculation was
        done.
    spectral_dos : SpectralDensityOfStates
        The atom-resolved LDOS of the transport calculation.

    """

    # Number of thermal energies the Fermi window of each component must
    # stay away from the energy window boundaries.
    window_margin = 10.0

    def __init__(
        self,
        density: NDArray,
        potential: NDArray,
        spectral_dos: SpectralDensityOfStates,
    ):
        """Initializes the LDOS density model."""
        self.reference_density = np.asarray(density, dtype=float)
        self.reference_potential = np.asarray(potential, dtype=float)

        # NOTE: The Mulliken-projected LDOS can be slightly negative.
        # It is clipped to keep the density monotonic in the potential
        # (i.e., a negative semi-definite contribution to the Jacobian).
        self.spectral_dos = np.clip(
            np.asarray(spectral_dos.spectral_dos, dtype=float), 0.0, None
        )
        self.energies = np.asarray(spectral_dos.energies, dtype=float)

        num_components, num_atoms, num_energies = self.spectral_dos.shape
        chemical_potentials = np.asarray(spectral_dos.chemical_potentials, dtype=float)
        temperatures = np.asarray(spectral_dos.temperatures, dtype=float)
        if (
            self.reference_density.shape != (num_atoms,)
            or self.reference_potential.shape != (num_atoms,)
            or self.energies.shape != (num_energies,)
            or chemical_potentials.shape
            not in ((num_components,), (num_components, num_atoms))
            or temperatures.shape not in ((), (num_components,))
        ):
            raise ValueError(
                "Inconsistent shapes: spectral_dos "
                f"{self.spectral_dos.shape}, density "
                f"{self.reference_density.shape}, potential "
                f"{self.reference_potential.shape}, energies "
                f"{self.energies.shape}, chemical_potentials "
                f"{chemical_potentials.shape}, temperatures "
                f"{temperatures.shape}."
            )

        # Chemical potentials per component and atom.
        self.chemical_potentials = np.broadcast_to(
            chemical_potentials.reshape(num_components, -1),
            (num_components, num_atoms),
        )
        self.thermal_energies = k_B * np.broadcast_to(temperatures, (num_components,))

        self._warned_window = False
        self.reference_occupation = self._occupation(np.zeros(num_atoms))

    def _occupations(self, shift: NDArray) -> list[tuple[NDArray, float]]:
        """Fermi-Dirac occupations for each component at the shifted energies."""
        self._check_window(shift)
        return [
            (
                expit(
                    -(self.energies[np.newaxis, :] + (shift - mu)[:, np.newaxis]) / kT
                ),
                kT,
            )
            for mu, kT in zip(self.chemical_potentials, self.thermal_energies)
        ]

    def _occupation(self, shift: NDArray) -> NDArray:
        """Number of occupied states per atom for the given shift."""
        occupation = np.zeros(shift.shape)
        for dos, (f, __) in zip(self.spectral_dos, self._occupations(shift)):
            occupation += np.trapezoid(dos * f, self.energies, axis=-1)
        return occupation

    def _check_window(self, shift: NDArray):
        """Warns if a Fermi window is shifted out of the energy window."""
        if self._warned_window or shift.size == 0:
            return
        margin = self.window_margin * self.thermal_energies[:, np.newaxis]
        # The occupation at energy E + shift is filled up to mu, i.e.
        # the Fermi window is centered at mu - shift.
        center = self.chemical_potentials - shift[np.newaxis, :]
        if np.any(center - margin < self.energies[0]) or np.any(
            center + margin > self.energies[-1]
        ):
            self._warned_window = True
            warnings.warn(
                "The potential update shifts a Fermi window outside of the "
                "transport energy window. The LDOS density model "
                "underestimates the density response there. Consider "
                "widening the energy window.",
                RuntimeWarning,
            )

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
        shift = np.asarray(potential, dtype=float) - self.reference_potential
        return (
            self.reference_density + self._occupation(shift) - self.reference_occupation
        )

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
        shift = np.asarray(potential, dtype=float) - self.reference_potential
        derivative = np.zeros(shift.shape)
        for dos, (f, kT) in zip(self.spectral_dos, self._occupations(shift)):
            derivative -= np.trapezoid(dos * f * (1 - f), self.energies, axis=-1) / kT
        return derivative
