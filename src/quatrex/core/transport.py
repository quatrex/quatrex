# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the abstract base class for transport solvers."""

from abc import ABC, abstractmethod

from qttools import NDArray
from quatrex.electrostatics.density_response import SpectralDensityOfStates


class TransportSolver(ABC):
    """Abstract base class for transport solvers."""

    @abstractmethod
    def set_potential(self, potential: NDArray) -> None:
        """Sets the potential for the transport solver."""
        ...

    @abstractmethod
    def get_charge_density(self) -> NDArray:
        """Gets the charge density from the transport solver."""
        ...

    def get_spectral_density_of_states(self) -> SpectralDensityOfStates:
        """Gets the atom- and contact-resolved LDOS of the last run.

        This is needed for the "ldos" density model of the nonlinear
        Poisson solver.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide the spectral density "
            "of states needed for the 'ldos' density model."
        )

    @abstractmethod
    def run(self) -> None:
        """Solves the transport problem."""
        ...
