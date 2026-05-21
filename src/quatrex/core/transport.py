# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from abc import ABC, abstractmethod

from qttools import NDArray


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

    @abstractmethod
    def run(self) -> None:
        """Solves the transport problem."""
        ...
