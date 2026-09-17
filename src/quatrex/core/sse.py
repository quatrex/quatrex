# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the abstract base class for scattering self-energy calculations."""

from abc import ABC, abstractmethod


class ScatteringSelfEnergy(ABC):
    @abstractmethod
    def compute(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Computes the scattering self-energy."""
        ...
