# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

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
