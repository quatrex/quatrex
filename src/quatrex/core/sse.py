# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from abc import ABC, abstractmethod

from qttools.datastructures import DSBSparse


class ScatteringSelfEnergy(ABC):
    @abstractmethod
    def compute(
        self,
        *args,
        **kwargs,
    ) -> DSBSparse:
        """Computes the scattering self-energy."""
        ...
