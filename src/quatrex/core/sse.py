# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

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
