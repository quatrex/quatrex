# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from abc import ABC, abstractmethod

from qttools.datastructures import DBSparse


class ScatteringSelfEnergy(ABC):
    @abstractmethod
    def compute(
        self,
        *args,
        **kwargs,
    ) -> DBSparse:
        """Computes the scattering self-energy."""
        ...
