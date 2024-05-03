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
