# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

import numpy as np


class OBC(ABC):
    """Abstract base class for the open-boundary condition solver."""

    @abstractmethod
    def __call__(
        self,
        a_ii: np.ndarray,
        a_ij: np.ndarray,
        a_ji: np.ndarray,
        contact: str,
        out=None | np.ndarray,
    ) -> np.ndarray | None:
        ...
