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
        """Returns the surface Green's function.

        Parameters
        ----------
        a_ii : np.ndarray
            Diagonal boundary block of a system matrix.
        a_ij : np.ndarray
            Off-diagonal boundary block of a system matrix, connecting
            lead (i) to system (j).
        a_ji : np.ndarray
            Off-diagonal boundary block of a system matrix, connecting
            system (j) to lead (i).
        contact : str
            The contact to which the boundary blocks belong.
        out : np.ndarray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : np.ndarray, optional
            The system's surface Green's function.

        """
        ...
