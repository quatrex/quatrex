from abc import ABC, abstractmethod

import numpy as np
from qttools.greens_function_solver import RGF, GFSolver, Inv
from qttools.obc import OBC, SanchoRubio

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import OBCConfig, QuatrexConfig


class SubsystemSolver(ABC):
    """Abstract core class for subsystem solvers.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.

    """

    @abstractmethod
    @property
    def system(self) -> str:
        ...

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
    ) -> None:
        """Initializes the solver."""
        self.energies = energies

        self.obc = self._configure_obc(getattr(quatrex_config, self.system).obc)
        self.solver = self._configure_solver(
            getattr(quatrex_config, self.system).solver
        )

    def _configure_obc(self, obc_config: OBCConfig) -> OBC:
        """Configures the OBC algorithm from the config."""
        if obc_config.algorithm == "sancho-rubio":
            return SanchoRubio(obc_config.max_iterations, obc_config.convergence_tol)

        raise NotImplementedError(
            f"OBC algorithm '{obc_config.algorithm}' not implemented."
        )

    def _configure_solver(self, solver: str) -> GFSolver:
        """Configures the solver algorithm from the config."""
        if solver == "rgf":
            return RGF()

        if solver == "inv":
            return Inv()

        raise NotImplementedError(f"Solver '{solver}' not implemented.")

    @abstractmethod
    def apply_obc(self, *args, **kwargs) -> None:
        """Applies the OBC."""
        ...

    @abstractmethod
    def assemble_system_matrix(self) -> None:
        """Assembles the system matrix for a given energy."""
        ...

    @abstractmethod
    def solve(self) -> None:
        """Solves the system for a given energy."""
        ...
