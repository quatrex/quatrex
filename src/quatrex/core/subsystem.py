# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from abc import ABC, abstractmethod

import numpy as np
from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import RGF, GFSolver, Inv
from qttools.obc.full import Full
from qttools.obc.obc import OBC
from qttools.obc.sancho_rubio import SanchoRubio

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import OBCConfig, QuatrexConfig


class SubsystemSolver(ABC):
    """Abstract core class for subsystem solvers.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.

    """

    @property
    @abstractmethod
    def system(self) -> str:
        ...

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
    ) -> None:
        """Initializes the solver."""
        self.dbsparse = compute_config.dbsparse
        self.energies = energies

        self.obc = self._configure_obc(getattr(quatrex_config, self.system).obc)
        self.solver = self._configure_solver(
            getattr(quatrex_config, self.system).solver
        )

    def _configure_obc(self, obc_config: OBCConfig) -> OBC:
        """Configures the OBC algorithm from the config."""
        if obc_config.algorithm == "sancho-rubio":
            return SanchoRubio(obc_config.max_iterations, obc_config.convergence_tol)

        if obc_config.algorithm == "full":
            return Full()

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
    def solve(
        self,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Solves the system for a given energy."""
        ...
