# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from abc import ABC, abstractmethod

import numpy as np
from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import RGF, GFSolver, Inv
from qttools.nevp import NEVP, Beyn, Full
from qttools.obc import OBC, SanchoRubio, Spectral
from qttools.utils.mpi_utils import get_local_slice

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import OBCConfig, QuatrexConfig


class SubsystemSolver(ABC):
    """Abstract base class for subsystem solvers.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies : np.ndarray
        The energies at which to solve.

    """

    @property
    @abstractmethod
    def system(self) -> str:
        """The physical system for which the solver is implemented."""
        ...

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
    ) -> None:
        """Initializes the solver."""
        self.energies = energies
        self.local_energies = get_local_slice(energies)

        self.obc = self._configure_obc(getattr(quatrex_config, self.system).obc)
        self.solver = self._configure_solver(
            getattr(quatrex_config, self.system).solver
        )

    def _configure_nevp(self, obc_config: OBCConfig) -> NEVP:
        """Configures the NEVP solver from the config."""
        if obc_config.nevp_solver == "beyn":
            return Beyn(
                r_o=obc_config.r_o,
                r_i=obc_config.r_i,
                c_hat=obc_config.c_hat,
                num_quad_points=obc_config.num_quad_points,
            )
        if obc_config.nevp_solver == "full":
            return Full()

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def _configure_obc(self, obc_config: OBCConfig) -> OBC:
        """Configures the OBC algorithm from the config."""
        if obc_config.algorithm == "sancho-rubio":
            return SanchoRubio(obc_config.max_iterations, obc_config.convergence_tol)

        if obc_config.algorithm == "spectral":
            nevp = self._configure_nevp(obc_config)
            return Spectral(
                nevp=nevp,
                block_sections=obc_config.block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
            )

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
