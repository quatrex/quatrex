# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the abstract base class for subsystem solvers."""

from abc import ABC, abstractmethod

from qttools import NDArray
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver import RGF, GFSolver, Inv, RGFDist
from qttools.utils.mpi_utils import get_local_slice
from quatrex.core.config import QuatrexConfig, SolverConfig
from quatrex.device import SCBADevice


class SubsystemSolver(ABC):
    """Abstract base class for subsystem solvers.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    device : SCBADevice
        The device for which to solve the subsystem.
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
        config: QuatrexConfig,
        device: SCBADevice,
        energies: NDArray,
    ) -> None:
        """Initializes the solver."""
        self.energies = energies
        self.local_energies = get_local_slice(energies)

        self.solver = self._configure_solver(getattr(config, self.system).solver)
        self.solver_dist = RGFDist(
            max_batch_size=getattr(config, self.system).solver.max_batch_size,
        )

        self.config = config
        self.device = device

    def _configure_solver(self, solver_config: SolverConfig) -> GFSolver:
        """Configures the solver algorithm from the config.

        Parameters
        ----------
        solver : SolverConfig
            The solver configuration.

        Returns
        -------
        GFSolver
            The configured solver.

        """
        if solver_config.algorithm == "rgf":
            return RGF(max_batch_size=solver_config.max_batch_size)

        if solver_config.algorithm == "inv":
            return Inv(max_batch_size=solver_config.max_batch_size)

        raise NotImplementedError(
            f"Solver '{solver_config.algorithm}' not implemented."
        )

    @abstractmethod
    def solve(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        sse_retarded: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Solves the system.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser self-energy.
        sse_greater : DSDBSparse
            The greater self-energy.
        sse_retarded : DSDBSparse
            The retarded self-energy.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """
        ...
