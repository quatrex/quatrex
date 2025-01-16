# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from abc import ABC, abstractmethod

from qttools import NDArray, lyapunov, obc
from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import RGF, GFSolver, Inv
from qttools.nevp import NEVP, Beyn, Full
from qttools.utils.mpi_utils import get_local_slice

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import LyapunovConfig, OBCConfig, QuatrexConfig


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
        energies: NDArray,
    ) -> None:
        """Initializes the solver."""
        self.energies = energies
        self.local_energies = get_local_slice(energies)

        self.obc = self._configure_obc(getattr(quatrex_config, self.system).obc)
        self.lyapunov = self._configure_lyapunov(
            getattr(quatrex_config, self.system).lyapunov
        )
        self.solver = self._configure_solver(
            getattr(quatrex_config, self.system).solver
        )

    def _configure_nevp(self, obc_config: OBCConfig) -> NEVP:
        """Configures the NEVP solver from the config.

        Parameters
        ----------
        obc_config : OBCConfig
            The OBC configuration.

        Returns
        -------
        NEVP
            The configured NEVP solver.

        """
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

    def _configure_obc(self, obc_config: OBCConfig) -> obc.OBCSolver:
        """Configures the OBC algorithm from the config.

        Parameters
        ----------
        obc_config : OBCConfig
            The OBC configuration.

        Returns
        -------
        obc.OBCSolver
            The configured OBC solver.

        """
        if obc_config.algorithm == "sancho-rubio":
            obc_solver = obc.SanchoRubio(
                obc_config.max_iterations, obc_config.convergence_tol
            )

        elif obc_config.algorithm == "spectral":
            nevp = self._configure_nevp(obc_config)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=obc_config.block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                x_ii_formula=obc_config.x_ii_formula,
                treat_pairwise=obc_config.treat_pairwise,
                pairing_threshold=obc_config.pairing_threshold,
                min_propagation=obc_config.min_propagation,
            )

        else:
            raise NotImplementedError(
                f"OBC algorithm '{obc_config.algorithm}' not implemented."
            )

        if obc_config.memoizer.enable:
            obc_solver = obc.OBCMemoizer(
                obc_solver,
                obc_config.memoizer.num_ref_iterations,
                obc_config.memoizer.convergence_tol,
            )

        return obc_solver

    def _configure_lyapunov(
        self, lyapunov_config: LyapunovConfig
    ) -> lyapunov.LyapunovSolver:
        """Configures the Lyapunov solver from the config.

        Parameters
        ----------
        lyapunov_config : LyapunovConfig
            The Lyapunov configuration.

        Returns
        -------
        lyapunov.LyapunovSolver
            The configured Lyapunov solver.

        """
        if lyapunov_config.algorithm == "spectral":
            lyapunov_solver = lyapunov.Spectral(
                num_ref_iterations=lyapunov_config.num_ref_iterations
            )
        elif lyapunov_config.algorithm == "doubling":
            lyapunov_solver = lyapunov.Doubling(
                lyapunov_config.max_iterations, lyapunov_config.convergence_tol
            )
        else:
            raise NotImplementedError(
                f"Lyapunov algorithm '{lyapunov_config.algorithm}' not implemented."
            )

        if lyapunov_config.memoizer.enable:
            lyapunov_solver = lyapunov.LyapunovMemoizer(
                lyapunov_solver,
                lyapunov_config.memoizer.num_ref_iterations,
                lyapunov_config.memoizer.convergence_tol,
            )
        return lyapunov_solver

    def _configure_solver(self, solver: str) -> GFSolver:
        """Configures the solver algorithm from the config.

        Parameters
        ----------
        solver : str
            The solver algorithm.

        Returns
        -------
        GFSolver
            The configured solver.

        """
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
        """Solves the system.

        Parameters
        ----------
        sse_lesser : DSBSparse
            The lesser self-energy.
        sse_greater : DSBSparse
            The greater self-energy.
        sse_retarded : DSBSparse
            The retarded self-energy.
        out : tuple[DSBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """
        ...
