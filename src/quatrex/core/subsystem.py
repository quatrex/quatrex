# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from abc import ABC, abstractmethod

from qttools import NDArray, lyapunov, obc
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver import RGF, GFSolver, Inv, RGFDist
from qttools.nevp import NEVP, Beyn, Full
from qttools.utils.mpi_utils import get_local_slice
from quatrex.core.config import (
    ComputeConfig,
    LyapunovComputeConfig,
    LyapunovConfig,
    NEVPConfig,
    OBCConfig,
    QuatrexConfig,
    SolverConfig,
)


class SubsystemSolver(ABC):
    """Abstract base class for subsystem solvers.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
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
        energies: NDArray,
    ) -> None:
        """Initializes the solver."""
        self.energies = energies
        self.local_energies = get_local_slice(energies)

        self.obc = self._configure_obc(
            getattr(config, self.system).obc, config.compute.nevp
        )
        self.lyapunov = self._configure_lyapunov(
            getattr(config, self.system).lyapunov,
            config.compute.lyapunov,
        )
        self.solver = self._configure_solver(getattr(config, self.system).solver, config.compute.lyapunov)
        self.solver_dist = RGFDist(
            max_batch_size=getattr(config, self.system).solver.max_batch_size,
        )

        self.config = config

    def _configure_nevp(self, obc_config: OBCConfig, nevp_config: NEVPConfig) -> NEVP:
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
                m_0=obc_config.m_0,
                num_quad_points=obc_config.num_quad_points,
                num_threads_contour=nevp_config.num_threads_contour,
                eig_compute_location=nevp_config.eig_compute_location,
                project_compute_location=nevp_config.project_compute_location,
                use_qr=nevp_config.use_qr,
                contour_batch_size=nevp_config.contour_batch_size,
                use_pinned_memory=nevp_config.use_pinned_memory,
            )
        if obc_config.nevp_solver == "full":
            return Full(
                eig_compute_location=nevp_config.eig_compute_location,
                reduce=nevp_config.reduce_sparsity,
            )

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def _configure_obc(
        self, obc_config: OBCConfig, nevp_config: NEVPConfig
    ) -> obc.OBCSolver:
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
            nevp = self._configure_nevp(obc_config, nevp_config)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=obc_config.block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                min_propagation=obc_config.min_propagation,
                residual_tolerance=obc_config.residual_tolerance,
                residual_normalization=obc_config.residual_normalization,
                warning_threshold=obc_config.warning_threshold,
                eta_decay=obc_config.eta_decay,
            )

        else:
            raise NotImplementedError(
                f"OBC algorithm '{obc_config.algorithm}' not implemented."
            )

        if obc_config.memoizer.mode != "off":
            obc_solver = obc.OBCMemoizer(
                obc_solver=obc_solver,
                num_ref_iterations=obc_config.memoizer.num_ref_iterations,
                memoize_rel_tol=obc_config.memoizer.relative_tol,
                memoize_abs_tol=obc_config.memoizer.absolute_tol,
                warning_threshold=obc_config.memoizer.warning_threshold,
                memoizing_mode=obc_config.memoizer.mode,
            )

        return obc_solver

    def _configure_lyapunov(
        self,
        lyapunov_config: LyapunovConfig,
        lyapunov_compute_config: LyapunovComputeConfig,
    ) -> lyapunov.LyapunovSolver:
        """Configures the Lyapunov solver from the config.

        Parameters
        ----------
        lyapunov_config : LyapunovConfig
            The Lyapunov configuration.
        lyapunov_compute_config : LyapunovComputeConfig
            The Lyapunov compute configuration.

        Returns
        -------
        lyapunov.LyapunovSolver
            The configured Lyapunov solver.

        """
        if lyapunov_config.algorithm == "spectral":
            lyapunov_solver = lyapunov.Spectral(
                num_ref_iterations=lyapunov_config.num_ref_iterations,
                warning_threshold=lyapunov_config.warning_threshold,
                eig_compute_location=lyapunov_compute_config.eig_compute_location,
                reduce_sparsity=lyapunov_config.reduce_sparsity,
                use_pinned_memory=lyapunov_compute_config.use_pinned_memory,
            )
        elif lyapunov_config.algorithm == "doubling":
            lyapunov_solver = lyapunov.Doubling(
                max_iterations=lyapunov_config.max_iterations,
                convergence_rel_tol=lyapunov_config.relative_tol,
                convergence_abs_tol=lyapunov_config.absolute_tol,
                reduce_sparsity=lyapunov_config.reduce_sparsity,
            )
        else:
            raise NotImplementedError(
                f"Lyapunov algorithm '{lyapunov_config.algorithm}' not implemented."
            )

        if lyapunov_config.memoizer.mode != "off":
            lyapunov_solver = lyapunov.LyapunovMemoizer(
                lyapunov_solver=lyapunov_solver,
                num_ref_iterations=lyapunov_config.memoizer.num_ref_iterations,
                memoize_rel_tol=lyapunov_config.memoizer.relative_tol,
                memoize_abs_tol=lyapunov_config.memoizer.absolute_tol,
                warning_threshold=lyapunov_config.memoizer.warning_threshold,
                memoizing_mode=lyapunov_config.memoizer.mode,
                reduce_sparsity=lyapunov_config.reduce_sparsity,
            )
        return lyapunov_solver

    def _configure_solver(self, solver_config: SolverConfig, compute_config: ComputeConfig) -> GFSolver:
        """Configures the solver algorithm from the config.

        Parameters
        ----------
        solver : SolverConfig
            The solver configuration.
        compute_config : ComputeConfig
            The compute configuration.

        Returns
        -------
        GFSolver
            The configured solver.

        """
        if solver_config.algorithm == "rgf":
            return RGF(max_batch_size=solver_config.max_batch_size, mask=compute_config.mixed_precision.rgf_mask)

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
