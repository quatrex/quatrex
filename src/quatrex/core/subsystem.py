import multiprocessing as mp
from abc import ABC, abstractmethod
from functools import partial

from mpi4py.MPI import COMM_WORLD

import numpy as np
import numpy.linalg as npla

from quatrex.core.config import OBCConfig, QuatrexConfig
from quantumtransporttoolbox.obc import (
    sancho_rubio,
)
from quantumtransporttoolbox.greens_function import rgf

from quatrex.core.coo import COOBatch


class SubsystemSolver(ABC):
    """Abstract core class for subsystem solvers.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.

    """

    @property
    @abstractmethod
    def system(self) -> str: ...

    def __init__(self, config: QuatrexConfig, *args: tuple, **kwargs: dict) -> None:
        """Initializes the solver."""

        self.energies = np.load(config.input_dir / "energies.npy")

        self.obc = self._configure_obc(getattr(config, self.system).obc)
        self.invert = self._configure_inversion(getattr(config, self.system).solver)
        self.solver = self._configure_solver(getattr(config, self.system).solver)

        (self.n_energies_per_rank, self.energy_slice) = self._compute_energy_slice()

    def _compute_energy_slice(self):
        """Computes the energy slice."""
        n_energies = len(self.energies) // COMM_WORLD.size

        if len(self.energies) % COMM_WORLD.size != 0:
            raise ValueError(
                "The number of energies must be divisible by the number of MPI processes."
            )

        start_energy = n_energies * COMM_WORLD.rank
        end_energy = start_energy + n_energies

        return (n_energies, slice(start_energy, end_energy))

    def _allocate_memory(self, nnz: int) -> tuple:
        # Inputs data
        system_matrices = COOBatch(self.n_energies, nnz)
        sigma_lesser = COOBatch(self.n_energies, nnz)
        sigma_greater = COOBatch(self.n_energies, nnz)

        # Results
        g_lesser = COOBatch(self.n_energies, nnz)
        g_greater = COOBatch(self.n_energies, nnz)

        return (system_matrices, sigma_lesser, sigma_greater, g_lesser, g_greater)

    def _configure_obc(self, obc_config: OBCConfig) -> callable:
        """Configures the OBC algorithm from the config.

        Parameters
        ----------
        obc_config : QuatrexConfig
            The OBC configuration.

        Returns
        -------
        obc : callable
            The configured OBC algorithm.

        """
        if obc_config.algorithm == "sancho-rubio":

            def obc(a_ii, a_ij, a_ji, side: str) -> np.ndarray:
                """Calculates the surface Green's function."""
                return partial(
                    sancho_rubio,
                    max_iterations=obc_config.max_iterations,
                    max_delta=obc_config.max_delta,
                )(a_ii, a_ij, a_ji)

            return obc

        raise NotImplementedError(
            f"OBC algorithm {obc_config.algorithm} not implemented."
        )

    def _configure_solver_retarded(self, solver: str) -> callable:
        """Configures the inversion algorithm from the config.

        Parameters
        ----------
        solver : str
            The inversion algorithm.

        Returns
        -------
        invert : callable
            The configured inversion algorithm.

        """
        if solver == "rgf":
            return rgf

        if solver == "dense_inv":

            def invert(a: bsp.BSparse) -> np.ndarray:
                inverse = npla.inv(a.toarray())
                return bsp.BCOO.from_array(inverse, sizes=(a.row_sizes, a.col_sizes))

            return invert

        raise NotImplementedError(f"Solver '{solver}' not implemented.")

    def _configure_solver_quadratic(self, solver: str) -> callable:
        """Configures the solver algorithm from the config.

        Parameters
        ----------
        solver : str
            The solver algorithm.

        Returns
        -------
        solver : callable
            The configured solver algorithm.

        """
        if solver == "rgf":
            return rgf

        if solver == "dense_inv":

            def solve(
                a: bsp.BSparse, sigma_l: bsp.BSparse, sigma_g: bsp.BSparse
            ) -> np.ndarray:
                y = npla.inv(a.toarray())
                x_l = bsp.BCOO.from_array(
                    (y @ sigma_l.toarray() @ y.conj().T),
                    sizes=(a.row_sizes, a.col_sizes),
                )
                x_g = bsp.BCOO.from_array(
                    (y @ sigma_g.toarray() @ y.conj().T),
                    sizes=(a.row_sizes, a.col_sizes),
                )

                return x_l, x_g

            return solve

        raise NotImplementedError(f"Solver '{solver}' not implemented.")

    @abstractmethod
    def apply_obc(self, *args, **kwargs) -> tuple[bsp.BSparse, ...]:
        """Applies the OBC."""
        ...

    @abstractmethod
    def assemble_system_matrix(self, energy: float) -> tuple[bsp.BSparse, ...]:
        """Assembles the system matrix for a given energy."""
        ...

    @abstractmethod
    def solve_retarded_at_energy(self, energy: float) -> bsp.BSparse:
        """Solves for the causal Green's function at a given energy."""
        ...

    def solve_retarded(self) -> list[bsp.BCOO]:
        """Solves for the causal Green's function for all energies.

        Returns
        -------
        d_causal : list[bsp.BCOO]
            The causal Green's function for all energies.

        """
        global _solve_causal_at_energy

        def _solve_causal_at_energy(energy: float):
            return self.solve_causal_at_energy(energy)

        self._set_omp_num_threads()
        with mp.Pool(
            self.num_processes, maxtasksperchild=self.maxtasksperchild
        ) as pool:
            if not hasattr(self, "cli") or not self.cli:
                return pool.map(_solve_causal_at_energy, self.energies)
            return track(
                pool.imap(_solve_causal_at_energy, self.energies),
                total=len(self.energies),
            )

    @abstractmethod
    def solve_lesser_greater_at_energy(
        self, energy: float, contour_order="lesser"
    ) -> bsp.BSparse:
        """Solves for the lesser Green's function at a given energy."""
        ...

    def solve_lesser_greater(self) -> list[bsp.BCOO]:
        """Solves for the lesser/greater Green's function for all energies."""
        global _solve_at_energy

        def _solve_at_energy(energy: float):
            return self.solve_at_energy(energy)

        self._set_omp_num_threads()
        with mp.Pool(
            self.num_processes, maxtasksperchild=self.maxtasksperchild
        ) as pool:
            if not hasattr(self, "cli") or not self.cli:
                res = pool.map(_solve_at_energy, self.energies)

            res = track(
                pool.imap(_solve_at_energy, self.energies), total=len(self.energies)
            )

        return zip(*res)
