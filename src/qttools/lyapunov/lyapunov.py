# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NCCL_AVAILABLE, NDArray, nccl_comm, xp
from qttools.lyapunov.utils import system_reduction
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_device, get_host, synchronize_current_stream
from qttools.utils.mpi_utils import check_gpu_aware_mpi

profiler = Profiler()

GPU_AWARE_MPI = check_gpu_aware_mpi()


class LyapunovSolver(ABC):
    r"""Solver interface for the discrete-time Lyapunov equation.

    The discrete-time Lyapunov equation is defined as:

    \[
        X - A X A^H = Q
    \]

    """

    @abstractmethod
    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : NDArray | None
            The solution of the discrete-time Lyapunov equation.

        """
        ...


class LyapunovMemoizer:
    """Memoization class to reuse the result of an Lyapunov evaluation.

    Parameters
    ----------
    lyapunov_solver : LyapunovSolver
        The Lyapunov solver to wrap.
    num_ref_iterations : int, optional
        The number of refinement iterations to do.
    memoize_tol : float, optional
        The required accuracy to only memoize.
    reduce_sparsity : bool, optional
        Whether to reduce the sparsity of the system matrix.
        If sparsity of any obc is changed during runtime, then the cache
        needs to be invalidated.
    force_memoizing: bool, optionak
        Force memoizing using q as the initial guess.

    """

    def __init__(
        self,
        lyapunov_solver: LyapunovSolver,
        num_ref_iterations: int = 10,
        memoize_tol: float = 1e-2,
        reduce_sparsity: bool = True,
        force_memoizing: bool = False,
    ) -> None:
        """Initializes the memoizer."""
        self.lyapunov_solver = lyapunov_solver
        self.num_ref_iterations = num_ref_iterations
        self.memoize_tol = memoize_tol
        self._cache = {}
        self.reduce_sparsity = reduce_sparsity
        self.force_memoizing = force_memoizing

    @profiler.profile(level="debug")
    def _call_with_cache(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Calls the wrapped Lyapunov function with cache handling.

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong. Used as a
            key for the cache.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : NDArray | None
            The solution of the discrete-time Lyapunov equation.

        """
        x = self.lyapunov_solver(a, q, contact, out=out)
        if out is None:
            self._cache[contact] = x.copy()
            return x

        self._cache[contact] = out.copy()
        return None

    @profiler.profile(level="api")
    def _solve(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        This is a memoized wrapper around a Lyapunov solver.

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : NDArray | None
            The solution of the discrete-time Lyapunov equation.

        """
        # Try to reuse the result from the cache.
        x = self._cache.get(contact, None)

        if self.force_memoizing:
            x = q

        if x is None:
            return self._call_with_cache(a, q, contact, out=out)

        x_ref = q + a @ x @ a.conj().swapaxes(-2, -1)

        # Check for convergence accross all MPI ranks.
        recursion_error = xp.mean(
            xp.linalg.norm(x_ref - x, axis=(-2, -1))
            / xp.linalg.norm(x_ref, axis=(-2, -1))
        )
        x = x_ref

        local_memoizing = xp.array(recursion_error < self.memoize_tol, dtype=int)
        memoizing = xp.empty_like(local_memoizing)

        # NCCL allreduce does not support op="and"
        synchronize_current_stream()
        # NCCL allreduce does not support op="and"
        if NCCL_AVAILABLE:
            nccl_comm.all_reduce(local_memoizing, memoizing, op="sum")
        elif GPU_AWARE_MPI:
            comm.Allreduce(local_memoizing, memoizing, op=MPI.SUM)
        else:
            local_memoizing = get_host(local_memoizing)
            # TODO: this memcopy is not necessary
            # but for consistency with the other cases
            memoizing = get_host(memoizing)
            comm.Allreduce(local_memoizing, memoizing, op=MPI.SUM)
            memoizing = get_device(memoizing)
        synchronize_current_stream()

        if comm.rank == 0:
            print(
                f"{memoizing} out of {comm.size} ranks want to memoize Lyapunov",
                flush=True,
            )

        if memoizing != comm.size and not self.force_memoizing:
            # If the result did not converge, recompute it from scratch.
            return self._call_with_cache(a, q, contact, out=out)

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x = q + a @ x @ a.conj().swapaxes(-2, -1)

        # TODO: we should allow data gathering of the final recursion error

        self._cache[contact] = x.copy()
        if out is None:
            return x
        out[:] = x
        return None

    @profiler.profile(level="api")
    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        The matrices a and q can have different ndims with q.ndim >= a.ndim (will broadcast)

        This is a memoized wrapper around a Lyapunov solver.

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : NDArray | None
            The solution of the discrete-time Lyapunov equation.

        """

        assert q.shape[-2:] == a.shape[-2:]
        assert q.ndim >= a.ndim

        # NOTE: possible to cache the sparsity reduction
        if self.reduce_sparsity:

            if hasattr(self.lyapunov_solver, "reduce_sparsity"):
                save_reduce_sparsity = self.lyapunov_solver.reduce_sparsity
                # Not reduce sparsity twice
                self.lyapunov_solver.reduce_sparsity = False

            out = system_reduction(a, q, contact, self._solve, out=out)

            if hasattr(self.lyapunov_solver, "reduce_sparsity"):
                self.lyapunov_solver.reduce_sparsity = save_reduce_sparsity

            return out

        return self._solve(a, q, contact, out=out)
