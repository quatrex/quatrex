# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import CUDA_AWARE_MPI, NCCL_AVAILABLE, NDArray, nccl_comm, xp
from qttools.kernels.linalg import inv
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_device, get_host, synchronize_current_stream

profiler = Profiler()


class OBCSolver(ABC):
    r"""Abstract base class for the open-boundary condition solver.

    The recursion relation for the surface Green's function is given by:

    \[
        x_{ii} = (a_{ii} - a_{ji} x_{ii} a_{ij})^{-1}
    \]

    """

    @abstractmethod
    def __call__(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Returns the surface Green's function.

        Parameters
        ----------
        a_ii : NDArray
            Diagonal boundary block of a system matrix.
        a_ij : NDArray
            Superdiagonal boundary block of a system matrix.
        a_ji : NDArray
            Subdiagonal boundary block of a system matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        ...


class OBCMemoizer:
    """Memoization class to reuse the result of an OBC function call.

    Parameters
    ----------
    obc_solver : OBCSolver
        The OBC solver to wrap.
    num_ref_iterations : int, optional
        The maximum number of refinement iterations to do.
    memoize_tol : float, optional
        The required accuracy to only memoize.
    force_memoizing: bool, optionak
        Force memoizing using q as the initial guess.

    """

    def __init__(
        self,
        obc_solver: "OBCSolver",
        num_ref_iterations: int = 3,
        memoize_tol: float = 1e-2,
        force_memoizing: bool = False,
    ) -> None:
        """Initalizes the memoizer."""
        self.obc_solver = obc_solver
        self.num_ref_iterations = num_ref_iterations
        self.memoize_tol = memoize_tol
        self.force_memoizing = force_memoizing
        self._cache = {}

    @profiler.profile(level="debug")
    def _call_with_cache(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Calls the wrapped OBC solver with cache handling.

        Parameters
        ----------
        a_ii : NDArray
            Diagonal boundary block of a system matrix.
        a_ij : NDArray
            Superdiagonal boundary block of a system matrix.
        a_ji : NDArray
            Subdiagonal boundary block of a system matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        x_ii = self.obc_solver(a_ii, a_ij, a_ji, contact, out=out)
        if out is None:
            self._cache[contact] = x_ii.copy()
            return x_ii

        self._cache[contact] = out.copy()
        return None

    @profiler.profile(level="api")
    def __call__(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Returns the surface Green's function.

        This is a memoized wrapper around an OBC solver.

        Parameters
        ----------
        a_ii : NDArray
            Diagonal boundary block of a system matrix.
        a_ij : NDArray
            Superdiagonal boundary block of a system matrix.
        a_ji : NDArray
            Subdiagonal boundary block of a system matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        # TODO: merge with Lyapunov memoizer
        # since there is code duplication

        # Try to reuse the result from the cache.
        x_ii = self._cache.get(contact, None)

        if self.force_memoizing:
            x_ii = inv(a_ii)

        if x_ii is None:
            return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)

        x_ii_ref = inv(a_ii - a_ji @ x_ii @ a_ij)

        # Check for convergence accross all MPI ranks.
        recursion_error = xp.max(
            xp.linalg.norm(x_ii_ref - x_ii, axis=(-2, -1))
            / xp.linalg.norm(x_ii_ref, axis=(-2, -1))
        )
        x_ii = x_ii_ref

        local_memoizing = xp.array(recursion_error < self.memoize_tol, dtype=int)
        memoizing = xp.empty_like(local_memoizing)

        # NCCL allreduce does not support op="and"
        synchronize_current_stream()
        # NCCL allreduce does not support op="and"
        if NCCL_AVAILABLE:
            nccl_comm.all_reduce(local_memoizing, memoizing, op="sum")
        elif CUDA_AWARE_MPI:
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
                f"{memoizing} out of {comm.size} ranks want to memoize OBC", flush=True
            )

        # NOTE: it would be possible to memoize even if few energies did not converge
        if memoizing != comm.size and not self.force_memoizing:
            # If the result did not converge, recompute it from scratch.
            return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x_ii = inv(a_ii - a_ji @ x_ii @ a_ij)

        # TODO: we should allow data gathering of the final recursion error

        self._cache[contact] = x_ii.copy()
        if out is None:
            return x_ii
        out[:] = x_ii
        return None
