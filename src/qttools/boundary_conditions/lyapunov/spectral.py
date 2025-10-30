# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.boundary_conditions.lyapunov.lyapunov import LyapunovSolver
from qttools.kernels import linalg


class Spectral(LyapunovSolver):
    """A solver for the Lyapunov equation by using the matrix spectrum.

    Parameters
    ----------
    num_ref_iterations : int, optional
        The number of refinement iterations to perform.
    eig_compute_location : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "numpy" or "cupy" or "nvmath".
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

    """

    def __init__(
        self,
        num_ref_iterations: int = 2,
        eig_compute_location: str = "numpy",
        use_pinned_memory: bool = True,
    ) -> None:
        """Initializes the spectral Lyapunov solver."""
        self.num_ref_iterations = num_ref_iterations
        self.eig_compute_location = eig_compute_location
        self.use_pinned_memory = use_pinned_memory

    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
    ) -> NDArray:
        """Computes the solution of the discrete-time Lyapunov equation.

        The matrices a and q can have different ndims with q.ndim >= a.ndim (will broadcast)

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.

        Returns
        -------
        x : NDArray
            The solution of the discrete-time Lyapunov equation.

        """

        assert q.shape[-2:] == a.shape[-2:]
        assert q.ndim >= a.ndim

        ws, vs = linalg.eig(
            a,
            compute_module=self.eig_compute_location,
            use_pinned_memory=self.use_pinned_memory,
        )

        inv_vs = linalg.inv(vs)
        inv_vs = xp.broadcast_to(inv_vs, q.shape)
        gamma = inv_vs @ q @ inv_vs.conj().swapaxes(-1, -2)

        phi = xp.ones_like(a) - xp.einsum("...i, ...j -> ...ij", ws, ws.conj())
        phi = xp.broadcast_to(phi, q.shape)
        x_tilde = 1 / phi * gamma

        x = vs @ x_tilde @ vs.conj().swapaxes(-1, -2)

        a = xp.broadcast_to(a, q.shape)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations):
            x = q + a @ x @ a.conj().swapaxes(-2, -1)

        return x
