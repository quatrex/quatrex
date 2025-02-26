# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.kernels.eig import eig
from qttools.lyapunov.lyapunov import LyapunovSolver


class Spectral(LyapunovSolver):
    """A solver for the Lyapunov equation by using the matrix spectrum.

    Parameters
    ----------
    num_ref_iterations : int, optional
        The number of refinement iterations to perform.
    warning_threshold : float, optional
        The threshold for the relative recursion error to issue a warning.
    eig_compute_location : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "numpy" or "cupy". Only relevant if cupy is used.

    """

    def __init__(
        self,
        num_ref_iterations: int = 3,
        warning_threshold: float = 1e-1,
        eig_compute_location: str = "numpy",
    ) -> None:
        """Initializes the spectral Lyapunov solver."""
        self.num_ref_iterations = num_ref_iterations
        self.warning_threshold = warning_threshold
        self.eig_compute_location = eig_compute_location

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

        if a.ndim == 2:
            a = a[xp.newaxis, ...]
            q = q[xp.newaxis, ...]

        ws, vs = eig(a, compute_module=self.eig_compute_location)

        inv_vs = xp.linalg.inv(vs)
        gamma = inv_vs @ q @ inv_vs.conj().swapaxes(-1, -2)

        phi = xp.ones_like(a) - xp.einsum("e...i, e...j -> e...ij", ws, ws.conj())
        x_tilde = 1 / phi * gamma

        x = vs @ x_tilde @ vs.conj().swapaxes(-1, -2)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x = q + a @ x @ a.conj().swapaxes(-2, -1)

        x_ref = q + a @ x @ a.conj().swapaxes(-2, -1)

        # Check the batch average recursion error.
        recursion_error = xp.mean(
            xp.linalg.norm(x_ref - x, axis=(-2, -1))
            / xp.linalg.norm(x_ref, axis=(-2, -1))
        )
        if recursion_error > self.warning_threshold:
            warnings.warn(
                f"High relative recursion error: {recursion_error:.2e}",
                RuntimeWarning,
            )

        if out is not None:
            out[...] = x_ref
            return

        return x_ref
