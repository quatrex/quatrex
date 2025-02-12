# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np

from qttools import NDArray, xp
from qttools.lyapunov.lyapunov import LyapunovSolver
from qttools.utils.gpu_utils import get_device, get_host


class Spectral(LyapunovSolver):
    """A solver for the Lyapunov equation by using the matrix spectrum."""

    def __init__(self, num_ref_iterations: int = 3):
        """Initializes the spectral Lyapunov solver."""
        self.num_ref_iterations = num_ref_iterations

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

        ws, vs = map(get_device, np.linalg.eig(get_host(a)))

        inv_vs = xp.linalg.inv(vs)
        gamma = inv_vs @ q @ inv_vs.conj().swapaxes(-1, -2)

        phi = xp.ones_like(a) - xp.einsum("e...i, e...j -> e...ij", ws, ws.conj())
        x_tilde = 1 / phi * gamma

        x = vs @ x_tilde @ vs.conj().swapaxes(-1, -2)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations):
            x = q + a @ x @ a.conj().swapaxes(-2, -1)

        if out is not None:
            out[...] = x
            return

        return x
