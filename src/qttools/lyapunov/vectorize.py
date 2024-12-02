# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.lyapunov.lyapunov import LyapunovSolver


class Vectorize(LyapunovSolver):
    """A solver for the Lyapunov equation using vectorization."""

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
            a = a[xp.newaxis, :, :]
            q = q[xp.newaxis, :, :]

        b, *__, n = a.shape

        a_ = -xp.einsum("eij,ekl->ekilj", a.conj(), a).reshape(b, n**2, n**2)
        a_ += xp.eye(n**2)[xp.newaxis].astype(a.dtype)
        q_ = q.reshape(b, n**2)
        x = xp.linalg.solve(a_, q_[..., xp.newaxis]).reshape(b, n, n)

        if out is not None:
            out[...] = x
            return

        return x
