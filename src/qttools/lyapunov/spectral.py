import numpy as np

from qttools import xp
from qttools.lyapunov.lyapunov import LyapunovSolver
from qttools.utils.gpu_utils import get_device, get_host


class Spectral(LyapunovSolver):
    """A solver for the Lyapunov equation by using the matrix spectrum."""

    def __call__(
        self,
        a: xp.ndarray,
        q: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:
        """Computes the solution of the discrete-time Lyapunov equation."""

        if a.ndim == 2:
            a = a[xp.newaxis, ...]
            q = q[xp.newaxis, ...]

        ws, vs = map(get_device, np.linalg.eig(get_host(a)))

        inv_vs = xp.linalg.inv(vs)
        gamma = inv_vs @ q @ inv_vs.conj().swapaxes(-1, -2)

        phi = xp.ones_like(a) - xp.einsum("e...i, e...j -> e...ij", ws, ws.conj())
        x_tilde = 1 / phi * gamma

        if out is not None:
            out[...] = vs @ x_tilde @ vs.conj().swapaxes(-1, -2)
            return

        return vs @ x_tilde @ vs.conj().swapaxes(-1, -2)
