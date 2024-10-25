from qttools.lyapunov.lyapunov import Lyapunov
from qttools.utils.gpu_utils import xp


class Vectorize(Lyapunov):
    """A solver for the Lyapunov equation using vectorization."""

    def __call__(
        self,
        a: xp.ndarray,
        q: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:
        """Computes the solution of the discrete-time Lyapunov equation."""
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
