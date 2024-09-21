import numpy.linalg as npla

from qttools.nevp.nevp import NEVP
from qttools.nevp.utils import operator_inverse
from qttools.utils.gpu_utils import get_device, get_host, xp

rng = xp.random.default_rng(42)


class Beyn(NEVP):
    """Beyn's integral method for solving NEVP.

    Parameters
    ----------
    r_o : float
        The outer radius of the annulus for the contour integration.
    r_i : float
        The inner radius of the annulus for the contour integration.
    c_hat : int
        Guess for the number of eigenvalues that lie in our subspace.
    num_quad_points : int
        The number of quadrature points for the contour integration.

    References
    ----------
    .. [1] W.-J. Beyn, An integral method for solving nonlinear
       eigenvalue problems, Linear Algebra and its Applications, 2012.
    .. [2] S. Brück, Ab-initio Quantum Transport Simulations for
       Nanoelectronic Devices, ETH Zurich, 2017.

    """

    def __init__(
        self,
        r_o: float,
        r_i: float,
        c_hat: int,
        num_quad_points: int,
    ):
        """Initializes the Beyn NEVP solver."""
        self.r_o = r_o
        self.r_i = r_i
        self.c_hat = c_hat
        self.num_quad_points = num_quad_points

    def __call__(
        self,
        a_xx: list[xp.ndarray],
    ):
        d = a_xx[0].shape[-1]
        in_type = a_xx[0].dtype

        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = [a_x[xp.newaxis, :, :] for a_x in a_xx]

        batchsize = a_xx[0].shape[0]

        # NOTE: Here we could also use a good initial guess.
        Y = rng.random((batchsize, d, self.c_hat)) + 1j * rng.random(
            (batchsize, d, self.c_hat)
        )

        # Determine quadrature points and weights.
        zeta = xp.arange(self.num_quad_points) + 1
        phase = xp.exp(2j * xp.pi * (zeta + 0.5) / self.num_quad_points)
        z_o = self.r_o * phase
        z_i = self.r_i * phase
        w = xp.ones(self.num_quad_points) / self.num_quad_points

        # Reshape to broadcast over batch dimension.
        z_o = z_o.reshape(1, -1, 1, 1)
        z_i = z_i.reshape(1, -1, 1, 1)
        w = w.reshape(1, -1, 1, 1)

        a_xx = [a_x[:, xp.newaxis, :, :] for a_x in a_xx]
        inv_Tz_o = operator_inverse(a_xx, z_o, in_type, in_type)
        inv_Tz_i = operator_inverse(a_xx, z_i, in_type, in_type)

        # Compute first and second moment.
        P_0 = xp.sum(w * z_o * inv_Tz_o - w * z_i * inv_Tz_i, axis=1) @ Y
        P_1 = xp.sum(w * z_o**2 * inv_Tz_o - w * z_i**2 * inv_Tz_i, axis=1) @ Y

        # Get the eigenvalues and eigenvectors.
        ws = xp.zeros((batchsize, self.c_hat), dtype=in_type)
        vs = Y.copy()

        # TODO: Batch even if the reduced size is smaller than c_hat.
        for i in range(batchsize):
            # Perform an SVD on the linear subspace projector.
            u, s, vh = xp.linalg.svd(P_0[i], full_matrices=False)

            # Remove the zero singular values (within numerical tolerance).
            eps_svd = s.max() * d * xp.finfo(in_type).eps

            inds = xp.where(s > eps_svd)[0]
            if len(inds) == self.c_hat:
                print("Search space too small. Relevant eigenvalues may be missing.")

            u, s, vh = u[:, inds], s[inds], vh[inds, :]

            # Probe second moment. No eigenvalues on the GPU :(
            w, v = npla.eig(
                get_host(u.conj().T @ P_1[i] @ vh.conj().T @ xp.diag(1 / s))
            )
            w, v = get_device(w), get_device(v)

            # Recover the full eigenvectors from the subspace.
            v = u @ v

            ws[i, : len(inds)] = w
            vs[i, :, : len(inds)] = v

        # Stack the batch dimension.
        ws = xp.hstack(ws)
        vs = xp.hstack(vs)

        return ws, vs
