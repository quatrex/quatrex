# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.kernels.eig import eig
from qttools.kernels.operator import operator_inverse
from qttools.kernels.svd import svd
from qttools.nevp.nevp import NEVP

rng = xp.random.default_rng(42)


class Beyn(NEVP):
    """Beyn's integral method for solving NEVP.[^1]

    This is implemented along the lines of what is described in [^2].

    [^1]: W.-J. Beyn, An integral method for solving nonlinear
    eigenvalue problems, Linear Algebra and its Applications, 2012.

    [^2]: S. Brück, Ab-initio Quantum Transport Simulations for
    Nanoelectronic Devices, ETH Zurich, 2017.

    Parameters
    ----------
    r_o : float
        The outer radius of the annulus for the contour integration.
    r_i : float
        The inner radius of the annulus for the contour integration.
    m_0 : int
        Guess for the number of eigenvalues that lie in the subspace.
    num_quad_points : int
        The number of quadrature points to use for the contour
        integration.
    num_threads_contour : int, optional
        The number of cuda threads to use for the contour integration kernel.
        Only relevant for GPU computations.
    eig_compute_location : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "numpy" or "cupy". Only relevant if cupy is used.
    svd_compute_location : str, optional
        The location where to compute the singular value decomposition.
        Can be either "numpy" or "cupy". Only relevant if cupy is
        used.

    """

    def __init__(
        self,
        r_o: float,
        r_i: float,
        m_0: int,
        num_quad_points: int,
        num_threads_contour: int = 1024,
        eig_compute_location: str = "numpy",
        svd_compute_location: str = "numpy",
    ):
        """Initializes the Beyn NEVP solver."""
        self.r_o = r_o
        self.r_i = r_i
        self.m_0 = m_0
        self.num_quad_points = num_quad_points
        self.num_threads_contour = num_threads_contour
        self.eig_compute_location = eig_compute_location
        self.svd_compute_location = svd_compute_location

    def _one_sided(self, a_xx: tuple[NDArray, ...]) -> tuple[NDArray, NDArray]:
        """Solves the plynomial eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.

        Returns
        -------
        ws : NDArray
            The eigenvalues.
        vs : NDArray
            The eigenvectors.

        """
        d = a_xx[0].shape[-1]
        in_type = a_xx[0].dtype

        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

        batchsize = a_xx[0].shape[0]

        # NOTE: Here we could also use a good initial guess.
        Y = rng.random((batchsize, d, self.m_0)) + 1j * rng.random(
            (batchsize, d, self.m_0)
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

        a_xx = tuple(a_x[:, xp.newaxis, :, :] for a_x in a_xx)
        inv_Tz_o = operator_inverse(
            a_xx, z_o, in_type, in_type, self.num_threads_contour
        )
        inv_Tz_i = operator_inverse(
            a_xx, z_i, in_type, in_type, self.num_threads_contour
        )

        # Compute first and second moment.
        P_0 = xp.sum(w * z_o * inv_Tz_o - w * z_i * inv_Tz_i, axis=1) @ Y
        P_1 = xp.sum(w * z_o**2 * inv_Tz_o - w * z_i**2 * inv_Tz_i, axis=1) @ Y

        # Get the eigenvalues and eigenvectors.
        ws = xp.zeros((batchsize, self.m_0), dtype=in_type)
        vs = Y.copy()

        # TODO: Batch even if the reduced size is smaller than m_0.
        for i in range(batchsize):
            # Perform an SVD on the linear subspace projector.
            u, s, vh = svd(
                P_0[i], full_matrices=False, compute_module=self.svd_compute_location
            )

            # Remove the zero singular values (within numerical tolerance).
            eps_svd = s.max() * d * xp.finfo(in_type).eps
            inds = xp.where(s > eps_svd)[0]

            u, s, vh = u[:, inds], s[inds], vh[inds, :]

            # Probe second moment.
            a = u.conj().T @ P_1[i] @ vh.conj().T / s
            w, v = eig(a, compute_module=self.eig_compute_location)

            # Recover the full eigenvectors from the subspace.
            ws[i, : len(inds)] = w
            vs[i, :, : len(inds)] = u @ v

        return ws, vs

    def _two_sided(
        self, a_xx: tuple[NDArray, ...]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Solves the plynomial eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.

        Returns
        -------
        ws : NDArray
            The right eigenvalues.
        vs : NDArray
            The right eigenvectors.
        wl : NDArray
            The left eigenvalues.
        vl : NDArray
            The left eigenvectors.

        """
        d = a_xx[0].shape[-1]
        in_type = a_xx[0].dtype

        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

        batchsize = a_xx[0].shape[0]

        # NOTE: Here we could also use a good initial guess.
        Y = rng.random((batchsize, d, self.m_0)) + 1j * rng.random(
            (batchsize, d, self.m_0)
        )
        Y_hat = rng.random((batchsize, d, self.m_0)) + 1j * rng.random(
            (batchsize, d, self.m_0)
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
        inv_Tz_o = operator_inverse(
            a_xx, z_o, in_type, in_type, self.num_threads_contour
        )
        inv_Tz_i = operator_inverse(
            a_xx, z_i, in_type, in_type, self.num_threads_contour
        )

        q0 = xp.sum(w * z_o * inv_Tz_o - w * z_i * inv_Tz_i, axis=1)
        q1 = xp.sum(w * z_o**2 * inv_Tz_o - w * z_i**2 * inv_Tz_i, axis=1)

        # Compute first and second moment.
        P_0 = q0 @ Y
        P_1 = q1 @ Y

        P_0_hat = Y_hat.conj().swapaxes(-2, -1) @ q0
        P_1_hat = Y_hat.conj().swapaxes(-2, -1) @ q1

        # Get the eigenvalues and eigenvectors.
        wrs = xp.zeros((batchsize, self.m_0), dtype=in_type)
        vrs = xp.zeros((batchsize, d, self.m_0), dtype=in_type)
        wls = xp.zeros((batchsize, self.m_0), dtype=in_type)
        vls = xp.zeros((batchsize, d, self.m_0), dtype=in_type)

        # TODO: Batch even if the reduced size is smaller than m_0.
        for i in range(batchsize):
            # Perform an SVD on the linear subspace projector.
            u, s, vh = svd(
                P_0[i], full_matrices=False, compute_module=self.svd_compute_location
            )
            u_hat, s_hat, vh_hat = svd(
                P_0_hat[i],
                full_matrices=False,
                compute_module=self.svd_compute_location,
            )

            # Remove the zero singular values (within numerical tolerance).
            # and orthogonalize projector
            eps_svd = s.max() * d * xp.finfo(in_type).eps
            eps_svd_hat = s_hat.max() * d * xp.finfo(in_type).eps
            inds = xp.where(s > eps_svd)[0]
            inds_hat = xp.where(s_hat > eps_svd_hat)[0]

            u, s, vh = u[:, inds], s[inds], vh[inds, :]
            u_hat, s_hat, vh_hat = (
                u_hat[:, inds_hat],
                s_hat[inds_hat],
                vh_hat[inds_hat, :],
            )

            # Probe second moment.
            a = u.conj().T @ P_1[i] @ vh.conj().T / s
            # NOTE: xp.diag is unnecessary, should be removed
            a_hat = xp.diag(1 / s_hat) @ u_hat.conj().T @ P_1_hat[i] @ vh_hat.conj().T

            w, v = eig(a, compute_module=self.eig_compute_location)
            w_hat, v_hat = eig(a_hat, compute_module=self.eig_compute_location)

            # Recover the full eigenvectors from the subspace.
            wrs[i, : len(inds)] = w
            wls[i, : len(inds_hat)] = w_hat

            vrs[i, :, : len(inds)] = u @ v
            vls[i, :, : len(inds_hat)] = xp.linalg.solve(v_hat, vh_hat).conj().T

        return wrs, vrs, wls, vls

    def __call__(
        self, a_xx: tuple[NDArray, ...], left: bool = False
    ) -> tuple[NDArray, NDArray]:
        """Solves the plynomial eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.
        left : bool, optional
            Whether to solve additionally for the left eigenvectors.

        Returns
        -------
        ws : NDArray
            The right eigenvalues.
        vs : NDArray
            The right eigenvectors.
        wl : NDArray, optional
            The left eigenvalues.
            Returned only if `left` is `True`.
        vl : NDArray, optional
            The left eigenvectors.
            Returned only if `left` is `True`.

        """
        if left:
            return self._two_sided(a_xx)
        return self._one_sided(a_xx)
