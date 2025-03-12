# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

import numpy as np

from qttools import NDArray, xp
from qttools.kernels import linalg
from qttools.kernels.operator import operator_inverse
from qttools.nevp.nevp import NEVP
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import (
    get_any_location,
    get_any_location_pinned,
    get_array_module_name,
)
from qttools.utils.mpi_utils import get_section_sizes

profiler = Profiler()


rng = xp.random.default_rng(42)


@decorate_methods(
    profiler.profile(level="debug"),
    exclude=["__call__", "__init__"],
)
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
    project_compute_location : str, optional
        The location where to compute the singular value
        or qr decomposition for the projector.
        Can be either "numpy" or "cupy". Only relevant if cupy is
        used.
    use_qr : bool, optional
        Whether to use QR decomposition for the projector instead of SVD.
        Default is `False`.
    contour_batch_size : int, optional
        The batch size for the contour integration kernel. If `None`,
        the batch size is set to `num_quad_points`.
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

    """

    def __init__(
        self,
        r_o: float,
        r_i: float,
        m_0: int,
        num_quad_points: int,
        num_threads_contour: int = 1024,
        eig_compute_location: str = "numpy",
        project_compute_location: str = "numpy",
        use_qr: bool = False,
        contour_batch_size: int | None = None,
        use_pinned_memory: bool = True,
    ):
        """Initializes the Beyn NEVP solver."""
        self.r_o = r_o
        self.r_i = r_i
        self.m_0 = m_0
        self.num_quad_points = num_quad_points
        self.num_threads_contour = num_threads_contour
        self.eig_compute_location = eig_compute_location
        self.project_compute_location = project_compute_location
        self.use_qr = use_qr
        if contour_batch_size is None:
            contour_batch_size = num_quad_points

        self.contour_batch_size = contour_batch_size

        contour_counts, _ = get_section_sizes(
            num_quad_points, int(np.ceil(num_quad_points / contour_batch_size))
        )

        self.contour_displacements = np.cumsum(
            np.concatenate(([0], np.array(contour_counts)))
        )
        self.use_pinned_memory = use_pinned_memory

    def _project_svd(
        self, P_0: NDArray, P_1: NDArray, left: bool = False
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Projects the systems onto the linear subspace with an SVD.

        Parameters
        ----------
        P_0 : NDArray
            The first moment of the system.
        P_1 : NDArray
            The second moment of the system.
        left : bool, optional
            Whether to project the system from the left.

        Returns
        -------
        a : list[NDArray]
            The projected systems.
        u_out : list[NDArray]
            The projectors.

        """
        batchsize = P_0.shape[0]
        d = P_0.shape[-1]
        input_location = get_array_module_name(P_0)

        if self.use_pinned_memory:
            P_0 = get_any_location_pinned(P_0, self.project_compute_location)
        else:
            P_0 = get_any_location(P_0, self.project_compute_location)

        # Perform an SVD on the linear subspace projector.
        u, s, vh = linalg.svd(
            P_0, full_matrices=False, compute_module=self.project_compute_location
        )

        # NOTE: this can lead to an extra memory copy on the host
        # kernels could be change to accept an output and the cache
        if self.use_pinned_memory:
            u = get_any_location_pinned(u, input_location)
            s = get_any_location_pinned(s, input_location)
            vh = get_any_location_pinned(vh, input_location)
        else:
            u = get_any_location(u, input_location)
            s = get_any_location(s, input_location)
            vh = get_any_location(vh, input_location)

        a = []
        u_out = []
        v_out = []
        for i in range(batchsize):

            ui, si, vhi = u[i], s[i], vh[i]

            # Remove the zero singular values (within numerical tolerance).
            eps_svd = si.max() * d * xp.finfo(P_0.dtype).eps
            inds = xp.where(si > eps_svd)[0]

            ui, si, vhi = ui[:, inds], si[inds], vhi[inds, :]
            u_out.append(ui)
            v_out.append(vhi)

            # Probe second moment.
            if left:
                a.append(xp.diag(1 / si) @ ui.conj().T @ P_1[i] @ vhi.conj().T)
            else:
                a.append(ui.conj().T @ P_1[i] @ vhi.conj().T / si)

        if left:
            return a, v_out
        else:
            return a, u_out

    def _project_qr(
        self, P_0: NDArray, P_1: NDArray, left: bool = False
    ) -> tuple[NDArray, NDArray]:
        """Projects the systems onto the linear subspace with a QR.

        Parameters
        ----------
        P_0 : NDArray
            The first moment of the system.
        P_1 : NDArray
            The second moment of the system.
        left : bool, optional
            Whether to project the system from the left.

        Returns
        -------
        a : NDArray
            The projected system.
        q : NDArray
            The projectors.

        """
        input_location = get_array_module_name(P_0)

        P_0 = get_any_location_pinned(P_0, self.project_compute_location)

        # Perform an QR on the linear subspace projector.
        if left:
            q, r = linalg.qr(
                P_0.conj().swapaxes(-2, -1),
                compute_module=self.project_compute_location,
            )
        else:
            q, r = linalg.qr(P_0, compute_module=self.project_compute_location)

        # NOTE: this can lead to an extra memory copy on the host
        # kernels could be change to accept an output and the cache
        if self.use_pinned_memory:
            # NOTE: this can lead to bugs if copies would async
            # and q/r have the same size (alias of the pinned buffers)
            q = get_any_location_pinned(q, input_location)
            r = get_any_location_pinned(r, input_location)
        else:
            q = get_any_location(q, input_location)
            r = get_any_location(r, input_location)

        if left:
            a = xp.linalg.inv(r.conj().swapaxes(-2, -1)) @ P_1 @ q
        else:
            a = q.conj().swapaxes(-2, -1) @ P_1 @ xp.linalg.inv(r)

        if left:
            return a, q.conj().swapaxes(-2, -1)
        else:
            return a, q

    def _contour_integrate(self, a_xx: tuple[NDArray, ...]) -> tuple[NDArray, NDArray]:
        """Computes the contour integral of the operator inverse.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.

        Returns
        -------
        q0 : NDArray
            The first moment of the contour integral.
        q1 : NDArray
            The second moment of the contour integral.

        """

        in_type = a_xx[0].dtype

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

        q0 = xp.zeros_like(a_xx[0])
        q1 = xp.zeros_like(a_xx[0])
        a_xx = [a_x[:, xp.newaxis, :, :] for a_x in a_xx]

        for j in range(len(self.contour_displacements) - 1):

            z_oj = z_o[
                :, self.contour_displacements[j] : self.contour_displacements[j + 1]
            ]
            z_ij = z_i[
                :, self.contour_displacements[j] : self.contour_displacements[j + 1]
            ]
            w_j = w[
                :, self.contour_displacements[j] : self.contour_displacements[j + 1]
            ]

            inv_Tz_o = operator_inverse(
                a_xx, z_oj, in_type, in_type, self.num_threads_contour
            )
            inv_Tz_i = operator_inverse(
                a_xx, z_ij, in_type, in_type, self.num_threads_contour
            )

            q0 += xp.sum(w_j * z_oj * inv_Tz_o - w_j * z_ij * inv_Tz_i, axis=1)
            q1 += xp.sum(w_j * z_oj**2 * inv_Tz_o - w_j * z_ij**2 * inv_Tz_i, axis=1)

        return q0, q1

    def _solve_reduced_system(
        self,
        a: list[NDArray] | NDArray,
        Y: NDArray,
        p_back: list[NDArray] | NDArray,
        left: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Solve the reduced system"""

        in_type = a[0].dtype
        input_location = get_array_module_name(Y)
        if isinstance(a, list):
            batchsize = len(a)
        else:
            batchsize = a.shape[0]

        if self.eig_compute_location == "numpy" and xp.__name__ == "cupy":

            if self.use_pinned_memory:
                if isinstance(a, list):
                    a = [
                        get_any_location_pinned(ai, self.eig_compute_location)
                        for ai in a
                    ]
                else:
                    a = get_any_location_pinned(a, self.eig_compute_location)
            else:
                if isinstance(a, list):
                    a = [get_any_location(ai, self.eig_compute_location) for ai in a]
                else:
                    a = get_any_location(a, self.eig_compute_location)

        # solve the reduced system
        w, v = linalg.eig(a, compute_module=self.eig_compute_location)

        if self.eig_compute_location == "numpy" and xp.__name__ == "cupy":
            if self.use_pinned_memory:
                if isinstance(a, list):
                    w = [get_any_location_pinned(wi, input_location) for wi in w]
                    v = [get_any_location_pinned(vi, input_location) for vi in v]
                else:
                    w = get_any_location_pinned(w, input_location)
                    v = get_any_location_pinned(v, input_location)
            else:
                if isinstance(a, list):
                    w = [get_any_location(wi, input_location) for wi in w]
                    v = [get_any_location(vi, input_location) for vi in v]
                else:
                    w = get_any_location(w, input_location)
                    v = get_any_location(v, input_location)

        # Get the eigenvalues and eigenvectors.
        ws = xp.zeros((batchsize, self.m_0), dtype=in_type)
        vs = Y.copy()

        for i in range(batchsize):
            len_w = len(w[i])
            # Recover the full eigenvectors from the subspace.
            ws[i, :len_w] = w[i]
            if left:
                vs[i, :, :len_w] = xp.linalg.solve(v[i], p_back[i]).conj().T
            else:
                vs[i, :, :len_w] = p_back[i] @ v[i]

        return ws, vs

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

        batchsize = a_xx[0].shape[0]

        # NOTE: Here we could also use a good initial guess.
        Y = rng.random((batchsize, d, self.m_0)) + 1j * rng.random(
            (batchsize, d, self.m_0)
        )

        # Compute the contour integral.
        q0, q1 = self._contour_integrate(a_xx)

        # Compute first and second moment.
        P_0 = q0 @ Y
        P_1 = q1 @ Y

        # project the system
        if self.use_qr:
            a, p_back = self._project_qr(P_0, P_1)
        else:
            a, p_back = self._project_svd(P_0, P_1)

        return self._solve_reduced_system(a, Y, p_back)

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

        batchsize = a_xx[0].shape[0]

        # NOTE: Here we could also use a good initial guess.
        Y = rng.random((batchsize, d, self.m_0)) + 1j * rng.random(
            (batchsize, d, self.m_0)
        )
        Y_hat = rng.random((batchsize, d, self.m_0)) + 1j * rng.random(
            (batchsize, d, self.m_0)
        )

        # Compute the contour integral.
        q0, q1 = self._contour_integrate(a_xx)

        # Compute first and second moment.
        P_0 = q0 @ Y
        P_1 = q1 @ Y

        P_0_hat = Y_hat.conj().swapaxes(-2, -1) @ q0
        P_1_hat = Y_hat.conj().swapaxes(-2, -1) @ q1

        # project the system
        if self.use_qr:
            a, p_back = self._project_qr(P_0, P_1)
            a_hat, p_back_hat = self._project_qr(P_0_hat, P_1_hat, left=True)
        else:
            a, p_back = self._project_svd(P_0, P_1)
            a_hat, p_back_hat = self._project_svd(P_0_hat, P_1_hat, left=True)

        wrs, vrs = self._solve_reduced_system(a, Y, p_back)
        wls, vls = self._solve_reduced_system(a_hat, Y_hat, p_back_hat, left=True)

        return wrs, vrs, wls, vls

    @profiler.profile(level="api")
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
        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

        batch_shape = a_xx[0].shape[:-2]
        d = a_xx[0].shape[-1]

        # allow for higher dimensional inputs
        a_xx = tuple(a_x.reshape(-1, d, d) for a_x in a_xx)

        if d < self.m_0 and self.use_qr:
            warnings.warn(
                f"Subspace guess {self.m_0} is larger than the "
                f"dimension of the system {a_xx[0].shape[-1]}. "
                f"Setting subspace guess to {a_xx[0].shape[-1]}."
            )
            self.old_m_0 = self.m_0
            self.m_0 = a_xx[0].shape[-1]

        if left:
            wrs, vrs, wls, vls = self._two_sided(a_xx)
            wrs = wrs.reshape(*batch_shape, self.m_0)
            vrs = vrs.reshape(*batch_shape, d, self.m_0)
            wls = wls.reshape(*batch_shape, self.m_0)
            vls = vls.reshape(*batch_shape, d, self.m_0)
            out = (wrs, vrs, wls, vls)

        else:
            ws, vs = self._one_sided(a_xx)
            ws = ws.reshape(*batch_shape, self.m_0)
            vs = vs.reshape(*batch_shape, d, self.m_0)
            out = (ws, vs)

        # reset subspace guess
        if hasattr(self, "old_m_0"):
            self.m_0 = self.old_m_0
        return out
