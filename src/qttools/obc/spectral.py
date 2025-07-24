# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import _block_view
from qttools.kernels import linalg
from qttools.nevp import NEVP
from qttools.obc.obc import OBCSolver
from qttools.profiling import Profiler, decorate_methods

profiler = Profiler()


@decorate_methods(
    profiler.profile(level="debug"),
    exclude=["__call__", "__init__"],
)
class Spectral(OBCSolver):
    """Spectral open-boundary condition solver.

    This technique of obtaining the surface Green's function is based on
    the solution of a non-linear eigenvalue problem (NEVP), defined via
    the system-matrix blocks in the semi-infinite contacts.

    Those eigenvalues corresponding to reflected modes are filtered out,
    so that only the ones that correspond to modes that propagate into
    the leads or those that decay away from the system are retained.

    The surface Green's function is then calculated from these filtered
    eigenvalues and eigenvectors.

    Parameters
    ----------
    nevp : NEVP
        The non-linear eigenvalue problem solver to use.
    block_sections : int, optional
        The number of sections to split the periodic matrix layer into.
    min_decay : float, optional
        The decay threshold after which modes are considered to be
        evanescent.
    max_decay : float, optional
        The maximum decay to consider for evanescent modes.
        The default is 6.9 which corresponds to 1000 in the eigenvalues.
    num_ref_iterations : int, optional
        The number of refinement iterations to perform on the surface
        Green's function.
    min_propagation : float, optional
        The minimum ratio between the real and imaginary part of the
        group velocity of a mode. This ratio is used to determine how
        clearly a mode propagates.
    residual_tolerance : float, optional
        The tolerance for the residual of the NEVP.
    residual_normalization : bool
        If the residual should be normalized by the eigenvalue.
    eta_decay : float, optional
        Small value to separate very slow decaying modes from
        non-decaying ones.

        [^1]: S. Brück, et al., Efficient algorithms for large-scale
        quantum transport calculations, The Journal of Chemical Physics,
        2017.

    """

    def __init__(
        self,
        nevp: NEVP,
        block_sections: int = 1,
        min_decay: float = 1e-3,
        max_decay: float = 6.9,
        num_ref_iterations: int = 2,
        min_propagation: float = 0.01,
        residual_tolerance: float = 1e-3,
        residual_normalization: bool = True,
        warning_threshold: float = 1e-1,
        eta_decay: float = 1e-14,
    ) -> None:
        """Initializes the spectral OBC solver."""
        self.nevp = nevp

        self.min_decay = min_decay
        self.max_decay = max_decay

        self.num_ref_iterations = num_ref_iterations
        self.block_sections = block_sections

        self.min_propagation = min_propagation
        self.residual_tolerance = residual_tolerance
        self.residual_normalization = residual_normalization
        self.warning_threshold = warning_threshold
        self.eta_decay = eta_decay

    def _extract_subblocks(
        self,
        a_ji: NDArray,
        a_ii: NDArray,
        a_ij: NDArray,
    ) -> tuple[NDArray, ...]:
        """Extracts the coefficient blocks from the periodic matrix.

        Parameters
        ----------
        a_ji : NDArray
            The subdiagonal block of the periodic matrix.
        a_ii : NDArray
            The diagonal block of the periodic matrix.
        a_ij : NDArray
            The superdiagonal block of the periodic matrix.

        Returns
        -------
        blocks : tuple[NDArray, ...]
            The non-zero blocks making up the matrix layer.

        """
        # Construct layer of periodic matrix in semi-infinite lead.
        layer = (a_ji, a_ii, a_ij)
        if self.block_sections == 1:
            return layer

        # Get a nested block view of the layer.
        view = _block_view(xp.concatenate(layer, axis=-1), -1, 3 * self.block_sections)
        view = _block_view(view, -2, self.block_sections)

        # Make sure that the reduction leads to periodic sublayers.
        relative_errors = xp.zeros(self.block_sections - 1)
        first_block_norm = xp.linalg.norm(view[0, :])
        for i in range(1, self.block_sections):
            relative_errors[i - 1] = (
                xp.linalg.norm(view[0, :] - xp.roll(view[i, :], -i, axis=0))
                / first_block_norm
            )

        if xp.max(relative_errors) > 1e-3:
            warnings.warn(
                f"Requested block sectioning is not periodic. ({xp.max(relative_errors):.2e})",
                RuntimeWarning,
            )

        # Select relevant blocks and remove empty ones.
        blocks = view[0, : -self.block_sections + 1]
        return tuple(block for block in blocks if xp.any(block))

    def _compute_dE_dk(self, ws: NDArray, vs: NDArray, a_xx: list[NDArray]) -> NDArray:
        """Computes the group velocity of the modes.

        Parameters
        ----------
        ws : NDArray
            The eigenvalues of the NEVP.
        vs : NDArray
            The right eigenvectors of the NEVP.
        a_xx : tuple[NDArray, ...]
            The blocks of the periodic matrix.

        Returns
        -------
        dEk_dk : NDArray
            The group velocity of the modes.

        """

        b = len(a_xx) // 2

        with warnings.catch_warnings(
            action="ignore", category=RuntimeWarning
        ):  # Ignore division by zero.

            dEk_dk = -sum(
                (1j * n)
                * xp.diagonal(
                    vs.conj().swapaxes(-1, -2) @ a_x @ vs,
                    axis1=-2,
                    axis2=-1,
                )
                * ws**n
                for a_x, n in zip(a_xx, range(-b, b + 1))
            )

        return dEk_dk

    def _find_reflected_modes(
        self,
        ws: NDArray,
        vs: NDArray,
        a_xx: list[NDArray],
        find_injected: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray, NDArray]:
        """Determines which eigenvalues correspond to reflected (and injected) modes.

        For the computation of the surface Green's function, only the
        eigenvalues corresponding to modes that propagate or decay into
        the leads are retained.

        Parameters
        ----------
        ws : NDArray
            The eigenvalues of the NEVP.
        vs : NDArray
            The right eigenvectors of the NEVP.
        a_xx : tuple[NDArray, ...]
            The blocks of the periodic matrix.
        find_injected: bool, optional
            Whether to find the injected eigenvector

        Returns
        -------
        mask_reflected : NDArray
            A boolean mask indicating which eigenvalues correspond to
            reflected modes.
        mask_injected : NDArray, optional
            A boolean mask indicating which eigenvalues correspond to
            injected modes.
        dEk_dK_injected : NDArray, optional
            List of dEk_dK values corresponding to injected modes

        """

        batchsize = a_xx[0].shape[0]

        if batchsize != 1 and find_injected:
            raise ValueError(
                "The injection vector can only be calculated with batchsize = 1"
            )

        # Calculate the residual
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):

            products = sum(
                a_x @ vs * ws[:, xp.newaxis, :] ** (i - len(a_xx) // 2)
                for i, a_x in enumerate(a_xx)
            )

            residuals = xp.linalg.norm(products, axis=-2)

            # eigenvectors are not necessarily normalized
            eigenvector_norm = xp.linalg.norm(vs, axis=-2)
            residuals /= eigenvector_norm

            if self.residual_normalization:
                residuals /= xp.abs(ws)

        # Calculate the group velocity to select propagation direction.
        # The formula can be derived by taking the derivative of the
        # polynomial eigenvalue equation with respect to k.
        # NOTE: This is actually only correct if we have no overlap.

        dEk_dk = self._compute_dE_dk(ws, vs, a_xx)

        with warnings.catch_warnings(
            action="ignore", category=RuntimeWarning
        ):  # Ignore zero log and division by zero.
            ks = -1j * xp.log(ws)

        # replace nan and infs with 0 due to zero eigenvalues
        dEk_dk = xp.nan_to_num(dEk_dk, nan=0, posinf=0, neginf=0)
        ks = xp.nan_to_num(ks, nan=0, posinf=0, neginf=0)

        # Find eigenvalues that correspond to reflected modes. These are
        # modes that either propagate into the leads or decay away from
        # the system.

        # Determine (matched) modes that decay slow enough to be
        # considered propagating.
        mask_propagating = xp.abs(ks.imag) < self.min_decay

        # fast enough propagation (group velocity)
        eta = xp.finfo(dEk_dk.dtype).eps
        mask_propagating &= self.min_propagation < abs(dEk_dk.real) / (
            abs(dEk_dk.imag) + eta
        )
        # propgation direction
        mask_propagating &= dEk_dk.real < 0

        # Make sure decaying modes decay fast enough.
        mask_decaying = ks.imag < -self.min_decay

        # capture slow decaying modes
        # modes that arent clearly propagating
        mask_decaying |= (
            self.min_propagation >= abs(dEk_dk.real) / (abs(dEk_dk.imag) + eta)
        ) & (ks.imag < -self.eta_decay)

        # ingore modes that decay incredibly fast
        mask_decaying &= ks.imag > -self.max_decay

        mask_reflected = (mask_propagating | mask_decaying) & (
            residuals < self.residual_tolerance
        )

        # Calulate injecting modes
        if find_injected:

            mask_injected = dEk_dk.real > 0
            mask_injected &= xp.abs(ks.imag) < self.min_decay
            mask_injected &= self.min_propagation < abs(dEk_dk.real) / (
                abs(dEk_dk.imag) + eta
            )

            dEk_dK_injected = dEk_dk[mask_injected]

            return mask_reflected, mask_injected, dEk_dK_injected

        return mask_reflected

    def _upscale_eigenmodes(
        self,
        ws: NDArray,
        vs: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Upscales the eigenvectors to the full periodic matrix layer.

        The extraction of subblocks and hence the solution of a higher-
        ordere, but smaller, NEVP leads to eigenvectors that are only
        defined on the reduced matrix layer. This function upscales the
        eigenvectors back to the full periodic matrix layer.

        Parameters
        ----------
        ws : NDArray
            The eigenvalues of the NEVP.
        vs : NDArray
            The eigenvectors of the (potentially) higher order NEVP.

        Returns
        -------
        ws : NDArray
            The upscaled eigenvalues.
        vs : NDArray
            The upscaled eigenvectors.

        """
        if self.block_sections == 1:
            return ws, vs / xp.linalg.norm(vs, axis=-2, keepdims=True)

        batchsize, subblock_size, num_modes = vs.shape
        block_size = subblock_size * self.block_sections

        ws_upscaled = xp.array([ws**n for n in range(self.block_sections)]).swapaxes(
            0, 1
        )

        vs_upscaled = (
            ws_upscaled[:, :, xp.newaxis, :] * vs[:, xp.newaxis, :, :]
        ).reshape(batchsize, block_size, num_modes)

        with warnings.catch_warnings(
            action="ignore", category=RuntimeWarning
        ):  # Ignore division by zero.
            vs_upscaled = vs_upscaled / xp.linalg.norm(
                vs_upscaled, axis=-2, keepdims=True
            )

        return ws**self.block_sections, vs_upscaled

    def _compute_x_ii(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        ws: NDArray,
        vs: NDArray,
        mask: NDArray,
    ) -> NDArray:
        """Computes the surface Green's function.

        Parameters
        ----------
        a_ii : NDArray
            The diagonal block of the periodic matrix.
        a_ij : NDArray
            The superdiagonal block of the periodic matrix.
        a_ji : NDArray
            The subdiagonal block of the periodic matrix.
        ws : NDArray
            The eigenvalues of the NEVP.
        vs : NDArray
            The right eigenvectors of the NEVP.
        mask : NDArray
            A boolean mask indicating which eigenvalues correspond to
            reflected modes.

        Returns
        -------
        x_ii : NDArray
            The surface Green's function.
        x_ii_a_ij : NDArray
            The Bloch matrix.

        """
        # Equation (13.1).
        x_ii_a_ij = xp.zeros((mask.shape[0], *a_ij.shape[-2:]), dtype=a_ij.dtype)
        for i, m in enumerate(mask):
            vr = vs[i][:, m]
            w = ws[i, m]
            # Moore-Penrose pseudoinverse.
            v_inv = linalg.inv(vr.conj().T @ vr) @ vr.conj().T
            x_ii_a_ij[i] = vr / w @ v_inv

        # Calculate the surface Green's function.
        return linalg.inv(a_ii + a_ji @ x_ii_a_ij)

    @profiler.profile(level="api")
    def __call__(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        return_injected: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray, NDArray]:
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
        return_injected: bool, optional
            Whether to return the injection vector

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.
        sigma_retarded: NDArray
            The boundary self energy. Returned only if return_injected
            is True. 
        inj: NDArray
            The Injection vector. Returned only if return_injected is
            True. (only compatible with batchsize = 1)
        w_inj: NDArray
            The eigenvalues of the injected modes. Returned only if
            return_injected is True. (only compatible with batchsize =
            1)

        """

        #if a_ii.ndim != 2 and return_injected:
        #   raise NotImplementedError

        if a_ii.ndim == 2:
            a_ii = a_ii[xp.newaxis, :, :]
            a_ij = a_ij[xp.newaxis, :, :]
            a_ji = a_ji[xp.newaxis, :, :]

        blocks = self._extract_subblocks(a_ji, a_ii, a_ij)
        ws, vs = self.nevp(blocks)

        ws, vs = self._upscale_eigenmodes(ws, vs)

        if return_injected:
            mask_reflected, mask_injected, dE_dK_injected = self._find_reflected_modes(
                ws,
                vs,
                a_xx=[a_ji, a_ii, a_ij],
                find_injected=return_injected,
            )
        else:
            mask_reflected = self._find_reflected_modes(
                ws,
                vs,
                a_xx=[a_ji, a_ii, a_ij],
            )

        x_ii = self._compute_x_ii(a_ii, a_ij, a_ji, ws, vs, mask_reflected)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x_ii = linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        x_ii_ref = linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        # Check the batch average recursion error.
        recursion_error = xp.max(
            xp.linalg.norm(x_ii_ref - x_ii, axis=(-2, -1))
            / xp.linalg.norm(x_ii_ref, axis=(-2, -1))
        )
        if recursion_error > self.warning_threshold:
            warnings.warn(
                f"High relative recursion error: {recursion_error:.2e}",
                RuntimeWarning,
            )

        # Calculate the injection vector and return it together with the boundary self-energy and the injected eigenvalues
        if return_injected:

            mask_injected = mask_injected[0, :]
            vrs_inj = vs[0][:, mask_injected]
            wrs_inj = xp.diag(ws[0, mask_injected])

            # Flux normalization
            vrs_inj = vrs_inj / xp.sqrt(dE_dK_injected[None, :])

            # Compute boundary self energy
            sigma_retarded = a_ji[0, :, :] @ x_ii[0, :, :] @ a_ij[0, :, :]

            # Compute injection vector
            injection = (
                -a_ji[0, :, :] @ vrs_inj @ linalg.inv(wrs_inj)
                - sigma_retarded @ vrs_inj
            )
            
            inj_phase = xp.angle(injection[0,:])
            injection = xp.divide(injection, xp.exp(1j * inj_phase))

            return x_ii_ref, sigma_retarded, injection, ws[0, mask_injected]

        return x_ii_ref
