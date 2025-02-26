# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.kernels.eig import eig
from qttools.nevp.nevp import NEVP


class Full(NEVP):
    """An NEVP solver based on linearization.

    Warning
    -------
    This solver will create very large matrices and should only be used
    for very small problems. It is intended as a reference
    implementation and should probably not be used in production code.

    Implemented along the lines of what is described in [^1].

    [^1]: S. Brück, Ab-initio Quantum Transport Simulations for
    Nanoelectronic Devices, ETH Zurich, 2017.

    """

    def _solve(
        self, a_xx: tuple[NDArray, ...], eig_compute_location: str = "numpy"
    ) -> tuple[NDArray, NDArray]:
        """Solves the plynomial eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.
        eig_compute_location : str, optional
            The location where to compute the eigenvalues and eigenvectors.
            Can be either "numpy" or "cupy". Only relevant if cupy is used.

        Returns
        -------
        ws : NDArray
            The eigenvalues.
        vs : NDArray
            The right eigenvectors.

        """
        inverse = xp.linalg.inv(sum(a_xx))

        # NOTE: CuPy does not expose a `block` function.
        row = xp.concatenate(
            [inverse @ sum(a_xx[:i]) for i in range(1, len(a_xx) - 1)]
            + [inverse @ -a_xx[-1]],
            axis=-1,
        )
        A = xp.concatenate([row] * (len(a_xx) - 1), axis=-2)
        B = xp.kron(xp.tri(len(a_xx) - 2).T, xp.eye(a_xx[0].shape[-1]))
        A[:, : B.shape[0], : B.shape[1]] -= B

        w, v = eig(A, compute_module=eig_compute_location)

        # Recover the original eigenvalues from the spectral transform.
        w = xp.where((xp.abs(w) == 0.0), -1.0, w)
        w = 1 / w + 1
        v = v[:, : a_xx[0].shape[-1]]

        return w, v

    def __call__(
        self,
        a_xx: tuple[NDArray, ...],
        left: bool = False,
        eig_compute_location: str = "numpy",
    ) -> tuple:
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
        eig_compute_location : str, optional
            The location where to compute the eigenvalues and eigenvectors.
            Can be either "cupy" or "numpy". Only relevant if cupy is used.

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

        wrs, vrs = self._solve(a_xx, eig_compute_location=eig_compute_location)

        if left:
            # solve for the left eigenvectors
            # by solving the right eigenvectors of the adjoint problem
            wls, vls = self._solve(
                tuple(a_x.conj().swapaxes(-2, -1) for a_x in a_xx),
                eig_compute_location=eig_compute_location,
            )
            wls = wls.conj()

            return wrs, vrs, wls, vls

        return wrs, vrs
