# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import numpy.linalg as npla

from qttools.obc.obc import OBC


class Full(OBC):
    """Calculates surface Green's function by full eigenvalue solution.

    Implemented along the lines of [1]_.

    Parameters
    ----------
    a_ii : array_like
        On-diagonal block of the system matrix.
    a_ij : array_like
        Super-diagonal block of the system matrix.
    a_ji : array_like, optional
        Sub-diagonal block of the system matrix
    contact : str
        The contact side.

    Returns
    -------
    x_ii : np.ndarray
        The surface Green's function.

    References
    ----------
    .. [1] S. Brück, Ab-initio Quantum Transport Simulations for
       Nanoelectronic Devices, ETH Zurich, 2017.

    """

    def __call__(
        self,
        a_ii: np.ndarray,
        a_ij: np.ndarray,
        a_ji: np.ndarray,
        contact: str,
        out: None | np.ndarray = None,
    ):
        """Returns the surface Green's function."""
        stack_size = 1 if a_ii.ndim == 2 else a_ii.shape[0]
        identity = np.stack([np.eye(a_ii.shape[-1])] * stack_size)

        # Assemble the spectral transform of the companion linear system.
        inverse = npla.inv(a_ii + a_ij + a_ji)
        A = np.block(
            [
                [inverse @ a_ji, inverse @ -a_ij],
                [inverse @ a_ji, inverse @ -a_ij],
            ]
        ) - np.block(
            [
                [identity, np.zeros_like(a_ij)],
                [np.zeros_like(a_ij), np.zeros_like(a_ij)],
            ]
        )
        ws, vs = npla.eig(A)

        # Recover the original eigenvalues from the spectral transform.
        ws = 1 / ws + 1

        # Select the eigenvalues corresponding to non-decaying modes.
        inds = [np.argwhere(np.abs(w) > 1.0)[:, 0] for w in ws]
        ws = np.stack([np.diag(w[ind]) for w, ind in zip(ws, inds)])
        vs = np.stack([v[: a_ii.shape[-1], ind] for v, ind in zip(vs, inds)])

        # Calculate the surface Green's function.
        inverse = npla.inv(
            vs.conj().transpose(0, 2, 1) @ a_ii @ vs
            + vs.conj().transpose(0, 2, 1) @ a_ji @ vs @ npla.inv(ws)
        )
        x_ii = vs @ inverse @ vs.conj().transpose(0, 2, 1)

        if out is not None:
            out[...] = x_ii
            return

        return x_ii
