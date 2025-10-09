# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.kernels import linalg
from qttools.nevp.nevp import NEVP
from qttools.profiling import Profiler

profiler = Profiler()


class Full(NEVP):
    """An NEVP solver based on linearization.

    Implemented along the lines of what is described in [^1].

    [^1]: S. Brück, Ab-initio Quantum Transport Simulations for
    Nanoelectronic Devices, ETH Zurich, 2017.

    Parameters
    ----------
    eig_compute_location : str, optional
        The location where to compute the eigenvalues and eigenvectors.
        Can be either "numpy" or "cupy" or "nvmath".
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.
    a_sparsity : tuple[NDArray, ...] | None, optional
        The sparsity patterns of the coefficient blocks of the NEVP.
        Every array is a 2D matrix with entries 0 or 1 indicating
        the sparsity pattern of the corresponding coefficient block.
    reduce : bool, optional
        Whether to reduce the problem size by eliminating columns
        that are zero in the first and last coefficient blocks.
        These corresponding eigenvalues are infinity or zero.

    """

    def __init__(
        self,
        eig_compute_location: str = "numpy",
        use_pinned_memory: bool = True,
        a_sparsity: tuple[NDArray, ...] | None = None,
        reduce: bool = False,
    ):
        """Initializes the Full NEVP solver."""
        self.eig_compute_location = eig_compute_location
        self.use_pinned_memory = use_pinned_memory

    @profiler.profile(level="api")
    def __call__(self, a_xx: tuple[NDArray, ...]) -> tuple[NDArray, NDArray]:
        """Solves the polynomial eigenvalue problem through linearization.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.
        side : str, optional
            Whether for the left or right eigenvectors should be solved.
            Relevant only for the sparsity reduction.

        Returns
        -------
        ws : NDArray
            The eigenvalues.
        vs : NDArray
            The right eigenvectors.

        """
        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)

        inverse = linalg.inv(sum(a_xx))

        # NOTE: CuPy does not expose a `block` function.
        row = xp.concatenate(
            [inverse @ sum(a_xx[:i]) for i in range(1, len(a_xx) - 1)]
            + [inverse @ -a_xx[-1]],
            axis=-1,
        )
        A = xp.concatenate([row] * (len(a_xx) - 1), axis=-2)
        B = xp.kron(xp.tri(len(a_xx) - 2).T, xp.eye(a_xx[0].shape[-1]))
        A[:, : B.shape[0], : B.shape[1]] -= B

        # Concatenate and delete
        if self.reduce:
            A_b = A[:, self.zero_indices[side], :][:, :, self.nonzero_indices[side]]
            A_c = A[:, self.zero_indices[side], :][:, :, self.zero_indices[side]]
            A = A[:, self.nonzero_indices[side], :][:, :, self.nonzero_indices[side]]

        w, v = linalg.eig(
            A,
            compute_module=self.eig_compute_location,
            use_pinned_memory=self.use_pinned_memory,
        )

        if self.reduce:
            v_zero = xp.divide(
                A_b @ v,
                w[:, xp.newaxis, :]
                - A_c.diagonal(axis1=-2, axis2=-1)[:, :, xp.newaxis],
            )

            tmp = xp.concatenate([v, v_zero], axis=1)
            v = xp.empty_like(tmp)
            v[:, self.all_indices[side], :] = tmp

        # Recover the original eigenvalues from the spectral transform.
        w = xp.where((xp.abs(w) == 0.0), -1.0, w)
        w = 1 / w + 1
        v = v[:, : a_xx[0].shape[-1]]

        return w, v
