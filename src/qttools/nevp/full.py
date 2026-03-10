# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, sparse, xp
from qttools.kernels import linalg
from qttools.nevp.nevp import NEVP


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
        Whether to use pinnend memory if cupy is used. Default is
       `True`.
    reduce : bool, optional
        Whether to reduce the problem size by eliminating columns that
        are zero in the first and last coefficient blocks. These columns
        correspond to eigenvalues that are infinity or zero.
    a_xx_sparsity : tuple[sparse.csc_matrix, ...] or None, optional
        The sparsity patterns of the coefficient blocks of the NEVP.
        If `reduce` is `True`, this can be provided at instantiation to
        identify the zero columns and perform the reduction. If `reduce`
        is `True` and `a_xx` is not provided, the zero columns will be
        identified at runtime, which may introduce some overhead.

    """

    def __init__(
        self,
        eig_compute_location: str = "numpy",
        use_pinned_memory: bool = True,
        reduce: bool = False,
        a_xx_sparsity: tuple[sparse.csc_matrix, ...] | None = None,
    ):
        """Initializes the Full NEVP solver."""
        self.eig_compute_location = eig_compute_location
        self.use_pinned_memory = use_pinned_memory

        self.reduce = reduce

        if reduce:
            if a_xx_sparsity is None:
                warnings.warn(
                    "Reduction is enabled but no coefficient blocks are provided. "
                    "Zero columns will be identified at runtime, which may "
                    "introduce overhead.",
                )
            else:
                self.zero_indices, self.nonzero_indices, self.all_indices = (
                    self._find_zero_columns(a_xx_sparsity)
                )

                if self.zero_indices is None:
                    warnings.warn(
                        "No columns are zero in the first and last blocks. "
                        "Reduction has no effect.",
                    )
                    self.reduce = False

    @staticmethod
    def _find_zero_columns(
        a_xx: tuple[sparse.csc_matrix, ...],
    ) -> tuple[NDArray, NDArray, NDArray] | tuple[None, None, None]:
        """Determines the reduction indices for the full NEVP solver.

        This method identifies the zero columns in the first and last
        coefficient blocks, which correspond to eigenvalues that are
        infinity or zero, respectively. It returns the indices of the
        zero columns, the non-zero columns, and the concatenation of both.

        Parameters
        ----------
        a_xx : tuple[sparse.csc_matrix, ...]
            The coefficient blocks of the NEVP.

        Returns
        -------
        zero_indices : NDArray or None
            The indices of the zero columns.
        nonzero_indices : NDArray or None
            The indices of the non-zero columns.
        all_indices : NDArray or None
            The concatenation of zero and non-zero column indices.

        """
        if not all(a.shape[0] == a.shape[1] for a in a_xx):
            raise ValueError("All arrays in a_xx must be square.")

        if not all(a.shape[0] == a_xx[0].shape[0] for a in a_xx):
            raise ValueError("All arrays in a_xx must have the same shape.")

        zero_inds_first = xp.where(xp.diff(a_xx[0].indptr) == 0)[0]
        zero_inds_last = xp.where(xp.diff(a_xx[-1].indptr) == 0)[0]
        offset = a_xx[0].shape[1] * (len(a_xx) - 2)

        zero_inds = xp.concatenate((zero_inds_first, zero_inds_last + offset))
        all_inds = xp.arange(a_xx[0].shape[1] * (len(a_xx) - 1))
        nonzero_inds = xp.setdiff1d(all_inds, zero_inds)

        if len(zero_inds) == 0:
            # No columns are zero, reduction will have no effect.
            return None, None, None

        return zero_inds, nonzero_inds, xp.concatenate((nonzero_inds, zero_inds))

    def __call__(self, a_xx: tuple[NDArray, ...]) -> tuple[NDArray, NDArray]:
        """Solves the polynomial eigenvalue problem through linearization.

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
            The right eigenvectors.

        """
        # Allow for batched input.
        if a_xx[0].ndim == 2:
            a_xx = tuple(a_x[xp.newaxis, :, :] for a_x in a_xx)
        batch_shape = a_xx[0].shape[:-2]
        if a_xx[0].ndim > 3:
            a_xx = tuple(a_x.reshape(-1, *a_x.shape[-2:]) for a_x in a_xx)

        if self.reduce and not hasattr(self, "zero_indices"):
            # Identify zero columns at runtime if not provided at instantiation.
            self.zero_indices, self.nonzero_indices, self.all_indices = (
                self._find_zero_columns(
                    tuple(sparse.csc_matrix(a_x[0]) for a_x in a_xx)
                )
            )

            if self.zero_indices is None:
                # No columns are zero.
                warnings.warn(
                    "No columns are zero in the first and last blocks. "
                    "Reduction has no effect.",
                )
                self.reduce = False

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
            A_b = A[:, self.zero_indices, :][:, :, self.nonzero_indices]
            A_c = A[:, self.zero_indices, :][:, :, self.zero_indices]
            A = A[:, self.nonzero_indices, :][:, :, self.nonzero_indices]

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
            v[:, self.all_indices, :] = tmp

        # Recover the original eigenvalues from the spectral transform.
        w = xp.where((xp.abs(w) == 0.0), -1.0, w)
        w = 1 / w + 1
        v = v[:, : a_xx[0].shape[-1]]

        # Reshape to original batch shape.
        w = w.reshape(*batch_shape, -1)
        v = v.reshape(*batch_shape, v.shape[1], -1)

        return w, v
