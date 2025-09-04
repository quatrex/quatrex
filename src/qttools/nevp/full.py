# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

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
        Can be either "numpy" or "cupy". Only relevant if cupy is used.
    use_pinned_memory : bool, optional
        Whether to use pinnend memory if cupy is used.
        Default is `True`.

    """

    def __init__(
        self,
        eig_compute_location: str = "cupy",
        use_pinned_memory: bool = True,
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

        print("FULL: Size of matrix before deleting elements:", A.shape)

        # Find coloumns with all zeros
        nnz = xp.count_nonzero(A[0], axis=0)
        diag = xp.diagonal(A[0])

        i_z = xp.where(nnz == 0)[0]

        # Find coloumns with -1 on the diag and with norm 1
        i_mo = xp.where((diag == -1) & (nnz == 1))[0]

        # Concatenate and delete
        i_B = xp.concatenate((i_z, i_mo))
        i_A = xp.setdiff1d(xp.arange(A.shape[-1]), i_B)
        i_tot = xp.concatenate((i_A, i_B))

        A_b = A[:, i_B, :][:, :, i_B].diagonal(axis1=-2, axis2=-1)
        A_c = A[:, i_B, :][:, :, i_A]
        A = A[:, i_A, :][:, :, i_A]

        print("FULL: Size of matrix after deleting elements:", A.shape)

        w, v = linalg.eig(
            A,
            compute_module=self.eig_compute_location,
            use_pinned_memory=self.use_pinned_memory,
        )

        v_y = xp.divide(
            A_c @ v, w[:, xp.newaxis, :] - A_b[:, :, xp.newaxis]
        )  # shape: (batch, len(i_B), num_eig)

        # v[:,i_tot,:] = xp.concatenate([v, v_y], axis=1) Without Memory copy it does not work

        v_pre_sort = xp.concatenate([v, v_y], axis=1)
        v = xp.empty_like(v_pre_sort)
        v[:, i_tot, :] = v_pre_sort

        # Recover the original eigenvalues from the spectral transform.
        w = xp.where((xp.abs(w) == 0.0), -1.0, w)
        w = 1 / w + 1
        v = v[:, : a_xx[0].shape[-1]]

        return w, v
