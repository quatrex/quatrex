# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the selected inversion solver."""

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
from qttools.kernels import linalg
from qttools.utils.solvers_utils import get_batches


class RGF(GFSolver):
    """Selected inversion solver based on the Schur complement.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        100.

    """

    def __init__(self, max_batch_size: int = 100) -> None:
        """Initializes the selected inversion solver."""
        self.max_batch_size = max_batch_size

    def selected_inv(
        self,
        a: DSDBSparse,
        out: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
    ) -> None:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        out : DSDBSparse
            Preallocated output matrix.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.

        """
        # Initialize dense temporary buffers for the diagonal blocks.
        x_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        x = out

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            # See if there is an OBC block for the current layer.
            obc = obc_blocks.retarded[0]
            a_00 = (
                a_.blocks[0, 0] if obc is None else a_.blocks[0, 0] - obc[stack_slice]
            )

            x_diag_blocks[0] = linalg.inv(a_00)

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # See if there is an OBC block for the current layer.
                obc = obc_blocks.retarded[j]
                a_jj = (
                    a_.blocks[j, j]
                    if obc is None
                    else a_.blocks[j, j] - obc[stack_slice]
                )

                x_diag_blocks[j] = linalg.inv(
                    a_jj - a_.blocks[j, i] @ x_diag_blocks[i] @ a_.blocks[i, j]
                )

            # We need to write the last diagonal block to the output.
            x_.blocks[a.num_blocks - 1, a.num_blocks - 1] = x_diag_blocks[-1]

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                x_ii = x_diag_blocks[i]
                x_jj = x_diag_blocks[j]
                a_ij = a_.blocks[i, j]

                x_ji = -x_jj @ a_.blocks[j, i] @ x_ii
                x_.blocks[j, i] = x_ji
                x_.blocks[i, j] = -x_ii @ a_ij @ x_jj

                # NOTE: Cursed Python multiple assignment syntax.
                x_.blocks[i, i] = x_diag_blocks[i] = x_ii - x_ii @ a_ij @ x_ji

    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        obc_blocks: OBCBlocks | None = None,
        a_hat: DSDBSparse | None = None,
        return_retarded: bool = False,
        return_meir_wingreen_current: bool = False,
        return_device_current: bool = False,
    ) -> None | tuple | NDArray:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        sigma_lesser : DSDBSparse
            Lesser matrix. This matrix is expected to be skew-hermitian,
            i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSDBSparse, ...]
            Preallocated output matrices.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        a_hat : DSDBSparse, optional
            The bare system matrix without self-energy contributions.
            This is used to compute the device current.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_meir_wingreen_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False.
        return_device_current : bool, optional
            Whether to additionally compute and return the coherent
            bond current between adjacent blocks, evaluated from the
            *dense* off-diagonal Green's function blocks. Only
            supported together with `return_meir_wingreen_current`. By default False.

        Returns
        -------
        None | tuple | NDArray
            If `return_meir_wingreen_current` is True, returns the current for each
            layer. If `return_device_current` is True, the
            bond current as well.

        """
        # Initialize empty lists for the dense diagonal blocks.
        xr_diag_blocks: list[NDArray | None] = [None] * sigma_lesser.num_blocks
        xl_diag_blocks: list[NDArray | None] = [None] * sigma_lesser.num_blocks
        xg_diag_blocks: list[NDArray | None] = [None] * sigma_lesser.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=sigma_lesser.num_blocks)

        # Allocate a buffer for the current. This includes current
        # between each layer and from/to the leads (in total
        # num_blocks + 1).
        if return_meir_wingreen_current:
            meir_wingreen_current = xp.zeros(
                (*sigma_lesser.local_stack_shape, sigma_lesser.num_blocks + 1),
                dtype=sigma_lesser.dtype,
            )
        if return_device_current:
            device_current = xp.zeros(
                (*sigma_lesser.local_stack_shape, sigma_lesser.num_blocks - 1),
                dtype=sigma_lesser.dtype,
            )
            if a_hat is None:
                raise ValueError(
                    "The bare system matrix must be provided to compute the device current."
                )

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(
            sigma_lesser.local_stack_shape[0], self.max_batch_size
        )

        # xr will be the third element of the tuple.
        xl, xg, *xr = out
        if return_retarded:
            if len(xr) != 1:
                raise ValueError("Invalid number of output matrices.")
            xr = xr[0]

        if xl.symmetry not in [None, "skew-hermitian"]:
            raise ValueError(
                "Invalid symmetry for lesser Green's function. "
                "Expected None or 'skew-hermitian'."
            )
        if xg.symmetry not in [None, "skew-hermitian"]:
            raise ValueError(
                "Invalid symmetry for greater Green's function. "
                "Expected None or 'skew-hermitian'."
            )

        # Perform the selected solve by batches.
        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            a_hat_ = a_hat.stack[stack_slice] if a_hat is not None else None
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xl_ = xl.stack[stack_slice]
            xg_ = xg.stack[stack_slice]
            if return_retarded:
                xr_ = xr.stack[stack_slice]

            # Check if there are OBC blocks for the current layer.
            obc_r = obc_blocks.retarded[0]
            a_jj = (
                a_.blocks[0, 0]
                if obc_r is None
                else a_.blocks[0, 0] - obc_r[stack_slice]
            )
            obc_l = obc_blocks.lesser[0]
            sl_jj = (
                sigma_lesser_.blocks[0, 0]
                if obc_l is None
                else sigma_lesser_.blocks[0, 0] + obc_l[stack_slice]
            )
            obc_g = obc_blocks.greater[0]
            sg_jj = (
                sigma_greater_.blocks[0, 0]
                if obc_g is None
                else sigma_greater_.blocks[0, 0] + obc_g[stack_slice]
            )

            xr_jj = linalg.inv(a_jj)
            xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
            xr_diag_blocks[0] = xr_jj
            xl_diag_blocks[0] = xr_jj @ sl_jj @ xr_jj_dagger
            xg_diag_blocks[0] = xr_jj @ sg_jj @ xr_jj_dagger

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # Check if there are OBC blocks for the current layer.
                obc_r = obc_blocks.retarded[j]
                a_jj = (
                    a_.blocks[j, j]
                    if obc_r is None
                    else a_.blocks[j, j] - obc_r[stack_slice]
                )
                obc_l = obc_blocks.lesser[j]
                sl_jj = (
                    sigma_lesser_.blocks[j, j]
                    if obc_l is None
                    else sigma_lesser_.blocks[j, j] + obc_l[stack_slice]
                )
                obc_g = obc_blocks.greater[j]
                sg_jj = (
                    sigma_greater_.blocks[j, j]
                    if obc_g is None
                    else sigma_greater_.blocks[j, j] + obc_g[stack_slice]
                )

                # Get the blocks that are used multiple times.
                a_ji = a_.blocks[j, i]
                xr_ii = xr_diag_blocks[i]

                # Precompute the transposes that are used multiple times.
                a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

                # Precompute some terms that are used multiple times.
                a_ji_xr_ii = a_ji @ xr_ii

                xr_jj = linalg.inv(a_jj - a_ji_xr_ii @ a_.blocks[i, j])
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_diag_blocks[j] = xr_jj

                a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_lesser_.blocks[i, j]

                xl_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sl_jj
                        + a_ji @ xl_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    @ xr_jj_dagger
                )

                a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_greater_.blocks[i, j]

                xg_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sg_jj
                        + a_ji @ xg_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    @ xr_jj_dagger
                )

            # We need to write the last diagonal blocks to the output.
            xl_.blocks[a.num_blocks - 1, a.num_blocks - 1] = 0.5 * (
                xl_diag_blocks[-1] - xl_diag_blocks[-1].conj().swapaxes(-2, -1)
            )
            xg_.blocks[a.num_blocks - 1, a.num_blocks - 1] = 0.5 * (
                xg_diag_blocks[-1] - xg_diag_blocks[-1].conj().swapaxes(-2, -1)
            )
            if return_retarded:
                xr_.blocks[a.num_blocks - 1, a.num_blocks - 1] = xr_diag_blocks[-1]

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                # Get the blocks that are used multiple times.
                xr_ii = xr_diag_blocks[i]
                xr_jj = xr_diag_blocks[j]
                a_ij = a_.blocks[i, j]
                a_ji = a_.blocks[j, i]
                xl_ii = xl_diag_blocks[i]
                xl_jj = xl_diag_blocks[j]
                xg_ii = xg_diag_blocks[i]
                xg_jj = xg_diag_blocks[j]
                sigma_lesser_ij = sigma_lesser_.blocks[i, j]
                sigma_greater_ij = sigma_greater_.blocks[i, j]

                # Precompute the transposes that are used multiple times.
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)

                # Precompute the terms that are used multiple times.
                xr_ii_a_ij = xr_ii @ a_ij
                a_ij_dagger_xr_ii_dagger = xr_ii_a_ij.conj().swapaxes(-2, -1)
                xr_jj_a_ji = xr_jj @ a_ji
                a_ji_dagger_xr_jj_dagger = xr_jj_a_ji.conj().swapaxes(-2, -1)
                xr_ii_a_ij_xr_jj = xr_ii_a_ij @ xr_jj
                xr_jj_dagger_a_ij_dagger_xr_ii_dagger = (
                    xr_ii_a_ij_xr_jj.conj().swapaxes(-2, -1)
                )
                xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji

                temp_1x = (
                    xr_ii_a_ij_xr_jj_a_ji @ xl_ii
                    - xr_ii @ sigma_lesser_ij @ xr_jj_dagger_a_ij_dagger_xr_ii_dagger
                )
                temp_1x -= temp_1x.conj().swapaxes(-2, -1)
                temp_2x = xr_ii_a_ij @ xl_jj

                xl_ij = (
                    -temp_2x
                    - xl_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
                )

                xl_.blocks[i, j] = xl_ij
                if xl_.symmetry is None:
                    xl_.blocks[j, i] = -xl_ij.conj().swapaxes(-2, -1)

                if return_device_current:
                    # Coherent bond current across the interface between
                    # block i and i+1, using the *dense* off-diagonal
                    # G^< block (xl_ij) and the full effective coupling
                    # from the system matrix (E*S - H).
                    a_hat_ij = a_hat_.blocks[i, j]
                    a_hat_ji = a_hat_.blocks[j, i]

                    gl_ji = -xl_ij.conj().swapaxes(-2, -1)
                    device_current[stack_slice, ..., i] = xp.trace(
                        xl_ij @ a_hat_ji - a_hat_ij @ gl_ji,
                        axis1=-2,
                        axis2=-1,
                    )

                xl_diag_blocks[i] = xl_ii + temp_2x @ a_ij_dagger_xr_ii_dagger + temp_1x
                xl_.blocks[i, i] = 0.5 * (
                    xl_diag_blocks[i] - xl_diag_blocks[i].conj().swapaxes(-2, -1)
                )

                temp_1x = (
                    xr_ii_a_ij_xr_jj_a_ji @ xg_ii
                    - xr_ii @ sigma_greater_ij @ xr_jj_dagger_a_ij_dagger_xr_ii_dagger
                )
                temp_1x -= temp_1x.conj().swapaxes(-2, -1)
                temp_2x = xr_ii_a_ij @ xg_jj

                xg_ij = (
                    -temp_2x
                    - xg_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_greater_ij @ xr_jj_dagger
                )

                xg_.blocks[i, j] = xg_ij
                if xg_.symmetry is None:
                    xg_.blocks[j, i] = -xg_ij.conj().swapaxes(-2, -1)

                xg_diag_blocks[i] = xg_ii + temp_2x @ a_ij_dagger_xr_ii_dagger + temp_1x
                xg_.blocks[i, i] = 0.5 * (
                    xg_diag_blocks[i] - xg_diag_blocks[i].conj().swapaxes(-2, -1)
                )

                if return_meir_wingreen_current:
                    a_ji_dagger = a_ji.conj().swapaxes(-2, -1)
                    a_ji_xr_ii = a_ji @ xr_ii
                    a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_lesser_ij
                    sigma_lesser_tilde = (
                        a_ji @ xl_ii @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    a_ji_xr_ii_sx_ij = a_ji_xr_ii @ sigma_greater_ij
                    sigma_greater_tilde = (
                        a_ji @ xg_ii @ a_ji_dagger
                        + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij
                    )
                    meir_wingreen_current[stack_slice, ..., j] = xp.trace(
                        sigma_greater_tilde @ xl_diag_blocks[j]
                        - xg_diag_blocks[j] @ sigma_lesser_tilde,
                        axis1=-2,
                        axis2=-1,
                    )

                xr_diag_blocks[i] = xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
                if return_retarded:
                    xr_.blocks[i, i] = xr_diag_blocks[i]

            # The contact (lead) currents from the boundary self-energies.
            if return_meir_wingreen_current:
                meir_wingreen_current[stack_slice, ..., 0] = xp.trace(
                    obc_blocks.greater[0][stack_slice] @ xl_diag_blocks[0]
                    - xg_diag_blocks[0] @ obc_blocks.lesser[0][stack_slice],
                    axis1=-2,
                    axis2=-1,
                )
                # NOTE: Negative sign is needed to get the current flowing
                # in the correct direction (positive from left to right).
                meir_wingreen_current[stack_slice, ..., -1] = -xp.trace(
                    obc_blocks.greater[-1][stack_slice] @ xl_diag_blocks[-1]
                    - xg_diag_blocks[-1] @ obc_blocks.lesser[-1][stack_slice],
                    axis1=-2,
                    axis2=-1,
                )

        if return_meir_wingreen_current and return_device_current:
            return meir_wingreen_current, device_current
        elif return_meir_wingreen_current:
            return meir_wingreen_current
        elif return_device_current:
            return device_current
