# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks
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
        a: DSBSparse,
        obc_blocks: OBCBlocks | None = None,
        out: DSBSparse | None = None,
    ) -> None | DSBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : DSBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSBSparse object.

        """
        # Initialize dense temporary buffers for the diagonal blocks.
        x_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        if out is not None:
            x = out
        else:
            x = a.__class__.zeros_like(a)

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            # See if there is an OBC block for the current layer.
            obc = obc_blocks.retarded[0]
            a_00 = (
                a_.blocks[0, 0] if obc is None else a_.blocks[0, 0] - obc[stack_slice]
            )

            x_diag_blocks[0] = xp.linalg.inv(a_00)

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

                x_diag_blocks[j] = xp.linalg.inv(
                    a_jj - a_.blocks[j, i] @ x_diag_blocks[i] @ a_.blocks[i, j]
                )

            # We need to write the last diagonal block to the output.
            x_.blocks[j, j] = x_diag_blocks[j]

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

        if out is None:
            return x

    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        obc_blocks: OBCBlocks | None = None,
        out: tuple[DSBSparse, ...] | None = None,
        return_retarded: bool = False,
        return_current: bool = False,
    ) -> None | tuple | NDArray:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        sigma_lesser : DSBSparse
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : tuple[DSBSparse, ...] | None, optional
            Preallocated output matrices, by default None
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False.

        Returns
        -------
        None | tuple | NDArray
            If `out` is None, returns None. Otherwise, the solutions are
            returned as DSBParse matrices. If `return_retarded` is True,
            returns a tuple with the retarded Green's function as the
            last element. If `return_current` is True, returns the
            current for each layer.

        """
        # Initialize empty lists for the dense diagonal blocks.
        xr_diag_blocks: list[NDArray | None] = [None] * a.num_blocks
        xl_diag_blocks: list[NDArray | None] = [None] * a.num_blocks
        xg_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        if return_current:
            # Allocate a buffer for the current.
            current = xp.zeros((a.shape[0], a.num_blocks - 1), dtype=a.dtype)

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        # If out is not none, xr will be the third element of the tuple.
        if out is not None:
            xl, xg, *xr = out
            if return_retarded:
                if len(xr) != 1:
                    raise ValueError("Invalid number of output matrices.")
                xr = xr[0]
        else:
            xl = a.__class__.zeros_like(a)
            xg = a.__class__.zeros_like(a)
            if return_retarded:
                xr = a.__class__.zeros_like(a)

        # Perform the selected solve by batches.
        for i in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[i]), int(batches_slices[i + 1]), 1)

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xl_ = xl.stack[stack_slice]
            xg_ = xg.stack[stack_slice]
            if return_retarded:
                xr_ = xr.stack[stack_slice]

            # Check if there are OBC blocks for the current layer.
            obc_r = obc_blocks.retarded[0]
            a_00 = (
                a_.blocks[0, 0]
                if obc_r is None
                else a_.blocks[0, 0] - obc_r[stack_slice]
            )
            obc_l = obc_blocks.lesser[0]
            sl_00 = (
                sigma_lesser_.blocks[0, 0]
                if obc_l is None
                else sigma_lesser_.blocks[0, 0] + obc_l[stack_slice]
            )
            obc_g = obc_blocks.greater[0]
            sg_00 = (
                sigma_greater_.blocks[0, 0]
                if obc_g is None
                else sigma_greater_.blocks[0, 0] + obc_g[stack_slice]
            )

            xr_00 = xp.linalg.inv(a_00)
            xr_00_dagger = xr_00.conj().swapaxes(-2, -1)
            xr_diag_blocks[0] = xr_00
            xl_diag_blocks[0] = xr_00 @ sl_00 @ xr_00_dagger
            xg_diag_blocks[0] = xr_00 @ sg_00 @ xr_00_dagger

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
                a_ji_xr_ii_sl_ij = a_ji_xr_ii @ sigma_lesser_.blocks[i, j]
                a_ji_xr_ii_sg_ij = a_ji_xr_ii @ sigma_greater_.blocks[i, j]

                xr_jj = xp.linalg.inv(a_jj - a_ji @ xr_ii @ a_.blocks[i, j])
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_diag_blocks[j] = xr_jj

                xl_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sl_jj
                        + a_ji @ xl_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sl_ij
                    )
                    @ xr_jj_dagger
                )

                xg_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sg_jj
                        + a_ji @ xg_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sg_ij
                    )
                    @ xr_jj_dagger
                )

            # We need to write the last diagonal blocks to the output.
            xl_.blocks[-1, -1] = xl_diag_blocks[-1]
            xg_.blocks[-1, -1] = xg_diag_blocks[-1]
            if return_retarded:
                xr_.blocks[-1, -1] = xr_diag_blocks[-1]

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
                xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
                a_ij_dagger = a_ij.conj().swapaxes(-2, -1)
                sigma_greater_ji = -sigma_greater_ij.conj().swapaxes(-2, -1)
                sigma_lesser_ji = -sigma_lesser_ij.conj().swapaxes(-2, -1)

                # Precompute the terms that are used multiple times.
                xr_jj_dagger_aij_dagger = xr_jj_dagger @ a_ij_dagger
                a_ji_dagger_xr_jj_dagger = a_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
                a_ij_dagger_xr_ii_dagger = a_ij_dagger @ xr_ii_dagger
                a_ij_xr_jj = a_ij @ xr_jj
                xr_ii_a_ij = xr_ii @ a_ij
                xr_jj_a_ji = xr_jj @ a_ji
                xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji
                xr_ii_a_ij_xl_jj = xr_ii_a_ij @ xl_jj
                xr_ii_a_ij_xg_jj = xr_ii_a_ij @ xg_jj

                temp_1_l = (
                    xr_ii
                    @ (
                        sigma_lesser_ij @ xr_jj_dagger_aij_dagger
                        + a_ij_xr_jj @ sigma_lesser_ji
                    )
                    @ xr_ii_dagger
                )

                temp_1_g = (
                    xr_ii
                    @ (
                        sigma_greater_ij @ xr_jj_dagger_aij_dagger
                        + a_ij_xr_jj @ sigma_greater_ji
                    )
                    @ xr_ii_dagger
                )

                temp_2_l = xr_ii_a_ij_xr_jj_a_ji @ xl_ii

                temp_2_g = xr_ii_a_ij_xr_jj_a_ji @ xg_ii

                xl_.blocks[i, j] = (
                    -xr_ii_a_ij_xl_jj
                    - xl_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
                )

                xg_.blocks[i, j] = (
                    -xr_ii_a_ij_xg_jj
                    - xg_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_greater_ij @ xr_jj_dagger
                )

                xl_.blocks[j, i] = (
                    -xl_jj @ a_ij_dagger_xr_ii_dagger
                    - xr_jj_a_ji @ xl_ii
                    + xr_jj @ sigma_lesser_ji @ xr_ii_dagger
                )

                xg_.blocks[j, i] = (
                    -xg_jj @ a_ij_dagger_xr_ii_dagger
                    - xr_jj_a_ji @ xg_ii
                    + xr_jj @ sigma_greater_ji @ xr_ii_dagger
                )

                if return_current:
                    a_ji_xr_ii_sl_ij = a_ji @ xr_ii @ sigma_lesser_ij
                    a_ji_xr_ii_sg_ij = a_ji @ xr_ii @ sigma_greater_ij
                    sigma_lesser_tilde = (
                        a_ji @ xl_ii @ a_ji_dagger
                        + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sl_ij
                    )
                    sigma_greater_tilde = (
                        a_ji @ xg_ii @ a_ji_dagger
                        + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sg_ij
                    )
                    current[stack_slice, i] = xp.trace(
                        sigma_greater_tilde @ xl_diag_blocks[j]
                        - xg_diag_blocks[j] @ sigma_lesser_tilde,
                        axis1=-2,
                        axis2=-1,
                    )

                # NOTE: Cursed Python multiple assignment syntax.
                xl_.blocks[i, i] = xl_diag_blocks[i] = (
                    xl_ii
                    + xr_ii_a_ij_xl_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
                )
                xg_.blocks[i, i] = xg_diag_blocks[i] = (
                    xg_ii
                    + xr_ii_a_ij_xg_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_g
                    + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
                )
                xr_diag_blocks[i] = xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
                if return_retarded:
                    xr_.blocks[i, i] = xr_diag_blocks[i]

        if out is None:
            if return_retarded:
                if return_current:
                    return xl, xg, xr, current
                return xl, xg, xr
            return xl, xg

        if return_current:
            return current
