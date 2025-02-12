# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.solvers_utils import get_batches


class RGF(GFSolver):
    """Selected inversion solver based on the Schur complement.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size to use when inverting the matrix, by default
        1.

    """

    def __init__(self, max_batch_size: int = 1) -> None:
        """Initializes the selected inversion solver."""
        self.max_batch_size = max_batch_size

    def selected_inv(
        self, a: DSBSparse, out: DSBSparse | None = None
    ) -> None | DSBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        out : DSBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSBSparse object.

        """
        # Initialize dense temporary buffers for the diagonal blocks.
        x_diag_blocks = [None] * a.num_blocks

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

            x_diag_blocks[0] = xp.linalg.inv(a_.blocks[0, 0])

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                x_diag_blocks[j] = xp.linalg.inv(
                    a_.blocks[j, j]
                    - a_.blocks[j, i] @ x_diag_blocks[i] @ a_.blocks[i, j]
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
        sigma_greater: DSBSparse | None = None,
        out: tuple[DSBSparse, ...] | None = None,
        return_retarded: bool = False,
    ) -> None | tuple:
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
        sigma_greater : DSBSparse | None, optional
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSBSparse, ...] | None, optional
            Preallocated output matrices, by default None
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False

        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, the solutions are
            returned as DSBParse matrices. If `return_retarded` is True,
            returns a tuple with the retarded Green's function as the
            last element.

        """
        # Initialize dense temporary buffers for the diagonal blocks.
        xr_diag_blocks = [None] * a.num_blocks
        xl_diag_blocks = [None] * a.num_blocks
        if sigma_greater is not None:
            xg_diag_blocks = [None] * a.num_blocks

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        # If out is not none, xr will be the last element of the tuple.
        if out is not None:
            if sigma_greater is not None:
                xl, xg, *xr = out
            else:
                xl, *xr = out
            if len(xr) == 0:
                xr = a.__class__.zeros_like(a)
            else:
                xr = xr[0]
        else:
            xr = a.__class__.zeros_like(a)
            xl = a.__class__.zeros_like(a)
            if sigma_greater is not None:
                xg = a.__class__.zeros_like(a)

        # Perform the selected solve by batches.
        for i in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[i]), int(batches_slices[i + 1]), 1)

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            if sigma_greater is not None:
                sigma_greater_ = sigma_greater.stack[stack_slice]

            xr_ = xr.stack[stack_slice]
            xl_ = xl.stack[stack_slice]
            if sigma_greater is not None:
                xg_ = xg.stack[stack_slice]

            xr_00 = xp.linalg.inv(a_.blocks[0, 0])
            xr_00_dagger = xr_00.conj().swapaxes(-2, -1)
            xr_diag_blocks[0] = xr_00
            xl_diag_blocks[0] = xr_00 @ sigma_lesser_.blocks[0, 0] @ xr_00_dagger
            if sigma_greater is not None:
                xg_diag_blocks[0] = xr_00 @ sigma_greater_.blocks[0, 0] @ xr_00_dagger

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # Get the blocks that are used multiple times.
                a_ji = a_.blocks[j, i]
                xr_ii = xr_diag_blocks[i]

                # Precompute the transposes that are used multiple times.
                a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

                # Precompute some terms that are used multiple times.
                xr_ii_dagger_aji_dagger = xr_ii.conj().swapaxes(-2, -1) @ a_ji_dagger
                a_ji_xr_ii = a_ji @ xr_ii

                xr_jj = xp.linalg.inv(a_.blocks[j, j] - a_ji @ xr_ii @ a_.blocks[i, j])
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_diag_blocks[j] = xr_jj

                xl_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sigma_lesser_.blocks[j, j]
                        + a_ji @ xl_diag_blocks[i] @ a_ji_dagger
                        - sigma_lesser_.blocks[j, i] @ xr_ii_dagger_aji_dagger
                        - a_ji_xr_ii @ sigma_lesser_.blocks[i, j]
                    )
                    @ xr_jj_dagger
                )

                if sigma_greater is not None:
                    xg_diag_blocks[j] = (
                        xr_jj
                        @ (
                            sigma_greater_.blocks[j, j]
                            + a_ji @ xg_diag_blocks[i] @ a_ji_dagger
                            - sigma_greater_.blocks[j, i] @ xr_ii_dagger_aji_dagger
                            - a_ji_xr_ii @ sigma_greater_.blocks[i, j]
                        )
                        @ xr_jj_dagger
                    )

            # We need to write the last diagonal blocks to the output.
            xr_.blocks[j, j] = xr_diag_blocks[j]
            xl_.blocks[j, j] = xl_diag_blocks[j]
            if sigma_greater is not None:
                xg_.blocks[j, j] = xg_diag_blocks[j]

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
                if sigma_greater is not None:
                    xg_ii = xg_diag_blocks[i]
                    xg_jj = xg_diag_blocks[j]
                sigma_lesser_ij = sigma_lesser_.blocks[i, j]
                sigma_lesser_ji = sigma_lesser_.blocks[j, i]
                if sigma_greater is not None:
                    sigma_greater_ij = sigma_greater_.blocks[i, j]
                    sigma_greater_ji = sigma_greater_.blocks[j, i]

                # Precompute the transposes that are used multiple times.
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
                a_ij_dagger = a_ij.conj().swapaxes(-2, -1)

                # Precompute the terms that are used multiple times.
                xr_jj_dagger_aij_dagger = xr_jj_dagger @ a_ij_dagger
                a_ji_dagger_xr_jj_dagger = a_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
                a_ij_dagger_xr_ii_dagger = a_ij_dagger @ xr_ii_dagger
                a_ij_xr_jj = a_ij @ xr_jj
                xr_ii_a_ij = xr_ii @ a_ij
                xr_jj_a_ji = xr_jj @ a_ji
                xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji
                xr_ii_a_ij_xl_jj = xr_ii_a_ij @ xl_jj
                if sigma_greater is not None:
                    xr_ii_a_ij_xg_jj = xr_ii_a_ij @ xg_jj

                temp_1_l = (
                    xr_ii
                    @ (
                        sigma_lesser_ij @ xr_jj_dagger_aij_dagger
                        + a_ij_xr_jj @ sigma_lesser_ji
                    )
                    @ xr_ii_dagger
                )

                if sigma_greater is not None:
                    temp_1_g = (
                        xr_ii
                        @ (
                            sigma_greater_ij @ xr_jj_dagger_aij_dagger
                            + a_ij_xr_jj @ sigma_greater_ji
                        )
                        @ xr_ii_dagger
                    )

                temp_2_l = xr_ii_a_ij_xr_jj_a_ji @ xl_ii

                if sigma_greater is not None:
                    temp_2_g = xr_ii_a_ij_xr_jj_a_ji @ xg_ii

                xl_.blocks[i, j] = (
                    -xr_ii_a_ij_xl_jj
                    - xl_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
                )

                if sigma_greater is not None:
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

                if sigma_greater is not None:
                    xg_.blocks[j, i] = (
                        -xg_jj @ a_ij_dagger_xr_ii_dagger
                        - xr_jj_a_ji @ xg_ii
                        + xr_jj @ sigma_greater_ji @ xr_ii_dagger
                    )

                # NOTE: Cursed Python multiple assignment syntax.
                xl_.blocks[i, i] = xl_diag_blocks[i] = (
                    xl_ii
                    + xr_ii_a_ij_xl_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
                )
                if sigma_greater is not None:
                    xg_.blocks[i, i] = xg_diag_blocks[i] = (
                        xg_ii
                        + xr_ii_a_ij_xg_jj @ a_ij_dagger_xr_ii_dagger
                        - temp_1_g
                        + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
                    )

                xr_.blocks[i, i] = xr_diag_blocks[i] = (
                    xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
                )

        if out is None:
            if return_retarded:
                if sigma_greater is not None:
                    return xl, xg, xr
                return xl, xr
            if sigma_greater is not None:
                return xl, xg
            return xl
