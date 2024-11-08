# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from qttools import xp
from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.solvers_utils import get_batches


class RGF(GFSolver):
    def selected_inv(
        self, a: DSBSparse, out=None, max_batch_size: int = 1
    ) -> None | DSBSparse:
        """
        Perform the selected inversion of a matrix in block-tridiagonal form.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        out : _type_, optional
            Output matrix, by default None.
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1.

        Returns
        -------
        None | DSBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DSBSparse object.
        """
        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], max_batch_size)

        if out is not None:
            x = out
        else:
            x = a.__class__.zeros_like(a)

        for b in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[b]), int(batches_slices[b + 1]), 1)

            a_ = a.stack[stack_slice]
            x_ = x.stack[stack_slice]

            x_.blocks[0, 0] = xp.linalg.inv(a_.blocks[0, 0])

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1
                x_.blocks[j, j] = xp.linalg.inv(
                    a_.blocks[j, j]
                    - a_.blocks[j, i] @ x_.blocks[i, i] @ a_.blocks[i, j]
                )

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                x_ii = x_.blocks[i, i]
                x_jj = x_.blocks[j, j]
                a_ij = a_.blocks[i, j]

                x_ji = -x_jj @ a_.blocks[j, i] @ x_ii
                x_.blocks[j, i] = x_ji
                x_.blocks[i, j] = -x_ii @ a_ij @ x_jj

                x_.blocks[i, i] = x_ii - x_ii @ a_ij @ x_ji

        if out is None:
            return x

    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        max_batch_size: int = 1,
    ) -> None | tuple:
        """Perform a selected-solve of the congruence matrix equation: A * X * A^T = B.

        Note
        ----
        If the diagonal blocks of the input matrix ```a: DSBSparse``` are not dense,
        the selected-solve will not be performed correctly. This happen because during
        the forward sweep, only the elements of the inverse that match the sparsity
        pattern of the input diagonal blocks will be stored. Leading to incomplete
        matrix-multiplications down the line.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        sigma_lesser : DSBSparse
            Lesser matrix. This matrix is expected to be skewed-hermitian.
        sigma_greater : DSBSparse
            Greater matrix. This matrix is expected to be skewed-hermitian.
        out : tuple | None, optional
            Output matrix, by default None
        return_retarded : bool, optional
            Weither the retarded Green's functioln should be returned, by default False
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1


        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DSBSparse object. If `return_retarded` is True, returns a tuple with
            the retarded Green's function as the last element.
        """
        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], max_batch_size)

        # If out is not none, xr will be the last element of the tuple.
        if out is not None:
            xl, xg, *xr = out
            if len(xr) == 0:
                xr = a.__class__.zeros_like(a)
            else:
                xr = xr[0]
        else:
            xr = a.__class__.zeros_like(a)
            xl = a.__class__.zeros_like(a)
            xg = a.__class__.zeros_like(a)

        # Perform the selected solve by batches.
        for i in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[i]), int(batches_slices[i + 1]), 1)

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xr_ = xr.stack[stack_slice]
            xl_ = xl.stack[stack_slice]
            xg_ = xg.stack[stack_slice]

            xr_00 = xp.linalg.inv(a_.blocks[0, 0])
            xr_.blocks[0, 0] = xr_00
            xl_.blocks[0, 0] = (
                xr_00 @ sigma_lesser_.blocks[0, 0] @ xr_00.conj().swapaxes(-2, -1)
            )
            xg_.blocks[0, 0] = (
                xr_00 @ sigma_greater_.blocks[0, 0] @ xr_00.conj().swapaxes(-2, -1)
            )

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # Densify the blocks that are used multiple times.
                a_ji = a_.blocks[j, i]
                xr_ii = xr_.blocks[i, i]

                xr_jj = xp.linalg.inv(a_.blocks[j, j] - a_ji @ xr_ii @ a_.blocks[i, j])
                xr_.blocks[j, j] = xr_jj

                # Precompute some terms that are used multiple times.
                xr_ii_dagger_aji_dagger = xr_ii.conj().swapaxes(
                    -2, -1
                ) @ a_ji.conj().swapaxes(-2, -1)
                a_ji_xr_ii = a_ji @ xr_ii

                xl_.blocks[j, j] = (
                    xr_jj
                    @ (
                        sigma_lesser_.blocks[j, j]
                        + a_ji @ xl_.blocks[i, i] @ a_ji.conj().swapaxes(-2, -1)
                        - sigma_lesser_.blocks[j, i] @ xr_ii_dagger_aji_dagger
                        - a_ji_xr_ii @ sigma_lesser_.blocks[i, j]
                    )
                    @ xr_jj.conj().swapaxes(-2, -1)
                )

                xg_.blocks[j, j] = (
                    xr_jj
                    @ (
                        sigma_greater_.blocks[j, j]
                        + a_ji @ xg_.blocks[i, i] @ a_ji.conj().swapaxes(-2, -1)
                        - sigma_greater_.blocks[j, i] @ xr_ii_dagger_aji_dagger
                        - a_ji_xr_ii @ sigma_greater_.blocks[i, j]
                    )
                    @ xr_jj.conj().swapaxes(-2, -1)
                )

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                # Densify the blocks that are used multiple times.
                xr_ii = xr_.blocks[i, i]
                xr_jj = xr_.blocks[j, j]
                a_ij = a_.blocks[i, j]
                a_ji = a_.blocks[j, i]
                xl_ii = xl_.blocks[i, i]
                xl_jj = xl_.blocks[j, j]
                xg_ii = xg_.blocks[i, i]
                xg_jj = xg_.blocks[j, j]
                sigma_lesser_ij = sigma_lesser_.blocks[i, j]
                sigma_lesser_ji = sigma_lesser_.blocks[j, i]
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

                xl_.blocks[i, i] = (
                    xl_ii
                    + xr_ii_a_ij_xl_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
                )
                xg_.blocks[i, i] = (
                    xg_ii
                    + xr_ii_a_ij_xg_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_g
                    + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
                )

                xr_.blocks[i, i] = xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii

        if out is None:
            if return_retarded:
                return xl, xg, xr
            return xl, xg
