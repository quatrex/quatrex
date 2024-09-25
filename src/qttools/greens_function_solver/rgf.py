# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.gpu_utils import xp
from qttools.utils.solvers_utils import get_batches


class RGF(GFSolver):
    def selected_inv(
        self, a: DSBSparse, out=None, max_batch_size: int = 1
    ) -> None | DSBSparse:
        """
        Perform the selected inversion of a matrix in block-tridiagonal form.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        out : _type_, optional
            Output matrix, by default None.
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1.

        Returns
        -------
        None | DBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object.
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
        a : DBSparse
            Matrix to invert.
        sigma_lesser : DBSparse
            Lesser matrix. This matrix is expected to be skewed-hermitian.
        sigma_greater : DBSparse
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
            as a DBSparse object. If `return_retarded` is True, returns a tuple with
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

            xr_.blocks[0, 0] = xp.linalg.inv(a_.blocks[0, 0])
            xl_.blocks[0, 0] = (
                xr_.blocks[0, 0]
                @ sigma_lesser_.blocks[0, 0]
                @ xr_.blocks[0, 0].conj().swapaxes(-2, -1)
            )
            xg_.blocks[0, 0] = (
                xr_.blocks[0, 0]
                @ sigma_greater_.blocks[0, 0]
                @ xr_.blocks[0, 0].conj().swapaxes(-2, -1)
            )

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                xr_.blocks[j, j] = xp.linalg.inv(
                    a_.blocks[j, j]
                    - a_.blocks[j, i] @ xr_.blocks[i, i] @ a_.blocks[i, j]
                )

                xl_.blocks[j, j] = (
                    xr_.blocks[j, j]
                    @ (
                        sigma_lesser_.blocks[j, j]
                        + a_.blocks[j, i]
                        @ xl_.blocks[i, i]
                        @ a_.blocks[j, i].conj().swapaxes(-2, -1)
                        - sigma_lesser_.blocks[j, i]
                        @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                        @ a_.blocks[j, i].conj().swapaxes(-2, -1)
                        - a_.blocks[j, i]
                        @ xr_.blocks[i, i]
                        @ sigma_lesser_.blocks[i, j]
                    )
                    @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                )

                xg_.blocks[j, j] = (
                    xr_.blocks[j, j]
                    @ (
                        sigma_greater_.blocks[j, j]
                        + a_.blocks[j, i]
                        @ xg_.blocks[i, i]
                        @ a_.blocks[j, i].conj().swapaxes(-2, -1)
                        - sigma_greater_.blocks[j, i]
                        @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                        @ a_.blocks[j, i].conj().swapaxes(-2, -1)
                        - a_.blocks[j, i]
                        @ xr_.blocks[i, i]
                        @ sigma_greater_.blocks[i, j]
                    )
                    @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                )

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                temp_1_l = (
                    xr_.blocks[i, i]
                    @ (
                        sigma_lesser_.blocks[i, j]
                        @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                        @ a_.blocks[i, j].conj().swapaxes(-2, -1)
                        + a_.blocks[i, j]
                        @ xr_.blocks[j, j]
                        @ sigma_lesser_.blocks[j, i]
                    )
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                )

                temp_1_g = (
                    xr_.blocks[i, i]
                    @ (
                        sigma_greater_.blocks[i, j]
                        @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                        @ a_.blocks[i, j].conj().swapaxes(-2, -1)
                        + a_.blocks[i, j]
                        @ xr_.blocks[j, j]
                        @ sigma_greater_.blocks[j, i]
                    )
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                )

                temp_2_l = (
                    xr_.blocks[i, i]
                    @ a_.blocks[i, j]
                    @ xr_.blocks[j, j]
                    @ a_.blocks[j, i]
                    @ xl_.blocks[i, i]
                )

                temp_2_g = (
                    xr_.blocks[i, i]
                    @ a_.blocks[i, j]
                    @ xr_.blocks[j, j]
                    @ a_.blocks[j, i]
                    @ xg_.blocks[i, i]
                )

                xl_.blocks[i, j] = (
                    -xr_.blocks[i, i] @ a_.blocks[i, j] @ xl_.blocks[j, j]
                    - xl_.blocks[i, i]
                    @ a_.blocks[j, i].conj().swapaxes(-2, -1)
                    @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                    + xr_.blocks[i, i]
                    @ sigma_lesser_.blocks[i, j]
                    @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                )

                xl_.blocks[j, i] = (
                    -xl_.blocks[j, j]
                    @ a_.blocks[i, j].conj().swapaxes(-2, -1)
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                    - xr_.blocks[j, j] @ a_.blocks[j, i] @ xl_.blocks[i, i]
                    + xr_.blocks[j, j]
                    @ sigma_lesser_.blocks[j, i]
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                )

                xg_.blocks[i, j] = (
                    -xr_.blocks[i, i] @ a_.blocks[i, j] @ xg_.blocks[j, j]
                    - xg_.blocks[i, i]
                    @ a_.blocks[j, i].conj().swapaxes(-2, -1)
                    @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                    + xr_.blocks[i, i]
                    @ sigma_greater_.blocks[i, j]
                    @ xr_.blocks[j, j].conj().swapaxes(-2, -1)
                )

                xg_.blocks[j, i] = (
                    -xg_.blocks[j, j]
                    @ a_.blocks[i, j].conj().swapaxes(-2, -1)
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                    - xr_.blocks[j, j] @ a_.blocks[j, i] @ xg_.blocks[i, i]
                    + xr_.blocks[j, j]
                    @ sigma_greater_.blocks[j, i]
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                )

                xl_.blocks[i, i] = (
                    xl_.blocks[i, i]
                    + xr_.blocks[i, i]
                    @ a_.blocks[i, j]
                    @ xl_.blocks[j, j]
                    @ a_.blocks[i, j].conj().swapaxes(-2, -1)
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
                )
                xg_.blocks[i, i] = (
                    xg_.blocks[i, i]
                    + xr_.blocks[i, i]
                    @ a_.blocks[i, j]
                    @ xg_.blocks[j, j]
                    @ a_.blocks[i, j].conj().swapaxes(-2, -1)
                    @ xr_.blocks[i, i].conj().swapaxes(-2, -1)
                    - temp_1_g
                    + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
                )
                xr_.blocks[i, i] = (
                    xr_.blocks[i, i]
                    + xr_.blocks[i, i]
                    @ a_.blocks[i, j]
                    @ xr_.blocks[j, j]
                    @ a_.blocks[j, i]
                    @ xr_.blocks[i, i]
                )

        if out is None:
            if return_retarded:
                return xl, xg, xr
            return xl, xg
