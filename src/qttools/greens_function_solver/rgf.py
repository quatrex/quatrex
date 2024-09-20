# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.solvers_utils import get_batches
from qttools.utils.gpu_utils import xp


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

        x.blocks[0, 0] = xp.linalg.inv(a.blocks[0, 0])

        # Forwards sweep.
        for i in range(a.num_blocks - 1):
            j = i + 1
            x.blocks[j, j] = xp.linalg.inv(
                a.blocks[j, j] - a.blocks[j, i] @ x.blocks[i, i] @ a.blocks[i, j]
            )

        # Backwards sweep.
        for i in range(a.num_blocks - 2, -1, -1):
            j = i + 1

            x_ii = x.blocks[i, i]
            x_jj = x.blocks[j, j]
            a_ij = a.blocks[i, j]

            x_ji = -x_jj @ a.blocks[j, i] @ x_ii
            x.blocks[j, i] = x_ji
            x.blocks[i, j] = -x_ii @ a_ij @ x_jj

            x.blocks[i, i] = x_ii - x_ii @ a_ij @ x_ji

        # for i in range(len(batches_sizes)):
        #     stack_slice = slice(batches_slices[i], batches_slices[i + 1], 1)

        #     x.blocks[0, 0][stack_slice] = xp.linalg.inv(a.blocks[0, 0][stack_slice])

        #     # Forwards sweep.
        #     for i in range(a.num_blocks - 1):
        #         j = i + 1
        #         x.blocks[j, j][stack_slice] = xp.linalg.inv(
        #             a.blocks[j, j][stack_slice]
        #             - a.blocks[j, i][stack_slice]
        #             @ x.blocks[i, i][stack_slice]
        #             @ a.blocks[i, j][stack_slice]
        #         )

        #     # Backwards sweep.
        #     for i in range(a.num_blocks - 2, -1, -1):
        #         j = i + 1

        #         x_ii = x.blocks[i, i][stack_slice]
        #         x_jj = x.blocks[j, j][stack_slice]
        #         a_ij = a.blocks[i, j][stack_slice]

        #         x_ji = -x_jj @ a.blocks[j, i][stack_slice] @ x_ii
        #         x.blocks[j, i][stack_slice] = x_ji
        #         x.blocks[i, j][stack_slice] = -x_ii @ a_ij @ x_jj

        #         x.blocks[i, i][stack_slice] = x_ii - x_ii @ a_ij @ x_ji

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

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        sigma_lesser : DBSparse
            Lesser matrix.
        sigma_greater : DBSparse
            Greater matrix.
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

        # If out is not none, x_r will be the last element of the tuple.
        if out is not None:
            x_l, x_g, *x_r = out
        else:
            x_r = a.__class__.zeros_like(a)
            x_l = a.__class__.zeros_like(a)
            x_g = a.__class__.zeros_like(a)

        for i in range(len(batches_sizes)):
            stack_slice = slice(batches_slices[i], batches_slices[i + 1], 1)

            x_r.blocks[0, 0][stack_slice] = xp.linalg.inv(a.blocks[0, 0][stack_slice])
            x_l.blocks[0, 0][stack_slice] = (
                x_r.blocks[0, 0][stack_slice]
                @ sigma_lesser.blocks[0, 0][stack_slice]
                @ x_r.blocks[0, 0][stack_slice].conj().T
            )
            x_g.blocks[0, 0][stack_slice] = (
                x_r.blocks[0, 0][stack_slice]
                @ sigma_greater.blocks[0, 0][stack_slice]
                @ x_r.blocks[0, 0][stack_slice].conj().T
            )

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                x_r.blocks[j, j][stack_slice] = xp.linalg.inv(
                    a.blocks[j, j][stack_slice]
                    - a.blocks[j, i][stack_slice]
                    @ x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                )

                x_l.blocks[j, j][stack_slice] = (
                    x_r.blocks[j, j][stack_slice]
                    @ (
                        sigma_lesser.blocks[j, j][stack_slice]
                        + a.blocks[j, i][stack_slice]
                        @ x_l.blocks[i, i][stack_slice]
                        @ a.blocks[j, i][stack_slice].conj().T
                        - sigma_lesser.blocks[j, i][stack_slice]
                        @ x_r.blocks[i, i][stack_slice].conj().T
                        @ a.blocks[j, i][stack_slice].conj().T
                        - a.blocks[j, i][stack_slice]
                        @ x_r.blocks[i, i][stack_slice]
                        @ sigma_lesser.blocks[i, j][stack_slice]
                    )
                    @ x_r.blocks[j, j][stack_slice].conj().T
                )
                x_g.blocks[j, j][stack_slice] = (
                    x_r.blocks[j, j][stack_slice]
                    @ (
                        sigma_greater.blocks[j, j][stack_slice]
                        + a.blocks[j, i][stack_slice]
                        @ x_g.blocks[i, i][stack_slice]
                        @ a.blocks[j, i][stack_slice].conj().T
                        - sigma_greater.blocks[j, i][stack_slice]
                        @ x_r.blocks[i, i][stack_slice].conj().T
                        @ a.blocks[j, i][stack_slice].conj().T
                        - a.blocks[j, i][stack_slice]
                        @ x_r.blocks[i, i][stack_slice]
                        @ sigma_greater.blocks[i, j][stack_slice]
                    )
                    @ x_r.blocks[j, j][stack_slice].conj().T
                )

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                temp_1_l = (
                    x_r.blocks[i, i][stack_slice]
                    @ (
                        sigma_lesser.blocks[i, j][stack_slice]
                        @ x_r.blocks[j, j][stack_slice].conj().T
                        @ a.blocks[i, j][stack_slice].conj().T
                        + a.blocks[i, j][stack_slice]
                        @ x_r.blocks[j, j][stack_slice]
                        @ sigma_lesser.blocks[j, i][stack_slice]
                    )
                    @ x_r.blocks[i, i][stack_slice].conj().T
                )
                temp_1_g = (
                    x_r.blocks[i, i][stack_slice]
                    @ (
                        sigma_greater.blocks[i, j][stack_slice]
                        @ x_r.blocks[j, j][stack_slice].conj().T
                        @ a.blocks[i, j][stack_slice].conj().T
                        + a.blocks[i, j][stack_slice]
                        @ x_r.blocks[j, j][stack_slice]
                        @ sigma_greater.blocks[j, i][stack_slice]
                    )
                    @ x_r.blocks[i, i][stack_slice].conj().T
                )
                temp_2_l = (
                    x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_r.blocks[j, j][stack_slice]
                    @ a.blocks[j, i][stack_slice]
                    @ x_l.blocks[i, i][stack_slice]
                )
                temp_2_g = (
                    x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_r.blocks[j, j][stack_slice]
                    @ a.blocks[j, i][stack_slice]
                    @ x_g.blocks[i, i][stack_slice]
                )

                x_l.blocks[i, j][stack_slice] = (
                    -x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_l.blocks[j, j][stack_slice]
                    - x_l.blocks[i, i][stack_slice]
                    @ a.blocks[j, i][stack_slice].conj().T
                    @ x_r.blocks[j, j][stack_slice].conj().T
                    + x_r.blocks[i, i][stack_slice]
                    @ sigma_lesser.blocks[i, j][stack_slice]
                    @ x_r.blocks[j, j][stack_slice].conj().T
                )

                x_l.blocks[j, i][stack_slice] = (
                    -x_l.blocks[j, j][stack_slice]
                    @ a.blocks[i, j][stack_slice].conj().T
                    @ x_r.blocks[i, i][stack_slice].conj().T
                    - x_r.blocks[j, j][stack_slice]
                    @ a.blocks[j, i][stack_slice]
                    @ x_l.blocks[i, i][stack_slice]
                    + x_r.blocks[j, j][stack_slice]
                    @ sigma_lesser.blocks[j, i][stack_slice]
                    @ x_r.blocks[i, i][stack_slice].conj().T
                )

                x_g.blocks[i, j][stack_slice] = (
                    -x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_g.blocks[j, j][stack_slice]
                    - x_g.blocks[i, i][stack_slice]
                    @ a.blocks[j, i][stack_slice].conj().T
                    @ x_r.blocks[j, j][stack_slice].conj().T
                    + x_r.blocks[i, i][stack_slice]
                    @ sigma_greater.blocks[i, j][stack_slice]
                    @ x_r.blocks[j, j][stack_slice].conj().T
                )

                x_g.blocks[j, i][stack_slice] = (
                    -x_g.blocks[j, j][stack_slice]
                    @ a.blocks[i, j][stack_slice].conj().T
                    @ x_r.blocks[i, i][stack_slice].conj().T
                    - x_r.blocks[j, j][stack_slice]
                    @ a.blocks[j, i][stack_slice]
                    @ x_g.blocks[i, i][stack_slice]
                    + x_r.blocks[j, j][stack_slice]
                    @ sigma_greater.blocks[j, i][stack_slice]
                    @ x_r.blocks[i, i][stack_slice].conj().T
                )

                x_l.blocks[i, i][stack_slice] = (
                    x_l.blocks[i, i][stack_slice]
                    + x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_l.blocks[j, j][stack_slice]
                    @ a.blocks[i, j][stack_slice].conj().T
                    @ x_r.blocks[i, i][stack_slice].conj().T
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().T)
                )
                x_g.blocks[i, i][stack_slice] = (
                    x_g.blocks[i, i][stack_slice]
                    + x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_g.blocks[j, j][stack_slice]
                    @ a.blocks[i, j][stack_slice].conj().T
                    @ x_r.blocks[i, i][stack_slice].conj().T
                    - temp_1_g
                    + (temp_2_g - temp_2_g.conj().T)
                )
                x_r.blocks[i, i][stack_slice] = (
                    x_r.blocks[i, i][stack_slice]
                    + x_r.blocks[i, i][stack_slice]
                    @ a.blocks[i, j][stack_slice]
                    @ x_r.blocks[j, j][stack_slice]
                    @ a.blocks[j, i][stack_slice]
                    @ x_r.blocks[i, i][stack_slice]
                )

        return x_l, x_g
