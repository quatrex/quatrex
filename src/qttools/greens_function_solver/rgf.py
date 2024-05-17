# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy.linalg as npla

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver


class RGF(GFSolver):
    def selected_inv(a: DSBSparse, out=None) -> None | DSBSparse:
        """
        Perform the selected inversion of a matrix in block-tridiagonal form.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        out : _type_, optional
            Output matrix, by default None.

        Returns
        -------
        None | DBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object.
        """
        if out is not None:
            x = out
        else:
            x = DSBSparse.zeros_like(a)

        x[0, 0] = npla.inv(a[0, 0])

        # Forwards sweep.
        for i in range(a.bshape[0] - 1):
            j = i + 1
            x[j, j] = npla.inv(a[j, j] - a[j, i] @ x[i, i] @ a[i, j])

        # Backwards sweep.
        for i in range(a.bshape[0] - 2, -1, -1):
            j = i + 1

            x_ii = x[i, i]
            x_jj = x[j, j]
            a_ij = a[i, j]

            x_ji = -x_jj @ a[j, i] @ x_ii
            x[j, i] = x_ji
            x[i, j] = -x_ii @ a_ij @ x_jj

            x[i, i] = x_ii - x_ii @ a_ij @ x_ji

        return x

    def selected_solve(
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        **kwargs,
    ) -> None | tuple:
        """Solve the selected quadratic matrix equation and compute only selected
        elements of it's inverse.

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

        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object. If `return_retarded` is True, returns a tuple with
            the retarded Green's function as the second element.
        """

        # If out is not none, x_r will bhe the first element of the tuple. and so on for x_l and x_g
        if out is not None:
            x_r = out[0]
            x_l = out[1]
            x_g = out[2]
        else:
            x_r = DSBSparse.zeros_like(a)
            x_l = DSBSparse.zeros_like(a)
            x_g = DSBSparse.zeros_like(a)

        x_r[0, 0] = npla.inv(a[0, 0])
        x_l[0, 0] = x_r[0, 0] @ sigma_lesser[0, 0] @ x_r[0, 0].conj().T
        x_g[0, 0] = x_r[0, 0] @ sigma_greater[0, 0] @ x_r[0, 0].conj().T

        # Forwards sweep.
        for i in range(a.bshape[0] - 1):
            j = i + 1

            x_r[j, j] = npla.inv(a[j, j] - a[j, i] @ x_r[i, i] @ a[i, j])

            x_l[j, j] = (
                x_r[j, j]
                @ (
                    sigma_lesser[j, j]
                    + a[j, i] @ x_l[i, i] @ a[j, i].conj().T
                    - sigma_lesser[j, i] @ x_r[i, i].conj().T @ a[j, i].conj().T
                    - a[j, i] @ x_r[i, i] @ sigma_lesser[i, j]
                )
                @ x_r[j, j].conj().T
            )
            x_g[j, j] = (
                x_r[j, j]
                @ (
                    sigma_greater[j, j]
                    + a[j, i] @ x_g[i, i] @ a[j, i].conj().T
                    - sigma_greater[j, i] @ x_r[i, i].conj().T @ a[j, i].conj().T
                    - a[j, i] @ x_r[i, i] @ sigma_greater[i, j]
                )
                @ x_r[j, j].conj().T
            )

        # Backwards sweep.
        for i in range(a.bshape[0] - 2, -1, -1):
            j = i + 1

            temp_1_l = (
                x_r[i, i]
                @ (
                    sigma_lesser[i, j] @ x_r[j, j].conj().T @ a[i, j].conj().T
                    + a[i, j] @ x_r[j, j] @ sigma_lesser[j, i]
                )
                @ x_r[i, i].conj().T
            )
            temp_1_g = (
                x_r[i, i]
                @ (
                    sigma_greater[i, j] @ x_r[j, j].conj().T @ a[i, j].conj().T
                    + a[i, j] @ x_r[j, j] @ sigma_greater[j, i]
                )
                @ x_r[i, i].conj().T
            )
            temp_2_l = x_r[i, i] @ a[i, j] @ x_r[j, j] @ a[j, i] @ x_l[i, i]
            temp_2_g = x_r[i, i] @ a[i, j] @ x_r[j, j] @ a[j, i] @ x_g[i, i]

            x_l[i, j] = (
                -x_r[i, i] @ a[i, j] @ x_l[j, j]
                - x_l[i, i] @ a[j, i].conj().T @ x_r[j, j].conj().T
                + x_r[i, i] @ sigma_lesser[i, j] @ x_r[j, j].conj().T
            )

            x_l[j, i] = (
                -x_l[j, j] @ a[i, j].conj().T @ x_r[i, i].conj().T
                - x_r[j, j] @ a[j, i] @ x_l[i, i]
                + x_r[j, j] @ sigma_lesser[j, i] @ x_r[i, i].conj().T
            )

            x_g[i, j] = (
                -x_r[i, i] @ a[i, j] @ x_g[j, j]
                - x_g[i, i] @ a[j, i].conj().T @ x_r[j, j].conj().T
                + x_r[i, i] @ sigma_greater[i, j] @ x_r[j, j].conj().T
            )

            x_g[j, i] = (
                -x_g[j, j] @ a[i, j].conj().T @ x_r[i, i].conj().T
                - x_r[j, j] @ a[j, i] @ x_g[i, i]
                + x_r[j, j] @ sigma_greater[j, i] @ x_r[i, i].conj().T
            )

            x_l[i, i] = (
                x_l[i, i]
                + x_r[i, i]
                @ a[i, j]
                @ x_l[j, j]
                @ a[i, j].conj().T
                @ x_r[i, i].conj().T
                - temp_1_l
                + (temp_2_l - temp_2_l.conj().T)
            )
            x_g[i, i] = (
                x_g[i, i]
                + x_r[i, i]
                @ a[i, j]
                @ x_g[j, j]
                @ a[i, j].conj().T
                @ x_r[i, i].conj().T
                - temp_1_g
                + (temp_2_g - temp_2_g.conj().T)
            )
            x_r[i, i] = (
                x_r[i, i] + x_r[i, i] @ a[i, j] @ x_r[j, j] @ a[j, i] @ x_r[i, i]
            )

        return x_l, x_g
