# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver


class Inv(GFSolver):
    def selected_inv(self, a: DSBSparse, out: DSBSparse = None) -> None | DSBSparse:
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

        inv_a = np.linalg.inv(a.to_dense())

        if out is None:
            rows, cols = a.spy()
            sel_inv_a = a.__class__.zeros_like(a)
            sel_inv_a.data[:] = inv_a[..., rows, cols]
            return sel_inv_a

        rows, cols = out.spy()
        out.data[:] = inv_a[..., rows, cols]

    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple[DSBSparse, ...] | None = None,
        return_retarded: bool = False,
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
        x_r = np.linalg.inv(a.to_dense())

        x_l = x_r @ sigma_lesser.to_dense() @ x_r.conj().transpose((0, 2, 1))
        x_g = x_r @ sigma_greater.to_dense() @ x_r.conj().transpose((0, 2, 1))

        if out is None:
            rows, cols = a.spy()
            sel_x_l = a.__class__.zeros_like(a)
            sel_x_g = a.__class__.zeros_like(a)
            sel_x_l.data[:] = x_l[..., rows, cols]
            sel_x_g.data[:] = x_g[..., rows, cols]

            if not return_retarded:
                return sel_x_l, sel_x_g

            sel_x_r = a.__class__.zeros_like(a)
            sel_x_r.data[:] = x_r[..., rows, cols]

            return sel_x_l, sel_x_g, sel_x_r

        x_l_out, x_g_out, *x_r_out = out

        rows_l, cols_l = x_l_out.spy()
        rows_g, cols_g = x_g_out.spy()

        x_l_out.data[:] = x_l[..., rows_l, cols_l]
        x_g_out.data[:] = x_g[..., rows_g, cols_g]

        if return_retarded:
            if len(x_r_out) == 0:
                raise ValueError("Missing output for the retarded Green's function.")
            x_r_out = x_r_out[0]

            rows_r, cols_r = x_r_out.spy()
            x_r_out.data[:] = x_r[..., rows_r, cols_r]
