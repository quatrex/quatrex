import numpy as np

from qttools.datastructures.dbsparse import DBSparse
from qttools.greens_function_solver.solver import GFSolver


class Inv(GFSolver):
    def selected_inv(a: DBSparse, out=None) -> None | DBSparse:
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

        if out is not None:
            out[:] = inv_a
            return ...
        else:
            return inv_a

    def selected_solve(
        a: DBSparse,
        sigma_lesser: DBSparse,
        sigma_greater: DBSparse,
        out: tuple | None = None,
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
        x = Inv.selected_inv(a)

        x_l = x @ sigma_lesser.to_dense() @ x.conj().transpose((0, 2, 1))
        x_g = x @ sigma_greater.to_dense() @ x.conj().transpose((0, 2, 1))

        x_lesser = DBSparse.zeros_like(a)
        x_greater = DBSparse.zeros_like(a)
        for i in range(sigma_lesser.num_blocks):
            _i = slice(*sigma_lesser.block_offsets[i : i + 2])
            x_lesser.set_block(i, i, x_l[..., _i, _i])
            x_greater.set_block(i, i, x_g[..., _i, _i])

        for i in range(sigma_lesser.num_blocks - 1):
            _i = slice(*sigma_lesser.block_offsets[i : i + 2])
            _j = slice(*sigma_lesser.block_offsets[i + 1 : i + 2 + 1])
            x_lesser.set_block(i, i + 1, x_l[..., _i, _j])
            x_lesser.set_block(i + 1, i, x_l[..., _j, _i])
            x_greater.set_block(i, i + 1, x_g[..., _i, _j])
            x_greater.set_block(i + 1, i, x_g[..., _j, _i])

        if not return_retarded:
            return x_lesser, x_greater

        x_retarded = DBSparse.zeros_like(a)
        for i in range(sigma_lesser.num_blocks):
            _i = slice(*sigma_lesser.block_offsets[i : i + 2])
            x_retarded.set_block(i, i, x[..., _i, _i])

        for i in range(sigma_lesser.num_blocks - 1):
            _i = slice(*sigma_lesser.block_offsets[i : i + 2])
            _j = slice(*sigma_lesser.block_offsets[i + 1 : i + 2 + 1])
            x_retarded.set_block(i, i + 1, x[..., _i, _j])
            x_retarded.set_block(i + 1, i, x[..., _j, _i])

        return x_lesser, x_greater, x_retarded
