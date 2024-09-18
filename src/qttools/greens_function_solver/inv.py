# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np

from qttools.datastructures.dsbsparse import DSBSparse
from qttools.greens_function_solver.solver import GFSolver
from qttools.utils.mpi_utils import get_section_sizes


class Inv(GFSolver):
    def selected_inv(
        self, a: DSBSparse, out: DSBSparse = None, max_batch_size: int = 1
    ) -> None | DSBSparse:
        """Perform the selected inversion of a matrix in block-tridiagonal form using
        batching through the first dimmension matrix stack.

        This method will invert the matrix as dense and then select the elements
        to keep by matching the sparse structure of the input matrix.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        out : DSBSparse, optional
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
        batches_sizes, _ = get_section_sizes(
            num_elements=a.shape[0],
            num_sections=a.shape[0] // min(max_batch_size, a.shape[0]),
        )
        batches_slices = np.cumsum([0] + batches_sizes)

        # Allocate batching buffer
        inv_a = np.zeros((max(batches_sizes), *a.shape[1:]), dtype=a.dtype)

        # Prepare output
        return_out = False
        if out is None:
            rows, cols = a.spy()
            out = a.__class__.zeros_like(a)
            return_out = True
        else:
            rows, cols = out.spy()

        # Perform the inversion in batches
        for i in range(len(batches_sizes)):
            stack_slice = slice(batches_slices[i], batches_slices[i + 1], 1)

            inv_a[: batches_sizes[i]] = np.linalg.inv(
                a.to_dense(stack_slice=stack_slice)
            )

            out.data[stack_slice,] = inv_a[: batches_sizes[i], rows, cols]

        if return_out:
            return out

    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple[DSBSparse, ...] | None = None,
        return_retarded: bool = False,
    ) -> None | tuple:
        """Solve the congruence matrix equation: A * X * A^T = B.

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
