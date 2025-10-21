# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.datastructures.routines import complex_gemm_to_real_with_mask
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
        obc_blocks: OBCBlocks | None = None,
        out: DSDBSparse | None = None,
        mm_mask: str | None = None,
        inv_mask: str = "fp64",
    ) -> None | DSDBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : DSDBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSDBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSDBSparse object.

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

                temp_mult = complex_gemm_to_real_with_mask(
                    a_.blocks[j, i], x_diag_blocks[i], mask=self.mm_mask
                )
                temp_result = complex_gemm_to_real_with_mask(
                    temp_mult, a_.blocks[i, j], mask=self.mm_mask
                )
                x_diag_blocks[j] = inv(a_jj - temp_result)

            # We need to write the last diagonal block to the output.
            x_.blocks[a.num_blocks - 1, a.num_blocks - 1] = x_diag_blocks[-1]

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                x_ii = x_diag_blocks[i]
                x_jj = x_diag_blocks[j]
                a_ij = a_.blocks[i, j]

                temp_mult1 = complex_gemm_to_real_with_mask(
                    x_jj, a_.blocks[j, i], mask=self.mm_mask
                )
                x_ji = -complex_gemm_to_real_with_mask(
                    temp_mult1, x_ii, mask=self.mm_mask
                )
                x_.blocks[j, i] = x_ji
                temp_mult2 = complex_gemm_to_real_with_mask(
                    x_ii, a_ij, mask=self.mm_mask
                )
                x_.blocks[i, j] = -complex_gemm_to_real_with_mask(
                    temp_mult2, x_jj, mask=self.mm_mask
                )

                # NOTE: Cursed Python multiple assignment syntax.
                temp_mult3 = complex_gemm_to_real_with_mask(
                    x_ii, a_ij, mask=self.mm_mask
                )
                temp_result3 = complex_gemm_to_real_with_mask(
                    temp_mult3, x_ji, mask=self.mm_mask
                )
                x_.blocks[i, i] = x_diag_blocks[i] = x_ii - temp_result3

        if out is None:
            return x

    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
        out: tuple[DSDBSparse, ...] | None = None,
        return_retarded: bool = False,
        return_current: bool = False,
        mm_mask: str | None = None,
        inv_mask: str = "fp64",
        tmp_mask: str = "fp64",
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
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : tuple[DSDBSparse, ...] | None, optional
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

        assert inv_mask in ["fp32", "fp64"], f"Invalid inversion mask: {inv_mask}"
        assert tmp_mask in ["fp32", "fp64"], f"Invalid temporary mask: {tmp_mask}"

        self.mm_mask = mm_mask

        if inv_mask == "fp32":
            inv_type = xp.complex64
        else:
            inv_type = xp.complex128

        if tmp_mask == "fp32":
            tmp_type = xp.complex64
        else:
            tmp_type = xp.complex128

        in_type = a.data.dtype

        # Initialize empty lists for the dense diagonal blocks.
        xr_diag_blocks: list[NDArray | None] = [None] * a.num_blocks
        xl_diag_blocks: list[NDArray | None] = [None] * a.num_blocks
        xg_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_blocks)

        if return_current:
            # Allocate a buffer for the current.
            current = xp.zeros((*a.shape[:-2], a.num_blocks - 1), dtype=a.dtype)

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
            a_jj = (
                a_.blocks[0, 0]
                if obc_r is None
                else a_.blocks[0, 0] - obc_r[stack_slice]
            ).astype(tmp_type)
            obc_l = obc_blocks.lesser[0]
            sl_jj = (
                sigma_lesser_.blocks[0, 0]
                if obc_l is None
                else sigma_lesser_.blocks[0, 0] + obc_l[stack_slice]
            ).astype(tmp_type)
            obc_g = obc_blocks.greater[0]
            sg_jj = (
                sigma_greater_.blocks[0, 0]
                if obc_g is None
                else sigma_greater_.blocks[0, 0] + obc_g[stack_slice]
            ).astype(tmp_type)

            xr_jj = inv(a_jj.astype(inv_type)).astype(tmp_type)
            xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
            xr_diag_blocks[0] = xr_jj
            temp_mult_l0 = complex_gemm_to_real_with_mask(
                xr_jj, sl_jj, mask=self.mm_mask
            )
            xl_diag_blocks[0] = complex_gemm_to_real_with_mask(
                temp_mult_l0, xr_jj_dagger, mask=self.mm_mask
            )
            temp_mult_g0 = complex_gemm_to_real_with_mask(
                xr_jj, sg_jj, mask=self.mm_mask
            )
            xg_diag_blocks[0] = complex_gemm_to_real_with_mask(
                temp_mult_g0, xr_jj_dagger, mask=self.mm_mask
            )

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # Check if there are OBC blocks for the current layer.
                obc_r = obc_blocks.retarded[j]
                a_jj = (
                    a_.blocks[j, j]
                    if obc_r is None
                    else a_.blocks[j, j] - obc_r[stack_slice]
                ).astype(tmp_type)
                obc_l = obc_blocks.lesser[j]
                sl_jj = (
                    sigma_lesser_.blocks[j, j]
                    if obc_l is None
                    else sigma_lesser_.blocks[j, j] + obc_l[stack_slice]
                ).astype(tmp_type)
                obc_g = obc_blocks.greater[j]
                sg_jj = (
                    sigma_greater_.blocks[j, j]
                    if obc_g is None
                    else sigma_greater_.blocks[j, j] + obc_g[stack_slice]
                ).astype(tmp_type)

                # Get the blocks that are used multiple times.
                a_ji = (a_.blocks[j, i]).astype(tmp_type)
                xr_ii = xr_diag_blocks[i]

                # Precompute the transposes that are used multiple times.
                a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

                # Precompute some terms that are used multiple times.
                a_ji_xr_ii = complex_gemm_to_real_with_mask(
                    a_ji, xr_ii, mask=self.mm_mask
                )

                temp_inv_arg = complex_gemm_to_real_with_mask(
                    a_ji_xr_ii, (a_.blocks[i, j]).astype(tmp_type), mask=self.mm_mask
                )
                xr_jj = inv((a_jj - temp_inv_arg).astype(inv_type)).astype(tmp_type)
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_diag_blocks[j] = xr_jj

                a_ji_xr_ii_sx_ij = complex_gemm_to_real_with_mask(
                    a_ji_xr_ii,
                    (sigma_lesser_.blocks[i, j]).astype(tmp_type),
                    mask=self.mm_mask,
                )

                temp_xl_inner1 = complex_gemm_to_real_with_mask(
                    a_ji, xl_diag_blocks[i], mask=self.mm_mask
                )
                temp_xl_inner2 = complex_gemm_to_real_with_mask(
                    temp_xl_inner1, a_ji_dagger, mask=self.mm_mask
                )
                temp_xl_sum = (
                    sl_jj
                    + temp_xl_inner2
                    + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                    - a_ji_xr_ii_sx_ij
                )
                temp_xl_mult1 = complex_gemm_to_real_with_mask(
                    xr_jj, temp_xl_sum, mask=self.mm_mask
                )
                xl_diag_blocks[j] = complex_gemm_to_real_with_mask(
                    temp_xl_mult1, xr_jj_dagger, mask=self.mm_mask
                )

                a_ji_xr_ii_sx_ij = complex_gemm_to_real_with_mask(
                    a_ji_xr_ii,
                    (sigma_greater_.blocks[i, j]).astype(tmp_type),
                    mask=self.mm_mask,
                )

                temp_xg_inner1 = complex_gemm_to_real_with_mask(
                    a_ji, xg_diag_blocks[i], mask=self.mm_mask
                )
                temp_xg_inner2 = complex_gemm_to_real_with_mask(
                    temp_xg_inner1, a_ji_dagger, mask=self.mm_mask
                )
                temp_xg_sum = (
                    sg_jj
                    + temp_xg_inner2
                    + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                    - a_ji_xr_ii_sx_ij
                )
                temp_xg_mult1 = complex_gemm_to_real_with_mask(
                    xr_jj, temp_xg_sum, mask=self.mm_mask
                )
                xg_diag_blocks[j] = complex_gemm_to_real_with_mask(
                    temp_xg_mult1, xr_jj_dagger, mask=self.mm_mask
                )

            # We need to write the last diagonal blocks to the output.
            xl_.blocks[a.num_blocks - 1, a.num_blocks - 1] = 0.5 * (
                xl_diag_blocks[-1] - xl_diag_blocks[-1].conj().swapaxes(-2, -1)
            ).astype(in_type)
            xg_.blocks[a.num_blocks - 1, a.num_blocks - 1] = 0.5 * (
                xg_diag_blocks[-1] - xg_diag_blocks[-1].conj().swapaxes(-2, -1)
            ).astype(in_type)
            if return_retarded:
                xr_.blocks[a.num_blocks - 1, a.num_blocks - 1] = (
                    xr_diag_blocks[-1]
                ).astype(in_type)

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                # Get the blocks that are used multiple times.
                xr_ii = xr_diag_blocks[i]
                xr_jj = xr_diag_blocks[j]
                a_ij = (a_.blocks[i, j]).astype(tmp_type)
                a_ji = (a_.blocks[j, i]).astype(tmp_type)
                xl_ii = xl_diag_blocks[i]
                xl_jj = xl_diag_blocks[j]
                xg_ii = xg_diag_blocks[i]
                xg_jj = xg_diag_blocks[j]
                sigma_lesser_ij = (sigma_lesser_.blocks[i, j]).astype(tmp_type)
                sigma_greater_ij = (sigma_greater_.blocks[i, j]).astype(tmp_type)

                # Precompute the transposes that are used multiple times.
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)

                # Precompute the terms that are used multiple times.
                xr_ii_a_ij = complex_gemm_to_real_with_mask(
                    xr_ii, a_ij, mask=self.mm_mask
                )
                a_ij_dagger_xr_ii_dagger = xr_ii_a_ij.conj().swapaxes(-2, -1)
                xr_jj_a_ji = complex_gemm_to_real_with_mask(
                    xr_jj, a_ji, mask=self.mm_mask
                )
                a_ji_dagger_xr_jj_dagger = xr_jj_a_ji.conj().swapaxes(-2, -1)
                xr_ii_a_ij_xr_jj = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij, xr_jj, mask=self.mm_mask
                )
                xr_jj_dagger_a_ij_dagger_xr_ii_dagger = (
                    xr_ii_a_ij_xr_jj.conj().swapaxes(-2, -1)
                )
                xr_ii_a_ij_xr_jj_a_ji = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij, xr_jj_a_ji, mask=self.mm_mask
                )

                temp_1x_part1 = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij_xr_jj_a_ji, xl_ii, mask=self.mm_mask
                )
                temp_1x_part2_1 = complex_gemm_to_real_with_mask(
                    xr_ii, sigma_lesser_ij, mask=self.mm_mask
                )
                temp_1x_part2 = complex_gemm_to_real_with_mask(
                    temp_1x_part2_1,
                    xr_jj_dagger_a_ij_dagger_xr_ii_dagger,
                    mask=self.mm_mask,
                )
                temp_1x = temp_1x_part1 - temp_1x_part2
                temp_1x -= temp_1x.conj().swapaxes(-2, -1)
                temp_2x = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij, xl_jj, mask=self.mm_mask
                )

                xl_ij_part1 = complex_gemm_to_real_with_mask(
                    xl_ii, a_ji_dagger_xr_jj_dagger, mask=self.mm_mask
                )
                xl_ij_part2_1 = complex_gemm_to_real_with_mask(
                    xr_ii, sigma_lesser_ij, mask=self.mm_mask
                )
                xl_ij_part2 = complex_gemm_to_real_with_mask(
                    xl_ij_part2_1, xr_jj_dagger, mask=self.mm_mask
                )
                xl_ij = -temp_2x - xl_ij_part1 + xl_ij_part2

                xl_.blocks[i, j] = (xl_ij).astype(in_type)
                if not xl_.symmetry:
                    xl_.blocks[j, i] = (-xl_ij.conj().swapaxes(-2, -1)).astype(in_type)

                xl_diag_part = complex_gemm_to_real_with_mask(
                    temp_2x, a_ij_dagger_xr_ii_dagger, mask=self.mm_mask
                )
                xl_diag_blocks[i] = xl_ii + xl_diag_part + temp_1x
                xl_.blocks[i, i] = 0.5 * (
                    xl_diag_blocks[i] - xl_diag_blocks[i].conj().swapaxes(-2, -1)
                ).astype(in_type)

                temp_1x_part1_g = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij_xr_jj_a_ji, xg_ii, mask=self.mm_mask
                )
                temp_1x_part2_1_g = complex_gemm_to_real_with_mask(
                    xr_ii, sigma_greater_ij, mask=self.mm_mask
                )
                temp_1x_part2_g = complex_gemm_to_real_with_mask(
                    temp_1x_part2_1_g,
                    xr_jj_dagger_a_ij_dagger_xr_ii_dagger,
                    mask=self.mm_mask,
                )
                temp_1x = temp_1x_part1_g - temp_1x_part2_g
                temp_1x -= temp_1x.conj().swapaxes(-2, -1)
                temp_2x = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij, xg_jj, mask=self.mm_mask
                )

                xg_ij_part1 = complex_gemm_to_real_with_mask(
                    xg_ii, a_ji_dagger_xr_jj_dagger, mask=self.mm_mask
                )
                xg_ij_part2_1 = complex_gemm_to_real_with_mask(
                    xr_ii, sigma_greater_ij, mask=self.mm_mask
                )
                xg_ij_part2 = complex_gemm_to_real_with_mask(
                    xg_ij_part2_1, xr_jj_dagger, mask=self.mm_mask
                )
                xg_ij = -temp_2x - xg_ij_part1 + xg_ij_part2

                xg_.blocks[i, j] = (xg_ij).astype(in_type)
                if not xg_.symmetry:
                    xg_.blocks[j, i] = (-xg_ij.conj().swapaxes(-2, -1)).astype(in_type)

                xg_diag_part = complex_gemm_to_real_with_mask(
                    temp_2x, a_ij_dagger_xr_ii_dagger, mask=self.mm_mask
                )
                xg_diag_blocks[i] = xg_ii + xg_diag_part + temp_1x
                xg_.blocks[i, i] = 0.5 * (
                    xg_diag_blocks[i] - xg_diag_blocks[i].conj().swapaxes(-2, -1)
                ).astype(in_type)

                if return_current:
                    a_ji_xr_ii_curr = complex_gemm_to_real_with_mask(
                        a_ji, xr_ii, mask=self.mm_mask
                    )
                    a_ji_xr_ii_sx_ij_curr = complex_gemm_to_real_with_mask(
                        a_ji_xr_ii_curr, sigma_lesser_ij, mask=self.mm_mask
                    )
                    temp_curr_l1 = complex_gemm_to_real_with_mask(
                        a_ji, xl_ii, mask=self.mm_mask
                    )
                    temp_curr_l2 = complex_gemm_to_real_with_mask(
                        temp_curr_l1, a_ji_dagger, mask=self.mm_mask
                    )
                    sigma_lesser_tilde = (
                        temp_curr_l2
                        + a_ji_xr_ii_sx_ij_curr.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij_curr
                    )
                    a_ji_xr_ii_sx_ij_curr = complex_gemm_to_real_with_mask(
                        a_ji_xr_ii_curr, sigma_greater_ij, mask=self.mm_mask
                    )
                    temp_curr_g1 = complex_gemm_to_real_with_mask(
                        a_ji, xg_ii, mask=self.mm_mask
                    )
                    temp_curr_g2 = complex_gemm_to_real_with_mask(
                        temp_curr_g1, a_ji_dagger, mask=self.mm_mask
                    )
                    sigma_greater_tilde = (
                        temp_curr_g2
                        + a_ji_xr_ii_sx_ij_curr.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sx_ij_curr
                    )
                    temp_trace1 = complex_gemm_to_real_with_mask(sigma_greater_tilde, xl_diag_blocks[j], mask=self.mm_mask)
                    temp_trace2 = complex_gemm_to_real_with_mask(xg_diag_blocks[j], sigma_lesser_tilde, mask=self.mm_mask)
                    current[stack_slice, ..., i] = xp.trace(
                        temp_trace1 - temp_trace2,
                    )

                xr_final_mult = complex_gemm_to_real_with_mask(
                    xr_ii_a_ij_xr_jj_a_ji, xr_ii, mask=self.mm_mask
                )
                xr_diag_blocks[i] = xr_ii + xr_final_mult
                if return_retarded:
                    xr_.blocks[i, i] = (xr_diag_blocks[i]).astype(in_type)

        if out is None:
            if return_retarded:
                if return_current:
                    return xl, xg, xr, current
                return xl, xg, xr
            return xl, xg

        if return_current:
            return current
