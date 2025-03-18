# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.


from qttools import NDArray, block_comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.greens_function_solver import _serinv
from qttools.greens_function_solver.solver import GFSolver, OBCBlocks


class RGFDist(GFSolver):
    """Distributed selected inversion solver.

    Parameters
    ----------
    solve_lesser : bool, optional
        Whether to solve the quadratic system associated with the lesser right-hand-side,
        by default False.
    solve_greater : bool, optional
        Whether to solve the quadratic system associated with the greater right-hand-side,
        by default False.
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
        out: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
    ) -> None | DSDBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        out : DSDBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSDBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSDBSparse object.

        """

        # Initialize temporary buffers.
        reduced_system = _serinv.ReducedSystem()

        # Initialize dense temporary buffers for the diagonal blocks and
        # the upper and lower auxiliary buffer blocks.
        x_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_local_blocks)

        if block_comm.rank == 0:
            # Direction: downward Schur-complement
            _serinv.downward_schur(
                a, x_diag_blocks, obc_blocks, invert_last_block=False
            )
        elif block_comm.rank == block_comm.size - 1:
            # Direction: upward Schur-complement
            _serinv.upward_schur(a, x_diag_blocks, obc_blocks, invert_last_block=False)
        else:
            # Permuted Schur-complement
            _serinv.permuted_schur(
                a, x_diag_blocks, buffer_lower, buffer_upper, obc_blocks
            )

        # Construct the reduced system.
        reduced_system.gather(a, x_diag_blocks, buffer_upper, buffer_lower)
        # Perform selected-inversion on the reduced system.
        reduced_system.solve()
        # Scatter the result to the output matrix.
        reduced_system.scatter(x_diag_blocks, buffer_upper, buffer_lower, out)

        if block_comm.rank == 0:
            # Direction: upward sell-inv
            _serinv.downward_selinv(a, x_diag_blocks, out)
        elif block_comm.rank == block_comm.size - 1:
            # Direction: downward sell-inv
            _serinv.upward_selinv(a, x_diag_blocks, out)
        else:
            # Permuted Sell-inv
            _serinv.permuted_selinv(a, x_diag_blocks, buffer_lower, buffer_upper, out)

    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        out: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
    ):
        """Performs selected inversion of a block-tridiagonal matrix.

        Can optionally solve the quadratic system associated with the
        Bl and Bg matrices in the equation AXA^T = B.

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
        out : tuple[DSBSparse, ...]
            Preallocated output matrices, by default None
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False

        """
        # Initialize temporary buffers.
        reduced_system = _serinv.ReducedSystem(selected_solve=True)

        xr_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        xr_buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        xr_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        xl_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        xl_buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        xl_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        xg_diag_blocks: list[NDArray | None] = [None] * a.num_local_blocks
        xg_buffer_lower: list[NDArray | None] = [None] * a.num_local_blocks
        xg_buffer_upper: list[NDArray | None] = [None] * a.num_local_blocks

        if obc_blocks is None:
            obc_blocks = OBCBlocks(num_blocks=a.num_local_blocks)

        xl_out, xg_out, *xr_out = out
        if return_retarded:
            if len(xr_out) != 1:
                raise ValueError("Invalid number of output matrices.")
            xr_out = xr_out[0]

        if block_comm.rank == 0:
            # Direction: downward Schur-complement
            _serinv.downward_schur(
                a=a,
                xr_diag_blocks=xr_diag_blocks,
                # Lesser quantities.
                sigma_lesser=sigma_lesser,
                xl_diag_blocks=xl_diag_blocks,
                # Greater quantities.
                sigma_greater=sigma_greater,
                xg_diag_blocks=xg_diag_blocks,
                # OBC and settings.
                obc_blocks=obc_blocks,
                invert_last_block=False,
                selected_solve=True,
            )
        elif block_comm.rank == block_comm.size - 1:
            # Direction: upward Schur-complement
            _serinv.upward_schur(
                a=a,
                xr_diag_blocks=xr_diag_blocks,
                # Lesser quantities.
                sigma_lesser=sigma_lesser,
                xl_diag_blocks=xl_diag_blocks,
                # Greater quantities.
                sigma_greater=sigma_greater,
                xg_diag_blocks=xg_diag_blocks,
                # OBC and settings.
                obc_blocks=obc_blocks,
                invert_last_block=False,
                selected_solve=True,
            )
        else:
            # Permuted Schur-complement
            _serinv.permuted_schur(
                a=a,
                xr_diag_blocks=xr_diag_blocks,
                xr_buffer_lower=xr_buffer_lower,
                xr_buffer_upper=xr_buffer_upper,
                # Lesser quantities.
                sigma_lesser=sigma_lesser,
                xl_diag_blocks=xl_diag_blocks,
                xl_buffer_lower=xl_buffer_lower,
                xl_buffer_upper=xl_buffer_upper,
                # Greater quantities.
                sigma_greater=sigma_greater,
                xg_diag_blocks=xg_diag_blocks,
                xg_buffer_lower=xg_buffer_lower,
                xg_buffer_upper=xg_buffer_upper,
                # OBC and settings.
                obc_blocks=obc_blocks,
                selected_solve=True,
            )

        # Construct the reduced system.
        reduced_system.gather(
            a=a,
            xr_diag_blocks=xr_diag_blocks,
            xr_buffer_lower=xr_buffer_lower,
            xr_buffer_upper=xr_buffer_upper,
            # Lesser quantities.
            sigma_lesser=sigma_lesser,
            xl_diag_blocks=xl_diag_blocks,
            xl_buffer_lower=xl_buffer_lower,
            xl_buffer_upper=xl_buffer_upper,
            # Greater quantities.
            sigma_greater=sigma_greater,
            xg_diag_blocks=xg_diag_blocks,
            xg_buffer_lower=xg_buffer_lower,
            xg_buffer_upper=xg_buffer_upper,
        )
        # Perform selected-inversion on the reduced system.
        reduced_system.solve()
        # Scatter the result to the output matrix.
        reduced_system.scatter(
            xr_diag_blocks=xr_diag_blocks,
            xr_buffer_lower=xr_buffer_lower,
            xr_buffer_upper=xr_buffer_upper,
            xr_out=xr_out,
            # Lesser quantities.
            xl_diag_blocks=xl_diag_blocks,
            xl_buffer_lower=xl_buffer_lower,
            xl_buffer_upper=xl_buffer_upper,
            xl_out=xl_out,
            # Greater quantities.
            xg_diag_blocks=xg_diag_blocks,
            xg_buffer_lower=xg_buffer_lower,
            xg_buffer_upper=xg_buffer_upper,
            xg_out=xg_out,
        )

        if block_comm.rank == 0:
            # Direction: upward sell-inv
            _serinv.downward_selinv(
                a=a,
                xr_diag_blocks=xr_diag_blocks,
                xr_out=xr_out,
                # Lesser quantities.
                sigma_lesser=sigma_lesser,
                xl_diag_blocks=xl_diag_blocks,
                xl_out=xl_out,
                # Greater quantities.
                sigma_greater=sigma_greater,
                xg_diag_blocks=xg_diag_blocks,
                xg_out=xg_out,
                selected_solve=True,
            )
        elif block_comm.rank == block_comm.size - 1:
            # Direction: downward sell-inv
            _serinv.upward_selinv(
                a=a,
                xr_diag_blocks=xr_diag_blocks,
                xr_out=xr_out,
                # Lesser quantities.
                sigma_lesser=sigma_lesser,
                xl_diag_blocks=xl_diag_blocks,
                xl_out=xl_out,
                # Greater quantities.
                sigma_greater=sigma_greater,
                xg_diag_blocks=xg_diag_blocks,
                xg_out=xg_out,
                selected_solve=True,
            )
        else:
            # Permuted Sell-inv
            _serinv.permuted_selinv(
                a=a,
                xr_diag_blocks=xr_diag_blocks,
                xr_buffer_lower=xr_buffer_lower,
                xr_buffer_upper=xr_buffer_upper,
                xr_out=xr_out,
                # Lesser quantities.
                sigma_lesser=sigma_lesser,
                xl_diag_blocks=xl_diag_blocks,
                xl_buffer_lower=xl_buffer_lower,
                xl_buffer_upper=xl_buffer_upper,
                xl_out=xl_out,
                # Greater quantities.
                sigma_greater=sigma_greater,
                xg_diag_blocks=xg_diag_blocks,
                xg_buffer_lower=xg_buffer_lower,
                xg_buffer_upper=xg_buffer_upper,
                xg_out=xg_out,
                selected_solve=True,
            )
