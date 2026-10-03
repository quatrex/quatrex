# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes a mock distributed wave function solver."""

import warnings

from qttools import NDArray, sparse, xp
from qttools.comm import comm as quatrex_comm
from qttools.comm.comm import _SubCommunicator
from qttools.wave_function_solver.auto_select import _select_non_distributed_solver
from qttools.wave_function_solver.solver import WFSolver


class Mock(WFSolver):
    """Mock wavefunction solver for when no real distributed solver is
    available.

    Note
    ----
    This solver only exists to allow for testing and development of
    distributed QTBM.

    Parameters
    ----------
    matrix_type : str, optional
        The type of the system matrix. This describes properties like
        symmetry and definiteness. If None, the solver will use a
        general matrix type.
    matrix_view : str, optional
        The view of the system matrix sparsity. This solver supports
        'full', 'upper', and 'lower' views. If None, the solver will use
        the 'full' view.
    comm : _SubCommunicator, optional
        The communicator for distributed solves. If None, the solver
        will assume a single-rank solve. This must be provided together
        with local_rows.
    local_rows : tuple, optional
        A tuple specifying the local row distribution for distributed
        solves. If None, the solver will assume a single-rank solve.

    """

    def __init__(
        self,
        matrix_type: str | None = None,
        matrix_view: str | None = None,
        comm: _SubCommunicator | None = None,
        local_rows: tuple | None = None,
    ):
        """Initializes the mock solver."""

        if (comm is None) != (local_rows is None):
            raise ValueError(
                "Both comm and local_rows must be provided together for "
                "distributed solves."
            )

        if quatrex_comm.rank == 0:
            warnings.warn(
                "Using MockDist solver. This is a mock implementation "
                "and may not be very fast.",
                UserWarning,
            )

        self._comm = comm
        self._local_rows = local_rows

        self._solver = _select_non_distributed_solver(
            matrix_type=matrix_type,
            matrix_view=matrix_view,
        )

    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
    ):
        """Solves the sparse linear system a @ x = b using cuDSS.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix in CSR format.
        b : NDArray
            The dense right-hand side array with shape (n, batchsize).
        reuse_analysis : bool, optional
            Whether to reuse the symbolic factorization from a previous
            solve. Default is False. This is useful when solving
            multiple linear systems with the same sparsity pattern but
            different numerical values.
        reuse_factorization : bool, optional
            Whether to reuse the numerical factorization from a previous
            solve. Default is False. This can only be True if
            reuse_analysis is also True. Note that this must only be
            True if the matrix values have not changed since the last
            factorization.

        Returns
        -------
        x : NDArray
            The solution array with shape (n, batchsize).

        """
        if reuse_analysis or reuse_factorization:
            raise NotImplementedError(
                "MockDist solver does not support reuse_analysis or "
                "reuse_factorization."
            )

        if self._comm is not None:
            a = a.tocoo()
            data = a.data
            row_ind = a.row
            col_ind = a.col

            local_length = xp.array(a.shape[0])
            recv_buffer = xp.empty((self._comm.size,), dtype=local_length.dtype)
            self._comm.all_gather(local_length, recv_buffer)
            row_offsets = xp.array([0] + list(xp.cumsum(recv_buffer)))

            data = self._comm.all_gather_v(data, axis=0)
            row_ind = self._comm.all_gather_v(
                row_ind + row_offsets[self._comm.rank], axis=0
            )
            col_ind = self._comm.all_gather_v(col_ind, axis=0)

            b = self._comm.all_gather_v(b, axis=0)

            a = sparse.csr_matrix(
                (data, (row_ind, col_ind)), shape=(row_offsets[-1], a.shape[1])
            )

        out = self._solver.solve(
            a,
            b,
            reuse_analysis=reuse_analysis,
            reuse_factorization=reuse_factorization,
        )

        if self._comm is not None:
            out = out[row_offsets[self._comm.rank] : row_offsets[self._comm.rank + 1]]

        return out
