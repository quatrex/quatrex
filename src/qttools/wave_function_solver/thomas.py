# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from functools import partial

from qttools import NDArray, sparse, xp
from qttools.kernels import linalg
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

if xp.__name__ == "cupy":
    from cupyx.cusparse import sparseToDense as _densify
else:

    def _densify(a: sparse.csr_matrix) -> NDArray:
        """Converts a sparse matrix to a dense array."""
        return a.toarray()


profiler = Profiler()


class Thomas(WFSolver):
    """Wave function solver using the block Thomas algorithm.

    Parameters
    ----------
    matrix_type : str, optional
        The type of the system matrix. Must be one of
        'real_symmetric_indefinite', 'complex_hermitian_indefinite',
        or 'complex_nonsymmetric'. Default is 'complex_nonsymmetric'.
    matrix_view : str, optional
        The view of the system matrix. Must be one of 'full',
        'upper', or 'lower'. 'upper' and 'lower' are only valid for
        symmetric or Hermitian matrices. Default is 'full'.

    """

    def __init__(
        self,
        matrix_type: str = "complex_nonsymmetric",
        matrix_view: str = "full",
    ):
        """Initializes the Thomas solver."""

        if matrix_type not in [
            "real_symmetric_indefinite",
            "complex_hermitian_indefinite",
            "complex_nonsymmetric",
        ]:
            raise ValueError(
                f"Invalid matrix_type: {matrix_type}. Must be "
                f"'real_symmetric_indefinite', 'complex_hermitian_indefinite', "
                "or 'complex_nonsymmetric'."
            )

        symmetric = matrix_type in [
            "real_symmetric_indefinite",
            "complex_hermitian_indefinite",
        ]

        if matrix_view not in ["full", "upper"]:
            raise ValueError(
                f"Invalid matrix_view: {matrix_view}. Must be 'full' or 'upper'."
            )

        if matrix_view == "full" and symmetric:
            raise ValueError(
                "Invalid matrix_view: 'full' is not valid for symmetric "
                "matrix types. Use 'upper' instead."
            )
        if matrix_view == "upper" and not symmetric:
            raise ValueError(
                "Invalid matrix_view: 'upper' is only valid for symmetric "
                "matrix types."
            )

        self.symmetric = symmetric
        self._block_slices = None

    def _get_block(self, a: sparse.csr_matrix, row: int, col: int) -> NDArray:
        """Extracts a block from the sparse matrix.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix.
        row : int
            The starting row index of the block.
        col : int
            The starting column index of the block.

        Returns
        -------
        NDArray
            The extracted block.

        """

        if self.symmetric and col < row:
            return self._get_block(a, row=col, col=row).conj().T

        block = _densify(a[self._block_slices[row], self._block_slices[col]])

        # Special case for diagonal blocks if symmetry.
        if self.symmetric and (col == row):
            block += block.conj().T
            xp.fill_diagonal(block, block.diagonal() / 2)

        return block

    def _get_slice(self, b: NDArray, row: int) -> NDArray:
        """Extracts a slice from the dense array.

        Parameters
        ----------
        b : NDArray
            The dense array.
        row : int
            The starting row index of the slice.

        Returns
        -------
        NDArray
            The extracted slice.

        """

        return b[self._block_slices[row], :]

    @profiler.profile("Thomas: analysis", level="default")
    def _analyze(self, a: sparse.csr_matrix) -> None:
        """Finds block structure of the sparse matrix a.

        This traverses the graph of the sparse matrix to find connected
        components. The blocks are contiguous and with variable size.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix.

        """

        n = a.shape[0]

        self._block_slices = []

        visited_max = -1
        frontier_max = 0

        while frontier_max > visited_max:
            start = visited_max + 1
            stop = frontier_max + 1

            self._block_slices.append(slice(start, stop))

            next_max = frontier_max
            for node in range(start, stop):
                row_start = a.indptr[node]
                row_stop = a.indptr[node + 1]
                if row_stop > row_start:
                    row_max = int(a.indices[row_stop - 1])
                    next_max = max(next_max, row_max)

            visited_max = frontier_max
            frontier_max = min(next_max, n - 1)

        if len(self._block_slices) > 1:
            self._block_slices[1] = slice(
                self._block_slices[0].start, self._block_slices[1].stop
            )
            self._block_slices.pop(0)

    @profiler.profile("Thomas: forward elimination", level="default")
    def _forward_elimination(self, a: sparse.spmatrix, b: NDArray):
        """Runs the forward elimination phase of the block Thomas
        algorithm.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.

        """

        num_blocks = len(self._block_slices)
        schur_complements = []
        a_ = partial(self._get_block, a)

        # First block.
        a_00_inv = linalg.inv(a_(0, 0))

        xp.matmul(
            a_00_inv,
            b[self._block_slices[0]],
            out=b[self._block_slices[0]],
        )

        schur_complements.append(a_00_inv @ a_(0, 1))

        # Forward elimination for the remaining blocks.
        for i in range(1, num_blocks):
            a_10 = a_(i, i - 1)

            a_00_inv = linalg.inv(a_(i, i) - a_10 @ schur_complements[i - 1])

            xp.matmul(
                a_00_inv,
                b[self._block_slices[i]] - a_10 @ b[self._block_slices[i - 1]],
                out=b[self._block_slices[i]],
            )

            # Store the Schur complement for the next block.
            # The last block is not needed.
            if i < num_blocks - 1:
                schur_complements.append(a_00_inv @ a_(i, i + 1))

        return schur_complements

    @profiler.profile("Thomas: backward substitution", level="default")
    def _backward_substitution(self, schur_complements: list[NDArray], b: NDArray):
        """Runs the backward substitution phase of the block Thomas
        algorithm.

        Parameters
        ----------
        schur_complements : list[NDArray]
            The Schur complements calculated during the forward
            elimination phase.
        b : NDArray
            The right-hand side vector.

        """

        for i in range(len(self._block_slices) - 2, -1, -1):
            b[self._block_slices[i]] -= (
                schur_complements[i] @ b[self._block_slices[i + 1]]
            )

    @profiler.profile("Thomas solve", level="default")
    def solve(
        self,
        a: sparse.csr_matrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
        overwrite_b: bool = True,
    ) -> NDArray:
        """Solves the sparse system a @ x = b using the block Thomas algorithm.

        Since b is directly modified in-place, b is lost after this
        call.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix.
        b : NDArray
            The dense right-hand side vector.
        reuse_analysis : bool, optional
            Whether to reuse the analysis phase from a previous solve,
            by by default False. This is useful when solving multiple
            linear systems with the same sparsity pattern but different
            numerical values.
        reuse_factorization : bool, optional
            Not implemented for this solver.
        overwrite_b : bool, optional
            Whether to overwrite the input b with the solution. Default
            is True. If False, a copy of b will be made before solving,
            and the original b will remain unchanged. This can be useful
            if the caller needs to keep the original right-hand side
            vector for later use.


        Returns
        -------
        x : NDArray
            The solution vector.

        """
        if reuse_factorization:
            raise NotImplementedError(
                "reuse_factorization is not implemented for Thomas solver."
            )

        if not overwrite_b:
            b = b.copy()

        if self._block_slices is None or not reuse_analysis:
            self._analyze(a)

        if a.shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch between a and b.")

        schur_complements = self._forward_elimination(a, b)
        self._backward_substitution(schur_complements, b)

        return b
