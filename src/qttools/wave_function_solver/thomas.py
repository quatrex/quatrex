# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

if "cupy" in sparse.__name__:
    from cupyx.cusparse import sparseToDense as densify
    from cupyx.scipy.linalg import lu_factor, lu_solve
else:

    def densify(a):
        return a.toarray()

    from scipy.linalg import lu_factor, lu_solve

profiler = Profiler()


class Thomas(WFSolver):
    """Wave function solver using block thomas algorithm."""

    def __init__(
        self, matrix_type: str = "complex_nonsymmetric", view: str = "default"
    ):
        """
        Initialize the Thomas solver.

        Parameters
        ----------
        matrix_type : str, optional
            The type of the system matrix. Must be one of 'real_symmetric_indefinite', 'complex_hermitian_indefinite', or 'complex_nonsymmetric'.
        view : str, optional
            The view of the system matrix. Must be one of 'default', 'up', or
            'down'. 'up' and 'down' are only valid for symmetric or Hermitian matrices.
        """

        if matrix_type not in [
            "real_symmetric_indefinite",
            "complex_hermitian_indefinite",
            "complex_nonsymmetric",
        ]:
            raise ValueError(
                f"Invalid matrix_type: {matrix_type}. Must be 'real_symmetric_indefinite', 'complex_hermitian_indefinite', or 'complex_nonsymmetric'."
            )
        if matrix_type in ["real_symmetric_indefinite", "complex_hermitian_indefinite"]:
            sym = True
        else:
            sym = False

        if view not in ["default", "up", "down"]:
            raise ValueError(
                f"Invalid view: {view}. Must be 'default', 'up', or 'down'."
            )
        if not sym and view != "default":
            raise ValueError(
                f"Invalid view: {view}. Must be 'default' when sym is False."
            )
        if view == "down":
            raise NotImplementedError("Down view is not implemented for Thomas solver.")

        self.sym = sym
        self.view = view
        self.blocks = None
        self.schur = None
        self._triu_cache = {}

    def _symmetrize_from_upper_inplace(self, block: NDArray) -> None:
        """Make block Hermitian by folding the lower triangle into the upper one.

        For each off-diagonal pair (i, j) with i < j:
            block[i, j] += conj(block[j, i])   # accumulate lower into upper
            block[j, i]  = conj(block[i, j])   # mirror back
        This handles permuted Hamiltonians where the lower triangle may carry
        values that have no corresponding entry in the upper triangle.
        """
        n = block.shape[0]
        if n <= 1:
            return

        idx = self._triu_cache.get(n)
        if idx is None:
            idx = xp.triu_indices(n, k=1)
            self._triu_cache[n] = idx

        i, j = idx
        block[j, i] = block[i, j].conj()

    def _plan(
        self,
        a: sparse.spmatrix,
    ) -> None:
        """
        Find block structure of the sparse matrix a and prepare for solving.
        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        """

        n = a.shape[0]

        self.blocks = []

        visited_max = -1
        frontier_max = 0

        while frontier_max > visited_max:
            start = visited_max + 1
            stop = frontier_max + 1

            current_block = xp.arange(start, stop)
            self.blocks.append(current_block)

            next_max = frontier_max
            for node in range(start, stop):
                row_start = a.indptr[node]
                row_stop = a.indptr[node + 1]
                if row_stop > row_start:
                    row_max = int(a.indices[row_stop - 1])
                    next_max = max(next_max, row_max)

            visited_max = frontier_max
            frontier_max = min(next_max, n - 1)

        if len(self.blocks) > 1:
            self.blocks[1] = xp.hstack([self.blocks[0], self.blocks[1]])
            self.blocks.pop(0)

    def _run_forward(self, a, b):
        """
        Run the forward elimination phase of the block Thomas algorithm.
        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.
        sol : NDArray
            The solution vector.
        """

        self.schur = []

        # First iteration
        a00 = densify(a[self.blocks[0], :][:, self.blocks[0]])
        if self.sym and self.view == "up":
            self._symmetrize_from_upper_inplace(a00)
        a01 = densify(a[self.blocks[0], :][:, self.blocks[1]])

        # LU factorization of the first block
        a00, piv = lu_factor(a00, overwrite_a=False)
        # Solve for the first block
        b[self.blocks[0]] = lu_solve((a00, piv), b[self.blocks[0]], overwrite_b=False)
        # Store the Schur complement for the next block
        self.schur.append(lu_solve((a00, piv), a01, overwrite_b=False))

        for i in range(1, len(self.blocks) - 1):
            if self.sym and self.view == "up":
                a10 = a01.T.conj()
            else:
                a10 = densify(a[self.blocks[i], :][:, self.blocks[i - 1]])
            a00 = densify(a[self.blocks[i], :][:, self.blocks[i]])
            if self.sym and self.view == "up":
                self._symmetrize_from_upper_inplace(a00)
            a01 = densify(a[self.blocks[i], :][:, self.blocks[i + 1]])
            # Update the current block with the Schur complement
            a00 -= a10 @ self.schur[i - 1]
            # LU factorization of the current block
            a00, piv = lu_factor(a00, overwrite_a=False)
            # Solve for the current block
            b[self.blocks[i]] = lu_solve(
                (a00, piv),
                b[self.blocks[i]] - a10 @ b[self.blocks[i - 1]],
                overwrite_b=False,
            )
            # Store the Schur complement for the next block
            self.schur.append(lu_solve((a00, piv), a01, overwrite_b=False))

        # Lu factorization of the last block
        if self.sym and self.view == "up":
            a10 = a01.T.conj()
        else:
            a10 = densify(a[self.blocks[-1], :][:, self.blocks[-2]])
        a00 = densify(a[self.blocks[-1], :][:, self.blocks[-1]])
        if self.sym and self.view == "up":
            self._symmetrize_from_upper_inplace(a00)
        a00 -= a10 @ self.schur[-1]
        a00, piv = lu_factor(a00, overwrite_a=False)
        # Solve for the last block
        b[self.blocks[-1]] = lu_solve(
            (a00, piv), b[self.blocks[-1]] - a10 @ b[self.blocks[-2]], overwrite_b=False
        )

    def _run_backward(self, b):
        """
        Run the backward substitution phase of the block Thomas algorithm.
        Parameters
        ----------
        b : NDArray
            The right-hand side vector.
        """

        for i in range(len(self.blocks) - 2, -1, -1):
            b[self.blocks[i]] -= self.schur[i] @ b[self.blocks[i + 1]]

    def _free_memory(self):
        """Free memory used for intermediate computations."""
        self.schur = None

    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_sym_fact: bool = False,
        reuse_fact: bool = False,
    ) -> NDArray:
        """Solves the sparse system a @ x = b using the block Thomas algorithm.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.

        Returns
        -------
        x : NDArray
            The solution vector.

        """

        if reuse_fact:
            raise NotImplementedError(
                "Factorization reuse is not implemented for Thomas solver."
            )

        if self.blocks is None or not reuse_sym_fact:
            self._plan(a)

        if a.shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch between a and b.")

        self._run_forward(a, b)
        self._run_backward(b)

        self._free_memory()

        return b
