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

    def __init__(self):
        self.blocks = None
        self.schur = None
        self._triu_cache = {}

    def _symmetrize_from_upper_inplace(self, block: NDArray) -> None:
        """Fill lower triangle from upper triangle: block[j,i] = conj(block[i,j])."""
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

        frontier = xp.zeros(n, dtype=bool)
        visited = xp.zeros(n, dtype=bool)
        frontier[0] = True

        self.blocks = []

        while xp.any(frontier):
            current_block = xp.where(frontier)[0]
            self.blocks.append(current_block)
            visited |= frontier

            new_frontier = xp.zeros(n, dtype=bool)
            for node in current_block.tolist():
                new_frontier[a.indices[a.indptr[node] : a.indptr[node + 1]]] = True

            frontier = new_frontier & (~visited)

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
        self._symmetrize_from_upper_inplace(a00)
        a01 = (
            densify(a[self.blocks[0], :][:, self.blocks[1]])
            + densify(a[self.blocks[1], :][:, self.blocks[0]]).T.conj()
        )

        # LU factorization of the first block
        a00, piv = lu_factor(a00, overwrite_a=False)
        # Solve for the first block
        b[self.blocks[0]] = lu_solve((a00, piv), b[self.blocks[0]], overwrite_b=False)
        # Store the Schur complement for the next block
        self.schur.append(lu_solve((a00, piv), a01, overwrite_b=False))

        for i in range(1, len(self.blocks) - 1):

            a10 = a01.T.conj()
            a00 = densify(a[self.blocks[i], :][:, self.blocks[i]])
            self._symmetrize_from_upper_inplace(a00)
            a01 = (
                densify(a[self.blocks[i], :][:, self.blocks[i + 1]])
                + densify(a[self.blocks[i + 1], :][:, self.blocks[i]]).T.conj()
            )
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
        a10 = a01.T.conj()
        a00 = densify(a[self.blocks[-1], :][:, self.blocks[-1]])
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

    @profiler.profile(level="api")
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

        return b
