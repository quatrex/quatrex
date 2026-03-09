# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray, xp
from qttools.boundary_conditions.boundary_system import BaseBoundarySystem
from qttools.boundary_conditions.system_reduction import SystemReducer


class LyapunovSolver(ABC):
    r"""Solver interface for the discrete-time Lyapunov equation.

    The discrete-time Lyapunov equation is defined as:

    \[
        X - A X A^H = Q
    \]

    """

    @abstractmethod
    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
    ) -> NDArray:
        r"""Computes the solution of the discrete-time Lyapunov equation.

            The equation is give by:

            $$\mathbf{x} = \mathbf{q} + \mathbf{a} \mathbf{x}  \mathbf{a}^H$$

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.

        Returns
        -------
        x : NDArray
            The solution of the discrete-time Lyapunov equation.

        """
        ...


class LyapunovSystemReducer(SystemReducer):
    r"""A lyapunov system reducer.

    Parameters
    ----------
    reduce_sparsity : bool, optional
        Whether to reduce the sparsity of the system matrix.
        If sparsity of any obc is changed during runtime, then the cache
        needs to be invalidated. Default is True.
    assume_constant_sparsity : bool, optional
        Whether to assume that the sparsity pattern of the system matrix
        remains constant during runtime. If True, the sparsity pattern
        is only computed once. Default is True.

    """

    def __init__(
        self,
        reduce_sparsity: bool = True,
        assume_constant_sparsity: bool = True,
    ) -> None:
        """Initializes the lyapunov system."""

        super().__init__()

        self.number_non_zero_rows = None
        self.number_non_zero_cols = None
        self.rows_reduced_system = None
        self.cols_reduced_system = None
        self.reduce_sparsity = reduce_sparsity
        self.assume_constant_sparsity = assume_constant_sparsity

    def __contract_system_zero_cols(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> tuple[NDArray, ...]:
        """Contract the boundary system to a reduced system
        when there are zero cols in the system matrix."""
        if self.cols_reduced_system is None:
            raise ValueError(
                "The system reduction information is missing.\n"
                + "Make sure to call '_contract_system' before contracting the system."
            )

        a, q = boundary_system
        a_hat = a[..., self.cols_reduced_system, self.cols_reduced_system]
        q_hat = q[..., self.cols_reduced_system, self.cols_reduced_system]
        return a_hat, q_hat

    def __contract_system_zero_rows(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> tuple[NDArray, ...]:
        """Contract the boundary system to a reduced system
        when there are zero rows in the system matrix."""
        if self.rows_reduced_system is None:
            raise ValueError(
                "The system reduction information is missing.\n"
                + "Make sure to call '_contract_system' before expanding the solution."
            )

        a, q = boundary_system
        a_hat = a[..., self.rows_reduced_system, self.rows_reduced_system]
        a = xp.broadcast_to(a, q.shape)

        x = q.copy()
        x[..., self.rows_reduced_system, self.rows_reduced_system] = 0
        q_hat = q[..., self.rows_reduced_system, self.rows_reduced_system] + (
            a[..., self.rows_reduced_system, :]
            @ x
            @ a[..., self.rows_reduced_system, :].conj().swapaxes(-2, -1)
        )

        return a_hat, q_hat

    def __compute_sparsity_pattern(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> None:
        """Compute the sparsity pattern of the system matrix."""
        a, _ = boundary_system
        if (
            (not self.assume_constant_sparsity)
            or (self.rows_reduced_system is None)
            or (self.cols_reduced_system is None)
            or (self.number_non_zero_rows is None)
            or (self.number_non_zero_cols is None)
        ):

            # get first and last row/cols with non-zero elements
            nonzero_rows = xp.sum(xp.abs(a), axis=-1) > 0
            nonzero_cols = xp.sum(xp.abs(a), axis=-2) > 0

            first_nonzero_row = xp.argmax(nonzero_rows, axis=-1)
            last_nonzero_row = a.shape[-1] - xp.argmax(nonzero_rows[..., ::-1], axis=-1)

            first_nonzero_col = xp.argmax(nonzero_cols, axis=-1)
            last_nonzero_col = a.shape[-2] - xp.argmax(nonzero_cols[..., ::-1], axis=-1)

            # any over column/row dims
            nonzero_energies_rows = xp.any(nonzero_rows, axis=-1)
            nonzero_energies_cols = xp.any(nonzero_cols, axis=-1)
            # sanitiy check
            assert xp.allclose(nonzero_energies_rows, nonzero_energies_cols)
            nonzero_energies = nonzero_energies_rows

            # account for only zero rows/cols
            # any over batch dims
            if not xp.any(nonzero_energies):
                # hack to avoid empty slices
                # system solve will return q anyway
                self.rows_reduced_system = slice(0, 1)
                self.cols_reduced_system = slice(0, 1)
            else:
                self.rows_reduced_system = slice(
                    xp.min(first_nonzero_row[nonzero_energies]),
                    xp.max(last_nonzero_row[nonzero_energies]),
                )
                self.cols_reduced_system = slice(
                    xp.min(first_nonzero_col[nonzero_energies]),
                    xp.max(last_nonzero_col[nonzero_energies]),
                )

            # reduction methods are differently expensive
            # but we choose cols reduction here arbitrarily for zero rows
            # which costs an outer product
            self.number_non_zero_rows = (
                self.rows_reduced_system.stop - self.rows_reduced_system.start
            )
            self.number_non_zero_cols = (
                self.cols_reduced_system.stop - self.cols_reduced_system.start
            )

    def contract_system(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> tuple[NDArray, ...]:
        """Contract the boundary system to a reduced system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.

        Returns
        -------
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system.

        """
        if not self.reduce_sparsity:
            return boundary_system

        a, q = boundary_system
        assert q.shape[-2:] == a.shape[-2:]

        # allows for broadcasting of a to q shape
        assert q.ndim >= a.ndim

        self.__compute_sparsity_pattern(boundary_system)

        if self.number_non_zero_cols is None or self.number_non_zero_rows is None:
            raise ValueError("The system reduction information is missing.")

        # more zero rows than cols -> system was reduced by rows
        if self.number_non_zero_rows < self.number_non_zero_cols:
            return self.__contract_system_zero_rows(boundary_system)

        return self.__contract_system_zero_cols(boundary_system)

    def __expand_solution_zero_cols(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_solution: NDArray,
    ) -> NDArray:
        """Expand the solution from the reduced system to the full system
        when there are zero cols in the system matrix."""
        if self.cols_reduced_system is None:
            raise ValueError(
                "The system reduction information is missing.\n"
                + "Make sure to call '_contract_system' before expanding the solution."
            )

        a, q = boundary_system
        a = xp.broadcast_to(a, q.shape)
        solution = q + a[..., :, self.cols_reduced_system] @ reduced_solution @ a[
            ..., :, self.cols_reduced_system
        ].conj().swapaxes(-2, -1)

        return solution

    def __expand_solution_zero_rows(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_solution: NDArray,
    ) -> NDArray:
        """Expand the solution from the reduced system to the full system
        when there are zero rows in the system matrix."""
        if self.rows_reduced_system is None:
            raise ValueError(
                "The system reduction information is missing.\n"
                + "Make sure to call '_contract_system' before expanding the solution."
            )

        _, q = boundary_system
        solution = q.copy()
        solution[..., self.rows_reduced_system, self.rows_reduced_system] = (
            reduced_solution
        )

        return solution

    def expand_solution(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        reduced_solution: NDArray,
    ) -> NDArray:
        """Expand the solution from the reduced system to the full system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the reduced system matrix
            and 'q' is the reduced right-hand side matrix.
        reduced_solution : NDArray
            The solution of the reduced system.

        Returns
        -------
        full_solution : NDArray
            The solution of the full system.

        """
        if not self.reduce_sparsity:
            return reduced_solution

        if self.number_non_zero_cols is None or self.number_non_zero_rows is None:
            raise ValueError(
                "The system reduction information is missing.\n"
                + "Make sure to call '_contract_system' before expanding the solution."
            )

        # more zero rows than cols -> system was reduced by rows
        if self.number_non_zero_rows < self.number_non_zero_cols:
            return self.__expand_solution_zero_rows(boundary_system, reduced_solution)

        return self.__expand_solution_zero_cols(boundary_system, reduced_solution)

    def expand_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Expand the residuals from the reduced system to the full system.

        The reduced lyapynov system has the same residuals as the full system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the reduced system matrix
            and 'q' is the reduced right-hand side matrix.
        rel_residuals : NDArray
            The relative residuals of the reduced system.
        abs_residuals : NDArray
            The absolute residuals of the reduced system.

        Returns
        -------
        full_rel_residuals : NDArray
            The relative residuals of the full system.
        full_abs_residuals : NDArray
            The absolute residuals of the full system.

        """
        return rel_residuals, abs_residuals


class LyapunovSystem(BaseBoundarySystem):
    r"""A lyapunov system solver with memoization and system reduction.

        The lyapyunov equation to be solved is given by:

        $$\mathbf{x} = \mathbf{q} + \mathbf{a} \mathbf{x}  \mathbf{a}^H$$

    Parameters
    ----------
    boundary_solver : LyapunovSolver
        The lyapunov solver to be memoized.
    cache_compressor : object, optional
        An object with 'compress' and 'decompress' methods to handle
        cache compression. If None, no compression is applied.
    num_ref_iterations : int, optional
        The number of fixed-point iterations to refine the solution. Default is 2.
    relative_tol : float, optional
        The relative tolerance for convergence. Default is 0.2.
    absolute_tol : float, optional
        The absolute tolerance for convergence. Default is 1e-6.
    warning_threshold : float, optional
        The threshold for issuing a warning about high residuals. Default is 0.1.
    mode : str, optional
        The memoization mode. Can be 'off', 'auto', 'force-after-first', or 'force'.
        Default is 'auto'.
    agreement_threshold : float, optional
        The threshold for agreement across MPI ranks to consider a memoized solution valid.
        Default is 0.999.
    reduce_sparsity : bool, optional
        Whether to reduce the sparsity of the system matrix.
        If sparsity of any obc is changed during runtime, then the cache
        needs to be invalidated. Default is True.
    assume_constant_sparsity : bool, optional
        Whether to assume that the sparsity pattern of the system matrix
        remains constant during runtime. If True, the sparsity pattern
        is only computed once. Default is True.

    """

    def __init__(
        self,
        boundary_solver: LyapunovSolver,
        cache_compressor: None = None,
        system_reducer: SystemReducer | None = None,
        num_ref_iterations: int = 2,
        relative_tol: float = 2e-1,
        absolute_tol: float = 1e-6,
        warning_threshold: float = 1e-1,
        mode: str = "auto",
        agreement_threshold: float = 0.999,
    ) -> None:
        """Initializes the lyapunov system."""
        super().__init__(
            boundary_solver=boundary_solver,
            cache_compressor=cache_compressor,
            system_reducer=system_reducer,
            num_ref_iterations=num_ref_iterations,
            relative_tol=relative_tol,
            absolute_tol=absolute_tol,
            warning_threshold=warning_threshold,
            mode=mode,
            agreement_threshold=agreement_threshold,
        )

    def _fixed_point_step(
        self,
        boundary_system: tuple[NDArray, ...],
        solution: NDArray,
    ) -> NDArray:
        r"""Perform a fixed-point iteration step to refine the solution.

            The fix-point iteration is given by:
            $$\mathbf{x}_{n+1} = \mathbf{q} + \mathbf{a} \mathbf{x}_{n}  \mathbf{a}^H$$

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        solution : NDArray
            The current solution to refine.

        Returns
        -------
        refined_solution : NDArray
            The refined solution after one fixed-point iteration step.

        """
        a, q = boundary_system
        return q + a @ solution @ a.conj().swapaxes(-2, -1)

    def _get_starting_guess(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> NDArray:
        """Get a starting guess for the boundary system.

        For the lyapunov equation, a good starting guess is simply the right-hand side 'q'.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.

        Returns
        -------
        starting_guess : NDArray
            The starting guess for the boundary system.

        """
        _, q = boundary_system
        return q

    def _get_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        test_solution: NDArray,
    ) -> tuple[NDArray, ...]:
        r"""Compute the residuals of a test solution.

            They are computed as follows:
            $$\mathbf{x}_{ref} = \mathbf{q} + \mathbf{a} \mathbf{x}_{test}  \mathbf{a}^H$$
            $$\mathbf{residual_{abs}} = \lvert \mathbf{x}_{ref} - \mathbf{x}_{test} \rvert$$
            $$\mathbf{residual_{rel}} = \frac{\mathbf{residual_{abs}}}{\lvert \mathbf{x}_{ref} \rvert}$$

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        test_solution : NDArray
            The test solution to evaluate.

        Returns
        -------
        rel_residuals : NDArray
            The relative residuals of the test solution.
        abs_residuals : NDArray
            The absolute residuals of the test solution.
        solution : NDArray
            The (possibly refined) solution of the boundary system.

        """
        solution_ref = self._fixed_point_step(boundary_system, test_solution)

        abs_residuals = xp.linalg.norm(solution_ref - test_solution, axis=(-2, -1))
        rel_residuals = abs_residuals / xp.linalg.norm(solution_ref, axis=(-2, -1))

        return rel_residuals, abs_residuals, solution_ref
