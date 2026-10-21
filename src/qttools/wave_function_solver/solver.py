# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the abstract base class for wave function solvers."""

from abc import ABC, abstractmethod

from qttools import NDArray, sparse, xp
from qttools.comm.comm import _SubCommunicator

MATRIX_DTYPES = [
    xp.dtype("float32"),
    xp.dtype("float64"),
    xp.dtype("complex64"),
    xp.dtype("complex128"),
]
MATRIX_PROPERTIES = [
    "general",
    "symmetric_indefinite",
    "hermitian_indefinite",
    "symmetric_positive_definite",
    "hermitian_positive_definite",
]
MATRIX_VIEWS = ["full", "upper"]


class WFSolver(ABC):
    """Abstract base class for wave function solvers.


    Parameters
    ----------
    matrix_dtype : xp.dtype[xp.inexact]
        The data type of the system matrix. This needs to be provided to
        correctly initialize the solver.
    matrix_property : str
        The property of the system matrix. This describes properties
        like symmetry and definiteness, which can be used by solvers to
        optimize the solution process. Can be None if the solver does
        not require this information or if it can be inferred from the
        matrix itself.
    matrix_view : str
        The view of the system matrix sparsity. Not all solvers are
        compatible with all views.
    reuse_analysis : bool, optional
        Whether to reuse the analysis phase when solve is called
        multiple times, by default False. This typically includes
        symbolic factorization and ordering but can vary between
        solvers. This is useful when solving multiple linear systems
        with the same sparsity pattern but different numerical values.
    reuse_factorization : bool, optional
        Whether to reuse the numerical factorization when solve is
        called multiple times, by default False. This can only be True
        if reuse_analysis is also True. Note that this must only be True
        if the matrix values have not changed since the last
        factorization.
    comm : _SubCommunicator, optional
        The communicator for distributed solves. If None, the solver
        will assume a single-GPU solve. This must be provided together
        with local_rows.
    local_rows : tuple, optional
        A tuple specifying the local row distribution for distributed
        solves. If None, the solver will assume a single-GPU solve.

    """

    def __init__(
        self,
        matrix_dtype: xp.dtype[xp.inexact],
        matrix_property: str,
        matrix_view: str,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
        comm: _SubCommunicator | None = None,
        local_rows: tuple | None = None,
    ) -> None:
        """Initializes the wave function solver."""

        # Comm and local_rows must be provided together or not at all.
        if (comm is None) != (local_rows is None):
            raise ValueError(
                "Both 'comm' and 'local_rows' must be provided together or not at all."
            )

        if matrix_dtype not in MATRIX_DTYPES:
            raise ValueError(
                f"Invalid matrix dtype '{matrix_dtype}'. "
                f"Valid options are: {MATRIX_DTYPES}"
            )
        if matrix_view not in MATRIX_VIEWS:
            raise ValueError(
                f"Invalid matrix view '{matrix_view}'. "
                f"Valid options are: {MATRIX_VIEWS}"
            )

        if matrix_property not in MATRIX_PROPERTIES:
            raise ValueError(
                f"Invalid matrix property '{matrix_property}'. "
                f"Valid options are: {MATRIX_PROPERTIES}"
            )

        if matrix_view == "upper" and matrix_property == "general":
            raise ValueError(
                "Matrix view 'upper' is not compatible with "
                "matrix property 'general'."
            )

        self._distributed = (
            comm is not None and comm.size > 1 and local_rows is not None
        )

        self.matrix_dtype = matrix_dtype
        self.matrix_view = matrix_view
        self.reuse_analysis = reuse_analysis
        self.reuse_factorization = reuse_factorization
        self._comm = comm
        self._local_rows = local_rows

    @staticmethod
    @abstractmethod
    def valid_views() -> list[str]:
        """Returns the valid matrix views for the solver.

        Returns
        -------
        list[str]
            The valid matrix views.

        """
        ...

    @abstractmethod
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
    ) -> NDArray:
        """Solves the sparse linear system a @ x = b.

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
        ...
