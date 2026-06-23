# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

try:
    import cupy as cp
    import nvmath
    from nvmath.bindings import cudss

    cudss_matrix_types = {
        "real_symmetric_positive_definite": cudss.MatrixType.SPD,
        "real_symmetric_indefinite": cudss.MatrixType.SYMMETRIC,
        "complex_hermitian_positive_definite": cudss.MatrixType.HPD,
        "complex_hermitian_indefinite": cudss.MatrixType.HERMITIAN,
        "real_nonsymmetric": cudss.MatrixType.GENERAL,
        "complex_nonsymmetric": cudss.MatrixType.GENERAL,
    }

    cudss_matrix_views = {
        "full": cudss.MatrixViewType.FULL,
        "upper": cudss.MatrixViewType.UPPER,
        "lower": cudss.MatrixViewType.LOWER,
    }

    cudss_value_types = {
        cp.dtype("float64"): nvmath.CudaDataType.CUDA_R_64F,
        cp.dtype("complex128"): nvmath.CudaDataType.CUDA_C_64F,
    }

    cudss_available = True


except ImportError:
    cudss_available = False

from qttools import NDArray, sparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import synchronize_current_stream
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


class cuDSS(WFSolver):
    """Wavefunction solver using NVIDIA's cuDSS library for sparse
    direct solves on GPUs.

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

    """

    def __init__(self, matrix_type: str | None = None, matrix_view: str | None = None):
        """Initializes the cuDSS solver."""
        if not cudss_available:
            raise ImportError(
                "nvmath or its cudss bindings are not available. "
                "Please install them to use this solver."
            )
        if matrix_type is not None and matrix_type not in cudss_matrix_types:
            raise ValueError(
                f"Invalid matrix type '{matrix_type}'. "
                f"Valid options are: {list(cudss_matrix_types.keys())}"
            )

        self._mtype = cudss_matrix_types.get(matrix_type, cudss.MatrixType.GENERAL)
        self._mview = cudss_matrix_views.get(matrix_view, cudss.MatrixViewType.FULL)

        self._solver_handle = cudss.create()
        self._solver_config = cudss.config_create()
        self._solver_data = cudss.data_create(self._solver_handle)

        self.analyzed = False
        self.factorized = False

    def _create_cudss_csr(self, a: sparse.csr_matrix) -> int:
        """Creates a cuDSS matrix wrapper for the sparse system matrix a
        in CSR format.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix in CSR format.

        Returns
        -------
        csr_handle : int
            The cuDSS matrix handle for the matrix a.

        """
        if a.indices.dtype != cp.int32 or a.indptr.dtype != cp.int32:
            raise ValueError(
                f"Matrix has unsupported index data type. "
                f"Expected int32 for both indices and indptr, "
                f"but got {a.indices.dtype} and {a.indptr.dtype}."
            )

        value_type = cudss_value_types.get(a.dtype)

        if value_type is None:
            raise ValueError(
                f"Matrix has unsupported value data type {a.dtype}. "
                f"Supported types are: {list(cudss_value_types.keys())}"
            )

        csr_handle = cudss.matrix_create_csr(
            nrows=a.shape[0],
            ncols=a.shape[1],
            nnz=a.nnz,
            row_start=a.indptr.data.ptr,  # Beginning of row offset array
            row_end=0,  # Not used in standard CSR
            col_indices=a.indices.data.ptr,
            values=a.data.data.ptr,
            index_type=nvmath.CudaDataType.CUDA_R_32I,
            value_type=value_type,
            mtype=self._mtype,
            mview=self._mview,
            index_base=cudss.IndexBase.ZERO,
        )

        return csr_handle

    def _create_cudss_array(self, arr: NDArray) -> int:
        """Create a cuDSS wrapper for a dense array.

        Used for the right-hand side and solution.

        Parameters
        ----------
        arr : NDArray
            The dense array for which to create the cuDSS wrapper.

        Returns
        -------
        array_handle : int
            The cuDSS matrix handle for the array arr.

        """

        value_type = cudss_value_types.get(arr.dtype)
        if value_type is None:
            raise ValueError(
                f"Array has unsupported value data type {arr.dtype}. "
                f"Supported types are: {list(cudss_value_types.keys())}"
            )

        array_handle = cudss.matrix_create_dn(
            nrows=arr.shape[0],
            ncols=arr.shape[1],
            ld=arr.shape[0],  # leading dimension
            values=arr.data.ptr,
            value_type=value_type,
            layout=cudss.Layout.COL_MAJOR,  # Fortran order
        )
        return array_handle

    def _execute_phase(
        self, phase: "cudss.Phase", matrix: int, solution: int, rhs: int
    ):
        """Executes a specific phase of the cuDSS solver.

        Parameters
        ----------
        phase : cudss.Phase
            The phase of the solver to execute (ANALYSIS, FACTORIZATION,
            SOLVE).
        matrix : int
            The cuDSS handle for the system matrix.
        solution : int
            The cuDSS handle for the solution array.
        rhs : int
            The cuDSS handle for the right-hand side array.

        """
        synchronize_current_stream()
        cudss.execute(
            handle=self._solver_handle,
            phase=phase,
            solver_config=self._solver_config,
            solver_data=self._solver_data,
            input_matrix=matrix,
            solution=solution,
            rhs=rhs,
        )
        synchronize_current_stream()

    @profiler.profile("cuDSS: analysis", level="default")
    def _analyze(self, matrix: int, solution: int, rhs: int):
        """Performs symbolic factorization of the system.

        Parameters
        ----------
        matrix : int
            The cuDSS handle for the system matrix.
        solution : int
            The cuDSS handle for the solution array.
        rhs : int
            The cuDSS handle for the right-hand side array.

        """
        self._execute_phase(cudss.Phase.ANALYSIS, matrix, solution, rhs)

    @profiler.profile("cuDSS: factorization", level="default")
    def _factorize(self, matrix: int, solution: int, rhs: int):
        """Performs numeric factorization of the system.

        Parameters
        ----------
        matrix : int
            The cuDSS handle for the system matrix.
        solution : int
            The cuDSS handle for the solution array.
        rhs : int
            The cuDSS handle for the right-hand side array.

        """
        self._execute_phase(cudss.Phase.FACTORIZATION, matrix, solution, rhs)

    def _solve(self, matrix: int, solution: int, rhs: int):
        """Solves the linear system a @ x = b.

        Parameters
        ----------
        matrix : int
            The cuDSS handle for the system matrix.
        solution : int
            The cuDSS handle for the solution array.
        rhs : int
            The cuDSS handle for the right-hand side array.

        """
        self._execute_phase(cudss.Phase.SOLVE, matrix, solution, rhs)

    @profiler.profile("cuDSS solve", level="default")
    def solve(
        self,
        a: sparse.csr_matrix,
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
        if reuse_factorization and not reuse_analysis:
            raise ValueError(
                "Cannot reuse total factorization without reusing symbolic factorization."
            )
        if a.dtype != b.dtype:
            raise ValueError(
                f"Data type of a ({a.dtype}) does not match data type of b ({b.dtype}). "
                "Please ensure they have the same data type."
            )

        x = cp.zeros_like(b)

        # Set up the linear system.
        matrix = self._create_cudss_csr(a)
        solution = self._create_cudss_array(x)
        rhs = self._create_cudss_array(b)

        if not self.analyzed or not reuse_analysis:
            self._analyze(matrix, solution, rhs)
            self.analyzed = True

        if not self.factorized or not reuse_factorization:
            self._factorize(matrix, solution, rhs)
            self.factorized = True

        self._solve(matrix, solution, rhs)

        # Free GPU memory used for cuDSS linear system.
        for handle in [matrix, rhs, solution]:
            cudss.matrix_destroy(handle)

        return x
