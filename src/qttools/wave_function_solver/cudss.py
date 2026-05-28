# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
try:
    import cupy as cp
    import nvmath
    from nvmath.bindings import cudss
    from nvmath.bindings.cudss import MatrixType, MatrixViewType

    nvmath_available = True

    matrix_combination = {
        "real_symmetric_positive_definite": MatrixType.SPD,
        "real_symmetric_indefinite": MatrixType.SYMMETRIC,
        "complex_hermitian_positive_definite": MatrixType.HPD,
        "complex_hermitian_indefinite": MatrixType.HERMITIAN,
        "real_nonsymmetric": MatrixType.GENERAL,
        "complex_nonsymmetric": MatrixType.GENERAL,
    }

except ImportError:
    nvmath_available = False

import time

from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.wave_function_solver.solver import WFSolver


def cudss_available():
    """Checks if the cuDSS solver is available."""
    return nvmath_available


class cuDSS(WFSolver):

    def _set_A(self, a: sparse.csr_matrix):
        """
        Create a cuDSS matrix wrapper for the sparse system matrix A in CSR format.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix A in CSR format.
        """
        n = a.shape[0]
        nnz = a.nnz
        if a.indices.dtype != cp.int32 or a.indptr.dtype != cp.int32:
            raise ValueError(
                f"Matrix A has unsupported index data type. Expected int32 for both indices and indptr, but got {a.indices.dtype} and {a.indptr.dtype}."
            )
        if a.dtype == cp.float64:
            type = nvmath.CudaDataType.CUDA_R_64F
        elif a.dtype == cp.complex128:
            type = nvmath.CudaDataType.CUDA_C_64F
        else:
            raise ValueError(
                f"Unsupported data type {a.dtype} for matrix A. Supported types are float64 and complex128."
            )
        self.A = cudss.matrix_create_csr(
            n,  # nrows
            n,  # ncols
            nnz,  # nnz
            a.indptr.data.ptr,  # row_start (beginning of row offset array)
            0,  # row_end (NULL/0 - not used in standard CSR)
            a.indices.data.ptr,  # column indices
            a.data.data.ptr,  # values
            nvmath.CudaDataType.CUDA_R_32I,  # index type (int32)
            type,
            self.M_type,  # matrix type (general)
            self.M_view,  # matrix view (full)
            cudss.IndexBase.ZERO,  # 0-based indexing
        )

    def _set_b(self, b: NDArray):
        """
        Create a cuDSS matrix wrapper for the dense right-hand side matrix B.

        Parameters
        ----------
        b : NDArray
            The dense right-hand side matrix B with shape (n, batchsize).
        """

        n = b.shape[0]
        batchsize = b.shape[1]

        if b.dtype == cp.float64:
            type = nvmath.CudaDataType.CUDA_R_64F
        elif b.dtype == cp.complex128:
            type = nvmath.CudaDataType.CUDA_C_64F
        else:
            raise ValueError(
                f"Unsupported data type {b.dtype} for matrix B. Supported types are float64 and complex128."
            )
        self.B = cudss.matrix_create_dn(
            n,  # nrows
            batchsize,  # ncols (number of RHS)
            n,  # leading dimension
            b.data.ptr,  # values
            type,
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

    def _set_x(self, x: NDArray, dtype):
        """
        Create a cuDSS matrix wrapper for the dense solution matrix X.

        Parameters
        ----------
        x : NDArray
            The dense solution matrix X with shape (n, batchsize).
        dtype : numpy.dtype
            The data type of the solution matrix X.
        """
        n = x.shape[0]
        batchsize = x.shape[1]
        if dtype == cp.float64:
            type = nvmath.CudaDataType.CUDA_R_64F
        elif dtype == cp.complex128:
            type = nvmath.CudaDataType.CUDA_C_64F
        else:
            raise ValueError(
                f"Unsupported data type {dtype} for matrix X. Supported types are float64 and complex128."
            )
        self.X = cudss.matrix_create_dn(
            n,  # nrows
            batchsize,  # ncols (number of RHS)
            n,  # leading dimension
            x.data.ptr,  # values (will be allocated by cuDSS)
            type,
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

    def __init__(
        self,
        matrix_type: str = None,
        view: str = None,
    ):
        """
        Initialize the cuDSS solver.

        Parameters
        ----------
        matrix_type : str, optional
            The type of the system matrix A.
        view : str, optional
            The view of the system matrix A.
        """
        if not nvmath_available:
            raise ImportError(
                "nvmath or its cudss bindings are not available. Please install them to use this solver."
            )

        self.sym_factorized = False
        self.factorized = False

        if matrix_type is not None:
            if matrix_type not in matrix_combination:
                raise ValueError(
                    f"Invalid matrix type '{matrix_type}'. Valid options are: {list(matrix_combination.keys())}"
                )
            self.M_type = matrix_combination[matrix_type]
        else:
            self.M_type = MatrixType.GENERAL
            self.M_view = MatrixViewType.FULL

        if view is not None:
            if view == "default":
                self.M_view = MatrixViewType.FULL
            elif view == "up":
                self.M_view = MatrixViewType.UPPER
            elif view == "down":
                self.M_view = MatrixViewType.LOWER
            else:
                raise ValueError(
                    f"Invalid view '{view}'. Valid options are: 'default', 'up', 'down'."
                )

        self.cudss_handle = cudss.create()
        self.cudss_config = cudss.config_create()
        self.cudss_data = cudss.data_create(self.cudss_handle)

    def analyse(self):
        """
        Perform symbolic factorization (analysis) of the system matrix A.
        """
        xp.cuda.Stream.null.synchronize()
        analysis_tic = time.perf_counter()
        cudss.execute(
            self.cudss_handle,
            cudss.Phase.ANALYSIS,
            self.cudss_config,
            self.cudss_data,
            self.A,
            self.X,
            self.B,
        )
        xp.cuda.Stream.null.synchronize()
        analysis_toc = time.perf_counter()

        return analysis_toc - analysis_tic

    def factorize(self):
        """
        Perform numeric factorization of the system matrix A.
        """
        xp.cuda.Stream.null.synchronize()
        numeric_tic = time.perf_counter()
        cudss.execute(
            self.cudss_handle,
            cudss.Phase.FACTORIZATION,
            self.cudss_config,
            self.cudss_data,
            self.A,
            self.X,
            self.B,
        )
        xp.cuda.Stream.null.synchronize()
        numeric_toc = time.perf_counter()

        return numeric_toc - numeric_tic

    def _solve(self):
        """
        Solve the linear system AX = B using the factorized form of A.
        """
        xp.cuda.Stream.null.synchronize()
        solve_tic = time.perf_counter()
        cudss.execute(
            self.cudss_handle,
            cudss.Phase.SOLVE,
            self.cudss_config,
            self.cudss_data,
            self.A,
            self.X,
            self.B,
        )
        xp.cuda.Stream.null.synchronize()
        solve_toc = time.perf_counter()

        return solve_toc - solve_tic

    def _destroy_data_wrappers(self):
        """
        Destroy the cuDSS matrix wrappers for A, B, and X to free GPU memory.
        """
        cudss.matrix_destroy(self.A)
        cudss.matrix_destroy(self.B)
        cudss.matrix_destroy(self.X)

    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_sym_fact: bool = False,
        reuse_fact: bool = False,
    ):
        """
        Solve the linear system AX = B using cuDSS.
        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix A in CSR format.
        b : NDArray
            The dense right-hand side matrix B with shape (n, batchsize).
        reuse_sym_fact : bool, optional
            Whether to reuse the symbolic factorization from a previous solve. Default is False.
        reuse_fact : bool, optional
            Whether to reuse the numeric factorization from a previous solve. Default is False.

        Return
        ------
        x : NDArray
            The dense solution matrix X with shape (n, batchsize) that satisfies AX = B
        """

        x = xp.zeros_like(b)

        self._set_A(a)
        self._set_b(b)
        self._set_x(x, b.dtype)

        if a.dtype != b.dtype:
            raise ValueError(
                f"Data type of matrix A ({a.dtype}) does not match data type of matrix B ({b.dtype}). Please ensure they have the same data type."
            )

        if not reuse_sym_fact and reuse_fact:
            raise ValueError(
                "Cannot reuse total factorization without reusing symbolic factorization."
            )

        if not reuse_sym_fact or not self.sym_factorized:
            analysis_time = self.analyse()
            if comm.rank == 0:
                print(
                    f"    CUDSS: Analysis time: {analysis_time:.6f} seconds", flush=True
                )
            self.sym_factorized = True

        if not reuse_fact or not self.factorized:
            factorization_time = self.factorize()
            if comm.rank == 0:
                print(
                    f"    CUDSS: Factorization time: {factorization_time:.6f} seconds",
                    flush=True,
                )
            self.factorized = True

        solve_time = self._solve()
        if comm.rank == 0:
            print(f"    CUDSS: Solve time: {solve_time:.6f} seconds", flush=True)

        self._destroy_data_wrappers()

        return x
