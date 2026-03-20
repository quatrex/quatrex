# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

try:
    import nvmath
    from nvmath.bindings import cudss
    from nvmath.bindings.cudss import MatrixType, MatrixViewType

    nvmath_available = True

    matrix_combination = {
        "real_symmetric_positive_definite": MatrixType.SYMMETRIC,
        "real_symmetric_indefinite": MatrixType.GENERAL,
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
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


class cuDSS(WFSolver):

    def _set_A(self, a: sparse.csr_matrix):
        n = a.shape[0]
        nnz = a.nnz
        self.A = cudss.matrix_create_csr(
            n,  # nrows
            n,  # ncols
            nnz,  # nnz
            a.indptr.data.ptr,  # row_start (beginning of row offset array)
            0,  # row_end (NULL/0 - not used in standard CSR)
            a.indices.data.ptr,  # column indices
            a.data.data.ptr,  # values
            nvmath.CudaDataType.CUDA_R_32I,  # index type (int32)
            nvmath.CudaDataType.CUDA_C_64F,  # value type (complex128)
            self.M_type,  # matrix type (general)
            self.M_view,  # matrix view (full)
            cudss.IndexBase.ZERO,  # 0-based indexing
        )

    def _set_b(self, b: NDArray):
        n = b.shape[0]
        batchsize = b.shape[1]
        self.B = cudss.matrix_create_dn(
            n,  # nrows
            batchsize,  # ncols (number of RHS)
            n,  # leading dimension
            b.data.ptr,  # values
            nvmath.CudaDataType.CUDA_C_64F,  # complex128
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

    def _set_x(self, x: NDArray):
        n = x.shape[0]
        batchsize = x.shape[1]
        self.X = cudss.matrix_create_dn(
            n,  # nrows
            batchsize,  # ncols (number of RHS)
            n,  # leading dimension
            x.data.ptr,  # values (will be allocated by cuDSS)
            nvmath.CudaDataType.CUDA_C_64F,  # complex128
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

    def __init__(
        self,
        matrix_type: str = None,
    ):
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
            self.M_view = MatrixViewType.UPPER

        self.cudss_handle = cudss.create()
        self.cudss_config = cudss.config_create()
        self.cudss_data = cudss.data_create(self.cudss_handle)

    def analyse(self):
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

    @profiler.profile(level="api")
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_sym_fact: bool = False,
        reuse_fact: bool = False,
    ):

        x = xp.zeros_like(b)

        self._set_A(a)
        self._set_b(b)
        self._set_x(x)

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

        return x
