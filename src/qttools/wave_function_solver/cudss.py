# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the cuDSS wave function solver."""

try:
    import ctypes

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

    NAME_LEN = 64

    ALLOC = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    FREE = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p
    )

    class DeviceMemHandler(ctypes.Structure):
        _fields_ = [
            ("ctx", ctypes.c_void_p),
            ("device_alloc", ALLOC),
            ("device_free", FREE),
            ("name", ctypes.c_char * NAME_LEN),
        ]

    _live = {}

    def _alloc(ctx, ptr_out, size, stream):
        try:
            with cp.cuda.ExternalStream(stream or 0):
                mem = cp.cuda.alloc(size)
            _live[mem.ptr] = mem
            ptr_out[0] = mem.ptr
            return 0
        except Exception:
            return 1

    def _free(ctx, ptr, size, stream):
        _live.pop(ptr, None)
        return 0

    # If the callbacks are garbage collected, cuDSS will crash.
    _alloc_cb, _free_cb = ALLOC(_alloc), FREE(_free)
    _handler = DeviceMemHandler(None, _alloc_cb, _free_cb, b"cupy_pool")
    cudss_available = True


except ImportError:
    cudss_available = False

import os

import numpy as np
from mpi4py import MPI

from qttools import NDArray, sparse
from qttools.comm import comm as quatrex_comm
from qttools.comm.comm import _SubCommunicator
from qttools.profiling import Profiler
from qttools.profiling.profiler import QTX_PROFILE_LEVEL
from qttools.utils.gpu_utils import get_array_module_name, synchronize_current_stream
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


class cuDSS(WFSolver):
    """Wavefunction solver using NVIDIA's cuDSS library for sparse
    direct solves on GPUs.

    For distributed solves, the user must provide a communicator and the
    local row distribution. The communicator must be compatible with the
    cuDSS communication layer, which can be set via the CUDSS_COMM_LIB
    environment variable. The local row distribution is specified as a
    tuple of the form (`start_row`, `end_row`) for each process.

    Note that even though cuDSS MGMN mode takes end_row to be inclusive,
    we use the exclusive convention in this solver to be consistent with
    Python slicing.

    The solver also supports multithreading via the cuDSS threading
    layer, which can be set via the CUDSS_THREADING_LIB environment
    variable.

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
        will assume a single-GPU solve. This must be provided together
        with local_rows.
    local_rows : tuple, optional
        A tuple specifying the local row distribution for distributed
        solves. If None, the solver will assume a single-GPU solve.

    """

    def __init__(
        self,
        matrix_type: str | None = None,
        matrix_view: str | None = None,
        comm: _SubCommunicator | None = None,
        local_rows: tuple | None = None,
    ):
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
        # Set the cudss memory handler to use CuPy's memory pool for
        # device allocations.
        cudss.set_device_mem_handler(self._solver_handle, ctypes.addressof(_handler))

        self._solver_config = cudss.config_create()
        self._solver_data = cudss.data_create(self._solver_handle)

        self.analyzed = False
        self.factorized = False
        self._matrix_handle = None
        self._comm = comm

        self._indptr_ptr = None
        self._indices_ptr = None
        self._local_rows = None

        # Comm and local_rows must be provided together or not at all.
        if (comm is None) != (local_rows is None):
            raise ValueError(
                "Both 'comm' and 'local_rows' must be provided together or not at all."
            )

        if local_rows is not None:
            start, stop = local_rows
            # NOTE: cuDSS uses inclusive end row
            self._local_rows = (int(start), int(stop) - 1)

        if comm is not None:
            comm_lib = os.getenv("CUDSS_COMM_LIB")

            if comm_lib is None:
                raise ValueError(
                    "CUDSS_COMM_LIB environment variable is not set. "
                    "Please set it to the path to your "
                    "'libcudss_commlayer_<mpi|nccl>.so' shared library."
                )

            # Set up communication layer.
            cudss.set_comm_layer(self._solver_handle, comm_lib)

            # Keep the communicator alive to avoid garbage collection of
            # the underlying MPI communicator.
            self._mpi_comm = comm._mpi_comm
            host_comm_address = MPI._addressof(comm._mpi_comm)
            host_comm_size = MPI._sizeof(MPI.Comm)
            cudss.data_set(
                self._solver_handle,
                self._solver_data,
                cudss.DataParam.COMM_HOST,
                host_comm_address,
                size_in_bytes=host_comm_size,
            )

            if "openmpi" in comm_lib.lower() or "mpich" in comm_lib.lower():
                device_comm_address = host_comm_address
                device_comm_size = host_comm_size

            elif "nccl" in comm_lib.lower():
                if not hasattr(comm, "_nccl_comm"):
                    raise ValueError(
                        "The provided communicator does not have an NCCL communicator. "
                        "Please set the backend to NCCL."
                    )
                # Keep this pointer alive to avoid garbage collection of
                # the underlying NCCL communicator.
                self._nccl_comm = ctypes.c_void_p(comm._nccl_comm._comm.comm)
                device_comm_address = ctypes.addressof(self._nccl_comm)
                device_comm_size = ctypes.sizeof(self._nccl_comm)
            else:
                raise ValueError(
                    f"Unsupported communication library '{comm_lib}'. "
                    "Please use a library that contains 'openmpi', 'mpich', or 'nccl' in its name."
                )

            cudss.data_set(
                self._solver_handle,
                self._solver_data,
                cudss.DataParam.COMM_DEVICE,
                device_comm_address,
                size_in_bytes=device_comm_size,
            )

        threading_lib = os.getenv("CUDSS_THREADING_LIB")
        if threading_lib is not None:
            cudss.set_threading_layer(self._solver_handle, threading_lib)

    def close(self):
        """Frees all cuDSS solver handles and internal data structures."""
        if getattr(self, "_matrix_handle", None) is not None:
            cudss.matrix_destroy(self._matrix_handle)
            self._matrix_handle = None
        if hasattr(self, "_solver_data") and self._solver_data is not None:
            cudss.data_destroy(self._solver_handle, self._solver_data)
            self._solver_data = None
        if hasattr(self, "_solver_config") and self._solver_config is not None:
            cudss.config_destroy(self._solver_config)
            self._solver_config = None
        if hasattr(self, "_solver_handle") and self._solver_handle is not None:
            cudss.destroy(self._solver_handle)
            self._solver_handle = None

    def __del__(self):
        self.close()

    def _create_cudss_csr(self, a: sparse.csr_matrix) -> int:
        """Creates a new cuDSS CSR descriptor (see _update_cudss_csr
        for reuse).

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

        # NOTE: We assume that the matrix is always square and in the
        # case of a distributed matrix, we partition the matrix along
        # the rows.
        nrows = ncols = a.shape[1]

        nnz = a.nnz
        # TODO: Official documentation says nnz should be global.
        # if self._comm is not None:
        #     nnz = self._comm._mpi_comm.allreduce(nnz)

        csr_handle = cudss.matrix_create_csr(
            nrows=nrows,
            ncols=ncols,
            nnz=nnz,
            row_start=a.indptr.data.ptr,
            row_end=0,
            col_indices=a.indices.data.ptr,
            values=a.data.data.ptr,
            offset_type=nvmath.CudaDataType.CUDA_R_32I,
            index_type=nvmath.CudaDataType.CUDA_R_32I,
            value_type=value_type,
            mtype=self._mtype,
            mview=self._mview,
            index_base=cudss.IndexBase.ZERO,
        )

        if self._local_rows is not None:
            cudss.matrix_set_distribution_row1d(csr_handle, *self._local_rows)

        return csr_handle

    def _update_cudss_csr(self, a: sparse.csr_matrix):
        """Updates the cuDSS CSR descriptor with the new matrix data.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix in CSR format.

        """
        cudss.matrix_set_csr_pointers(
            self._matrix_handle,
            a.indptr.data.ptr,
            0,
            a.indices.data.ptr,
            a.data.data.ptr,
        )

    def _create_cudss_array(self, arr: NDArray, nrows_global: int) -> int:
        """Create a cuDSS wrapper for a dense array.

        Used for the right-hand side and solution.

        Parameters
        ----------
        arr : NDArray
            The dense array for which to create the cuDSS wrapper.
        nrows_global : int
            The global number of rows of the array, which is required
            for distributed solves. If arr is a local slice of a
            distributed array, this should be the total number of rows
            in the global array.

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
            nrows=nrows_global,  # global number of rows
            ncols=arr.shape[1],  # global number of columns
            ld=arr.shape[0],  # local (!) leading dimension
            values=arr.data.ptr,
            value_type=value_type,
            layout=cudss.Layout.COL_MAJOR,  # Fortran order
        )

        if self._local_rows is not None:
            cudss.matrix_set_distribution_row1d(array_handle, *self._local_rows)

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
        # NOTE: We want to profile inside the method to be able to set
        # the communicator.
        with profiler.profile_range(
            label="cuDSS: analysis", level="default", comm=self._comm
        ):
            self._execute_phase(cudss.Phase.ANALYSIS, matrix, solution, rhs)

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
        with profiler.profile_range(
            label="cuDSS: factorization", level="default", comm=self._comm
        ):
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
        with profiler.profile_range(
            label="cuDSS: solve", level="default", comm=self._comm
        ):
            self._execute_phase(cudss.Phase.SOLVE, matrix, solution, rhs)

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
        with profiler.profile_range(
            label="cuDSS: solve", level="default", comm=self._comm
        ):

            if reuse_factorization and not reuse_analysis:
                raise ValueError(
                    "Cannot reuse total factorization without reusing symbolic factorization."
                )
            if a.dtype != b.dtype:
                raise ValueError(
                    f"Data type of a ({a.dtype}) does not match data type of b ({b.dtype}). "
                    "Please ensure they have the same data type."
                )

            b_shape = b.shape

            if b.ndim == 1:
                b = b.reshape(-1, 1)
            elif b.ndim > 2:
                raise ValueError(
                    f"Right-hand side b has invalid number of dimensions {b.ndim}. "
                    "Expected 1 or 2 dimensions."
                )

            if get_array_module_name(b) != "cupy":
                raise ValueError(
                    "Right-hand side b must be a CuPy array on the GPU. "
                    "Please transfer it to the GPU before calling this method."
                )
            if get_array_module_name(a) != "cupyx":
                raise ValueError(
                    "System matrix a must be a CuPy sparse CSR matrix on the GPU. "
                    "Please transfer it to the GPU before calling this method."
                )

            # b needs to be fortran contiguous for cuDSS
            b = cp.asfortranarray(b)

            x = cp.zeros_like(b)
            x = cp.asfortranarray(x)

            # Set up the linear system.
            # Dense descriptors are cheap and their column count changes, so
            # they are recreated on every call.
            if reuse_analysis:
                # It seems for reuse of analysis only indices need to stay
                # at the same memory location. The data can be updated.
                if self._indptr_ptr is None:
                    self._indptr_ptr = a.indptr.data.ptr
                elif self._indptr_ptr != a.indptr.data.ptr:
                    raise ValueError(
                        "Cannot reuse analysis with a different matrix row pointer. "
                        "Please ensure that the matrix row pointer has not changed since the last analysis."
                    )
                if self._indices_ptr is None:
                    self._indices_ptr = a.indices.data.ptr
                elif self._indices_ptr != a.indices.data.ptr:
                    raise ValueError(
                        "Cannot reuse analysis with a different matrix column indices pointer. "
                        "Please ensure that the matrix column indices have not changed since the last analysis."
                    )

            # NOTE: Check if we need to redo analysis if the number
            # of right-hand sides has changed.
            local_redo = (
                not reuse_analysis or not self.analyzed or self._matrix_handle is None
            )
            if self._comm is not None:
                redo_analysis = bool(
                    self._comm._mpi_comm.allreduce(int(local_redo), op=MPI.MAX)
                )

                # Print a warning when local and global disagree
                if redo_analysis != local_redo:
                    print(
                        f"Warning: local redo_analysis ({local_redo}) "
                        f"does not match global redo_analysis ({redo_analysis}).\n"
                        "This may indicate a mismatch in the local row distribution\n"
                        "or a change in the number of right-hand sides across processes.\n",
                        "On process: ",
                        quatrex_comm.rank,
                        flush=True,
                    )

            else:
                redo_analysis = local_redo

            _solution_handle = _rhs_handle = None
            try:
                _solution_handle = self._create_cudss_array(x, nrows_global=a.shape[1])
                _rhs_handle = self._create_cudss_array(b, nrows_global=a.shape[1])

                if redo_analysis:
                    if self._matrix_handle is not None:
                        cudss.matrix_destroy(self._matrix_handle)
                        self._matrix_handle = None
                    self.analyzed = False
                    self.factorized = False
                    self._matrix_handle = self._create_cudss_csr(a)
                    self._analyze(self._matrix_handle, _solution_handle, _rhs_handle)
                    self.analyzed = True
                else:
                    # NOTE: Unsure here since it seems we need to keep the
                    # pointers to the row and column indices the same for cuDSS
                    # to reuse the analysis. Seems the update does not do what
                    # we would expect.
                    self._update_cudss_csr(a)

                if QTX_PROFILE_LEVEL == "debug":
                    mem_estimates = np.zeros(16, dtype=np.int64)
                    size_written = np.zeros(1, dtype=np.uint64)
                    cudss.data_get(
                        self._solver_handle,
                        self._solver_data,
                        cudss.DataParam.MEMORY_ESTIMATES,
                        mem_estimates.ctypes.data,
                        mem_estimates.nbytes,
                        size_written.ctypes.data,
                    )
                    message = (
                        f"\n[Memory, rank {quatrex_comm.rank}]: cudss memory estimates"
                    )
                    message += "\n" + "-" * 80
                    message += f"\nPermanent Device Memory: {mem_estimates[0] / 1e9} GB"
                    message += f"\nPeak Device Memory: {mem_estimates[1] / 1e9} GB"
                    message += f"\nPermanent Host Memory: {mem_estimates[2] / 1e9} GB"
                    message += f"\nPeak Host Memory: {mem_estimates[3] / 1e9} GB"
                    message += "\n" + "-" * 80 + "\n"
                    print(message, flush=True)

                if redo_analysis or not self.factorized or not reuse_factorization:
                    self._factorize(self._matrix_handle, _solution_handle, _rhs_handle)
                    self.factorized = True

                self._solve(self._matrix_handle, _solution_handle, _rhs_handle)

            finally:
                for handle in [_solution_handle, _rhs_handle]:
                    if handle is not None:
                        cudss.matrix_destroy(handle)

            return x.reshape(b_shape)
