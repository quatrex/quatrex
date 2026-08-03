# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


try:

    import petsc4py
    from petsc4py import PETSc as petsc

    petsc_matrix_types = {
        "real_symmetric_positive_definite": petsc.Mat.Option.SPD,
        "real_symmetric_indefinite": petsc.Mat.Option.SYMMETRIC,
        "complex_hermitian_indefinite": petsc.Mat.Option.HERMITIAN,
        "real_nonsymmetric": None,
        "complex_nonsymmetric": None,
    }
    # TODO: Default should probably be structurally_symmetric?

    # NOTE: petsc4py does not provide a direct way to access the
    # underlying PETSc C library, so we use ctypes to load it manually.
    # Specifically, we want to access the `MatSetPreallocationCOO` and
    # `MatSetValuesCOO` functions, which allow us to set up a matrix in
    # COO format directly on the GPU. Other construction methods (e.g.,
    # `MatSetValues`) would require transferring the matrix to the CPU
    # first, and then back to the GPU, which is not so nice. Same story
    # for `MatCreateXXXAIJWithArrays`.
    import ctypes
    import os

    # The get_config() function ignores environment variables but the
    # actual `from petsc4py import PETSc` import logic does not.
    petsc4py_config = petsc4py.get_config()
    petsc_dir = petsc4py_config.get("PETSC_DIR", None)
    petsc_arch = os.environ.get("PETSC_ARCH", petsc4py_config.get("PETSC_ARCH", None))
    libpetsc = ctypes.CDLL(f"{petsc_dir}/{petsc_arch}/lib/libpetsc.so")
    libpetsc.MatSetPreallocationCOO.argtypes = [
        ctypes.c_void_p,  # Mat A
        ctypes.c_longlong,  # PetscCount ncoo
        ctypes.c_void_p,  # PetscInt coo_i[]
        ctypes.c_void_p,  # PetscInt coo_j[]
    ]
    libpetsc.MatSetPreallocationCOO.restype = ctypes.c_int  # PetscErrorCode

    libpetsc.MatSetValuesCOO.argtypes = [
        ctypes.c_void_p,  # Mat A
        ctypes.c_void_p,  # const PetscScalar coo_v[]
        ctypes.c_int,  # InsertMode imode
    ]
    libpetsc.MatSetValuesCOO.restype = ctypes.c_int  # PetscErrorCode

    # NOTE: To have a copy-free allocation of the rhs and the solution
    # vector on the GPU, we need to use another PETSc function that is
    # not exposed in petsc4py.
    libpetsc.MatSeqDenseSetPreallocation.argtypes = [
        ctypes.c_void_p,  # Mat B
        ctypes.c_void_p,  # PetscScalar data[]
    ]
    libpetsc.MatSeqDenseSetPreallocation.restype = ctypes.c_int  # PetscErrorCode

    libpetsc.MatMPIDenseSetPreallocation.argtypes = [
        ctypes.c_void_p,  # Mat B
        ctypes.c_void_p,  # PetscScalar *data
    ]
    libpetsc.MatMPIDenseSetPreallocation.restype = ctypes.c_int  # PetscErrorCode

    petsc_available = True


except ImportError:
    petsc_available = False

from qttools import NDArray, sparse, xp
from qttools.comm.comm import _SubCommunicator
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


def _get_data_pointer(arr: NDArray) -> ctypes.c_void_p:
    """Returns a ctypes pointer to the data of an array.

    This function handles both CuPy and NumPy arrays, returning a
    ctypes pointer to the underlying data buffer.

    Parameters
    ----------
    arr : NDArray
        The array to get the data pointer from.

    Returns
    -------
    ctypes.c_void_p
        A ctypes pointer to the data of the array.

    """
    if xp.__name__ == "cupy":
        return ctypes.c_void_p(arr.data.ptr)

    return ctypes.c_void_p(arr.ctypes.data)


# Maps the combination of (backend, dense) to the appropriate PETSc
# matrix type suffix.
_petsc_mat_type_suffixes = {
    ("cpu", False): "aij",
    ("cpu", True): "dense",
    ("cuda", False): "aijcusparse",
    ("cuda", True): "densecuda",
    ("hip", False): "aijhipsparse",
    ("hip", True): "densehip",
}


def _get_petsc_mat_type(distributed: bool, dense: bool) -> str:
    """Resolves PETSc matrix type based on the current configuration.

    Checks context in terms of distribted/sequential, dense/sparse, and
    GPU/CPU backend to return the appropriate PETSc matrix type string.

    Parameters
    ----------
    distributed : bool
        Whether the matrix is distributed.
    dense : bool, optional
        Whether to return a dense matrix type. If False, a sparse
        matrix type will be returned. Default is False.

    Returns
    -------
    str
        The PETSc matrix type string.

    """
    prefix = "mpi" if distributed else "seq"

    if xp.__name__ != "cupy":
        backend = "cpu"
    elif xp.cuda.runtime.is_hip:
        backend = "hip"
    else:
        backend = "cuda"

    return prefix + _petsc_mat_type_suffixes[backend, dense]


class PETSc(WFSolver):
    """Wavefunction solver using PETSc for sparse matrix solving.

    For distributed solves, the user must provide a communicator and the
    local row distribution. The local row distribution is specified as a
    tuple of the form (start_row, end_row) for each process.

    Note that PETSc assumes that vectors always have a contiguous range
    of vector entries stored on each MPI rank. Rank 0 will always own
    the first `local_rows[1]` entries, rank 1 will own the next
    `local_rows[1] - local_rows[0]` entries, and so on. The user must
    ensure that the local row distribution is consistent with this
    assumption.

    Note
    ----
    petsc4py can be built against multiple different architectures. The
    specific PETSc architecture to be used at runtime can be set using
    the `PETSC_ARCH` environment variable.

    Parameters
    ----------
    matrix_type : str, optional
        The type of the system matrix. This describes properties like
        symmetry and definiteness. If None, the solver will use a
        general matrix type.
    matrix_view : str, optional
        The view of the system matrix sparsity. This solver supports
        'full' and 'upper' views. If None, the solver will use the PETSc
        default for the given `matrix_type`.
    comm : _SubCommunicator, optional
        The communicator for distributed solves. If None, the solver
        will assume a single-rank solve. This must be provided together
        with local_rows.
    local_rows : tuple, optional
        A tuple specifying the local row distribution for distributed
        solves. If None, the solver will assume a single-rank solve.
    petsc_options : dict, optional
        A dictionary of PETSc options to set. If None, the solver will
        use the default PETSc options. This can be used to set options
        like the solver type, preconditioner, and tolerances. To select
        an external direct solver, something like `{ksp_type: "preonly",
        "pc_type": "lu", "pc_factor_mat_solver_type": "superlu_dist"}`
        can be used. See the PETSc documentation for more details on
        available options.

    """

    def __init__(
        self,
        matrix_type: str | None = None,
        matrix_view: str | None = None,
        comm: _SubCommunicator | None = None,
        local_rows: tuple[int, int] | None = None,
        petsc_options: dict | None = None,
    ):
        """Initializes the PETSc solver."""
        if not petsc_available:
            raise ImportError(
                "petsc4py is not available. Please install it to use this solver."
            )

        if matrix_type is not None and matrix_type not in petsc_matrix_types:
            raise ValueError(
                f"Invalid matrix type '{matrix_type}'. "
                f"Valid options are: {list(petsc_matrix_types.keys())}"
            )
        if matrix_view not in [None, "full", "upper"]:
            raise ValueError(
                f"Invalid view '{matrix_view}'. "
                "Valid options are: None, 'full', 'upper'."
            )

        if (
            "real" in matrix_type
            and petsc.ScalarType != xp.float64
            or "complex" in matrix_type
            and petsc.ScalarType != xp.complex128
        ):
            raise ValueError(
                f"Requested matrix type '{matrix_type}' does not match "
                f"the PETSc scalar type '{petsc.ScalarType}'. Please "
                "ensure that PETSc is compiled with a matching scalar "
                "type and/or set the `PETSC_ARCH` environment variable"
                "accordingly."
            )

        # Comm and local_rows must be provided together or not at all.
        if (comm is None) != (local_rows is None):
            raise ValueError(
                "Both 'comm' and 'local_rows' must be provided together or not at all."
            )

        distributed = comm is not None and local_rows is not None

        self.comm = comm._mpi_comm if distributed else petsc.COMM_SELF
        self.local_rows = local_rows

        self._sparse_mat_type = _get_petsc_mat_type(distributed, dense=False)
        self._dense_mat_type = _get_petsc_mat_type(distributed, dense=True)

        options = petsc.Options()
        if petsc_options is not None:
            for key, value in petsc_options.items():
                options.setValue(key, value)

        self._ksp = petsc.KSP().create(comm=self.comm)
        self._ksp.setFromOptions()

    def _create_petsc_csr(self, a: sparse.csr_matrix) -> petsc.Mat:
        """Creates a PETSc matrix from a CSR matrix.

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse matrix in CSR format to convert to a PETSc
            matrix.

        Returns
        -------
        petsc.Mat
            The PETSc matrix corresponding to the input CSR matrix.

        """
        # NOTE: We assume that the matrix is always square and in the
        # case of a distributed matrix, we partition the matrix along
        # the rows.
        n = a.shape[1]

        rows = xp.repeat(xp.arange(n, dtype=xp.int32), xp.diff(a.indptr))

        sizes = (n, n)
        if self.local_rows is not None:
            num_local_rows = self.local_rows[1] - self.local_rows[0]
            # NOTE: PETSc requires that the row and colum distribution
            # is the same, especially since we want to get only a part
            # of the solution vector in the end. Since we feed in the
            # values along a range of rows, the setup here will probably
            # incur some communication.
            sizes = ((num_local_rows, n), (num_local_rows, n))

            # Include the rank offset.
            rows += self.local_rows[0]

        mat = petsc.Mat().create(comm=self.comm)
        mat.setSizes(sizes)
        mat.setType(self._sparse_mat_type)

        libpetsc.MatSetPreallocationCOO(
            mat.handle,
            a.data.size,
            _get_data_pointer(rows),
            _get_data_pointer(a.indices),
        )

        # TODO: Later one will just have to update the `data` buffer
        # in-place, and call MatSetValuesCOO again.
        libpetsc.MatSetValuesCOO(
            mat.handle,
            _get_data_pointer(a.data),
            petsc.InsertMode.INSERT_VALUES,
        )

        return mat

    def _create_petsc_array(self, arr: NDArray) -> petsc.Mat:
        """Creates a dense PETSc matrix from an array.

        Parameters
        ----------
        arr : NDArray
            The array to convert to a PETSc matrix.

        """
        sizes = arr.shape
        if self.local_rows is not None:
            num_local_rows = self.local_rows[1] - self.local_rows[0]
            sizes = ((num_local_rows, petsc.DETERMINE), (arr.shape[1], arr.shape[1]))

        mat = petsc.Mat().create(comm=self.comm)
        mat.setSizes(sizes)
        mat.setType(self._dense_mat_type)

        if self.local_rows is not None:
            libpetsc.MatMPIDenseSetPreallocation(mat.handle, _get_data_pointer(arr))
        else:
            libpetsc.MatSeqDenseSetPreallocation(mat.handle, _get_data_pointer(arr))

        return mat

    @profiler.profile("PETSc solve", level="default")
    def solve(
        self,
        a: sparse.csr_matrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
    ):
        """Solves the sparse linear system a @ x = b using PETSc.

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
        if a.dtype != petsc.ScalarType:
            raise ValueError(
                f"Data type of a ({a.dtype}) does not match PETSc "
                f"scalar type ({petsc.ScalarType})."
            )
        if b.dtype != petsc.ScalarType:
            raise ValueError(
                f"Data type of b ({b.dtype}) does not match PETSc "
                f"scalar type ({petsc.ScalarType})."
            )
        if a.dtype != b.dtype:
            raise ValueError(
                f"Data type of a ({a.dtype}) does not match data type "
                f"of b ({b.dtype})."
            )

        x = xp.zeros_like(b)

        matrix = self._create_petsc_csr(a)
        rhs = self._create_petsc_array(b)
        solution = self._create_petsc_array(x)

        if reuse_analysis:
            self._ksp.pc.setReusePreconditioner(True)

        if not reuse_factorization:
            self._ksp.reset()
            self._ksp.setOperators(matrix)

        self._ksp.matSolve(rhs, solution)

        rhs.destroy()
        solution.destroy()

        return x
