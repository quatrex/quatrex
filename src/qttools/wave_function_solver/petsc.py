# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from __future__ import annotations

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
import warnings

from qttools.utils.gpu_utils import get_array_module_name

try:

    import petsc4py
    from petsc4py import PETSc as petsc

    petsc_matrix_properties = {
        "symmetric_positive_definite": petsc.Mat.Option.SPD,
        "symmetric_indefinite": petsc.Mat.Option.SYMMETRIC,
        "hermitian_indefinite": petsc.Mat.Option.HERMITIAN,
        "general": petsc.Mat.Option.STRUCTURALLY_SYMMETRIC,
    }
    # NOTE: We set structural symmetric for complex matrices, since in
    # QTBM, the matrices are always structural symmetric. In general,
    # this could be problematic.

    # The get_config() function ignores environment variables but the
    # actual `from petsc4py import PETSc` import logic does not.
    petsc4py_config = petsc4py.get_config()
    petsc_dir = os.environ.get("PETSC_DIR", petsc4py_config.get("PETSC_DIR", None))
    petsc_arch = os.environ.get("PETSC_ARCH", petsc4py_config.get("PETSC_ARCH", None))

    try:
        libpetsc = ctypes.CDLL(f"{petsc_dir}/{petsc_arch}/lib/libpetsc.so")
    except OSError:
        # If the PETSc library is not found, we can try to load it from
        # the system library path. This may work if PETSc was installed
        # system-wide or via a package manager.
        print(
            f"Could not load PETSc library from {petsc_dir}/{petsc_arch}. "
            f"Trying to load from system library path."
        )
        libpetsc = ctypes.CDLL("libpetsc.so")

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


except (ImportError, OSError):
    petsc_available = False

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm as quatrex_comm
from qttools.comm.comm import _SubCommunicator
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()

_ALLOWED_KSP_TYPES = {"preonly"}
_ALLOWED_PC_TYPES = {"lu"}

_ALLOWED_MATRIX_VIEWS = ["full"]
_ALLOWED_MATRIX_DTYPES = [
    xp.dtype("float64"),
    xp.dtype("complex128"),
]

DEFAULT_PETSC_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist",
}


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
    if type(arr).__module__ == "cupy":
        return ctypes.c_void_p(arr.data.ptr)

    return ctypes.c_void_p(arr.ctypes.data)


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

    if dense:
        return prefix + "dense"

    if xp.__name__ != "cupy":
        return prefix + "aij"

    if xp.cuda.runtime.is_hip:
        return prefix + "aijhipsparse"

    return prefix + "aijcusparse"


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
    matrix_dtype : xp.dtype[xp.inexact]
        The data type of the system matrix. This needs to be
        provided to correctly initialize the solver.
    matrix_property : str
        The property of the system matrix. This describes properties
        like symmetry and definiteness, which can be used by solvers
        to optimize the solution process. Can be None if the solver
        does not require this information or if it can be inferred
        from the matrix itself.
    matrix_view : str
        The view of the system matrix sparsity. Not all solvers are
        compatible with all views.
    reuse_analysis : bool, optional
        Whether to reuse the analysis phase when solve is called
        multiple times, by default False. This typically includes
        symbolic factorization and ordering but can vary between
        solvers. This is useful when solving multiple linear systems
        with the same sparsity pattern but different numerical
        values.
    reuse_factorization : bool, optional
        Whether to reuse the numerical factorization when solve is
        called multiple times, by default False. This can only be
        True if reuse_analysis is also True. Note that this must
        only be True if the matrix values have not changed since the
        last factorization.
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

    @staticmethod
    def valid_views() -> list[str]:
        """Returns the valid matrix views.

        Returns
        -------
        list[str]
            The valid matrix views.

        """
        return _ALLOWED_MATRIX_VIEWS

    def __init__(
        self,
        matrix_dtype: xp.dtype[xp.inexact],
        matrix_property: str,
        matrix_view: str,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
        comm: _SubCommunicator | None = None,
        local_rows: tuple[int, int] | None = None,
        petsc_options: dict | None = None,
    ):
        """Initializes the PETSc solver."""

        # NOTE: This is a bit of hack to not need to modify the logic
        # when allocating the solver in QTBM.
        if matrix_view not in self.valid_views():
            if quatrex_comm.rank == 0:
                warnings.warn(
                    f"Matrix view '{matrix_view}' is not supported by the PETSc solver. "
                    f"Using 'full' instead."
                )
            matrix_view = "full"

        if reuse_factorization or reuse_analysis:
            if quatrex_comm.rank == 0:
                warnings.warn(
                    "Reuse of analysis or factorization is not yet supported in the PETSc solver."
                )
            reuse_factorization = False
            reuse_analysis = False

        super().__init__(
            matrix_dtype=matrix_dtype,
            matrix_property=matrix_property,
            matrix_view=matrix_view,
            reuse_analysis=reuse_analysis,
            reuse_factorization=reuse_factorization,
            comm=comm,
            local_rows=local_rows,
        )

        if not petsc_available:
            raise ImportError(
                "petsc4py is not available. Please install it to use this solver."
            )

        if matrix_property not in petsc_matrix_properties:
            raise ValueError(
                f"Invalid matrix property '{matrix_property}'. "
                f"Valid options are: {list(petsc_matrix_properties.keys())}"
            )

        if matrix_dtype not in _ALLOWED_MATRIX_DTYPES:
            raise ValueError(
                f"Invalid matrix dtype '{matrix_dtype}'. "
                f"Valid options are: {_ALLOWED_MATRIX_DTYPES}."
            )

        if petsc.ScalarType != matrix_dtype:
            raise ValueError(
                f"Requested matrix type '{matrix_dtype}' does not match "
                f"the PETSc scalar type '{petsc.ScalarType}'. Please "
                "ensure that PETSc is compiled with a matching scalar "
                "type and/or set the `PETSC_ARCH` environment variable"
                "accordingly."
            )

        self._mpi_comm = comm._mpi_comm if self._distributed else petsc.COMM_SELF

        self._sparse_mat_type = _get_petsc_mat_type(self._distributed, dense=False)
        self._dense_mat_type = _get_petsc_mat_type(self._distributed, dense=True)
        self._mproperty = petsc_matrix_properties[matrix_property]

        petsc_options = (
            dict(petsc_options) if petsc_options else DEFAULT_PETSC_OPTIONS.copy()
        )

        if petsc_options["ksp_type"] not in _ALLOWED_KSP_TYPES:
            raise ValueError(
                f"Iterative KSP type '{petsc_options['ksp_type']}' is not "
                f"supported. Use one of {sorted(_ALLOWED_KSP_TYPES)}."
            )
        if petsc_options["pc_type"] not in _ALLOWED_PC_TYPES:
            raise ValueError(
                f"PC type '{petsc_options['pc_type']}' is not a direct solver. "
                f"Use one of {sorted(_ALLOWED_PC_TYPES)}."
            )

        # NOTE: Object specific options are set using a prefix
        prefix = f"qttools_{id(self)}_"
        self._ksp = petsc.KSP().create(comm=self._mpi_comm)
        self._ksp.setOptionsPrefix(prefix)
        options = petsc.Options(prefix)
        for key, value in petsc_options.items():
            options.setValue(key, value)
        self._ksp.setFromOptions()

    def __del__(self):
        """Cleans up the PETSc solver."""
        if getattr(self, "_ksp", None) is not None:
            self._ksp.destroy()
            self._ksp = None

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
        n_local, n = a.shape

        rows = xp.repeat(xp.arange(n_local, dtype=a.indptr.dtype), xp.diff(a.indptr))

        sizes = (n, n)
        if self._local_rows is not None:
            num_local_rows = self._local_rows[1] - self._local_rows[0]
            # NOTE: PETSc requires that the row and colum distribution
            # is the same, especially since we want to get only a part
            # of the solution vector in the end. Since we feed in the
            # values along a range of rows, the setup here will probably
            # incur some communication.
            sizes = ((num_local_rows, n), (num_local_rows, n))

            # Include the rank offset.
            rows += self._local_rows[0]

        mat = petsc.Mat().create(comm=self._mpi_comm)
        mat.setSizes(sizes)
        mat.setType(self._sparse_mat_type)

        if self._mproperty is not None:
            mat.setOption(self._mproperty, True)

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

        mat.assemble()

        return mat

    def _create_petsc_array(self, arr: NDArray) -> petsc.Mat:
        """Creates a dense PETSc matrix from an array.

        Parameters
        ----------
        arr : NDArray
            The array to convert to a PETSc matrix.

        """
        sizes = arr.shape
        if self._distributed:
            num_local_rows = self._local_rows[1] - self._local_rows[0]
            if sizes[0] != num_local_rows:
                raise ValueError(
                    f"Local array shape {sizes} does not match the "
                    f"expected local row count {num_local_rows}."
                )
            sizes = ((num_local_rows, petsc.DETERMINE), (petsc.DECIDE, sizes[1]))

        mat = petsc.Mat().create(comm=self._mpi_comm)
        mat.setSizes(sizes)
        mat.setType(self._dense_mat_type)

        if self._distributed:
            libpetsc.MatMPIDenseSetPreallocation(mat.handle, _get_data_pointer(arr))
        else:
            libpetsc.MatSeqDenseSetPreallocation(mat.handle, _get_data_pointer(arr))

        mat.assemble()

        return mat

    @profiler.profile("PETSc solve", level="default")
    def solve(
        self,
        a: sparse.csr_matrix,
        b: NDArray,
    ):
        """Solves the sparse linear system a @ x = b using PETSc.

        Note
        ----
        PETSc only accepts right-hand side arrays on the host. If the
        input array is a CuPy array, it will be transferred to the host
        before solving. The solution will also be returned in the same
        format as the input array (i.e., if the input is a CuPy array,
        the solution will be a CuPy array).

        Parameters
        ----------
        a : sparse.csr_matrix
            The sparse system matrix in CSR format.
        b : NDArray
            The dense right-hand side array with shape (n, batchsize).

        Returns
        -------
        x : NDArray
            The solution array with shape (n, batchsize).

        """
        if (a.dtype != petsc.ScalarType) or (a.dtype != self.matrix_dtype):
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
        if not sparse.isspmatrix_csr(a):
            raise ValueError("Matrix a must be in CSR format.")
        if a.indptr.dtype != petsc.IntType:
            raise ValueError(
                f"Data type of a.indptr ({a.indptr.dtype}) does not match "
                f"PETSc int type ({petsc.IntType})."
            )
        if a.indices.dtype != petsc.IntType:
            raise ValueError(
                f"Data type of a.indices ({a.indices.dtype}) does not match "
                f"PETSc int type ({petsc.IntType})."
            )

        # TODO: Currently, we need to copy since we allow petsc to
        # modify the matrix in-place.
        a = a.copy()

        if not xp.isfortran(b):
            b = xp.asfortranarray(b)

        if b.ndim == 1:
            b = b[:, xp.newaxis]

        transferred = False
        if get_array_module_name(b) == "cupy":
            # NOTE: PETSc cannot handle CuPy arrays for the RHS and the
            # solution directly, so we need to transfer the data to
            # NumPy arrays.
            b = np.asfortranarray(b.get())
            transferred = True

        x = np.empty_like(b, order="F")

        matrix = self._create_petsc_csr(a)
        rhs = self._create_petsc_array(b)
        solution = self._create_petsc_array(x)

        self._ksp.setOperators(matrix)

        self._ksp.matSolve(rhs, solution)

        rhs.destroy()
        solution.destroy()
        matrix.destroy()
        self._ksp.reset()

        if transferred:
            # NOTE: Transfer the solution back to the original CuPy
            # array.
            x = xp.asarray(x)

        return x
