# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

# TODO: LLM generated, refactor and cleanup.

"""Python binding for the distributed STRUMPACK sparse solver."""

from __future__ import annotations

import ctypes
import os
import pathlib
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from mpi4py import MPI

from qttools import NDArray, sparse, xp
from qttools.comm import comm as quatrex_comm
from qttools.comm.comm import _SubCommunicator
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_array_module_name
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()

# Modes understood by the shim (see strumpack_shim.c).
_QT_DOFACT, _QT_SAMEPATTERN, _QT_FACTORED = 0, 1, 2
_ERR_PATTERN, _ERR_NOFACTOR, _ERR_ALLOC, _ERR_DIST, _ERR_OVERFLOW = (
    -1001,
    -1002,
    -1003,
    -1004,
    -1005,
)
# Positive return codes are STRUMPACK_RETURN_CODE values.
_STRUMPACK_CODES = {
    1: "MATRIX_NOT_SET",
    2: "REORDERING_ERROR",
    3: "ZERO_PIVOT",
    4: "NO_CONVERGENCE",
    5: "INACCURATE_INERTIA",
}


def _load_shim() -> ctypes.CDLL | None:
    """Load the shim library (see strumpack_shim.c)."""
    candidates = []
    env = os.environ.get("QTTOOLS_STRUMPACK_SHIM")
    if env:
        candidates.append(env)
    for name in ["libqt_strumpack_shim.so"]:
        candidates.append(str(pathlib.Path(__file__).with_name(name)))
        candidates.append(name)

    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue

        lib.qt_create.argtypes = [
            ctypes.c_int,  # is_complex
            ctypes.c_int,  # use64 (64-bit indices)
            ctypes.c_int,  # Fortran MPI communicator
            ctypes.c_char_p,  # STRUMPACK command line flags
        ]
        lib.qt_create.restype = ctypes.c_void_p  # pointer, not int!

        lib.qt_solve.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_int,  # mode
            ctypes.c_longlong,  # n
            ctypes.c_longlong,  # m_loc
            ctypes.c_longlong,  # fst_row
            ctypes.c_longlong,  # nnz_loc
            ctypes.c_void_p,  # nzval
            ctypes.c_void_p,  # colind
            ctypes.c_void_p,  # rowptr
            ctypes.c_void_p,  # b (input)
            ctypes.c_void_p,  # x (output)
            ctypes.c_int,  # ld
            ctypes.c_int,  # nrhs
        ]
        lib.qt_solve.restype = ctypes.c_int

        lib.qt_destroy.argtypes = [ctypes.c_void_p]
        lib.qt_destroy.restype = None

        lib.qt_set_option.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        lib.qt_set_option.restype = ctypes.c_int
        return lib
    return None


_lib = _load_shim()
strumpack_available = _lib is not None


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.ctypes.data)


class Strumpack(WFSolver):
    """Wavefunction solver using the distributed STRUMPACK sparse solver.

    Supports real and complex systems. Index width
    (32 or 64 bit) is chosen at runtime`.

    Parameters
    ----------
    matrix_dtype : xp.dtype
        Must be float64 or complex128.
    matrix_property : str
        Matrix property (see ``WFSolver``). The solver does a general sparse
        LU, so this is only validated, not exploited.
    matrix_view : str
        Must be "full".
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
        will assume a single-GPU solve. This must be provided together
        with local_rows.
    local_rows : tuple, optional
        A tuple specifying the local row distribution for distributed
        solves. If None, the solver will assume a single-GPU solve.
    strumpack_options : mapping, optional
        STRUMPACK options to set on the solver. Values may be bools, ints,
        floats or enum names (case-insensitive). Supported names:

        - choices: Compression (NONE, HSS, BLR, HODLR, BLR_HODLR,
          ZFP_BLR_HODLR, LOSSLESS, LOSSY), ReorderingMethod (METIS, PARMETIS,
          SCOTCH, PTSCOTCH, RCM, GEOMETRIC, AMD, MMD, AND, MLF, SPECTRAL,
          NATURAL), Matching, KrylovSolver (AUTO, DIRECT, REFINE, PREC_GMRES,
          GMRES, PREC_BICGSTAB, BICGSTAB), GramSchmidt
        - booleans: UseGPU, MetisNodeNDP, SymmetricPattern, UseMatsolve
        - integers: MaxIt, GMRESRestart, NDParam, CompressionMinSepSize,
          CompressionMinFrontSize, CompressionLeafSize,
          CompressionButterflyLevels, CompressionLossyPrecision
        - floats: RelTol, AbsTol, CompressionRelTol, CompressionAbsTol

        ``UseMatsolve=True`` solves all right-hand sides in one
        ``STRUMPACK_matsolve`` call instead of one ``STRUMPACK_solve`` per
        column (faster for many RHS; check it works with your STRUMPACK build).
    strumpack_args : str or sequence of str, optional
        Raw STRUMPACK command line flags (e.g. ``"--sp_enable_gpu"``), parsed
        before ``strumpack_options`` are applied (which take precedence).

    """

    def __init__(
        self,
        matrix_dtype,
        matrix_property: str,
        matrix_view: str,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
        comm: _SubCommunicator | None = None,
        local_rows: tuple | None = None,
        strumpack_options: Mapping[str, int | float | str | bool] | None = None,
        strumpack_args: str | Sequence[str] | None = None,
    ):
        if not strumpack_available:
            raise ImportError("The STRUMPACK shim library is not available.")

        if matrix_view not in self.valid_views():
            if quatrex_comm.rank == 0:
                warnings.warn(
                    f"Matrix view '{matrix_view}' not supported; "
                    f"valid views: {self.valid_views()}."
                )
            matrix_view = "full"

        super().__init__(
            matrix_dtype=matrix_dtype,
            matrix_property=matrix_property,
            matrix_view=matrix_view,
            reuse_analysis=reuse_analysis,
            reuse_factorization=reuse_factorization,
            comm=comm,
            local_rows=local_rows,
        )

        if np.dtype(matrix_dtype) not in (np.float64, np.complex128):
            raise ValueError(
                f"Unsupported matrix dtype {matrix_dtype}; "
                "STRUMPACK binding needs float64 or complex128."
            )

        self._mpi_comm = comm._mpi_comm if comm is not None else MPI.COMM_SELF

        self._options = {
            k: self._option_to_str(v) for k, v in (strumpack_options or {}).items()
        }
        if isinstance(strumpack_args, str):
            self._args = strumpack_args
        else:
            self._args = " ".join(strumpack_args) if strumpack_args else ""

        # One STRUMPACK handle per scalar type (real / complex), created lazily.
        self._handles: dict[bool, int] = {}
        self._factored: dict[bool, bool] = {}
        self._int_dtype: dict[bool, type] = {}

    @staticmethod
    def valid_views() -> list[str]:
        """Returns the valid matrix views for the solver.

        Returns
        -------
        list[str]
            The valid matrix views.

        """
        return ["full"]

    # -- options --------------------------------------------------------
    @staticmethod
    def _option_to_str(value) -> str:
        if isinstance(value, bool):
            return "YES" if value else "NO"
        return str(value)

    def _apply_options(self, handle: int) -> None:
        errors = {
            -1: "unknown option",
            -2: "invalid value",
            -3: "managed by the solver and cannot be set",
        }
        for name, value in self._options.items():
            rc = _lib.qt_set_option(handle, name.encode(), value.encode())
            if rc != 0:
                _lib.qt_destroy(handle)
                raise ValueError(
                    f"STRUMPACK option {name}={value!r}: "
                    f"{errors.get(rc, f'error {rc}')}."
                )

    def _get_handle(self, is_complex: bool, n: int, nnz_local: int) -> int:
        handle = self._handles.get(is_complex)
        if handle is None:
            # Collective: all ranks agree on the index width.
            nnz_global = int(self._mpi_comm.allreduce(int(nnz_local)))
            use64 = max(n, nnz_global) >= 2**31 - 1

            # Collective over the communicator.
            handle = _lib.qt_create(
                int(is_complex),
                int(use64),
                self._mpi_comm.py2f(),
                self._args.encode(),
            )
            if not handle:
                raise RuntimeError("STRUMPACK handle creation failed.")
            self._apply_options(handle)
            self._handles[is_complex] = handle
            self._factored[is_complex] = False
            self._int_dtype[is_complex] = np.int64 if use64 else np.int32
        return handle

    def _close(self) -> None:
        """Free all STRUMPACK resources (collective)."""
        for handle in self._handles.values():
            _lib.qt_destroy(handle)
        self._handles.clear()
        self._factored.clear()
        self._int_dtype.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._close()

    def __del__(self):
        # Destruction is collective, so it must not happen implicitly at GC
        # time (ranks could deadlock). Only warn.
        if getattr(self, "_handles", None):
            warnings.warn(
                "Strumpack was not closed; call close() before MPI_Finalize.",
                ResourceWarning,
            )

    @profiler.profile("STRUMPACK solve", level="default")
    def solve(self, a: sparse.csr_matrix, b: NDArray) -> NDArray:
        """Solve ``a @ x = b``.

        Parameters
        ----------
        a : sparse.csr_matrix
            Local rows of the matrix, shape ``(m_loc, n)``, with *global*
            column indices. May live on the GPU (copied to host).
        b : NDArray
            Local rows of the right-hand side, shape ``(m_loc,)`` or
            ``(m_loc, nrhs)``.

        Returns
        -------
        NDArray
            Local rows of the solution, same shape and array library as ``b``.
        """
        if not sparse.isspmatrix_csr(a):
            raise ValueError("Matrix a must be in CSR format.")

        # Move everything to the host: STRUMPACK's sparse solver is used here
        # through host pointers.
        if get_array_module_name(a.data) == "cupy":
            a = a.get()
        b_on_gpu = get_array_module_name(b) == "cupy"
        b_host = b.get() if b_on_gpu else np.asarray(b)

        # Common scalar type: float64 or complex128.
        dtype = np.result_type(a.dtype, b_host.dtype, np.float64)
        if dtype not in (np.float64, np.complex128):
            raise ValueError(f"Unsupported dtype {dtype}; need float64 or complex128.")
        is_complex = bool(dtype == np.complex128)

        # Shapes / distribution.
        m_loc, n = a.shape
        if self._local_rows is not None:
            start, stop = self._local_rows
            if m_loc != stop - start:
                raise ValueError(
                    f"a has {m_loc} rows but local_rows spans {stop - start}."
                )
            fst_row = start
        else:
            if m_loc != n:
                raise ValueError("Without a communicator, a must be square.")
            fst_row = 0

        squeeze = b_host.ndim == 1
        if b_host.shape[0] != m_loc:
            raise ValueError(f"b has {b_host.shape[0]} local rows, expected {m_loc}.")

        # Fortran-ordered input and a separate output array (STRUMPACK solves
        # out of place; the caller's b stays untouched). ld == m_loc.
        rhs = np.asfortranarray(b_host.reshape(m_loc, -1), dtype=dtype)
        nrhs = rhs.shape[1]
        x = np.empty((m_loc, nrhs), dtype=dtype, order="F")
        ld = max(m_loc, 1)

        # CSR arrays in exactly the layout STRUMPACK expects.
        if not a.has_sorted_indices:
            a = a.copy()
            a.sort_indices()

        handle = self._get_handle(is_complex, n, a.nnz)
        int_dtype = self._int_dtype[is_complex]
        if int_dtype is np.int32 and max(n, a.nnz) >= 2**31 - 1:
            raise OverflowError(
                "Problem too large for the 32-bit STRUMPACK handle created at "
                "the first solve."
            )

        data = np.ascontiguousarray(a.data, dtype=dtype)
        indices = np.ascontiguousarray(a.indices, dtype=int_dtype)
        indptr = np.ascontiguousarray(a.indptr, dtype=int_dtype)

        if self.reuse_factorization and self._factored[is_complex]:
            mode = _QT_FACTORED
        elif self.reuse_analysis and self._factored[is_complex]:
            mode = _QT_SAMEPATTERN
        else:
            mode = _QT_DOFACT

        info = _lib.qt_solve(
            handle,
            mode,
            n,
            m_loc,
            fst_row,
            a.nnz,
            _ptr(data),
            _ptr(indices),
            _ptr(indptr),
            _ptr(rhs),
            _ptr(x),
            ld,
            nrhs,
        )

        if info == _ERR_PATTERN:
            self._factored[is_complex] = False
            raise ValueError(
                "reuse_analysis=True but the sparsity pattern changed since "
                "the last factorization."
            )
        if info == _ERR_NOFACTOR:
            self._factored[is_complex] = False
            raise RuntimeError("reuse_factorization=True but nothing is factored.")
        if info == _ERR_DIST:
            self._factored[is_complex] = False
            raise ValueError(
                "Row blocks are not contiguous and ordered by rank "
                "(rank r must own rows [local_rows[0], local_rows[1]) and the "
                "blocks must tile 0..n-1 in rank order)."
            )
        if info in (_ERR_ALLOC, _ERR_OVERFLOW):
            self._factored[is_complex] = False
            raise RuntimeError(
                "STRUMPACK shim failed: "
                + (
                    "allocation error."
                    if info == _ERR_ALLOC
                    else "32-bit index overflow."
                )
            )
        if info != 0:
            # NO_CONVERGENCE (iterative solve) leaves the factorization valid,
            # everything else forces a fresh analysis next time.
            if info != 4:
                self._factored[is_complex] = False
            raise RuntimeError(
                f"STRUMPACK solve failed with info={info} "
                f"({_STRUMPACK_CODES.get(info, 'unknown')})."
            )

        self._factored[is_complex] = True

        if squeeze:
            x = x[:, 0]
        return xp.asarray(x) if b_on_gpu else x
