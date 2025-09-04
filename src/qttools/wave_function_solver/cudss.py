# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import ctypes.util
from collections.abc import Sequence

from qttools import NDArray, sparse
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

try:
    from nvmath.bindings import cudss
    from nvmath.internal import tensor_wrapper, utils
    from nvmath.sparse._internal import cudss_utils
    from nvmath.sparse.advanced import DirectSolver
    from nvmath.sparse.advanced.direct_solver import (
        axis_order_in_memory,
        calculate_strides,
    )

    nvmath_available = True

except ImportError:
    nvmath_available = False


profiler = Profiler()

# Possible multithreading libraries for cuDSS in descending order of
# preference.
_mtlayer_libs = [
    "libcudss_mtlayer_gomp.so.0",
    "libcudss_mtlayer_gomp.so.0.5.0",
]


class cuDSS(WFSolver):
    """Wave function solver using cuDSS for sparse matrix solving.

    This solver uses the cuDSS library to solve sparse linear systems
    on NVIDIA GPUs.

    Parameters
    ----------
    explicitely_reset_operands : str, optional
        String indicating which operands to reset explicitly. If "a" is
        in the string, the system matrix `a` will be reset before
        solving. If "b" is in the string, the right-hand side vector `b`
        will be reset before solving. Default is "b", meaning only the
        right-hand side vector will be reset.
    use_multithreading : bool, optional
        Whether to use multithreading for the solver. If True, it will
        attempt to find a suitable multithreading library. Default is
        True.

    """

    def __init__(
        self,
        explicitely_reset_operands: str = "b",
        use_multithreading: bool = True,
    ) -> None:
        """Initializes the cuDSS wave function solver."""
        if not nvmath_available:
            raise ImportError(
                "cuDSS is not available. Please install it to use this solver."
            )

        self.direct_solver = None
        self.plan_info = None
        self.explicitely_reset_a = "a" in explicitely_reset_operands
        self.explicitely_reset_b = "b" in explicitely_reset_operands

        self.solver_options = {}

        if use_multithreading:
            # Try to find a multithreading library for cuDSS.
            multithreading_lib = [
                ctypes.util.find_library(lib) for lib in _mtlayer_libs
            ].pop(0)

            if multithreading_lib is None:
                print("WARNING! No suitable multithreading library found for cuDSS.")
            else:
                self.solver_options["multithreading_lib"] = multithreading_lib

    def __del__(self) -> None:
        """Cleans up the cuDSS solver context."""
        if self.direct_solver is not None:
            self.direct_solver.free()
            self.direct_solver = None

    def _explicitely_reset_b(self, b: NDArray) -> None:
        """Resets the right-hand side in the direct solver.

        This is a workaround for the current limitation in nvmath where
        resetting the right-hand side does not support different shapes
        or strides than the original one.

        Parameters
        ----------
        b : NDArray
            The new right-hand side vector to set.

        """
        ds = self.direct_solver

        stream_holder = utils.get_or_create_stream(
            device_id=ds.device_id, stream=None, op_package=ds.rhs_package
        )

        # Do the checks that nvmath does too.
        # NOTE: Check the direct_solver.py source code for the original
        # implementation.
        explicitly_batched = isinstance(b, Sequence)
        if explicitly_batched:
            raise NotImplementedError(
                "Explicitly batched right-hand sides are not supported."
            )
        b = tensor_wrapper.wrap_operand(b)
        rhs_package = utils.infer_object_package(b.tensor)

        device_id = b.device_id
        memory_space = b.device
        value_type = b.dtype
        shape = b.shape
        strides = b.strides

        # Handle cupy <> numpy asymmetry. See note #2.
        if rhs_package == "numpy":
            rhs_package = "cupy"

        # Check package, device ID, shape, strides, and dtype.
        if rhs_package != ds.rhs_package:
            raise TypeError(
                f"The package for 'b' ({rhs_package}) doesn't match the original one ({ds.rhs_package})."
            )
        if memory_space != ds.memory_space:
            raise TypeError(
                f"The memory space for 'b' ({memory_space}) doesn't match the original one ({ds.memory_space})."
            )
        if device_id != "cpu" and device_id != ds.device_id:
            raise TypeError(
                f"The device id for 'b' ({device_id}) doesn't match the original one ({ds.device_id})."
            )
        if value_type != ds.value_type:
            raise TypeError(
                f"The dtype for 'b' ({value_type}) doesn't match the original one ({ds.value_type})."
            )

        if ds.copy_across_memspace:
            raise NotImplementedError(
                "Copying across memory spaces is not supported for resetting the right-hand side."
            )
        ds.b = b

        # Update the rhs and result shape and strides.
        ds.rhs_shape = shape
        ds.rhs_strides = strides

        ds.result_shape = shape
        # NOTE: I do not know what these two lines do.
        # For single or implicitly-batched RHS, the matrix may not be
        # compact so we use the axis ordering to determine the strides.
        axis_order = axis_order_in_memory(ds.rhs_shape, ds.rhs_strides)
        ds.result_strides = calculate_strides(ds.rhs_shape, axis_order)

        # NOTE: Now it is not enough to just update the pointer
        # references, and keep reference to the internal buffers.
        # Instead, we need to destroy the existing resources and create
        # new ones with the updated right-hand side.

        # Free matrix pointers.
        cudss.matrix_destroy(ds.x_ptr)
        cudss.matrix_destroy(ds.b_ptr)

        ds.resources_b, ds.b_ptr = cudss_utils.create_cudss_dense_wrapper(
            ds.cuda_index_type,
            ds.cuda_value_type,
            ds.index_type,
            ds.batch_indices,
            ds.b,
            stream_holder,
        )
        # Use `b` for creating the (potentially explicitly or implicitly
        # batched) solution matrix or vector. The pointers will be
        # updated later in execute.
        ds.resources_x, ds.x_ptr = cudss_utils.create_cudss_dense_wrapper(
            ds.cuda_index_type,
            ds.cuda_value_type,
            ds.index_type,
            ds.batch_indices,
            ds.b,
            stream_holder,
        )

    @profiler.profile(level="api")
    def solve(self, a: sparse.spmatrix, b: NDArray) -> NDArray:
        """Solves the sparse linear system a @ x = b using cuDSS.

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
        if self.direct_solver is None or self.plan_info is None:
            self.direct_solver = DirectSolver(a, b, options=self.solver_options)
            # NOTE: By default this uses a custom nested dissectioning
            # scheme based on METIS. Other options could in principle be
            # exposed as a parameter.
            self.plan_info = self.direct_solver.plan()

        if self.explicitely_reset_a:
            self.direct_solver.reset_operands(a=a)
            # After resetting a, we need to re-plan.
            self.plan_info = self.direct_solver.plan()

        if self.explicitely_reset_b:
            # TODO: This does not support setting a right-hand side that
            # has a different shape or strides than the original one.
            # Until it is fixed in nvmath, we will hack around it by
            # resetting the operands.
            # self.direct_solver.reset_operands(b=b)
            self._explicitely_reset_b(b)

        self.direct_solver.factorize()
        return self.direct_solver.solve()
