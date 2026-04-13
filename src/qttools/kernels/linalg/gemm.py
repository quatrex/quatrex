# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import cupy as cp
import numpy as np

try:

    import torch
    from emu_gemm.cupy_interface import (  # zgemm_emu1,; zgemm_emu1_workspace_size,
        zgemm_emu2,
        zgemm_emu2_workspace_size,
    )

    def cupy_to_torch(a: cp.ndarray) -> torch.Tensor:
        """Zero-copy cupy ndarray → torch CUDA tensor."""
        return torch.from_dlpack(a)

    def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
        """Zero-copy torch CUDA tensor → cupy ndarray."""
        return cp.from_dlpack(t)

    def matmul(a, b, ozaki: None | int = None, slices: None | int = None):
        if ozaki is None:
            return a @ b
        elif ozaki == 1:
            raise NotImplementedError("Ozaki method 1 is not implemented yet")
        elif ozaki == 2:
            if slices is None:
                raise ValueError("slices must be specified when ozaki is 2")
            # Implement Ozaki method 2 here

            if a.ndim == 3 and b.ndim == 3:
                if a.shape[0] != b.shape[0]:
                    batch_size = max(a.shape[0], b.shape[0])

                    # broadcast a and b to the same batch size
                    a = cp.broadcast_to(a, (batch_size, a.shape[1], a.shape[2]))
                    b = cp.broadcast_to(b, (batch_size, b.shape[1], b.shape[2]))
            elif a.ndim == 3 or b.ndim == 3:
                if a.ndim == 3:
                    b = cp.broadcast_to(b, (a.shape[0], b.shape[1], b.shape[2]))
                else:
                    a = cp.broadcast_to(a, (b.shape[0], a.shape[1], a.shape[2]))
            elif a.ndim == 4 and b.ndim == 4:
                if a.shape[0] != b.shape[0]:
                    batch_size = max(a.shape[0], b.shape[0])
                    # broadcast a and b to the same batch size
                    a = cp.broadcast_to(
                        a, (batch_size, a.shape[1], a.shape[2], a.shape[3])
                    )
                    b = cp.broadcast_to(
                        b, (batch_size, b.shape[1], b.shape[2], b.shape[3])
                    )
            elif a.ndim == 4 or b.ndim == 4:
                if a.ndim == 4:
                    b = cp.broadcast_to(
                        b, (a.shape[0], a.shape[1], a.shape[2], a.shape[3])
                    )
                else:
                    a = cp.broadcast_to(
                        a, (b.shape[0], a.shape[1], a.shape[2], a.shape[3])
                    )

            assert slices > 0, "slices must be a positive integer"

            M, N, K = a.shape[-2], b.shape[-1], a.shape[-1]

            # check for C order
            if not a.flags.c_contiguous:
                a = cp.ascontiguousarray(a)
            if not b.flags.c_contiguous:
                b = cp.ascontiguousarray(b)

            if a.ndim > 2:

                ws = cp.empty(
                    zgemm_emu2_workspace_size(M, N, K, slices), dtype=cp.uint8
                )
                c = cp.empty((*a.shape[:-2], M, N), dtype=cp.complex128)

                for batch in np.ndindex(a.shape[:-2]):
                    zgemm_emu2(a[batch], b[batch], c[batch], ws, slices, b_k_major=True)

            else:
                ws = cp.empty(
                    zgemm_emu2_workspace_size(M, N, K, slices), dtype=cp.uint8
                )
                c = cp.empty((M, N), dtype=cp.complex128)
                zgemm_emu2(a, b, c, ws, slices, b_k_major=True)

            return c

        else:
            raise ValueError(f"Invalid value for ozaki: {ozaki}")

except (ImportError, ModuleNotFoundError):

    def cupy_to_torch(a):
        return a

    def torch_to_cupy(t):
        return t

    import warnings

    def matmul(a, b, ozaki: None | int = None, slices: None | int = None):
        if ozaki is not None:
            warnings.warn(
                "Ozaki method is not available because the emu_gemm package is not installed",
                RuntimeWarning,
            )
        return a @ b
