# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import cupy as cp
import numpy as np
import torch

from qttools.kernels.linalg.crt_utils import zgemm3m_crt


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
                a = cp.broadcast_to(a, (batch_size, a.shape[1], a.shape[2], a.shape[3]))
                b = cp.broadcast_to(b, (batch_size, b.shape[1], b.shape[2], b.shape[3]))
        elif a.ndim == 4 or b.ndim == 4:
            if a.ndim == 4:
                b = cp.broadcast_to(b, (a.shape[0], a.shape[1], a.shape[2], a.shape[3]))
            else:
                a = cp.broadcast_to(a, (b.shape[0], a.shape[1], a.shape[2], a.shape[3]))

        c = cp.empty_like(a)

        assert slices > 0, "slices must be a positive integer"

        for batch in np.ndindex(a.shape[:-2]):
            a_re = cupy_to_torch(a[batch].real.copy())
            a_im = cupy_to_torch(a[batch].imag.copy())
            b_re = cupy_to_torch(b[batch].real.T.copy())
            b_im = cupy_to_torch(b[batch].imag.T.copy())

            c_re_crt, c_im_crt = zgemm3m_crt(a_re, a_im, b_re, b_im, num_moduli=slices)

            c[batch] = torch_to_cupy(c_re_crt + 1j * c_im_crt)

        return c

    else:
        raise ValueError(f"Invalid value for ozaki: {ozaki}")
