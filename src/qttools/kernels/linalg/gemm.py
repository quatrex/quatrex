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

        if a.ndim > 2:

            c = cp.empty_like(a)

            assert slices > 0, "slices must be a positive integer"

            for batch in np.ndindex(a.shape[:-2]):
                a_re = cupy_to_torch(a[batch].real.copy())
                a_im = cupy_to_torch(a[batch].imag.copy())
                b_slice = b[batch].T.copy()
                b_re = cupy_to_torch(b_slice.real.copy())
                b_im = cupy_to_torch(b_slice.imag.copy())

                # pad to power of 2 if necessary for better performance
                # if the shape is not a power of 2
                new_shape_a = [2 ** int(np.ceil(np.log2(s))) for s in a_re.shape]
                new_shape_b = [2 ** int(np.ceil(np.log2(s))) for s in b_re.shape]

                a_re_padded = torch.zeros(
                    new_shape_a, dtype=a_re.dtype, device=a_re.device
                )
                a_im_padded = torch.zeros(
                    new_shape_a, dtype=a_im.dtype, device=a_im.device
                )
                b_re_padded = torch.zeros(
                    new_shape_b, dtype=b_re.dtype, device=b_re.device
                )
                b_im_padded = torch.zeros(
                    new_shape_b, dtype=b_im.dtype, device=b_im.device
                )

                a_re_padded[: a_re.shape[0], : a_re.shape[1]] = a_re
                a_im_padded[: a_im.shape[0], : a_im.shape[1]] = a_im
                b_re_padded[: b_re.shape[0], : b_re.shape[1]] = b_re
                b_im_padded[: b_im.shape[0], : b_im.shape[1]] = b_im

                c_re_crt, c_im_crt = zgemm3m_crt(
                    a_re_padded,
                    a_im_padded,
                    b_re_padded,
                    b_im_padded,
                    num_moduli=slices,
                )

                c[batch] = torch_to_cupy(
                    c_re_crt[: a_re.shape[0], : b_re.shape[1]]
                ) + 1j * torch_to_cupy(c_im_crt[: a_im.shape[0], : b_im.shape[1]])

        else:
            a_re = cupy_to_torch(a.real.copy())
            a_im = cupy_to_torch(a.imag.copy())
            b_slice = b.T.copy()
            b_re = cupy_to_torch(b_slice.real.copy())
            b_im = cupy_to_torch(b_slice.imag.copy())

            # pad to power of 2 if necessary for better performance
            # if the shape is not a power of 2
            new_shape_a = [2 ** int(np.ceil(np.log2(s))) for s in a_re.shape]
            new_shape_b = [2 ** int(np.ceil(np.log2(s))) for s in b_re.shape]

            a_re_padded = torch.zeros(new_shape_a, dtype=a_re.dtype, device=a_re.device)
            a_im_padded = torch.zeros(new_shape_a, dtype=a_im.dtype, device=a_im.device)
            b_re_padded = torch.zeros(new_shape_b, dtype=b_re.dtype, device=b_re.device)
            b_im_padded = torch.zeros(new_shape_b, dtype=b_im.dtype, device=b_im.device)

            a_re_padded[: a_re.shape[0], : a_re.shape[1]] = a_re
            a_im_padded[: a_im.shape[0], : a_im.shape[1]] = a_im
            b_re_padded[: b_re.shape[0], : b_re.shape[1]] = b_re
            b_im_padded[: b_im.shape[0], : b_im.shape[1]] = b_im

            c_re_crt, c_im_crt = zgemm3m_crt(
                a_re_padded, a_im_padded, b_re_padded, b_im_padded, num_moduli=slices
            )

            c = torch_to_cupy(
                c_re_crt[: a_re.shape[0], : b_re.shape[1]]
            ) + 1j * torch_to_cupy(c_im_crt[: a_im.shape[0], : b_im.shape[1]])

        return c

    else:
        raise ValueError(f"Invalid value for ozaki: {ozaki}")
