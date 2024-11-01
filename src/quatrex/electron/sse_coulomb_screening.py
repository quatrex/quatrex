# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import qttools.datastructures as DSBSparse
from mpi4py.MPI import COMM_WORLD as comm
from qttools.utils.gpu_utils import xp

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


def fft_convolve(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the convolution of two arrays using the FFT."""
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b, (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


def fft_correlate(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the correlation of two arrays using the FFT."""
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


def hilbert_transform(a: xp.ndarray, energies: xp.ndarray) -> xp.ndarray:
    """Computes the Hilbert transform of the array a."""
    return fft_convolve(a, 1 / energies)


class SigmaCoulombScreening(ScatteringSelfEnergy):
    def __init__(self, config: QuatrexConfig, electron_energies: xp.ndarray):
        self.energies = electron_energies
        self.prefactor = 1j / np.pi * (self.energies[1] - self.energies[0])

    def compute(
        self,
        g_lesser: DSBSparse,
        g_greater: DSBSparse,
        w_lesser: DSBSparse,
        w_greater: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Computes the GW self-energy."""
        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (
            g_lesser,
            g_greater,
            w_lesser,
            w_greater,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the full self-energy using the convolution theorem.
        sigma_lesser_full = self.prefactor * fft_convolve(g_lesser.data, w_lesser.data)
        sigma_greater_full = self.prefactor * fft_convolve(
            g_greater.data, w_greater.data
        )
        # Corrections for negative frequencies that where cut off by the polarization calculation.
        sigma_lesser_full -= self.prefactor * fft_correlate(
            g_lesser.data, w_greater.data.conj()
        )
        sigma_greater_full -= self.prefactor * fft_correlate(
            g_greater.data, w_lesser.data.conj()
        )

        # Compute retarded self-energy using the Kramer-Kronig relation with a Hilbert transform (Principle integral).
        sigma_retarded_full = self.prefactor * hilbert_transform(
            1j
            * (
                sigma_lesser_full[: g_lesser.shape[0]]
                - sigma_greater_full[: g_greater.shape[0]]
            ),
            self.energies,
        )

        sigma_lesser._data[
            sigma_lesser._stack_padding_mask,
            ...,
            : sigma_lesser.nnz_section_sizes[comm.rank],
        ] = sigma_lesser_full[: g_lesser.shape[0]]
        sigma_greater._data[
            sigma_greater._stack_padding_mask,
            ...,
            : sigma_greater.nnz_section_sizes[comm.rank],
        ] = sigma_greater_full[: g_greater.shape[0]]
        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] = sigma_retarded_full[: g_lesser.shape[0]]

        # Transpose the matrices to stack distribution.
        for m in (
            g_lesser,
            g_greater,
            w_lesser,
            w_greater,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "stack" else None
