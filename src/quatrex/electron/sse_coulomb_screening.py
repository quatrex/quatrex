# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
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
    energy_differences = (energies - energies[0]).reshape(-1, 1)
    ne = energies.size
    eta = 1e-6
    b = (
        fft_convolve(a, 1 / (energy_differences + 1j * eta))[:ne]
        + fft_convolve(a, 1 / (-energy_differences[::-1] + 1j * eta))[ne - 1 :]
    )
    # The factor 1j / 4 * eta is a bit arbitrary (needed for identity sr-sa = sg-sl). Can probably be proved (principal value?).
    return 1j / 4 * eta * b


class SigmaCoulombScreening(ScatteringSelfEnergy):
    def __init__(self, config: QuatrexConfig, electron_energies: xp.ndarray):
        self.energies = electron_energies
        self.ne = self.energies.size
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

        # If the w_lesser and w_greater don't have the same sparsity pattern as
        # g_lesser and g_greater, we have to reduce them to the same sparsity pattern.
        if w_lesser.nnz != g_lesser.nnz:
            w_lesser_reduced = w_lesser.reduce_to(
                g_lesser.rows, g_lesser.cols, g_lesser.block_sizes
            )
        if w_greater.nnz != g_greater.nnz:
            w_greater_reduced = w_greater.reduce_to(
                g_greater.rows, g_greater.cols, g_greater.block_sizes
            )

        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (
            g_lesser,
            g_greater,
            w_lesser_reduced,
            w_greater_reduced,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the full self-energy using the convolution theorem.
        # Second term are corrections for positive frequencies that where cut off by the polarization calculation.
        sigma_lesser._data[
            sigma_lesser._stack_padding_mask,
            ...,
            : sigma_lesser.nnz_section_sizes[comm.rank],
        ] += self.prefactor * (
            fft_convolve(g_lesser.data, w_lesser_reduced.data)[self.ne - 1 :]
            - fft_correlate(g_lesser.data, w_greater_reduced.data.conj())[: self.ne]
        )
        sigma_greater._data[
            sigma_greater._stack_padding_mask,
            ...,
            : sigma_greater.nnz_section_sizes[comm.rank],
        ] += self.prefactor * (
            fft_convolve(g_greater.data, w_greater_reduced.data)[self.ne - 1 :]
            - fft_correlate(g_greater.data, w_lesser_reduced.data.conj())[: self.ne]
        )

        # Compute retarded self-energy with a Hilbert transform.
        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] += hilbert_transform(sigma_greater.data - sigma_lesser.data, self.energies)

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


class SigmaFock(ScatteringSelfEnergy):
    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: xp.ndarray,
    ):
        self.energies = electron_energies
        self.prefactor = 1j / xp.pi * (self.energies[1] - self.energies[0])
        coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        self.coulomb_matrix = compute_config.dbsparse_type.from_sparray(
            coulomb_matrix_sparray,
            block_sizes,
            (self.energies.size,),
            densify_blocks=[
                (i, j)
                for i in range(len(block_sizes))
                for j in range(max(i - 1, 0), min(i + 2, len(block_sizes)))
            ],
        )

    def compute(
        self,
        g_lesser: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Computes the Fock self-energy."""
        (sigma_retarded,) = out
        for m in (g_lesser, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "nnz" else None
        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] += (
            gl_density * self.coulomb_matrix.data
        )
        for m in (g_lesser, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "stack" else None
