# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host, synchronize_device
from qttools.utils.mpi_utils import distributed_load, get_section_sizes

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


@profiler.profile(level="api")
def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    """Computes the convolution of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The convolution of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fft(a, n, axis=0)
    b_fft = xp.fft.fft(b, n, axis=0)
    return xp.fft.ifft(a_fft * b_fft, axis=0)


@profiler.profile(level="api")
def fft_correlate(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The cross-correlation of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fft(a, n, axis=0)
    b_fft = xp.fft.fft(b[::-1], n, axis=0)
    return xp.fft.ifft(a_fft * b_fft, axis=0)


@profiler.profile(level="api")
def hilbert_transform(a: NDArray, energies: NDArray, eta=1e-8) -> NDArray:
    """Computes the Hilbert transform of the array a.

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    a : NDArray
        The array to transform.
    energies : NDArray
        The energy values corresponding to the first axis of a.
    eta : float, optional
        For the principle part. Small part to avoid singularity, by
        default 1e-8.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    # Add a small imaginary part to avoid singularity.
    energy_differences = (energies - energies[0]).reshape(-1, 1)
    ne = energies.size
    # eta for removing the singularity. See Cauchy principal value.
    b = (
        fft_convolve(a, 1 / (energy_differences + eta))[:ne]
        + fft_convolve(a, 1 / (-energy_differences[::-1] - eta))[ne - 1 :]
    )
    # Not sure about the prefactor. Currently gives the same value as the old code.
    return b / (2 * xp.pi) * (energies[1] - energies[0])


class SigmaCoulombScreening(ScatteringSelfEnergy):
    """Computes the scattering self-energy from the Coulomb screening.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: NDArray,
    ):
        """Initializes the scattering self-energy."""
        self.energies = electron_energies
        self.num_energies = self.energies.size
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])

        self.big_block_sizes = None
        self.batch_size = compute_config.convolve.batch_size

    @profiler.profile(level="api")
    def compute(
        self,
        g_lesser: DSBSparse,
        g_greater: DSBSparse,
        w_lesser: DSBSparse,
        w_greater: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        w_lesser : DSBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.

        """

        t_block_reorder_start = time.perf_counter()
        if w_lesser.nnz != g_lesser.nnz:
            raise ValueError(
                "The sparsity pattern of w_lesser and g_lesser must match."
                "Something went horribly wrong."
            )

        # Save the block sizes for later.
        if self.big_block_sizes is None:
            self.big_block_sizes = w_lesser.block_sizes

        # Enforce that the block sizes are the same. NOTE: This triggers
        # a block-reordering in the DSBSparse object.
        w_lesser.block_sizes = g_lesser.block_sizes
        w_greater.block_sizes = g_greater.block_sizes

        sigma_lesser, sigma_greater, sigma_retarded = out

        synchronize_device()
        t_block_reorder_end = time.perf_counter()
        comm.Barrier()
        t_block_reorder_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaCoulombScreening: block reorder: {t_block_reorder_end - t_block_reorder_start:.3f} s", flush=True)
            print(f"    SigmaCoulombScreening: block reorder all: {t_block_reorder_end_all - t_block_reorder_start:.3f} s", flush=True)

        t_all2all_start = time.perf_counter()
        with profiler.profile_range("stack->nnz transpose", level="debug"):
            # Transpose the matrices to nnz distribution.
            for m in (
                w_lesser,
                w_greater,
                g_lesser,
                g_greater,
                sigma_lesser,
                sigma_greater,
                sigma_retarded,
            ):
                # The electron Green's functions and self-energies should
                # ideally already be in nnz-distribution. We cannot discard
                # the data here, as we cannot be sure that this is the
                # first/only interaction.
                m.dtranspose() if m.distribution_state != "nnz" else None
        synchronize_device()
        t_all2all_end = time.perf_counter()
        comm.Barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaCoulombScreening: stack->nnz transpose: {t_all2all_end - t_all2all_start:.3f} s", flush=True)
            print(f"    SigmaCoulombScreening: stack->nnz transpose all: {t_all2all_end_all - t_all2all_start:.3f} s", flush=True)


        t_sse_start = time.perf_counter()
        # Because of padding there could be no ij elements
        if g_greater.data.shape[-1] != 0:

            with profiler.profile_range("SSE computation", level="debug"):
                self.batch_size = None
                if self.batch_size is None:
                    # NOTE: This is a temporary solution. The batch size should be
                    # calculated in the configuration.
                    self.batch_size = g_greater.data.shape[-1]

                batch_counts, _ = get_section_sizes(
                    g_greater.data.shape[-1],
                    int(np.ceil(g_greater.data.shape[-1] / self.batch_size)),
                )

                batch_displacements = np.cumsum(
                    np.concatenate(([0], np.array(batch_counts)))
                )

                energy_differences = (self.energies - self.energies[0]).reshape(-1, 1)
                eta = 1e-8
                # Add a small imaginary part to avoid singularity.
                # eta for removing the singularity. See Cauchy principal value.
                n = g_lesser.data.shape[0] + g_greater.data.shape[0] - 1
                ne = g_lesser.data.shape[0]

                hilbert_kernel_fft = xp.fft.fft(1 / (energy_differences + eta), n, axis=0).T

                for start, end in zip(batch_displacements, batch_displacements[1:]):
                    batch = slice(start, end)

                    g_x_fft = xp.fft.fft(g_lesser.data[:, batch].T, n, axis=1)
                    w_lesser_fft = xp.fft.fft(w_lesser.data[:, batch].T, n, axis=1)
                    w_greater_fft = xp.fft.fft(w_greater.data[:, batch].T, n, axis=1)

                    sigma_x_fft = xp.multiply(g_x_fft, w_lesser_fft)
                    sigma_x_fft -= xp.multiply(
                        g_x_fft, w_greater_fft.conj()
                    )  # negative energy part
                    lesser = self.prefactor * xp.fft.ifft(sigma_x_fft, axis=1)[:, :ne]
                    sigma_lesser._data[
                        sigma_lesser._stack_padding_mask, ..., batch
                    ] += lesser.T

                    # antihermitian_fft = -sigma_x_fft

                    g_x_fft = xp.fft.fft(g_greater.data[:, batch].T, n, axis=1)
                    sigma_x_fft = xp.multiply(g_x_fft, w_greater_fft)
                    sigma_x_fft -= xp.multiply(
                        g_x_fft, w_lesser_fft.conj()
                    )  # negative energy part
                    greater = self.prefactor * xp.fft.ifft(sigma_x_fft, axis=1)[:, :ne]
                    sigma_greater._data[
                        sigma_greater._stack_padding_mask, ..., batch
                    ] += greater.T

                    # antihermitian_fft += sigma_x_fft

                    # Compute retarded self-energy with a Hilbert transform.
                    antihermitian = 1j * xp.imag(greater - lesser)
                    antihermitian_fft = xp.fft.fft(antihermitian, n, axis=1)
                    # TODO check this: impose the causality in the FFT domain instead of taking the
                    # imaginary part in the real domain, we have one less fft to do
                    # antihermitian_fft *= self.prefactor * 0.5
                    # antihermitian_fft -=  xp.flip(antihermitian_fft.conj(), axis=0) # remove the hermitian part X(-t) = X(t).conj()

                    sigma_x_fft = xp.multiply(antihermitian_fft, hilbert_kernel_fft)
                    sigma_x_fft -= xp.multiply(
                        antihermitian_fft, hilbert_kernel_fft.conj()
                    )  # negative energy part
                    sigma_retarded._data[
                        sigma_retarded._stack_padding_mask, ..., batch
                    ] += (
                        self.prefactor * xp.fft.ifft(sigma_x_fft, axis=1)[:, :ne]
                        + antihermitian / 2
                    ).T

                    # Compute the full self-energy using the convolution theorem.
                    # Second term are corrections for negative frequencies that
                    # where cut off by the polarization calculation.
                    # TODO: the datastructures does not allow for easy slicing of the
                    # data. This is a workaround.
                    # sigma_lesser._data[
                    #     sigma_lesser._stack_padding_mask, ..., batch
                    # ] += self.prefactor * (
                    #     fft_convolve(g_lesser.data[:, batch], w_lesser.data[:, batch])[
                    #         : self.num_energies
                    #     ]
                    #     - fft_correlate(
                    #         g_lesser.data[:, batch], w_greater.data[:, batch].conj()
                    #     )[self.num_energies - 1 :]
                    # )
                    # sigma_greater._data[
                    #     sigma_greater._stack_padding_mask, ..., batch
                    # ] += self.prefactor * (
                    #     fft_convolve(g_greater.data[:, batch], w_greater.data[:, batch])[
                    #         : self.num_energies
                    #     ]
                    #     - fft_correlate(
                    #         g_greater.data[:, batch], w_lesser.data[:, batch].conj()
                    #     )[self.num_energies - 1 :]
                    # )
                    

                    # Compute retarded self-energy with a Hilbert transform.
                    # sigma_antihermitian = 1j * xp.imag(
                    #     sigma_greater.data[:, batch] - sigma_lesser.data[:, batch]
                    # )
                    # sigma_hermitian = hilbert_transform(sigma_antihermitian, self.energies)
                    # sigma_retarded._data[
                    #     sigma_retarded._stack_padding_mask, ..., batch
                    # ] += (1j * sigma_hermitian + sigma_antihermitian / 2)

        synchronize_device()
        t_sse_end = time.perf_counter()
        comm.Barrier()
        t_sse_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaCoulombScreening: SSE computation: {t_sse_end - t_sse_start:.3f} s", flush=True)
            print(f"    SigmaCoulombScreening: SSE computation all: {t_sse_end_all - t_sse_start:.3f} s", flush=True)

        t_all2all_start = time.perf_counter()
        # Transpose the matrices to stack distribution.
        with profiler.profile_range("nnz->stack transpose", level="debug"):
            for m in (w_lesser, w_greater):
                m.dtranspose(discard=True) if m.distribution_state != "stack" else None
            # NOTE: The electron Green's functions and self-energies must
            # not be transposed back to stack distribution, as they are
            # needed in nnz distribution for the other interactions.

        t_all2all_end = time.perf_counter()
        comm.Barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaCoulombScreening: nnz->stack transpose: {t_all2all_end - t_all2all_start:.3f} s", flush=True)
            print(f"    SigmaCoulombScreening: nnz->stack transpose all: {t_all2all_end_all - t_all2all_start:.3f} s", flush=True)

        t_block_reorder_start = time.perf_counter()
        # Recover original block sizes.
        w_lesser.block_sizes = self.big_block_sizes
        w_greater.block_sizes = self.big_block_sizes
        synchronize_device()
        t_block_reorder_end = time.perf_counter()
        comm.Barrier()
        t_block_reorder_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaCoulombScreening: block reorder back: {t_block_reorder_end - t_block_reorder_start:.3f} s", flush=True)
            print(f"    SigmaCoulombScreening: block reorder back all: {t_block_reorder_end_all - t_block_reorder_start:.3f} s", flush=True)


class SigmaFock(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the bare Fock self-energy."""
        self.energies = electron_energies
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)
        # Make sure that the Coulomb matrix is Hermitian.
        coulomb_matrix_sparray = (
            0.5 * (coulomb_matrix_sparray + coulomb_matrix_sparray.conj().T)
        ).tocoo()

        # Load block sizes for the coulomb matrix.
        block_sizes = get_host(
            distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        )

        # Create the DSBSparse object.
        # TODO: This is pretty wasteful memory-wise.
        # Workaround: Use comm size as global stack shape.
        coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=(comm.size,),
        )
        coulomb_matrix.data = 0.0
        coulomb_matrix += coulomb_matrix_sparray
        coulomb_matrix.dtranspose()
        self.coulomb_matrix_data = coulomb_matrix.data[0]

    @profiler.profile(level="api")
    def compute(self, g_lesser: DSBSparse, out: tuple[DSBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        t_all2all_start = time.perf_counter()
        (sigma_retarded,) = out
        for m in (g_lesser, sigma_retarded):
            # These should both already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None
        synchronize_device()
        t_all2all_end = time.perf_counter()
        comm.Barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaFock: stack->nnz transpose: {t_all2all_end - t_all2all_start:.3f} s", flush=True)
            print(f"    SigmaFock: stack->nnz transpose all: {t_all2all_end_all - t_all2all_start:.3f} s", flush=True)


        t_sse_start = time.perf_counter()
        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded.data += xp.real(gl_density * self.coulomb_matrix_data)
        synchronize_device()
        t_sse_end = time.perf_counter()
        comm.Barrier()
        t_sse_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    SigmaFock: SSE computation: {t_sse_end - t_sse_start:.3f} s", flush=True)
            print(f"    SigmaFock: SSE computation all: {t_sse_end_all - t_sse_start:.3f} s", flush=True)

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
