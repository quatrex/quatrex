# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from qttools.fft import fft_convolve, fft_convolve_kpoints, fft_correlate_kpoints
from qttools.kernels.mixed_precision import compress, decompress
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import free_mempool
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()

if xp.__name__ == "cupy":
    cache = xp.fft.config.get_plan_cache()


def hilbert_transform(sl: NDArray, sg: NDArray, energies: NDArray) -> NDArray:
    """Computes the Hilbert transform.

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    sl : NDArray
        The lesser self-energy on the grid |-----|-----|xxxxx|.
    sg : NDArray
        The greater self-energy on the grid |xxxxx|-----|-----|.
    energies : NDArray
        The energy values corresponding to the first axis of sl/sg.

    Returns
    -------
    NDArray
         The Hilbert transformation of sg - sl.

    """
    ne = energies.size
    nk = sg.shape[1:-1]
    # Add empty dimensions for each k-point.
    energy_differences = (energies - energies[0]).reshape(-1, *(len(nk) + 1) * (1,))
    # eta for removing the singularity. See Cauchy principal value.
    eta = (energies[1] - energies[0]) / 2
    hilbert_kernel = 1 / (energy_differences + eta)

    sr = fft_convolve(sg[:ne] - sl[-ne:], hilbert_kernel)[:ne]
    # Correct for left edge
    sr += fft_convolve(-sl[:ne], hilbert_kernel)[-ne:]
    # Next account for negative frequencies
    hilbert_kernel = -hilbert_kernel[::-1]
    sr += fft_convolve(sg[:ne] - sl[-ne:], hilbert_kernel)[-ne:]
    # Correct for right edge
    sr += fft_convolve(sg[-ne:], hilbert_kernel)[:ne]

    return sr


class SigmaCoulombScreening(ScatteringSelfEnergy):
    """Computes the scattering self-energy from the Coulomb screening.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        config: QuatrexConfig,
        electron_energies: NDArray,
    ):
        """Initializes the scattering self-energy."""
        self.config = config
        self.energies = electron_energies
        self.kpoint_volume = np.prod(config.device.kpoint_grid)
        # self.num_energies = self.energies.size
        self.prefactor = (
            1j
            / (2 * xp.pi)
            * (self.energies[1] - self.energies[0])
            / self.kpoint_volume
        )
        self.big_block_sizes = None
        self.batch_size = config.compute.convolve.batch_size

        self.apply_hilbert_correction = (
            config.coulomb_screening.apply_hilbert_correction
        )

    def _compute_without_correction(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        batch: slice,
        hilbert_kernel_fft: NDArray,
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        w_lesser : DSDBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSDBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.
        batch : slice
            The batch slice for the current computation.
        hilbert_kernel_fft : NDArray
            The precomputed Hilbert kernel in Fourier space.

        """
        sigma_lesser, sigma_greater, sigma_retarded = out

        n = g_lesser.data.shape[0] + g_greater.data.shape[0] - 1
        ne = g_lesser.data.shape[0]

        if self.config.compute.num_bits is not None:
            nk = g_lesser.data.shape[1:-2]
        else:
            nk = g_lesser.data.shape[1:-1]

        if self.config.compute.num_bits is not None:
            _g_lesser = decompress(g_lesser.data[..., batch, :], g_lesser.bits)
            _g_greater = decompress(g_greater.data[..., batch, :], g_lesser.bits)
            _w_lesser = decompress(w_lesser.data[..., batch, :], g_lesser.bits)
            _w_greater = decompress(w_greater.data[..., batch, :], g_lesser.bits)
        else:
            _g_lesser = g_lesser.data[..., batch]
            _g_greater = g_greater.data[..., batch]
            _w_lesser = w_lesser.data[..., batch]
            _w_greater = w_greater.data[..., batch]

        g_x_fft = xp.fft.fftn(_g_lesser, (n,) + nk, axes=tuple(range(len(nk) + 1)))
        w_lesser_fft = xp.fft.fftn(_w_lesser, (n,) + nk, axes=tuple(range(len(nk) + 1)))
        w_greater_fft = xp.fft.fftn(
            _w_greater,
            (n,) + nk,
            axes=tuple(range(len(nk) + 1)),
        )

        sigma_x_fft = xp.multiply(g_x_fft, w_lesser_fft)
        sigma_x_fft -= xp.multiply(
            g_x_fft, w_greater_fft.conj()
        )  # negative energy part
        lesser = (
            self.prefactor
            * xp.fft.ifftn(sigma_x_fft, axes=tuple(range(len(nk) + 1)))[:ne]
        )
        if self.config.compute.num_bits is not None:
            sigma_lesser.data[..., batch, :] = compress(
                lesser
                + decompress(sigma_lesser.data[..., batch, :], sigma_lesser.bits),
                sigma_lesser.bits,
            )
        else:
            sigma_lesser.data[..., batch] += lesser

        g_x_fft = xp.fft.fftn(
            _g_greater,
            (n,) + nk,
            axes=tuple(range(len(nk) + 1)),
        )
        sigma_x_fft = xp.multiply(g_x_fft, w_greater_fft)
        sigma_x_fft -= xp.multiply(g_x_fft, w_lesser_fft.conj())  # negative energy part
        greater = (
            self.prefactor
            * xp.fft.ifftn(sigma_x_fft, axes=tuple(range(len(nk) + 1)))[:ne]
        )
        if self.config.compute.num_bits is not None:
            sigma_greater.data[..., batch, :] = compress(
                greater
                + decompress(sigma_greater.data[..., batch, :], sigma_greater.bits),
                sigma_greater.bits,
            )
        else:
            sigma_greater.data[..., batch] += greater

        # Compute retarded self-energy with a Hilbert transform.
        antihermitian = greater - lesser
        antihermitian_fft = xp.fft.fft(antihermitian, n, axis=0)

        sigma_x_fft = xp.multiply(antihermitian_fft, hilbert_kernel_fft)
        # negative energy part
        sigma_x_fft -= xp.multiply(antihermitian_fft, hilbert_kernel_fft.conj())

        retarded = (
            self.prefactor * xp.fft.ifft(sigma_x_fft, axis=0)[:ne] * self.kpoint_volume
        )

        if self.config.compute.num_bits is not None:
            sigma_retarded.data[..., batch, :] = compress(
                retarded
                + decompress(sigma_retarded.data[..., batch, :], sigma_retarded.bits),
                sigma_retarded.bits,
            )
        else:
            sigma_retarded.data[..., batch] += retarded

    def _compute_with_correction(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        batch: slice,
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        w_lesser : DSDBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSDBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.
        batch : slice
            The batch slice for the current computation.

        """
        sigma_lesser, sigma_greater, sigma_retarded = out
        ne = g_lesser.data.shape[0]

        # Lesser self-energy
        sl = self.prefactor * fft_correlate_kpoints(
            g_lesser.data[..., batch],
            -w_greater.data[..., batch].conj(),
        )
        sl[-ne:] += (
            self.prefactor
            * fft_convolve_kpoints(
                g_lesser.data[..., batch], w_lesser.data[..., batch]
            )[:ne]
        )

        # Greater self-energy
        sg = self.prefactor * fft_convolve_kpoints(
            g_greater.data[..., batch], w_greater.data[..., batch]
        )
        sg[:ne] += (
            self.prefactor
            * fft_correlate_kpoints(
                g_greater.data[..., batch],
                -w_lesser.data[..., batch].conj(),
            )[-ne:]
        )

        sigma_lesser.data[..., batch] += sl[-ne:]
        sigma_greater.data[..., batch] += sg[:ne]

        sigma_retarded.data[..., batch] += (
            self.prefactor
            * hilbert_transform(sl, sg, self.energies)
            * self.kpoint_volume
        )

    @profiler.profile(label="SigmaCoulombScreening", level="default", comm=comm)
    def compute(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        w_lesser : DSDBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSDBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.

        """

        with profiler.profile_range(
            label="SigmaCoulombScreening: block reorder", level="default", comm=comm
        ):

            # Save the block sizes for later.
            if self.big_block_sizes is None:
                self.big_block_sizes = w_lesser.block_sizes

            # Enforce that the block sizes are the same. NOTE: This triggers
            # a block-reordering in the DSDBSparse object.
            w_lesser.block_sizes = g_lesser.block_sizes
            w_greater.block_sizes = g_greater.block_sizes

            sigma_lesser, sigma_greater, sigma_retarded = out

        with profiler.profile_range(
            label="SigmaCoulombScreening: stack->nnz transpose",
            level="default",
            comm=comm,
        ):

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

        with profiler.profile_range(
            label="SigmaCoulombScreening: SSE computation", level="default", comm=comm
        ):

            # Because of padding there could be no ij elements

            if self.config.compute.num_bits is not None:
                ne = g_lesser.data.shape[0]
                nks = g_lesser.data.shape[1:-2]
                no = g_lesser.data.shape[-2]
            else:
                ne = g_lesser.data.shape[0]
                nks = g_lesser.data.shape[1:-1]
                no = g_lesser.data.shape[-1]

            nk = np.prod(nks)

            if no != 0:
                if xp.__name__ == "cupy":
                    cache.clear()
                    free_mempool()
                    free_memory, _ = xp.cuda.Device().mem_info
                    num_buffers = 35  # closer to 8 but overapproximating
                    avail_buffer_size = free_memory // num_buffers
                    batch_size = avail_buffer_size // (
                        2 * ne * nk * 50
                    )  # 16 bytes for complex128
                    batch_size = max(min(batch_size, no), 1)
                    batches = int(np.ceil(no / batch_size))
                    batch_size = int(np.ceil(no / batches))  # Balance last batch
                    # if self.batch_size is not None and batch_size < self.batch_size:
                    #     cache.clear()
                    self.batch_size = batch_size
                    if comm.rank == 0:
                        print(
                            f"Free GiB: {free_memory/(1024**3):.3f}, Batches: {batches}, Batch size: {batch_size}",
                            flush=True,
                        )
                        print(cache.show_info(), flush=True)
                else:
                    if self.batch_size is None:
                        # NOTE: This is a temporary solution. The batch size should be
                        # calculated in the configuration.
                        self.batch_size = no

                batch_counts, _ = get_section_sizes(
                    no,
                    int(np.ceil(no / self.batch_size)),
                )

                batch_displacements = np.cumsum(
                    np.concatenate(([0], np.array(batch_counts)))
                )

                if self.apply_hilbert_correction:
                    for start, end in zip(batch_displacements, batch_displacements[1:]):
                        self._compute_with_correction(
                            g_lesser,
                            g_greater,
                            w_lesser,
                            w_greater,
                            out,
                            slice(start, end),
                        )
                else:
                    n = 2 * ne - 1

                    # Add empty dimensions for each k-point.
                    energy_differences = (self.energies - self.energies[0]).reshape(
                        -1, *(len(nks) + 1) * (1,)
                    )

                    # NOTE: Same eta as in the other computation, but fewer
                    # ffts are computed in this case.
                    eta = (self.energies[1] - self.energies[0]) / 2
                    hilbert_kernel_fft = xp.fft.fft(
                        1 / (energy_differences + eta), n, axis=0
                    )
                    for start, end in zip(batch_displacements, batch_displacements[1:]):
                        self._compute_without_correction(
                            g_lesser,
                            g_greater,
                            w_lesser,
                            w_greater,
                            out,
                            slice(start, end),
                            hilbert_kernel_fft,
                        )
            cache.clear()
        # Transpose the matrices to stack distribution.
        with profiler.profile_range(
            label="SigmaCoulombScreening: nnz->stack transpose",
            level="default",
            comm=comm,
        ):
            for m in (w_lesser, w_greater):
                m.dtranspose(discard=True) if m.distribution_state != "stack" else None
            # NOTE: The electron Green's functions and self-energies must
            # not be transposed back to stack distribution, as they are
            # needed in nnz distribution for the other interactions.

        with profiler.profile_range(
            label="SigmaCoulombScreening: block reorder back",
            level="default",
            comm=comm,
        ):
            # Recover original block sizes.
            w_lesser.block_sizes = self.big_block_sizes
            w_greater.block_sizes = self.big_block_sizes
