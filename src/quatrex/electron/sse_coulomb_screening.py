# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np
from scipy import interpolate
import finufft

from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from qttools.fft import fft_convolve, fft_convolve_kpoints, fft_correlate_kpoints
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import free_mempool
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()

if xp.__name__ == "cupy":
    cache = xp.fft.config.get_plan_cache()


def create_hilbert_kernel(energies: NDArray, nk: tuple, r: float) -> NDArray:
    """Creates the FFT(Hilbert kernel) for the given energy grid and oversampling ratio.
    
    Parameters
    ----------
    energies : NDArray
        The energy values.
    nk : tuple
        The k-point grid dimensions.
    r : float
        The oversampling ratio.

    Returns
    -------
    NDArray
        The Hilbert kernel and its FFT.
    """
    # liyongda (25 Mar 2026): Hilbert kernel dE should be the fine grid dE
    ne = energies.size
    ne_fine = int(r * ne)
    n_conv = 2*ne_fine - 1      # length of convolution output

    energies_fine = np.linspace(energies[0], energies[-1], ne_fine)

    # eta for removing the singularity. See Cauchy principal value.
    eta = (energies_fine[1] - energies_fine[0]) / 2

    # Add empty dimensions for each k-point.
    energy_differences = (energies_fine - energies_fine[0]).reshape(-1, *(len(nk) + 1) * (1,))

    # Create the Hilbert kernel in real space.    
    hilbert_kernel = 1 / (energy_differences + eta)

    # Transform to Fourier space.
    hilbert_kernel_fft = xp.fft.fft(hilbert_kernel, n_conv, axis=0)

    return hilbert_kernel, hilbert_kernel_fft


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

    hilbert_kernel, _ = create_hilbert_kernel(energies, nk, r=1)

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

        self.include_energy_renormalization = (
            config.coulomb_screening.include_energy_renormalization
            in ("self-energy", "both")
        )
        self.apply_hilbert_correction = (
            config.coulomb_screening.apply_hilbert_correction
        )

    def update_energies(self, new_energies: NDArray) -> None:
        """Updates the energies for the Coulomb screening.

        This is needed if the energies for the electron system change during
        the self-consistent loop.

        Parameters
        ----------
        new_energies : NDArray
            The new energies for the electron system.

        """
        self.energies = new_energies

    def _compute_without_correction(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        source_grid1: NDArray,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
        source_grid2: NDArray,
        out: tuple[DSDBSparse, ...],
        batch: slice,
        hilbert_kernel_fft: NDArray,
        target_grid: NDArray,
        use_adaptive: bool = False,
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        source_grid1: NDArray
            The energy grid corresponding to g_lesser and g_greater.
        w_lesser : DSDBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSDBSparse
            The greater screened Coulomb interaction.
        source_grid2: NDArray
            The energy grid corresponding to w_lesser and w_greater.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.
        batch : slice
            The batch slice for the current computation.
        hilbert_kernel_fft : NDArray
            The precomputed Hilbert kernel in Fourier space.
        target_grid: NDArray
            The energy grid corresponding to the output self-energies.
        use_adaptive: bool
            Whether to use adaptive interpolation for the convolution.
        """
        sigma_lesser, sigma_greater, sigma_retarded = out

        # setup parameters, n=length of fft output, ne=number of energy points, nk=number of k-points
        n = g_lesser.data.shape[0] + g_greater.data.shape[0] - 1
        ne = g_lesser.data.shape[0]
        nk = g_lesser.data.shape[1:-1]

        k = self.config.scba.adaptive_interpolation_order

        # oversampling ratio, how much to blow up the adaptive grid in interpolation
        r = self.config.scba.adaptive_interpolation_oversampling_ratio
        n_fine = int(r * ne)        # r could be 0.5
        n_conv_fine = 2*n_fine -1
        if use_adaptive:
            # self.prefactor is only computed 1x on init, assuming uniform energy grid
            adaptive_prefactor = self.prefactor / r  # because of interpolation to a finer grid, we need to divide by the oversampling ratio to keep the prefactor consistent
            fine_energies_g = np.linspace(source_grid1[0], source_grid1[-1], n_fine)
            fine_energies_w = np.linspace(source_grid2[0], source_grid2[-1], n_fine)
            # Interpolate g and w to the fine grid. This is needed for the FFT-based convolution.
            g_lesser_fine = interpolate.make_interp_spline(source_grid1, g_lesser.data[..., batch], axis=0, k=k)(fine_energies_g)
            g_greater_fine = interpolate.make_interp_spline(source_grid1, g_greater.data[..., batch], axis=0, k=k)(fine_energies_g)
            w_lesser_fine = interpolate.make_interp_spline(source_grid2, w_lesser.data[..., batch], axis=0, k=k)(fine_energies_w)
            w_greater_fine = interpolate.make_interp_spline(source_grid2, w_greater.data[..., batch], axis=0, k=k)(fine_energies_w)

            # liyongda (23 Mar 2026):
            # G grid is -15 to 5 eV
            # W grid is 0 to 20 eV
            # W data ends at 0 eV, but need to extrapolate to -15 eV
            #   it just takes a linear interpolation of the last segment --> straight line
            # if we need to extrapolate anything, set it to 0
            # w_lesser_fine [fine_energies < source_grid2[0]] = 0
            # w_greater_fine [fine_energies < source_grid2[0]] = 0

            # sigma_lesser
            sigma_x_fft = xp.multiply(xp.fft.fft(g_lesser_fine, n_conv_fine, axis=0), xp.fft.fft(w_lesser_fine, n_conv_fine, axis=0))
            sigma_x_fft -= xp.multiply(xp.fft.fft(g_lesser_fine, n_conv_fine, axis=0), xp.fft.fft(w_greater_fine, n_conv_fine, axis=0).conj())
            
            # liyongda (03 Mar 2026) todo: how many points do we target when going back to real space?
            lesser = adaptive_prefactor * xp.fft.ifft(sigma_x_fft, axis=0)[:n_fine]

            # sigma_greater
            sigma_x_fft = xp.multiply(xp.fft.fft(g_greater_fine, n_conv_fine, axis=0), xp.fft.fft(w_greater_fine, n_conv_fine, axis=0))
            sigma_x_fft -= xp.multiply(xp.fft.fft(g_greater_fine, n_conv_fine, axis=0), xp.fft.fft(w_lesser_fine, n_conv_fine, axis=0).conj())
            greater = adaptive_prefactor * xp.fft.ifft(sigma_x_fft, axis=0)[:n_fine]

            # interpolate to target grid
            ## liyongda (23 Mar 2026): what is the source grid for the interpolation here...?
            lesser_projected = interpolate.make_interp_spline(fine_energies_g, lesser, axis=0, k=k)(target_grid)
            greater_projected = interpolate.make_interp_spline(fine_energies_g, greater, axis=0, k=k)(target_grid)
            
            # update self-energy data structure
            sigma_lesser.data[..., batch] += lesser_projected
            sigma_greater.data[..., batch] += greater_projected


            # Compute retarded self-energy with a Hilbert transform
            # liyongda (01 Apr 2026): use the blown up dense grid for Hilbert transform (requires uniform grid)
            #   Hilbert kernel is already adjusted for the interpolation grid
            antihermitian = greater - lesser
            antihermitian_fft = xp.fft.fft(antihermitian, n_conv_fine, axis=0)

            sigma_x_fft = xp.multiply(antihermitian_fft, hilbert_kernel_fft)
            # negative energy part
            sigma_x_fft -= xp.multiply(antihermitian_fft, hilbert_kernel_fft.conj())

            retarded = adaptive_prefactor * xp.fft.ifft(sigma_x_fft, axis=0)[:n_fine] * self.kpoint_volume
            retarded_projected = interpolate.make_interp_spline(fine_energies_g, retarded, axis=0, k=k)(target_grid)
            sigma_retarded.data[..., batch] += retarded_projected

        else:
            # compute transforms
            g_x_fft = xp.fft.fftn(
                g_lesser.data[..., batch], (n,) + nk, axes=tuple(range(len(nk) + 1))
            )
            w_lesser_fft = xp.fft.fftn(
                w_lesser.data[..., batch], (n,) + nk, axes=tuple(range(len(nk) + 1))
            )
            w_greater_fft = xp.fft.fftn(
                w_greater.data[..., batch],
                (n,) + nk,
                axes=tuple(range(len(nk) + 1)),
            )

            # sigma_lesser: point-wise multiplication in Fourier space and inverse transform to get convolution
            sigma_x_fft = xp.multiply(g_x_fft, w_lesser_fft)
            sigma_x_fft -= xp.multiply(
                g_x_fft, w_greater_fft.conj()
            )  # negative energy part
            lesser = (
                self.prefactor
                * xp.fft.ifftn(sigma_x_fft, axes=tuple(range(len(nk) + 1)))[:ne]
            )
            sigma_lesser.data[..., batch] += lesser

            # compute transforms
            g_x_fft = xp.fft.fftn(
                g_greater.data[..., batch],
                (n,) + nk,
                axes=tuple(range(len(nk) + 1)),
            )
            # sigma_greater: point-wise multiplication in Fourier space and inverse transform to get convolution
            sigma_x_fft = xp.multiply(g_x_fft, w_greater_fft)
            sigma_x_fft -= xp.multiply(g_x_fft, w_lesser_fft.conj())  # negative energy part
            greater = (
                self.prefactor
                * xp.fft.ifftn(sigma_x_fft, axes=tuple(range(len(nk) + 1)))[:ne]
            )
            sigma_greater.data[..., batch] += greater

        if self.include_energy_renormalization:
            # Compute retarded self-energy with a Hilbert transform.
            antihermitian = greater - lesser
            antihermitian_fft = xp.fft.fft(antihermitian, n, axis=0)

            sigma_x_fft = xp.multiply(antihermitian_fft, hilbert_kernel_fft)
            # negative energy part
            sigma_x_fft -= xp.multiply(antihermitian_fft, hilbert_kernel_fft.conj())

            # NOTE: The anti-Hermitian (sigma_greater - sigma_lesser)
            # part of the retarded self-energy is added outside in the
            # main SCBA loop, so it is not added here.
            sigma_retarded.data[..., batch] += (
                self.prefactor
                * xp.fft.ifft(sigma_x_fft, axis=0)[:ne]
                * self.kpoint_volume
            )

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

        if self.include_energy_renormalization:
            # NOTE: The anti-Hermitian (sigma_greater - sigma_lesser)
            # part of the retarded self-energy is added outside in the
            # main SCBA loop, so it is not added here.
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
        source_grid1: NDArray,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
        source_grid2: NDArray,
        out: tuple[DSDBSparse, ...],
        target_grid: NDArray,
        use_adaptive: bool = False,
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
            if g_greater.data.shape[-1] != 0:
                if xp.__name__ == "cupy":
                    free_mempool()
                    free_memory, _ = xp.cuda.Device().mem_info
                    num_buffers = 12  # closer to 8 but overapproximating
                    avail_buffer_size = free_memory // num_buffers
                    ne = g_lesser.data.shape[0]
                    nk = np.prod(g_lesser.data.shape[1:-1])
                    no = g_lesser.data.shape[-1]
                    batch_size = avail_buffer_size // (
                        2 * ne * nk * 16
                    )  # 16 bytes for complex128
                    batch_size = max(min(batch_size, no), 1)
                    batches = int(np.ceil(no / batch_size))
                    batch_size = int(np.ceil(no / batches))  # Balance last batch
                    if self.batch_size is not None and batch_size < self.batch_size:
                        cache.clear()
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
                        self.batch_size = g_greater.data.shape[-1]

                batch_counts, _ = get_section_sizes(
                    g_greater.data.shape[-1],
                    int(np.ceil(g_greater.data.shape[-1] / self.batch_size)),
                )

                batch_displacements = np.cumsum(
                    np.concatenate(([0], np.array(batch_counts)))
                )

                if self.apply_hilbert_correction:
                    if target_grid is not None:
                        raise NotImplementedError(
                            "Scattering Self Energy computation Hilbert correction with non-uniform target grid is not implemented. Please use a uniform grid."
                        )
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
                    nk = g_lesser.data.shape[1:-1]
                    r = self.config.scba.adaptive_interpolation_oversampling_ratio if use_adaptive else 1
                    _, hilbert_kernel_fft = create_hilbert_kernel(self.energies, nk, r)

                    for start, end in zip(batch_displacements, batch_displacements[1:]):
                        self._compute_without_correction(
                            g_lesser = g_lesser,
                            g_greater = g_greater,
                            source_grid1 = source_grid1,
                            w_lesser = w_lesser,
                            w_greater = w_greater,
                            source_grid2 = source_grid2,
                            out = out,
                            batch = slice(start, end),
                            hilbert_kernel_fft = hilbert_kernel_fft,
                            target_grid = target_grid,
                            use_adaptive = use_adaptive,
                        )

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
