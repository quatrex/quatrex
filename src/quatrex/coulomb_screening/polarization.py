# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the Polarization class."""

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from qttools.fft import fft_convolve, fft_correlate_kpoints
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import clear_fft_plan_cache, free_mempool
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.grid import get_equal_spacing

profiler = Profiler()

if xp.__name__ == "cupy":
    cache = xp.fft.config.get_plan_cache()


def hilbert_transform(a: NDArray, energies: NDArray) -> NDArray:
    r"""Computes the Hilbert transform of the array a, assuming the symmetries of the
    polarization, i.e \([P^{\lessgtr}_{ij}(\omega)]^{\dagger} = -P^{\gtrless}_{ij}(-\omega)\).
    This becomes \(a(-\omega)=a^{*}(\omega)\), where a is \(a=P^>-P^<\).

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    a : NDArray
        The array to transform.
    energies : NDArray
        The energy values corresponding to the first axis of a.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    # eta for removing the singularity. See Cauchy principal value.
    energy_differences = xp.expand_dims(energies - energies[0], tuple(range(1, a.ndim)))
    # Set energy differences to inf at the singularity to avoid division by zero.
    energy_differences[0] = xp.inf
    ne = energies.size

    hilbert_kernel = 1 / energy_differences
    b = fft_convolve(a, hilbert_kernel)[:ne]
    # Negative frequencies of a
    b += fft_convolve(a[::-1].conj(), hilbert_kernel)[-ne:]
    # Negative frequencies of the kernel
    hilbert_kernel = -hilbert_kernel[::-1]
    b += fft_convolve(a, hilbert_kernel)[-ne:]

    return b


class PCoulombScreening(ScatteringSelfEnergy):
    """Computes the dynamic polarization from the electronic system.

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object.
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(
        self, config: QuatrexConfig, coulomb_screening_energies: NDArray
    ) -> None:
        """Initializes the polarization."""
        self.energies = coulomb_screening_energies
        self.kpoint_volume = np.prod(config.device.kpoint_grid)
        self.ne = len(self.energies)
        self.prefactor = (
            -1j
            / (xp.pi)
            * xp.abs(get_equal_spacing(self.energies))
            / self.kpoint_volume
        )
        self.num_batches = config.compute.convolve.num_batches

        self.align_to_complex_axes = (
            config.coulomb_screening.align_polarization_to_complex_axes
        )
        self.include_energy_renormalization = (
            config.coulomb_screening.include_energy_renormalization
            in ("polarization", "both")
        )

    @profiler.profile(label="PCoulombScreening", level="default", comm=comm)
    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the polarization. The order is
            p_lesser, p_greater, p_retarded_hermitian.

        """
        p_lesser, p_greater, p_retarded_hermitian = out

        # Barrier to synchronize ranks.
        with profiler.profile_range(
            label="PCoulombScreening: stack->nnz transpose", level="default", comm=comm
        ):

            # Transpose the matrices to nnz distribution.
            for m in (g_lesser, g_greater):
                # These should ideally already be in nnz-distribution.
                m.dtranspose() if m.distribution_state != "nnz" else None
            for m in (p_lesser, p_greater):
                # These only need the correct shape, so discard the data.
                m.dtranspose(discard=True) if m.distribution_state != "nnz" else None
            if self.include_energy_renormalization:
                (
                    p_retarded_hermitian.dtranspose(discard=True)
                    if (p_retarded_hermitian.distribution_state != "nnz")
                    else None
                )

        with profiler.profile_range(
            label="PCoulombScreening: Polarization computation",
            level="default",
            comm=comm,
        ):

            # Because of padding there could be no ij elements.
            if p_greater.data.shape[-1] != 0:
                clear_fft_plan_cache()
                free_mempool()

                batch_counts, _ = get_section_sizes(
                    p_greater.data.shape[-1], self.num_batches
                )

                batch_displacements = np.cumsum(
                    np.concatenate(([0], np.array(batch_counts)))
                )

                for start, end in zip(batch_displacements, batch_displacements[1:]):
                    batch = slice(start, end)

                    p_g_full = self.prefactor * fft_correlate_kpoints(
                        g_greater.data[..., batch], -g_lesser.data[..., batch].conj()
                    )
                    p_l_full = -p_g_full[::-1].conj()
                    # Fill the matrices with the data. Take second part
                    # of the energy convolution. # Note that only the
                    # anti-hermitian part is computed here.
                    p_lesser.data[..., batch] = p_l_full[self.ne - 1 :]
                    p_greater.data[..., batch] = p_g_full[self.ne - 1 :]

                    if self.include_energy_renormalization:
                        p_retarded_hermitian.data[..., batch] = (
                            -(self.prefactor / 2)
                            * (
                                hilbert_transform(
                                    (
                                        p_greater.data[..., batch]
                                        - p_lesser.data[..., batch]
                                    ),
                                    self.energies,
                                )
                            )
                            * self.kpoint_volume
                        )

                clear_fft_plan_cache()
                free_mempool()

        with profiler.profile_range(
            label="PCoulombScreening: nnz->stack transpose", level="default", comm=comm
        ):

            # Transpose the matrices to stack distribution.
            for m in (p_lesser, p_greater):
                m.dtranspose() if m.distribution_state != "stack" else None
            if self.include_energy_renormalization:
                (
                    p_retarded_hermitian.dtranspose()
                    if (p_retarded_hermitian.distribution_state != "stack")
                    else None
                )
            # NOTE: The Green's functions must not be transposed back to
            # stack distribution, as they are needed in nnz distribution for
            # the other interactions.

        # Enforce anti-Hermitian symmetry and calculate Pr.
        with profiler.profile_range(
            label="PCoulombScreening: Symmetrization", level="default", comm=comm
        ):
            if p_lesser.symmetry is None:
                p_lesser.symmetrize("skew-hermitian")
                p_greater.symmetrize("skew-hermitian")
                p_retarded_hermitian.symmetrize("hermitian")

            if not self.include_energy_renormalization:
                p_retarded_hermitian.data[:] = 0

            # Discard the real part of lesser/greater and imag part of retarded
            if self.align_to_complex_axes:
                p_lesser.data.real = 0
                p_greater.data.real = 0
                p_retarded_hermitian.data.imag = 0
