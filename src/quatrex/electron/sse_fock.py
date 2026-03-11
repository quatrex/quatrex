# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.fft import fft_circular_convolve
from qttools.kernels.mixed_precision import compress, decompress
from qttools.profiling import Profiler
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


class SigmaFock(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

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
        coulomb_matrix: DSDBSparse,
        electron_energies: NDArray,
    ):
        """Initializes the bare Fock self-energy."""
        self.config = config
        self.energies = electron_energies
        self.kpoint_volume = np.prod(config.device.kpoint_grid)
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        (
            coulomb_matrix.dtranspose()
            if coulomb_matrix.distribution_state != "nnz"
            else None
        )
        self.coulomb_matrix_data = coulomb_matrix.data[0]

    @profiler.profile(label="SigmaFock", level="default", comm=comm)
    def compute(self, g_lesser: DSDBSparse, out: tuple[DSDBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        with profiler.profile_range(
            label="SigmaFock: stack->nnz transpose", level="default", comm=comm
        ):
            (sigma_retarded,) = out
            for m in (g_lesser, sigma_retarded):
                # These should both already be in nnz-distribution.
                m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the electron density by summing over energies.
        with profiler.profile_range(
            label="SigmaFock: SSE computation", level="default", comm=comm
        ):
            if g_lesser.data.shape[-1] != 0:
                if self.config.compute.num_bits is None:
                    gl_density = self.prefactor * g_lesser.data.sum(axis=0)
                    sigma_retarded.data += (
                        fft_circular_convolve(
                            gl_density,
                            self.coulomb_matrix_data,
                            axes=tuple(range(gl_density.ndim - 1)),
                        )
                        / self.kpoint_volume
                    )

                else:
                    gl_density = self.prefactor * decompress(
                        g_lesser.data, g_lesser.bits
                    ).sum(axis=0)
                    sigma_retarded.data += compress(
                        (
                            fft_circular_convolve(
                                gl_density,
                                decompress(self.coulomb_matrix_data, g_lesser.bits),
                                axes=tuple(range(gl_density.ndim - 1)),
                            )
                            / self.kpoint_volume
                        ),
                        g_lesser.bits,
                    )

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
