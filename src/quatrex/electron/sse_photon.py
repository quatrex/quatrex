# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.routines import bd_sandwich_distr
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


def _update_sigma_retarded_from_lesser_greater(
    sigma_lesser: DSDBSparse,
    sigma_greater: DSDBSparse,
    sigma_retarded: DSDBSparse,
) -> None:
    sigma_retarded.data = sigma_retarded.data.real + 0.5 * 1j * (
        sigma_greater.data.imag - sigma_lesser.data.imag
    )


class SigmaPhoton(ScatteringSelfEnergy):
    """Computes the electron-photon self-energy.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.
    electron_energies : NDArray, optional
        The electron energies.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        sparsity_pattern: sparse.coo_matrix,
        electron_energies: NDArray | None = None,
    ) -> None:
        """Initializes the self-energy."""
        self.compute_config = compute_config
        self.model = quatrex_config.photon.model
        if quatrex_config.photon.model == "negf":
            raise NotImplementedError

        if quatrex_config.photon.model == "pseudo-scattering":

            if electron_energies is None:
                raise ValueError(
                    "Electron energies must be provided for deformation potential model."
                )

            self.electron_energies = electron_energies

            # Load block sizes.
            self.block_sizes = get_host(
                distributed_load(quatrex_config.input_dir / "block_sizes.npy").astype(
                    xp.int32
                )
            )

            self.interaction_matrix = compute_config.dsdbsparse_type.from_sparray(
                sparsity_pattern.astype(xp.float32),
                block_sizes=self.block_sizes,
                global_stack_shape=(comm.stack.size,),
                symmetry=quatrex_config.scba.symmetric,
                symmetry_op=-xp.conj,
            )
            self.interaction_matrix.data = distributed_load(
                quatrex_config.input_dir / self.photon.interaction_matrix_file
            )

            return

        raise ValueError(f"Unknown photon model: {quatrex_config.photon.model}")

    @profiler.profile(level="basic")
    def compute(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        d_lesser: NDArray | DSDBSparse | None = None,
        d_greater: NDArray | DSDBSparse | None = None,
        photon_energies: NDArray | None = None,
        light_intensity: NDArray | None = None,
    ) -> None:
        """Computes the electron-photon self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.

        """
        if self.model == "pseudo-scattering":
            if photon_energies is None:
                raise ValueError("Photon energies must be provided.")
            if light_intensity is None:
                raise ValueError("Light intensity must be provided.")
            return self._compute_pseudo_scattering(
                g_lesser, g_greater, photon_energies, light_intensity, out
            )

    def _compute_pseudo_scattering(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        photon_energies: NDArray,
        light_intensity: NDArray,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Computes the photon self-energy of a coherent and monochromatic light beam.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        out : tuple[DSDBSparse, ...]
            The lesser, greater and retarded self-energies.

        """
        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded):
            # These should ideally already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "stack" else None

        local_blocks, _ = get_section_sizes(
            len(self.interaction_matrix.block_sizes), comm.block.size
        )
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        lesser = self.compute_config.dsdbsparse_type.empty_like(sigma_lesser)
        greater = self.compute_config.dsdbsparse_type.empty_like(sigma_greater)

        prefactor = 1

        # locally compute the sigma^<> = M G^<> M for local energies

        bd_sandwich_distr(
            self.interaction_matrix,
            g_lesser,
            out=lesser,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        lesser.data *= prefactor  # M@G^<@M

        bd_sandwich_distr(
            self.interaction_matrix,
            g_greater,
            out=greater,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        greater.data *= prefactor  # M@G^>@M

        for m in (sigma_lesser, sigma_greater, sigma_retarded, lesser, greater):
            m.dtranspose() if m.distribution_state != "nnz" else None

        if photon_energies.shape > 0:
            n = g_lesser.data.shape[0] + photon_energies.shape[0] - 1
            ne = g_lesser.data.shape[0]

            lesser_fft = xp.fft.fft(lesser.data.T, n, axis=1)
            greater_fft = xp.fft.fft(greater.data.T, n, axis=1)

            d_greater_fft = xp.zeros(photon_energies.shape[0], dtype=lesser_fft.dtype)

            d_greater_fft[:, int(photon_energies / self.energy_resolution)] = (
                light_intensity / photon_energies * (-1j)
            )

            tmp = xp.fft.ifft(lesser_fft * d_greater_fft.conj(), axis=1)[:, :ne]
            sigma_lesser.data = tmp.T

            tmp = xp.fft.ifft(greater_fft * d_greater_fft, axis=1)[:, :ne]
            sigma_greater.data = tmp.T
        else:
            # energy + hbar * omega
            # <=> xp.roll(self.electron_energies, -upshift)[:-upshift]
            self.upshift = xp.argmin(
                xp.abs(
                    self.electron_energies
                    - (self.electron_energies[0] + photon_energies)
                )
            )
            # energy - hbar * omega
            # <=> xp.roll(self.electron_energies, downshift)[downshift:]
            self.downshift = (
                self.electron_energies.size
                - xp.argmin(
                    xp.abs(
                        self.electron_energies
                        - (self.electron_energies[-1] - photon_energies)
                    )
                )
                - 1
            )

            sigma_greater.data[self.upshift :, ...] += greater.data[
                : -self.upshift, ...
            ]
            sigma_greater.data[: -self.downshift :, ...] += greater.data[
                self.downshift :, ...
            ]

            sigma_lesser.data[self.upshift :, ...] += lesser.data[: -self.upshift, ...]
            sigma_lesser.data[: -self.downshift :, ...] += lesser.data[
                self.downshift :, ...
            ]

        _update_sigma_retarded_from_lesser_greater(
            sigma_lesser, sigma_greater, sigma_retarded
        )
