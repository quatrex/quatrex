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
        if quatrex_config.photon.model == "negf":
            raise NotImplementedError

        if quatrex_config.photon.model == "pseudo-scattering":
            if electron_energies is None:
                raise ValueError("Electron energies must be provided.")
            self.photon_energy = quatrex_config.photon.photon_energy

            self.monochromatic_injection = quatrex_config.photon.monochromatic_injection
            self.light_intensity = quatrex_config.photon.light_intensity

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

            # energy + hbar * omega
            # <=> xp.roll(self.electron_energies, -upshift)[:-upshift]
            self.upshift = xp.argmin(
                xp.abs(electron_energies - (electron_energies[0] + self.photon_energy))
            )
            # energy - hbar * omega
            # <=> xp.roll(self.electron_energies, downshift)[downshift:]
            self.downshift = (
                electron_energies.size
                - xp.argmin(
                    xp.abs(
                        electron_energies - (electron_energies[-1] - self.photon_energy)
                    )
                )
                - 1
            )

            self.valid_slice = (
                slice(self.downshift, -self.upshift)
                if self.upshift != 0
                else slice(None)
            )

            totalshift = self.upshift + self.downshift

            self.upslice = slice(None) if totalshift == 0 else slice(-totalshift)
            self.downslice = slice(totalshift, None)

            return

        raise ValueError(f"Unknown photon model: {quatrex_config.photon.model}")

    @profiler.profile(level="basic")
    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
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
        if self.monochromatic_injection:
            return self._compute_monochrome(g_lesser, g_greater, out)

    def _compute_monochrome(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
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
            m.dtranspose() if m.distribution_state != "nnz" else None

        local_blocks, _ = get_section_sizes(
            len(self.interaction_matrix.block_sizes), comm.block.size
        )
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        tmp = self.compute_config.dsdbsparse_type.empty_like(sigma_lesser)

        prefactor = self.light_intensity

        bd_sandwich_distr(
            self.interaction_matrix,
            g_lesser,
            out=tmp,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        tmp.data *= prefactor
        sigma_lesser.data[self.upshift :, ...] += tmp.data[: -self.upshift, ...]
        sigma_lesser.data[: -self.downshift :, ...] += tmp.data[self.downshift :, ...]

        bd_sandwich_distr(
            self.interaction_matrix,
            g_greater,
            out=tmp,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        tmp.data *= prefactor
        sigma_greater.data[self.upshift :, ...] += tmp.data[: -self.upshift, ...]
        sigma_greater.data[: -self.downshift :, ...] += tmp.data[self.downshift :, ...]

        _update_sigma_retarded_from_lesser_greater(
            sigma_lesser, sigma_greater, sigma_retarded
        )
