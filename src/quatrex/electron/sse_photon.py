# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from scipy.constants import (
    angstrom,
    electron_mass,
    elementary_charge,
    epsilon_0,
    hbar,
    speed_of_light,
)

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.routines import bd_sandwich_distr
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.electron import ElectronSolver

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
            self.photon_energy = quatrex_config.photon.photon_energy

            if electron_energies is None:
                raise ValueError(
                    "Electron energies must be provided for pseudo-scattering model."
                )

            self.upshift = int(
                self.photon_energy / (electron_energies[1] - electron_energies[0])
            )
            self.downshift = self.upshift

            hamiltonian_sparray, block_sizes = ElectronSolver.load_hamiltonian(
                quatrex_config
            )
            orbital_positions = distributed_load(quatrex_config.input_dir / "grid.npy")

            polarization = xp.array(quatrex_config.photon.polarization, dtype=float)
            intensity = quatrex_config.photon.light_intensity

            polarization = polarization / xp.linalg.norm(polarization)
            m_sparray = self.compute_interaction_matrix(
                polarization, hamiltonian_sparray, orbital_positions, intensity
            )

            self.interaction_matrix = compute_config.dsdbsparse_type.from_sparray(
                m_sparray,
                block_sizes=block_sizes,
                global_stack_shape=(comm.stack.size,),
                symmetry=quatrex_config.scba.symmetric,
                symmetry_op=lambda a: -a.conj(),
            )

            return

        raise ValueError(f"Unknown photon model: {quatrex_config.photon.model}")

    def compute_interaction_matrix(
        self,
        polarization: NDArray,
        hamiltonian_sparray: sparse.coo_matrix,
        orbital_positions: NDArray,
        intensity: float,
    ) -> sparse.coo_matrix:
        prefactor = (
            angstrom
            * 1j
            * (hbar * elementary_charge / electron_mass)
            / (2.0e0 * epsilon_0 * speed_of_light)
        )
        prefactor *= intensity / elementary_charge / self.photon_energy**2
        interaction_matrix = sparse.coo_matrix(hamiltonian_sparray)
        for s, (i, j) in enumerate(
            zip(hamiltonian_sparray.row, hamiltonian_sparray.col)
        ):
            interaction_matrix.data[s] = (
                xp.dot((orbital_positions[i] - orbital_positions[j]), polarization)
                * hamiltonian_sparray.data[s]
            )
        interaction_matrix.data *= prefactor
        return interaction_matrix

    @profiler.profile(level="basic")
    def compute(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        d_lesser: NDArray | DSDBSparse | None = None,
        d_greater: NDArray | DSDBSparse | None = None,
    ):
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

            return self._compute_pseudo_scattering(g_lesser, g_greater, out)

    def _compute_pseudo_scattering(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
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

        lesser = self.compute_config.dsdbsparse_type.zeros_like(sigma_lesser)
        greater = self.compute_config.dsdbsparse_type.zeros_like(sigma_greater)

        # Enforce that the block sizes are the same. NOTE: This triggers
        # a block-reordering in the DSDBSparse object.
        self.interaction_matrix.block_sizes = g_lesser.block_sizes

        # locally compute the sigma^<> = M G^<> M for local energies

        bd_sandwich_distr(
            self.interaction_matrix,
            g_lesser,
            out=lesser,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        # M@G^<@M

        bd_sandwich_distr(
            self.interaction_matrix,
            g_greater,
            out=greater,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        # M@G^>@M

        for m in (sigma_lesser, sigma_greater, sigma_retarded, lesser, greater):
            m.dtranspose() if m.distribution_state != "nnz" else None

        num_energies = sigma_greater.data.shape[0]

        if self.upshift > num_energies or self.downshift > num_energies:
            raise RuntimeError

        sigma_greater.data[self.upshift :, ...] += greater.data[: -self.upshift, ...]
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
