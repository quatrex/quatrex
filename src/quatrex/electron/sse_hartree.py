# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import distributed_load
from quatrex.bandstructure.band_edges import local_band_edges
from quatrex.core.observables import density
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


class SigmaHartree(ScatteringSelfEnergy):
    """Computes the bare Hartree self-energy.

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
        coulomb_matrix: DSDBSparse,
        electron_energies: NDArray,
        electron_potential: NDArray,
    ):
        """Initializes the bare Hartree self-energy."""
        self.energies = electron_energies
        self.prefactor = 1 / (xp.pi) * (self.energies[1] - self.energies[0])

        (
            coulomb_matrix.dtranspose()
            if coulomb_matrix.distribution_state != "stack"
            else None
        )
        # Gamma-point Coulomb matrix
        # TODO: This is only for a single transverse direction.
        self.gamma_coulomb_matrix = coulomb_matrix.stack[0, 0]

        # Guess for the mid-gap energies, will be updated in the first iteration.
        self.mid_gap_energies = (
            quatrex_config.electron.left_fermi_level + electron_potential
        )

        lattice_vectors = distributed_load(
            quatrex_config.input_dir / "lattice_vectors.npy"
        )
        # TODO: So far this only works for 2D systems.
        # Unit cell area/volume in cm^2
        unit_cell_area = xp.linalg.det(lattice_vectors[:2, :2]) * 1e-16  # A^2 to cm^2
        # Charge per unit area/volume
        doping = quatrex_config.electron.doping * unit_cell_area

        # States per unit area/volume
        uc_size = quatrex_config.device.neighbor_cell_cutoff[
            "xyz".index(quatrex_config.device.transport_direction)
        ]
        block_size_0 = coulomb_matrix.block_sizes[0]
        assert (
            block_size_0 % uc_size == 0
        ), "Block size must be divisible by unit cell size."
        num_states = block_size_0 // uc_size

        # Doping per state
        self.doping_per_state = doping / num_states

        self.hartree_potential = xp.zeros(
            coulomb_matrix.shape[-1], dtype=coulomb_matrix.dtype
        )
        self.mem_factor = 0.01

        unit_cells_per_block = quatrex_config.device.neighbor_cell_cutoff[
            "xyz".index(quatrex_config.device.transport_direction)
        ]
        small_block_size = block_size_0 // unit_cells_per_block
        assert (
            block_size_0 % small_block_size == 0
        ), "Block size must be divisible by small block size."
        self.ucpb = unit_cells_per_block
        self.sbs = small_block_size

        self.compute_counter = 0

    @profiler.profile(level="api")
    def compute(
        self, g_retarded: DSDBSparse, g_lesser: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the Hartree self-energy.

        Parameters
        ----------
        g_retarded : DSDBSparse
            The retarded Green's function.
        g_lesser : DSDBSparse
            The lesser Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.
        """
        sigma_retarded = out[0]
        # For the Hartree self-energy, we need the mid-gap energies to determine the excess charge.
        ldos = density(g_retarded)
        # Mean over k-points
        ldos = ldos.mean(axis=tuple(range(1, ldos.ndim - 1)))
        vb, cb = local_band_edges(ldos, self.energies, self.mid_gap_energies)
        self.mid_gap_energies = (vb + cb) / 2

        # Compute the Charge density summed over all k-points and energies
        # TODO: This does not work in the 'nnz' distribution, how to solve this?
        # Quick fix: transpose the coulomb matrix to the 'stack' distribution, but this is not ideal.
        (g_lesser.dtranspose() if g_lesser.distribution_state != "stack" else None)
        electron_density = density(g_lesser)
        excess_charge = xp.zeros(g_lesser.shape[-1], dtype=g_lesser.dtype)
        for i, mg in enumerate(self.mid_gap_energies):
            mask = self.energies > mg
            if xp.any(mask):
                excess_charge[i] = self.prefactor * electron_density[mask, ..., i].sum()

        hartree_potential = xp.zeros_like(excess_charge)
        # Perform the matrix-vector product to get the Hartree potential.
        block_sizes = g_lesser.block_sizes
        num_blocks = len(block_sizes)
        for i in range(num_blocks):
            row_start = sum(block_sizes[:i])
            row_end = row_start + block_sizes[i]
            for j in range(max(0, i - 1), min(num_blocks, i + 2)):
                col_start = sum(block_sizes[:j])
                col_end = col_start + block_sizes[j]
                hartree_potential[
                    row_start:row_end
                ] += self.gamma_coulomb_matrix.blocks[i, j] @ (
                    excess_charge[col_start:col_end] - self.doping_per_state
                )
            # hartree_potential[row_start:row_end] += self.gamma_coulomb_matrix.blocks[i, i] @ (excess_charge[row_start:row_end] - self.doping_per_state)
        # TODO: Fix boundary conditions for the multiplication!
        # Corrections for the boundary conditions, e.g. for a infinite system.
        # Tile the excess charge of the small block size to the full block size, then perform the multiplication with the first and last block of the Coulomb matrix.
        # lead_charge = np.tile(excess_charge[:self.sbs], self.ucpb)
        # hartree_potential[:block_sizes[0]] += self.gamma_coulomb_matrix.blocks[1, 0] @ (lead_charge - self.doping_per_state)
        # lead_charge = np.tile(excess_charge[-self.sbs:], self.ucpb)
        # hartree_potential[-block_sizes[-1]:] += self.gamma_coulomb_matrix.blocks[num_blocks-2, num_blocks-1] @ (lead_charge - self.doping_per_state)

        # Can maybe try some other update scheme here, e.g. mixing with previous iteration or using a damping factor.
        self.hartree_potential += self.mem_factor * hartree_potential
        # self.hartree_potential = self.mem_factor * hartree_potential
        if comm.rank == 0:
            np.save(f"excess_charge_{self.compute_counter}.npy", excess_charge)
            np.save(
                f"doping_per_state_{self.compute_counter}.npy", self.doping_per_state
            )
            np.save(f"hartree_potential_{self.compute_counter}.npy", hartree_potential)
        self.compute_counter += 1

        # NOTE: To do this assignment, sigma_retarded should already be in the 'stack' distribution
        (
            sigma_retarded.dtranspose()
            if sigma_retarded.distribution_state != "stack"
            else None
        )
        sigma_retarded += sparse.diags(self.hartree_potential, format="csr")
