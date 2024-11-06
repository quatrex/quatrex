from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


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
            (comm.size,),
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
