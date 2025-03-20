# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

from qttools import NDArray, global_comm, sparse, stack_comm, xp
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


class SigmaFockDist(ScatteringSelfEnergy):
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

        # Create the DSDBSparse object.
        # TODO: This is pretty wasteful memory-wise.
        # Workaround: Use comm size as global stack shape.
        coulomb_matrix = compute_config.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=(stack_comm.size,),
        )
        coulomb_matrix.data = 0.0
        coulomb_matrix += coulomb_matrix_sparray
        coulomb_matrix.dtranspose()
        self.coulomb_matrix_data = coulomb_matrix.data[0]

    @profiler.profile(level="api")
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
        (sigma_retarded,) = out
        t0 = time.perf_counter()
        for m in (g_lesser, sigma_retarded):
            # These should both already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None
        t1 = time.perf_counter()
        if global_comm.rank == 0:
            print(f"SigmaFock: stack->nnz transpose time: {t1-t0}", flush=True)

        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded.data += xp.real(gl_density * self.coulomb_matrix_data)

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
