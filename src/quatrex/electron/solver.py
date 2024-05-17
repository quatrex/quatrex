# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
import numpy as np
from qttools.datastructures.dsbcsr import DSBCSR
from qttools.utils import mpi_utils
from scipy import sparse

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver


class ElectronSolver(SubsystemSolver):
    """Solves for the lesser electron Green's function."""

    system = "electron"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(quatrex_config, compute_config, energies)

        # load Hamiltonian matrix and block_sizes vector, raise error if not found
        self.hamiltonian_sparray = mpi_utils.distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        )
        self.block_sizes = mpi_utils.distributed_load(
            quatrex_config.input_dir / "block_sizes.npy"
        )

        # Check wether total sizes from blocksizes and hamiltonian shape match
        if self.block_sizes.sum() != self.hamiltonian_sparray.shape[0]:
            raise ValueError(
                f"Sum of block sizes does not match Hamiltonian size. {self.block_sizes.sum()} != {self.hamiltonian_sparray.shape[0]}"
            )

        # load overlap matrix, set to identity if None
        try:
            self.overlap_sparray = mpi_utils.distributed_load(
                quatrex_config.input_dir / "overlap.npz"
            )
        except OSError:
            self.overlap_sparray = sparse.eye(
                self.hamiltonian_sparray.shape[0], format="coo"
            )

        if self.overlap_sparray.shape != self.hamiltonian_sparray.shape:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )

        self.bare_system_matrix = DSBCSR.from_sparray(
            self.overlap_sparray,
            block_sizes=self.block_sizes,
            global_stack_shape=(self.energies.size,),
            densify_blocks=[(0, 0), (-1, -1)],
        )
        self.bare_system_matrix.data[:] = (
            mpi_utils.get_local_slice(self.energies) * self.bare_system_matrix.data[:]
        )
        self.bare_system_matrix -= self.hamiltonian_sparray

        # load potential matrix, set to diagonal zero if None
        try:
            self.potential = mpi_utils.distributed_load(
                quatrex_config.input_dir / "potential.npz"
            )
        except OSError:
            # File does not exist. Set potential to zero.
            self.potential = 0 * sparse.eye(self.hamiltonian.shape[0], format="coo")

        if self.potential.shape != self.hamiltonian.shape:
            raise ValueError(
                "Potential matrix and Hamiltonian matrix have different shapes."
            )

    def apply_obc(self, *args, **kwargs) -> None:
        ...

    def assemble_system_matrix(self) -> None:
        ...

    def solve(self) -> None:
        ...
