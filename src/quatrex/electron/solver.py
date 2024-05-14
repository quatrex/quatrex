# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import os

import numpy as np
from qttools.datastructures.dbcsr import DBCSR
from qttools.utils.mpi_utils import distributed_load
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

        # load Hamiltonian matrix, raise error if not found
        hamiltonian_path = quatrex_config.input_dir / "hamiltonian.npz"
        if os.path.isfile(hamiltonian_path):
            self.hamiltonian_sparray = distributed_load(
                quatrex_config.input_dir / "hamiltonian.npz"
            )
            self.hamiltonian = DBCSR.from_sparray(
                self.hamiltonian_sparray, stackshape=(1,)
            )
        else:
            raise FileNotFoundError('Hamiltonian file not found: "hamiltonian.npz"')

        # load overlap matrix, set to identity if None
        overlap_path = quatrex_config.input_dir / "overlap.npz"
        if os.path.isfile(overlap_path):
            self.overlap_sparray = distributed_load(overlap_path)
        else:
            self.overlap_sparray = sparse.eye(
                self.hamiltonian_sparray.shape[0], format="coo"
            )
        self.overlap = DBCSR.from_sparray(
            self.overlap_sparray, stackshape=self.local_energies.shape
        )

        self.system_matrix = self.local_energies * self.overlap - self.hamiltonian

        # load potential matrix, set to diagonal zero if None
        potential_path = quatrex_config.input_dir / "potential.npz"
        if os.path.isfile(potential_path):
            self.potential = distributed_load(potential_path)
        else:
            self.potential = 0 * sparse.eye(self.hamiltonian.shape[0], format="coo")
        self.potential_dbsparse = DBCSR.from_sparray(self.potential, stackshape=(1,))

    def apply_obc(self, *args, **kwargs) -> None:
        ...

    def assemble_system_matrix(self) -> None:
        ...

    def solve(self) -> None:
        ...
