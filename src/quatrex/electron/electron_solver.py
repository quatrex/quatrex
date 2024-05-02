# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.
import os

import numpy as np

from scipy import sparse


from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.statistics import fermi_dirac
from quatrex.utils.dist_utils import distributed_load

from qttools.datastructures.dbcsr import DBCSR


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
        super().__init__(quatrex_config)

        self.local_energies = energies

        # load Hamiltonian matrix, raise error if not found
        hamiltonian_path = quatrex_config.input_dir / "hamiltonian.npz"
        if os.path.isfile(hamiltonian_path):
            self.hamiltonian = distributed_load(
                quatrex_config.input_dir / "hamiltonian.npz"
            )
            self.hamiltonian_dbsparse = DBCSR.from_sparray(
                self.hamiltonian, stackshape=self.local_energies.shape
            )
        else:
            raise FileNotFoundError('Hamiltonian file not found: "hamiltonian.npz"')

        # load overlap matrix, set to identity if None
        overlap_path = quatrex_config.input_dir / "overlap.npz"
        if os.path.isfile(overlap_path):
            self.overlap = distributed_load(overlap_path)
        else:
            self.overlap = sparse.eye(self.hamiltonian.shape[0], format="coo")
        self.overlap_dbsparse = DBCSR.from_sparray(self.overlap, stackshape=(1,))

        # load potential matrix, set to diagonal zero if None
        potential_path = quatrex_config.input_dir / "potential.npz"
        if os.path.isfile(potential_path):
            self.potential = distributed_load(potential_path)
        else:
            self.potential = 0 * sparse.eye(self.hamiltonian.shape[0], format="coo")
        self.potential_dbsparse = DBCSR.from_sparray(self.potential, stackshape=(1,))
