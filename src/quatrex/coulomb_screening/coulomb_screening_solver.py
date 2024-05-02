# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.
import numpy as np

from scipy import sparse

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.statistics import fermi_dirac

from qttools.datastructures.dbcsr import DBCSR


class CoulombScreeningSolver(SubsystemSolver):

    system = "coulomb_screening"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(quatrex_config)

        ...
