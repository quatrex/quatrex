# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from quatrex.core.config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver


class PhotonSolver(SubsystemSolver):
    system = "photon"

    def __init__(
        self,
        config: QuatrexConfig,
        energies: np.ndarray,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        ...
