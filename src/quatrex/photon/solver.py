import numpy as np

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver


class PhotonSolver(SubsystemSolver):
    system = "photon"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
    ) -> None:
        """Initializes the solver."""
        super().__init__(quatrex_config)

        ...
