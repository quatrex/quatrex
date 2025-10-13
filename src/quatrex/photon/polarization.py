# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import numpy as np
from quatrex.core.sse import ScatteringSelfEnergy


class PiPhoton(ScatteringSelfEnergy):

    def __init__(
        self, #config: QuatrexConfig,
        electron_energies: np.array,
    ) -> None:
            
        # self.photon_energy = config.photon.energy  # or however it's stored
        self.electron_energies = electron_energies
        return  

    def compute(self, g_lesser, g_greater,
    ) -> None:
        """Computes the photon polarization function.

        Args:
            distances: Distances at which to evaluate the polarization function.

        Returns:
            The photon polarization function evaluated at the given distances.
        """

        PiPhoton

        return 