# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver


class DysonSolver(SubsystemSolver):
    """class for the Dyson Equation solver of interacting Green's function."""

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: xp.ndarray,
    ) -> None:
        """Initializes the GF solver."""
        super().__init__(quatrex_config, compute_config, energies)

    def solve(
        self,
        g0: DSBSparse,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Solves the Dyson Equation

        \[
           G^R = (I - G_0 \Sigma^R)^{-1} G_0
        \]

        \[
           G^\lessgtr = G^R \Sigma^\lessgtr G^R^\dagger
        \]

        """
