# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os

from qttools import xp
from qttools.comm import comm
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.quatrex_config import QuatrexConfig


def get_electron_energies(quatrex_config: QuatrexConfig) -> xp.ndarray:
    """Get the electron energies based on the configuration.
    If an energy window is specified in the configuration, it generates
    the energies using linspace. Otherwise, it attempts to load the energies
    from a file named 'electron_energies.npy' in the input directory.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.

    Returns
    -------
    electron_energies : xp.ndarray
        Array of electron energies.

    Raises
    -------
    ValueError
        If both or neither of `energy_window_num` and `energy_window_num_per_rank` are set.
    FileNotFoundError
        If the energies file is not found and no energy window is specified.

    """

    if (quatrex_config.electron.energy_window_max is not None) and (
        quatrex_config.electron.energy_window_min is not None
    ):
        if quatrex_config.electron.energy_window_num is not None:
            if quatrex_config.electron.energy_window_num_per_rank is not None:
                raise ValueError(
                    "Should **exclusively** set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
                )
            electron_energies = xp.linspace(
                quatrex_config.electron.energy_window_min,
                quatrex_config.electron.energy_window_max,
                quatrex_config.electron.energy_window_num,
            )
        elif quatrex_config.electron.energy_window_num_per_rank is not None:
            energy_window_num = (
                quatrex_config.electron.energy_window_num_per_rank * comm.stack.size
            )
            electron_energies = xp.linspace(
                quatrex_config.electron.energy_window_min,
                quatrex_config.electron.energy_window_max,
                energy_window_num,
            )
        else:
            raise ValueError(
                "Should set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
            )
    else:
        energies_path = quatrex_config.input_dir / "electron_energies.npy"
        if os.path.isfile(energies_path):
            electron_energies = distributed_load(energies_path)
        else:
            raise FileNotFoundError(
                f"Could not find electron energies file at {energies_path}. Please provide an energy window in the config."
            )
    return electron_energies
