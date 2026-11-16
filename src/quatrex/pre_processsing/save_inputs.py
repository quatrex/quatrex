import os

import ase
import ase.io
import numpy as np

from quatrex.core.config import QuatrexConfig
from quatrex.device import BaseDevice


def save_processed_inputs(
    config: QuatrexConfig,
    device: BaseDevice,
) -> None:
    """Saves the atomic positions after quatrex' pre-processing (e.g.
    upscaling)

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    device : BaseDevice
        The device object.

    """

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    atoms = ase.Atoms(
        symbols=device.atomic_species,
        positions=np.asarray(device.atom_coordinates),
    )

    ase.io.write(config.output_dir / "structure.xyz", atoms, format="extxyz")
