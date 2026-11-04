# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the device and contact classes."""

from quatrex.core.config import QuatrexConfig
from quatrex.device.base import BaseDevice
from quatrex.device.qtbm import QTBMDevice
from quatrex.device.scba import SCBADevice
from quatrex.device.schroedinger import SchroedingerDevice


def create_device(
    config: QuatrexConfig,
    validate_contacts: bool = True,
) -> QTBMDevice | SCBADevice | SchroedingerDevice:
    """Creates a device object based on the configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The main quatrex configuration.
    validate_contacts : bool, optional
        Whether to validate the contacts after creation, by default
        True.

    Returns
    -------
    QTBMDevice | SCBADevice | SchroedingerDevice
        The created device object.

    """
    if config.formalism == "wf":
        device = QTBMDevice(config)
    elif config.formalism == "negf":
        device = SCBADevice(config)
    elif config.formalism == "closed":
        device = SchroedingerDevice(config)
    else:
        raise ValueError(f"Unknown formalism {config.formalism}")

    if validate_contacts:
        device._validate_contacts()

    return device


__all__ = [
    "BaseDevice",
    "QTBMDevice",
    "SCBADevice",
    "SchroedingerDevice",
    "create_device",
]
