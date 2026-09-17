# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the device and contact classes."""

from quatrex.core.config import QuatrexConfig
from quatrex.device.base import BaseDevice
from quatrex.device.qtbm import QTBMDevice
from quatrex.device.scba import SCBADevice


def create_device(
    config: QuatrexConfig,
    validate_contacts: bool = True,
) -> QTBMDevice | SCBADevice:
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
    QTBMDevice | SCBADevice
        The created device object.

    """
    if config.formalism == "wf":
        device = QTBMDevice(config)
    elif config.formalism == "negf":
        device = SCBADevice(config)
    else:
        raise ValueError(f"Unknown formalism {config.formalism}")

    if validate_contacts:
        device._validate_contacts()

    return device


__all__ = [
    "BaseDevice",
    "QTBMDevice",
    "SCBADevice",
    "create_device",
]
