# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes main pre-processing function."""

from qttools.comm import comm
from quatrex.core.config import QuatrexConfig
from quatrex.device import Device
from quatrex.pre_processsing.contact_bandstructure import plot_contact_band_structure
from quatrex.pre_processsing.contact_fermi_level import pre_process_fermi_level


def pre_process(config: QuatrexConfig):
    """Main function for pre-processing.

    The functionality includes the following:
    - Plotting the contact band structure if enabled in the configuration.


    Parameters
    ----------
    config : QuatrexConfig
        The main quatrex configuration.

    """

    if comm.size > 1:
        raise RuntimeError(
            "The 'pre-process' command can only be run on a single process."
        )

    device = None
    if config.formalism == "wf":
        device = Device(config)

    if config.pre_process.compute_fermi_level:
        pre_process_fermi_level(config, device)

    if config.pre_process.plot_contact_band_structure:
        plot_contact_band_structure(config, device)
