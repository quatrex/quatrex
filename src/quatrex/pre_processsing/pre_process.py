# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes main pre-processing function."""

from quatrex.core.config import QuatrexConfig
from quatrex.pre_processsing.contact_bandstructure import plot_contact_band_structure


def pre_process(config: QuatrexConfig):
    """Main function for pre-processing.

    The functionality includes the following:
    - Plotting the contact band structure if enabled in the configuration.


    Parameters
    ----------
    config : QuatrexConfig
        The main quatrex configuration.

    """

    if config.pre_process.plot_contact_band_structure:
        plot_contact_band_structure(config)
