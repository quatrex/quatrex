# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes functionality to compute and write the Fermi level of the contacts."""

import warnings

from qttools.comm import comm
from quatrex.bandstructure.contact import compute_contact_band_properties
from quatrex.core.config import QuatrexConfig
from quatrex.device import Device


def pre_process_fermi_level(
    config: QuatrexConfig,
    device: Device | None = None,
) -> None:
    """Computes the Fermi level of the contacts for a given quatrex
    configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    device : Device | None
        The device object. It is `None` for NEGF simulations.

    """

    if comm.rank != 0:
        return

    if comm.size > 1:
        warnings.warn(
            "Pre-processing is only performed on rank 0. "
            "If you are running a parallel simulation, please ensure that "
            "the pre-processing steps are completed before starting the parallel run."
        )

    if config.formalism == "wf":
        for contact_config, contact in zip(config.device.contacts, device.contacts):
            if comm.rank == 0:
                print(
                    f"Computing Fermi level for contact {contact_config.name}",
                    flush=True,
                )

            (
                fermi_level,
                mid_gap_energy,
                delta_fermi_level_conduction_band,
            ) = compute_contact_band_properties(
                contact=contact,
                contact_config=contact_config,
                device=device,
                device_config=config.device,
            )

            # TODO update the config with the new values.

        # TODO update the config file with the new values.

    else:
        warnings.warn(
            "Automatic Fermi level computation does nothing for NEGF simulations"
        )
