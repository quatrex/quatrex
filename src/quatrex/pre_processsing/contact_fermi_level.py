# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes functionality to compute and write the Fermi level of the contacts."""

import warnings
from textwrap import dedent

import tomlkit

from qttools.comm import comm
from quatrex.bandstructure.contact import compute_contact_band_properties
from quatrex.config import merge_toml
from quatrex.core.config import QuatrexConfig, _parse_config
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
        # load config and save the original config
        config_file = config.config_file
        if config_file is None:
            raise ValueError(
                "The configuration file path is not set. "
                "An error occured while loading the configuration file."
            )

        new_config = tomlkit.parse(config_file.read_text())
        original_config_file = config_file.with_name(
            config_file.stem + "_original.toml"
        )
        if original_config_file.exists():
            # Guard against calling pre-processing multiple times, which
            # would overwrite the original config file.
            raise ValueError(
                f"The original configuration file {original_config_file} already exists.\n"
                "This is most likely because you have already run the pre-processing step.\n"
                "Backup and remove the original configuration file"
                "before running the pre-processing step again."
            )
        original_config_file.write_text(tomlkit.dumps(new_config))

        # NOTE: Not the most efficient code since we do naive loops. The
        # code could be potentially batched, but this should not be a
        # bottleneck since it is only pre-processing.
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
            # Make sure that they are float and not numpy.float64
            fermi_level = float(fermi_level)
            mid_gap_energy = float(mid_gap_energy)
            delta_fermi_level_conduction_band = float(delta_fermi_level_conduction_band)
            contact.fermi_level = fermi_level
            contact.mid_gap_energy = mid_gap_energy
            contact.delta_fermi_level_conduction_band = (
                delta_fermi_level_conduction_band
            )

            patch = dedent(
                f"""
                [[device.contacts]]
                name = "{contact.name}"
                {fermi_level=}
                {mid_gap_energy=}
                {delta_fermi_level_conduction_band=}
                """
            )
            merge_toml(
                new_config,
                tomlkit.parse(patch),
                id_keys={"contacts": "name"},
            )

        # parse the new config to make sure that it is valid need to
        # unwrap due the tomlkit lib not being compatible with
        # posixpaths
        QuatrexConfig(**_parse_config(config_file, new_config.unwrap()))
        config_file.write_text(tomlkit.dumps(new_config))

    else:
        warnings.warn(
            "Automatic Fermi level computation does nothing for NEGF simulations"
        )
