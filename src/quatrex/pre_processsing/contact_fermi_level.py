# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes functionality to compute and write the Fermi level of the contacts."""

import warnings

import numpy as np

from qttools import xp
from qttools.comm import comm
from quatrex.bandstructure.contact import (
    contact_band_edges,
    contact_doping_density,
    contact_fermi_level,
)
from quatrex.core.config import ContactConfig, DeviceConfig, QuatrexConfig
from quatrex.device import Contact, Device
from quatrex.grid import monkhorst_pack
from quatrex.pre_processsing.contact_bandstructure import slice_expand_bandstructure


def _compute_fermi_level(
    contact: Contact,
    contact_config: ContactConfig,
    device: Device,
    device_config: DeviceConfig,
) -> tuple[float, float, float]:
    """Computes the Fermi level for the contact.

    Parameters
    ----------
    contact : Contact
        The contact object.
    contact_config : ContactConfig
        The contact configuration object.
    device : Device
        The device object.
    device_config : DeviceConfig
        The device configuration object.

    Returns
    -------
    fermi_level : float
        The computed Fermi level in eV.
    mid_gap_energy : float
        The recomputed mid-gap energy based on the band structure.
    delta_fermi_level_conduction_band : float
        The energy difference between the Fermi level and the
        conduction band edge.

    """
    kpoints_transport = np.linspace(
        -np.pi,
        np.pi,
        contact_config.num_kpoints_transport,
        endpoint=False,
    )

    transverse_axes = [0, 1, 2]
    transverse_axes.remove(contact.direction)

    kpoints = monkhorst_pack(device_config.kpoint_grid, device_config.kpoint_shift)

    e_k = xp.zeros(
        (
            len(kpoints_transport),
            kpoints.shape[0],
            len(contact.origin_orbital_indices) * contact.transport_repetitions,
        ),
        dtype=float,
    )

    hamiltonians = device.hamiltonians
    overlaps = device.overlap_matrices
    for m, kpoint in enumerate(kpoints):
        hamiltonian = sum(
            np.exp(2j * np.pi * np.dot(kpoint, r)) * h for r, h in hamiltonians.items()
        )
        overlap = sum(
            np.exp(2j * np.pi * np.dot(kpoint, r)) * s for r, s in overlaps.items()
        )

        e_k[:, m, :] = slice_expand_bandstructure(
            hamiltonian=hamiltonian,
            overlap=overlap,
            kpoint=kpoint,
            contact=contact,
            kpoints_transport=kpoints_transport,
        )

    # Average over transverse k-points.
    e_k = xp.mean(e_k, axis=1)

    doping_density = contact_doping_density(
        coordinates=device.orbital_coordinates[contact.origin_orbital_indices],
        geometry_regions=device_config.geometry.regions,
    )

    fermi_level = contact_fermi_level(
        e_k=e_k,
        kpoints=kpoints_transport,
        mid_gap_energy=contact.mid_gap_energy,
        cell_volume=contact_config.cell_volume,
        doping_density=doping_density,
        temperature=contact.temperature,
    )

    # Recompute the actual mid-gap energy from the band structure.
    valence_band_edge, conduction_band_edge = contact_band_edges(
        e_k, contact.mid_gap_energy
    )
    mid_gap_energy = 0.5 * (conduction_band_edge + valence_band_edge)
    delta_fermi_level_conduction_band = conduction_band_edge - fermi_level

    if comm.rank == 0:
        print(f"    Doping density: {doping_density} Å^-3", flush=True)
        print(f"    Fermi level: {fermi_level} eV", flush=True)
        print(f"    Conduction band minimum: {conduction_band_edge} eV", flush=True)
        print(f"    Valence band maximum: {valence_band_edge} eV", flush=True)
        print(f"    Recomputed mid-gap energy: {mid_gap_energy} eV", flush=True)
        print(
            f"    Delta Fermi level conduction band: {delta_fermi_level_conduction_band} eV",
            flush=True,
        )

    return fermi_level, mid_gap_energy, delta_fermi_level_conduction_band


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
            ) = _compute_fermi_level(
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
