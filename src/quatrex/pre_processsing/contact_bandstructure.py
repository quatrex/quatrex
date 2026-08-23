# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes functions to plot the contact band structure for a given quatrex configuration."""

import os

import numpy as np
from matplotlib import pyplot as plt

from qttools import sparse
from quatrex.bandstructure.contact import contact_band_structure
from quatrex.core.config import QuatrexConfig
from quatrex.device import Device
from quatrex.device.inputs import assemble_matrix, expand_circulant_cell
from quatrex.grid import monkhorst_pack


def _plot(
    ax: plt.Axes,
    kpoints_transport: np.ndarray,
    e_k: np.ndarray,
) -> None:
    """Plots the contact band structure for a given contact.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    kpoints_transport : np.ndarray
        The k-points along the transport direction.
    e_k : np.ndarray
        The eigenvalues for the contact band structure.

    """
    e_k = e_k.squeeze()
    k_repeated = np.repeat(kpoints_transport, e_k.shape[1])
    ax.scatter(k_repeated, e_k, color="blue", s=10)


def _plot_wf(config: QuatrexConfig, axes: plt.Axes, device: Device) -> None:
    """Plots the contact band structure for a wavefunction simulation.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    axes : plt.Axes
        The axes to plot on.
    device : Device
        The device object.

    """

    kpoint_grid = config.device.kpoint_grid
    kpoints = monkhorst_pack(kpoint_grid, config.device.kpoint_shift)

    if device.gamma_only and kpoint_grid != (1, 1, 1):
        raise ValueError(
            "The device only has a Gamma point Hamiltonian, "
            "but more than one k-point is configured."
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

        for n, (contact, contact_config) in enumerate(
            zip(device.contacts, config.device.contacts)
        ):
            kpoints_transport = np.linspace(
                -np.pi,
                np.pi,
                contact_config.num_kpoints_transport,
                endpoint=False,
            )

            h_xx = {}
            # slice the layer that corresponds to the contact unit cell
            hamiltonian_origin = hamiltonian[contact.origin_orbital_indices, :]
            hamiltonian_origin += hamiltonian[
                :, contact.origin_orbital_indices
            ].T.conj()
            hamiltonian_origin[:, contact.origin_orbital_indices] -= (
                sparse.diags(
                    hamiltonian_origin[:, contact.origin_orbital_indices].diagonal()
                )
                / 2
            )
            ny, nz = contact.transverse_repetition_grid
            for j, k, i in np.ndindex(ny, nz, contact.num_transport_cells + 1):
                h_xx[i, j, k] = hamiltonian_origin[
                    :, contact.unit_cell_orbital_indices[i, j, k]
                ].todense()

            s_xx = {}
            overlap_origin = overlap[contact.origin_orbital_indices, :]
            overlap_origin += overlap[:, contact.origin_orbital_indices].T.conj()
            overlap_origin[:, contact.origin_orbital_indices] -= (
                sparse.diags(
                    overlap_origin[:, contact.origin_orbital_indices].diagonal()
                )
                / 2
            )
            ny, nz = contact.transverse_repetition_grid
            for j, k, i in np.ndindex(ny, nz, contact.num_transport_cells + 1):
                s_xx[i, j, k] = overlap_origin[
                    :, contact.unit_cell_orbital_indices[i, j, k]
                ].todense()

            transport_ind = "xyz".index(config.device.transport_direction)

            # upscale the blocks
            shifts = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
            for i in range(3):
                shifts[i][transport_ind] = contact.num_transport_cells * (i - 1)

            phases = tuple(np.exp(2j * np.pi * k) for k in kpoint)
            phases = phases[:transport_ind] + phases[transport_ind + 1 :]

            h_xx = tuple(
                expand_circulant_cell(
                    h_xx,
                    contact.num_transport_cells,
                    transport_ind,
                    tuple(shift),
                    (ny, nz),
                    phases=phases,
                    key_assumption="half",
                )
                for shift in shifts
            )

            s_xx = tuple(
                expand_circulant_cell(
                    s_xx,
                    contact.num_transport_cells,
                    transport_ind,
                    tuple(shift),
                    (ny, nz),
                    phases=phases,
                    key_assumption="half",
                )
                for shift in shifts
            )

            kpoints_transport = np.linspace(
                -np.pi,
                np.pi,
                contact_config.num_kpoints_transport,
                endpoint=False,
            )

            e_k = contact_band_structure(kpoints_transport, h_xx, s_xx)
            if contact_config.voltage is not None:
                e_k += contact_config.voltage
            _plot(
                ax=axes[m, n],
                kpoints_transport=kpoints_transport,
                e_k=e_k,
            )


def _plot_negf(config: QuatrexConfig, axes: plt.Axes) -> None:
    """Plots the contact band structure for a NEGF simulation.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    axes : plt.Axes
        The axes to plot on.

    """

    # Initialize the device
    hamiltonian, __ = assemble_matrix(
        config=config,
        matrix_name="hamiltonian",
        sparsity_pattern=None,
        shift_kpoints=False,
    )

    try:
        # Attempt to load the device overlap matrix.
        overlap, __ = assemble_matrix(
            config=config,
            matrix_name="overlap",
            sparsity_pattern=None,
            shift_kpoints=False,
        )
        print("Non-orthogonal basis detected.", flush=True)

    except FileNotFoundError:
        overlap = None
        print("No overlap matrix found. Assuming orthogonal basis.", flush=True)

    for i, contact_config in enumerate(
        [config.electron.left_contact, config.electron.right_contact]
    ):
        n = hamiltonian.num_local_blocks - 1
        m = n - 1
        diagonal_inds = (0, 0) if contact_config.name == "left" else (n, n)
        upper_inds = (0, 1) if contact_config.name == "left" else (n, m)

        h_xx = (
            hamiltonian.blocks[*upper_inds[::-1]],
            hamiltonian.blocks[*diagonal_inds],
            hamiltonian.blocks[*upper_inds],
        )

        if overlap is not None:
            s_xx = (
                overlap.blocks[*upper_inds[::-1]],
                overlap.blocks[*diagonal_inds],
                overlap.blocks[*upper_inds],
            )
        else:
            s_xx = None

        kpoints_transport = np.linspace(
            -np.pi,
            np.pi,
            contact_config.num_kpoints_transport,
            endpoint=False,
        )

        e_k = contact_band_structure(kpoints_transport, h_xx, s_xx)
        if contact_config.voltage is not None:
            e_k += contact_config.voltage
        for j, kpoint in enumerate(np.ndindex(h_xx[0].shape[:-2])):
            _plot(
                ax=axes[j, i],
                kpoints_transport=kpoints_transport,
                e_k=e_k[:, *kpoint, :],
            )


def plot_contact_band_structure(config: QuatrexConfig) -> None:
    """Plots the contact band structure for a given quatrex configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.

    """

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    kpoint_grid = config.device.kpoint_grid
    kpoints = monkhorst_pack(kpoint_grid, config.device.kpoint_shift)

    if config.formalism == "wf":
        device = Device(config)
        contacts = device.contacts
    elif config.formalism == "negf":
        contacts = [config.electron.left_contact, config.electron.right_contact]
    else:
        raise ValueError(f"Unknown formalism: {config.formalism}")

    plt.rcParams.update({"font.size": 16})
    __, axes = plt.subplots(
        len(kpoints),
        len(contacts),
        figsize=(12, 6),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    if config.formalism == "wf":
        _plot_wf(config, axes, device)
    elif config.formalism == "negf":
        _plot_negf(config, axes)

    for ax, contact in zip(axes[0], contacts):
        if contact.fermi_level is not None:
            ax.axhline(
                contact.fermi_level, color="red", linestyle="--", label="Fermi Level"
            )
        if contact.mid_gap_energy is not None:
            ax.axhline(
                contact.mid_gap_energy,
                color="green",
                linestyle=":",
                label="Mid-gap Energy",
            )
        ax.legend(loc="upper left")

    for ax, contact in zip(axes[0], contacts):
        ax.set_title(f"{contact.name.capitalize()} Contact")

    for ax, kpoint in zip(axes[:, 0], kpoints):
        ax.set_ylabel(f"k-point: {kpoint}\nEnergy (eV)")

    for ax in axes[-1]:
        ax.set_xlabel("k")

    if config.pre_process.plot_window is None:
        center_energy = (
            contacts[0].mid_gap_energy
            if contacts[0].mid_gap_energy is not None
            else contacts[0].fermi_level
        )
        # NOTE: Naively take +-1eV around the center energy for the plot window.
        plot_window = (center_energy - 1, center_energy + 1)
    else:
        plot_window = config.pre_process.plot_window

    plt.ylim(plot_window)
    plt.tight_layout()
    plt.savefig(config.output_dir / "band_structure.png", dpi=300)
    plt.close()
