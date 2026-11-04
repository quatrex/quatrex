# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes functions to plot the contact band structure for a given quatrex configuration."""

import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.utils.gpu_utils import get_host
from quatrex.core.config import QuatrexConfig
from quatrex.device import BaseDevice


def _plot(
    ax: plt.Axes,
    kpoints_transport: NDArray,
    e_k: NDArray,
) -> None:
    """Plots the contact band structure for a given contact.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    kpoints_transport : NDArray
        The k-points along the transport direction.
    e_k : NDArray
        The eigenvalues for the contact band structure.

    """
    e_k = e_k.squeeze()
    k_repeated = np.repeat(get_host(kpoints_transport), e_k.shape[1])
    ax.scatter(k_repeated, get_host(e_k), color="blue", s=10)


def _generate_plots(config: QuatrexConfig, axes: plt.Axes, device: BaseDevice) -> None:
    """Plots the contact band structure.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    axes : plt.Axes
        The axes to plot on.
    device : BaseDevice
        The device object.

    """
    # NOTE: Not the most efficient code since we do naive loops. The
    # code could be potentially batched, but this should not be a
    # bottleneck since it is only pre-processing.
    for n, (contact, contact_config) in enumerate(
        zip(device.contacts, config.device.contacts)
    ):
        kpoints_transport = xp.linspace(
            -xp.pi,
            xp.pi,
            contact_config.num_kpoints_transport,
            endpoint=False,
        )

        e_k = contact.compute_contact_bandstructure(
            kpoints_transport=kpoints_transport,
        )

        if contact_config.voltage is not None:
            e_k += contact_config.voltage

        for m in range(len(device.kpoints)):
            _plot(
                ax=axes[m, n],
                kpoints_transport=kpoints_transport,
                e_k=e_k[:, m, :],
            )


def plot_contact_band_structure(
    config: QuatrexConfig,
    device: BaseDevice,
) -> None:
    """Plots the contact band structure for a given quatrex configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    device : BaseDevice
        The device object.

    """

    if comm.size > 1:
        raise ValueError(
            "Pre-processing can be only performed on a single process. "
            "If you are running a parallel simulation, please ensure that "
            "the pre-processing steps are completed before starting the parallel run."
        )

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    contacts = device.contacts

    # Turn off interactive plotting only for this step and increase font
    # size
    backend = matplotlib.get_backend()
    plt.switch_backend("Agg")
    original_rc = plt.rcParams.copy()
    plt.rcParams.update({"font.size": 16})

    fig, axes = plt.subplots(
        len(device.kpoints),
        len(contacts),
        figsize=(12, 6),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    _generate_plots(config, axes, device)

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
        if (contact.voltage is not None) and (contact.fermi_level is not None):
            ax.axhline(
                contact.fermi_level - contact.voltage,
                color="orange",
                linestyle="-.",
                label=r"Chemical Potential ($\mu$)",
            )

        ax.legend(loc="upper left")

    for ax, contact in zip(axes[0], contacts):
        ax.set_title(f"{contact.name.capitalize()} Contact")

    for ax, kpoint in zip(axes[:, 0], device.kpoints):
        ax.set_ylabel(f"k-point:\n{kpoint}\nEnergy (eV)")

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

    for ax in axes.flatten():
        ax.set_ylim(plot_window)
    fig.tight_layout()
    fig.savefig(config.output_dir / "band_structure.png", dpi=300)
    plt.close(fig)
    plt.rcParams.update(original_rc)
    plt.switch_backend(backend)
