# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes contact band structure related functions."""

import numpy as np
from scipy.optimize import minimize_scalar

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.kernels import linalg
from quatrex.core.config import ContactConfig, DeviceConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.device import Contact, Device
from quatrex.device.inputs import expand_circulant_cell
from quatrex.electrostatics.geometry_config import Region, VolumeProperties
from quatrex.electrostatics.meshing import inside_shape
from quatrex.grid import monkhorst_pack


def _real_to_kspace(
    kpoints: NDArray, a_xx: tuple[NDArray, NDArray, NDArray] | None
) -> NDArray:
    """Transforms real-space operator blocks to k-space."""
    a_10, a_00, a_01 = a_xx
    a_k = (
        np.einsum("k, ...-> k...", np.exp(-1j * kpoints), a_10)
        + a_00
        + np.einsum("k, ...-> k...", np.exp(1j * kpoints), a_01)
    )
    return a_k


def contact_band_structure(
    kpoints: NDArray,
    h_xx: tuple[NDArray, NDArray, NDArray],
    s_xx: tuple[NDArray, NDArray, NDArray] | None = None,
) -> NDArray:
    """Computes the band structure of a device contact.

    Parameters
    ----------
    kpoints : NDArray
        The k-points at which to compute the band structure. This should
        have shape (num_kpoints,).
    h_xx : tuple[NDArray, NDArray, NDArray]
        Hamiltonian matrix blocks of a single contact layer.
    s_xx : tuple[NDArray, NDArray, NDArray] | None
        Overlap matrix blocks. If None, the overlap matrix is assumed to
        be the identity.

    Returns
    -------
    e_k : NDArray
        The sorted eigenvalues in energy and k. This will have shape
        (num_kpoints, num_bands).

    """
    h_k = _real_to_kspace(kpoints, h_xx)
    s_k = _real_to_kspace(kpoints, s_xx) if s_xx is not None else None

    e_k = linalg.eigvalsh(h_k, s_k, compute_module="numpy")

    return xp.sort(e_k, axis=-1)


def contact_band_edges(e_k: NDArray, mid_gap_energy: float) -> tuple[float, float]:
    """Computes the band edges from band structure and mid-gap energy.

    Parameters
    ----------
    e_k : NDArray
        The sorted eigenvalues in energy and k. This should have shape
        (num_kpoints, num_bands).
    mid_gap_energy : float
        A guess for the mid-gap energy, which is used to determine the
        number of valence bands.

    Returns
    -------
    valence_band_edge : float
        The energy of the valence band edge in eV.
    conduction_band_edge : float
        The energy of the conduction band edge in eV.

    """
    valence_bands_mask = e_k < mid_gap_energy

    valence_band_edge = e_k[valence_bands_mask].max()
    conduction_band_edge = e_k[~valence_bands_mask].min()

    # NOTE: This is a implicit copy to the host
    # this is done since max() gives a 0-dim with cupy
    # and a scalar with numpy
    return float(valence_band_edge), float(conduction_band_edge)


def contact_doping_density(
    coordinates: NDArray, geometry_regions: list[Region]
) -> float:
    """Computes the doping density of a device contact.

    This function checks which geometry region the contact coordinates
    are located in and returns the corresponding doping density. If the
    contact is located in multiple regions, the doping density of the
    first detected region is returned. If the contact is not located in
    any region, a doping density of 0 is returned.

    Parameters
    ----------
    coordinates : NDArray
        The coordinates of the contact.
    geometry_regions : list[Region]
        The geometry regions to check.

    Returns
    -------
    doping_density : float
        The doping density in Å^-3.

    """
    for region in geometry_regions:
        if not isinstance(region.properties, VolumeProperties):
            continue

        if (
            region.properties.donor_concentration == 0.0
            and region.properties.acceptor_concentration == 0.0
        ):
            continue

        if np.all(inside_shape(coordinates, region.shape)):
            # Convert from cm^-3 to Å^-3
            doping_density = 1e-24 * (
                region.properties.donor_concentration
                - region.properties.acceptor_concentration
            )
            return doping_density

    return 0.0


def contact_fermi_level(
    e_k: NDArray,
    kpoints: NDArray,
    mid_gap_energy: float,
    cell_volume: float,
    doping_density: float,
    temperature: float = 300,
) -> float:
    """Computes the Fermi level of a device contact.

    The cell volume and the doping density must be given in the same
    units, so both should be in m^3/m^-3 or in nm^3/nm^-3, etc.

    Parameters
    ----------
    e_k : NDArray
        The sorted eigenvalues in energy and k. This should have shape
        (num_kpoints, num_bands).
    kpoints : NDArray
        The corresponding k-points. This should have shape
        (num_kpoints,).
    mid_gap_energy : float
        A guess for the mid-gap energy, which is used to determine the
        number of valence bands.
    cell_volume : float
        The volume of the unit cell. This needs to have the same units
        as the doping density.
    doping_density : float
        The doping density. This needs to have the same units as the
        cell volume.
    temperature : float, optional
        The temperature in K. Default is 300 K.

    Returns
    -------
    fermi_level : float
        The Fermi level in eV.

    """

    num_valence_bands_k = (e_k < mid_gap_energy).sum(axis=1)
    if not np.all(num_valence_bands_k == num_valence_bands_k[0]):
        raise ValueError(
            "The number of valence bands is not the same for all "
            "k-points. You may have to adjust the mid-gap energy guess."
        )

    # NOTE: int() does a implicit copy to the host which is needed since
    # the input for split needs to be a int.
    num_valence_bands = int(num_valence_bands_k[0])
    e_k_valence, e_k_conduction = xp.split(e_k, [num_valence_bands], axis=1)

    def objective_function(fermi_level):
        """Charge neutrality objective function."""
        n_k = fermi_dirac(e_k_conduction - fermi_level, temperature).sum(axis=1)
        p_k = fermi_dirac(fermi_level - e_k_valence, temperature).sum(axis=1)
        n = xp.trapezoid(n_k, kpoints)
        p = xp.trapezoid(p_k, kpoints)

        rho = (n - p) / (2 * xp.pi * cell_volume)
        rho *= 2  # Spin

        return float((rho - doping_density) ** 2)

    # NOTE: cupyx does not support minimize_scalar
    result = minimize_scalar(
        objective_function,
        bounds=(float(e_k.min()), float(e_k.max())),
        method="bounded",
    )

    return result.x


def compute_contact_bandstructure(
    hamiltonian: sparse.spmatrix,
    overlap: sparse.spmatrix,
    kpoint: NDArray,
    contact: Contact,
    kpoints_transport: NDArray,
) -> NDArray:
    """Slices and expands the inptut matrices, and computes the contact
    band structure for a given contact at a specific k-point.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian matrix of the device. It is expected to be the
        full Hamiltonian for the specific k-point.
    overlap : sparse.spmatrix
        The overlap matrix of the device. It is expected to be the full
        overlap matrix for the specific k-point.
    kpoint : NDArray
        The k-point at which to compute the band structure.
    contact : Contact
        The contact object containing information about the contact.
    kpoints_transport : NDArray
        The k-points along the transport direction.

    Returns
    -------
    e_k : np.ndarray
        The eigenvalues for the contact band structure.

    """
    grid = (contact.transport_repetitions + 1,) + contact.transverse_repetition_grid
    h_sliced = Contact.slice_matrix(
        M=hamiltonian,
        origin_orbital_indices=contact.origin_orbital_indices,
        unit_cell_orbital_indices=contact.unit_cell_orbital_indices,
        grid=grid,
        upper=True,
    )
    s_sliced = Contact.slice_matrix(
        M=overlap,
        origin_orbital_indices=contact.origin_orbital_indices,
        unit_cell_orbital_indices=contact.unit_cell_orbital_indices,
        grid=grid,
        upper=True,
    )

    h_xx = {}
    s_xx = {}
    # shuffle keys to to have natural order a,b,c
    for i, j, k in np.ndindex(*grid):
        index = [j, k]
        index.insert(contact.direction, i)
        index = tuple(index)
        h_xx[index] = h_sliced[i, j, k].toarray()
        s_xx[index] = s_sliced[i, j, k].toarray()

    phases = tuple(np.exp(2j * np.pi * k) for k in kpoint)
    phases = phases[: contact.direction] + phases[contact.direction + 1 :]

    h_xx = tuple(
        expand_circulant_cell(
            matrix_dict=h_xx,
            transport_cell_size=contact.transport_repetitions,
            transport_ind=contact.direction,
            index=index,
            sections=contact.transverse_repetition_grid,
            phases=phases,
            key_assumption="half",
        )
        for index in [-1, 0, 1]
    )
    s_xx = tuple(
        expand_circulant_cell(
            matrix_dict=s_xx,
            transport_cell_size=contact.transport_repetitions,
            transport_ind=contact.direction,
            index=index,
            sections=contact.transverse_repetition_grid,
            phases=phases,
            key_assumption="half",
        )
        for index in [-1, 0, 1]
    )

    return contact_band_structure(kpoints_transport, h_xx, s_xx)


def compute_contact_band_properties(
    contact: Contact,
    contact_config: ContactConfig,
    device: Device,
    device_config: DeviceConfig,
) -> tuple[float, float, float]:
    """Computes the Fermi level for the contact from the Hamiltonian and
    overlap matrices.

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
        The energy difference between the Fermi level and the conduction
        band edge.

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

        e_k[:, m, :] = compute_contact_bandstructure(
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
        print(f"Doping density: {doping_density} Å^-3", flush=True)
        print(f"Fermi level: {fermi_level} eV", flush=True)
        print(f"Conduction band minimum: {conduction_band_edge} eV", flush=True)
        print(f"Valence band maximum: {valence_band_edge} eV", flush=True)
        print(f"Recomputed mid-gap energy: {mid_gap_energy} eV", flush=True)
        print(
            f"Delta Fermi level conduction band: {delta_fermi_level_conduction_band} eV",
            flush=True,
        )

    return fermi_level, mid_gap_energy, delta_fermi_level_conduction_band
