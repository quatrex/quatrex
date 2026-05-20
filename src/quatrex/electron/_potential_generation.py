import numpy as np
from scipy.interpolate import interp1d

from quatrex.core.statistics import fermi_dirac


def generate_potential_profile(
    grid, transport_direction, bias, potential_function="tanh", flat_length=0
):
    """
    Generates a potential profile on a given grid using a specified potential function.

    Parameters
    ----------
    grid : ndarray
        The spatial grid on which to evaluate the potential. Shape should be (N, 3) for 3D space.
    transport_direction : int
        The index of the transport direction (0 for x, 1 for y, 2 for z).
    bias : float
        The bias to apply to the potential profile. A positive bias creates a drop along transport.
    potential_function : str
        "tanh" or "linear". The type of potential function to use. Default is "tanh".
    flat_length : float
        Length of the flat part of the potential profile at the beginning and end.

    Returns
    -------
    potential_profile : ndarray
        The potential profile evaluated on the grid. Shape will be (N,).
    """
    coords = np.asarray(grid[:, transport_direction], dtype=float)
    coord0 = coords.min()
    transport_length = coords.max() - coord0

    if transport_length <= 0:
        raise ValueError("Grid has zero transport length in the chosen direction.")
    if flat_length < 0:
        raise ValueError("flat_length must be non-negative.")
    if 2 * flat_length >= transport_length:
        raise ValueError("flat_length is too large for the given transport length.")

    # Shift to [0, transport_length] so masks are coordinate-based, not index-based.
    s = coords - coord0
    left_mask = s <= flat_length
    right_mask = s >= (transport_length - flat_length)
    middle_mask = ~(left_mask | right_mask)

    potential_profile = np.empty_like(s)
    potential_profile[left_mask] = 0.0
    potential_profile[right_mask] = -bias

    if np.any(middle_mask):
        drop_coords = s[middle_mask] - flat_length
        drop_length = transport_length - 2 * flat_length

        if potential_function == "tanh":
            xi = drop_coords / drop_length
            # potential_profile[middle_mask] = -0.5 * bias * (1.0 + np.tanh(5 * (2 * xi - 1)))
            potential_profile[middle_mask] = (
                -0.5 * bias * (1.0 + np.tanh(3 * (2 * xi - 1)))
            )
        elif potential_function == "linear":
            potential_profile[middle_mask] = -bias * (drop_coords / drop_length)
        else:
            raise ValueError("Invalid potential function. Choose 'tanh' or 'linear'.")

    return potential_profile


def _trapezoidal_weights(energies):
    w = np.empty_like(energies)
    w[0] = (energies[1] - energies[0]) / 2
    w[-1] = (energies[-1] - energies[-2]) / 2
    w[1:-1] = (energies[2:] - energies[:-2]) / 2
    return w


def compute_charge_for_fermi_levels(fermi_levels, ldos, midgap_energies, energies):
    w = _trapezoidal_weights(energies)
    occupations = fermi_dirac(energies[:, None] - fermi_levels[None, :], 300)
    # Masks for above/below midgap for all indices: shape (ne, nindices)
    mask_above = energies[:, None] > midgap_energies[None, :]
    mask_below = ~mask_above
    # Weighted LDOS for above/below: shape (ne, nindices)
    wldos_above = ldos * mask_above * w[:, None]
    wldos_below = ldos * mask_below * w[:, None]
    # Compute charges: (nmu x nindices) = (nmu x ne) @ (ne x nindices)
    electron = occupations.T @ wldos_above
    hole = (1.0 - occupations).T @ wldos_below
    return electron - hole


def find_energy_shift(
    charge_per_fermi_level, fermi_levels, target_charge, current_charge
):
    energy_shift = np.empty(charge_per_fermi_level.shape[1])
    for j in range(charge_per_fermi_level.shape[1]):
        cp = charge_per_fermi_level[:, j]
        # ensure monotonicity for interp; if not monotonic, sort
        if not (np.all(np.diff(cp) >= 0) or np.all(np.diff(cp) <= 0)):
            idx = np.argsort(cp)
            cp_sorted = cp[idx]
            fm_sorted = fermi_levels[idx]
            interp = interp1d(cp_sorted, fm_sorted, bounds_error=True)
        else:
            interp = interp1d(cp, fermi_levels, bounds_error=True)
        energy_target = interp(target_charge)
        energy_current = interp(current_charge[j])
        energy_shift[j] = energy_target - energy_current
    return energy_shift
