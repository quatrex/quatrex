# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from types import SimpleNamespace

import numpy as np
import pytest

from quatrex.core.constants import hbar, k_B
from quatrex.core.qtbm import QTBM
from quatrex.core.statistics import fermi_dirac
from quatrex.electrostatics.density_response import (
    GrapheneDensityModel,
    LDOSDensityModel,
    SpectralDensityOfStates,
    dirac_net_density,
)

TEMPERATURE = 300.0
KT = k_B * TEMPERATURE
ENERGIES = np.linspace(-1.5, 1.5, 6001)


def _spectral_dos(dos, chemical_potentials) -> SpectralDensityOfStates:
    chemical_potentials = np.atleast_1d(chemical_potentials)
    return SpectralDensityOfStates(
        spectral_dos=np.asarray(dos),
        energies=ENERGIES,
        chemical_potentials=chemical_potentials,
        temperatures=np.full(chemical_potentials.shape, TEMPERATURE),
    )


def _random_model(seed: int = 0) -> LDOSDensityModel:
    """Two contacts at different chemical potentials, structured DOS."""
    rng = np.random.default_rng(seed)
    num_atoms = 5
    peaks = rng.uniform(-0.3, 0.3, (2, num_atoms, 1))
    dos = 0.5 + np.exp(-(((ENERGIES - peaks) / 0.05) ** 2))
    return LDOSDensityModel(
        density=rng.uniform(-1e-3, 1e-3, num_atoms),
        potential=rng.uniform(-0.1, 0.1, num_atoms),
        spectral_dos=_spectral_dos(dos, [0.0, -0.05]),
    )


def test_reference_density():
    """Tests that the transport density is reproduced (up to round-off)."""
    model = _random_model()
    assert np.allclose(
        model.density(model.reference_potential),
        model.reference_density,
        rtol=0,
        atol=1e-14,
    )


def test_constant_dos():
    """Tests that a constant DOS gives dn/dV = -D and n linear in V."""
    dos = 0.7
    model = LDOSDensityModel(
        density=np.zeros(3),
        potential=np.zeros(3),
        spectral_dos=_spectral_dos(np.full((1, 3, ENERGIES.size), dos), 0.0),
    )
    potential = np.array([-0.2, 0.0, 0.3])
    assert np.allclose(model.density_derivative(potential), -dos, rtol=1e-5)
    assert np.allclose(model.density(potential), -dos * potential, rtol=1e-5)


def test_density_derivative():
    """Tests the derivative against central finite differences."""
    model = _random_model()
    potential = model.reference_potential + np.linspace(-0.1, 0.1, 5)
    step = 1e-5
    finite_difference = (
        model.density(potential + step) - model.density(potential - step)
    ) / (2 * step)
    assert np.allclose(
        model.density_derivative(potential), finite_difference, rtol=1e-6
    )


def test_matches_graphene_model():
    """Tests that a Dirac-cone LDOS reproduces the graphene model."""
    area_per_atom = 3 * np.sqrt(3) / 4 * 1.42**2
    hbar_v_F = hbar * 1e6 * 1e10  # eV Å
    # Graphene DOS per atom including spin and valley degeneracy.
    dos = 2 / np.pi * np.abs(ENERGIES) / hbar_v_F**2 * area_per_atom
    effective_dos = 2 / np.pi * (KT / hbar_v_F) ** 2 * area_per_atom

    # Charge neutrality level at E = 0, the electrochemical potential is
    # 0.1 eV above it at zero potential.
    chemical_potential = 0.1
    density = effective_dos * dirac_net_density(np.array([chemical_potential / KT]))
    potential = np.zeros(1)

    ldos_model = LDOSDensityModel(
        density, potential, _spectral_dos(dos[np.newaxis, np.newaxis], 0.1)
    )
    graphene_model = GrapheneDensityModel(density, potential)

    potentials = np.linspace(-0.4, 0.4, 17)
    for value in potentials:
        phi = np.array([value])
        assert np.allclose(
            ldos_model.density(phi), graphene_model.density(phi), rtol=1e-4
        )
        assert np.allclose(
            ldos_model.density_derivative(phi),
            graphene_model.density_derivative(phi),
            rtol=1e-4,
        )


def test_window_warning():
    """Tests the warning when a Fermi window leaves the energy window."""
    model = _random_model()
    with pytest.warns(RuntimeWarning, match="energy window"):
        model.density(model.reference_potential + 1.4)


def test_inconsistent_shapes():
    """Tests that inconsistent shapes are rejected."""
    with pytest.raises(ValueError, match="Inconsistent shapes"):
        LDOSDensityModel(
            density=np.zeros(4),
            potential=np.zeros(3),
            spectral_dos=_spectral_dos(np.ones((1, 3, ENERGIES.size)), 0.0),
        )


class _Contact:
    """Hashable stand-in for a QTBM contact."""

    def __init__(self, fermi_level, voltage, mid_gap_energy):
        self.fermi_level = fermi_level
        self.voltage = voltage
        self.mid_gap_energy = mid_gap_energy
        self.temperature = TEMPERATURE


def test_qtbm_spectral_dos_consistent_with_charge_density():
    """Tests that the QTBM spectral DOS reproduces the QTBM charge density.

    The net excess charge is the occupied part of the spectral DOS minus
    all states below the (local) charge neutrality level.

    """
    rng = np.random.default_rng(1)
    num_kpoints, orbitals_per_atom, num_atoms = 3, [2, 3, 1, 2], 4
    num_orbitals = sum(orbitals_per_atom)
    energies = np.linspace(-1.0, 1.0, 801)
    contacts = [_Contact(0.05, 0.0, 0.0), _Contact(0.05, -0.08, 0.0)]
    ldos = {
        contact: rng.uniform(0, 1, (num_kpoints, num_orbitals, energies.size))
        for contact in contacts
    }
    potential = rng.uniform(-0.1, 0.1, num_orbitals)

    qtbm = SimpleNamespace(
        observables=SimpleNamespace(electron_ldos=ldos),
        electron_energies=energies,
        num_orbitals=num_orbitals,
        device=SimpleNamespace(
            contacts=contacts,
            potential=potential,
            orbital_offsets=np.hstack(([0], np.cumsum(orbitals_per_atom))),
        ),
    )
    qtbm._compute_excess_charge_densities = (
        lambda: QTBM._compute_excess_charge_densities(qtbm)
    )

    charge_density = QTBM.get_charge_density(qtbm)
    spectral_dos = QTBM.get_spectral_density_of_states(qtbm)

    assert spectral_dos.spectral_dos.shape == (2, num_atoms, energies.size)
    assert np.allclose(spectral_dos.chemical_potentials, [0.05, 0.13])

    # Orbital-resolved states below the local neutrality level.
    below = energies <= potential[:, np.newaxis]
    total_ldos = 2 * sum(value.mean(axis=0) for value in ldos.values())
    valence = np.add.reduceat(
        np.trapezoid(total_ldos * below, energies, axis=1),
        qtbm.device.orbital_offsets[:-1],
    )
    occupied = sum(
        np.trapezoid(dos * fermi_dirac(energies - mu, TEMPERATURE), energies, axis=1)
        for dos, mu in zip(spectral_dos.spectral_dos, spectral_dos.chemical_potentials)
    )
    assert np.allclose(occupied - valence, charge_density, atol=1e-2)
