# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from types import SimpleNamespace

import numpy as np
import pytest

from quatrex.core.constants import hbar, k_B
from quatrex.core.qtbm import QTBM
from quatrex.core.scba import SCBA
from quatrex.core.statistics import fermi_dirac
from quatrex.core.utils import split_trapezoid
from quatrex.electrostatics.density_response import (
    GrapheneDensityModel,
    LDOSDensityModel,
    SpectralDensityOfStates,
    dirac_net_density,
    local_quasi_fermi_levels,
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
    total_ldos = 2 * sum(value.mean(axis=0) for value in ldos.values())
    valence = np.add.reduceat(
        split_trapezoid(total_ldos, energies, potential)[0],
        qtbm.device.orbital_offsets[:-1],
    )
    occupied = sum(
        np.trapezoid(dos * fermi_dirac(energies - mu, TEMPERATURE), energies, axis=1)
        for dos, mu in zip(spectral_dos.spectral_dos, spectral_dos.chemical_potentials)
    )
    assert np.allclose(occupied - valence, charge_density, rtol=1e-10, atol=1e-12)


def _net_excess(dos, chemical_potential, neutrality_levels):
    """Reference net excess density (electrons above, holes below)."""
    f = fermi_dirac(ENERGIES - chemical_potential, TEMPERATURE)
    __, electrons = split_trapezoid(dos * f, ENERGIES, neutrality_levels)
    holes, __ = split_trapezoid(dos * (1 - f), ENERGIES, neutrality_levels)
    return electrons - holes


def _structured_dos(seed: int, num_components: int, num_atoms: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    peaks = rng.uniform(-0.3, 0.3, (num_components, num_atoms, 1))
    return 0.5 + np.exp(-(((ENERGIES - peaks) / 0.05) ** 2))


def test_quasi_fermi_levels_equilibrium():
    """Tests that the fit recovers Fermi-Dirac chemical potentials,
    including one inside a gap of the spectral function."""
    dos = _structured_dos(seed=2, num_components=1, num_atoms=4)[0]
    chemical_potentials = np.array([-0.2, 0.0, 0.05, 0.3])
    # Gap of +-0.3 eV around the chemical potential of the last atom.
    dos[-1, np.abs(ENERGIES - chemical_potentials[-1]) < 0.3] = 0.0
    neutrality_levels = np.array([0.0, -0.1, 0.1, 0.3])

    net_density = np.array(
        [
            _net_excess(dos[[i]], mu, neutrality_levels[[i]])[0]
            for i, mu in enumerate(chemical_potentials)
        ]
    )
    fitted = local_quasi_fermi_levels(
        dos, net_density, ENERGIES, neutrality_levels, TEMPERATURE
    )
    assert np.allclose(fitted, chemical_potentials, atol=1e-6)


def _contact_resolved_and_fitted(chemical_potentials):
    """Contact-resolved model and its single-component (NEGF) analogue."""
    num_atoms = 5
    dos = _structured_dos(seed=0, num_components=2, num_atoms=num_atoms)
    neutrality_levels = np.full(num_atoms, -0.4)
    density = sum(
        _net_excess(d, mu, neutrality_levels) for d, mu in zip(dos, chemical_potentials)
    )
    potential = np.zeros(num_atoms)

    contact_resolved = LDOSDensityModel(
        density, potential, _spectral_dos(dos, chemical_potentials)
    )
    spectral_function = dos.sum(axis=0)
    fitted = local_quasi_fermi_levels(
        spectral_function, density, ENERGIES, neutrality_levels, TEMPERATURE
    )
    single_component = LDOSDensityModel(
        density,
        potential,
        SpectralDensityOfStates(
            spectral_dos=spectral_function[np.newaxis],
            energies=ENERGIES,
            chemical_potentials=fitted[np.newaxis],
            temperatures=np.array([TEMPERATURE]),
        ),
    )
    return contact_resolved, single_component, fitted


def test_quasi_fermi_model_equilibrium():
    """Tests that, in equilibrium, the spectral function with fitted
    quasi-Fermi levels is identical to the contact-resolved model."""
    contact_resolved, single_component, fitted = _contact_resolved_and_fitted(
        np.array([0.02, 0.02])
    )
    assert np.allclose(fitted, 0.02, atol=1e-6)
    for value in np.linspace(-0.3, 0.3, 7):
        potential = np.full(5, value)
        assert np.allclose(
            single_component.density(potential),
            contact_resolved.density(potential),
            rtol=1e-6,
            atol=1e-10,
        )
        assert np.allclose(
            single_component.density_derivative(potential),
            contact_resolved.density_derivative(potential),
            rtol=1e-5,
        )


def test_quasi_fermi_model_nonequilibrium():
    """Tests the quasi-Fermi approximation under a bias of ~2 k_B T.

    The density is still exact at the reference potential (so the SCSP
    fixed point is unaffected), and the derivative stays close to the
    contact-resolved one (it only affects the convergence rate).

    """
    contact_resolved, single_component, fitted = _contact_resolved_and_fitted(
        np.array([0.0, -0.05])
    )
    assert np.all((fitted > -0.05) & (fitted < 0.0))

    potential = np.zeros(5)
    assert np.allclose(
        single_component.density(potential),
        contact_resolved.density(potential),
        rtol=0,
        atol=1e-14,
    )
    for value in (-0.1, 0.0, 0.05):
        potential = np.full(5, value)
        assert np.allclose(
            single_component.density_derivative(potential),
            contact_resolved.density_derivative(potential),
            rtol=0.2,
        )


def test_scba_spectral_dos():
    """Tests the SCBA spectral DOS on a stub in equilibrium.

    The fitted quasi-Fermi levels must recover the chemical potential,
    and the resulting model must reproduce the SCBA charge density.

    """
    rng = np.random.default_rng(3)
    orbitals_per_atom = [2, 3, 1, 2]
    num_orbitals = sum(orbitals_per_atom)
    energies = np.linspace(-1.0, 1.0, 801)
    chemical_potential = 0.05
    contacts = [_Contact(chemical_potential, 0.0, 0.0), _Contact(0.05, 0.0, 0.0)]

    # Spectral electron and hole densities with shape (E, orbitals).
    spectral_function = rng.uniform(0, 1, (energies.size, num_orbitals))
    occupation = fermi_dirac(energies - chemical_potential, TEMPERATURE)[:, None]
    potential = np.repeat(
        rng.uniform(-0.1, 0.1, len(orbitals_per_atom)), orbitals_per_atom
    )

    scba = SimpleNamespace(
        observables=SimpleNamespace(
            electron_density=spectral_function * occupation,
            hole_density=spectral_function * (1 - occupation),
        ),
        electron_energies=energies,
        data=SimpleNamespace(orbitals_per_atom=orbitals_per_atom),
        device=SimpleNamespace(contacts=contacts, potential=potential),
    )
    scba._reference_contact = lambda: SCBA._reference_contact(scba)
    scba._compute_excess_charge_densities = (
        lambda: SCBA._compute_excess_charge_densities(scba)
    )
    scba.get_charge_density = lambda: SCBA.get_charge_density(scba)

    spectral_dos = SCBA.get_spectral_density_of_states(scba)
    charge_density = SCBA.get_charge_density(scba)

    assert spectral_dos.spectral_dos.shape == (1, 4, energies.size)
    assert spectral_dos.chemical_potentials.shape == (1, 4)
    assert np.allclose(spectral_dos.chemical_potentials, chemical_potential, atol=1e-4)

    atom_potential = potential[np.hstack(([0], np.cumsum(orbitals_per_atom)))[:-1]]
    model = LDOSDensityModel(charge_density, atom_potential, spectral_dos)
    assert np.allclose(model.density(atom_potential), charge_density)
