#!/usr/bin/env python3
"""
Integration test for PiPhoton polarization computation.

This test creates a small device, initializes PiPhoton, and verifies:
1. The polarization can be computed without errors
2. Physical properties are satisfied (anti-Hermitian, imaginary diagonal)
3. Results match reference implementation
"""

import numpy as np
import pytest
from pathlib import Path
from scipy import sparse as sps

from qttools import xp, sparse
from qttools.comm import comm as qttools_comm
from qttools.datastructures import DSDBCSR

from quatrex.photon.polarization import PiPhoton


class MockQuatrexConfig:
    """Mock configuration matching SigmaPhoton pattern."""

    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        self.photon = type('obj', (object,), {
            'polarization': [1.0, 0.0, 0.0],
            'light_intensity': 1e10,
            'photon_energy': 0.2  # Match energy step
        })
        self.scba = type('obj', (object,), {'symmetric': False})
        self.device = type('obj', (object,), {'construct_from_unit_cell': False})


class MockComputeConfig:
    """Mock compute configuration."""

    def __init__(self):
        self.dsdbsparse_type = DSDBCSR
        self.convolve = type('obj', (object,), {'batch_size': None})


@pytest.fixture
def small_device(tmp_path):
    """Create a small device similar to carbon nanotube structure."""
    return _create_small_device(tmp_path)


def _create_small_device(tmp_path):
    """Helper to create a small device (can be called directly)."""
    # Small system for fast testing
    n_orb = 6
    n_energy = 12
    
    # Energy grids
    electron_energies = np.linspace(-1.0, 1.0, n_energy)
    photon_energies = electron_energies.copy()
    
    # Block structure: 3 blocks of 2 orbitals each
    block_sizes = np.array([2, 2, 2])
    
    # Create a realistic block-tridiagonal Hamiltonian
    # This mimics a 1D chain structure like CNT
    h_sparray = sps.lil_matrix((n_orb, n_orb), dtype=complex)
    
    np.random.seed(123)
    
    # Onsite terms (diagonal blocks)
    for ib in range(3):
        start = ib * 2
        end = start + 2
        onsite = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        onsite = (onsite + onsite.T.conj()) / 2  # Hermitian onsite
        h_sparray[start:end, start:end] = onsite
    
    # Nearest-neighbor hopping (off-diagonal blocks)
    for ib in range(2):
        start1 = ib * 2
        end1 = start1 + 2
        start2 = (ib + 1) * 2
        end2 = start2 + 2
        
        hopping = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        h_sparray[start1:end1, start2:end2] = hopping
        h_sparray[start2:end2, start1:end1] = hopping.T.conj()  # Hermitian
    
    h_sparray = h_sparray.tocoo()
    
    # Orbital positions (1D chain along x-axis)
    orbital_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
    ])
    
    # Save input files
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    
    # Save as sparse matrix in npz format
    sps.save_npz(input_dir / "hamiltonian.npz", h_sparray.tocsr())
    np.save(input_dir / "block_sizes.npy", block_sizes)
    np.save(input_dir / "grid.npy", orbital_positions)
    
    return {
        'n_orb': n_orb,
        'n_energy': n_energy,
        'electron_energies': electron_energies,
        'photon_energies': photon_energies,
        'block_sizes': block_sizes,
        'h_sparray': h_sparray,
        'orbital_positions': orbital_positions,
        'input_dir': input_dir,
    }


def create_test_greens_functions(n_energy, n_orb, block_sizes, dsdbsparse_type):
    """Create test Green's functions as distributed sparse matrices.
    
    Creates physically reasonable anti-Hermitian G^< and G^> matrices.
    """
    np.random.seed(456)
    
    # Create energy-dependent anti-Hermitian matrices
    g_lesser_dense = xp.zeros((n_energy, n_orb, n_orb), dtype=complex)
    g_greater_dense = xp.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    for iE in range(n_energy):
        # G^< should be anti-Hermitian: G^<† = -G^<
        base_l = xp.random.randn(n_orb, n_orb) + 1j * xp.random.randn(n_orb, n_orb)
        g_lesser_dense[iE] = (base_l - base_l.T.conj()) / 2.0
        
        # G^> should also be anti-Hermitian
        base_g = xp.random.randn(n_orb, n_orb) + 1j * xp.random.randn(n_orb, n_orb)
        g_greater_dense[iE] = (base_g - base_g.T.conj()) / 2.0
    
    # Convert to xp arrays (CuPy if available)
    g_lesser_xp = xp.array(g_lesser_dense)
    g_greater_xp = xp.array(g_greater_dense)
    block_sizes_np = np.array(block_sizes)
    
    # Create sparse matrices from the first energy slice
    g_lesser_sparse = sparse.coo_matrix(g_lesser_dense[0])
    g_greater_sparse = sparse.coo_matrix(g_greater_dense[0])
    
    # Create DSDBCSR matrices in stack distribution
    g_lesser = dsdbsparse_type.from_sparray(
        g_lesser_sparse,
        block_sizes=block_sizes_np,
        global_stack_shape=(qttools_comm.stack.size,),
        symmetry=False,
    )
    g_greater = dsdbsparse_type.from_sparray(
        g_greater_sparse,
        block_sizes=block_sizes_np,
        global_stack_shape=(qttools_comm.stack.size,),
        symmetry=False,
    )
    
    # Set the energy-dependent data directly
    g_lesser.data = g_lesser_xp.reshape(n_energy, -1)[: g_lesser.data.shape[0], : g_lesser.data.shape[1]]
    g_greater.data = g_greater_xp.reshape(n_energy, -1)[: g_greater.data.shape[0], : g_greater.data.shape[1]]
    
    return g_lesser, g_greater, g_lesser_dense, g_greater_dense


def test_piphoton_initialization(small_device):
    """Test that PiPhoton can be initialized."""
    config = MockQuatrexConfig(small_device['input_dir'])
    compute_config = MockComputeConfig()
    
    pi_photon = PiPhoton(
        quatrex_config=config,
        compute_config=compute_config,
        photon_energies=small_device['photon_energies'],
        electron_energies=small_device['electron_energies'],
    )
    
    # Check that interaction matrix was created
    assert pi_photon.interaction_matrix is not None
    assert pi_photon.prefactor is not None
    
    print("✓ PiPhoton initialized successfully")


def test_piphoton_compute_runs(small_device):
    """Test that PiPhoton.compute() runs without errors."""
    config = MockQuatrexConfig(small_device['input_dir'])
    compute_config = MockComputeConfig()
    
    pi_photon = PiPhoton(
        quatrex_config=config,
        compute_config=compute_config,
        photon_energies=small_device['photon_energies'],
        electron_energies=small_device['electron_energies'],
    )
    
    # Create Green's functions
    g_lesser, g_greater, _, _ = create_test_greens_functions(
        small_device['n_energy'],
        small_device['n_orb'],
        small_device['block_sizes'],
        compute_config.dsdbsparse_type,
    )
    
    # Create output matrices
    pi_lesser = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    pi_greater = compute_config.dsdbsparse_type.zeros_like(g_greater)
    pi_retarded = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    
    # Run computation
    pi_photon.compute(g_lesser, g_greater, (pi_lesser, pi_greater, pi_retarded))
    
    # Check outputs are not all zero
    assert not np.allclose(pi_lesser.data, 0), "pi_lesser should not be all zeros"
    assert not np.allclose(pi_greater.data, 0), "pi_greater should not be all zeros"
    
    print("✓ PiPhoton.compute() completed successfully")


def test_piphoton_physical_properties(small_device):
    """Test that computed polarization satisfies physical properties."""
    config = MockQuatrexConfig(small_device['input_dir'])
    compute_config = MockComputeConfig()
    
    pi_photon = PiPhoton(
        quatrex_config=config,
        compute_config=compute_config,
        photon_energies=small_device['photon_energies'],
        electron_energies=small_device['electron_energies'],
    )
    
    # Create Green's functions
    g_lesser, g_greater, _, _ = create_test_greens_functions(
        small_device['n_energy'],
        small_device['n_orb'],
        small_device['block_sizes'],
        compute_config.dsdbsparse_type,
    )
    
    # Create output matrices
    pi_lesser = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    pi_greater = compute_config.dsdbsparse_type.zeros_like(g_greater)
    pi_retarded = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    
    # Compute polarization
    pi_photon.compute(g_lesser, g_greater, (pi_lesser, pi_greater, pi_retarded))
    
    # Extract data
    pi_l_data = pi_lesser.data
    pi_g_data = pi_greater.data
    
    n_energy = pi_l_data.shape[0]
    
    # Test 1: Anti-Hermitian property (π^<)† = -π^<
    print("\nTesting anti-Hermitian property...")
    max_antiherm_violation = 0.0
    for iE in range(n_energy):
        violation = np.max(np.abs(pi_l_data[iE] + pi_l_data[iE].conj().T))
        max_antiherm_violation = max(max_antiherm_violation, violation)
    
    print(f"  Max anti-Hermitian violation: {max_antiherm_violation:.2e}")
    assert max_antiherm_violation < 1e-10, "π^< should be anti-Hermitian"
    print("  ✓ π^< is anti-Hermitian")
    
    # Test 2: Diagonal should be purely imaginary
    print("\nTesting diagonal is purely imaginary...")
    max_diag_real = 0.0
    for iE in range(n_energy):
        diag = np.diag(pi_l_data[iE])
        max_diag_real = max(max_diag_real, np.max(np.abs(diag.real)))
    
    print(f"  Max diagonal real part: {max_diag_real:.2e}")
    assert max_diag_real < 1e-10, "Diagonal should be purely imaginary"
    print("  ✓ Diagonal is purely imaginary")
    
    # Test 3: Energy symmetry π^>(E) = -π^<(-E)†
    print("\nTesting energy symmetry...")
    max_energy_sym_violation = 0.0
    for iE in range(n_energy):
        # π^>(E) should equal -π^<(-E)†
        # For discrete grid: π^>[iE] = -π^<[ne-1-iE]†
        expected = -pi_l_data[n_energy - 1 - iE].conj()
        violation = np.max(np.abs(pi_g_data[iE] - expected))
        max_energy_sym_violation = max(max_energy_sym_violation, violation)
    
    print(f"  Max energy symmetry violation: {max_energy_sym_violation:.2e}")
    assert max_energy_sym_violation < 1e-10, "Energy symmetry should hold"
    print("  ✓ Energy symmetry satisfied")
    
    print("\n✅ All physical properties verified!")


def test_piphoton_matches_reference(small_device):
    """Test that PiPhoton output matches reference einsum implementation."""
    config = MockQuatrexConfig(small_device['input_dir'])
    compute_config = MockComputeConfig()
    
    pi_photon = PiPhoton(
        quatrex_config=config,
        compute_config=compute_config,
        photon_energies=small_device['photon_energies'],
        electron_energies=small_device['electron_energies'],
    )
    
    # Create Green's functions
    g_lesser, g_greater, g_l_dense, g_g_dense = create_test_greens_functions(
        small_device['n_energy'],
        small_device['n_orb'],
        small_device['block_sizes'],
        compute_config.dsdbsparse_type,
    )
    
    # Create output matrices
    pi_lesser = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    pi_greater = compute_config.dsdbsparse_type.zeros_like(g_greater)
    pi_retarded = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    
    # Compute with PiPhoton
    pi_photon.compute(g_lesser, g_greater, (pi_lesser, pi_greater, pi_retarded))
    
    # Get interaction matrix
    m_matrix = pi_photon.interaction_matrix.to_array()
    
    # Compute reference using einsum
    pi_reference = compute_reference_4term(
        m_matrix,
        g_l_dense,
        g_g_dense,
        small_device['electron_energies']
    )
    
    # Compare
    diff = np.max(np.abs(pi_lesser.data - pi_reference))
    print(f"\nMax difference from reference: {diff:.2e}")
    
    # Allow some numerical tolerance
    assert diff < 1e-10, f"PiPhoton output should match reference (diff={diff})"
    print("✓ PiPhoton matches reference implementation")


def compute_reference_4term(m_matrix, g_lesser, g_greater, energies):
    """Reference implementation using einsum with corrected FFT correlation."""
    n_energy = g_lesser.shape[0]
    n_orb = m_matrix.shape[0]
    
    dE = energies[1] - energies[0]
    prefactor = 1j / (2 * np.pi) * dE
    
    pi = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    # Compute using the corrected formula: sum_{E_diff} A(E+E_diff) * B(E_diff)
    for iE in range(n_energy):
        for iE_diff in range(n_energy):
            if iE + iE_diff >= n_energy:
                continue
            iEp = iE + iE_diff
            
            # Term 1: M@G<@M ⊙ G>
            term1 = np.einsum('ij,jk,kl,li->il', 
                            m_matrix, g_lesser[iEp], m_matrix, g_greater[iE_diff])
            
            # Term 2: G<@M ⊙ G>@M
            term2 = np.einsum('ij,jk,kl,li->il',
                            g_lesser[iEp], m_matrix, g_greater[iE_diff], m_matrix)
            
            # Term 3: M@G< ⊙ M@G>
            term3 = np.einsum('ij,jl,lk,ki->il',
                            m_matrix, g_lesser[iEp], m_matrix, g_greater[iE_diff])
            
            # Term 4: G< ⊙ M@G>@M
            term4 = np.einsum('il,jk,kl,ji->il',
                            g_lesser[iEp], g_greater[iE_diff], m_matrix, m_matrix)
            
            pi[iE] += prefactor * (term1 + term2 + term3 + term4)
    
    return pi


if __name__ == "__main__":
    # Run tests manually for debugging
    import tempfile
    
    print("="*80)
    print("PiPhoton Integration Tests")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create device
        print("\nCreating small device...")
        device = _create_small_device(tmp_path)
        print(f"  Orbitals: {device['n_orb']}")
        print(f"  Energies: {device['n_energy']}")
        print(f"  Blocks: {device['block_sizes']}")
        
        # Test 1
        print("\n" + "="*80)
        print("TEST 1: Initialization")
        print("="*80)
        test_piphoton_initialization(device)
        
        # Test 2
        print("\n" + "="*80)
        print("TEST 2: Compute Runs")
        print("="*80)
        test_piphoton_compute_runs(device)
        
        # Test 3
        print("\n" + "="*80)
        print("TEST 3: Physical Properties")
        print("="*80)
        test_piphoton_physical_properties(device)
        
        # Test 4
        print("\n" + "="*80)
        print("TEST 4: Matches Reference")
        print("="*80)
        test_piphoton_matches_reference(device)
        
    print("\n" + "="*80)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*80)
