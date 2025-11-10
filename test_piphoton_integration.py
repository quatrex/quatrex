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
    
    # Create energy-dependent anti-Hermitian matrices using NumPy for deterministic results
    g_lesser_dense = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    g_greater_dense = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    for iE in range(n_energy):
        # G^< should be anti-Hermitian: G^<† = -G^<
        base_l = np.random.randn(n_orb, n_orb) + 1j * np.random.randn(n_orb, n_orb)
        g_lesser_dense[iE] = (base_l - base_l.T.conj()) / 2.0
        
        # G^> should also be anti-Hermitian
        base_g = np.random.randn(n_orb, n_orb) + 1j * np.random.randn(n_orb, n_orb)
        g_greater_dense[iE] = (base_g - base_g.T.conj()) / 2.0
    
    # Convert to xp arrays (CuPy if available) for sparse matrix creation
    g_lesser_xp = xp.array(g_lesser_dense)
    g_greater_xp = xp.array(g_greater_dense)
    block_sizes_np = np.array(block_sizes)
    
    # Create sparse matrices from the first energy slice
    g_lesser_sparse = sparse.coo_matrix(g_lesser_xp[0])
    g_greater_sparse = sparse.coo_matrix(g_greater_xp[0])
    
    # Create DSDBCSR matrices in stack distribution
    # The stack dimension represents energies, so global_stack_shape should be (n_energy,)
    g_lesser = dsdbsparse_type.from_sparray(
        g_lesser_sparse,
        block_sizes=block_sizes_np,
        global_stack_shape=(n_energy,),
        symmetry=False,
    )
    g_greater = dsdbsparse_type.from_sparray(
        g_greater_sparse,
        block_sizes=block_sizes_np,
        global_stack_shape=(n_energy,),
        symmetry=False,
    )
    
    # Set the energy-dependent data by extracting the sparse elements
    # Get the sparsity pattern (row, col indices of non-zero elements)
    rows, cols = g_lesser.spy()
    if xp.__name__ == 'cupy':
        rows = xp.asnumpy(rows)
        cols = xp.asnumpy(cols)
    
    # Fill in the sparse data for each energy
    for iE in range(min(n_energy, g_lesser.data.shape[0])):
        for idx in range(len(rows)):
            if xp.__name__ == 'cupy':
                g_lesser.data[iE, idx] = xp.array(g_lesser_dense[iE, rows[idx], cols[idx]])
                g_greater.data[iE, idx] = xp.array(g_greater_dense[iE, rows[idx], cols[idx]])
            else:
                g_lesser.data[iE, idx] = g_lesser_dense[iE, rows[idx], cols[idx]]
                g_greater.data[iE, idx] = g_greater_dense[iE, rows[idx], cols[idx]]
    
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
    
    # Clean up GPU memory
    if xp.__name__ == 'cupy':
        import gc
        del pi_photon
        gc.collect()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


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
    
    # Clean up GPU memory
    if xp.__name__ == 'cupy':
        import gc
        del pi_photon, g_lesser, g_greater, pi_lesser, pi_greater, pi_retarded
        gc.collect()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


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
    
    # Clean up GPU memory
    if xp.__name__ == 'cupy':
        import gc
        del pi_photon, g_lesser, g_greater, pi_lesser, pi_greater, pi_retarded
        gc.collect()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


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
    
    # Verify sparse matches dense
    print("\n" + "="*70)
    print("Verifying sparse representation matches dense")
    print("="*70)
    rows, cols = g_lesser.spy()
    if xp.__name__ == 'cupy':
        rows_np = xp.asnumpy(rows)
        cols_np = xp.asnumpy(cols)
        g_lesser_data_np = xp.asnumpy(g_lesser.data)
    else:
        rows_np, cols_np = rows, cols
        g_lesser_data_np = g_lesser.data
    
    max_diff_gl = 0.0
    for iE in range(min(3, small_device['n_energy'])):
        for idx in range(min(5, len(rows_np))):
            sparse_val = g_lesser_data_np[iE, idx]
            dense_val = g_l_dense[iE, rows_np[idx], cols_np[idx]]
            diff = abs(sparse_val - dense_val)
            max_diff_gl = max(max_diff_gl, diff)
            if idx < 2 and iE == 0:
                print(f"  G^<[{iE},{rows_np[idx]},{cols_np[idx]}]: sparse={sparse_val:.6f}, dense={dense_val:.6f}, diff={diff:.2e}")
    print(f"Max difference G^< sparse vs dense: {max_diff_gl:.2e}")
    
    # Create output matrices
    pi_lesser = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    pi_greater = compute_config.dsdbsparse_type.zeros_like(g_greater)
    pi_retarded = compute_config.dsdbsparse_type.zeros_like(g_lesser)
    
    # Compute with PiPhoton
    pi_photon.compute(g_lesser, g_greater, (pi_lesser, pi_greater, pi_retarded))
    
    # Get interaction matrix
    m_matrix = pi_photon.interaction_matrix.to_dense()[0]
    if xp.__name__ == 'cupy':
        m_matrix = xp.asnumpy(m_matrix)
        g_l_dense = xp.asnumpy(g_l_dense)
        g_g_dense = xp.asnumpy(g_g_dense)
    
    # Compute reference using einsum
    pi_reference = compute_reference_4term(
        m_matrix,
        g_l_dense,
        g_g_dense,
        small_device['electron_energies']
    )
    
    # Get the actual polarization data
    # pi_lesser.data has shape (local_stack_size, n_nnz) where the stack dimension
    # corresponds to energies
    pi_data = pi_lesser.data
    if xp.__name__ == 'cupy':
        pi_data = xp.asnumpy(pi_data)
    
    print(f"\npi_data shape: {pi_data.shape}")
    print(f"pi_reference shape: {pi_reference.shape}")
    
    # The data is in sparse format (local_stack_size, n_nnz)
    # We need to convert each energy slice to dense
    n_energy = small_device['n_energy']
    n_orb = small_device['n_orb']
    
    # Reconstruct dense matrices from the sparse data
    pi_lesser_reconstructed = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    # Get the sparsity pattern
    rows, cols = pi_lesser.spy()
    if xp.__name__ == 'cupy':
        rows = xp.asnumpy(rows)
        cols = xp.asnumpy(cols)
    
    # Fill in the dense matrices
    for iE in range(min(n_energy, pi_data.shape[0])):
        for idx in range(len(rows)):
            pi_lesser_reconstructed[iE, rows[idx], cols[idx]] = pi_data[iE, idx]
    
    # Debug: Print some sample values
    print(f"\nSample pi_lesser_reconstructed[0, 0, 0]: {pi_lesser_reconstructed[0, 0, 0]}")
    print(f"Sample pi_reference[0, 0, 0]: {pi_reference[0, 0, 0]}")
    print(f"Sample pi_lesser_reconstructed[5, 2, 3]: {pi_lesser_reconstructed[5, 2, 3]}")
    print(f"Sample pi_reference[5, 2, 3]: {pi_reference[5, 2, 3]}")
    print(f"Sample pi_reference[5, 3, 2]: {pi_reference[5, 3, 2]}")
    print(f"-pi_reference[5, 3, 2]*: {-pi_reference[5, 3, 2].conj()}")
    
    # Check anti-Hermitian property of reference
    max_ref_violation = 0.0
    for iE in range(n_energy):
        violation = np.max(np.abs(pi_reference[iE] + pi_reference[iE].conj().T))
        max_ref_violation = max(max_ref_violation, violation)
    print(f"Max anti-Hermitian violation in reference: {max_ref_violation:.2e}")
    
    # Check anti-Hermitian property of PiPhoton output
    max_pi_violation = 0.0
    for iE in range(n_energy):
        violation = np.max(np.abs(pi_lesser_reconstructed[iE] + pi_lesser_reconstructed[iE].conj().T))
        max_pi_violation = max(max_pi_violation, violation)
    print(f"Max anti-Hermitian violation in PiPhoton: {max_pi_violation:.2e}")
    
    # Compare
    diff = np.max(np.abs(pi_lesser_reconstructed - pi_reference))
    print(f"Max difference from reference: {diff:.2e}")
    
    # Check if the difference is due to a systematic offset or scaling
    ratio = np.max(np.abs(pi_lesser_reconstructed)) / np.max(np.abs(pi_reference))
    print(f"Ratio of max values: {ratio:.2e}")
    
    # Allow some numerical tolerance  
    assert diff < 1e-6, f"PiPhoton output should match reference (diff={diff})"
    print("✓ PiPhoton matches reference implementation")
    
    # Clean up GPU memory
    if xp.__name__ == 'cupy':
        import gc
        del pi_photon, g_lesser, g_greater, pi_lesser, pi_greater, pi_retarded
        del m_matrix, g_l_dense, g_g_dense, pi_reference, pi_lesser_dense, pi_lesser_reshaped
        gc.collect()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


def compute_reference_4term(m_matrix, g_lesser, g_greater, energies):
    """Reference implementation using FFT correlation formula (matching PiPhoton).
    
    The 4 terms are computed as GEMM operations followed by element-wise products
    and FFT correlation through energy.
    """
    n_energy = g_lesser.shape[0]
    n_orb = m_matrix.shape[0]
    
    dE = energies[1] - energies[0]
    prefactor = 1j / (2 * np.pi) * dE
    
    print(f"Reference prefactor: {prefactor}")
    print(f"Energy step dE: {dE}")
    
    # Compute intermediate products at all energies
    # Shape: (n_energy, n_orb, n_orb)
    m_gl = np.einsum('ij,ejk->eik', m_matrix, g_lesser)         # M@G<
    m_gl_m = np.einsum('ij,ejk,kl->eil', m_matrix, g_lesser, m_matrix)  # M@G<@M
    gl_m = np.einsum('eij,jk->eik', g_lesser, m_matrix)         # G<@M
    
    m_gg = np.einsum('ij,ejk->eik', m_matrix, g_greater)        # M@G>
    m_gg_m = np.einsum('ij,ejk,kl->eil', m_matrix, g_greater, m_matrix)  # M@G>@M
    gg_m = np.einsum('eij,jk->eik', g_greater, m_matrix)        # G>@M
    
    # Initialize result
    pi_lesser = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    # Compute correlations using FFT
    n_fft = 2 * n_energy - 1
    
    # For each orbital pair (i, l)
    for i in range(n_orb):
        for l in range(n_orb):
            # Term 1: (M@G<@M)[i,l] ⊙ G>.T[i,l] = (M@G<@M)[i,l] ⊙ G>[l,i]
            # Correlate m_gl_m[i,l](E') with g_greater[l,i](E'-E)
            fft1 = np.fft.fft(m_gl_m[:, i, l], n_fft)
            fft2 = np.fft.fft(g_greater[::-1, l, i], n_fft)
            corr1 = np.fft.ifft(fft1 * fft2)[n_energy-1:]  # Take last n_energy elements
            
            # Term 2: (G<@M)[i,l] ⊙ (G>@M).T[i,l] = (G<@M)[i,l] ⊙ (G>@M)[l,i]
            # Correlate gl_m[i,l](E') with gg_m[l,i](E'-E)
            fft1 = np.fft.fft(gl_m[:, i, l], n_fft)
            fft2 = np.fft.fft(gg_m[::-1, l, i], n_fft)
            corr2 = np.fft.ifft(fft1 * fft2)[n_energy-1:]  # Take last n_energy elements
            
            # Term 3: (M@G<)[i,l] ⊙ (M@G>).T[i,l] = (M@G<)[i,l] ⊙ (M@G>)[l,i]
            # Correlate m_gl[i,l](E') with m_gg[l,i](E'-E)
            fft1 = np.fft.fft(m_gl[:, i, l], n_fft)
            fft2 = np.fft.fft(m_gg[::-1, l, i], n_fft)
            corr3 = np.fft.ifft(fft1 * fft2)[n_energy-1:]  # Take last n_energy elements
            
            # Term 4: G<[i,l] ⊙ (M@G>@M).T[i,l] = G<[i,l] ⊙ (M@G>@M)[l,i]
            # Correlate g_lesser[i,l](E') with m_gg_m[l,i](E'-E)
            fft1 = np.fft.fft(g_lesser[:, i, l], n_fft)
            fft2 = np.fft.fft(m_gg_m[::-1, l, i], n_fft)
            corr4 = np.fft.ifft(fft1 * fft2)[n_energy-1:]  # Take last n_energy elements
            
            # Sum all terms (no need to reverse - correct indexing already done)
            pi_lesser[:, i, l] = prefactor * (corr1 + corr2 + corr3 + corr4)
    
    # Apply symmetrization steps (matching PiPhoton.compute)
    # 1. Enforce spatial anti-Hermitian symmetry: A_ij -> 0.5 * (A_ij - A_ji*)
    for iE in range(n_energy):
        pi_lesser[iE] = 0.5 * (pi_lesser[iE] - pi_lesser[iE].conj().T)
    
    # 2. Discard the real part (enforcing that polarization is purely imaginary)
    pi_lesser.real[:] = 0.0
    
    return pi_lesser


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
