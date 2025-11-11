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
    """Create test Green's functions as distributed sparse block-tridiagonal matrices.
    
    Creates physically reasonable anti-Hermitian G^< and G^> matrices with
    block-tridiagonal structure (required for bd_matmul_distr to work correctly).
    
    For n_blocks=3 with block_size=2, the structure is:
    [G00  G01   0 ]
    [G10  G11  G12]
    [ 0   G21  G22]
    
    Each Gij is a 2x2 block.
    """
    np.random.seed(456)
    n_blocks = len(block_sizes)
    
    # Create block-tridiagonal sparse matrices
    g_lesser_dense = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    g_greater_dense = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    for iE in range(n_energy):
        # Create block-tridiagonal structure
        for ib in range(n_blocks):
            # Diagonal blocks
            start = sum(block_sizes[:ib])
            end = start + block_sizes[ib]
            
            # Create anti-Hermitian diagonal block
            base_l = np.random.randn(block_sizes[ib], block_sizes[ib]) + 1j * np.random.randn(block_sizes[ib], block_sizes[ib])
            g_lesser_dense[iE, start:end, start:end] = (base_l - base_l.T.conj()) / 2.0
            
            base_g = np.random.randn(block_sizes[ib], block_sizes[ib]) + 1j * np.random.randn(block_sizes[ib], block_sizes[ib])
            g_greater_dense[iE, start:end, start:end] = (base_g - base_g.T.conj()) / 2.0
            
            # Off-diagonal blocks (only nearest neighbors)
            if ib < n_blocks - 1:
                start1 = sum(block_sizes[:ib])
                end1 = start1 + block_sizes[ib]
                start2 = sum(block_sizes[:ib+1])
                end2 = start2 + block_sizes[ib+1]
                
                # Upper off-diagonal
                off_l = np.random.randn(block_sizes[ib], block_sizes[ib+1]) + 1j * np.random.randn(block_sizes[ib], block_sizes[ib+1])
                g_lesser_dense[iE, start1:end1, start2:end2] = off_l
                # Lower off-diagonal (enforce anti-Hermitian: G^† = -G)
                g_lesser_dense[iE, start2:end2, start1:end1] = -off_l.T.conj()
                
                off_g = np.random.randn(block_sizes[ib], block_sizes[ib+1]) + 1j * np.random.randn(block_sizes[ib], block_sizes[ib+1])
                g_greater_dense[iE, start1:end1, start2:end2] = off_g
                g_greater_dense[iE, start2:end2, start1:end1] = -off_g.T.conj()
    
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
        symmetry_op=lambda a: -a.conj(),
    )
    g_greater = dsdbsparse_type.from_sparray(
        g_greater_sparse,
        block_sizes=block_sizes_np,
        global_stack_shape=(n_energy,),
        symmetry=False,
        symmetry_op=lambda a: -a.conj(),
    )

    g_lesser.symmetrize(xp.subtract)
    g_greater.symmetrize(xp.subtract)
    
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
    
    # Extract sparse data
    pi_l_data = pi_lesser.data
    pi_g_data = pi_greater.data
    
    if xp.__name__ == 'cupy':
        pi_l_data = xp.asnumpy(pi_l_data)
        pi_g_data = xp.asnumpy(pi_g_data)
    
    n_energy = pi_l_data.shape[0]
    n_orb = small_device['n_orb']
    
    # Get the sparsity pattern to convert sparse data to dense matrices
    rows, cols = pi_lesser.spy()
    if xp.__name__ == 'cupy':
        rows = xp.asnumpy(rows)
        cols = xp.asnumpy(cols)
    
    # Convert sparse data to dense matrices per energy
    pi_l_dense = pi_lesser.to_dense()
    pi_g_dense = pi_greater.to_dense()
    
    # Test 1: Anti-Hermitian property (π^<)† = -π^<
    print("\nTesting anti-Hermitian property...")
    max_antiherm_violation = 0.0
    for iE in range(n_energy):
        violation = np.max(np.abs(pi_l_dense[iE] + pi_l_dense[iE].conj().T))
        max_antiherm_violation = max(max_antiherm_violation, violation)
    
    print(f"  Max anti-Hermitian violation: {max_antiherm_violation:.2e}")
    assert max_antiherm_violation < 1e-10, "π^< should be anti-Hermitian"
    print("  ✓ π^< is anti-Hermitian")
    
    # Test 2: Diagonal should be purely imaginary
    print("\nTesting diagonal is purely imaginary...")
    max_diag_real = 0.0
    for iE in range(n_energy):
        diag = np.diag(pi_l_dense[iE])
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
        expected = -pi_l_dense[n_energy - 1 - iE].conj()
        violation = np.max(np.abs(pi_g_dense[iE] - expected))
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
    
    # Debug: Check PiPhoton's polarization BEFORE extracting to dense
    # to see if there's any processing we're missing
    print(f"\nPiPhoton polarization stats (in sparse format):")
    print(f"  pi_lesser.data shape: {pi_lesser.data.shape}")
    print(f"  pi_lesser.data max abs: {np.max(np.abs(xp.asnumpy(pi_lesser.data) if xp.__name__ == 'cupy' else pi_lesser.data)):.6e}")
    
    # Get interaction matrix
    m_matrix = pi_photon.interaction_matrix.to_dense()[0]
    # if xp.__name__ == 'cupy':
    #     m_matrix = xp.asnumpy(m_matrix)
    #     g_l_dense = xp.asnumpy(g_l_dense)
    #     g_g_dense = xp.asnumpy(g_g_dense)
    
    # Debug: Print M matrix properties
    print(f"\nM matrix shape: {m_matrix.shape}")
    print(f"M matrix max abs: {np.max(np.abs(m_matrix)):.6e}")
    print(f"M matrix sample [0,0]: {m_matrix[0,0]:.6e}")
    print(f"M matrix sample [0,1]: {m_matrix[0,1]:.6e}")
    print(f"M matrix is Hermitian: {np.allclose(m_matrix, m_matrix.conj().T)}")
    print(f"M matrix sparsity: {np.sum(np.abs(m_matrix) > 1e-12)} / {m_matrix.size} elements non-zero")
    
    # Debug: Print G matrix properties
    print(f"\nG< matrix (first energy) max abs: {np.max(np.abs(g_l_dense[0])):.6e}")
    print(f"G< sample [0,0,0]: {g_l_dense[0,0,0]:.6e}")
    print(f"G> matrix (first energy) max abs: {np.max(np.abs(g_g_dense[0])):.6e}")
    
    # Get sparsity pattern from the Green's functions  
    g_sparsity_mask = np.abs(g_l_dense[0]) > 0  # Boolean mask of non-zero elements in G
    print(f"\nG sparsity pattern: {np.sum(g_sparsity_mask)} / {g_sparsity_mask.size} elements non-zero")
    
    # Get sparsity from M matrix (which is sparse)
    m_sparsity_mask = np.abs(m_matrix) > 0
    print(f"M sparsity pattern: {np.sum(m_sparsity_mask)} / {m_sparsity_mask.size} elements non-zero")
    
    # Use direct summation approach (validated in test_piphoton_dense.py)
    # This matches FFT correlation with machine precision
    print(f"Using direct summation reference (validated approach)")
    
    # Compute reference using direct summation
    pi_reference = compute_reference_4term(
        m_matrix,
        g_l_dense,
        g_g_dense,
        small_device['electron_energies'],
        sparsity_mask=g_sparsity_mask,
        output_mask=g_sparsity_mask  # Only compute elements in G's sparsity pattern
    )
    
    print(f"\nReference polarization stats:")
    print(f"  pi_reference max abs: {np.max(np.abs(pi_reference)):.6e}")
    print(f"  pi_reference sample [0,0,0]: {pi_reference[0,0,0]:.6e}")
    
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
    pi_lesser_reconstructed = pi_lesser.to_dense()
    
    # Debug: Print some sample values
    print(f"\nSample pi_lesser_reconstructed[0, 0, 0]: {pi_lesser_reconstructed[0, 0, 0]}")
    print(f"Sample pi_reference[0, 0, 0]: {pi_reference[0, 0, 0]}")
    
    # Only compare elements that are in PiPhoton's output sparsity pattern
    print(f"\nComparing only elements in PiPhoton's sparsity pattern ({len(rows)} elements)...")
    pi_reference[:, g_sparsity_mask == False] = 0.0  # Zero out elements not in sparsity pattern
    max_diff_sparse = np.max(np.abs(xp.asnumpy(pi_lesser_reconstructed) - xp.asnumpy(pi_reference)))
    
    print(f"Max difference on sparse elements: {max_diff_sparse:.2e}")
    
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
    
    # Compare only sparse elements
    diff = max_diff_sparse
    
    # Check relative error
    rel_error = diff / np.max(np.abs(pi_lesser_reconstructed))
    print(f"Relative error: {rel_error:.2e}")
    
    # Check if the difference is due to a systematic offset or scaling
    ratio = np.max(np.abs(pi_lesser_reconstructed)) / np.max(np.abs(pi_reference))
    print(f"Ratio of max values: {ratio:.2e}")
    
    # Allow numerical tolerance for sparse vs dense comparison
    # Sparse block-distributed operations may have different rounding than dense einsum
    # A relative error of ~5% is acceptable for comparing different numerical implementations
    assert diff < 1e-15, f"PiPhoton output should match reference (diff={diff}, rel_err={rel_error})"
    print("✓ PiPhoton matches reference implementation (within numerical tolerance)")
    
    # Clean up GPU memory
    if xp.__name__ == 'cupy':
        import gc
        del pi_photon, g_lesser, g_greater, pi_lesser, pi_greater, pi_retarded
        del m_matrix, g_l_dense, g_g_dense, pi_reference
        gc.collect()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


def compute_reference_4term(m_matrix, g_lesser, g_greater, energies, sparsity_mask, output_mask=None):
    """Reference implementation using direct summation (validated approach).
    
    Computes polarization using the formula:
    π^<_{il}(E) = (i/2π) * dE * ∑_{E'} ∑_{jk}[
        M_{ij}·G^<_{jk}(E')·M_{kl}·G^>_{li}(E'-E) +
        G^<_{ij}(E')·M_{jk}·G^>_{kl}(E'-E)·M_{li} +
        M_{ij}·G^<_{jl}(E')·M_{lk}·G^>_{ki}(E'-E) +
        M_{ji}·G^<_{il}(E')·M_{lk}·G^>_{kj}(E'-E)
    ]
    
    Expressed as element-wise correlations:
    c[E] = ∫ a(E') b(E'-E) dE' = sum_E' a(E') * b(E'-E) * dE
    For discrete: c[iE] = sum_iE' a[iE'] * b[iE'-iE] * dE
    
    Term 1: (M@G<@M)[i,l](E') ⊗ G>[l,i](E'-E)
    Term 2: (G<@M)[i,l](E') ⊗ (G>@M)[l,i](E'-E)
    Term 3: (M@G<)[i,l](E') ⊗ (M@G>)[l,i](E'-E)
    Term 4: G<[i,l](E') ⊗ (M@G>@M)[l,i](E'-E)
    
    This implementation uses the validated approach from test_piphoton_dense.py
    which matches FFT correlation with machine precision (diff < 1e-15).
    
    Parameters
    ----------
    m_matrix : np.ndarray
        Interaction matrix M with shape (n_orb, n_orb)
    g_lesser : np.ndarray
        Lesser Green's function with shape (n_energy, n_orb, n_orb)
    g_greater : np.ndarray
        Greater Green's function with shape (n_energy, n_orb, n_orb)
    energies : np.ndarray
        Energy grid
    sparsity_mask : np.ndarray, optional
        Not used in this implementation (kept for API compatibility)
    output_mask : np.ndarray, optional
        Boolean mask with shape (n_orb, n_orb) for which (i,l) pairs to compute.
    """
    n_energy = g_lesser.shape[0]
    n_orb = m_matrix.shape[0]
    
    dE = energies[1] - energies[0]
    prefactor = 1j / (2 * np.pi) * dE
    
    print(f"Reference prefactor: {prefactor}")
    print(f"Energy step dE: {dE}")
    
    # Pre-compute intermediate products at all energies
    gl_m = np.einsum('eij,jk->eik', g_lesser, m_matrix)                  # G<@M
    gl_m[:, sparsity_mask == False] = 0.0  # Zero out non-sparse elements if mask provided
    m_gl = np.einsum('ij,ejk->eik', m_matrix, g_lesser)                  # M@G<
    m_gl[:, sparsity_mask == False] = 0.0  # Zero out non-sparse elements if mask provided
    m_gl_m = np.einsum('eik,kl->eil', m_gl, m_matrix)  # M@G<@M
    m_gl_m[:, sparsity_mask == False] = 0.0  # Zero out non-sparse elements if mask provided
    gg_m = np.einsum('eij,jk->eik', g_greater, m_matrix)                 # G>@M
    gg_m[:, sparsity_mask == False] = 0.0  # Zero out non-sparse elements if mask provided
    m_gg = np.einsum('ij,ejk->eik', m_matrix, g_greater)                 # M@G>
    m_gg[:, sparsity_mask == False] = 0.0  # Zero out non-sparse elements if mask provided
    m_gg_m = np.einsum('eik,kl->eil', m_gg, m_matrix)  # M@G>@M
    m_gg_m[:, sparsity_mask == False] = 0.0  # Zero out non-sparse elements if mask provided
    
    print(f"\nIntermediate products:")
    print(f"  M@G< max abs: {np.max(np.abs(m_gl)):.6e}")
    print(f"  M@G<@M max abs: {np.max(np.abs(m_gl_m)):.6e}")
    print(f"  G<@M max abs: {np.max(np.abs(gl_m)):.6e}")
    print(f"  M@G> max abs: {np.max(np.abs(m_gg)):.6e}")
    print(f"  M@G>@M max abs: {np.max(np.abs(m_gg_m)):.6e}")
    print(f"  G>@M max abs: {np.max(np.abs(gg_m)):.6e}")
    
    # Initialize result
    pi_lesser = np.zeros((n_energy, n_orb, n_orb), dtype=complex)
    
    # Direct summation over energy
    # c[iE] = sum_iE' a[iE'] * b[iE'-iE] * dE
    # where b is evaluated at (E'-E), so we need b[iE_diff] with iE_diff = iE' - iE
    print(f"\nComputing polarization via direct summation...")
    
    # Count how many terms contribute
    term_count = 0
    
    for iE in range(n_energy):
        for iE_prime in range(n_energy):
            # Energy difference index: iE_diff = iE' - iE
            iE_diff = iE_prime - iE
            
            # Skip negative energy differences (assume functions are zero outside range)
            if iE_diff < 0:
                continue
            
            if iE_diff < n_energy:
                term_count += 1
                
                # Determine which (i,l) pairs to compute
                if output_mask is not None:
                    pairs = [(i, l) for i in range(n_orb) for l in range(n_orb) if output_mask[i, l]]
                else:
                    pairs = [(i, l) for i in range(n_orb) for l in range(n_orb)]
                
                # Compute all 4 terms for each (i,l) pair
                for i, l in pairs:
                    pi_lesser[iE, i, l] += (
                        m_gl_m[iE_prime, i, l] * g_greater[iE_diff, l, i] +
                        gl_m[iE_prime, i, l] * gg_m[iE_diff, l, i] +
                        m_gl[iE_prime, i, l] * m_gg[iE_diff, l, i] +
                        g_lesser[iE_prime, i, l] * m_gg_m[iE_diff, l, i]
                    )
    
    print(f"Total energy correlation terms: {term_count}")
    print(f"Polarization (before prefactor) max abs: {np.max(np.abs(pi_lesser)):.6e}")
    
    # Apply prefactor
    pi_lesser *= prefactor
    
    print(f"Polarization (after prefactor, before symmetrization) max abs: {np.max(np.abs(pi_lesser)):.6e}")
    
    # Apply symmetrization steps (matching PiPhoton.compute)
    # 1. Enforce spatial anti-Hermitian symmetry: A_ij -> 0.5 * (A_ij - A_ji*)
    for iE in range(n_energy):
        pi_lesser[iE] = 0.5 * (pi_lesser[iE] - pi_lesser[iE].conj().T)
    
    print(f"Polarization (after symmetrization) max abs: {np.max(np.abs(pi_lesser)):.6e}")
    
    # 2. Discard the real part (enforcing that polarization is purely imaginary)
    pi_lesser.real[:] = 0.0
    
    print(f"Polarization (after discarding real part) max abs: {np.max(np.abs(pi_lesser)):.6e}")
    
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
