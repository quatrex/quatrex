# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import numpy as np
import pytest
from pathlib import Path
from scipy import sparse as scipy_sparse

from qttools import xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse, DSDBCSR

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.photon.polarization import PiPhoton


@pytest.fixture
def artificial_system():
    """Create artificial system data for testing."""
    # System parameters
    num_orbitals = 20
    num_blocks = 4
    orbitals_per_block = num_orbitals // num_blocks
    num_energies = 50
    
    # Energy grid
    energy_min = -1.0
    energy_max = 1.0
    energies = np.linspace(energy_min, energy_max, num_energies)
    
    # Block sizes
    block_sizes = [orbitals_per_block] * num_blocks
    
    # Create artificial Hamiltonian (tridiagonal block structure)
    hamiltonian = scipy_sparse.lil_matrix((num_orbitals, num_orbitals), dtype=complex)
    
    # Fill diagonal blocks
    for i in range(num_blocks):
        start = i * orbitals_per_block
        end = (i + 1) * orbitals_per_block
        # On-site energies
        for j in range(start, end):
            hamiltonian[j, j] = -0.5 + 0.1j * np.random.randn()
        # Nearest-neighbor hopping within block
        for j in range(start, end - 1):
            t = -0.3 + 0.05j * np.random.randn()
            hamiltonian[j, j + 1] = t
            hamiltonian[j + 1, j] = np.conj(t)
    
    # Off-diagonal blocks (inter-block coupling)
    for i in range(num_blocks - 1):
        start_i = i * orbitals_per_block
        end_i = (i + 1) * orbitals_per_block
        start_j = (i + 1) * orbitals_per_block
        end_j = (i + 2) * orbitals_per_block
        
        # Couple last orbital of block i to first orbital of block i+1
        t = -0.2 + 0.03j * np.random.randn()
        hamiltonian[end_i - 1, start_j] = t
        hamiltonian[start_j, end_i - 1] = np.conj(t)
    
    hamiltonian = hamiltonian.tocoo()
    
    # Orbital positions (1D chain for simplicity)
    orbital_positions = np.zeros((num_orbitals, 3))
    orbital_positions[:, 0] = np.arange(num_orbitals) * 2.5  # Angstroms
    
    return {
        "num_orbitals": num_orbitals,
        "num_blocks": num_blocks,
        "block_sizes": block_sizes,
        "num_energies": num_energies,
        "energies": energies,
        "hamiltonian": hamiltonian,
        "orbital_positions": orbital_positions,
    }


@pytest.fixture
def mock_config(tmp_path, artificial_system):
    """Create mock configurations."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    
    # Save artificial data
    np.save(input_dir / "grid.npy", artificial_system["orbital_positions"])
    
    hamiltonian_coo = artificial_system["hamiltonian"]
    np.savez(
        input_dir / "hamiltonian.npz",
        data=hamiltonian_coo.data,
        row=hamiltonian_coo.row,
        col=hamiltonian_coo.col,
        shape=hamiltonian_coo.shape,
    )
    
    # Create mock QuatrexConfig
    class MockPhotonConfig:
        model = "pseudo-scattering"
        photon_energy = 0.1  # eV
        polarization = [1.0, 0.0, 0.0]  # x-direction
        light_intensity = 1e10  # W/m^2
    
    class MockSCBAConfig:
        symmetric = False
    
    class MockQuatrexConfig:
        def __init__(self, input_path):
            self.input_dir = input_path
            self.photon = MockPhotonConfig()
            self.scba = MockSCBAConfig()
    
    # Create mock ComputeConfig
    class MockConvolveConfig:
        batch_size = None
    
    class MockComputeConfig:
        dsdbsparse_type = DSDBCSR
        convolve = MockConvolveConfig()
    
    quatrex_config = MockQuatrexConfig(input_dir)
    compute_config = MockComputeConfig()
    
    return quatrex_config, compute_config, artificial_system


def create_artificial_greens_functions(artificial_system):
    """Create artificial Green's functions for testing."""
    num_energies = artificial_system["num_energies"]
    num_orbitals = artificial_system["num_orbitals"]
    block_sizes = artificial_system["block_sizes"]
    
    # Create simple model Green's functions
    # G^< and G^> should satisfy physical properties
    
    # Initialize data arrays
    # Shape: (num_energies, num_orbitals, num_orbitals)
    g_lesser_data = np.zeros((num_energies, num_orbitals, num_orbitals), dtype=complex)
    g_greater_data = np.zeros((num_energies, num_orbitals, num_orbitals), dtype=complex)
    
    # Fill with artificial but physically reasonable values
    for ie in range(num_energies):
        # Diagonal dominant structure
        for i in range(num_orbitals):
            # G^< is negative semidefinite (for fermions)
            g_lesser_data[ie, i, i] = -0.5j * np.exp(-((ie - num_energies//2)**2) / (num_energies/4)**2)
            # G^> is positive semidefinite (for fermions)
            g_greater_data[ie, i, i] = 0.5j * np.exp(-((ie - num_energies//2)**2) / (num_energies/4)**2)
            
            # Small off-diagonal elements
            if i < num_orbitals - 1:
                g_lesser_data[ie, i, i+1] = -0.1j * np.random.randn()
                g_lesser_data[ie, i+1, i] = g_lesser_data[ie, i, i+1].conj()
                
                g_greater_data[ie, i, i+1] = 0.1j * np.random.randn()
                g_greater_data[ie, i+1, i] = g_greater_data[ie, i, i+1].conj()
    
    return g_lesser_data, g_greater_data, block_sizes


class TestPiPhoton:
    """Test suite for photon polarization computation."""
    
    def test_initialization(self, mock_config, artificial_system):
        """Test PiPhoton initialization."""
        quatrex_config, compute_config, system = mock_config
        
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        assert pi_photon.num_photon_energies == system["num_energies"]
        assert pi_photon.num_electron_energies == system["num_energies"]
        assert pi_photon.interaction_matrix is not None
        
    def test_interaction_matrix_properties(self, mock_config, artificial_system):
        """Test properties of the interaction matrix."""
        quatrex_config, compute_config, system = mock_config
        
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        # Interaction matrix should have same sparsity pattern as Hamiltonian
        assert pi_photon.interaction_matrix.num_blocks == system["num_blocks"]
        
    def test_polarization_computation(self, mock_config, artificial_system):
        """Test the polarization computation with artificial Green's functions."""
        quatrex_config, compute_config, system = mock_config
        
        # Create PiPhoton instance
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        # Create artificial Green's functions
        g_lesser_data, g_greater_data, block_sizes = create_artificial_greens_functions(
            artificial_system
        )
        
        # Convert to DSDBSparse format
        g_lesser = DSDBCSR.from_dense_stack(
            xp.array(g_lesser_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        g_greater = DSDBCSR.from_dense_stack(
            xp.array(g_greater_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        # Create output matrices
        pi_lesser = DSDBCSR.zeros_like(g_lesser)
        pi_greater = DSDBCSR.zeros_like(g_greater)
        pi_retarded = DSDBCSR.zeros_like(g_lesser)
        
        # Compute polarization
        pi_photon.compute(
            g_lesser=g_lesser,
            g_greater=g_greater,
            out=(pi_lesser, pi_greater, pi_retarded),
        )
        
        # Basic sanity checks
        assert pi_lesser.data is not None
        assert pi_greater.data is not None
        assert pi_retarded.data is not None
        assert not np.all(pi_lesser.data == 0)
        assert not np.all(pi_greater.data == 0)
        
    def test_energy_symmetry(self, mock_config, artificial_system):
        """Test that π^>(E) = -π^<(-E)† is satisfied."""
        quatrex_config, compute_config, system = mock_config
        
        # Create PiPhoton instance
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        # Create artificial Green's functions
        g_lesser_data, g_greater_data, block_sizes = create_artificial_greens_functions(
            artificial_system
        )
        
        # Convert to DSDBSparse format
        g_lesser = DSDBCSR.from_dense_stack(
            xp.array(g_lesser_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        g_greater = DSDBCSR.from_dense_stack(
            xp.array(g_greater_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        # Create output matrices
        pi_lesser = DSDBCSR.zeros_like(g_lesser)
        pi_greater = DSDBCSR.zeros_like(g_greater)
        pi_retarded = DSDBCSR.zeros_like(g_lesser)
        
        # Compute polarization
        pi_photon.compute(
            g_lesser=g_lesser,
            g_greater=g_greater,
            out=(pi_lesser, pi_greater, pi_retarded),
        )
        
        # Test energy symmetry: π^>(E) = -π^<(-E)†
        pi_greater_check = -pi_lesser.data[::-1].conj()
        
        # Should be very close due to symmetrization in compute()
        np.testing.assert_allclose(
            pi_greater.data,
            pi_greater_check,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Energy symmetry π^>(E) = -π^<(-E)† not satisfied"
        )
        
    def test_spatial_antihermitian_symmetry(self, mock_config, artificial_system):
        """Test that π is anti-Hermitian in spatial indices."""
        quatrex_config, compute_config, system = mock_config
        
        # Create PiPhoton instance
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        # Create artificial Green's functions
        g_lesser_data, g_greater_data, block_sizes = create_artificial_greens_functions(
            artificial_system
        )
        
        # Convert to DSDBSparse format
        g_lesser = DSDBCSR.from_dense_stack(
            xp.array(g_lesser_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        g_greater = DSDBCSR.from_dense_stack(
            xp.array(g_greater_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        # Create output matrices
        pi_lesser = DSDBCSR.zeros_like(g_lesser)
        pi_greater = DSDBCSR.zeros_like(g_greater)
        pi_retarded = DSDBCSR.zeros_like(g_lesser)
        
        # Compute polarization
        pi_photon.compute(
            g_lesser=g_lesser,
            g_greater=g_greater,
            out=(pi_lesser, pi_greater, pi_retarded),
        )
        
        # Convert to dense for easy checking
        pi_l_dense = pi_lesser.to_dense_stack()
        
        # Check anti-Hermitian property for each energy
        for ie in range(system["num_energies"]):
            pi_mat = pi_l_dense[ie]
            # π_ij = -π_ji*
            np.testing.assert_allclose(
                pi_mat,
                -pi_mat.T.conj(),
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Spatial anti-Hermitian symmetry not satisfied at energy {ie}"
            )
            
    def test_purely_imaginary(self, mock_config, artificial_system):
        """Test that polarization is purely imaginary (anti-Hermitian)."""
        quatrex_config, compute_config, system = mock_config
        
        # Create PiPhoton instance
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        # Create artificial Green's functions
        g_lesser_data, g_greater_data, block_sizes = create_artificial_greens_functions(
            artificial_system
        )
        
        # Convert to DSDBSparse format
        g_lesser = DSDBCSR.from_dense_stack(
            xp.array(g_lesser_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        g_greater = DSDBCSR.from_dense_stack(
            xp.array(g_greater_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        # Create output matrices
        pi_lesser = DSDBCSR.zeros_like(g_lesser)
        pi_greater = DSDBCSR.zeros_like(g_greater)
        pi_retarded = DSDBCSR.zeros_like(g_lesser)
        
        # Compute polarization
        pi_photon.compute(
            g_lesser=g_lesser,
            g_greater=g_greater,
            out=(pi_lesser, pi_greater, pi_retarded),
        )
        
        # Check that real part is zero (within numerical precision)
        np.testing.assert_allclose(
            pi_lesser.data.real,
            0.0,
            rtol=1e-10,
            atol=1e-12,
            err_msg="π^< has non-zero real part"
        )
        
        np.testing.assert_allclose(
            pi_greater.data.real,
            0.0,
            rtol=1e-10,
            atol=1e-12,
            err_msg="π^> has non-zero real part"
        )
        
    def test_retarded_relation(self, mock_config, artificial_system):
        """Test that π^r = (π^> - π^<) / 2."""
        quatrex_config, compute_config, system = mock_config
        
        # Create PiPhoton instance
        pi_photon = PiPhoton(
            quatrex_config=quatrex_config,
            compute_config=compute_config,
            photon_energies=system["energies"],
            electron_energies=system["energies"],
        )
        
        # Create artificial Green's functions
        g_lesser_data, g_greater_data, block_sizes = create_artificial_greens_functions(
            artificial_system
        )
        
        # Convert to DSDBSparse format
        g_lesser = DSDBCSR.from_dense_stack(
            xp.array(g_lesser_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        g_greater = DSDBCSR.from_dense_stack(
            xp.array(g_greater_data),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,),
        )
        
        # Create output matrices
        pi_lesser = DSDBCSR.zeros_like(g_lesser)
        pi_greater = DSDBCSR.zeros_like(g_greater)
        pi_retarded = DSDBCSR.zeros_like(g_lesser)
        
        # Compute polarization
        pi_photon.compute(
            g_lesser=g_lesser,
            g_greater=g_greater,
            out=(pi_lesser, pi_greater, pi_retarded),
        )
        
        # Check relation
        pi_r_expected = (pi_greater.data - pi_lesser.data) / 2
        
        np.testing.assert_allclose(
            pi_retarded.data,
            pi_r_expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Retarded polarization relation π^r = (π^> - π^<)/2 not satisfied"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
