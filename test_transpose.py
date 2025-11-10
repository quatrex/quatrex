"""Test the transpose method for DSDBSparse matrices."""

import numpy as np
from qttools import xp
from qttools.comm import comm
from qttools.datastructures import DSDBCSR, DSDBCOO
from scipy import sparse

# Initialize communicator
_default_config = {
    "all_to_all": "host_mpi",
    "all_gather": "host_mpi",
    "all_reduce": "host_mpi",
    "bcast": "host_mpi",
}
comm.configure(
    block_comm_size=1,
    block_comm_config=_default_config,
    stack_comm_config=_default_config,
    override=True,
)

# Create a simple test matrix
n = 6
block_sizes = np.array([2, 2, 2])

# Create a random sparse matrix
np.random.seed(42)
dense = np.random.randn(n, n) + 1j * np.random.randn(n, n)
# Make it sparse
mask = np.random.rand(n, n) > 0.7
dense = dense * mask

# Convert to sparse using xp to ensure compatibility
if xp.__name__ == 'cupy':
    import cupyx.scipy.sparse as cp_sparse
    sparray = cp_sparse.csr_matrix(xp.array(dense))
else:
    sparray = sparse.csr_matrix(dense)

print("Original dense matrix:")
print(dense)
print("\nExpected transpose:")
print(dense.T)

# Test DSDBCSR
print("\n" + "="*60)
print("Testing DSDBCSR")
print("="*60)

dsdbcsr = DSDBCSR.from_sparray(
    sparray,
    block_sizes=block_sizes,
    global_stack_shape=(1,),
    symmetry=False
)

# Create output matrix
dsdbcsr_T = DSDBCSR.zeros_like(dsdbcsr)

# Perform transpose
dsdbcsr.transpose(out=dsdbcsr_T)

# Convert back to dense
result = dsdbcsr_T.to_dense()[0]
if xp.__name__ == 'cupy':
    result = xp.asnumpy(result)
expected = dense.T

print("\nTranspose result:")
print(result)
print("\nDifference:")
print(np.abs(result - expected))
print(f"\nMax error: {np.max(np.abs(result - expected)):.2e}")

if np.allclose(result, expected):
    print("✓ DSDBCSR transpose PASSED")
else:
    print("✗ DSDBCSR transpose FAILED")

# Test DSDBCOO
print("\n" + "="*60)
print("Testing DSDBCOO")
print("="*60)

dsdbcoo = DSDBCOO.from_sparray(
    sparray,
    block_sizes=block_sizes,
    global_stack_shape=(1,),
    symmetry=False
)

# Create output matrix
dsdbcoo_T = DSDBCOO.zeros_like(dsdbcoo)

# Perform transpose
dsdbcoo.transpose(out=dsdbcoo_T)

# Convert back to dense
result_coo = dsdbcoo_T.to_dense()[0]
if xp.__name__ == 'cupy':
    result_coo = xp.asnumpy(result_coo)

print("\nTranspose result:")
print(result_coo)
print("\nDifference:")
print(np.abs(result_coo - expected))
print(f"\nMax error: {np.max(np.abs(result_coo - expected)):.2e}")

if np.allclose(result_coo, expected):
    print("✓ DSDBCOO transpose PASSED")
else:
    print("✗ DSDBCOO transpose FAILED")

# Test that both implementations give the same result
print("\n" + "="*60)
print("Comparing DSDBCSR and DSDBCOO transposes")
print("="*60)
print(f"Max difference: {np.max(np.abs(result - result_coo)):.2e}")
if np.allclose(result, result_coo):
    print("✓ Both implementations agree")
else:
    print("✗ Implementations differ")
