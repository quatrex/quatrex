"""Minimal illustrative example wiring EKIndexer with a DSDBCOO matrix.

This example is intentionally lightweight and avoids MPI branching logic;
wrap guards (if comm.rank == 0) as needed when integrating into tests.

It demonstrates:
 1. Constructing an E-per-k mapping (uniform energies per k).
 2. Creating dummy COO sparsity (diagonal) per (k,E) stack slice.
 3. Building a DSDBCOO with global_stack_shape=(N_ke,).
 4. Accessing a single (k,E) via the flattened stack index.

NOTE: This example assumes that qttools initialization of communicators
(comm.stack, comm.block) has been performed externally. If running in a
single-rank context, ensure comm.stack.size == 1 and comm.block.size == 1.
"""

from qttools import xp
from qttools.datastructures.dsdbcoo import DSDBCOO
from qttools.utils import EKIndexer


def build_example():
    # Define k-points and uniform energy grid
    k_points = xp.linspace(-1.0, 1.0, 3)  # N_k = 3
    energies = xp.linspace(-2.0, 2.0, 4)  # N_e = 4
    ek = EKIndexer(k_points=k_points, energies=energies)

    # Global matrix size (for illustration) choose block_sizes summing to M
    # Here we create a trivial 4x4 matrix (single block) per (k,E)
    block_sizes = xp.asarray([4], dtype=xp.int32)
    M = int(block_sizes.sum())

    # Build a diagonal COO sparsity pattern with nnz=4 per (k,E)
    rows = xp.arange(M, dtype=xp.int32)
    cols = xp.arange(M, dtype=xp.int32)

    # For each (k,E) we store the same diagonal data but scaled
    # Shape required by DSDBCOO: (local_stack_len, nnz)
    # In a single-rank setting local_stack_len == ek.N_ke
    data = xp.zeros((ek.N_ke, rows.size), dtype=xp.float32)
    for flat in range(ek.N_ke):
        k_idx, e_idx = ek.from_flat(flat)
        # Simple value: k + energy
        val = float(ek.k_at(k_idx) + ek.energy_at(k_idx, e_idx))
        data[flat, :] = val

    ds = DSDBCOO(
        data=data,
        rows=rows,
        cols=cols,
        block_sizes=block_sizes,
        global_stack_shape=(ek.N_ke,),
        return_dense=True,
    )

    # Access the block (only one) for a specific (k,E)
    target_k, target_e = 1, 2
    flat = ek.to_flat(target_k, target_e)
    # Using the stack indexer to target a single stack slice.
    diag_block = ds.stack[(flat,)].blocks[0, 0]

    return ek, ds, diag_block


if __name__ == "__main__":
    ek, ds, block = build_example()
    print("Flattened size N_ke=", ek.N_ke)
    arr = xp.asarray(block)
    print("Example block shape:", arr.shape)
    print("Block contents:\n", arr)
