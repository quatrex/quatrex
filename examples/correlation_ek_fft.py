"""Correlation across energy and k using EKIndexer and DSDBCOO via FFT.

This example shows how to:
 1) Build an E-per-k flattened stack index using EKIndexer
 2) Assemble a toy DSDBCOO whose stack axis corresponds to (k,E)
 3) Extract a scalar field per (k,E) from the sparse structure
 4) Compute a 2D auto-correlation over (k,E) using FFTs

Ragged E-per-k is optionally supported by zero-padding each k-row to the
maximum number of energies before performing FFT along the E dimension.

Note: This is a conceptual demo. For performance, avoid Python loops and
batch operations where possible in production code.
"""

from __future__ import annotations

from qttools import xp
from qttools.datastructures.dsdbcoo import DSDBCOO
from qttools.utils import EKIndexer


def build_demo_dsdbcoo(ek: EKIndexer) -> DSDBCOO:
    """Create a small diagonal DSDBCOO example aligned with EKIndexer.

    Each (k,E) stack slice stores a 4x4 diagonal block with constant value
    f(k,E) = k + E for demonstration.
    """
    block_sizes = xp.asarray([4], dtype=xp.int32)
    M = int(block_sizes.sum())
    rows = xp.arange(M, dtype=xp.int32)
    cols = xp.arange(M, dtype=xp.int32)

    # local stack length (single-rank example)
    data = xp.zeros((ek.N_ke, rows.size), dtype=xp.float32)
    for flat in range(ek.N_ke):
        k_idx, e_idx = ek.from_flat(flat)
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
    return ds


def extract_scalar_field(ds: DSDBCOO, ek: EKIndexer) -> xp.ndarray:
    """Extract a scalar per (k,E) from DSDBCOO.

    For demo, we take the trace of the only block (0,0) per stack slice.
    Returns a flat array of length N_ke.
    """
    vals = xp.zeros(ek.N_ke, dtype=ds.dtype)
    for flat in range(ek.N_ke):
        block = ds.stack[(flat,)].blocks[0, 0]
        arr = xp.asarray(block)
        vals[flat] = xp.trace(arr)
    return vals


def reshape_flat_to_ke(
    arr_flat: xp.ndarray, ek: EKIndexer, pad: bool = True
) -> xp.ndarray:
    """Reshape a flat vector to a (N_k, N_e) or (N_k, max_Ne) array.

    - Uniform energies: reshape to (N_k, N_e) (no padding needed)
    - Ragged energies: if pad=True, zero-pad each k-row to max length
      and return shape (N_k, max_Ne). If pad=False, raises ValueError.
    """
    if ek.uniform:
        return arr_flat.reshape(ek.N_k, ek.N_e)

    if not pad:
        raise ValueError("Ragged energies require padding to form a 2D array.")

    # Determine max energies per k and build padded array
    lengths = xp.asarray(
        [ek.rowptr[i + 1] - ek.rowptr[i] for i in range(ek.N_k)], dtype=xp.int32
    )
    max_ne = int(lengths.max())
    out = xp.zeros((ek.N_k, max_ne), dtype=arr_flat.dtype)
    for k in range(ek.N_k):
        start = int(ek.rowptr[k])
        stop = int(ek.rowptr[k + 1])
        ne = stop - start
        if ne > 0:
            out[k, :ne] = arr_flat[start:stop]
    return out


def autocorrelation_fft2_ke(field_ke: xp.ndarray) -> xp.ndarray:
    """Compute 2D auto-correlation over (k,E) via FFT.

    Returns the real-valued correlation array with the same shape.
    """
    # 2D FFT over both axes
    F = xp.fft.fftn(field_ke, axes=(-2, -1))
    P = xp.abs(F) ** 2
    R = xp.fft.ifftn(P, axes=(-2, -1))
    return R.real


def main():
    # Build an E-per-k mapping
    N_k = 16
    N_e = 64
    k_points = xp.linspace(-1.0, 1.0, N_k)
    energies = xp.linspace(-2.0, 2.0, N_e)
    ek = EKIndexer(k_points=k_points, energies=energies)

    # Build demo DSDBCOO and extract a scalar field per (k,E)
    ds = build_demo_dsdbcoo(ek)
    field_flat = extract_scalar_field(ds, ek)

    # Reshape to (N_k, N_e)
    field_ke = reshape_flat_to_ke(field_flat, ek)

    # Compute 2D auto-correlation via FFT
    corr_ke = autocorrelation_fft2_ke(field_ke)

    # Example outputs
    print("field_ke shape:", tuple(field_ke.shape))
    print("corr_ke shape:", tuple(corr_ke.shape))
    # Show central correlation value and a few neighbors
    kc, ec = field_ke.shape[0] // 2, field_ke.shape[1] // 2
    window = corr_ke[max(0, kc - 1) : kc + 2, max(0, ec - 2) : ec + 3]
    print("Correlation window around center:\n", xp.asarray(window))


if __name__ == "__main__":
    main()
