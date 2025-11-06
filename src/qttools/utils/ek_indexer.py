"""
EK indexer utilities for flattening an E-per-k grid to a 1D stack index.

This provides a light-weight mapping for scenarios where for each k-point
you have a (possibly non-uniform, possibly ragged) set of energies.

It is designed to integrate with the DSDBSparse family by using the flattened
index as the first (stack) dimension. With this, you can set
    global_stack_shape = (ek.N_ke,)
and address a specific (k_idx, e_local_idx) via
    flat = ek.to_flat(k_idx, e_local_idx)
    ds.stack[(flat,)]  # substack view for that single (k,E)

Two modes are supported:
- uniform energies across k: provide a single 1D energies array
- ragged energies per k: provide a list/tuple of 1D arrays (one per k)

This module is backend-agnostic with respect to qttools.xp (NumPy/CuPy).

Example
-------
>>> from qttools.utils.ek_indexer import EKIndexer
>>> import numpy as np
>>> k_points = np.linspace(-1.0, 1.0, 3)        # 3 k-points
>>> energies = np.linspace(-2.0, 2.0, 5)        # same 5 energies per k
>>> ek = EKIndexer(k_points=k_points, energies=energies)
>>> ek.N_k, ek.N_e, ek.N_ke
(3, 5, 15)
>>> ek.to_flat(1, 3)     # k=1, E-index=3
8
>>> ek.from_flat(8)
(1, 3)
>>> ek.energy_at(1, 3)
1.0
>>> ek.k_at(1)
0.0

Usage with DSDBCOO (conceptual)
--------------------------------
Assuming you assemble a DSDBCOO where each stack slice corresponds to a
distinct (k,E) pair:

    ek = EKIndexer(k_points, energies)  # or energies_per_k list
    ds = DSDBCOO(
        data=local_data,             # shape: (local_stack_len, ..., nnz)
        rows=rows, cols=cols,
        block_sizes=block_sizes,
        global_stack_shape=(ek.N_ke,),
        return_dense=True,
    )

Then you can select a single (k,E):

    flat = ek.to_flat(k_idx, e_local_idx)
    val = ds.stack[(flat,)][rows, cols]

And for all energies at a fixed k, iterate over the contiguous range:

    start, stop = ek.rowptr[k_idx], ek.rowptr[k_idx+1]
    for flat in range(int(start), int(stop)):
        ... = ds.stack[(flat,)]

Note: The DSDBCOO/DSDBSparse stack distribution only uses the first axis, so
flattening (k,E) to 1D ensures a balanced distribution when using your
existing "balanced" strategy for stack sectioning.
"""

from __future__ import annotations

from typing import Sequence

from qttools import NDArray, xp


class EKIndexer:
    def reshape_flat(self, arr_flat: NDArray, pad: bool = True) -> NDArray:
        """Reshape a flat array to (N_k, N_e) or (N_k, max_Ne) with zero-padding.

        Parameters
        ----------
        arr_flat : NDArray
            Flat array of length N_ke (as produced by EKIndexer flattening).
        pad : bool, optional
            If True and ragged, zero-pad each k-row to max length. If False and ragged, raises ValueError.

        Returns
        -------
        arr_2d : NDArray
            Array of shape (N_k, N_e) (uniform) or (N_k, max_Ne) (ragged).
        """
        arr_flat = xp.asarray(arr_flat)
        if self.uniform:
            if self.N_e is None:
                raise ValueError("Uniform mode requires N_e to be set.")
            return arr_flat.reshape(self.N_k, int(self.N_e))
        if not pad:
            raise ValueError("Ragged energies require padding to form a 2D array.")
        lengths = xp.asarray(
            [self.rowptr[i + 1] - self.rowptr[i] for i in range(self.N_k)],
            dtype=xp.int32,
        )
        max_ne = int(lengths.max())
        out = xp.zeros((self.N_k, max_ne), dtype=arr_flat.dtype)
        for k in range(self.N_k):
            start = int(self.rowptr[k])
            stop = int(self.rowptr[k + 1])
            ne = stop - start
            if ne > 0:
                out[k, :ne] = arr_flat[start:stop]
        return out

    """Mapping utilities for an E-per-k grid flattened to 1D.

    Parameters
    ----------
    k_points : array-like
        Array of k-points of shape (N_k,) or (N_k, d). Only the length (N_k)
        is used for indexing; values are stored for convenience.
    energies : array-like | Sequence[array-like]
        - Uniform mode: a 1D array of shape (N_e,) used for every k.
        - Ragged mode: a list/tuple with one 1D array per k (length N_k),
          allowing a different number of energies per k.
    energy_weights : None | array-like | Sequence[array-like]
        Optional quadrature weights, matching the structure of `energies`.
        If omitted, weights of 1 are assumed.

    Attributes
    ----------
    N_k : int
        Number of k-points.
    N_e : int | None
        Number of energies per k in uniform mode; None in ragged mode.
    N_ke : int
        Total number of (k,E) pairs after flattening.
    uniform : bool
        Whether energies are uniform across all k.
    rowptr : NDArray[int32]
        CSR-like row pointer of shape (N_k+1,), so that the flattened
        indices for a given k are in [rowptr[k], rowptr[k+1]).
    energies_concat : NDArray
        Concatenated energies in ragged mode; in uniform mode, stores the
        single 1D energy array.
    energy_weights_concat : NDArray | None
        Concatenated weights matching `energies_concat` (ragged) or the
        single 1D weight array (uniform), if provided.
    """

    def __init__(
        self,
        *,
        k_points: NDArray,
        energies: NDArray | Sequence[NDArray],
        energy_weights: NDArray | Sequence[NDArray] | None = None,
    ) -> None:
        self.k_points = xp.asarray(k_points)
        self.N_k = int(self.k_points.shape[0])

        # Determine mode and build rowptr + energy storage.
        if _is_sequence_of_arrays(energies):
            # Ragged energies per k
            if len(energies) != self.N_k:
                raise ValueError(
                    "Length of energies list must match number of k-points."
                )
            lengths = xp.asarray(
                [xp.asarray(e).shape[0] for e in energies], dtype=xp.int32
            )
            self.rowptr: NDArray = xp.hstack(
                (xp.array([0], dtype=xp.int32), xp.cumsum(lengths))
            )
            self.energies_concat: NDArray = xp.asarray(
                xp.hstack([xp.asarray(e) for e in energies])
            )
            self.uniform = False
            self.N_e = None

            if energy_weights is None:
                self.energy_weights_concat = None
            else:
                if (
                    not _is_sequence_of_arrays(energy_weights)
                    or len(energy_weights) != self.N_k
                ):
                    raise ValueError(
                        "energy_weights must be a list/tuple with one 1D array per k"
                    )
                self.energy_weights_concat = xp.asarray(
                    xp.hstack([xp.asarray(w) for w in energy_weights])
                )
        else:
            # Uniform energies across k
            e_arr = xp.asarray(energies)
            if e_arr.ndim != 1:
                raise ValueError("Uniform energies must be 1D array")
            self.N_e = int(e_arr.shape[0])
            # rowptr: 0, N_e, 2N_e, ..., N_k*N_e
            self.rowptr = xp.arange(
                0, (self.N_k + 1) * self.N_e, self.N_e, dtype=xp.int32
            )
            self.energies_concat = e_arr
            self.uniform = True

            if energy_weights is None:
                self.energy_weights_concat = None
            else:
                ew_arr = xp.asarray(energy_weights)
                if ew_arr.shape != e_arr.shape:
                    raise ValueError("energy_weights must match shape of energies")
                self.energy_weights_concat = ew_arr

        self.N_ke = int(self.rowptr[-1])

    # ---------------------------- mapping ----------------------------
    def to_flat(self, k_idx: int, e_local_idx: int) -> int:
        """Map (k_idx, e_local_idx) -> flat index in [0, N_ke).

        This is O(1).
        """
        return int(self.rowptr[int(k_idx)] + int(e_local_idx))

    def from_flat(self, flat_idx: int) -> tuple[int, int]:
        """Inverse mapping flat index -> (k_idx, e_local_idx).

        Uses searchsorted over rowptr; O(log N_k).
        """
        flat_idx = int(flat_idx)
        if flat_idx < 0 or flat_idx >= self.N_ke:
            raise IndexError("flat_idx out of range")
        k_idx = int(xp.searchsorted(self.rowptr, flat_idx, side="right") - 1)
        e_local_idx = int(flat_idx - self.rowptr[k_idx])
        return k_idx, e_local_idx

    # ------------------------- data accessors ------------------------
    def k_at(self, k_idx: int):
        """Return the k-point value (array or scalar) at index k_idx."""
        return self.k_points[int(k_idx)]

    def energy_at(self, k_idx: int, e_local_idx: int):
        """Return energy value for given (k_idx, e_local_idx)."""
        if self.uniform:
            return self.energies_concat[int(e_local_idx)]
        base = self.rowptr[int(k_idx)]
        return self.energies_concat[int(base + e_local_idx)]

    def weights_at(self, k_idx: int, e_local_idx: int):
        """Return quadrature weight for (k_idx, e_local_idx) if provided, else 1."""
        if self.energy_weights_concat is None:
            return 1.0
        if self.uniform:
            return self.energy_weights_concat[int(e_local_idx)]
        base = self.rowptr[int(k_idx)]
        return self.energy_weights_concat[int(base + e_local_idx)]

    # --------------------------- helpers -----------------------------
    def range_for_k(self, k_idx: int) -> tuple[int, int]:
        """Return (start, stop) flat index range for a fixed k_idx."""
        k_idx = int(k_idx)
        return int(self.rowptr[k_idx]), int(self.rowptr[k_idx + 1])

    def flat_indices_for_k(self, k_idx: int) -> NDArray:
        """Return flat indices for all energies at fixed k_idx as a 1D array."""
        start, stop = self.range_for_k(k_idx)
        if stop <= start:
            return xp.asarray([], dtype=xp.int32)
        return xp.arange(start, stop, dtype=xp.int32)

    def weights_flat(self) -> NDArray:
        """Return a length-N_ke vector of weights.

        If no weights were provided, returns a vector of ones.
        In uniform mode, the 1D weights are repeated N_k times.
        """
        if self.energy_weights_concat is None:
            return xp.ones(self.N_ke, dtype=xp.result_type(1.0))
        if self.uniform:
            # Repeat weights N_k times
            return xp.tile(self.energy_weights_concat, self.N_k)
        return self.energy_weights_concat


def _is_sequence_of_arrays(x: object) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and not _is_array_like(x)


def _is_array_like(x: object) -> bool:
    try:
        arr = xp.asarray(x)
        return True and arr is not None
    except Exception:
        return False
