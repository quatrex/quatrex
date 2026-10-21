# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes an own nested dissection algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import cKDTree

from qttools import NDArray

# TODO: GPU optimizations


@dataclass
class Block:
    # "leaf" or "separator"
    kind: str
    # depth of the tree node that created it
    depth: int
    # original atom indices (in new order)
    atoms: np.ndarray
    # orbital range in the NEW ordering
    orb_start: int = 0
    orb_stop: int = 0


@dataclass
class NDResult:
    # new atom position i <- old atom atom_perm[i]
    atom_perm: np.ndarray
    # new orbital position i <- old orbital orbital_perm[i]
    orbital_perm: np.ndarray
    # old orbital j -> new position orbital_inv[j]
    orbital_inv: np.ndarray
    # post-order: leaves and separators as they appear
    blocks: list[Block]

    @property
    def leaves(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "leaf"]

    @property
    def separators(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "separator"]

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)


def atom_graph_from_matrices(
    matrices: list,
    orbitals_per_atom: NDArray,
) -> sp.csr_matrix:
    """Atom-level graph from the union of the sparsity patterns of
    `matrices`
    (orbital x orbital, ORIGINAL ordering, e.g. all H_r and S_r).

    Parameters
    ----------
    matrices : list of array-like, shape (num_orbitals, num_orbitals)
        The matrices to use for the atom graph.
    orbitals_per_atom : array-like, shape (num_atoms,)
        The number of orbitals per atom.

    Returns
    -------
    G : scipy.sparse.csr_matrix, shape (num_atoms, num_atoms)
        The atom graph, where G[i, j] = 1 if atoms i and j are coupled,
        and 0 otherwise.

    """
    num_atoms, num_orbitals = len(orbitals_per_atom), int(orbitals_per_atom.sum())
    P = sp.csr_matrix(
        (
            np.ones(num_orbitals),
            (
                np.arange(num_orbitals),
                np.repeat(np.arange(num_atoms), orbitals_per_atom),
            ),
        ),
        shape=(num_orbitals, num_atoms),
    )
    M = None
    for m in matrices:
        m = sp.csr_matrix(m.get() if hasattr(m, "get") else m)
        m.eliminate_zeros()
        pat = m.copy()
        pat.data = np.ones(len(pat.data))
        M = pat if M is None else M + pat
    G = (P.T @ M @ P).tocsr()
    G = G + G.T
    G.setdiag(0)
    G.eliminate_zeros()
    G.data[:] = 1.0
    return G


def atom_graph_from_cutoff(
    atom_coords: np.ndarray,
    cutoff: float,
) -> sp.csr_matrix:
    """Atom graph: two atoms are coupled if closer than `cutoff`.

    Parameters
    ----------
    atom_coords : array-like, shape (num_atoms, 3)
        The coordinates of the atoms.
    cutoff : float
        The cutoff distance for the atom graph.

    Returns
    -------
    G : scipy.sparse.csr_matrix, shape (num_atoms, num_atoms)
        The atom graph, where G[i, j] = 1 if atoms i and j are coupled,
        and 0 otherwise.

    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    pairs = cKDTree(atom_coords).query_pairs(cutoff, output_type="ndarray")
    num_atoms = len(atom_coords)
    G = sp.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(num_atoms, num_atoms)
    )
    G = (G + G.T).tocsr()
    G.data[:] = 1.0
    return G


def nested_dissection_orbitals(
    atom_coordinates: NDArray,
    orbitals_per_atom: NDArray,
    max_depth: int,
    matrices: list | None = None,
    cutoff: float | None = None,
    plane_tol: float = 1e-3,
    block_order: str = "geometric",
) -> NDResult:
    """Nested dissection ordering of orbitals.

    Parameters
    ----------
    atom_coordinates : NDArray
        The coordinates of the atoms.
    orbitals_per_atom : NDArray
        The number of orbitals per atom.
    max_depth : int
        The maximum depth of the nested dissection tree.
    matrices : list of array-like, optional
        The matrices to use for the atom graph. If not
        provided, `cutoff` must be provided.
    cutoff : float, optional
        The cutoff distance for the atom graph. If not provided,
        `matrices` must be provided.
    plane_tol : float, optional
        The tolerance for grouping atoms into planes. Default is 1e-3.
    block_order : str, optional
        The ordering of atoms inside each block. Can be "geometric",
        "rcm", or "none". Default is "geometric".

    Returns
    -------
    NDResult
        The result of the nested dissection ordering, including the new
        atom and orbital permutations, and the list of blocks (leaves
        and separators).

    """
    atom_coordinates = np.asarray(
        (
            atom_coordinates.get()
            if hasattr(atom_coordinates, "get")
            else atom_coordinates
        ),
        dtype=float,
    )
    orbitals_per_atom = np.asarray(orbitals_per_atom, dtype=int)
    num_atoms = len(atom_coordinates)
    if len(orbitals_per_atom) != num_atoms:
        raise ValueError(
            f"orbitals_per_atom has {len(orbitals_per_atom)} entries, expected {num_atoms}"
        )
    num_orbitals = int(orbitals_per_atom.sum())

    if matrices is not None and cutoff is not None:
        raise ValueError("Pass either `matrices` or `cutoff`, not both.")

    if matrices is not None:
        G = atom_graph_from_matrices(matrices, orbitals_per_atom)
    elif cutoff is not None:
        G = atom_graph_from_cutoff(atom_coordinates, cutoff)
    else:
        raise ValueError(
            "Pass either `matrices` or `cutoff` to define the coupling graph."
        )

    # Global transport axis (longest extent of the whole device), used for
    # ordering atoms INSIDE every block so that all blocks are consistent.
    g_axis = int(np.argmax(atom_coordinates.max(axis=0) - atom_coordinates.min(axis=0)))
    g_a, g_b = [i for i in range(3) if i != g_axis]

    def _order_block(atoms):
        """Order atoms inside a leaf/separator so that nearby atoms are
        adjacent -> banded local structure -> large supernodes."""
        atoms = np.asarray(atoms)
        if len(atoms) <= 2 or block_order == "none":
            return np.sort(atoms)

        if block_order == "rcm":
            sub = G[atoms][:, atoms].tocsr()
            if sub.nnz > 0:
                # start from a geometric order so RCM ties are deterministic
                perm = reverse_cuthill_mckee(sub, symmetric_mode=True)
                return atoms[perm]
            # no internal edges: fall through to geometric

        pts = atom_coordinates[atoms]
        key = np.round(pts[:, g_axis] / plane_tol).astype(np.int64)
        # primary: transport plane, then the two in-plane coordinates
        order = np.lexsort((pts[:, g_b], pts[:, g_a], key))
        return atoms[order]

    def _split(atoms):
        pts = atom_coordinates[atoms]
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        a, b = [i for i in range(3) if i != axis]
        key = np.round(pts[:, axis] / plane_tol).astype(np.int64)
        order = np.lexsort((pts[:, b], pts[:, a], key))
        s, key = atoms[order], key[order]
        cum = np.cumsum(orbitals_per_atom[s])
        bounds = np.nonzero(np.diff(key))[0] + 1
        if len(bounds) == 0:
            bounds = np.arange(1, len(s))
        k = int(bounds[np.argmin(np.abs(cum[bounds - 1] - cum[-1] / 2.0))])
        return s[:k], s[k:]

    def _separate(left, right):
        in_left = np.zeros(num_atoms)
        in_left[left] = 1.0
        in_right = np.zeros(num_atoms)
        in_right[right] = 1.0
        left_touch = (G[left] @ in_right) > 0
        right_touch = (G[right] @ in_left) > 0
        if left_touch.sum() <= right_touch.sum():
            return left[~left_touch], right, left[left_touch]
        return left, right[~right_touch], right[right_touch]

    blocks: list[Block] = []

    def _recurse(atoms, depth):
        if depth < max_depth and len(atoms) >= 2:
            left, right = _split(atoms)
            left, right, sep = _separate(left, right)
            if len(left) > 0 and len(right) > 0:
                _recurse(left, depth + 1)
                _recurse(right, depth + 1)
                if len(sep) > 0:
                    blocks.append(Block("separator", depth, _order_block(sep)))
                return
        blocks.append(Block("leaf", depth, _order_block(atoms)))

    _recurse(np.arange(num_atoms), 0)

    atom_perm = np.concatenate([b.atoms for b in blocks])
    orb_start = np.concatenate([[0], np.cumsum(orbitals_per_atom)[:-1]])
    orbital_perm = np.concatenate(
        [
            np.arange(orb_start[a], orb_start[a] + orbitals_per_atom[a])
            for a in atom_perm
        ]
    )
    orbital_inv = np.empty_like(orbital_perm)
    orbital_inv[orbital_perm] = np.arange(num_orbitals)

    pos = 0
    for b in blocks:
        b.orb_start = pos
        pos += int(orbitals_per_atom[b.atoms].sum())
        b.orb_stop = pos

    assert np.array_equal(np.sort(atom_perm), np.arange(num_atoms))
    assert pos == num_orbitals
    return NDResult(atom_perm, orbital_perm, orbital_inv, blocks)
