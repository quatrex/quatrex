"""quatrex inputs for a 5-AGNR n-i-n double-gate FET (Schrodinger-Poisson example).

p_z tight binding of a hydrogen-passivated 5-AGNR (Son, Cohen, Louie, PRL 97,
216803 (2006)): nearest-neighbour hopping t = -2.7 eV, the C-C dimers on the two
edges strengthened by 12 % (t_edge = 1.12 t), which opens the 3p+2 gap. One
orbital per C, orthogonal basis (no overlap.h5), H atoms not included.

Transport along x (armchair direction, period a = 3 a_CC); the device is
NCELL unit cells long, centred in a box of LY x LZ (vacuum).

The bare Coulomb matrix (for GW) uses the Ohno interpolation between the on-site
repulsion U of a carbon p_z orbital and the point-charge limit e^2 / (4 pi eps0 d):
V(d) = U / sqrt(1 + (4 pi eps0 U d / e^2)^2). It is not screened: the permittivity of
the surroundings is `coulomb_screening.epsilon_r` in the quatrex config.

Writes inputs/hamiltonian.h5 and inputs/coulomb_matrix.h5 (key [0,0,0] = full
device), inputs/structure.xyz and inputs/summary.json (unit-cell bands, gap,
mid-gap, Coulomb parameters).
"""
import json
from pathlib import Path

import ase
import ase.io
import numpy as np
from qttools.utils.hdf5_utils import save_hdf5_dict
from scipy import sparse
from scipy.constants import e as Q_E
from scipy.constants import epsilon_0

ACC = 1.42           # C-C bond (A)
T = -2.7             # NN hopping (eV)
DELTA = 0.12         # edge-dimer hopping correction
NW = 5               # dimer lines (5-AGNR)
NCELL = 40           # unit cells along x
LY, LZ = 20.0, 20.0  # box (A)
U = 11.26            # on-site Coulomb repulsion of C p_z (eV), Ohno / PPP value
E2 = Q_E / (4 * np.pi * epsilon_0) * 1e10  # e^2 / (4 pi eps0) in eV A

A = 3 * ACC
DY = np.sqrt(3) / 2 * ACC


def unit_cell():
    """10 C of the 5-AGNR cell, x in [0, A), dimer lines along y."""
    pos = []
    for j in range(NW):
        x0 = 0.5 * ACC if j % 2 == 0 else 0.0
        pos += [[x0, j * DY], [x0 + (ACC if j % 2 == 0 else 2 * ACC), j * DY]]
    return np.array(pos)


def hopping(pos_i, pos_j):
    """NN hoppings between two sets of sites (2D)."""
    d = np.linalg.norm(pos_i[:, None] - pos_j[None], axis=-1)
    h = np.where(np.abs(d - ACC) < 0.05, T, 0.0)
    # edge dimers: both sites on the same outermost line, bond along x
    yi, yj = pos_i[:, 1][:, None], pos_j[:, 1][None]
    edge = (np.abs(yi - yj) < 1e-6) & ((np.abs(yi) < 1e-6) | (np.abs(yi - (NW - 1) * DY) < 1e-6))
    return np.where(edge & (h != 0), T * (1 + DELTA), h)


def coulomb(pos_i, pos_j):
    """Bare Coulomb matrix elements V_ij (eV), Ohno interpolation."""
    d = np.linalg.norm(pos_i[:, None] - pos_j[None], axis=-1)
    return U / np.sqrt(1 + (U * d / E2) ** 2)


def main():
    out = Path(__file__).resolve().parent / "inputs"
    out.mkdir(exist_ok=True)
    uc = unit_cell()
    h00 = hopping(uc, uc)
    h01 = hopping(uc, uc + [A, 0])

    # bands of the unit cell
    k = np.linspace(-np.pi, np.pi, 401)
    e = np.array([np.linalg.eigvalsh(h00 + h01 * np.exp(1j * q) + h01.T * np.exp(-1j * q)) for q in k])
    vb, cb = e[e < 0].max(), e[e > 0].min()
    nc = len(uc)

    # device: block tridiagonal, NCELL cells
    H = sparse.kron(sparse.eye(NCELL), h00) + sparse.kron(sparse.eye(NCELL, k=1), h01) \
        + sparse.kron(sparse.eye(NCELL, k=-1), h01.T)
    H = sparse.csr_matrix(H, dtype=complex)
    save_hdf5_dict(str(out / "hamiltonian.h5"), {"[0,0,0]": H})

    y0, z0 = (LY - (NW - 1) * DY) / 2, LZ / 2
    xyz = np.concatenate([np.c_[uc[:, 0] + n * A, uc[:, 1] + y0, np.full(nc, z0)] for n in range(NCELL)])

    # bare Coulomb matrix of the full device; quatrex truncates it to
    # |dx| < coulomb_screening.interaction_cutoff and divides by epsilon_r
    V = coulomb(xyz, xyz)
    save_hdf5_dict(str(out / "coulomb_matrix.h5"), {"[0,0,0]": sparse.csr_matrix(V, dtype=complex)})

    atoms = ase.Atoms("C" * len(xyz), positions=xyz, cell=[NCELL * A, LY, LZ], pbc=False)
    ase.io.write(out / "structure.xyz", atoms, format="extxyz")

    np.savez(out / "bands.npz", k=k, e=e)
    summary = dict(a=A, ncell=NCELL, norb=len(xyz), length=NCELL * A, box=[NCELL * A, LY, LZ],
                   y_ribbon=y0 + (NW - 1) * DY / 2, z_ribbon=z0, xmin=float(xyz[:, 0].min()),
                   vbm=float(vb), cbm=float(cb), gap=float(cb - vb), mid_gap=float((cb + vb) / 2),
                   t=T, delta=DELTA, u=U, v_nn=float(coulomb(np.zeros((1, 2)), np.array([[ACC, 0.0]]))[0, 0]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
