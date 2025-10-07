import numpy as np, math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, eye, lil_matrix, save_npz, load_npz
from scipy import sparse
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import math

# import scipy.constants import physical_constants
import warnings
import scipy.constants as const

grid = np.load("grid.npy", mmap_mode="r")

# load orbital positions
R = np.load("grid.npy")  # shape: (N, 3) for N orbitals?, units unknown? Angstrom
N = R.shape[0]
# distance between orbitals
dR = R[:, np.newaxis, :] - R[np.newaxis, :, :]  # (N, N, 3)

# invented energy grid (eV)
E_eV = np.linspace(0.1, 4, 20)  # np.ndarray (eV)
np.save("invented_energy_eV.npy", E_eV)

# physical constants
e_charge = 1.602176634e-19  # Coulomb
hbar = 1.054571817e-34  # J*s
eV_to_J = 1.602176634e-19
c = 299792458.0 * 1e10  # m/s A/s tocorrect

# get dimension information
print("Grid Dimensions:", grid.shape)


# plot the sparsity pattern of a sparse matrix in CSR format.
def plot_sparsity(sparse_mat, title=None, ax=None, figsize=(3, 3), cmap="viridis", s=1):

    coo = sparse_mat.tocoo()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    sc = ax.scatter(coo.col, coo.row, c=np.abs(coo.data), s=s, cmap=cmap)
    ax.invert_yaxis()  # rows read top→bottom
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)

    # add colorbar only if standalone
    if ax is plt.gca():
        plt.colorbar(
            sc, ax=ax, label="|value|"
        )  # maybe put ax = 0 or something here ? to avoid multiple colorbars and the crash of them

    return ax

    # plt.figure(figsize=figsize)
    # plt.scatter(coo.col, coo.row, c=np.abs(coo.data), s=s, cmap=cmap)
    # plt.gca().invert_yaxis() #rows read top→bottom
    # plt.xlabel("col")
    # plt.ylabel("row")
    # plt.colorbar(label="|value|")
    # plt.show()
    # ax.matshow(sparse_mat.toarray, cmap=cmap)


# Create a Tridiagonal Hamiltonian with exponential Decay


def exponential_decay_hamiltonian(dR, t0_eV=1.0, H_cutoff_factor=4):
    """
    Create a sparse Hamiltonian matrix with exponential decay based on distances between orbitals.

    Parameters:
    R (np.ndarray): Array of shape (N, 3) representing the positions of N orbitals in 3D space.
    t0_eV (float): Hopping prefactor in eV.(INVENTED) large (small) |t| :strong (weak) coupling, broad (narrow) energy bands, delocalized (localized) electrons.
    H_cutoff_factor (float): Factor to determine the cutoff distance for including Hamiltonian entries. include H entries for r <= H_cutoff_factor * H_decay_length / increases the number of kept pairs roughly like the area/volume of the cutoff region

    Returns:
    H_csr: The resulting sparse Hamiltonian matrix.
    """
    dmat = np.linalg.norm(dR, axis=2)  # (N, N) distances

    nonzero_dist = dmat[dmat > 1e-12]  # ignore zeros on diagonal
    H_decay_length = np.median(nonzero_dist) if nonzero_dist.size > 0 else 1.0
    H_cutoff = H_cutoff_factor * H_decay_length

    # Build sparse H
    rows = []
    cols = []
    vals = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = dmat[i, j]
            if r <= H_cutoff:
                val = -t0_eV * math.exp(-r / (H_decay_length + 1e-30))
                if abs(val) > 0.0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(val)

    H_csr = sp.coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))), shape=(N, N)
    ).tocsr()

    return H_csr


H_exp_decay = exponential_decay_hamiltonian(dR, t0_eV=1.0, H_cutoff_factor=4)
print("H sparse: nnz =", H_exp_decay.nnz)
plot_sparsity(H_exp_decay, title="H sparsity (values colored by |val|)")


H_coo = H_exp_decay.tocoo()


# TODO check the new version. Dense array checkout   . Sparse tensor umschreiben py.


def coupling_matrices(H_coo, dR):
    """
    Calculate the coupling matrices Mx, My, Mz from the Hamiltonian and position differences.

    Parameters:
    H_coo (scipy.sparse.coo_matrix): Sparse Hamiltonian matrix in COO format.
    dR (np.ndarray): Array of shape (N, N, 3) representing position differences between orbitals.

    Returns:
    Mx_csr, My_csr, Mz_csr: Coupling matrices in CSR format.
    """

    prefactor = (-e_charge / 2.0) * (1j / hbar)

    # indices = np.array([[i, j] for i, j in zip(H_coo.row, H_coo.col)], dtype=np.int64)
    # values = H_coo.data.astype(np.complex64)
    # shape = (N, N)
    # M = tf.Sparsetensor(indices = indices , values = values, dense_shape = shape)
    # M_dense = tf.sparse.to_dense(M)

    data_H = H_coo.data
    rows_H = H_coo.row
    cols_H = H_coo.col

    vecs = dR[rows_H, cols_H]  # vecs[i] == R[rows_H[i]] - R[cols_H[i]]

    # Calculate the coupling matrix elements
    compx = prefactor * vecs[:, 0] * data_H
    compy = prefactor * vecs[:, 1] * data_H
    compz = prefactor * vecs[:, 2] * data_H

    # rows_indices
    # col_indices
    values = prefactor * vecs * data_H
    M = torch.tensor(rows_indices, cols_indices, values, layout=torch.sparse_csr)

    Mx_csr = sp.coo_matrix(
        (compx.astype(complex), (rows_H, cols_H)), shape=(N, N)
    ).tocsr()
    My_csr = sp.coo_matrix(
        (compy.astype(complex), (rows_H, cols_H)), shape=(N, N)
    ).tocsr()
    Mz_csr = sp.coo_matrix(
        (compz.astype(complex), (rows_H, cols_H)), shape=(N, N)
    ).tocsr()

    return Mx_csr, My_csr, Mz_csr


Mx_csr, My_csr, Mz_csr = coupling_matrices(H_coo, dR)
print("Mx/My/Mz nnz:", Mx_csr.nnz, My_csr.nnz, Mz_csr.nnz)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
plot_sparsity(Mx_csr, "$M_{mn}^x$", ax=axes[0])
plot_sparsity(My_csr, "$M_{mn}^y$", ax=axes[1])
plot_sparsity(Mz_csr, "$M_{mn}^z$", ax=axes[2])
fig.colorbar(axes[0].collections[0], ax=axes, label="|value|")
plt.tight_layout()
plt.show()


def delta_perp(dR, sigma=1e-10):
    """
    (N,N,3,3) tensor δ^⊥_{ij}(r) between each pair (m,n)
    dR : (N,N,3) array of distance vectors between orbitals
    sigma : set how large (thin) the guaussian approximation will be for the delta(r) regularization via a Gaussian
    """

    pref = 1.0 / (4.0 * math.pi)

    # same as np.linalg.norm(dR, axis=2) but need r^2 later
    r_mn_2 = np.sum(dR**2, axis=2)
    r_mn = np.sqrt(r_mn_2)

    mark = r_mn > 0
    I3 = np.eye(3)

    # δ_ij δ(r) ~ gaussienne 3D
    norm = (2.0 * sigma**2 * math.pi) ** (-3 / 2)
    delta_3D = norm * np.exp(-r_mn_2 / (2.0 * sigma**2))  # (N,N)
    delta_multi = delta_3D[..., None, None] * I3  # (N,N,3,3)

    # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
    # On met à zéro sur la diagonale r=0 (gérée par le terme δ^{(3)})
    rirj_prod = dR[..., :, None] * dR[..., None, :]  # (N,N,3,3)
    r5 = r_mn**5

    Hessian = np.zeros((N, N, 3, 3), dtype=dR.dtype)
    # avoid division per zero : marks only entrie where r nonzero
    Hessian[mark] = (3.0 * rirj_prod[mark] - r_mn_2[mark][..., None, None] * I3) / r5[
        mark
    ][..., None, None]

    return delta_multi + pref * Hessian


delta_tensor = delta_perp(dR)  # Call the function to get the tensor
print("delta_perp tensor shape:", delta_tensor.shape)  # Should be (N, N, 3, 3)


def delta_perp_sparse(dR, sigma=1e-10, tol=0.0):
    """
    Retourne un dict {(i,j): csr_matrix (N,N)} pour δ^⊥_{ij}.
    dR : (N,N,3) array of distance vectors between orbitals
    sigma : set how large (thin) the guaussian approximation will be for the delta(r) regularization via a Gaussian
    tol: tolerance level: avoid to small values, too close to zero
    """
    N = dR.shape[0]
    pref = 1.0 / (4.0 * math.pi)

    # distances
    r_mn_2 = np.sum(dR**2, axis=2)
    r_mn = np.sqrt(r_mn_2)

    mark = r_mn > 0
    I3 = np.eye(3)

    # δ^{(3)}(r) ~ gaussienne 3D
    norm = (2.0 * sigma**2 * math.pi) ** (-3 / 2)
    delta_3D = norm * np.exp(-r_mn_2 / (2.0 * sigma**2))  # (N,N)

    # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
    inv_r5 = np.zeros_like(r_mn)
    inv_r5[mark] = 1.0 / (r_mn[mark] ** 5)

    delta_csr = {}
    for i in range(3):
        ri = dR[..., i]  # (N,N)
        for j in range(3):
            rj = dR[..., j]
            delta_ij = 1.0 if i == j else 0.0

            # δ⊥_{ij}} = δ_ab δ^{(3)}(r)  +  pref * [ 3 r_a r_b / r^5  - δ_ab * r^2 / r^5 ]
            delta_transversal = delta_ij * delta_3D + pref * (
                3.0 * ri * rj * inv_r5 - delta_ij * r_mn_2 * inv_r5
            )

            if tol > 0.0:
                keep = np.abs(delta_transversal) > tol
                rows, cols = np.nonzero(keep)
                data = delta_transversal[keep]
                delta_csr[(i, j)] = sp.coo_matrix(
                    (data, (rows, cols)), shape=(N, N)
                ).tocsr()
            else:
                delta_csr[(i, j)] = sp.csr_matrix(delta_transversal)

    return delta_csr


delta_csr = delta_perp_sparse(dR, sigma=1e-10, tol=1e-12)

plot_sparsity(delta_csr[(0, 0)], title="delta_perp_xx sparsity")


def D0_matrix(R_positions, E_eV_array):
    """
    3D tensor D0[m, i, j] of Initial Photon Green's functions between each pair (n,m) of positions
    R_positions : (N,3) array of positions of orbitals
    E_eV_array : (M,) array of energies in eV ((invented))
    """

    E = (
        np.asarray(E_eV_array, float).ravel() * eV_to_J
    )  # Convert eV to J because copilote forced me to
    omega = E / hbar
    k = omega / c

    R = np.asarray(R_positions, dtype=float) * 1e-10
    N = R.shape[0]
    # Compute all pairwise distances (shape: (N, N))
    dR = R[:, np.newaxis, :] - R[np.newaxis, :, :]  # (N, N, 3)

    r = np.linalg.norm(dR, axis=2)  # (N, N)
    r_norm = r.copy()
    np.fill_diagonal(r_norm, np.inf)

    # D0 goes through all positions.
    D0 = np.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (
        4 * np.pi * r_norm[None, :, :]
    )  # (M,N,N)

    # trying to debug that one warning!
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D0 = np.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (
            4 * np.pi * r_norm[None, :, :]
        )

    # # Set diagonal to zero (Do we exclude self-interaction here?)
    # for m in range(D0.shape[0]):
    #     np.fill_diagonal(D0[m], 0.0)

    return D0


def show_tensor_cuts(D0, energies, e=None, i=None, j=None):
    """
    Visualize 3 orthogonal cuts through the 3D tensor D0[m, i, j].

    e : energy index
    i : fixed row index    #probably useless
    j : fixed column index #probably useless
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    M, N, N = D0.shape
    if e is None:
        e = M // 2
    if i is None:
        i = N // 2
    if j is None:
        j = N // 2

    # 1. energy slice (fixed e)
    im0 = axes[0].imshow(np.abs(D0[e]), origin="lower", aspect="auto")
    axes[0].set_title(f"Energy slice e={e}  (E={energies[e]:.3f} eV)")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    fig.colorbar(im0, ax=axes[0])

    # 2. row slice (fixed i)
    im1 = axes[1].imshow(
        np.imag(D0[:, i, :]),
        origin="lower",
        extent=[0, N, energies[0], energies[-1]],
        aspect="auto",
    )
    axes[1].set_title(f"Row slice i={i}")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("Energy (eV)")
    fig.colorbar(im1, ax=axes[1])

    # 3. column slice (fixed j)
    im2 = axes[2].imshow(
        np.abs(D0[:, :, j]),
        origin="lower",
        extent=[0, N, energies[0], energies[-1]],
        aspect="auto",
    )
    axes[2].set_title(f"Column slice j={j}")
    axes[2].set_xlabel("i")
    axes[2].set_ylabel("Energy (eV)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


D0 = D0_matrix(R, E_eV)
show_tensor_cuts(D0, E_eV, e=1, i=150, j=150)
