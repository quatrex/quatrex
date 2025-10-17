
import numpy as np, math
import cupy as cp
import matplotlib.pyplot as plt

#TODO Later: update these imports
#from qttools import NDArray, sparse, xp
#from quatrex.core.constants import hbar, e_charge, c, eV_to_J

from scipy.constants import physical_constants
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, eye, save_npz, load_npz
from scipy.spatial.distance import cdist
import math

import warnings

# physical constants
e_charge = 1.602176634e-19         # Coulomb
hbar = 1.054571817e-34             # J*s
eV_to_J = 1.602176634e-19          # J/eV
c = 299792458.0 * 1e10             # A/s 


# Temporary : Create a Tridiagonal Hamiltonian with exponential Decay
def exponential_decay_hamiltonian(distance, t0_eV=1.0, H_cutoff_factor=4):
    """
    Create a sparse Hamiltonian matrix with exponential decay based on distances between orbitals.
    
    Parameters:
    R (np.ndarray): Array of shape (N, 3) representing the positions of N orbitals in 3D space.
    t0_eV (float): Hopping prefactor in eV.(INVENTED) large (small) |t| :strong (weak) coupling, broad (narrow) energy bands, delocalized (localized) electrons.
    H_cutoff_factor (float): Factor to determine the cutoff distance for including Hamiltonian entries. include H entries for r <= H_cutoff_factor * H_decay_length / increases the number of kept pairs roughly like the area/volume of the cutoff region
    
    Returns:
    H_csr: The resulting sparse Hamiltonian matrix.
    """
    dmat = np.linalg.norm(distance, axis=2)                # (N, N) distances
    N = distance.shape[0]
    nonzero_dist = dmat[dmat > 1e-12]                # ignore zeros on diagonal  
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
            r = dmat[i,j]
            if r <= H_cutoff:
                val = -t0_eV * math.exp(-r / (H_decay_length + 1e-30))
                if abs(val) > 0.0:
                    rows.append(i); cols.append(j); vals.append(val)

    H_csr = sp.coo_matrix(( np.array(vals), (np.array(rows), np.array(cols)) ), shape=(N, N)).tocsr()
    
    return H_csr

def interaction_tensor(distance_matrix, hamiltonian):
    """
    Calculate the coupling matrices Mx, My, Mz from the Hamiltonian and position differences.
    
    Parameters:
    H_coo (scipy.sparse.coo_matrix): Sparse Hamiltonian matrix in COO format.
    distance (np.ndarray): Array of shape (N, N, 3) representing position differences between orbitals.
    
    Returns:M    """  
    prefactor = (- e_charge / 2.0) * (1j / hbar)

    M = prefactor * hamiltonian.toarray()[..., np.newaxis] * distance_matrix #CPU
    #M = prefactor * hamiltonian[..., cp.newaxis] * distance_matrix

    # M = np.ndarray(comp.astype(complex), dtype=complex)
    
    return M

def delta_perp_sparse(distance, sigma=1e-10, tol=0.0):
    """
    Retourne un dict {(i,j): csr_matrix (N,N)} pour δ^⊥_{ij}.
    distance : (N,N,3) array of distance vectors between orbitals
    sigma : set how large (thin) the guaussian approximation will be for the delta(r) regularization via a Gaussian
    tol: tolerance level: avoid to small values, too close to zero 
    """
    N = distance.shape[0]
    pref = 1.0 / (4.0 * math.pi)

    # distances
    r_mn_2 = np.sum(distance**2, axis=2)                   
    r_mn = np.sqrt(r_mn_2)             
    
    mark = r_mn > 0
    I3 = np.eye(3)

    # δ^{(3)}(r) ~ gaussienne 3D
    norm = (2.0* sigma**2 *math.pi)**(-3/2)
    delta_3D = norm * np.exp(-r_mn_2/(2.0*sigma**2))  # (N,N)

    # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
    inv_r5 = np.zeros_like(r_mn)
    inv_r5[mark] = 1.0 / (r_mn[mark]**5)

    delta_csr = {}
    for i in range(3):
        ri = distance[..., i]                               # (N,N)
        for j in range(3):
            rj = distance[..., j]
            delta_ij = 1.0 if i == j else 0.0

            # δ⊥_{ij}} = δ_ab δ^{(3)}(r)  +  pref * [ 3 r_a r_b / r^5  - δ_ab * r^2 / r^5 ]
            delta_transversal = delta_ij * delta_3D \
                   + pref * (3.0 * ri * rj * inv_r5 - delta_ij * r_mn_2 * inv_r5)

            if tol > 0.0:
                keep = np.abs(delta_transversal) > tol
                rows, cols = np.nonzero(keep)
                data = delta_transversal[keep]
                delta_csr[(i, j)] = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
            else:
                delta_csr[(i, j)] = sp.csr_matrix(delta_transversal)

    return delta_csr

def D0_matrix(distance, E_eV_array):
    
    """
    3D tensor D0[m, i, j] of Initial Photon Green's functions between each pair (n,m) of positions
    R_positions : (N,3) array of positions of orbitals
    E_eV_array : (M,) array of energies in eV ((invented))
    """
    
    E = np.asarray(E_eV_array, float).ravel() * eV_to_J # Convert eV to J because copilote forced me to
    omega = E / hbar  
    k = omega / c
    N = distance.shape[0]

    r = np.linalg.norm(distance, axis=2)  # (N, N)
    r_norm = r.copy()
    np.fill_diagonal(r_norm, np.inf)

    # D0 goes through all positions.
    D0 = np.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (4 * np.pi * r_norm[None, :, :])  # (M,N,N)
    
    #trying to debug that one warning!
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D0 = np.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (4 * np.pi * r_norm[None, :, :])

    # # Set diagonal to zero (Do we exclude self-interaction here?)
    # for m in range(D0.shape[0]):
    #     np.fill_diagonal(D0[m], 0.0)

    return D0

#plot the sparsity pattern of a sparse matrix in CSR format.
def plot_sparsity(sparse_mat, title=None, ax = None, figsize=(3,3), cmap='viridis', s=1):
    
    coo = sparse_mat.tocoo()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))

    sc = ax.scatter(coo.col, coo.row, c=np.abs(coo.data), s=s, cmap=cmap)
    ax.invert_yaxis()  # rows read top→bottom
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)

    # add colorbar only if standalone
    if ax is plt.gca():
        plt.colorbar(sc, ax=ax, label="|value|")  #maybe put ax = 0 or something here ? to avoid multiple colorbars and the crash of them
    return ax

    # plt.figure(figsize=figsize)
    # plt.scatter(coo.col, coo.row, c=np.abs(coo.data), s=s, cmap=cmap)
    # plt.gca().invert_yaxis() #rows read top→bottom
    # plt.xlabel("col")
    # plt.ylabel("row")
    # plt.colorbar(label="|value|")
    # plt.show()
    #ax.matshow(sparse_mat.toarray, cmap=cmap)
#plot the tensor cuts for D0
def show_tensor_cuts(D0, energies, e=None, i=None, j=None):
    
    """
    Visualize 3 orthogonal cuts through the 3D tensor D0[m, i, j].

    e : energy index
    i : fixed row index    #probably useless
    j : fixed column index #probably useless
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    M, N, N = D0.shape
    if e is None: e = M // 2
    if i is None: i = N // 2
    if j is None: j = N // 2

    # 1. energy slice (fixed e)
    im0 = axes[0].imshow(np.abs(D0[e]), origin='lower', aspect='auto')
    axes[0].set_title(f"Energy slice e={e}  (E={energies[e]:.3f} eV)")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    fig.colorbar(im0, ax=axes[0])

    # 2. row slice (fixed i)
    im1 = axes[1].imshow(np.imag(D0[:, i, :]), origin='lower',
                         extent=[0, N, energies[0], energies[-1]],
                         aspect='auto')
    axes[1].set_title(f"Row slice i={i}")
    axes[1].set_xlabel("j"); axes[1].set_ylabel("Energy (eV)")
    fig.colorbar(im1, ax=axes[1])

    # 3. column slice (fixed j)
    im2 = axes[2].imshow(np.abs(D0[:, :, j]), origin='lower',
                         extent=[0, N, energies[0], energies[-1]],
                         aspect='auto')
    axes[2].set_title(f"Column slice j={j}")
    axes[2].set_xlabel("i"); axes[2].set_ylabel("Energy (eV)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


    import numpy as np

#Input enrgy

def make_grids(E_min, E_max, n_points, photon_energy_min, photon_energy_max, tol=1e-9):
    """
    Build a uniformly spaced energy grid and a photon *angular frequency* grid
    whose spacing matches Δω = ΔE / ħ. The photon range is provided in *energy*.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    # Energy grid (uniform)
    energy_grid = np.linspace(E_min, E_max, n_points)
    dE = (E_max - E_min) / (n_points - 1)

    # Target Δω that must match your FFT convention
    domega = dE

    # Convert requested photon *energy* range to angular frequency range
    omega_min = photon_energy_min 
    omega_max = photon_energy_max 

    # Choose integer count so that spacing is exactly Δω
    n_omega = int(round((omega_max - omega_min) / domega)) + 1
    if n_omega < 2:
        n_omega = 2

    photon_energy = omega_min + domega * np.arange(n_omega)

    # Sanity checks
    # 1) uniform grids
    dE_all = np.diff(energy_grid)
    if not np.allclose(dE_all, dE_all[0], rtol=1e-12, atol=1e-18):
        raise ValueError("energy_grid must be uniformly spaced")

    dω_all = np.diff(photon_energy)
    if not np.allclose(dω_all, domega, rtol=1e-12, atol=1e-18):
        raise ValueError("constructed photon_energy is not uniformly spaced as ΔE/ħ")

    # 2) end coverage (we hit or overshoot the requested max within tolerance)
    omega_requested_span = omega_max - omega_min
    omega_built_span     = photon_energy[-1] - photon_energy[0]
    if abs(omega_built_span - omega_requested_span) > (abs(domega) + tol):
        # optional: trim last point if we overshoot too much
        while len(photon_energy) > 2 and photon_energy[-1] - omega_max > domega * 0.51:
            photon_energy = photon_energy[:-1]

    return energy_grid, photon_energy
