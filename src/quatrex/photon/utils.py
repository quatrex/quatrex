
import numpy as np,math
import cupy as cp
import matplotlib.pyplot as plt
import opt_einsum as oe
#TODO Later: update these imports
from qttools import NDArray, sparse, xp

from quatrex.core.constants import hbar, e, c_0

import scipy.sparse as sp
import math

import warnings

eV_to_J = 1.602176634e-19  # Conversion factor from eV to Joules

def interaction_tensor(distances, hamiltonian):
    """
    Calculate the coupling matrices Mx, My, Mz from the Hamiltonian and position differences.
    
    Parameters:
    H_coo (scipy.sparse.coo_matrix): Sparse Hamiltonian matrix in COO format.
    distances (xp.ndarray): Array of shape (N, N, 3) representing position differences between orbitals.
    
    Returns:M    """  
    prefactor = (- e / 2.0) * (1j / hbar* eV_to_J)  # in SI units (C * s / J)

    M = prefactor * hamiltonian.toarray()[..., np.newaxis] * distances #CPU

    # M = xp.ndarray(comp.astype(complex), dtype=complex)
    
    return M

def D0_tensor(distances, photon_energies):
    
    """
    3D tensor D0[m, i, j] of Initial Photon Green's functions between each pair (n,m) of positions
    R_positions : (N,3) array of positions of orbitals
    E_eV_array : (M,) array of energies in eV ((invented))
    """
    omega = photon_energies / (hbar* eV_to_J)  # angular frequencies in rad/s  
    k = omega / c_0

    r = xp.linalg.norm(distances, axis=2)  # (N, N)
    r_norm = r.copy()
    xp.fill_diagonal(r_norm, xp.inf)

    # # D0 goes through all positions.
    # D0 = xp.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (4 * xp.pi * r_norm[None, :, :])  # (M,N,N)
    
    #trying to debug that one warning!
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D0 = xp.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (4 * xp.pi * r_norm[None, :, :])

    # # Set diagonal to zero (Do we exclude self-interaction here?)
    # for m in range(D0.shape[0]):
    #     xp.fill_diagonal(D0[m], 0.0)

    return D0 # shape (Nw,N,N)

def D0_delta_product(distances, photon_energies):
    
    D0 = D0_tensor(distances, photon_energies)
    delta_transverse = delta_perp_sparse(distances)
    
    num_orbital = distances.shape[0]

    # stack into dense tensor Delta[i,j,u,v]
    Delta = xp.empty((num_orbital, num_orbital, 3, 3), dtype=float)           # or complex if needed
    for u in range(3):
        for v in range(3):
            D_t= delta_transverse[(u, v)]
            Delta[:, :, u, v] = D_t.toarray() if sp.issparse(D_t) else xp.asarray(D_t)

    out = oe.contract('wij,jkuv->wikuv', D0, Delta)

    return out    # (Nw, N, N, 3, 3)

def delta_perp_sparse(distances, sigma=1e-10, tol=0.0):
    """
    Retourne un dict {(i,j): csr_matrix (N,N)} pour δ^⊥_{ij}.
    distances : (N,N,3) array of distances vectors between orbitals
    sigma : set how large (thin) the guaussian approximation will be for the delta(r) regularization via a Gaussian
    tol: tolerance level: avoid to small values, too close to zero 
    """
    N = distances.shape[0]
    pref = 1.0 / (4.0 * xp.pi)

    # distances
    r_mn_2 = xp.sum(distances**2, axis=2)                   
    r_mn = xp.sqrt(r_mn_2)             
    
    mark = r_mn > 0

    # δ^{(3)}(r) ~ gaussienne 3D
    norm = (2.0* sigma**2 *xp.pi)**(-3/2)
    delta_3D = norm * xp.exp(-r_mn_2/(2.0*sigma**2))  # (N,N)

    # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
    inv_r5 = xp.zeros_like(r_mn)
    inv_r5[mark] = 1.0 / (r_mn[mark]**5)

    delta_csr = {}
    for i in range(3):
        ri = distances[..., i]                               # (N,N)
        for j in range(3):
            rj = distances[..., j]
            delta_ij = 1.0 if i == j else 0.0

            # δ⊥_{ij}} = δ_ab δ^{(3)}(r)  +  pref * [ 3 r_a r_b / r^5  - δ_ab * r^2 / r^5 ]
            delta_transversal = delta_ij * delta_3D \
                   + pref * (3.0 * ri * rj * inv_r5 - delta_ij * r_mn_2 * inv_r5)

            if tol > 0.0:
                keep = xp.abs(delta_transversal) > tol
                rows, cols = xp.nonzero(keep)
                data = delta_transversal[keep]
                delta_csr[(i, j)] = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
            else:
                delta_csr[(i, j)] = sp.csr_matrix(delta_transversal)

    return delta_csr 


####################### Temporary Staff #######################

# Create a Tridiagonal Hamiltonian with exponential Decay
def exponential_decay_hamiltonian(distances, t0_eV=1.0, H_cutoff_factor=4):
    """
    Create a sparse Hamiltonian matrix with exponential decay based on distances between orbitals.
    
    Parameters:
    R (xp.ndarray): Array of shape (N, 3) representing the positions of N orbitals in 3D space.
    t0_eV (float): Hopping prefactor in eV.(INVENTED) large (small) |t| :strong (weak) coupling, broad (narrow) energy bands, delocalized (localized) electrons.
    H_cutoff_factor (float): Factor to determine the cutoff distances for including Hamiltonian entries. include H entries for r <= H_cutoff_factor * H_decay_length / increases the number of kept pairs roughly like the area/volume of the cutoff region
    
    Returns:
    H_csr: The resulting sparse Hamiltonian matrix.
    """
    dmat = xp.linalg.norm(distances, axis=2)                # (N, N) distances
    N = distances.shape[0]
    nonzero_dist = dmat[dmat > 1e-12]                # ignore zeros on diagonal  
    H_decay_length = xp.median(nonzero_dist) if nonzero_dist.size > 0 else 1.0
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
                val = -t0_eV * xp.exp(-r / (H_decay_length + 1e-30))
                if abs(val) > 0.0:
                    rows.append(i); cols.append(j); vals.append(val)

    H_csr = sp.coo_matrix(( xp.array(vals), (xp.array(rows), xp.array(cols)) ), shape=(N, N)).tocsr()
    
    return H_csr
#plot the sparsity pattern of a sparse matrix in CSR format.
def plot_sparsity(sparse_mat, title=None, ax = None, figsize=(3,3), cmap='viridis', s=1):
    
    coo = sparse_mat.tocoo()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))

    sc = ax.scatter(coo.col, coo.row, c=xp.abs(coo.data), s=s, cmap=cmap)
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
    # plt.scatter(coo.col, coo.row, c=xp.abs(coo.data), s=s, cmap=cmap)
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
    im0 = axes[0].imshow(xp.abs(D0[e]), origin='lower', aspect='auto')
    axes[0].set_title(f"Energy slice e={e}  (E={energies[e]:.3f} eV)")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    fig.colorbar(im0, ax=axes[0])

    # 2. row slice (fixed i)
    im1 = axes[1].imshow(xp.imag(D0[:, i, :]), origin='lower',
                         extent=[0, N, energies[0], energies[-1]],
                         aspect='auto')
    axes[1].set_title(f"Row slice i={i}")
    axes[1].set_xlabel("j"); axes[1].set_ylabel("Energy (eV)")
    fig.colorbar(im1, ax=axes[1])

    # 3. column slice (fixed j)
    im2 = axes[2].imshow(xp.abs(D0[:, :, j]), origin='lower',
                         extent=[0, N, energies[0], energies[-1]],
                         aspect='auto')
    axes[2].set_title(f"Column slice j={j}")
    axes[2].set_xlabel("i"); axes[2].set_ylabel("Energy (eV)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


#Input enrgy
def make_grids(E_min, E_max, n_points, photon_energy_min, photon_energy_max, tol=1e-9):
    """
        Create uniform energy and photon energy grids suitable for FFT-based convolution.
        
        Parameters:
        E_min (float): Minimum energy value (eV).
        E_max (float): Maximum energy value (eV).
        n_points (int): Number of points in the energy grid.
        photon_energy_min (float): Minimum photon energy value (eV).
        photon_energy_max (float): Maximum photon energy value (eV).
        tol (float): Tolerance for grid coverage checks.
        
        Returns:
        energy_grid (xp.ndarray): Uniform energy grid (eV).
        photon_energy (xp.ndarray): Uniform photon energy grid (eV).
    """ 

    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    # Energy grid (uniform)
    energy_grid = xp.linspace(E_min, E_max, n_points)
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

    photon_energy = omega_min + domega * xp.arange(n_omega)

    # Sanity checks
    # 1) uniform grids
    dE_all = xp.diff(energy_grid)
    if not xp.allclose(dE_all, dE_all[0], rtol=1e-12, atol=1e-18):
        raise ValueError("energy_grid must be uniformly spaced")

    dω_all = xp.diff(photon_energy)
    if not xp.allclose(dω_all, domega, rtol=1e-12, atol=1e-18):
        raise ValueError("constructed photon_energy is not uniformly spaced as ΔE/ħ")

    # 2) end coverage (we hit or overshoot the requested max within tolerance)
    omega_requested_span = omega_max - omega_min
    omega_built_span     = photon_energy[-1] - photon_energy[0]
    if abs(omega_built_span - omega_requested_span) > (abs(domega) + tol):
        # optional: trim last point if we overshoot too much
        while len(photon_energy) > 2 and photon_energy[-1] - omega_max > domega * 0.51:
            photon_energy = photon_energy[:-1]

    return energy_grid, photon_energy
