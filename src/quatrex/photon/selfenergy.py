import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import opt_einsum as oe
from quatrex.photon.utils import plot_sparsity, delta_perp_sparse, exponential_decay_hamiltonian,interaction_tensor, show_tensor_cuts, D0_matrix
import timeit
import scipy.sparse as sp
from scipy.interpolate import CubicSpline

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSDBSparse

mu0 = 8.854e-12
hbar = 1.054571817e-34             # J*s

def transver_self_energy (energy_grid:NDArray,photon_energy:NDArray, g:DSDBSparse, m_interaction, d:DSDBSparse, pad_factor = 2):
    """
    Compute D with g form energy via einstein sum - Toy exemple
    g:              (N, N) complex
    d:              (N, N) complex
    m_interaction:  (N, N, 3)   real/complex, energy-independent
    mu0:            vacuum permeability

    Returns:
      Σ:            (N, N, 3, 3) complex
    """
    
    dE = np.diff(energy_grid).mean()
    if not np.allclose(np.diff(energy_grid), dE, rtol=1e-6, atol=1e-12):
        raise ValueError("energy_grid should be uniformly spaced for FFT")

    dhw = np.diff(photon_energy).mean()
    if not np.allclose(np.diff(photon_energy),dhw,rtol=1e-6,atol=1e-12):
        raise ValueError("photon_energy should be uniformly spaced for FFT")
    
    if not np.isclose(dhw, dE):
        raise ValueError(f"Mismatch in spacing : Δω={dhw:.3e} vs ΔEs={dE:.3e}")
    
    Ne = energy_grid.size
    Nw = photon_energy.size
    prefactor = 1j * mu0 * (1 / (2*np.pi)) 

    pad_width = ((0, int(pad_factor * max(Nw,Ne)) - Nw), (0, 0), (0, 0))      
    g_pad = np.pad(g, pad_width, mode='constant')
    d_pad = np.pad(d, pad_width, mode='constant')    

    #Inverse FFT: energy/frequency domain to time domain: energy -> tau
    G_IFFT = (np.fft.fft(g_pad, axis=0)).conj()  # (Np, N, N) #TODO: change to the fastest option
    D_IFFT = np.fft.fft(d_pad, axis=0)  # (Np, N, N)

    #Get the term for the polarization 
    T1 = oe.contract("jiu,il,lkv,ikuv->jk",m_interaction,G_IFFT,m_interaction,D_IFFT)
    T2 = oe.contract("iju,il,lkv,iluv->jk",m_interaction,G_IFFT,m_interaction,D_IFFT) #weird
    T3 = oe.contract("iju,il,lkv,lkuv->jk",m_interaction,G_IFFT,m_interaction,D_IFFT) #make sense
    T4 = oe.contract("iju,il,lkv,jluv->jk",m_interaction,G_IFFT,m_interaction,D_IFFT) #weird
    SUM = (T1 + T2 + T3 + T4)

    SUM_FFT = np.fft.ifft(SUM, axis=0) 

    trans_self_energy = prefactor * SUM_FFT

    #TODO: different : energy now
    E_min = energy_grid[0]
    E_max = energy_grid[-1]
    mask = (dE >= E_min) & (dE <= E_max)

    if not np.any(mask):
      raise ValueError("Requested photon range is outside computed frequency grid.")

    # select only those frequencies and corresponding polarization values
    trans_self_energy_selected = trans_self_energy[mask, ...]

    return trans_self_energy_selected #lent