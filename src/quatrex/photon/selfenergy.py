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

def transver_self_energy (energy_grid:NDArray,photon_frequency:NDArray, g:DSDBSparse, m_interaction, d:DSDBSparse, pad_factor = 2):
    """
    Compute D with g form energy via einstein sum - Toy exemple
    g:              (N, N) complex
    d:              (N, N) complex
    m_interaction:  (N, N, 3)   real/complex, energy-independent
    mu0:            vacuum permeability

    Returns:
      Σ:            (N, N, 3, 3) complex
    """
    Nw = photon_frequency

    dw = np.diff(photon_frequency).mean

    if not np.allclose(np.diff(photon_grid),dw, rtol = 1e-6,atol = 1e-12):
        raise ValueError("photon frequency grid should be uniformely spaced")
    
    prefactor = 1j * mu0 * (1 / (2*np.pi)) 

    pad_width = ((0, int(pad_factor * Nw) - Nw), (0, 0), (0, 0))      
    g_pad = np.pad(g, pad_width, mode='constant')
    d_pad = np.pad(d, pad_width, mode='constant')    

    #Inverse FFT: energy/frequency domain to time domain: energy -> tau
    G1_IFFT = np.fft.ifft(g_pad, axis=0)  # (Np, N, N) 
    G2_IFFT = np.fft.ifft(d_pad, axis=0)  # (Np, N, N)

    #Get the term for the polarization 
    T1 = oe.contract("jiu,il,lkv,ikuv->jk",m_interaction,g,m_interaction,d)
    T2 = oe.contract("iju,il,lkv,iluv->jk",m_interaction,g,m_interaction,d) #weird
    T3 = oe.contract("iju,il,lkv,lkuv->jk",m_interaction,g,m_interaction,d) #make sense
    T4 = oe.contract("iju,il,lkv,jluv->jk",m_interaction,g,m_interaction,d) #weird
    SUM = (T1 + T2 + T3 + T4)

    SUM_FFT = np.fft.fft(SUM, axis=0) 

    trans_self_energy_tot = prefactor * SUM_FFT

    #TODO: different : energy now
   #get the energy step out
    Np,_,_,_,_ = trans_self_energy_tot.shape
    dtau = 2 * np.pi * hbar / (Np * dE) # 1/(N * ws) sampling rate dtau
    omega = 2 * np.pi * np.fft.fftfreq(Np, d=dtau)  # (window length, invert of sample rate) rad/s return the Discrete Fourier Transform sample frequencies.

    omega_target = photon_frequency

    #Interpolation 
    order = np.argsort(omega)
    omega = omega[order]
    trans_self_energy_tot = trans_self_energy_tot[order, ...]
    
    trans_self_energy = CubicSpline(omega, trans_self_energy_tot,axis=0,extrapolate=True) #could lead to overshooting (twice continuously differentiable), alternatives np.interp,PchipInterpolator,make_interp_spline

    return trans_self_energy #lent