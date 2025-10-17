import numpy as np
import time
#import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import colors
import opt_einsum as oe
from quatrex.photon.utils import plot_sparsity, delta_perp_sparse, exponential_decay_hamiltonian,interaction_tensor, show_tensor_cuts, D0_matrix
import timeit
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSDBSparse


#Constants
mu0 = 8.854e-12
hbar = 1.054571817e-34             # J*s
  
def polarization(G1, G2, m_interaction):
    
    """
    Compute Π with G1 and G2 indpt form energy via einstein sum - Toy exemple

    G1:             (N, N) complex 
    G2:             (N, N) complex
    m_interaction:  (N, N, 3)   real/complex, energy-independent
    mu0:            vacuum permeability

    Returns:
      Pi   :    (N, N, 3, 3) complex
    """
    pref = 1j * mu0 * (1 / (2*np.pi))

    #Get the term for the polarization shape NxNx3x3 
    T1 = oe.contract("imu,mj,jnv,ni->mnuv",m_interaction,G1,m_interaction,G2)
    T2 = oe.contract("imu,mn,njv,ji->mnuv",m_interaction,G1,m_interaction,G2)
    T3 = oe.contract("miu,ij,jnv,nm->mnuv",m_interaction,G1,m_interaction,G2)
    T4 = oe.contract("miu,in,njv,jm->mnuv",m_interaction,G1,m_interaction,G2)

    SUM = T1 + T2 + T3 + T4
  
    pi = pref * SUM

    return pi 

def terms_sparse(G1, G2, m_interaction_u,m_interaction_v):
    """
    Compute the sum of the terms necessary to compute Π 
    G1:         (N, N) complex
    G2:         (N, N) complex
    M:          (N, N, 3)   real/complex, energy-independent
    mu0:        vacuum permeability

    Returns:
      sum of the terms (N, N, 3, 3) complex
    """
    # change to the csr (col to row) compress sparse row matrix, because more efficient
    G1 = sp.csr_matrix(G1) 
    G2 = sp.csr_matrix(G2) 
    Mu = sp.csr_matrix(m_interaction_u) 
    Mv = sp.csr_matrix(m_interaction_v) 
    
    # T1 
    A1 = (G1 @ Mv)                              # matrix multiplication in scipy.sparse (@) and get mn matrix 
    B1 = (Mu.transpose() @ G2.transpose())      # matrix multiplication in scipy.sparse (@) and get mn matrix
    T1 = A1.multiply(B1)                        # #Hadamard multiplication:element-wise multiplication of two mn matrices (could have use *)

    # T2
    A2 = G1                                     # mn
    B2 = Mu.transpose() @ (Mv @ G2).transpose() # mn
    T2 = A2.multiply(B2)                        # mn                        

    # T3 
    A3 = Mu @ (G1 @ Mv) 
    B3 = G2.transpose()          
    T3 = A3.multiply(B3)

    # T4 
    A4 = (Mu @ G1)          
    B4 = (Mv @ G2).transpose() 
    T4 = A4.multiply(B4)

    return T1 + T2 + T3 + T4 

def polarizasion_sparse(g_greater, g_lesser, m_interaction):
    """
    Compute Π with G1 and G2 indpt form energy via sparse - Toy exemple

    G1:         (N, N) complex 
    G2:         (N, N) complex
    M:          (N, N, 3)   real/complex, energy-independent
    mu0:        vacuum permeability

    Returns:
      pi_greater   :    (N, N, 3, 3) complex
      pi_lesser    :    (N, N, 3, 3) complex
    """
    prefactor = (1j * mu0) * (1.0 / (2.0*np.pi))

    for u in range(3):
        Mu = m_interaction[:,:,u]
        for v in range(3):
            Mv = m_interaction[:, :, v]
            tsum_greater = terms_sparse(g_greater, g_lesser, Mu, Mv)   
            tsum_lesser = terms_sparse(g_lesser, g_greater, Mu, Mv)    

    pi_greater =  prefactor * tsum_greater.toarray()    
    pi_lesser =  prefactor * tsum_lesser.toarray()    

    return pi_greater, pi_lesser


def transver_self_energy (g, m_interaction, d):
    """
    Compute D with g form energy via einstein sum - Toy exemple
    g:              (N, N) complex
    d:              (N, N) complex
    m_interaction:  (N, N, 3)   real/complex, energy-independent
    mu0:            vacuum permeability

    Returns:
      Σ:            (N, N, 3, 3) complex
    """
    pref = 1j * mu0 * (1 / (2*np.pi)) 

    #Get the term for the polarization 
    __,path_info1 = oe.contract_path("jiu,il,lkv,ikuv->jk",m_interaction,g,m_interaction,d)
    print(path_info1)
    t_0 = time.perf_counter()
    T1 = oe.contract_path("jiu,il,lkv,ikuv->jk",m_interaction,g,m_interaction,d)
    t_1 = time.perf_counter()
    print(f" time for t1: {t_1-t_0} ")
    
    __,path_info2 = oe.contract_path("iju,il,lkv,iluv->jk",m_interaction,g,m_interaction,d) #weird
    print(path_info2)
    t_2 = time.perf_counter()
    T2 = oe.contract("iju,il,lkv,iluv->jk",m_interaction,g,m_interaction,d) #weird
    t_3 = time.perf_counter()
    print(f" time for t2: {t_3-t_2} ")
    

    __,path_info3 = oe.contract_path("iju,il,lkv,lkuv->jk",m_interaction,g,m_interaction,d) #make sense
    print(path_info3)
    t_4 = time.perf_counter()
    T3 = oe.contract("iju,il,lkv,lkuv->jk",m_interaction,g,m_interaction,d) #make sense
    t_5 = time.perf_counter()
    print(f" time for t3: {t_5-t_4} ")
    
    __,path_info4 = oe.contract_path("iju,il,lkv,jluv->jk",m_interaction,g,m_interaction,d) #weird
    print(path_info4)
    t_6 = time.perf_counter()
    T4 = oe.contract("iju,il,lkv,jluv->jk",m_interaction,g,m_interaction,d) #weird
    t_7 = time.perf_counter()
    print(f" time for t4: {t_7-t_6} ")
    

    trans_self_energy = pref*(T1 + T2 + T3 + T4)

    return trans_self_energy  

def polarization_FFT(energy_grid:NDArray, photon_energy:NDArray, G1:DSDBSparse, G2:DSDBSparse, M, pad_factor=2):
    
    """
    Compute Π^>(ω) using FFTs to turn the energy convolution into a time product.

    energy_grid: (nE,) uniformly spaced energies (eV or J). Use ħ in matching units.
    G1:          (nE, N, N) complex
    G2:          (nE, N, N) complex
    M:           (N, N, 3)   real/complex, energy-independent
    mu0:         vacuum permeability
    pad_factor:  zero-padding factor along energy axis (>=2 recommended)
    hbar:        ħ in units consistent with E (default eV·s if E in eV)

    Returns:
      omega:                  (Np,) angular frequencies (rad/s)
      p_polarization:         (Np, N, N, 3, 3) complex
    """

    Ne = energy_grid.size #discretization goes from 0 to Ne-1 (have to add padding)
    
    dE = np.diff(energy_grid).mean()
    if not np.allclose(np.diff(energy_grid), dE, rtol=1e-6, atol=1e-12):
        raise ValueError("energy_grid should be uniformly spaced for FFT")

    dhw = np.diff(photon_energy).mean()
    if not np.allclose(np.diff(photon_energy),dhw,rtol=1e-6,atol=1e-12):
        raise ValueError("photon_energy should be uniformly spaced for FFT")
    
    if not np.isclose(dhw, dE):
        raise ValueError(f"Mismatch in spacing : Δω={dhw:.3e} vs ΔEs={dE:.3e}")

    prefactor = 1j * mu0 * (1 / (2 * np.pi))

    #padding implementation pad_factor >= 2
    pad_width = ((0, int(pad_factor * Ne) - Ne), (0, 0), (0, 0))      
    G1_pad = np.pad(G1, pad_width, mode='constant')
    G2_pad = np.pad(G2, pad_width, mode='constant')    

    #Inverse FFT: energy/frequency domain to time domain: energy -> tau
    G1_IFFT = np.fft.ifft(G1_pad, axis=0)  # (Np, N, N) 
    G2_IFFT = np.fft.ifft(G2_pad, axis=0)  # (Np, N, N)

    #Get the term for the polarization via multiplication
    T1 = oe.contract("imu,tmj,jnv,tni->tmnuv",M,G1_IFFT,M,G2_IFFT) # (time, N, N, 3, 3)
    T2 = oe.contract("imu,tmn,njv,tji->tmnuv",M,G1_IFFT,M,G2_IFFT)
    T3 = oe.contract("miu,tij,jnv,tnm->tmnuv",M,G1_IFFT,M,G2_IFFT)
    T4 = oe.contract("miu,tin,njv,tjm->tmnuv",M,G1_IFFT,M,G2_IFFT)
    SUM = (T1 + T2 + T3 + T4)

    #FFT back:  tau -> omega
    SUM_FFT = np.fft.fft(SUM, axis=0) 
    #the multplication implemented the convolution in energy domain : now Pi(hw)
    p_polarization =  prefactor * SUM_FFT # (Np, N, N, 3, 3)
    
    hw_min = photon_energy[0]
    hw_max = photon_energy[-1]
    mask = (dhw >= hw_min) & (dhw <= hw_max)

    if not np.any(mask):
      raise ValueError("Requested photon range is outside computed frequency grid.")

    # select only those frequencies and corresponding polarization values
    p_polarization_selected =  p_polarization[mask, ...]

    return p_polarization_selected #(Np, N, N, 3, 3)

