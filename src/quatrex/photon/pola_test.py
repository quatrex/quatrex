from threadpoolctl import threadpool_limits, threadpool_info
import numpy as np
#import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import colors
import opt_einsum as oe
from .utils import plot_sparsity, delta_perp_sparse, exponential_decay_hamiltonian,interaction_tensor, show_tensor_cuts, D0_matrix
import timeit
import scipy.sparse as sp
#import cupyx.scipy.sparse as cpx_sp

# input 

mu0 = 8.854e-12
hbar = 1.054571817e-34             # J*s

E = np.linspace(0.1, 4, 20)   # np.ndarray (eV)
dE = (E[1] - E[0]) if E.size > 1 else 1.0
N = E.shape[0]
Np = 2*N
pad = ((0, N), (0,0), (0,0))

def polarization(num_orbitals,g_lesser, g_greater, M):

    #Start implementation of polarizatio

    Pi_lesser = np.zeros(((num_orbitals, num_orbitals, 3, 3)))
    Pi_greater = np.zeros(((num_orbitals, num_orbitals, 3, 3)))

    pref = 1j * mu0 * (dE / (2*np.pi))

    #Get the term for the polarization shape NxNx3x3
    term_1 = oe.contract("imu,mj,jnv,ni->mnuv",M,g_lesser,M,g_greater)
    term_2 = oe.contract("imu,mn,jnv,ji->mnuv",M,g_lesser,M,g_greater)
    term_3 = oe.contract("imu,ij,njv,nm->mnuv",M,g_lesser,M,g_greater)
    term_4 = oe.contract("imu,in,jnv,jm->mnuv",M,g_lesser,M,g_greater)

    Pi_greater = pref*(term_1 + term_2+term_3+term_4)

    #convinient plot example
    # fig, ax = plt.subplots()
    # im = ax.matshow(np.abs(hamiltonian.toarray()), norm=colors.LogNorm())
    # fig.colorbar(im)
    # distancematrix.diagonal()

    # fig, ax = plt.subplots()
    # im = ax.matshow(np.linalg.norm(Pi_greater, axis=(-1, -2)), norm=colors.LogNorm())
    # plt.show()
    # path, path_repr = np.einsum_path(
    #     "imu,mj,jnv,ni->mnuv",
    #     M,
    #     g_lesser,
    #     M,
    #     g_greater,
    #     optimize="greedy",
    # )
    path, path_info = oe.contract_path(
        "imu,mj,jnv,ni->mnuv",
        M,
        g_lesser,
        M,
        g_greater,
    )
    print(path_info)

    return Pi_greater #Pi_lesser


def terms_uv(G1, G2, M_u,M_v):

    #change to the csr (col to row)
    # efficient format : csr compress sparse row matrix
    G1 = sp.csr_matrix(G1) 
    G2 = sp.csr_matrix(G2) 
    Mu = sp.csr_matrix(M_u) 
    Mv = sp.csr_matrix(M_v) 
    
    # G1= cpx_sp.csr_matrix(G1).toarray()
    # G2= cpx_sp.csr_matrix(G1).toarray()
    # Mu = cpx_sp.csr_matrix(G1).toarray()
    # Mv = cpx_sp.csr_matrix(G1).toarray()
    # T1 = (G1 @ Mv) ⊙ (G2 @ Mu).T
    A1 = (G1@ Mv)          # mn 
    B1 = (Mu.transpose() @ G2.transpose())    #mn
    T1 = A1.multiply(B1.transpose())

    # T2 = G1 ⊙ (M_im_u.T @ (G2.T @ M_nj_v.T))
    A2 = G1                      # mn
    B2 = Mu.transpose() @ (Mv.transpose() @ G2)  #(G2.transpose() @ Mv).transpose()  # matrix multiplication (m,n)
    T2 = A2.multiply(B2)                         # Hadamard multiplication:element-wise multiplication (ij) (COULD HAVE USE *), entrywise product

    # T3 = (M_mi_u @ G1 @ M_jn_v) ⊙ G2.T
    A3 = Mu.transpose() @ (G1 @ Mv)  # (m,n)
    B3 = G2.transpose()          # (m,n)
    T3 = A3.multiply(B3)

    # T4 = (M_mi_u @ G1) ⊙ (M_nj_v @ G2).T
    A4 = (Mu.transpose() @ G1)           # (m,n)
    B4 = (G2.transpose() @ Mv) #(Mv.transpose() @ G2).transpose()  # (m,n)
    T4 = A4.multiply(B4)

    return T1 + T2 + T3 + T4  # keep them sparse

def polarizasion_sparse(num_orbitals,G1, G2, M):
    
    prefactor = (1j * mu0) * (1.0 / (2.0*np.pi))
    # Pi_blocks = [[sp.csr_matrix((num_orbitals, num_orbitals), dtype=complex) for _ in range(3)] for _ in range(3)]
    Pi_dense = np.zeros((num_orbitals, num_orbitals, 3, 3),dtype=complex)

    for u in range(3):
        Mu = M[:,:,u]
        for v in range(3):
            Mv = M[:, :, v]
            Tsum = terms_uv(G1, G2, Mu, Mv)    # (N,N) sparse
            # Pi_blocks[u][v] = Tsum
            Pi_dense[:, :, u, v] =  prefactor * Tsum.toarray()    # only if you’ll plot


    # Plot (norme sur les 2 derniers axes 3x3)
    # fig, ax = plt.subplots()
    # im = ax.matshow(np.linalg.norm(Pi_dense, axis=(-1, -2)), norm=colors.LogNorm())
    # plt.show()

    return Pi_dense


def transver_self_energy(num_orbitals,g_lesser, g_greater, M, D):

    #Start implementation of polarization

    sigam_perp_l = np.zeros(((num_orbitals, num_orbitals)))
    sigam_perp_g = np.zeros(((num_orbitals, num_orbitals)))

    pref = 1j * mu0 * (dE / (2*np.pi)) #dhw en vrai
    #Get the term for the polarization shape NxNx3x3
    term_1 = oe.contract("iju,il,klv,ikuv->jk",M,g_lesser,M,D)
    term_2 = oe.contract("iju,il,klv,iluv->jk",M,g_lesser,M,D)
    term_3 = oe.contract("iju,il,klv,jkuv->jk",M,g_lesser,M,D)
    term_4 = oe.contract("iju,il,klv,jluv->jk",M,g_lesser,M,D)

    sigam_perp_g = pref*(term_1 + term_2+term_3+term_4)

    #convinient plot example
    # fig, ax = plt.subplots()
    # im = ax.matshow(np.abs(hamiltonian.toarray()), norm=colors.LogNorm())
    # fig.colorbar(im)
    # distancematrix.diagonal()

    # fig, ax = plt.subplots()
    # im = ax.matshow(np.linalg.norm(sigam_perp_g, axis=(-1, -2)), norm=colors.LogNorm())
    # plt.show()
   

    return sigam_perp_g #sigam_perp_l


def polarization_FFT(energy_grid,num_orbitals, g_lesser_e, g_greater_e, M, pad_factor=2):
    """
    Compute Π^>(ω) using FFTs to turn the energy convolution into a time product.

    E:          (nE,) uniformly spaced energies (eV or J). Use ħ in matching units.
    g_lesser:   (nE, N, N) complex
    g_greater:  (nE, N, N) complex
    M:          (N, N, 3)   real/complex, energy-independent
    mu0:        vacuum permeability
    pad_factor: zero-padding factor along energy axis (>=2 recommended)
    hbar:       ħ in units consistent with E (default eV·s if E in eV)

    Returns:
      omega:            (Np,) angular frequencies (rad/s)
      Pi_greater_omega: (Np, N, N, 3, 3) complex
    """

    #inputs :
    Ne, _, _ = g_lesser_e.shape #ExNxN discretization goes from 0 to Ne-1 (have to add padding)
    E = np.asarray(energy_grid)
    dE = float(E[1] - E[0]) #Np samples along the energy axis spaced by dE = Pb?
    #prepare the input
        # a[0] should contain the zero frequency term,
        #a[1:n//2] should contain the positive-frequency terms,
        #a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.
        #can be complex

    #prepare the input by padding it = adding zeros at the end (basic way: can lead to pb)
    #Padding in energy narrows the resulting τ-domain sampling and reduces circular convolution artifacts when you FFT.
    pad_width = ((0, int(pad_factor * Ne) - Ne), (0, 0), (0, 0))    
    # wir wollen nur in x direction padden & pad_width = (pad_before, pad_after) 
    # no padding before = 0 and pad*factor*Ne-Ne pad width after to avoid problems 
    # FFT-based convolution assumes your data starts at the first grid point and continues forward in energy.
    # padding before would shift the phase of the transform

    gL_E = np.pad(g_lesser_e, pad_width, mode='constant')
    gG_E = np.pad(g_greater_e, pad_width, mode='constant')    #Start implementation of polarizatio

    # Inverse FFT E -> tau (relative time). 
    # NumPy's ifft : 1D inverse FFT is unnormalized.
    # nee d to be normalised? by dE/(2*np.pi*hbar)
    gL_IFFT = dE/(2*np.pi*hbar) * np.fft.ifft(gL_E, axis=0)  # (Np, N, N) Fourrier transform done on the energy axis which is 0
    gG_IFFT = dE/(2*np.pi*hbar) * np.fft.ifft(gG_E, axis=0)  # (Np, N, N)

    #Do the multiplication hihi
    #Get the term for the polarization shape timexNxNx3x3
    term_1 = oe.contract("imu,tmj,jnv,tni->tmnuv",M,gL_IFFT,M,gG_IFFT)
    term_2 = oe.contract("imu,tmn,jnv,tji->tmnuv",M,gL_IFFT,M,gG_IFFT)
    term_3 = oe.contract("imu,tij,njv,tnm->tmnuv",M,gL_IFFT,M,gG_IFFT)
    term_4 = oe.contract("imu,tin,jnv,tjm->tmnuv",M,gL_IFFT,M,gG_IFFT)

    sum = (term_1 + term_2+term_3+term_4)

    #FFT back tau -> omega
    sum_FFT = np.fft.fft(sum, axis=0) 

    prefactor = 1j * mu0 * (1 / (2 * np.pi))
    Pi_greater_omega =  prefactor * sum_FFT # (Np, N, N, 3, 3)
    
    #time step (in sec) and frequency axis``  (in rad/s)
    Np,_,_,_,_ = Pi_greater_omega.shape
    dtau = 2 * np.pi * hbar / (Np * dE) # 1/(N * ws) sampling rate dtau
    omega = 2 * np.pi * np.fft.fftfreq(Np, d=dtau)  # (window length, invert of sample rate) rad/s return the Discrete Fourier Transform sample frequencies.

    #TODO: remove the padding and get back in Ne = 20
    #TOASK: hbar omega : unsure about it 

    return omega, Pi_greater_omega


    # #convinient plot example
    # # fig, ax = plt.subplots()
    # # im = ax.matshow(np.abs(hamiltonian.toarray()), norm=colors.LogNorm())
    # # fig.colorbar(im)
    # # distancematrix.diagonal()

    # fig, ax = plt.subplots()
    # im = ax.matshow(np.linalg.norm(Pi_greater, axis=(-1, -2)), norm=colors.LogNorm())
    # plt.show()
    # path, path_repr = np.einsum_path(
    #     "imu,mj,jnv,ni->mnuv",
    #     M,
    #     g_lesser,
    #     M,
    #     g_greater,
    #     optimize="greedy",
    # )
    # path, path_info = oe.contract_path(
    #     "imu,mj,jnv,ni->mnuv",
    #     M,
    #     g_lesser,
    #     M,
    #     g_greater,
    # )
    # print(path_repr)

