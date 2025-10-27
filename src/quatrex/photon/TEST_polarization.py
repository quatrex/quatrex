import time

import opt_einsum as oe
import scipy
import scipy.sparse as sp
from opt_einsum import shared_intermediates

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from quatrex.core.constants import hbar, mu_0


# Final Version
def polarization(
    energies: NDArray,
    photon_energy: NDArray,
    g_lesser: DSDBSparse,
    g_greater: DSDBSparse,
    m_interaction,
):
    """
    Compute Π^>(ω) using FFTs to turn the energy convolution into a time product.

    energies: (nE,) uniformly spaced energies (eV or J). Use ħ in matching units.
    G1:          (nE, N, N) complex
    G2:          (nE, N, N) complex
    M:           (N, N, 3)   real/complex, energy-independent
    mu_0:         vacuum permeability
    pad_factor:  zero-padding factor along energy axis (>=2 recommended)
    hbar:        ħ in units consistent with E (default eV·s if E in eV)

    Returns:
      omega:                  (Np,) angular frequencies (rad/s)
      p_polarization:         (Np, N, N, 3, 3) complex
    """

    Ne = energies.size  # discretization goes from 0 to Ne-1 (have to add padding)

    dE = xp.diff(energies).mean()
    if not xp.allclose(xp.diff(energies), dE, rtol=1e-6, atol=1e-12):
        raise ValueError("energies should be uniformly spaced for FFT")

    dhw = xp.diff(photon_energy).mean()
    if not xp.allclose(xp.diff(photon_energy), dhw, rtol=1e-6, atol=1e-12):
        raise ValueError("photon_energy should be uniformly spaced for FFT")

    if not xp.isclose(dhw, dE):
        raise ValueError(f"Mismatch in spacing : Δω={dhw:.3e} vs ΔEs={dE:.3e}")

    prefactor = 1j * mu_0 * (1 / (2 * xp.pi))  # *(1/hbar) ! units!!
    n = Ne + Ne - 1
    # padding implementation pad_factor >= 2
    # pad_width = ((0, int(pad_factor * Ne) - Ne), (0, 0), (0, 0))
    # G1_pad = xp.pad(g_lessser, pad_width, mode='constant')
    # G2_pad = xp.pad(g_greater, pad_width, mode='constant')

    # FFT: energy/frequency domain to time domain: energy -> tau
    start_fft_timer = time.perf_counter()
    G1_IFFT = scipy.fft.fft(g_lesser, n, axis=0, workers=128)  # (Np, N, N)
    G2_IFFT = scipy.fft.fft(g_greater, n, axis=0, workers=128)  # (Np, N, N)
    end_fft_timer = time.perf_counter()
    m_interaction = m_interaction.astype(xp.complex64, copy=False)
    print(f"fft took {end_fft_timer - start_fft_timer:.3f}s")

    # Get the term for the polarization via multiplication
    indices_list = [
        "miu,tmj,jnv,tni->tmnuv",
        "miu,tmn,njv,tji->tmnuv",
        "miu,tij,jnv,tnm->tmnuv",
        "miu,tin,njv,tjm->tmnuv",
    ]

    SUM = None
    for i in indices_list:
        start = time.perf_counter()
        path, path_info = oe.contract_path(
            i,
            m_interaction,
            G1_IFFT,
            m_interaction,
            G2_IFFT,
            optimize="optimal",
            memory_limit="max_input",
        )
        end = time.perf_counter()
        print(path_info)
        print(end - start)
        Term = oe.contract(
            i,
            m_interaction,
            G1_IFFT,
            m_interaction,
            G2_IFFT,
            optimize=path,
            memory_limit="max_input",
        )
        if SUM is None:
            SUM = Term
        else:
            SUM += Term
        # free the temporary ASAP
        del Term

    print("Be patient, FFT back is starting...")
    # FFT back:  tau -> omega
    time_FFT_start = time.perf_counter()
    SUM_FFT = scipy.fft.ifft(SUM, axis=0, workers=128)
    SUM_FFT /= SUM.shape[0]
    time_FFT_end = time.perf_counter()
    print(f"fft took {time_FFT_end - time_FFT_start:.3f}s")

    Tpad = SUM_FFT.shape[0]
    # index array
    idx = xp.round((photon_energy - photon_energy[0]) / dE).astype(int)
    # idx = xp.mod(idx, Tpad)

    if xp.any((idx < 0) | (idx >= Tpad)):
        bad = photon_energy[(idx < 0) | (idx >= Tpad)]
        raise ValueError(
            f"Some requested photon energies fall outside the FFT grid: {bad}"
        )

    # if not xp.any(idx):
    #   raise ValueError("Requested photon range is outside computed frequency grid.")

    # select only those frequencies and corresponding polarization values
    # the multplication implemented the convolution in energy domain : now Pi(hw)
    p_polarization_selected = prefactor * SUM_FFT[idx, ...]  # (Nw, N, N, 3, 3)
    p_lesser = p_polarization_selected

    # --- detailed balance: Π^>(ω) = iΠ^<(-hbarω) ---
    # photon_energy est ton tableau des ħω (en eV) correspondant aux lignes sélectionnées
    # p_greater = xp.conj(p_lesser[::-1, :, :, :, :].swapaxes(1,2).swapaxes(3,4))
    p_greater = xp.conj(p_lesser[::-1].transpose(0, 2, 1, 4, 3))

    p_retarded = 0.5 * (p_lesser - p_greater)

    return p_lesser, p_greater, p_retarded  # (Nw, N, N, 3, 3)


# Simple version
def polarization_simple(G1, G2, m_interaction):
    """
    Compute Π with G1 and G2 indpt form energy via einstein sum - Toy exemple

    G1:             (N, N) complex
    G2:             (N, N) complex
    m_interaction:  (N, N, 3)   real/complex, energy-independent
    mu_0:            vacuum permeability

    Returns:
      Pi   :    (N, N, 3, 3) complex
    """
    pref = 1j * mu_0 * (1 / (2 * xp.pi))

    # Get the term for the polarization shape NxNx3x3
    T1 = oe.contract("imu,mj,jnv,ni->mnuv", m_interaction, G1, m_interaction, G2)
    T2 = oe.contract("imu,mn,njv,ji->mnuv", m_interaction, G1, m_interaction, G2)
    T3 = oe.contract("miu,ij,jnv,nm->mnuv", m_interaction, G1, m_interaction, G2)
    T4 = oe.contract("miu,in,njv,jm->mnuv", m_interaction, G1, m_interaction, G2)

    SUM = T1 + T2 + T3 + T4

    pi = pref * SUM

    return pi


def terms_sparse(G1, G2, m_interaction_u, m_interaction_v):
    """
    Compute the sum of the terms necessary to compute Π
    G1:         (N, N) complex
    G2:         (N, N) complex
    M:          (N, N, 3)   real/complex, energy-independent
    mu_0:        vacuum permeability

    Returns:
      sum of the terms (N, N, 3, 3) complex
    """
    # change to the csr (col to row) compress sparse row matrix, because more efficient
    G1 = sp.csr_matrix(G1)
    G2 = sp.csr_matrix(G2)
    Mu = sp.csr_matrix(m_interaction_u)
    Mv = sp.csr_matrix(m_interaction_v)

    # T1
    A1 = G1 @ Mv  # matrix multiplication in scipy.sparse (@) and get mn matrix
    B1 = (
        Mu.transpose() @ G2.transpose()
    )  # matrix multiplication in scipy.sparse (@) and get mn matrix
    T1 = A1.multiply(
        B1
    )  # #Hadamard multiplication:element-wise multiplication of two mn matrices (could have use *)

    # T2
    A2 = G1  # mn
    B2 = Mu.transpose() @ (Mv @ G2).transpose()  # mn
    T2 = A2.multiply(B2)  # mn

    # T3
    A3 = Mu @ (G1 @ Mv)
    B3 = G2.transpose()
    T3 = A3.multiply(B3)

    # T4
    A4 = Mu @ G1
    B4 = (Mv @ G2).transpose()
    T4 = A4.multiply(B4)

    return T1 + T2 + T3 + T4


def polarizasion_sparse(g_greater, g_lesser, m_interaction):
    """
    Compute Π with G1 and G2 indpt form energy via sparse - Toy exemple

    G1:         (N, N) complex
    G2:         (N, N) complex
    M:          (N, N, 3)   real/complex, energy-independent
    mu_0:        vacuum permeability

    Returns:
      pi_greater   :    (N, N, 3, 3) complex
      pi_lesser    :    (N, N, 3, 3) complex
    """
    prefactor = (1j * mu_0) * (1.0 / (2.0 * xp.pi))

    for u in range(3):
        Mu = m_interaction[:, :, u]
        for v in range(3):
            Mv = m_interaction[:, :, v]
            tsum_greater = terms_sparse(g_greater, g_lesser, Mu, Mv)
            tsum_lesser = terms_sparse(g_lesser, g_greater, Mu, Mv)

    pi_greater = prefactor * tsum_greater.toarray()
    pi_lesser = prefactor * tsum_lesser.toarray()

    return pi_greater, pi_lesser
