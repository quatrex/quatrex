import opt_einsum as oe
import time
from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from quatrex.core.constants import mu_0
import scipy 

def transver_self_energy(energy_grid:NDArray,photon_energy:NDArray, g:DSDBSparse, m_interaction, d:DSDBSparse):
    
  """
  Compute D with g form energy via einstein sum - Toy exemple
  g:              (N, N) complex
  d:              (N, N) complex
  m_interaction:  (N, N, 3)   real/complex, energy-independent
  mu_0:            vacuum permeability

  Returns:
    Σ:            (N, N, 3, 3) complex
  """
  
  dE = xp.diff(energy_grid).mean()
  if not xp.allclose(xp.diff(energy_grid), dE, rtol=1e-6, atol=1e-12):
    raise ValueError("energy_grid should be uniformly spaced for FFT")

  dhw = xp.diff(photon_energy).mean()
  if not xp.allclose(xp.diff(photon_energy),dhw,rtol=1e-6,atol=1e-12):
    raise ValueError("photon_energy should be uniformly spaced for FFT")
  
  if not xp.isclose(dhw, dE):
    raise ValueError(f"Mismatch in spacing : Δω={dhw:.3e} vs ΔEs={dE:.3e}")
  
  Ne = energy_grid.size
  Nw = photon_energy.size
  prefactor = 1j * mu_0 * (1 / (2*xp.pi)) 

  # pad_width = ((0, int(pad_factor * max(Nw,Ne)) - Nw), (0, 0), (0, 0))      
  # g_pad = xp.pad(g, pad_width, mode='constant')
  # d_pad = xp.pad(d, pad_width, mode='constant') 
  # 
     
  n = Nw + Nw - 1  # padding
  start_ifft_timer = time.perf_counter()
  # FFT: energy/frequency domain to time domain: energy -> tau
  G_IFFT = scipy.fft.fft(g, n, axis=0,workers = 128)  # (Np, N, N) #TODO: change to the fastest option
  G_IFFT = xp.conj(G_IFFT[::-1, ...])  # reverse the order to get G(tau)
  D_IFFT = scipy.fft.fft(d, n, axis=0,workers =128)  # (Np, N, N)
  end_ifft_timer = time.perf_counter()
  print(
      f"first fourier transform took {end_ifft_timer - start_ifft_timer:.3f}s"
  ) 
  
  indices_list = [
    "iju,til,lkv,tikuv->tjk",
    "iju,til,lkv,tiluv->tjk",  # optimized scaling at 6
    "iju,til,lkv,tjkuv->tjk",  # optimized scaling at 6
    "iju,til,lkv,tjluv->tjk",
  ]
  #Get the term for the polarization 
  SUM = None
  for i in indices_list:

    start = time.perf_counter()
    path, path_info = oe.contract_path(
      i,
      m_interaction,
      G_IFFT,
      m_interaction,
      D_IFFT,
      optimize="optimal",
      memory_limit="max_input", 
    )
    end = time.perf_counter()
    print(
      path_info,
    )  # optionnel: affiche le plan de contraction
    print(end - start)

    Term = oe.contract(
      i,
      m_interaction,
      G_IFFT,
      m_interaction,
      D_IFFT,
      optimize=path,
      memory_limit="max_input",
    )
    # later passes: mutate in place
    if SUM is None:
      # first pass: take a writable copy, do NOT add twice
      SUM = Term + 0
    else:
      SUM += Term

    del Term

  print("Be patient, FFT back is starting...")

  time_FFT_start = time.perf_counter()
  Sigma_full = scipy.fft.ifft(SUM, axis=0, workers=128)  # (n, N, N, 3, 3)
  Sigma_full = prefactor * Sigma_full
  time_FFT_end = time.perf_counter()
  print(
    f"back fourier transform took {time_FFT_end - time_FFT_start:.3f}s"
  )  # in np : 0.583s | scipy : 0.149s

  # index array
  idx = xp.round((energy_grid - energy_grid[0]) / dhw).astype(int)

  if xp.any((idx < 0) | (idx >= Sigma_full.shape[0])):

    bad = photon_energy[(idx < 0) | (idx >= Sigma_full.shape[0])]
    raise ValueError(f"Some requeste energies fall outside the FFT grid: {bad}")

  # select only selected electron energies and corresponding polarization values
  sigma_selected = Sigma_full[idx, ...]  # (NE, N, N, 3, 3)

  s_lesser = sigma_selected
  s_greater = xp.conj(s_lesser[::-1].transpose(0, 2, 1))
  # s_greater[...] = -xp.conj(s_lesser.transpose(0, 2, 1, 4, 3)) #fermionic nature
  s_retarded = 0.5 * (s_lesser - s_greater)

  return s_lesser, s_greater,s_retarded

#Simple version
def transver_self_energy_simple (g, m_interaction, d):
  """
  Compute D with g form energy via einstein sum - Toy exemple
  g:              (N, N) complex
  d:              (N, N) complex
  m_interaction:  (N, N, 3)   real/complex, energy-independent
  mu_0:            vacuum permeability

  Returns:
    Σ:            (N, N, 3, 3) complex
  """
  pref = 1j * mu_0 * (1 / (2*xp.pi)) 

  #Get the term for the polarization 
  __,path_info1 = oe.contract_path("jiu,il,lkv,ikuv->jk",m_interaction,g,m_interaction,d)
  print(path_info1)
  t_0 = time.perf_counter()
  T1 = oe.contract("jiu,il,lkv,ikuv->jk",m_interaction,g,m_interaction,d)
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
