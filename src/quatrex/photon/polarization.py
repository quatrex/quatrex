# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import opt_einsum as oe
import time
import scipy
import scipy.sparse as sp
from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse

# for initialization

from quatrex.photon.load import IOConfig, load_distances, load_hamiltonian_sparse

from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.photon.utils import interaction_tensor
from quatrex.core.constants import hbar, mu_0


class PiPhoton(ScatteringSelfEnergy):

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: NDArray,
        photon_energies: NDArray,
    ) -> None:
        """
        Initialize the PiPhoton object.
        quatrex_config: QuatrexConfig object with general configuration
        compute_config: ComputeConfig object with compute-specific configuration
        energies: (nE,) uniformly spaced energies (eV).
        photon_energies: (nω,) uniformly spaced photon energies (eV)

        """
        # super().__init__(quatrex_config, compute_config, photon_energies)

        # small usefull informations
        self.energies = energies
        self.photon_energies = photon_energies

        self.Ne = len(
            self.energies
        )  # discretization goes from 0 to Ne-1 (have to add padding)
        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))
        self.dE = xp.diff(energies).mean()
        self.dhw = xp.diff(photon_energies).mean()

        # --- configuration extraction ---
        io_cfg = IOConfig(
            input_dir=quatrex_config.input_dir,
            device=quatrex_config.device,
            example_input_dir=Path(
                "/home/sem25h7/project2/quatrex/examples/carbon-nanotube/inputs/"
            ),
        )
        # LOAD distances & Hamiltonian
        distance_unit_cells = load_distances(io_cfg)
        hamiltonian_sparray, _ = load_hamiltonian_sparse(io_cfg)

        # initialise M
        self.m_interaction = interaction_tensor(
            distance_unit_cells, hamiltonian_sparray
        ).astype(xp.complex64, copy=False)
        del hamiltonian_sparray

    def compute(
        self,
        g_lesser: DSDBSparse | NDArray,
        g_greater: DSDBSparse | NDArray,
        out: tuple[NDArray, NDArray, NDArray],
    ) -> None:
        """
        Compute Π^(ω) using FFTs to turn the energy convolution into a time product.

        G1:          (nE, N, N) complex
        G2:          (nE, N, N) complex

        Returns:
        p_polarization:         (Np, N, N, 3, 3) complex
        """
        p_lesser, p_greater, p_retarded = out

        if not xp.allclose(xp.diff(self.energies), self.dE, rtol=1e-6, atol=1e-12):
            raise ValueError("energy_grid should be uniformly spaced for FFT")

        if not xp.allclose(xp.diff(self.energies), self.dhw, rtol=1e-6, atol=1e-12):
            raise ValueError("photon_energy should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e}"
            )

        # Inverse FFT: energy/frequency domain to time domain: energy -> tau
        n = self.Ne + self.Ne - 1  # padding
        start_fft_timer = time.perf_counter()
        G1_IFFT = xp.fft.fft(g_lesser, n, axis=0)  # (Np, N, N)
        G2_IFFT = xp.fft.fft(g_greater, n, axis=0)  # (Np, N, N)
        M = self.m_interaction.astype(xp.complex64, copy=False)
        end_fft_timer = time.perf_counter()
        print(
            f"fft took {end_fft_timer - start_fft_timer:.3f}s"
        )  # np : 9.933s  | scipy : 9.911s

        # Get the term for the polarization via multiplication

        # self.system_matrix = (T1 + T2 + T3 + T4)
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
                i, M, G1_IFFT, M, G2_IFFT, optimize="optimal", memory_limit="max_input"
            )
            end = time.perf_counter()
            print(
                path_info,
            )
            print(end - start)
            Term = oe.contract(
                i, M, G1_IFFT, M, G2_IFFT, optimize=path, memory_limit="max_input"
            )
            if SUM is None:
                SUM = Term + 0
            else:
                SUM += Term

            del Term

        print("Be patient, FFT back is starting...")
        # FFT back:  tau -> omega
        time_FFT_start = time.perf_counter()
        Pi_omega_full = xp.fft.ifft(SUM, axis=0)  # (n, N, N, 3, 3)
        Pi_omega_full = self.prefactor * Pi_omega_full
        time_FFT_end = time.perf_counter()
        print(
            f"fft took {time_FFT_end - time_FFT_start:.3f}s"
        )  # np: 0.591s | scipy : 0.595s

        # index array
        idx = xp.round(
            (self.photon_energies - self.photon_energies[0]) / self.dE
        ).astype(int)
        # idx = xp.mod(idx, Tpad)

        if xp.any((idx < 0) | (idx >= Pi_omega_full.shape[0])):

            bad = self.photon_energies[(idx < 0) | (idx >= Pi_omega_full.shape[0])]
            raise ValueError(
                f"Some requested photon energies fall outside the FFT grid: {bad}"
            )

        # select only those frequencies and corresponding polarization values
        p_polarization_selected = Pi_omega_full[idx, ...]  # (Nw, N, N, 3, 3)

        p_lesser[...] = p_polarization_selected
        # --- detailed balance: Π^>(ω) = iΠ^<(-hbarω) ---
        p_greater[...] = xp.conj(
            p_lesser[::-1].transpose(0, 2, 1, 4, 3)
        )  # reorders the axes : to keeps axis 0 (energy) first,then swaps 1<->2 (i <-> j) and 3<->4 (u <-> v).
        p_retarded[...] = 0.5 * (p_lesser - p_greater)


# if __name__ == "__main__":
#     from quatrex.core.quatrex_config import parse_config
#     quatrex_config = parse_config(...)


#     pi_photon = PiPhoton();

if __name__ == "__main__":

    from qttools import NDArray, xp
    from quatrex.photon.utils import make_grids
    from pathlib import Path
    from matplotlib import pyplot as plt
    from matplotlib import colors

    # tiny test sizes
    input_dir = Path("/home/sem25h7/project2/quatrex/examples/carbon-nanotube/inputs/")
    hamiltonian = sp.load_npz(input_dir / "hamiltonian.npz").tocsr()
    num_orbitals = 768

    energies, photon_energies = make_grids(
        E_min=-0.1, E_max=0.1, n_points=21, photon_energy_min=0.1, photon_energy_max=0.3
    )
    num_photon_energies = photon_energies.size
    num_electron_energies = energies.size

    class DummyCfg:
        class Dev:
            construct_from_unit_cell = False

        device = Dev()
        input_dir = Path(".")  # Path object

    qc = DummyCfg()
    cc = object()  # not used here

    # build the object
    pi = PiPhoton(qc, cc, energies, photon_energies)

    ##### Provisorisch : creation of the G_lesser and G_greater
    random_matrix = xp.random.rand(num_orbitals, num_orbitals) + 1j * xp.random.rand(
        num_orbitals, num_orbitals
    )
    g_lesser = xp.zeros(
        (num_electron_energies, num_orbitals, num_orbitals), dtype=complex
    )
    g_greater = xp.zeros(
        (num_electron_energies, num_orbitals, num_orbitals), dtype=complex
    )

    nonzero_indices = xp.nonzero(hamiltonian)

    g_lesser[:, *nonzero_indices] = random_matrix[nonzero_indices]
    g_greater[:, *nonzero_indices] = random_matrix[nonzero_indices]

    # outputs
    P_less = xp.empty(
        (num_photon_energies, num_orbitals, num_orbitals, 3, 3), dtype=complex
    )
    P_grea = xp.empty_like(P_less)
    P_ret = xp.empty_like(P_less)

    print("Computation of Polarization is starting…")
    t0 = time.perf_counter()
    pi.compute(g_lesser, g_greater, (P_less, P_grea, P_ret))
    t1 = time.perf_counter()
    print(f"computation of Polarization finished in {t1 - t0:.3f}s")
    print("shapes:", P_less.shape, P_grea.shape, P_ret.shape)

    fig, ax = plt.subplots()
    im = ax.matshow(xp.abs(P_less[2, :, :, 0, 0]), norm=colors.LogNorm())
    plt.colorbar(im, ax=ax)
    plt.savefig("pi_less_slice.png", dpi=150)
