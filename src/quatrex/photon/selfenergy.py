import opt_einsum as oe
import time
import scipy
from pathlib import Path
import scipy.sparse as sp

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSDBSparse

# for initialization
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from qttools.comm import comm
from qttools.utils.gpu_utils import get_host
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr

from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.photon.utils import interaction_tensor
from quatrex.photon.load import IOConfig, load_distances, load_hamiltonian_sparse
from quatrex.core.constants import hbar, mu_0


class PhotonSelfEnergy(ScatteringSelfEnergy):
    """Photon self-energy within the self-consistent Born approximation (SCBA).

    Attributes:
      compute_config:  ComputeConfig
      qc:              QuatrexConfig
      m_interaction:   (N, N, 3)   real/complex, energy-independent
    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: NDArray,
        photon_energies: NDArray,
    ) -> None:

        self.energies = energies
        self.photon_energies = photon_energies

        self.Ne = energies.size
        self.Nw = photon_energies.size
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

        self.m_interaction = interaction_tensor(
            distance_unit_cells, hamiltonian_sparray
        ).astype(xp.complex64, copy=False)
        del hamiltonian_sparray

    def compute(
        self,
        g_lesser: DSDBSparse,
        d_lesser: DSDBSparse,
        out: tuple[NDArray, NDArray, NDArray],
    ) -> None:
        """Compute the photon self-energy Σ.

        Args:
          g:        (Ne, N, N) complex
          d:        (N, N) complex
          outputs:  tuple of NDArray to store results
                    each of shape (Nw, N, N, 3, 3) complex
        """

        s_lesser, s_greater, s_retarded = out

        # compute self-energy

        if not xp.allclose(xp.diff(self.energies), self.dE, rtol=1e-6, atol=1e-12):
            raise ValueError("energy_grid should be uniformly spaced for FFT")

        if not xp.allclose(xp.diff(self.energies), self.dhw, rtol=1e-6, atol=1e-12):
            raise ValueError("photon_energy should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e}"
            )

        n = self.Nw + self.Nw - 1  # padding
        start_ifft_timer = time.perf_counter()
        # FFT: energy/frequency domain to time domain: energy -> tau
        G_IFFT = scipy.fft.fft(
            g_lesser, n, axis=0, workers=128
        )  # (Np, N, N) #TODO: change to the fastest option
        G_IFFT = xp.flip(G_IFFT, axis=0)  # reverse the order to get G(tau)
        #G_IFFT = xp.conj(G_IFFT[::-1, ...])  # reverse the order to get G(tau)
        D_IFFT = scipy.fft.fft(d_lesser, n, axis=0, workers=128)  # (Np, N, N)
        end_ifft_timer = time.perf_counter()
        print(
            f"first fourier transform took {end_ifft_timer - start_ifft_timer:.3f}s"
        )  # np : 27.7 sec | scipy : 0.989s

        # Get the term for the transverse self-energy
        indices_list = [
            "iju,til,lkv,tikuv->tjk",
            "iju,til,lkv,tiluv->tjk",  # optimized scaling at 6
            "iju,til,lkv,tjkuv->tjk",  # optimized scaling at 6
            "iju,til,lkv,tjluv->tjk",
        ]
        SUM = None
        for i in indices_list:

            start = time.perf_counter()
            path, path_info = oe.contract_path(
                i,
                self.m_interaction,
                G_IFFT,
                self.m_interaction,
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
                self.m_interaction,
                G_IFFT,
                self.m_interaction,
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
        Sigma_full = xp.fft.ifft(SUM, axis=0)  # (n, N, N, 3, 3)
        Sigma_full = self.prefactor * Sigma_full
        time_FFT_end = time.perf_counter()
        print(
            f"back fourier transform took {time_FFT_end - time_FFT_start:.3f}s"
        )  # in np : 0.583s | scipy : 0.149s

        # index array
        idx = xp.round((self.energies - self.energies[0]) / self.dhw).astype(int)

        if xp.any((idx < 0) | (idx >= Sigma_full.shape[0])):

            bad = self.photon_energies[(idx < 0) | (idx >= Sigma_full.shape[0])]
            raise ValueError(f"Some requeste energies fall outside the FFT grid: {bad}")

        # select only selected electron energies and corresponding polarization values
        sigma_selected = Sigma_full[idx, ...]  # (NE, N, N, 3, 3)

        s_lesser[...] = sigma_selected
        s_greater[...] = xp.conj(s_lesser[::-1].transpose(0, 2, 1))
        # s_greater[...] = -xp.conj(s_lesser.transpose(0, 2, 1, 4, 3)) #fermionic nature
        s_retarded[...] = 0.5 * (s_lesser - s_greater)


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
        input_dir = Path(".")

    qc = DummyCfg()
    cc = object()  # not used here

    # build the object
    si = PhotonSelfEnergy(
        quatrex_config=qc,
        compute_config=cc,
        energies=energies,
        photon_energies=photon_energies,
    )

    ##### Provisorisch : creation of the G_lesser and G_greater
    random_matrix_g = xp.random.rand(num_orbitals, num_orbitals) + 1j * xp.random.rand(
        num_orbitals, num_orbitals
    )
    g_lesser = xp.zeros(
        (num_electron_energies, num_orbitals, num_orbitals), dtype=complex
    )
    nonzero_indices = xp.nonzero(hamiltonian)
    g_lesser[:, *nonzero_indices] = random_matrix_g[nonzero_indices]

    ##### Provisorisch : creation of the D_lesser and D_greater
    rows, cols = nonzero_indices
    d_lesser = xp.zeros(
        (num_electron_energies, num_orbitals, num_orbitals, 3, 3), dtype=complex
    )
    random_matrix_d = xp.random.rand(
        num_electron_energies, len(rows), 3, 3
    ) + 1j * xp.random.rand(num_electron_energies, len(rows), 3, 3)
    for idx, (i, j) in enumerate(zip(rows, cols)):
        d_lesser[:, i, j, :, :] = random_matrix_d[:, idx, :, :]

    # outputs
    s_less = xp.empty(
        (num_electron_energies, num_orbitals, num_orbitals), dtype=complex
    )
    s_grea = xp.empty_like(s_less)
    s_ret = xp.empty_like(s_less)

    print("Computation of Transverse Self Energy is starting…")
    t0 = time.perf_counter()
    si.compute(g_lesser, d_lesser, (s_less, s_grea, s_ret))
    t1 = time.perf_counter()
    print(f"computation of Transverse Self Energy finished in {t1 - t0:.3f}s")
    print("shapes:", s_less.shape, s_grea.shape, s_ret.shape)

    fig, ax = plt.subplots()
    im = ax.matshow(xp.abs(s_less[2, :, :]), norm=colors.LogNorm())
    plt.colorbar(im, ax=ax)
    plt.savefig("si_less_slice.png", dpi=150)  # because working on remote server
