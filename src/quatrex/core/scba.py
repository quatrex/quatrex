import numpy.linalg as npla

from quatrex.core.config import QuatrexConfig
from quatrex.electron.sigma_phonon import SigmaPhonon
from quatrex.electron.electron_solver import ElectronSolver

from qttools.datastructures.dbcsr import DBCSR


@dataclass
class SCBAState:
    sigma_phonon: SigmaPhonon | None = None
    electron_solver: ElectronSolver | None = None

    # pi_electron: PiElectron | None = None
    # phonon_solver: PhononSolver | None = None


class SCBA:
    """Computes the self-consistent Born approximation to convergence.

    Parameters
    ----------
    config_file : Path
        The configuration for the LDOS calculation.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        self.config = config

        self.electron_solver = None
        self.sigma_phonon = None

    def _warm_up_iteration(self) -> SCBAState:
        """Performs a warm-up iteration to compute the starting self-energy."""

        electron_solver = ElectronSolver(self.config, cli=True)
        g_lesser, g_greater = electron_solver.solve_lesser_greater()

        sigma_lesser = DBCSR(electron_solver.num_energies_per_rank, electron_solver.nnz)
        sigma_greater = DBCSR(electron_solver.num_energies_per_rank, electron_solver.nnz)

        scba = SCBAState()

        if self.config.scba.phonon:
            if self.config.phonon.model == "greens_function":
                raise NotImplementedError

            elif self.config.phonon.model == "deformation_potential":
                phonon_self_energy = SigmaPhonon(self.config, g_lesser, g_greater)

                scba.sigma_phonon = phonon_self_energy

                sigma_phonon_lesser, sigma_phonon_greater = scba.sigma_phonon.compute()

                sigma_lesser += sigma_phonon_lesser
                sigma_greater += sigma_phonon_greater

        electron_solver.sigma_lesser = sigma_lesser
        electron_solver.sigma_greater = sigma_greater

        scba.electron_solver = electron_solver

        return scba

    def run(self) -> None:

        scba = self._warm_up_iteration(self.config)

        for __ in range(1, self.config.scba.max_iterations + 1):
            g_lesser, g_greater = scba.electron_solver.solve_lesser_greater()

            sigma_lesser = None
            sigma_greater = None

            if self.config.scba.phonon:
                if self.config.phonon.model == "greens_function":
                    raise NotImplementedError

                elif self.config.phonon.model == "deformation_potential":
                    scba.sigma_phonon.g_lesser = g_lesser
                    scba.sigma_phonon.g_greater = g_greater

                    sigma_phonon_lesser, sigma_phonon_greater = (
                        scba.sigma_phonon.compute()
                    )

                    sigma_lesser += sigma_phonon_lesser
                    sigma_greater += sigma_phonon_greater

            # Compute difference between current and preceeding self-energy.
            sigma_retarded = 0.5 * (sigma_greater - sigma_lesser)

            prev_sigma_retarded = 0.5 * (
                scba.electron_solver.sigma_greater - scba.electron_solver.sigma_lesser
            )

            diff_causal = sigma_retarded - prev_sigma_retarded

            abs_norm_diff_causal = npla.norm(diff_causal)
            rel_norm_diff_causal = abs_norm_diff_causal / npla.norm(sigma_retarded)

            # Update self-energy.
            m = self.config.scba.mixing_factor
            scba.electron_solver.sigma_lesser = (
                m * scba.electron_solver.sigma_lesser + (1 - m) * sigma_lesser
            )
            scba.electron_solver.sigma_greater = (
                m * scba.electron_solver.sigma_greater + (1 - m) * sigma_greater
            )

            if rel_norm_diff_causal < self.config.scba.tolerance:
                print(f"SCBA converged after {__} iterations.")
                break

        else:  # Did not break, i.e. max_iterations reached.
            print(f"SCBA did not converge after {__} iterations.")
