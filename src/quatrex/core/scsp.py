import numpy as np

from qttools.comm import comm
from quatrex.core.config import QuatrexConfig, SCSPConfig
from quatrex.core.qtbm import QTBM
from quatrex.core.transport import TransportSolver
from quatrex.electrostatics.electrostatics import ElectrostaticSolver
from quatrex.electrostatics.mixer import DIIS, Mixer, UnderRelaxation


class SCSP:
    """Self-consistent Schrödinger-Poisson solver.

    This class implements a self-consistent Schrödinger-Poisson (SCSP)
    solver that iteratively solves the Schrödinger equation for the
    charge density and the Poisson equation for the electrostatic
    potential until convergence is achieved.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object containing all the necessary parameters
        for setting up and running the SCSP solver, including parameters
        for the electrostatic solver, transport solver, and mixer.

    """

    def __init__(self, config: QuatrexConfig):
        """Initializes the SCSP solver."""
        self.config = config

        self.transport_solver = self._configure_transport_solver(config)
        self.electrostatic_solver = ElectrostaticSolver(config)

        self.mixer = self._configure_mixer(config.scsp)

        self.convergence_tol = config.scsp.convergence_tol

    @staticmethod
    def _configure_transport_solver(config: QuatrexConfig) -> TransportSolver:
        """Configures the transport solver.

        Parameters
        ----------
        config : QuatrexConfig
            The configuration object containing all the necessary
            parameters for setting up the transport solver, including
            the choice of transport formalism (e.g., wavefunction-based
            or NEGF) and any relevant parameters for the chosen
            formalism.

        Returns
        -------
        TransportSolver
            The configured transport solver, which can be either a QTBM
            or an SCBA solver depending on the specified transport
            formalism in the configuration. The transport solver is
            responsible for computing the charge density based on the
            current potential.

        """
        if config.formalism == "wf":
            from quatrex.core.qtbm import QTBM
            from quatrex.device import Device

            device = Device(config)

            return QTBM(device, config)

        if config.formalism == "negf":
            from quatrex.core.scba import SCBA

            return SCBA(config)

        raise ValueError(f"Unknown transport formalism: {config.formalism}.")

    @staticmethod
    def _configure_mixer(scsp_config: SCSPConfig) -> Mixer:
        """Configures the mixer for the potential updates.

        Parameters
        ----------
        scsp_config : SCSPConfig
            The configuration object containing the parameters for the
            SCSP solver.

        Returns
        -------
        Mixer
            The configured mixer.

        """
        if scsp_config.mixer == "under-relaxation":
            return UnderRelaxation(alpha=scsp_config.mixing_factor)
        if scsp_config.mixer == "diis":
            return DIIS(
                max_history=scsp_config.max_history,
                epsilon=scsp_config.epsilon,
                alpha=scsp_config.mixing_factor,
                extrapolation_interval=scsp_config.extrapolation_interval,
            )

        raise ValueError(f"Unknown mixer type: {scsp_config.mixer}")

    def run(self):
        """Runs the self-consistent Schrödinger-Poisson solver."""

        potential = self.electrostatic_solver.generate_initial_guess()

        for iteration in range(self.config.scsp.max_iterations):

            if isinstance(self.transport_solver, QTBM):
                # TODO: The QTBM solver is currently torn down and
                # re-initialized at each iteration. The issue was that
                # it is a bit harder to reset the observables
                # after they have been allgathered in place.
                self.transport_solver = self._configure_transport_solver(self.config)

            if comm.rank == 0:
                np.save(
                    self.config.output_dir / f"potential_{iteration}.npy",
                    potential,
                )

            self.transport_solver.set_potential(potential)
            self.transport_solver.run()
            charge_density = self.transport_solver.get_charge_density()

            new_potential = self.electrostatic_solver.solve(charge_density, potential)

            if np.max(np.abs(potential - new_potential)) < self.convergence_tol:
                break

            potential = self.mixer.mix(potential, new_potential)

        else:  # max iterations reached without convergence.
            if comm.rank == 0:
                print(
                    "Warning: SCSP did not converge after "
                    f"{self.config.scsp.max_iterations} iterations."
                )

        if comm.rank == 0:
            print(f"SCSP converged after {iteration} iterations.")
            np.save(self.config.output_dir / "potential_final.npy", potential)

        return potential
