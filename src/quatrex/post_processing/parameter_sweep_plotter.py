"""Parameter sweep plotter for analyzing SLURM simulation outputs."""

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from quatrex.core.quatrex_config import parse_config


class ParameterSweepPlotter:
    """
    A class for parsing and plotting results from parameter sweep simulations.

    Looks for folders with a specified parameter name (e.g., 'eps*'), parses the
    SLURM output files with the largest number in each folder, extracts relevant
    data, and provides methods for visualization.

    Attributes
    ----------
    root_folder : Path
        Root folder containing parameter sweep directories
    parameter_pattern : str
        Pattern to match parameter folders (e.g., 'eps*')
    data : dict
        Extracted data organized by parameter value
    """

    def __init__(self, root_folder: str, parameter_pattern: str) -> None:
        """
        Initialize the ParameterSweepPlotter.

        Parameters
        ----------
        root_folder : str
            Root folder containing parameter sweep directories
        parameter_pattern : str
            Pattern to match parameter folders (e.g., 'eps*')
        """
        self.root_folder = Path(root_folder)
        self.parameter_pattern = parameter_pattern
        self.data: Dict[float, Dict[str, List[float]]] = {}
        self.parameter_folders: Dict[float, Path] = {}
        conf_folder = self._parse_all_folders()
        self.config = parse_config(conf_folder / "quatrex_config.toml")
        self.energies = np.linspace(
            self.config.electron.energy_window_min,
            self.config.electron.energy_window_max,
            self.config.electron.energy_window_num,
        )
        self.lattice_vectors = np.load(
            conf_folder / "inputs" / "lattice_vectors.npy"
        )

    def _find_parameter_folders(self) -> Dict[float, Path]:
        """
        Find all folders matching the parameter pattern and extract parameter values.

        Returns
        -------
        dict
            Dictionary mapping parameter values to folder paths
        """
        parameter_folders: Dict[float, Path] = {}

        for folder in self.root_folder.glob(self.parameter_pattern):
            if folder.is_dir():
                # Extract numerical value from folder name
                match = re.search(r"(\d+\.?\d*(?:[eE][-+]?\d+)?)", folder.name)
                if match:
                    param_value = float(match.group(1))
                    parameter_folders[param_value] = folder

        return parameter_folders

    def _find_latest_slurm_file(self, folder: Path) -> Optional[Path]:
        """
        Find the SLURM file with the largest number in the given folder.

        Parameters
        ----------
        folder : Path
            Directory to search for SLURM files

        Returns
        -------
        Path or None
            Path to the SLURM file with the largest number, or None if not found
        """
        slurm_files = list(folder.glob("slurm-*"))

        if not slurm_files:
            return None

        def extract_number(filename: Path) -> int:
            match = re.search(r"slurm-(\d+)", filename.name)
            return int(match.group(1)) if match else -1

        return max(slurm_files, key=extract_number)

    def _extract_values_from_slurm(self, filepath: Path) -> Dict[str, List[float]]:
        """
        Extract all values from a SLURM file.

        Parameters
        ----------
        filepath : Path
            Path to the SLURM file

        Returns
        -------
        dict
            Dictionary containing lists of extracted values for each metric
        """
        extracted_data = {
            "conduction_band": [],
            "valence_band": [],
            "fermi_level": [],
            "charge": [],
            "valley_diff_k_q": [],
            "valley_diff_k_g": [],
            "max_self_energy_update": [],
        }

        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Regex pattern for floating point numbers (including scientific notation)
            float_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

            # Extract Conduction Band Edge
            matches = re.findall(rf"Conduction Band Edge:\s*{float_pattern}", content)
            extracted_data["conduction_band"] = [float(m) for m in matches]

            # Extract Valence Band Edge
            matches = re.findall(rf"Valence Band Edge:\s*{float_pattern}", content)
            extracted_data["valence_band"] = [float(m) for m in matches]

            # Extract Fermi level
            matches = re.findall(rf"Fermi level:\s*{float_pattern}", content)
            extracted_data["fermi_level"] = [float(m) for m in matches]

            # Extract Previous charge
            matches = re.findall(rf"Previous charge:\s*{float_pattern}", content)
            extracted_data["charge"] = [float(m) for m in matches]

            # Extract Valley difference K-Q
            matches = re.findall(
                rf"Valley difference between K and Q symmetry points:\s*{float_pattern}",
                content,
            )
            extracted_data["valley_diff_k_q"] = [float(m) for m in matches]

            # Extract Valley difference K-G
            matches = re.findall(
                rf"Valley difference between K and G symmetry points:\s*{float_pattern}",
                content,
            )
            extracted_data["valley_diff_k_g"] = [float(m) for m in matches]

            # Extract Maximum Self-Energy Update
            matches = re.findall(
                rf"Maximum Self-Energy Update:\s*{float_pattern}", content
            )
            extracted_data["max_self_energy_update"] = [float(m) for m in matches]

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

        return extracted_data

    def _parse_all_folders(self) -> None:
        """Parse all parameter folders and extract data."""
        parameter_folders = self._find_parameter_folders()
        self.parameter_folders = parameter_folders
        if not parameter_folders:
            print(
                f"No parameter folders found matching pattern '{self.parameter_pattern}' in {self.root_folder}"
            )
            return

        for param_value, folder in sorted(parameter_folders.items()):
            slurm_file = self._find_latest_slurm_file(folder)
            if slurm_file:
                self.data[param_value] = self._extract_values_from_slurm(slurm_file)
            else:
                print(f"Warning: No SLURM file found in {folder}")
        # Return a folder for additional processing of shared data
        return list(parameter_folders.values())[0]

    def _get_parameter_folder(self, parameter_value: Union[str, float]) -> Path:
        """Return the sweep folder associated with a parameter value."""
        par_value_float = float(parameter_value)
        if par_value_float not in self.parameter_folders:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )

        return self.parameter_folders[par_value_float]

    def _get_last_values(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get parameter values and last iteration values for all metrics.

        Returns
        -------
        tuple
            Parameters array, conduction band array, valence band array,
            fermi level array, valley diff K-Q array, valley diff K-G array
        """
        params = sorted(self.data.keys())
        conduction_bands = []
        valence_bands = []
        fermi_levels = []
        valley_diff_k_q = []
        valley_diff_k_g = []

        for param in params:
            data = self.data[param]
            if data["conduction_band"]:
                conduction_bands.append(data["conduction_band"][-1])
            if data["valence_band"]:
                valence_bands.append(data["valence_band"][-1])
            if data["fermi_level"]:
                fermi_levels.append(data["fermi_level"][-1])
            if data["valley_diff_k_q"]:
                valley_diff_k_q.append(data["valley_diff_k_q"][-1])
            if data["valley_diff_k_g"]:
                valley_diff_k_g.append(data["valley_diff_k_g"][-1])

        return (
            np.array(params),
            np.array(conduction_bands),
            np.array(valence_bands),
            np.array(fermi_levels),
            np.array(valley_diff_k_q),
            np.array(valley_diff_k_g),
        )

    def plot_band_edges_and_fermi(self, ax=None) -> plt.Axes:
        """
        Plot conduction band, valence band, and Fermi level at the last iteration.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots()

        params, cb, vb, fermi, _, _ = self._get_last_values()

        ax.plot(params, cb, "o-", label="Conduction Band Edge")
        ax.plot(params, vb, "s-", label="Valence Band Edge")
        ax.plot(params, fermi, "^-", label="Fermi Level")

        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Band Edges and Fermi Level vs Parameter")
        ax.legend()
        ax.grid(True)

        return ax

    def plot_bandgap_vs_parameter(self, ax=None, reference_bandgap=None) -> plt.Axes:
        """
        Plot bandgap (conduction band - valence band) at the last iteration vs parameter.

        Includes a horizontal reference line showing the bandgap from the first iteration
        of the first parameter value.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots()

        params, cb, vb, _, _, _ = self._get_last_values()
        bandgap = cb - vb

        # Get reference bandgap from first iteration of first parameter
        first_param = sorted(self.data.keys())[0]
        first_cb = self.data[first_param]["conduction_band"][0]
        first_vb = self.data[first_param]["valence_band"][0]
        dft_bandgap = first_cb - first_vb

        ax.plot(params, bandgap, "o-", label="Bandgap (final iteration)")
        ax.axhline(
            y=dft_bandgap,
            color="red",
            linestyle="--",
            label=f"DFT gap (initial: {dft_bandgap:.4f} eV)",
        )
        if reference_bandgap is not None:
            ax.axhline(
                y=reference_bandgap,
                color="red",
                linestyle="--",
                label=f"Reference {reference_bandgap:.4f} eV)",
            )

        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Bandgap (eV)")
        ax.set_title("Bandgap vs Parameter")
        ax.legend()
        ax.grid(True)

        return ax

    def plot_charge_vs_iteration(self, parameter_value: float, ax=None) -> plt.Axes:
        """
        Plot charge as a function of iteration for a specific parameter value.

        Parameters
        ----------
        parameter_value : float
            The parameter value to plot
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot

        Raises
        ------
        ValueError
            If the parameter value is not found in the data
        """
        if ax is None:
            fig, ax = plt.subplots()

        if parameter_value not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {sorted(self.data.keys())}"
            )

        charges = self.data[parameter_value]["charge"]

        # Normalize over k-points
        charges = np.array(charges) / np.prod(self.config.device.kpoint_grid)
        # # Calculate the number of charge per cm^2 using the lattice vectors
        area_per_unit_cell = np.cross(self.lattice_vectors[0], self.lattice_vectors[1])[
            2
        ]  # Area in Å^2
        area_per_unit_cell_cm2 = area_per_unit_cell * 1e-16  # Convert to cm^2
        charge_per_cm2 = charges / area_per_unit_cell_cm2  # Charge density in e/cm^2
        # Factor 2 for spin degeneracy
        charge_per_cm2 *= 2

        iterations = np.arange(len(charges))

        ax.plot(iterations, charge_per_cm2, "o-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Charge Density (e/cm²)")
        ax.set_title(f"Charge Density vs Iteration (Parameter = {parameter_value})")
        ax.grid(True)

        return ax

    def plot_valley_differences(
        self, kq_ref: float = None, kg_ref: float = None
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Plot valley differences as a function of parameter in side-by-side subplots.

        Parameters
        ----------
        kq_ref : float, optional
            Reference K-Q valley difference value to plot as a horizontal line
        kg_ref : float, optional
            Reference K-G valley difference value to plot as a horizontal line

        Returns
        -------
        tuple of matplotlib.axes.Axes
            Tuple of axes objects (ax_k_q, ax_k_g) for K-Q and K-G valley differences
        """
        fig, (ax_k_q, ax_k_g) = plt.subplots(1, 2, figsize=(12, 4))

        params, _, _, _, valley_k_q, valley_k_g = self._get_last_values()

        # Get reference valley differences from first iteration of first parameter
        first_param = sorted(self.data.keys())[0]
        reference_k_q = (
            self.data[first_param]["valley_diff_k_q"][0]
            if self.data[first_param]["valley_diff_k_q"]
            else None
        )
        reference_k_g = (
            self.data[first_param]["valley_diff_k_g"][0]
            if self.data[first_param]["valley_diff_k_g"]
            else None
        )

        # Plot K-Q valley difference
        ax_k_q.plot(params, valley_k_q, "o-", color="C0")
        ax_k_q.set_xlabel("Parameter Value")
        ax_k_q.set_ylabel("Valley Difference")
        ax_k_q.set_title("Valley Difference (K-Q)")
        ax_k_q.grid(True)
        ax_k_q.axhline(
            y=reference_k_q,
            color="red",
            linestyle="--",
            label=(
                f"Reference (initial: {reference_k_q:.4f} eV)"
                if reference_k_q is not None
                else "Reference (initial: N/A)"
            ),
        )
        if kq_ref is not None:
            ax_k_q.axhline(
                y=kq_ref,
                color="blue",
                linestyle="--",
                label=f"K-Q Reference: {kq_ref:.4f} eV",
            )
        ax_k_q.legend()

        # Plot K-G valley difference
        ax_k_g.plot(params, valley_k_g, "s-", color="C1")
        ax_k_g.set_xlabel("Parameter Value")
        ax_k_g.set_ylabel("Valley Difference")
        ax_k_g.set_title("Valley Difference (K-G)")
        ax_k_g.grid(True)
        ax_k_g.axhline(
            y=reference_k_g,
            color="red",
            linestyle="--",
            label=(
                f"Reference (initial: {reference_k_g:.4f} eV)"
                if reference_k_g is not None
                else "Reference (initial: N/A)"
            ),
        )
        if kg_ref is not None:
            ax_k_g.axhline(
                y=kg_ref,
                color="blue",
                linestyle="--",
                label=f"K-G Reference: {kg_ref:.4f} eV",
            )
        ax_k_g.legend()

        fig.tight_layout()

        return ax_k_q, ax_k_g

    def plot_max_self_energy_vs_iteration(
        self, parameter_value: float, ax=None
    ) -> plt.Axes:
        """
        Plot maximum self-energy update as a function of iteration for a specific parameter.

        Parameters
        ----------
        parameter_value : float
            The parameter value to plot
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot

        Raises
        ------
        ValueError
            If the parameter value is not found in the data
        """
        if ax is None:
            fig, ax = plt.subplots()

        if parameter_value not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {sorted(self.data.keys())}"
            )

        max_self_energy = self.data[parameter_value]["max_self_energy_update"]
        iterations = np.arange(len(max_self_energy))

        ax.plot(iterations, max_self_energy, "o-")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Update [eV]")
        ax.set_title(
            f"Max Self-Energy Update vs Iteration (Parameter = {parameter_value})"
        )
        ax.grid(True)

        return ax
    
    def get_dos(self, parameter_value: str, iteration: int, kp: tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get density of states (DOS) for a specific parameter value and iteration.

        Parameters
        ----------
        parameter_value : str
            The parameter value to get the DOS for
        iteration : int
            The iteration number to get the DOS for
        kp : tuple[int, int], optional
            The k-point indices to get the DOS for. If None, returns the sum over all k-points.

        Returns
        -------
        tuple of np.ndarray
            Tuple containing energy array and corresponding DOS values

        Raises
        ------
        ValueError
            If the parameter value is not found in the data or if DOS data is not available
        """
        par_value_float = float(parameter_value)
        if par_value_float not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )

        dos_energy = self.energies
        dos_file = self._get_parameter_folder(parameter_value) / "outputs" / f"electron_ldos_{iteration}.npy"
        if not dos_file.exists():
            raise ValueError(f"DOS file {dos_file} not found for parameter value {parameter_value} and iteration {iteration}")
        
        if kp is None:
            # k-point resolved DOS
            dos_values = np.load(dos_file)
        else:
            dos_values = np.load(dos_file)[:, kp[0], kp[1]]

        return dos_energy, dos_values

    def plot_dos(self, parameter_value: str, iteration: int, ax=None, kp: tuple[int, int] = None) -> plt.Axes:
        """
        Plot density of states (DOS) for a specific parameter value.

        Parameters
        ----------
        parameter_value : str
            The parameter value to plot
        iteration : int
            The iteration number to plot the DOS for
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        kp : tuple[int, int], optional
            The k-point indices to plot. If None, plots the sum over all k-points.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot

        Raises
        ------
        ValueError
            If the parameter value is not found in the data or if DOS data is not available
        """
        if ax is None:
            fig, ax = plt.subplots()

        par_value_float = float(parameter_value)
        if par_value_float not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )

        dos_energy = self.energies
        parameter_folder = self._get_parameter_folder(parameter_value)
        dos_file = parameter_folder / "outputs" / f"electron_ldos_{iteration}.npy"
        # Have first iteration as reference
        dos_ref_file = parameter_folder / "outputs" / "electron_ldos_0.npy"
        if kp is None:
            # Sum over k-points
            dos_values = np.load(dos_file).mean(axis=(-2, -1))
            dos_ref = np.load(dos_ref_file).mean(axis=(-2, -1))
        else:
            dos_values = np.load(dos_file)[:, kp[0], kp[1]]
            dos_ref = np.load(dos_ref_file)[:, kp[0], kp[1]]

        ax.plot(dos_energy, dos_ref, label="Reference (Iteration 0)", color="lightgray", alpha=0.7)
        ax.plot(dos_energy, dos_values, label=f"Iteration {iteration}")
        # Add vertical lines for Fermi level
        fermi_level = self.data[par_value_float]["fermi_level"][iteration]
        ax.axvline(x=fermi_level, color="red", linestyle="--", label="Fermi Level")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Density of States")
        ax.set_title(f"Density of States (Parameter = {parameter_value})")
        ax.grid(True)

        return ax
    
    def occupation(self, fermi_level: float) -> np.ndarray:
        """
        Calculate the Fermi-Dirac occupation function.

        Parameters
        ----------
        energy : np.ndarray
            Array of energy values
        fermi_level : float
            The Fermi level energy

        Returns
        -------
        np.ndarray
            Array of occupation probabilities corresponding to the input energies
        """
        k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        T = self.config.electron.temperature  # Temperature in Kelvin (can be made a parameter if needed)
        return 1 / (np.exp((self.energies - fermi_level) / (k_B * T)) + 1)

    def _extract_charge_at_kp(self, parameter_value: str, iteration: int, kp: tuple[int, int]) -> float:
        """
        Extract the charge at a specific k-point for a given parameter value and iteration.

        Parameters
        ----------
        parameter_value : str
            The parameter value to extract the charge for
        iteration : int
            The iteration number to extract the charge for
        kp : tuple[int, int]
            The k-point indices to extract the charge from

        Returns
        -------
        float
            The charge at the specified k-point

        Raises
        ------
        ValueError
            If the parameter value is not found in the data or if charge data is not available
        """
        par_value_float = float(parameter_value)
        if par_value_float not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )

        dos_energy = self.energies
        dos_file = self._get_parameter_folder(parameter_value) / "outputs" / f"electron_ldos_{iteration}.npy"
        # Note that the DOS file is expected to have shape (num_energies, num_kpoints_x, num_kpoints_y)
        # The orbital index is already summed over to save disk space.
        dos_values = np.load(dos_file)[:, kp[0], kp[1]]
        # Integrate DOS up to Fermi level to get charge at this k-point
        fermi_level = self.data[par_value_float]["fermi_level"][iteration]
        mid_bandgap = (self.data[par_value_float]["conduction_band"][iteration] + self.data[par_value_float]["valence_band"][iteration]) / 2
        excess_occupation = self.occupation(fermi_level) - self.occupation(mid_bandgap)  # Occupation relative to mid-gap
        # NOTE: In the code the np.pi factor is not included. It actually should be included
        # but current results are not normalized by it, so I will keep it consistent with the current code.
        charge_kp = np.sum(dos_values * excess_occupation) * (dos_energy[1] - dos_energy[0]) / (np.pi)
        # Calculate the number of charge per cm^2 using the lattice vectors
        area_per_unit_cell = np.cross(self.lattice_vectors[0], self.lattice_vectors[1])[
            2
        ] * 1e-16 # Area in cm^2
        charge_kp_density = charge_kp / area_per_unit_cell  # Charge density in e/cm^2
        # When calculating charge density, you usually take the mean over all k-points, 
        # so should I divide by the total number of k-points to get the charge density at this k-point? Yes!
        num_kpoints = np.prod(self.config.device.kpoint_grid)
        charge_kp_density /= num_kpoints
        # Factor 2 for spin degeneracy
        charge_kp_density *= 2
        return charge_kp_density
    
    def plot_charge_at_kp_vs_parameter(
        self,
        kp: tuple[int, int],
        exclude_params: List[str] = None,
        ax=None,
    ) -> plt.Axes:
        """
        Plot charge at a specific k-point as a function of parameter value.

        The plot also overlays the charge percentage on a secondary y-axis.

        Parameters
        ----------
        kp : tuple[int, int]
            The k-point indices to plot the charge for
        exclude_params : list of str, optional
            List of parameter values to exclude from the plot (e.g., [0, 1e12])
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot

        Raises
        ------
        ValueError
            If the parameter value is not found in the data or if charge data is not available
        """
        if ax is None:
            fig, ax = plt.subplots()

        params = sorted(self.data.keys())
        if exclude_params:
            params = [param for param in params if param not in exclude_params]
        kp_display = tuple(int(index) for index in kp)
        charges_kp = []
        for param in params:
            # Extract the number of iterations for this parameter to get the last iteration number
            num_iterations = len(self.data[param]["fermi_level"])
            charge_kp = self._extract_charge_at_kp(param, num_iterations - 1, kp)  # Get charge at last iteration
            charges_kp.append(charge_kp)

        # Assume that the parameter value is the total charge (this only works if the charge is the parameter being swept)
        if self.parameter_pattern.startswith("doping"):
            total_charges = [float(param) for param in params]
        else:
            raise NotImplementedError(
                "Total charge is not directly available from the parameter values. "
                "Please implement a method to calculate total charge for each parameter value."
            )

        charge_percentages = [
            100 * (charge / total) if total != 0 else 0
            for charge, total in zip(charges_kp, total_charges)
        ]

        ax.plot(params, charges_kp, "o-")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Charge Density (e/cm²)")
        ax.set_title(f"Charge Density at k-point {kp_display}")
        ax.grid(True)

        ax_percentage = ax.twinx()
        ax_percentage.plot(params, charge_percentages, "s--", color="C1")
        ax_percentage.set_ylabel("Charge Percentage (%)")

        handles_charge, labels_charge = ax.get_legend_handles_labels()
        handles_percentage, labels_percentage = ax_percentage.get_legend_handles_labels()
        ax.legend(handles_charge + handles_percentage, labels_charge + labels_percentage)

        return ax

    def plot_charge_percentage_at_kp_vs_parameter(self, kp: tuple[int, int], exclude_params: List[str] = None, ax=None) -> plt.Axes:
        """
        Backward-compatible wrapper for plotting charge and charge percentage at a k-point.
        Parameters
        ----------
        kp : tuple[int, int]
            The k-point indices to plot the charge for
        exclude_params : list of str, optional
            List of parameter values to exclude from the plot (e.g., [0, 1e12])
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        return self.plot_charge_at_kp_vs_parameter(kp=kp, exclude_params=exclude_params, ax=ax)

    def plot_fermi_level_vs_parameter(self, exclude_params: List[str] = None, ax=None) -> plt.Axes:
        """
        Plot Fermi level as a function of all parameter values. Also plots the same thing but as a function of the 
        Fermi level difference to the conduction band edge.

        In the plot is also the energy minimums of the K and Q valleys, so that one can easily see how the Fermi 
        level moves with respect to the band edges.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        exclude_params : list of str, optional
            List of parameter values to exclude from the plot (e.g., [0, 1e12])

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        params, conduction_band, _, fermi, K_Q_valley_diff, _ = self._get_last_values()
        if exclude_params:
            mask = [p not in exclude_params for p in params]
            params = params[mask]
            fermi = fermi[mask]
            conduction_band = conduction_band[mask]
            K_Q_valley_diff = K_Q_valley_diff[mask]

        # Calculate K and Q valley minimums from conduction band edge and K-Q valley difference
        K_valley_min = np.zeros_like(conduction_band)
        Q_valley_min = np.zeros_like(conduction_band)
        for i, diff in enumerate(K_Q_valley_diff):
            if diff < 0:
                # K valley is lower than Q valley and thus the conduction band minimum
                K_valley_min[i] = conduction_band[i]
                Q_valley_min[i] = conduction_band[i] - diff
            else:
                # Q valley is lower than K valley and thus the conduction band minimum
                Q_valley_min[i] = conduction_band[i]
                K_valley_min[i] = conduction_band[i] + diff

        ax[0].plot(params, fermi, "o-", label="Fermi Level")
        ax[0].plot(params, K_valley_min, "--", label="K Valley Minimum")
        ax[0].plot(params, Q_valley_min, "--", label="Q Valley Minimum")
        ax[0].set_xlabel("Parameter Value")
        ax[0].set_ylabel("Fermi Level (eV)")
        ax[0].set_title("Fermi Level vs Parameter")
        ax[0].legend()
        ax[0].grid(True)
        # Fermi level difference to conduction band edge
        fermi_diff_cb = fermi - conduction_band
        ax[1].plot(params, fermi_diff_cb, "o-", label="Fermi Level - Conduction Band Edge")
        ax[1].plot(params, K_valley_min - conduction_band, "--", label="K Valley Minimum - Conduction Band Edge")
        ax[1].plot(params, Q_valley_min - conduction_band, "--", label="Q Valley Minimum - Conduction Band Edge")
        ax[1].set_xlabel("Parameter Value")
        ax[1].set_ylabel("Energy Difference (eV)")
        ax[1].set_title("Fermi Level Difference to Conduction Band Edge vs Parameter")
        ax[1].legend()
        ax[1].grid(True)
        return ax

    def get_total_num_states(self, parameter_value: str) -> int:
        """
        Get the total number of states for a specific parameter value.

        Parameters
        ----------
        parameter_value : str
            The parameter value to get the total number of states for

        Returns
        -------
        int
            The total number of states for the given parameter value
        """
        par_value_float = float(parameter_value)
        if par_value_float not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )

        dos_energy = self.energies
        folder = self._get_parameter_folder(parameter_value) / "outputs"
        dos_files = list(folder.glob(f"electron_ldos_*.npy"))
        if not dos_files:
            raise ValueError(f"No DOS files found for parameter value {parameter_value}")
        
        # Sort files by iteration number
        dos_files.sort(key=lambda f: int(re.search(r"electron_ldos_(\d+).npy", f.name).group(1)))
        dos_e = np.array([np.load(f).mean(axis=(-2, -1)) for f in dos_files])  # Shape: (num_iterations, num_energies)
        # Sum over energies to get total number of states
        de = dos_energy[1] - dos_energy[0]
        num_states = np.sum(dos_e, axis=1) * de / (np.pi)  # Sum over energy bins to get total number of states
        return num_states
    
    def get_mid_bandgap(self, parameter_value: str, iteration: int) -> float:
        """
        Get the mid-gap energy for a specific parameter value and iteration.

        Parameters
        ----------
        parameter_value : str
            The parameter value to get the mid-gap energy for
        iteration : int
            The iteration number to get the mid-gap energy for
        Returns
        -------
        float
            The mid-gap energy for the given parameter value and iteration
        """
        par_value_float = float(parameter_value)
        if par_value_float not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )
        conduction_band_edge = self.data[par_value_float]["conduction_band"][iteration]
        valence_band_edge = self.data[par_value_float]["valence_band"][iteration]
        mid_gap = (conduction_band_edge + valence_band_edge) / 2
        return mid_gap

    def plot_band_edges_vs_kpoint(self, parameter_value: str, iteration: int, ax_cb=None, ax_vb=None) -> plt.Axes:
        """
        Plot conduction and valence band edges as a function of k-point index for a specific parameter value and iteration.
        The k-point path is along the G to K, i.e. the kp index goes from (0, 0) to (num_kp, num_kp).

        Parameters
        ----------
        parameter_value : str
            The parameter value to plot
        iteration : int
            The iteration number to plot the band edges for
        ax_cb : matplotlib.axes.Axes, optional
            Axes object to plot the conduction band edges on. If None, creates a new figure.
        ax_vb : matplotlib.axes.Axes, optional
            Axes object to plot the valence band edges on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot

        Raises
        ------
        ValueError
            If the parameter value is not found in the data or if band edge data is not available
        """
        if ax_cb is None or ax_vb is None:
            fig, (ax_cb, ax_vb) = plt.subplots(1, 2, figsize=(12, 5))

        par_value_float = float(parameter_value)
        if par_value_float not in self.data:
            raise ValueError(
                f"Parameter value {parameter_value} not found in data. "
                f"Available values: {[f'{par:.1e}' for par in sorted(self.data.keys())]}"
            )

        # To plot the band edges as a function of k-point index, we need to extract the band edge values for each k-point from the DOS files.
        dos_energy = self.energies
        dos_file = self._get_parameter_folder(parameter_value) / "outputs" / f"electron_ldos_{iteration}.npy"
        if not dos_file.exists():
            raise ValueError(f"DOS file {dos_file} not found for parameter value {parameter_value} and iteration {iteration}")
        dos_values = np.load(dos_file)  # Shape: (num_energies, num_kpoints_x, num_kpoints_y)
        mid_bandgap = (self.data[par_value_float]["conduction_band"][iteration] + self.data[par_value_float]["valence_band"][iteration]) / 2
        num_kpoints_x = dos_values.shape[1]
        num_kpoints_y = dos_values.shape[2]
        kpoint_indices = [(i, i) for i in range(min(num_kpoints_x, num_kpoints_y))]  # G to K path
        conduction_band_edges = []
        valence_band_edges = []
        for kp in kpoint_indices:
            dos_kp = dos_values[:, kp[0], kp[1]]
            peaks, _ = find_peaks(dos_kp, height=0.01)
            bands = dos_energy[peaks]
            # Find the conduction and valence band edges.
            conduction_band_edge = np.min(bands[bands > mid_bandgap])
            valence_band_edge = np.max(bands[bands < mid_bandgap])
            conduction_band_edges.append(conduction_band_edge)
            valence_band_edges.append(valence_band_edge)
        ax_cb.plot(range(len(kpoint_indices)), conduction_band_edges, "o-")
        ax_cb.set_xlabel("K-point Index (G to K)")
        ax_cb.set_ylabel("Conduction Band Edge (eV)")
        ax_cb.set_title(f"Conduction Band Edge vs K-point \n (Parameter = {parameter_value}, Iteration = {iteration})")
        ax_cb.grid(True)
        ax_vb.plot(range(len(kpoint_indices)), valence_band_edges, "s-")
        ax_vb.set_xlabel("K-point Index (G to K)")
        ax_vb.set_ylabel("Valence Band Edge (eV)")
        ax_vb.set_title(f"Valence Band Edge vs K-point \n (Parameter = {parameter_value}, Iteration = {iteration})")
        ax_vb.grid(True)
        return ax_cb, ax_vb
        

    def get_available_parameters(self) -> List[float]:
        """
        Get a sorted list of all available parameter values.

        Returns
        -------
        list
            Sorted list of parameter values
        """
        return sorted(self.data.keys())

    def get_data_summary(self) -> Dict:
        """
        Get a summary of the extracted data.

        Returns
        -------
        dict
            Summary statistics for each parameter value
        """
        summary = {}
        for param_value, data in sorted(self.data.items()):
            summary[param_value] = {
                key: {
                    "count": len(values),
                    "last_value": values[-1] if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
                for key, values in data.items()
            }
        return summary
