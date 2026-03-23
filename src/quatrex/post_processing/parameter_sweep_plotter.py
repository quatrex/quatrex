"""Parameter sweep plotter for analyzing SLURM simulation outputs."""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
        dos_file = (
            self.root_folder
            / self.parameter_pattern.replace("*", parameter_value)
            / "outputs"
            / f"electron_ldos_{iteration}.npy"
        )
        # Have first iteration as reference
        dos_ref_file = (
            self.root_folder
            / self.parameter_pattern.replace("*", parameter_value)
            / "outputs"
            / f"electron_ldos_0.npy"
        )
        if kp is None:
            # Sum over k-points
            dos_values = np.load(dos_file).mean(axis=(-2, -1))
            dos_ref = np.load(dos_ref_file).mean(axis=(-2, -1))
        else:
            dos_values = np.load(dos_file)[:, kp[0], kp[1]]
            dos_ref = np.load(dos_ref_file)[:, kp[0], kp[1]]

        ax.plot(dos_energy, dos_ref, label="Reference (Iteration 0)", color="lightgray", alpha=0.7)
        ax.plot(dos_energy, dos_values, label=f"Iteration {iteration}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Density of States")
        ax.set_title(f"Density of States (Parameter = {parameter_value})")
        ax.grid(True)

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
        folder = self.root_folder / self.parameter_pattern.replace("*", parameter_value) / "outputs"
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
