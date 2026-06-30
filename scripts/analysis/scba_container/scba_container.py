import os
import matplotlib.pyplot as plt
import numpy as np

flag = "hello world"


# data class to hold data across iterations
class SCBAContainer:
    def __init__(
        self,
        max_iterations: int,
        energy_window_min: int,
        energy_window_max: int,
        energy_window_num: int,
        num_samples: int,
        config=None,
    ):
        self.g_lesser = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.g_greater = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.g_retarded = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )

        self.p_lesser = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.p_greater = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.p_retarded = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )

        self.w_lesser = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.w_greater = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )

        self.sigma_lesser = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.sigma_greater = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )
        self.sigma_retarded = np.empty(
            [max_iterations, energy_window_num, num_samples], dtype=np.complex128
        )

        self.sample_indices = None

        self.energies = np.linspace(
            energy_window_min, energy_window_max, energy_window_num, endpoint=True
        )

        self.adaptive_electron_energies_for_g_sigma = None
        self.adaptive_electron_energies_for_p_w = None

    def load_g_data(self, archive_file_prefix: str, iteration: int):
        self.g_lesser[iteration, :, :] = np.load(
            f"{archive_file_prefix}_g_lesser_iter{iteration:02}.npy"
        )
        self.g_greater[iteration, :, :] = np.load(
            f"{archive_file_prefix}_g_greater_iter{iteration:02}.npy"
        )
        self.g_retarded[iteration, :, :] = np.load(
            f"{archive_file_prefix}_g_retarded_iter{iteration:02}.npy"
        )

    def load_p_data(self, archive_file_prefix: str, iteration: int):
        self.p_lesser[iteration, :, :] = np.load(
            f"{archive_file_prefix}_p_lesser_iter{iteration:02}.npy"
        )
        self.p_greater[iteration, :, :] = np.load(
            f"{archive_file_prefix}_p_greater_iter{iteration:02}.npy"
        )
        self.p_retarded[iteration, :, :] = np.load(
            f"{archive_file_prefix}_p_retarded_hermitian_iter{iteration:02}.npy"
        )

    def load_w_data(self, archive_file_prefix: str, iteration: int):
        self.w_lesser[iteration, :, :] = np.load(
            f"{archive_file_prefix}_w_lesser_iter{iteration:02}.npy"
        )
        self.w_greater[iteration, :, :] = np.load(
            f"{archive_file_prefix}_w_greater_iter{iteration:02}.npy"
        )

    def load_sigma_data(self, archive_file_prefix: str, iteration: int):
        self.sigma_lesser[iteration, :, :] = np.load(
            f"{archive_file_prefix}_sigma_lesser_iter{iteration:02}.npy"
        )
        self.sigma_greater[iteration, :, :] = np.load(
            f"{archive_file_prefix}_sigma_greater_iter{iteration:02}.npy"
        )
        self.sigma_retarded[iteration, :, :] = np.load(
            f"{archive_file_prefix}_sigma_retarded_hermitian_iter{iteration:02}.npy"
        )

    def load_sample_indices(self, archive_file_prefix: str):
        if os.path.exists(f"{archive_file_prefix}_sample_indices.npy"):
            self.sample_indices = np.load(f"{archive_file_prefix}_sample_indices.npy")
        else:
            print(
                f"Warning: sample indices file {archive_file_prefix}_sample_indices.npy not found. Sample indices will be None."
            )

    def load_adaptive_grids(self, data_dir: str):
        if os.path.exists(f"{data_dir}/adaptive_electron_energies_for_g_sigma.npy"):
            self.adaptive_electron_energies_for_g_sigma = np.load(
                f"{data_dir}/adaptive_electron_energies_for_g_sigma.npy"
            )
        else:
            print(
                f"Warning: adaptive electron energies for g and sigma file {data_dir}/adaptive_electron_energies_for_g_sigma.npy not found. This data will be None."
            )
        
        if os.path.exists(f"{data_dir}/adaptive_electron_energies_for_p_w.npy"):
            self.adaptive_electron_energies_for_p_w = np.load(
                f"{data_dir}/adaptive_electron_energies_for_p_w.npy"
            )
        else:
            print(
                f"Warning: adaptive electron energies for p and w file {data_dir}/adaptive_electron_energies_for_p_w.npy not found. This data will be None."
            )

    def plot_iteration(
        self,
        axs,
        iteration,
        idx,
        adaptive_start_iteration=100,
        alpha=1.0,
        colorReal=None,
        colorImag=None,
        linewidthReal=1,
        linewidthImag=1,
        markersize=1
    ):
        """expect 4,3 subplot axs
        axs: np.ndarray
        iteration: int
        idx: int -- straight index of the data to plot
        """

        if iteration < 0 or iteration >= self.g_lesser.shape[0]:
            raise ValueError(
                f"Iteration {iteration} is out of bounds. Must be between 0 and {self.g_lesser.shape[0]-1}."
            )
        if idx < 0 or idx >= self.g_lesser.shape[2]:
            raise ValueError(
                f"NNZ Index {idx} is out of bounds. Must be between 0 and {self.g_lesser.shape[2]-1}."
            )

        if colorReal is None:
            colorReal = "tab:blue"

        if colorImag is None:
            colorImag = "tab:orange"

        conv_energies = np.linspace(0, max(self.energies) - min(self.energies), len(self.energies))

        # if there's a lot of points, make the makersize smaller
        if len(self.energies) > 10000:
            markersize = 1
        elif len(self.energies) > 1000:
            markersize = 3
        else:
            markersize = 5

        # uniform grid
        if iteration < adaptive_start_iteration or self.adaptive_electron_energies_for_g_sigma is None:
            x_axis = self.energies
            linestyle = "-"
            title_suffix = ""
        # adaptive grid
        else:
            x_axis = self.adaptive_electron_energies_for_g_sigma
            linestyle = "."
            title_suffix = " (adaptive grid)"
        
        axs[0, 0].set_title(f"G Lesser{title_suffix}")
        axs[0, 0].plot(
            x_axis,
            np.real(self.g_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[0, 0].plot(
            x_axis,
            np.imag(self.g_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[0, 0].grid()
        axs[0, 0].legend()

        axs[0, 1].set_title(f"G Greater{title_suffix}")
        axs[0, 1].plot(
            x_axis,
            np.real(self.g_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[0, 1].plot(
            x_axis,
            np.imag(self.g_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[0, 1].grid()
        axs[0, 1].legend()

        axs[0, 2].set_title(f"G Retarded{title_suffix}")
        axs[0, 2].plot(
            x_axis,
            np.real(self.g_retarded[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[0, 2].plot(
            x_axis,
            np.imag(self.g_retarded[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[0, 2].grid()
        axs[0, 2].legend()

        # uniform grid
        if iteration < adaptive_start_iteration or self.adaptive_electron_energies_for_p_w is None:
            conv_energies = np.linspace(0, max(self.energies) - min(self.energies), len(self.energies))
            linestyle = "-"
            title_suffix = ""
        # adaptive grid
        else:
            conv_energies = self.adaptive_electron_energies_for_p_w
            linestyle = "."
            title_suffix = " (adaptive grid)"

        axs[1, 0].set_title(f"P Lesser{title_suffix}")
        axs[1, 0].plot(
            conv_energies,
            np.real(self.p_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[1, 0].plot(
            conv_energies,
            np.imag(self.p_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[1, 0].grid()
        axs[1, 0].legend()

        axs[1, 1].set_title(f"P Greater{title_suffix}")
        axs[1, 1].plot(
            conv_energies,
            np.real(self.p_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[1, 1].plot(
            conv_energies,
            np.imag(self.p_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[1, 1].grid()
        axs[1, 1].legend()

        axs[1, 2].set_title(f"P Retarded{title_suffix}")
        axs[1, 2].plot(
            conv_energies,
            np.real(self.p_retarded[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[1, 2].plot(
            conv_energies,
            np.imag(self.p_retarded[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[1, 2].grid()
        axs[1, 2].legend()

        if iteration < adaptive_start_iteration or self.adaptive_electron_energies_for_p_w is None:
            conv_energies = np.linspace(0, max(self.energies) - min(self.energies), len(self.energies))
            linestyle = "-"
            title_suffix = ""
        else:
            conv_energies = self.adaptive_electron_energies_for_p_w
            linestyle = "."
            title_suffix = " (adaptive grid)"

        axs[2, 0].set_title(f"W Lesser{title_suffix}")
        axs[2, 0].plot(
            conv_energies,
            np.real(self.w_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[2, 0].plot(
            conv_energies,
            np.imag(self.w_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[2, 0].grid()
        axs[2, 0].legend()

        axs[2, 1].set_title(f"W Greater{title_suffix}")
        axs[2, 1].plot(
            conv_energies,
            np.real(self.w_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[2, 1].plot(
            conv_energies,
            np.imag(self.w_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[2, 1].grid()
        axs[2, 1].legend()

        if iteration < adaptive_start_iteration or self.adaptive_electron_energies_for_g_sigma is None:
            x_axis = self.energies
            linestyle = "-"
            title_suffix = ""
        else:
            x_axis = self.adaptive_electron_energies_for_g_sigma
            linestyle = "."
            title_suffix = " (adaptive grid)"

        axs[3, 0].set_title(f"Sigma Lesser{title_suffix}")
        axs[3, 0].plot(
            x_axis,
            np.real(self.sigma_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[3, 0].plot(
            x_axis,
            np.imag(self.sigma_lesser[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[3, 0].grid()
        axs[3, 0].legend()

        axs[3, 1].set_title(f"Sigma Greater{title_suffix}")
        axs[3, 1].plot(
            x_axis,
            np.real(self.sigma_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[3, 1].plot(
            x_axis,
            np.imag(self.sigma_greater[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[3, 1].grid()
        axs[3, 1].legend()

        axs[3, 2].set_title(f"Sigma Retarded{title_suffix}")
        axs[3, 2].plot(
            x_axis,
            np.real(self.sigma_retarded[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[3, 2].plot(
            x_axis,
            np.imag(self.sigma_retarded[iteration, :, idx]),
            linestyle,
            markersize=markersize,
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[3, 2].grid()
        axs[3, 2].legend()
