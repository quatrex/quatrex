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

    def load_sample_indices(self, archive_file_prefix: str):
        self.sample_indices = np.load(f"{archive_file_prefix}_sample_indices.npy")

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
            f"{archive_file_prefix}_p_retarded_iter{iteration:02}.npy"
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
            f"{archive_file_prefix}_sigma_retarded_iter{iteration:02}.npy"
        )

    def plot_iteration(
        self,
        axs,
        iteration,
        idx,
        alpha=1.0,
        colorReal=None,
        colorImag=None,
        linewidthReal=1,
        linewidthImag=1,
    ):
        """expect 4,3 subplot axs
        axs: np.ndarray
        iteration: int
        idx: int -- straight index of the data to plot
        """

        if colorReal is None:
            colorReal = "tab:blue"

        if colorImag is None:
            colorImag = "tab:orange"

        conv_energies = np.linspace(0, max(self.energies) - min(self.energies), len(self.energies))

        axs[0, 0].set_title("G Lesser")
        axs[0, 0].plot(
            self.energies,
            np.real(self.g_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[0, 0].plot(
            self.energies,
            np.imag(self.g_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[0, 0].grid()
        axs[0, 0].legend()

        axs[0, 1].set_title("G Greater")
        axs[0, 1].plot(
            self.energies,
            np.real(self.g_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[0, 1].plot(
            self.energies,
            np.imag(self.g_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[0, 1].grid()
        axs[0, 1].legend()

        axs[0, 2].set_title("G Retarded")
        axs[0, 2].plot(
            self.energies,
            np.real(self.g_retarded[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[0, 2].plot(
            self.energies,
            np.imag(self.g_retarded[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[0, 2].grid()
        axs[0, 2].legend()

        axs[1, 0].set_title("P Lesser")
        axs[1, 0].plot(
            conv_energies,
            np.real(self.p_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[1, 0].plot(
            conv_energies,
            np.imag(self.p_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[1, 0].grid()
        axs[1, 0].legend()

        axs[1, 1].set_title("P Greater")
        axs[1, 1].plot(
            conv_energies,
            np.real(self.p_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[1, 1].plot(
            conv_energies,
            np.imag(self.p_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[1, 1].grid()
        axs[1, 1].legend()

        axs[1, 2].set_title("P Retarded")
        axs[1, 2].plot(
            conv_energies,
            np.real(self.p_retarded[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[1, 2].plot(
            conv_energies,
            np.imag(self.p_retarded[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[1, 2].grid()
        axs[1, 2].legend()

        axs[2, 0].set_title("W Lesser")
        axs[2, 0].plot(
            conv_energies,
            np.real(self.w_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[2, 0].plot(
            conv_energies,
            np.imag(self.w_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[2, 0].grid()
        axs[2, 0].legend()

        axs[2, 1].set_title("W Greater")
        axs[2, 1].plot(
            conv_energies,
            np.real(self.w_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[2, 1].plot(
            conv_energies,
            np.imag(self.w_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[2, 1].grid()
        axs[2, 1].legend()

        axs[3, 0].set_title("Sigma Lesser")
        axs[3, 0].plot(
            self.energies,
            np.real(self.sigma_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[3, 0].plot(
            self.energies,
            np.imag(self.sigma_lesser[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[3, 0].grid()
        axs[3, 0].legend()

        axs[3, 1].set_title("Sigma Greater")
        axs[3, 1].plot(
            self.energies,
            np.real(self.sigma_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[3, 1].plot(
            self.energies,
            np.imag(self.sigma_greater[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[3, 1].grid()
        axs[3, 1].legend()

        axs[3, 2].set_title("Sigma Retarded")
        axs[3, 2].plot(
            self.energies,
            np.real(self.sigma_retarded[iteration, :, idx]),
            alpha=alpha,
            color=colorReal,
            linewidth=linewidthReal,
            label=f"real Iter {iteration}",
        )
        axs[3, 2].plot(
            self.energies,
            np.imag(self.sigma_retarded[iteration, :, idx]),
            alpha=alpha,
            color=colorImag,
            linewidth=linewidthImag,
            label=f"imag Iter {iteration}",
        )
        axs[3, 2].grid()
        axs[3, 2].legend()
