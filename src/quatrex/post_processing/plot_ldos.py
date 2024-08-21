import matplotlib.pyplot as plt
import numpy as np


def get_averages(
    quantity: np.ndarray,
    num_energies: int,
    num_cells: int,
    method: str = "rolling",
):
    cell_quantity = quantity.reshape(num_energies, num_cells, -1)
    if method == "cell-average":
        return cell_quantity.mean(axis=-1)
    if method == "rolling":
        kernel = np.ones(cell_quantity.shape[-1]) / cell_quantity.shape[-1]
        return np.array([np.convolve(q, kernel, mode="valid") for q in quantity])
    raise ValueError(f"Unknown averaging method: {method}")


def spectral_plot(
    ax: plt.Axes,
    averages: np.ndarray, 
    energies: np.ndarray,
    colorbar_label: str | None = None,
    **pcolormesh_kwargs: dict,
):
    qm = ax.pcolormesh(
        np.arange(averages.shape[1]), energies, averages, **pcolormesh_kwargs
    )
    ax.set_ylabel("Energy (eV)")
    plt.gcf().colorbar(qm, ax=ax, label=colorbar_label)
