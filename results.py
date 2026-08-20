import numpy as np
import matplotlib.pyplot as plt
import os
import tomllib

spin = True

config = "examples/w90/carbon-nanotube/gw/quatrex_config.toml"

with open(config, "rb") as f:
    data = tomllib.load(f)

OUTPUT_DIR = "plots_temp/spin/" if spin is True else "plots_temp/no-spin/"
INPUT_DIR = "/home/sem26f56/MSDocuments/quatrex/examples/w90/carbon-nanotube/gw/outputs/"

energy_window_min = data["electron"]["energy_window_min"]
energy_window_max = data["electron"]["energy_window_max"]
energy_window_num = data["electron"]["energy_window_num"]

"""
quantity q choose from:

"device_current", "w_lesser_density", "electron_density", "electron_ldos", "meir_wingreen_current", etc.
"""

quantities = ["electron_density", "device_current", "w_lesser_density", "electron_ldos", "meir_wingreen_current"]

for q in quantities:

    if not os.path.isdir(OUTPUT_DIR + q):
        os.mkdir(OUTPUT_DIR + q)

    it = np.arange(0,14,1)

    for x in os.listdir(INPUT_DIR):
        if x.endswith(".npy") and q in x:
            it = x.removeprefix(q).removesuffix(".npy")

            data = np.load(INPUT_DIR + x)
            w, h = data.shape
            if data.dtype == 'complex128':
                data = np.real(data)

            fig, ax = plt.subplots(figsize = (10,10))
            im = ax.imshow(data, interpolation = "bilinear", cmap = "coolwarm", aspect=h/w,
                        origin = 'lower')
            fig.colorbar(im, ax = ax)

            fig.tight_layout()
            plt.savefig(OUTPUT_DIR + q + '/' + q + it)
            plt.close()

            if "electron_ldos" in q:

                if not os.path.isdir(OUTPUT_DIR + "electron_dos"):
                    os.mkdir(OUTPUT_DIR + "electron_dos")
                
                E, N_AO = data.shape
                dos = np.zeros(E)
                dos[:] = np.sum(data, axis = 1)
                E = np.linspace(energy_window_min, energy_window_max, energy_window_num)

                fig,ax = plt.subplots(figsize = (10, 10))
                ax.plot(E, dos)

                plt.xlabel("Energy [eV]")
                plt.ylabel("DOS[(eV)^-1]")
                plt.savefig(OUTPUT_DIR + 'electron_dos/' + 'electron_dos' + it)
                plt.close()


