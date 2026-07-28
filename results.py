import numpy as np
import matplotlib.pyplot as plt

spin = True

input = "outputs/"
out = "spin/" if spin is True else "no_spin/"
DIR = "/home/sem26f56/MSDocuments/quatrex/examples/w90/carbon-nanotube/gw-dist/" + input
q = "electron_ldos" + "_"
it = np.arange(1,11,1)

for _it in it:
    data = np.load(DIR + q + str(_it) + ".npy")
    x, y = data.shape
    if data.dtype == 'complex128':
        data = np.abs(data)

    fig, ax = plt.subplots(figsize = (10,10))
    im = ax.imshow(data, interpolation = "bilinear", cmap = "hot", aspect=y/x,
                origin = 'lower')
    fig.colorbar(im, ax = ax)

    fig.tight_layout()
    plt.savefig('plots/' + out + q + str(_it))



