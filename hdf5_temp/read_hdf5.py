from qttools.utils.hdf5_utils import load_hdf5_dict
import _pickle as pickle
from matplotlib import pyplot as plt




# DIR = "/home/sem26f56/MSDocuments/quatrex/examples/w90/carbon-nanotube/inputs/coulomb_matrix.h5"
DIR = "/home/sem26f56/MSDocuments/quatrex/examples/w90/carbon-nanotube/inputs/coulomb_matrix.h5"
data = load_hdf5_dict(DIR)

coulomb_matrix = data["[0, 0, 0]"]

fig, ax = plt.subplots()
im = plt.imshow(coulomb_matrix, cmap = "coolwarm", interpolation= "bilinear")
fig.colorbar(im, ax = ax)
fig.savefig("hdf5/test.png")
