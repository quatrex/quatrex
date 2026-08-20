from qttools.utils.hdf5_utils import save_hdf5_dict
import numpy as np

def make_dict(matrix):
    return {"[0, 0, 0]": matrix}


def scale(matrix: np.array, num_blocks: int = 24) -> np.array:
    _, _, z, d, _ = matrix.shape

    scaled_matrix = np.zeros(shape = (d * num_blocks, d * num_blocks))

    blocks = np.zeros(shape = [int(z / 2) + 1, d, d])

    num_off_diag, _, _ = blocks.shape
    for i in range(num_off_diag):
        blocks[i] = matrix[0,0,i, :, :]

    for i in range(num_blocks):
        for j in range(num_blocks):

            if abs(i -j) >= 4:
                continue

            row_range = np.arange(i * d, (i + 1) * d)
            col_range = np.arange(j * d, (j + 1) * d)
            grid = np.meshgrid(row_range, col_range)
            idx_rows = grid[0].flatten()
            idx_cols = grid[1].flatten()

            if j >= i:
                scaled_matrix[idx_rows, idx_cols] = blocks[j - i].flatten()
            else:
                scaled_matrix[idx_rows, idx_cols] = np.transpose(blocks[i-j]).flatten()

    return scaled_matrix



DIR_coulomb = "/usr/scratch/mont-fort11/sem26f56/coulomb_matrix/CNT/outputs/coulomb_matrix.npy"
DIR_exchange = "/usr/scratch/mont-fort11/sem26f56/coulomb_matrix/CNT/outputs/exchange_matrix.npy"

output = "v_p" # or "v_ap"

coulomb_matrix = scale(np.load(DIR_coulomb))
exchange_matrix = scale(np.load(DIR_exchange))
matrix = np.zeros(shape = coulomb_matrix.shape)

match output:
    case "v_p":
        matrix = np.subtract(coulomb_matrix, exchange_matrix)
    case "v_ap":
        matrix = coulomb_matrix

save_hdf5_dict("/home/sem26f56/MSDocuments/quatrex/hdf5/" + output + ".h5", make_dict(matrix))

