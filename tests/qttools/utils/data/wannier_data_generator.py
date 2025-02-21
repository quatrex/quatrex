from qttools import NDArray, xp


def save_as_wannierhr(save_path: str, R: NDArray, MR: NDArray) -> NDArray:
    """ "
    Saves a matrix M in a wannier90_hr.dat format.

    Parameters
    ----------
    save_path : str
        Path to save the file
    R : ndarray
        Lattice vectors pointing to shifted unit cells
    MR : ndarray
        Wannier Hamiltonian or Coulomb matrix
    """
    # datetime object containing current date and time
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y at %H:%M:%S")

    header = "Created on " + dt_string + "\n"
    num_wann = MR[0].shape[0]
    nrpts = len(R)  # Number of Wigner-Seitz lattice vectors
    header += f"       {num_wann}\n       {nrpts}"
    for i in range(nrpts):
        if i % 15 == 0:
            header += "\n"
        header += "    1"
    header += "\n"

    rows = xp.tile(xp.arange(num_wann), num_wann) + 1
    cols = xp.repeat(xp.arange(num_wann), num_wann) + 1

    with open(save_path, "w") as f:
        f.write(header)
        for i, r in enumerate(R):
            line_R = list(r)
            interactions = MR[i]
            for i, interaction in enumerate(interactions.T.ravel()):
                line = line_R + [
                    rows[i],
                    cols[i],
                    xp.real(interaction),
                    xp.imag(interaction),
                ]
                f.write(
                    "{:> 5.0f}{:> 5.0f}{:> 5.0f}{:> 5.0f}{:> 5.0f}{:> 12.6f}{:> 12.6f}\n".format(
                        *line
                    )
                )


def main():
    R = xp.array(
        [
            [-2, -2, 0],
            [-2, -1, 0],
            [-2, 0, 0],
            [-2, 1, 0],
            [-2, 2, 0],
            [-1, -2, 0],
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [-1, 2, 0],
            [0, -2, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
            [1, -2, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 2, 0],
            [2, -2, 0],
            [2, -1, 0],
            [2, 0, 0],
            [2, 1, 0],
            [2, 2, 0],
        ]
    )
    MR = xp.ones((25, 10, 10), dtype=xp.complex128)
    for i in range(25):
        MR[i] = R[i][0] + 1j * R[i][1]
    save_as_wannierhr("wannier90_hr.dat", R, MR)
    xp.savetxt("R_ref.txt", R, fmt="%2d")


if __name__ == "__main__":
    main()
