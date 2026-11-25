# Carbon Nanotube

The electronic structure data for this (8, 0) carbon nanotube was
constructed using VASP and transformed into a basis of
maximally-localized Wannier functions using Wannier90.

## Software Versions

- VASP: `vasp.6.3.0 20Jan22 (build Mar 14 2022 17:30:40) complex`
- Wannier90: `Release: 3.1.0        5th March    2020`

## Geometry & Relaxation

An initial geometry can be constructed using `ase` for instance:

```python
import ase.build
import ase.io

cnt = ase.build.nanotube(8, 0, vacuum=10.0)

cnt.rotate("z", "x", rotate_cell=True)
ase.io.write("POSCAR", cnt)
```

## Self-Consistent Field (SCF) Calculation

After relaxing this geometry using VASP, we perform a self-consistent
field (SCF) calculation to obtain the electronic structure of the
system. The following input files are used for this calculation:

<details>
<summary>INCAR</summary>

```INCAR
ENCUT = 550 eV
ALGO = Normal

ISMEAR = 0
SIGMA = 0.05

NELM = 100
NELMIN = 10
EDIFF = 1E-10

GGA = PE
NBANDS = 84

PREC = Accurate
ADDGRID = .TRUE.
```

</details>

<details>
<summary>KPOINTS</summary>

```KPOINTS
21x1x1 kpoint grid
0
Gamma
21 1 1
0 0 0
```

</details>

<details>
<summary>POSCAR</summary>

```POSCAR
 CNT
   1.0000000000000000
    4.2761526107999996    0.0000000000000000     0.0000000000000000
     0.0000000000000000   40.0000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    24.6881122588999986
   C
    32
Direct
  0.0000000000000000  0.1714150000000032  0.4050532456727254
  0.4999821555947719  0.1773974999999979  0.4537771005946922
  0.0000000000000000  0.1944324999999978  0.4950844305883990
  0.4999821555947719  0.2199274999999972  0.5226847587485395
  0.0000000000000000  0.2500000000000000  0.5323776829174847
  0.4999821555947719  0.2800725000000028  0.5226847587485395
  0.0000000000000000  0.3055675000000022  0.4950844305883990
  0.4999821555947719  0.3226025000000021  0.4537771005946922
  0.0000000000000000  0.3285849999999968  0.4050532456727254
  0.4999821555947719  0.3226025000000021  0.3563293907507514
  0.0000000000000000  0.3055675000000022  0.3150220607570446
  0.4999821555947719  0.2800725000000028  0.2874217325969113
  0.0000000000000000  0.2500000000000000  0.2777288084279590
  0.4999821555947719  0.2199274999999972  0.2874217325969113
  0.0000000000000000  0.1944324999999978  0.3150220607570446
  0.4999821555947719  0.1773974999999979  0.3563293907507514
  0.3333370274016758  0.1714150000000032  0.4050532456727254
  0.8333191829964548  0.1773974999999979  0.4537771005946922
  0.3333370274016758  0.1944324999999978  0.4950844305883990
  0.8333191829964548  0.2199274999999972  0.5226847587485395
  0.3333370274016758  0.2500000000000000  0.5323776829174847
  0.8333191829964548  0.2800725000000028  0.5226847587485395
  0.3333370274016758  0.3055675000000022  0.4950844305883990
  0.8333191829964548  0.3226025000000021  0.4537771005946922
  0.3333370274016758  0.3285849999999968  0.4050532456727254
  0.8333191829964548  0.3226025000000021  0.3563293907507514
  0.3333370274016758  0.3055675000000022  0.3150220607570446
  0.8333191829964548  0.2800725000000028  0.2874217325969113
  0.3333370274016758  0.2500000000000000  0.2777288084279590
  0.8333191829964548  0.2199274999999972  0.2874217325969113
  0.3333370274016758  0.1944324999999978  0.3150220607570446
  0.8333191829964548  0.1773974999999979  0.3563293907507514

  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
```

</details>

The calculations were performed with the standard VASP `PAW_PBE` carbon
pseudopotential (`C`, 08Apr2002) obtained from the VASP portal.

Using these input files, the SCF calculation can be run (parallelizing
over the bands) with the following command:

```bash
mpiexec -n 84 vasp
```

At the end of this calculation we find a Fermi energy of -3.8599622677
eV for this system.

## Wannierization

We wannierize the electronic structure of the CNT using Wannier90
through the VASP interface (`LWANNIER90 = .TRUE.`). We use atom-centered
pz orbitals as initial projections for the wannierization. The input
file can be found below. [This
tutorial](https://www.wanniertools.org/tutorials/high-quality-wfs/) can
be helpful to find suitable parameters for the wannierization.

<details>
<summary>wannier90.win</summary>

```wannier90.win
num_bands = 84
num_wann = 32

Begin Projections
C:pz
End Projections

dis_num_iter = 500000
num_iter = 500000

guiding_centres = True

dis_win_min = -12
dis_win_max = 5
dis_froz_min = -4
dis_froz_max = -2

write_hr = True
write_xyz= True
translate_home_cell = True

begin unit_cell_cart
     4.2761526     0.0000000     0.0000000
     0.0000000    40.0000000     0.0000000
     0.0000000     0.0000000    24.6881123
end unit_cell_cart
begin atoms_cart
C        0.0000000     6.8566000    10.0000000
C        2.1380000     7.0959000    11.2029000
C        0.0000000     7.7773000    12.2227000
C        2.1380000     8.7971000    12.9041000
C        0.0000000    10.0000000    13.1434000
C        2.1380000    11.2029000    12.9041000
C        0.0000000    12.2227000    12.2227000
C        2.1380000    12.9041000    11.2029000
C        0.0000000    13.1434000    10.0000000
C        2.1380000    12.9041000     8.7971000
C        0.0000000    12.2227000     7.7773000
C        2.1380000    11.2029000     7.0959000
C        0.0000000    10.0000000     6.8566000
C        2.1380000     8.7971000     7.0959000
C        0.0000000     7.7773000     7.7773000
C        2.1380000     7.0959000     8.7971000
C        1.4254000     6.8566000    10.0000000
C        3.5634000     7.0959000    11.2029000
C        1.4254000     7.7773000    12.2227000
C        3.5634000     8.7971000    12.9041000
C        1.4254000    10.0000000    13.1434000
C        3.5634000    11.2029000    12.9041000
C        1.4254000    12.2227000    12.2227000
C        3.5634000    12.9041000    11.2029000
C        1.4254000    13.1434000    10.0000000
C        3.5634000    12.9041000     8.7971000
C        1.4254000    12.2227000     7.7773000
C        3.5634000    11.2029000     7.0959000
C        1.4254000    10.0000000     6.8566000
C        3.5634000     8.7971000     7.0959000
C        1.4254000     7.7773000     7.7773000
C        3.5634000     7.0959000     8.7971000
end atoms_cart
mp_grid =    21     1     1
begin kpoints
      0.000000000000      0.000000000000      0.000000000000
      0.047619047619      0.000000000000      0.000000000000
      0.095238095238      0.000000000000      0.000000000000
      0.142857142857      0.000000000000      0.000000000000
      0.190476190476      0.000000000000      0.000000000000
      0.238095238095      0.000000000000      0.000000000000
      0.285714285714      0.000000000000      0.000000000000
      0.333333333333      0.000000000000      0.000000000000
      0.380952380952      0.000000000000      0.000000000000
      0.428571428571      0.000000000000      0.000000000000
      0.476190476190      0.000000000000      0.000000000000
     -0.047619047619      0.000000000000      0.000000000000
     -0.095238095238      0.000000000000      0.000000000000
     -0.142857142857      0.000000000000      0.000000000000
     -0.190476190476      0.000000000000      0.000000000000
     -0.238095238095      0.000000000000      0.000000000000
     -0.285714285714      0.000000000000      0.000000000000
     -0.333333333333      0.000000000000      0.000000000000
     -0.380952380952      0.000000000000      0.000000000000
     -0.428571428571      0.000000000000      0.000000000000
     -0.476190476190      0.000000000000      0.000000000000
end kpoints
```

</details>

At the end of the Wannier90 calculation, we find spreads of roughly 3
Å^2 for most of the 32 Wannier functions, which should is satisfactory
for this example, especially given the small number of projections used
for this wannierization. The DFT band structure and the
Wannier-interpolated band structure are in good agreement. The resulting
Hamiltonian, stored in the `wannier90_hr.dat` file, is used to construct
inputs for the transport simulations.

## Constructing Transport Hamiltonian and Structure Files

### Automatic Upscaling from Unit Cell to Transport Hamiltonian

The Wannier90 Hamiltonian can be converted to `quatrex`'s HDF5 format
and used directly in transport calculations, where the upscaling from
unit cell to transport Hamiltonian (`device.construct_from_unit_cell =
true`) is handled by `quatrex` (see the `gw-unit-cell` example).

### Manual Upscaling from Unit Cell to Transport Hamiltonian

Alternatively, the Hamiltonian can be manually converted to a transport
Hamiltonian and the corresponding structure file.

Let's say we want to construct a transport Hamiltonian along the `"a"`
direction that consists of 12 transport cells, and each transport cell
should take into account the two neighboring unit cells. In this case,
we will set the following parameters:

```python
transport_direction = "a"
transport_index = "abc".index(transport_direction)
neighbor_cell_cutoff = (2, 0, 0)
num_transport_cells = 12
```

After loading in the `wannier_centers` and the `lattice_vector` of the
unit cell, you can use `ase` and `quatrex` to construct the upscaled
device `structure.xyz` file for the transport calculation. The following
code snippet shows how to do this:

```python
import ase.io
from quatrex.device.inputs import create_coordinate_grid

num_unit_cells = num_transport_cells * neighbor_cell_cutoff[transport_index]

structure = create_coordinate_grid(
    wannier_centers, num_unit_cells, transport_index, lattice_vectors
)
```

In a similar way, the Wannier90 Hamiltonian can be converted to a
transport Hamiltonian by gluing together the unit cell hopping terms.
First, the hopping terms are cut off according to the
`neighbor_cell_cutoff` parameter.

```python
hamiltonian = {
    r: h_r
    for r, h_r in hamiltonian.items()
    if all(abs(r_i) <= cutoff for r_i, cutoff in zip(r, neighbor_cell_cutoff))
}
```

Then, the hopping terms are upscaled to the transport Hamiltonian using
the `quatrex.device.inputs._expand_tight_binding_matrix` function:

```python
from quatrex.device.inputs import _expand_tight_binding_matrix

device_hamiltonian = _expand_tight_binding_matrix(
    hamiltonian, num_transport_cells, transport_index
)
```

## Bare Coulomb Matrix for GW calculations

For GW calculations, we need to construct the a Coulomb matrix in the
basis of the Wannier functions. While there are more accurate methods
available, one simple way to compute an approximate bare Coulomb matrix
is to assume that the Wannier functions are point charges located at the
Wannier centers. The following code snippet shows how to compute the
bare Coulomb matrix in this approximation:

```python
from scipy.constants import physical_constants

epsilon_0 = physical_constants["electric constant"][0] * 1e-10  # F/Å
e = physical_constants["elementary charge"][0]  # C

d = np.linalg.norm(structure[:, np.newaxis, :] - structure[np.newaxis, :, :], axis=-1)

coulomb_matrix = e / (4 * np.pi * epsilon_0 * d)
np.fill_diagonal(coulomb_matrix, 0)
```

## Phonon Dispersion

The `"deformation-potential"` electron-phonon scattering model requires
the `phonon_dispersion.npy` input file containing the angular velocities of
the phonons. For the required format, see User Guide -> Input Data ->
Phonon Data -> Phonon Dispersion in the documentation.

To obtain the dispersion, one can start from the Force Constants Matrix
$\mathbf{\Phi}_{ij}^{kk'}$, which gives the force on each atom $k$ in
any unit cell $i$ given the atomic displacements $\vec{u}_i^{k}$:

$$
\vec{F}_i^k = m^k \ddot{\vec{u}}_i^k 
= - \sum_{jk'}\mathbf{\Phi}_{ij}^{kk'} \cdot \vec{u}_j^{k'}
$$

Modeling the forces like this corresponds to the harmonic approximation.

In the following we assume a 1D structure that is periodic in one
dimension and confined in the other two. Let $\vec{a}$ label the
displacement between two neighboring unit cells. We will also ignore
interactions extending beyond neighboring unit cells. Then we can define
the dynamical matrix as $$ \mathbf{D}_{\vec{q}}^{kk'} := \left[
\mathbf{\Phi}_{0, -1}^{kk'} e^{-i\vec{q} \cdot \vec{a}} +
\mathbf{\Phi}_{0, 0}^{kk'} + \mathbf{\Phi}_{0, 1}^{kk'} e^{i\vec{q}
\cdot \vec{a}} \right] e^{i\vec{q} \cdot \left( \vec{R}_0^{k'} -
\vec{R}_0^{k} \right)}. $$

The angular velocities $\omega_{\vec{q}\lambda}^2$ and polarizations
$\vec{\epsilon}_{\vec{q}\lambda}$ are now obtained by solving

$$ 
-m^k \omega_{\vec{q}\lambda}^2 \vec{\epsilon}_{\vec{q}\lambda}^k 
= -\sum_{k'} \mathbf{D}_{\vec{q}}^{kk'}
\vec{\epsilon}_{\vec{q}\lambda}^{k'}.
$$

This is a standard eigenvalue problem for each momentum $\vec{q}$ if we
combine the coordinates of all atoms in the unit cell into a single
basis.

After solving for the modes at different phonon momenta, the modes are
not necessarily sorted equally for each momentum. We can obtain the
correct sorting by iterating through the momenta and sorting the modes
such that the overlap with the previous modes is maximized. This can be
done with the help of `scipy.optimize.linear_sum_assignment` for
example.

The following script reads the required parts of the force constants
matrix
(`H00` := $\mathbf{\Phi}_{0, 0}^{kk'}$,
`H10` := $\mathbf{\Phi}_{0,-1}^{kk'}$,
`H01` := $\mathbf{\Phi}_{0, 1}^{kk'}$,
`H` := $\mathbf{D}_{\vec{q}}^{kk'}$)
and computes the corresponding modes, making sure that the sorting is
always the same. It also ensures that the longitudinal acoustic (LA)
mode comes first, followed by the two transverse acoustic (TA) modes.

```python
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Avoiding a dependency on qt


load_path = "/path/to/data/"
save_path = load_path + "output/"


def get_dispersion(qx_a, *, H00, H10, H01):
    """
    Returns:
        omega[mode]
        epsilon[atom/dimension, mode]
    """

    H = H10 * np.exp(-1j * qx_a) + H00 + H01 * np.exp(1j * qx_a)

    result = np.linalg.eig(H)

    omega2 = result.eigenvalues
    if not np.allclose(np.imag(omega2) / np.real(omega2), 0, atol=1e-5):
        print(
            "Complex eigenvalues encountered at qx={qx_a} / a:",
            omega2[
                np.logical_not(
                    np.isclose(np.imag(omega2) / np.real(omega2), 0, atol=1e-5)
                )
            ],
        )
    omega2 = np.real(result.eigenvalues)
    sorting = np.argsort(omega2)
    nonphysical = omega2 < 0
    if np.any(nonphysical):
        print(
            f"Nonphysical values encountered for omega2 at qx={qx_a} / a:",
            omega2[nonphysical],
        )
        omega2[nonphysical] = float("nan")
    omega = np.sqrt(omega2)

    epsilon = result.eigenvectors

    return omega[sorting], epsilon[:, sorting]


# H00/H01/H10 are in a combined basis of the 3 dimensions and the N atoms
# in the unit cell. They are ordered as atom1 x, atom1 y, atom1 z,
# atom2 x, ...
H00 = np.loadtxt(load_path + "H00.dat", delimiter=",")
H01 = np.loadtxt(load_path + "H01.dat", delimiter=",")
H10 = np.loadtxt(load_path + "H10.dat", delimiter=",")
# Choosing an even N_q to avoid qx=0, where degeneracies make it difficult to
# assign the modes.
N_q = 250
hbar = scipy.constants.physical_constants["reduced Planck constant in eV s"][0]

assert N_q % 2 == 0
N_q_computed = N_q // 2
N_modes = H00.shape[0]
N_atoms = N_modes // 3
# qxs_a: momenta multiplied by the lattice constant
qxs_a = np.linspace(-np.pi, np.pi, N_q)

positive_qxs_a = qxs_a[N_q_computed:]
negative_qxs_a = qxs_a[:N_q_computed]
assert np.all(positive_qxs_a > 0)
assert np.all(negative_qxs_a < 0)
assert np.allclose(negative_qxs_a, -np.flip(positive_qxs_a))

# Obtain the full phonon dispersion for the positive momenta
omega = np.zeros((N_modes, N_q))
prev_epsilon = None
for shifted_qx_index, qx_a in enumerate(positive_qxs_a):
    qx_index = shifted_qx_index + N_q_computed
    omega[:, qx_index], epsilon = get_dispersion(qx_a, H00=H00, H10=H10, H01=H01)

    if shifted_qx_index == 0:
        # At the first momentum we determine the acoustic branches.
        # This requires the first momentum to be small and positive.
        # reshaped_epsilon[atom, dimension, mode]
        reshaped_epsilon = np.reshape(epsilon, (N_atoms, 3, N_modes))
        # avg_epsilon[dimension, mode]: Average over all atoms
        avg_epsilon = np.mean(reshaped_epsilon, axis=0)
        epsilon_deviation = np.abs(
            np.expand_dims(avg_epsilon, axis=0) - reshaped_epsilon
        )
        max_epsilon_deviation = np.max(epsilon_deviation, axis=(0, 1))
        acoustic_mode_indices = np.argmax(avg_epsilon, axis=1)
        print("Acoustic modes along x/y/z:", acoustic_mode_indices)
        assert len(np.unique(acoustic_mode_indices)) == 3, "Missing an acoustic mode"

        # Move acoustic mode to the beginning in the correct order (x, y, z, optical)
        order = list(acoustic_mode_indices) + [
            mode_index
            for mode_index in range(N_modes)
            if mode_index not in acoustic_mode_indices
        ]
        assert len(order) == N_modes
        omega[:, qx_index] = omega[order, qx_index]
        epsilon = epsilon[:, order]
        avg_epsilon = avg_epsilon[:, order]
        max_epsilon_deviation = max_epsilon_deviation[order]

        fig, ax = plt.subplots()
        ax.plot(np.real(avg_epsilon.T), "o", label=["x", "y", "z"])
        ax.set_prop_cycle(None)
        ax.plot(np.imag(avg_epsilon.T), "+", label=None)
        ax.plot(hbar * omega[:, qx_index], color="black", label="energy [eV]")
        ax.plot(max_epsilon_deviation, "x", color="black", label="max deviation")
        ax.legend()
        ax.set_xlabel("Mode index")
        ax.set_ylabel("epsilon average")
        fig.savefig(save_path + "polarizations.svg")

    else:
        assert prev_epsilon is not None  # Type narrowing
        # Sort the modes the same as in the previous momentum by finding the
        # eigenvectors that match the best
        # epsilon_overlaps[mode, mode]: overlap between the two modes
        epsilon_overlaps = np.abs(
            np.matmul(np.conj(np.transpose(prev_epsilon)), epsilon)
        )
        # linear_sum_assignement: Unique indices which maximize the total overlap
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            epsilon_overlaps, maximize=True
        )
        matching_modes = col_ind
        omega[:, qx_index] = omega[matching_modes, qx_index]
        epsilon = epsilon[:, matching_modes]

    prev_epsilon = epsilon

# Infer omega at negative momenta from symmetry
omega[:, :N_q_computed] = np.flip(omega[:, N_q_computed:], axis=1)

fig, ax = plt.subplots()
ax.plot(qxs_a / np.pi, hbar * omega.T)
ax.set_xlabel("qx [pi/a]")
ax.set_ylabel("E [eV]")
fig.savefig(save_path + "dispersion.svg")

with open(save_path + "phonon_dispersion.npy", "wb") as f:
    np.save(f, omega)
```

## `negf-gw-poisson`: Gated carbon nanotube with NEGF-GW and Poisson

Self-consistent Schrödinger-Poisson (SCSP) simulation with GW and
pseudo-phonon scattering, in the same spirit as the `negf-poisson`
example of graphene, which is ballistic. The tube (24 unit cells,
102.8 Å) is embedded in SiO2, with n-doped leads (2e19 cm$^{-3}$, eight
unit cells each), a gate-all-around of radius 12 Å and length 26 Å at
$V_g = -0.3$ V, and a drain bias of 0.1 V at the right contact.

```bash
mpirun -n 64 quatrex run quatrex_config.toml
```

With 64 ranks (one thread each) on an AMD EPYC 7742 node, this takes
about 40 min. The `outputs` need about 2 GB. The inputs are links to
`../inputs`, except for the mesh, which belongs to this device and can be
regenerated with `quatrex mesh quatrex_config.toml --off-screen`.

### Contact levels

GW renormalizes the bands, but band edge tracking is turned off in SCSP
runs. The Fermi levels, mid-gap energies and band edges of the contacts
are therefore set to those of the homogeneous doped tube with GW:

| | without GW | with GW |
|---|---|---|
| Valence band edge (eV) | -4.174 | -3.735 |
| Conduction band edge (eV) | -3.566 | -3.080 |
| `mid_gap_energy` (eV) | -3.870 | -3.41 |
| `fermi_level` (eV) for 2e19 cm$^{-3}$ | -3.422 | -2.896 |

The band edges with GW were read off the spectral function (25 meV grid)
of a standalone SCBA run (no `[scsp]`, zero potential,
`band_edge_tracking = true`, 30 iterations). The Fermi level was obtained
from standalone SCBA runs with the settings of this example and fixed Fermi
levels (45 iterations), by interpolating the charge above mid-gap to the
doping density:

| `fermi_level` (eV) | -2.902 | -2.896 (interpolated) | -2.887 | -2.873 |
|---|---|---|---|---|
| Electrons per unit cell | 0.206 | 0.214 | 0.225 | 0.245 |

Note that the Fermi level must not be shifted rigidly on the spectral
function of a run with a different charge, as the gap with GW depends on
the charge.

**The levels depend on the doping, on `epsilon_r`, on the cutoffs and on
the energy grid**, and have to be recalibrated if any of these change.
As a check, the potential in the source should be close to zero (here
-0.01 V) and that in the drain close to minus the drain bias (-0.11 V).

### Settings

- **Interleaved loops.** `scba_iterations_per_step = 2` runs two SCBA
  iterations per potential, starting from the previous self-energies.
- **Dielectric.** `default_epsilon_r = 3.9` everywhere, close to the
  `epsilon_r = 4` of the GW Coulomb matrix. The `oxide` region between the
  tube and the gate (radius 7.5 to 12 Å) has the same permittivity, so it
  does not change the results. It is there so that its `epsilon_r` can be
  varied without a new mesh. With 11.7 (as in
  `qtbm-poisson`) the screening length of the lead is comparable to
  its length, so that it is not neutral. With 1.0, the point-charge
  dopants cause a variation of about 0.4 V within a unit cell.
- **5 meV energy grid.** The gate raises the valence band in the channel
  above that of the leads, which binds hole states there (at about
  -3.64 eV, with a full width at half maximum of about 5 meV). They are
  narrower than a 25 meV grid, on which the self-energies and potential
  spike unless a large `eta` is used. See below.
- **No `eta` and `eta_obc`.** Both are left at their defaults. An `eta`
  of 2e-3 eV overestimates the current by more than 10 %.
- **`scba.convergence_tol = 0.5`.** The maximum self-energy update levels
  off at 0.05 - 0.35 eV, due to valence states below -4.9 eV. The
  tolerance was chosen above this floor, so in practice only the potential
  determines when the loop stops. Between the last two SCBA iterations, the
  spectral electron density above mid-gap changes by at most 2e-4 per eV
  and orbital.

### Saving the Green's functions and self-energies

With `scba_matrices = ["g", "p", "w", "sigma"]` in `[outputs]`, the
Green's functions, polarizations, screened interactions and self-energies
of the last SCBA iteration are written to HDF5 files at the end of the
run. Individual matrices can be selected as well, e.g. `["g_lesser"]`.
All of them need 27 GB.

### Results and limitations

The loop converges in 24 iterations, with a barrier (channel maximum minus
source) of 0.31 V. Ballistic QTBM with the bare levels gives 0.30 V (its
potential update levels off at 1.9 mV).

The dependence on the grid and on `eta` was determined with
`fermi_level = -2.887` and otherwise identical settings. The currents
(arbitrary units) are averages over the last 20 SCBA iterations.

| Grid | 25 meV | 25 meV | 5 meV | 5 meV | 1 meV |
|---|---|---|---|---|---|
| `eta` (eV) | 2e-3 (`eta_obc` 5e-3) | 1e-4 | 1e-4 | default | 1e-4 |
| Iterations to convergence (2 mV) | 23 | 41 (spikes) | 23 | 23 | 23 |
| Barrier (V) | 0.308 | 0.304 | 0.304 | 0.303 | 0.304 |
| Largest difference of the potential to 1 meV (mV) | 6.3 | 5.9 | 0.4 | 0.6 | - |
| Current | 1.97e-3 | 1.75e-3 | 1.71e-3 | 1.70e-3 | 1.71e-3 |
| Standard deviation of the current (%) | 1.8 | 4.7 | 0.7 | 1.8 | 1.1 |
| Spectral weight per orbital (exact: 2) | 1.945 | 1.993 | 1.993 | 1.996 | 1.993 |

The run time on 64 ranks is 12 min (25 meV, `eta = 1e-4`) and 34 min
(5 meV), and 2.8 h on 128 ranks with 1 meV.

- **The current fluctuates** from one SCBA iteration to the next by
  1 - 2 % (standard deviation), up to 5 % between the extremes of the last
  20 iterations. Average over the last iterations.
- **Transient negative densities.** In single SCBA iterations (3 of 48), the
  spectral electron density gets negative (down to -18 per eV) at deep
  valence states in the channel, between -5.0 and -5.4 eV, and recovers
  in one or two iterations. No such values occur above mid-gap, the
  potential updates stay smooth, and the last iteration has none. The cause
  is not known. They also occur on the 25 meV grid, and, in one run with
  `eta = 1e-4` on the 5 meV grid, were weaker (one iteration, -1.6).
- **The potential is converged to about 10 mV**, not to the 2 mV tolerance:
  the updates shrink by only 17 % per iteration (estimated from the 1 meV
  run).
- The current is sensitive to the contact levels: 9 meV in the Fermi level
  (-2.896 instead of -2.887) change it by about 9 %.
- The current differs by 0.2 % between the contacts, and by 0.3 % between
  the source contact and the middle of the device.
- **The leads are not neutral**: they hold 0.17 - 0.18 electrons per unit
  cell, compared to 0.214 dopants. This does not depend on `eta`,
  `eta_obc` or the energy grid, and hardly on the Fermi level. The potential
  varies by 0.22 V between the atoms of a unit cell in the leads, and the
  lowest subband is about 75 meV higher than expected from the flat tube.
  Whether this, or the length of the leads, causes the missing charge has
  not been determined.
