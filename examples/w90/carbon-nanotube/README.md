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
