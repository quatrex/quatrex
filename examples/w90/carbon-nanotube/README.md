# Carbon Nanotube

The electronic structure data for this (8, 0) carbon nanotube was
constructed using VASP and transformed into a basis of
maximally-localized Wannier functions using Wannier90.

## Software Versions

- VASP: `vasp.6.6.0 06Mar2026 (build Aug  4 2026 15:38:09) complex`
- Wannier90: `4.0.1`

## Geometry & Relaxation

An initial geometry can be constructed using `ase` for instance. The
script below generates the `POSCAR` input file for VASP as well as a
`projections.txt` file that will be used in the wannierization.

<details>
<summary>generate_structure.py</summary>

```python
#!/usr/bin/env python

import ase.build
import ase.geometry
import ase.io
import numpy as np


def get_coords_string(coords: np.ndarray):
    assert coords.shape == (3,)
    string = ""
    for coord in coords:
        string += f"{coord:.7f},"

    return string[:-1]  # Remove the final comma


# Parameters
cnt_type = (8, 0)  # Type of the CNT in the (n, m) notation
spacing = 50  # Length of the transverse lattice vectors in Å
bond_length = 1.42  # Å
periodic_dim = 0  # Dimension in which the CNT extends infinitely: 0, 1 or 2

# Construct CNT
cnt = ase.build.nanotube(*cnt_type, bond=bond_length)

# Apply a permutation to orient the CNT along `periodic_dim`
permutation = [0, 1, 2]
# By default, the CNT is oriented along the third dimension.
initial_periodic_dim = 2
permutation[periodic_dim], permutation[initial_periodic_dim] = (
    permutation[initial_periodic_dim],
    permutation[periodic_dim],
)
cnt = ase.geometry.permute_axes(cnt, permutation)

# Obtain the radial coordinates before translating
radial_coordinates = np.copy(cnt.positions)
radial_coordinates[:, periodic_dim] = 0

# Apply spacing
cell = cnt.cell.array
for dim in range(3):
    if dim == periodic_dim:
        continue
    cell[dim, dim] = spacing
cnt.set_cell(cell, scale_atoms=False)
translation = np.full(3, spacing / 2)
translation[periodic_dim] = 0
cnt.translate(translation)

# Move to the middle of the unit cell in the periodic direction to avoid
# atoms on the boundary
cnt.center(axis=periodic_dim)

# Obtain the bond centers
cnt_neighbor_list = ase.neighborlist.neighbor_list(
    "ijD", cnt, cutoff=bond_length * 1.01
)
bond_centers = []
for i, j, D in zip(*cnt_neighbor_list):
    if j > i:
        # Avoid double-counting without removing bonds to the same atom
        # in different unit cells. In newer versions of ase one could
        # probably handle this with the `bothways` keyword in
        # `ase.neighborlist.neighbor_list()`.
        continue
    bond_center = cnt.positions[i, :] + D / 2
    # Wrap the center into the unit cell
    bond_center = ase.geometry.wrap_positions(
        positions=bond_center.reshape(1, 3), cell=cnt.cell.array, pbc=cnt.pbc
    ).reshape(3)
    bond_centers.append(bond_center)

# Construct the projections block (radial pz orbitals on the atoms and s
# orbitals on the bonds) in the .win file
projections_string = "begin projections  # From projections.txt\nang\n"
for atom_index in range(cnt.positions.shape[0]):
    projections_string += (
        "    c="
        + get_coords_string(cnt.positions[atom_index, :])
        + " : pz : z="
        + get_coords_string(radial_coordinates[atom_index, :])
        + "\n"
    )
for bond_center in bond_centers:
    projections_string += "    c=" + get_coords_string(bond_center) + " : s\n"
projections_string += "end projections"

# Save outputs
ase.io.write("POSCAR", cnt)
with open("projections.txt", "w") as text_file:
    text_file.write(projections_string)
```

</details>

## Self-Consistent Field (SCF) Calculation

After relaxing this geometry using VASP, we perform a self-consistent
field (SCF) calculation to obtain the electronic structure of the
system. The following input files are used for this calculation:

<details>
<summary>INCAR</summary>

```INCAR
ENCUT = 550 eV
NBANDS = 308

# Recommended default smearing configuration: https://vasp.at/wiki/Smearing_technique
ISMEAR = 0
SIGMA = 0.05
EFERMI = MIDGAP

# EDIFF recommended for FD phonon calculations: https://vasp.at/wiki/index.php/EDIFF
EDIFF = 1E-7
NELMIN = 10

LREAL = Auto

LORBIT = 11  # Save overlaps with atomic orbitals in PROCAR

# Wannier90 configuration
LWANNIER90 = .TRUE.  # Produce wannier90 input files
LWRITE_UNK = .TRUE.  # Required for wannier function plots
NUM_WANN = 80
WANNIER90_WIN = "# From INCAR

num_iter = 1000
dis_num_iter  = 10000
dis_mix_ratio = 0.75
dis_froz_min = -10
dis_froz_max = -2
dis_win_min = -30
dis_win_max = 15
guiding_centres = true

write_hr = true  # Save the Hamiltonian to seedname_hr.dat
write_xyz = true  # Save the atomic positions & Wannier centers to seedname_centres.xyz

# Plotting
wannier_plot = true  # Plot wannier functions, slow & requires UNK files
wannier_plot_supercell = 5 1 1
bands_plot = true  # View wannier90_band.gnu with gnuplot
begin kpoint_path
-X  -0.5  0.0  0.0   X  0.5  0.0  0.0
end kpoint_path

begin projections  # From projections.txt
ang
    c=0.3550000,25.0000000,28.1315481 : pz : z=0.0000000,0.0000000,3.1315481
    c=3.9050000,26.1983916,27.8931732 : pz : z=0.0000000,1.1983916,2.8931732
    c=2.4850000,26.1983916,27.8931732 : pz : z=0.0000000,1.1983916,2.8931732
    c=1.7750000,27.2143389,27.2143389 : pz : z=0.0000000,2.2143389,2.2143389
    c=0.3550000,27.2143389,27.2143389 : pz : z=0.0000000,2.2143389,2.2143389
    c=3.9050000,27.8931732,26.1983916 : pz : z=0.0000000,2.8931732,1.1983916
    c=2.4850000,27.8931732,26.1983916 : pz : z=0.0000000,2.8931732,1.1983916
    c=1.7750000,28.1315481,25.0000000 : pz : z=0.0000000,3.1315481,0.0000000
    c=0.3550000,28.1315481,25.0000000 : pz : z=0.0000000,3.1315481,0.0000000
    c=3.9050000,27.8931732,23.8016084 : pz : z=0.0000000,2.8931732,-1.1983916
    c=2.4850000,27.8931732,23.8016084 : pz : z=0.0000000,2.8931732,-1.1983916
    c=1.7750000,27.2143389,22.7856611 : pz : z=0.0000000,2.2143389,-2.2143389
    c=0.3550000,27.2143389,22.7856611 : pz : z=0.0000000,2.2143389,-2.2143389
    c=3.9050000,26.1983916,22.1068268 : pz : z=0.0000000,1.1983916,-2.8931732
    c=2.4850000,26.1983916,22.1068268 : pz : z=0.0000000,1.1983916,-2.8931732
    c=1.7750000,25.0000000,21.8684519 : pz : z=0.0000000,0.0000000,-3.1315481
    c=0.3550000,25.0000000,21.8684519 : pz : z=0.0000000,0.0000000,-3.1315481
    c=3.9050000,23.8016084,22.1068268 : pz : z=0.0000000,-1.1983916,-2.8931732
    c=2.4850000,23.8016084,22.1068268 : pz : z=0.0000000,-1.1983916,-2.8931732
    c=1.7750000,22.7856611,22.7856611 : pz : z=0.0000000,-2.2143389,-2.2143389
    c=0.3550000,22.7856611,22.7856611 : pz : z=0.0000000,-2.2143389,-2.2143389
    c=3.9050000,22.1068268,23.8016084 : pz : z=0.0000000,-2.8931732,-1.1983916
    c=2.4850000,22.1068268,23.8016084 : pz : z=0.0000000,-2.8931732,-1.1983916
    c=1.7750000,21.8684519,25.0000000 : pz : z=0.0000000,-3.1315481,-0.0000000
    c=0.3550000,21.8684519,25.0000000 : pz : z=0.0000000,-3.1315481,-0.0000000
    c=3.9050000,22.1068268,26.1983916 : pz : z=0.0000000,-2.8931732,1.1983916
    c=2.4850000,22.1068268,26.1983916 : pz : z=0.0000000,-2.8931732,1.1983916
    c=1.7750000,22.7856611,27.2143389 : pz : z=0.0000000,-2.2143389,2.2143389
    c=0.3550000,22.7856611,27.2143389 : pz : z=0.0000000,-2.2143389,2.2143389
    c=3.9050000,23.8016084,27.8931732 : pz : z=0.0000000,-1.1983916,2.8931732
    c=2.4850000,23.8016084,27.8931732 : pz : z=0.0000000,-1.1983916,2.8931732
    c=1.7750000,25.0000000,28.1315481 : pz : z=0.0000000,-0.0000000,3.1315481
    c=0.0000000,25.5991958,28.0123607 : s
    c=3.1950000,26.1983916,27.8931732 : s
    c=2.1300000,26.7063653,27.5537561 : s
    c=1.0650000,27.2143389,27.2143389 : s
    c=-0.0000000,26.7063653,27.5537561 : s
    c=0.0000000,27.5537561,26.7063653 : s
    c=2.1300000,27.5537561,26.7063653 : s
    c=3.1950000,27.8931732,26.1983916 : s
    c=2.1300000,28.0123607,25.5991958 : s
    c=-0.0000000,28.0123607,25.5991958 : s
    c=1.0650000,28.1315481,25.0000000 : s
    c=-0.0000000,28.0123607,24.4008042 : s
    c=2.1300000,28.0123607,24.4008042 : s
    c=3.1950000,27.8931732,23.8016084 : s
    c=2.1300000,27.5537561,23.2936347 : s
    c=1.0650000,27.2143389,22.7856611 : s
    c=-0.0000000,27.5537561,23.2936347 : s
    c=0.0000000,26.7063653,22.4462439 : s
    c=3.1950000,26.1983916,22.1068268 : s
    c=2.1300000,26.7063653,22.4462439 : s
    c=2.1300000,25.5991958,21.9876393 : s
    c=0.0000000,25.5991958,21.9876393 : s
    c=1.0650000,25.0000000,21.8684519 : s
    c=0.0000000,24.4008042,21.9876393 : s
    c=3.1950000,23.8016084,22.1068268 : s
    c=2.1300000,24.4008042,21.9876393 : s
    c=2.1300000,23.2936347,22.4462439 : s
    c=1.0650000,22.7856611,22.7856611 : s
    c=0.0000000,23.2936347,22.4462439 : s
    c=-0.0000000,22.4462439,23.2936347 : s
    c=2.1300000,22.4462439,23.2936347 : s
    c=3.1950000,22.1068268,23.8016084 : s
    c=2.1300000,21.9876393,24.4008042 : s
    c=1.0650000,21.8684519,25.0000000 : s
    c=-0.0000000,21.9876393,24.4008042 : s
    c=0.0000000,21.9876393,25.5991958 : s
    c=3.1950000,22.1068268,26.1983916 : s
    c=2.1300000,21.9876393,25.5991958 : s
    c=2.1300000,22.4462439,26.7063653 : s
    c=1.0650000,22.7856611,27.2143389 : s
    c=0.0000000,22.4462439,26.7063653 : s
    c=0.0000000,23.2936347,27.5537561 : s
    c=0.0000000,24.4008042,28.0123607 : s
    c=3.1950000,23.8016084,27.8931732 : s
    c=2.1300000,23.2936347,27.5537561 : s
    c=1.0650000,25.0000000,28.1315481 : s
    c=2.1300000,25.5991958,28.0123607 : s
    c=2.1300000,24.4008042,28.0123607 : s
end projections
"
```

</details>

<details>
<summary>KPOINTS</summary>

```KPOINTS
Regular 11x1x1 k-point mesh
0
Gamma
11 1 1
```

</details>

<details>
<summary>POSCAR</summary>

```POSCAR
C 
 1.0000000000000000
     4.2599999999999998    0.0000000000000000    0.0000000000000000
     0.0000000000000000   50.0000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000   50.0000000000000000
 C  
  32
Cartesian
  0.3549999999999998 25.0000000000000000 28.1315481259957778
  3.9049999999999994 26.1983915854725282 27.8931732186815751
  2.4849999999999994 26.1983915854725282 27.8931732186815751
  1.7749999999999997 27.2143389155036388 27.2143389155036388
  0.3549999999999998 27.2143389155036388 27.2143389155036388
  3.9049999999999994 27.8931732186815751 26.1983915854725282
  2.4850000000000003 27.8931732186815751 26.1983915854725282
  1.7750000000000001 28.1315481259957778 25.0000000000000000
  0.3549999999999998 28.1315481259957778 25.0000000000000000
  3.9049999999999985 27.8931732186815751 23.8016084145274718
  2.4849999999999994 27.8931732186815751 23.8016084145274718
  1.7749999999999992 27.2143389155036388 22.7856610844963612
  0.3549999999999998 27.2143389155036388 22.7856610844963612
  3.9050000000000002 26.1983915854725282 22.1068267813184249
  2.4849999999999994 26.1983915854725282 22.1068267813184249
  1.7749999999999992 25.0000000000000000 21.8684518740042222
  0.3549999999999998 25.0000000000000000 21.8684518740042222
  3.9050000000000002 23.8016084145274718 22.1068267813184249
  2.4849999999999994 23.8016084145274718 22.1068267813184249
  1.7749999999999992 22.7856610844963612 22.7856610844963612
  0.3549999999999998 22.7856610844963612 22.7856610844963612
  3.9049999999999976 22.1068267813184249 23.8016084145274682
  2.4849999999999994 22.1068267813184249 23.8016084145274682
  1.7749999999999992 21.8684518740042222 25.0000000000000000
  0.3549999999999998 21.8684518740042222 25.0000000000000000
  3.9050000000000002 22.1068267813184249 26.1983915854725282
  2.4849999999999994 22.1068267813184249 26.1983915854725282
  1.7749999999999992 22.7856610844963612 27.2143389155036424
  0.3549999999999998 22.7856610844963612 27.2143389155036388
  3.9050000000000002 23.8016084145274682 27.8931732186815751
  2.4849999999999994 23.8016084145274682 27.8931732186815751
  1.7749999999999992 25.0000000000000000 28.1315481259957778
```

</details>

The `POSCAR` file as well as the projections block in `INCAR` were
generated using `generate_structure.py` from the previous section.

The DFT calculations were performed with the standard VASP `PAW_PBE`
carbon pseudopotential (`C`, 08Apr2002) obtained from the VASP portal.

Using these input files, the SCF calculation can be run (parallelizing
over bands) with the following command:

```bash
mpiexec -n 14 vasp_std
```

At the end of this calculation we find a Fermi energy of -4.1559415666
eV for this system.

## Wannierization

We wannierize the electronic structure of the CNT using Wannier90
through the VASP interface (`LWANNIER90 = .TRUE.`). We use atom-centered
pz orbitals and bond-centered s orbitals as initial projections for the
wannierization. The input file generated by VASP can be found below.
[This
tutorial](https://www.wanniertools.org/tutorials/high-quality-wfs/) can
be helpful to find suitable parameters for the wannierization.

<details>
<summary>wannier90.win</summary>

```wannier90.win
# From INCAR

num_iter = 1000
dis_num_iter  = 10000
dis_mix_ratio = 0.75
dis_froz_min = -10
dis_froz_max = -2
dis_win_min = -30
dis_win_max = 15
guiding_centres = true

write_hr = true  # Save the Hamiltonian to seedname_hr.dat
write_xyz = true  # Save the atomic positions & Wannier centers to seedname_centres.xyz

# Plotting
wannier_plot = true  # Plot wannier functions, slow & requires UNK files
wannier_plot_supercell = 5 1 1
bands_plot = true  # View wannier90_band.gnu with gnuplot
begin kpoint_path
-X  -0.5  0.0  0.0   X  0.5  0.0  0.0
end kpoint_path

begin projections  # From projections.txt
ang
    c=0.3550000,25.0000000,28.1315481 : pz : z=0.0000000,0.0000000,3.1315481
    c=3.9050000,26.1983916,27.8931732 : pz : z=0.0000000,1.1983916,2.8931732
    c=2.4850000,26.1983916,27.8931732 : pz : z=0.0000000,1.1983916,2.8931732
    c=1.7750000,27.2143389,27.2143389 : pz : z=0.0000000,2.2143389,2.2143389
    c=0.3550000,27.2143389,27.2143389 : pz : z=0.0000000,2.2143389,2.2143389
    c=3.9050000,27.8931732,26.1983916 : pz : z=0.0000000,2.8931732,1.1983916
    c=2.4850000,27.8931732,26.1983916 : pz : z=0.0000000,2.8931732,1.1983916
    c=1.7750000,28.1315481,25.0000000 : pz : z=0.0000000,3.1315481,0.0000000
    c=0.3550000,28.1315481,25.0000000 : pz : z=0.0000000,3.1315481,0.0000000
    c=3.9050000,27.8931732,23.8016084 : pz : z=0.0000000,2.8931732,-1.1983916
    c=2.4850000,27.8931732,23.8016084 : pz : z=0.0000000,2.8931732,-1.1983916
    c=1.7750000,27.2143389,22.7856611 : pz : z=0.0000000,2.2143389,-2.2143389
    c=0.3550000,27.2143389,22.7856611 : pz : z=0.0000000,2.2143389,-2.2143389
    c=3.9050000,26.1983916,22.1068268 : pz : z=0.0000000,1.1983916,-2.8931732
    c=2.4850000,26.1983916,22.1068268 : pz : z=0.0000000,1.1983916,-2.8931732
    c=1.7750000,25.0000000,21.8684519 : pz : z=0.0000000,0.0000000,-3.1315481
    c=0.3550000,25.0000000,21.8684519 : pz : z=0.0000000,0.0000000,-3.1315481
    c=3.9050000,23.8016084,22.1068268 : pz : z=0.0000000,-1.1983916,-2.8931732
    c=2.4850000,23.8016084,22.1068268 : pz : z=0.0000000,-1.1983916,-2.8931732
    c=1.7750000,22.7856611,22.7856611 : pz : z=0.0000000,-2.2143389,-2.2143389
    c=0.3550000,22.7856611,22.7856611 : pz : z=0.0000000,-2.2143389,-2.2143389
    c=3.9050000,22.1068268,23.8016084 : pz : z=0.0000000,-2.8931732,-1.1983916
    c=2.4850000,22.1068268,23.8016084 : pz : z=0.0000000,-2.8931732,-1.1983916
    c=1.7750000,21.8684519,25.0000000 : pz : z=0.0000000,-3.1315481,-0.0000000
    c=0.3550000,21.8684519,25.0000000 : pz : z=0.0000000,-3.1315481,-0.0000000
    c=3.9050000,22.1068268,26.1983916 : pz : z=0.0000000,-2.8931732,1.1983916
    c=2.4850000,22.1068268,26.1983916 : pz : z=0.0000000,-2.8931732,1.1983916
    c=1.7750000,22.7856611,27.2143389 : pz : z=0.0000000,-2.2143389,2.2143389
    c=0.3550000,22.7856611,27.2143389 : pz : z=0.0000000,-2.2143389,2.2143389
    c=3.9050000,23.8016084,27.8931732 : pz : z=0.0000000,-1.1983916,2.8931732
    c=2.4850000,23.8016084,27.8931732 : pz : z=0.0000000,-1.1983916,2.8931732
    c=1.7750000,25.0000000,28.1315481 : pz : z=0.0000000,-0.0000000,3.1315481
    c=0.0000000,25.5991958,28.0123607 : s
    c=3.1950000,26.1983916,27.8931732 : s
    c=2.1300000,26.7063653,27.5537561 : s
    c=1.0650000,27.2143389,27.2143389 : s
    c=-0.0000000,26.7063653,27.5537561 : s
    c=0.0000000,27.5537561,26.7063653 : s
    c=2.1300000,27.5537561,26.7063653 : s
    c=3.1950000,27.8931732,26.1983916 : s
    c=2.1300000,28.0123607,25.5991958 : s
    c=-0.0000000,28.0123607,25.5991958 : s
    c=1.0650000,28.1315481,25.0000000 : s
    c=-0.0000000,28.0123607,24.4008042 : s
    c=2.1300000,28.0123607,24.4008042 : s
    c=3.1950000,27.8931732,23.8016084 : s
    c=2.1300000,27.5537561,23.2936347 : s
    c=1.0650000,27.2143389,22.7856611 : s
    c=-0.0000000,27.5537561,23.2936347 : s
    c=0.0000000,26.7063653,22.4462439 : s
    c=3.1950000,26.1983916,22.1068268 : s
    c=2.1300000,26.7063653,22.4462439 : s
    c=2.1300000,25.5991958,21.9876393 : s
    c=0.0000000,25.5991958,21.9876393 : s
    c=1.0650000,25.0000000,21.8684519 : s
    c=0.0000000,24.4008042,21.9876393 : s
    c=3.1950000,23.8016084,22.1068268 : s
    c=2.1300000,24.4008042,21.9876393 : s
    c=2.1300000,23.2936347,22.4462439 : s
    c=1.0650000,22.7856611,22.7856611 : s
    c=0.0000000,23.2936347,22.4462439 : s
    c=-0.0000000,22.4462439,23.2936347 : s
    c=2.1300000,22.4462439,23.2936347 : s
    c=3.1950000,22.1068268,23.8016084 : s
    c=2.1300000,21.9876393,24.4008042 : s
    c=1.0650000,21.8684519,25.0000000 : s
    c=-0.0000000,21.9876393,24.4008042 : s
    c=0.0000000,21.9876393,25.5991958 : s
    c=3.1950000,22.1068268,26.1983916 : s
    c=2.1300000,21.9876393,25.5991958 : s
    c=2.1300000,22.4462439,26.7063653 : s
    c=1.0650000,22.7856611,27.2143389 : s
    c=0.0000000,22.4462439,26.7063653 : s
    c=0.0000000,23.2936347,27.5537561 : s
    c=0.0000000,24.4008042,28.0123607 : s
    c=3.1950000,23.8016084,27.8931732 : s
    c=2.1300000,23.2936347,27.5537561 : s
    c=1.0650000,25.0000000,28.1315481 : s
    c=2.1300000,25.5991958,28.0123607 : s
    c=2.1300000,24.4008042,28.0123607 : s
end projections

# This part was generated automatically by VASP
num_bands = 308
num_wann = 80
begin unit_cell_cart
     4.2600000     0.0000000     0.0000000
     0.0000000    50.0000000     0.0000000
     0.0000000     0.0000000    50.0000000
end unit_cell_cart
begin atoms_cart
C        0.3550000    25.0000000    28.1315481
C        3.9050000    26.1983916    27.8931732
C        2.4850000    26.1983916    27.8931732
C        1.7750000    27.2143389    27.2143389
C        0.3550000    27.2143389    27.2143389
C        3.9050000    27.8931732    26.1983916
C        2.4850000    27.8931732    26.1983916
C        1.7750000    28.1315481    25.0000000
C        0.3550000    28.1315481    25.0000000
C        3.9050000    27.8931732    23.8016084
C        2.4850000    27.8931732    23.8016084
C        1.7750000    27.2143389    22.7856611
C        0.3550000    27.2143389    22.7856611
C        3.9050000    26.1983916    22.1068268
C        2.4850000    26.1983916    22.1068268
C        1.7750000    25.0000000    21.8684519
C        0.3550000    25.0000000    21.8684519
C        3.9050000    23.8016084    22.1068268
C        2.4850000    23.8016084    22.1068268
C        1.7750000    22.7856611    22.7856611
C        0.3550000    22.7856611    22.7856611
C        3.9050000    22.1068268    23.8016084
C        2.4850000    22.1068268    23.8016084
C        1.7750000    21.8684519    25.0000000
C        0.3550000    21.8684519    25.0000000
C        3.9050000    22.1068268    26.1983916
C        2.4850000    22.1068268    26.1983916
C        1.7750000    22.7856611    27.2143389
C        0.3550000    22.7856611    27.2143389
C        3.9050000    23.8016084    27.8931732
C        2.4850000    23.8016084    27.8931732
C        1.7750000    25.0000000    28.1315481
end atoms_cart
mp_grid =    11     1     1
begin kpoints
      0.000000000000      0.000000000000      0.000000000000
      0.090909090909      0.000000000000      0.000000000000
      0.181818181818      0.000000000000      0.000000000000
      0.272727272727      0.000000000000      0.000000000000
      0.363636363636      0.000000000000      0.000000000000
      0.454545454545      0.000000000000      0.000000000000
     -0.090909090909     -0.000000000000     -0.000000000000
     -0.181818181818     -0.000000000000     -0.000000000000
     -0.272727272727     -0.000000000000     -0.000000000000
     -0.363636363636     -0.000000000000     -0.000000000000
     -0.454545454545     -0.000000000000     -0.000000000000
end kpoints
```

</details>

At the end of the Wannier90 calculation, we find spreads around 1
Å^2 for all 80 Wannier functions. The DFT band structure and the
Wannier-interpolated band structure are in good agreement. The resulting
Hamiltonian, stored in the `wannier90_hr.dat` file, is used to construct
inputs for the transport simulations.

## Constructing Transport Hamiltonian and Structure Files

The Wannier90 Hamiltonian can be converted to `quatrex`'s HDF5 format
and used directly in transport calculations, where the upscaling from
unit cell to transport Hamiltonian (`device.construct_from_unit_cell =
true`) is handled by `quatrex`.

The following script generates the input data required by `quatrex`.

<details>
<summary>get_input_files.py</summary>

```python
#!/usr/bin/env python

import ase.geometry
import ase.io
import numpy as np
from pathlib import Path
import scipy.constants
import scipy.interpolate

import qttools.utils.hdf5_utils


def read_wannier_hr(path: Path, dtype=np.complex128) -> dict:
    """Read the contents of a Wannier90 `hr.dat` file.

    Parameters
    ----------
    path : Path
        Path to a `hr.dat` file.
    dtype : optional
        The data type for the Hamiltonian matrix elements. Defaults to
        `numpy.complex128`.

    Returns
    -------
    hamiltonian : dict
        A dictionary containing the Hamiltonian hopping matrices. The
        keys are the hopping cells, and the values are the Hamiltonian
        hopping matrices.

    """

    # Strip info from header.
    num_wannier_functions, num_cells = np.loadtxt(
        path, skiprows=1, max_rows=2, dtype=int
    )

    # Read wannier data (skipping degeneracy info). Wannier90 uses some
    # Fortran formatting. Thus, the degeneracy information is arranged
    # into 15 values per line.
    degenerate_rows = int(np.ceil(num_cells / 15.0))
    wannier_data = np.loadtxt(path, skiprows=3 + degenerate_rows)

    cells = wannier_data[:, :3].astype(int)

    # Initialize Hamiltonian dictionary. The keys are the hopping cells,
    # and the values are the Hamiltonian hopping matrices.
    hamiltonian = {
        (rx.item(), ry.item(), rz.item()): np.zeros(
            (num_wannier_functions, num_wannier_functions), dtype=dtype
        )
        for rx, ry, rz in np.unique(cells, axis=0)
    }

    # Read the Hamiltonian matrix elements.
    for line in wannier_data:
        rx, ry, rz = line[:3].astype(int)
        key = (rx.item(), ry.item(), rz.item())
        i, j = line[3:5].astype(int)
        h_ij_real, h_ij_imag = line[5:]
        hamiltonian[key][i - 1, j - 1] = h_ij_real + 1j * h_ij_imag

    return hamiltonian


def get_coulomb_matrix(sites: tuple[ase.Atoms, ase.Atoms]):
    """
    Returns an approximate bare Coulomb matrix containing the force a
    point charge at any site in sites[0] exerts on a test charge in
    sites[1]. It is an approximation to assume that the Wannier
    functions are point charges located at the Wannier centers.
    """

    epsilon_0 = (
        scipy.constants.physical_constants["electric constant"][0] * 1e-10
    )  # F/Å
    e = scipy.constants.physical_constants["elementary charge"][0]  # C

    d = np.linalg.norm(
        sites[0].positions[:, np.newaxis, :] - sites[1].positions[np.newaxis, :, :],
        axis=-1,
    )
    with np.errstate(divide="ignore"):
        # Will divide by zero on the diagonal, that's handled below.
        coulomb_matrix = e / (4 * np.pi * epsilon_0 * d)
    if sites[0] == sites[1]:
        np.fill_diagonal(coulomb_matrix, 0)
    assert np.all(np.isfinite(coulomb_matrix))

    return coulomb_matrix


def get_composed_coulomb_matrix(
    sites: ase.Atoms,
    *,
    neighbor_cell_cutoff: int,
    repeated_dim: int,
    lattice_constant: float,
):
    """
    Force a point charge at site i in the unit cell exerts on a test
    charge at site j in cell m. Only implemented for 1D structures.
    """

    coulomb_matrix = {}

    for m in range(-neighbor_cell_cutoff, neighbor_cell_cutoff + 1):
        cell_coordinate = [0] * 3
        cell_coordinate[repeated_dim] = m
        cell_coordinate = tuple(cell_coordinate)

        translation = lattice_constant * np.array(cell_coordinate)
        offset_sites = sites.copy()
        offset_sites.translate(translation)

        coulomb_matrix[cell_coordinate] = get_coulomb_matrix((sites, offset_sites))

    return coulomb_matrix


def save_blocks_dict(filename: str, blocks_dict: dict):
    """
    Saves a dictionary with keys `(a, b, c)` as an HDF5 file called
    `filename` with keys "[str(a), str(b), str(c)]". We need to do this
    conversion in order to save as an HDF5 file.
    """

    # Turn the `blocks_dict`'s keys into strings
    string_keys_blocks_dict = {str(list(r)): h_r for r, h_r in blocks_dict.items()}

    qttools.utils.hdf5_utils.save_hdf5_dict(filename, string_keys_blocks_dict)


# Parameters
transport_index = 0
neighbor_cell_cutoff_coulomb_matrix = 10

# Load data
atoms = ase.io.read("POSCAR")
wannier_centers = ase.io.read("wannier90_centres.xyz")
# Remove atoms, only keep centers
wannier_centers = wannier_centers[wannier_centers.symbols == "X"]
# Wrap the centers into the unit cell
# This could be done with `translate_home_cell = true` in wannier90 in
# theory, but this does not seem to work in our case.
wannier_centers.set_cell(atoms.cell)
wrap_dim = [True] * 3
# Don't wrap in the transport direction: The Hamiltonian as well as the
# Coulomb matrix would have to be adapted accordingly. This is done in
# quatrex.
wrap_dim[transport_index] = False
wannier_centers.wrap(pbc=wrap_dim)
pbc = [False] * 3
pbc[transport_index] = True
wannier_centers.set_pbc(pbc)
hamiltonian = read_wannier_hr(path=Path("wannier90_hr.dat"))

lattice_constant = atoms.cell.array[transport_index, transport_index]
coulomb_matrix = get_composed_coulomb_matrix(
    wannier_centers,
    neighbor_cell_cutoff=neighbor_cell_cutoff_coulomb_matrix,
    repeated_dim=transport_index,
    lattice_constant=lattice_constant,
)

# Save unit cell data
Path("inputs-unit-cell").mkdir(exist_ok=True)
ase.io.write("inputs-unit-cell/structure.xyz", wannier_centers)
ase.io.write("atoms_and_centers.xyz", (atoms, wannier_centers))
save_blocks_dict("inputs-unit-cell/hamiltonian.h5", hamiltonian)
save_blocks_dict("inputs-unit-cell/coulomb_matrix.h5", coulomb_matrix)
```

</details>

The `read_wannier_hr` function is taken from `quatrex`' documentation
(User guide -> Running Transport Simulations -> Electronic Structure
Data -> Plane-Wave DFT & Wannier90).

For GW calculations, we need to construct the a Coulomb matrix in the
basis of the Wannier functions. While there are more accurate methods
available, the script above computes an approximate bare Coulomb matrix
assuming that the Wannier functions are point charges located at the
Wannier centers.

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

<details>
<summary>get_dispersion.py</summary>

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

</details>