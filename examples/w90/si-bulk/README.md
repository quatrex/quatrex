# Bulk Crystalline Silicon

The electronic structure data for this crystalline silicon structure was
constructed using VASP and transformed into a basis of
maximally-localized Wannier functions using Wannier90.

## Software Versions

- VASP: `vasp.6.3.0 20Jan22 (build Mar 14 2022 17:30:40) complex`
- Wannier90: `Release: 3.1.0        5th March    2020`

## Geometry

You can create a cubic cell of crystalline silicon using `ase`, and
create a POSCAR file like this:

```python
import ase.build
import ase.io

si = ase.build.bulk("Si", "diamond", a=5.4437, cubic=True)
si.center()

ase.io.write("POSCAR", si)
```

## Self-Consistent Field (SCF) Calculation & Wannierization

After relaxing this geometry using VASP, we perform a self-consistent
field (SCF) calculation and compute the Wannier projections for the
system. The following input files are used for this calculation:

<details>
<summary>INCAR</summary>

```INCAR
ENCUT = 550
ALGO = Normal

NELM = 100
NELMIN = 1
EDIFF = 1e-6

ISMEAR = 0
SIGMA = 0.05
PREC = Accurate

ADDGRID = True
GGA = PE

KPAR=27

NBANDS = 64

LWANNIER90 = True

NUM_WANN = 32
WANNIER90_WIN = "
Begin Projections
    Si:sp3
End Projections

write_hr = True

# Disentanglement parameters
dis_num_iter = 10000
dis_mix_ratio   = 0.5

dis_froz_min = -10
dis_froz_max = 12

dis_win_min = -15
dis_win_max = 30

num_iter = 10000
"
```

</details>

<details>
<summary>KPOINTS</summary>

```KPOINTS
21x21x21 kpoint grid
 0
Monkhorst Pack
 21 21 21
 0  0  0
```

</details>

```POSCAR
Si8
1.0
   5.4437023729394527    0.0000000000000000    0.0000000000000003
  -0.0000000000000003    5.4437023729394527    0.0000000000000003
   0.0000000000000000    0.0000000000000000    5.4437023729394527
Si
8
direct
   0.7500000000000000    0.7500000000000000    0.2500000000000000
   0.0000000000000000    0.5000000000000000    0.5000000000000000
   0.7500000000000000    0.2500000000000000    0.7500000000000000
   0.0000000000000000    0.0000000000000000    0.0000000000000000
   0.2500000000000000    0.7500000000000000    0.7500000000000000
   0.5000000000000000    0.5000000000000000    0.0000000000000000
   0.2500000000000000    0.2500000000000000    0.2500000000000000
   0.5000000000000000    0.0000000000000000    0.5000000000000000
```

</details>

All calculations were performed with the standard VASP `PAW_PBE` silicon
pseudopotential (`Si`, 05Jan2001) obtained from the VASP portal.

Using these input files, the SCF calculation can be run (parallelizing
over the bands), e.g., with the following command:

```bash
mpiexec -n 27 vasp
```

At the end of this calculation we find a Fermi energy of 5.9897097901 eV
for this system. For the Wannierization we use sets of atom-centered sp3
orbitals as initial projections. [This
tutorial](https://www.wanniertools.org/tutorials/high-quality-wfs/) can
be helpful to find suitable parameters for the wannierization.

The disentanglement should converge in roughly 3000 iterations and at
the end of the spread minimization, we find spreads of roughly 3 Å^2 for
all 32 Wannier functions, which is suitable for this example. The DFT
band structure and the Wannier-interpolated band structure are in good
agreement. The resulting Hamiltonian, stored in the `wannier90_hr.dat`
file, is used to construct inputs for the transport simulations.

## Constructing Transport Hamiltonian and Structure Files

For information on how to construct the transport Hamiltonian and
structure files from the Wannier90 output files, please refer to the
README for the `w90/carbon-nanotube` example.
