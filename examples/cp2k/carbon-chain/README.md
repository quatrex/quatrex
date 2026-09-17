# Chain of carbon atoms

The electronic structure data for this 25 nm long chain of carbon atoms
was obtained from a CP2K calculation using the DZVP basis set and the
PBE functional.

## Software Versions

- CP2K: `CP2K version 2025.1 (git:9635df4)`

## Geometry and Relaxation

The structure used in this example is a periodic chain of 20 carbon
atoms in an orthorhombic supercell, that will later be repeated 10 times
along the transport direction (x-axis) to construct the actual device
structure.

<details>
<summary>carbon_chain.xyz</summary>

```xyz
20
Lattice="25.703291 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3 pbc="T T T"
C      -10.93402500      -0.00008535       0.00002697
C      -12.19892500       0.00004545      -0.00002408
C       -8.36369300      -0.00004005      -0.00001201
C       -9.62861200       0.00008425      -0.00002962
C       -5.79339900      -0.00009125       0.00003025
C       -7.05824100       0.00004315      -0.00001723
C       -3.22303600      -0.00003005      -0.00000507
C       -4.48793700       0.00008635      -0.00002308
C       -0.65273590      -0.00008475       0.00003273
C       -1.91759600       0.00005515      -0.00001280
C        1.91762000      -0.00002595      -0.00000698
C        0.65272640       0.00009125      -0.00002096
C        4.48794000      -0.00007975       0.00002833
C        3.22304100       0.00005985      -0.00001920
C        7.05825400      -0.00003165      -0.00001459
C        5.79340100       0.00008955      -0.00002883
C        9.62861000      -0.00007855       0.00002632
C        8.36370700       0.00005575      -0.00002466
C       12.19892500      -0.00003985      -0.00001758
C       10.93403500       0.00008625      -0.00003273
```

</details>

For CP2K real-space matrix output, make sure all atoms are centered in
the unit cell and their fractional coordinates sit in the range `[-0.5,
0.5)`, otherwise the output matrices will end up being incorrect. The
above structure is already centered and satisfies this requirement.

You can center structures about the origin using `ase`:

```python
chain = ase.io.read("carbon_chain.xyz")
chain.center(about=(0.0, 0.0, 0.0))
```

To go from the above structure to the actual device structure, we repeat
this unit cell 10 times along the x-axis:

```python
chain.repeat((10, 1, 1))
```

## Self-Consistent Field (SCF) Calculation

We run a periodic DFT calculation in CP2K and print Hamiltonian and
overlap matrices in binary output files (`KS_CSR_WRITE` and
`S_CSR_WRITE`) with `REAL_SPACE` enabled.

<details>
<summary>carbon_chain.inp</summary>

```text
&GLOBAL
    PROJECT carbon_chain
    RUN_TYPE ENERGY
    PRINT_LEVEL MEDIUM
    EXTENDED_FFT_LENGTHS
    PREFERRED_DIAG_LIBRARY SL
&END GLOBAL
&FORCE_EVAL
    METHOD QS
    STRESS_TENSOR ANALYTICAL
    &SUBSYS
        &CELL
            A 25.703291 0.00000000 0.00000000
            B 0.00000000 10 0.00000000
            C 0.00000000 0.00000000 10
            MULTIPLE_UNIT_CELL 1 1 1
            SYMMETRY ORTHORHOMBIC
        &END CELL
        &TOPOLOGY
            MULTIPLE_UNIT_CELL 1 1 1
            COORD_FILE_NAME carbon_chain.xyz
            COORD_FILE_FORMAT XYZ
        &END TOPOLOGY
        &KIND C
            ELEMENT C
            BASIS_SET DZVP-MOLOPT-PBE-GTH-q4
            POTENTIAL GTH-PBE-q4
        &END KIND
    &END SUBSYS
    &DFT
        CHARGE 0
        BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH
        POTENTIAL_FILE_NAME POTENTIAL_UZH
        &QS
            METHOD GPW
            EPS_DEFAULT 1e-18
        &END QS
        &MGRID
            NGRIDS 5
            CUTOFF 1400
            REL_CUTOFF 80
        &END MGRID
        &SCF
            EPS_SCF 5e-8
            MAX_SCF 400
            ADDED_MOS 60
            &DIAGONALIZATION T
                ALGORITHM STANDARD
            &END DIAGONALIZATION
            &MIXING T
                NBUFFER 8
                BETA 0.15
                ALPHA 0.1
                METHOD BROYDEN_MIXING
            &END MIXING
            &PRINT
                &RESTART OFF
                &END RESTART
            &END PRINT
            &SMEAR ON
                METHOD FERMI_DIRAC
                ELECTRONIC_TEMPERATURE [K] 50
            &END SMEAR
        &END SCF
        &XC
            &XC_FUNCTIONAL PBE
            &END XC_FUNCTIONAL
        &END XC
        &KPOINTS
            SCHEME MONKHORST-PACK 5 1 1
        &END KPOINTS
        &PRINT
            &KS_CSR_WRITE
                ADD_LAST SYMBOLIC
                &EACH
                    CELL_OPT 0
                    GEO_OPT 0
                    QS_SCF 0
                &END EACH
                THRESHOLD 1e-8
                BINARY
                REAL_SPACE
            &END KS_CSR_WRITE
            &S_CSR_WRITE
                ADD_LAST SYMBOLIC
                &EACH
                    CELL_OPT 0
                    GEO_OPT 0
                    QS_SCF 0
                &END EACH
                THRESHOLD 1e-8
                BINARY
                REAL_SPACE
            &END S_CSR_WRITE
            &BAND_STRUCTURE
                FILE_NAME bands.dat
                &KPOINT_SET
                    UNITS B_VECTOR
                    SPECIAL_POINT -0.5 0 0
                    SPECIAL_POINT 0 0 0
                    SPECIAL_POINT 0.5 0 0
                    NPOINTS 10
                &END
            &END BAND_STRUCTURE
        &END PRINT
    &END DFT
&END FORCE_EVAL
```

</details>

After running the above input file in CP2K, you should see something like the following in the output file:

```carbon_chain.out
  ...
  *** SCF run converged in    18 steps ***


  Electronic density on regular grids:        -80.0000000000       -0.0000000000
  Core density on regular grids:               80.0000000000        0.0000000000
  Total charge density on r-space grids:        0.0000000000
  Total charge density g-space grids:          -0.0000000000

  Overlap energy of the core charge distribution:               0.00005660086465
  Self energy of the core charge distribution:               -266.63433616670648
  Core Hamiltonian energy:                                     82.16139318008054
  Hartree energy:                                             105.32072933538288
  Exchange-correlation energy:                                -34.00127996023918
  Electronic entropic energy:                                  -0.00000000000000
  Fermi energy:                                                -0.16549329070170

  Total energy:                                              -113.15343701061758
...
```

## Generated Electronic Structure Files

CP2K writes the real-space matrices as binary CSR files with the
following naming convention:

- `carbon_chain-KS_SPIN_1_R_*-1_0.csr`
- `carbon_chain-S_SPIN_1_R_*-1_0.csr`

These files can be read following the CP2K example in
`docs/user_guide/input_data.md`. The mapping from file number to unit
cell image can be obtained from the CP2K output file:

<details>
<summary>carbon_chain.out</summary>

```out
...

 KS CSR write|  27 periodic images
      Number    X      Y      Z
         1      0      0      0
         2     -1      0      0
         3      1      0      0
         4      0     -1      0
         5      0      1      0
         6      0      0     -1
         7      0      0      1
         8     -1     -1      0
         9      1     -1      0
        10     -1      1      0
        11      1      1      0
        12     -1      0     -1
        13      1      0     -1
        14     -1      0      1
        15      1      0      1
        16      0     -1     -1
        17      0      1     -1
        18      0     -1      1
        19      0      1      1
        20     -1     -1     -1
        21      1     -1     -1
        22     -1      1     -1
        23     -1     -1      1
        24      1      1     -1
        25      1     -1      1
        26     -1      1      1
        27      1      1      1

  S CSR write|  27 periodic images
      Number    X      Y      Z
         1      0      0      0
         2     -1      0      0
         3      1      0      0
         4      0     -1      0
         5      0      1      0
         6      0      0     -1
         7      0      0      1
         8     -1     -1      0
         9      1     -1      0
        10     -1      1      0
        11      1      1      0
        12     -1      0     -1
        13      1      0     -1
        14     -1      0      1
        15      1      0      1
        16      0     -1     -1
        17      0      1     -1
        18      0     -1      1
        19      0      1      1
        20     -1     -1     -1
        21      1     -1     -1
        22     -1      1     -1
        23     -1     -1      1
        24      1      1     -1
        25      1     -1      1
        26     -1      1      1
        27      1      1      1

...
```

</details>

## Unit Conversion (Hartree to eV)

The Hamiltonian matrix elements from CP2K are in Hartree. For `quatrex`
calculations, these must be converted to eV:

```python
from scipy.constants import physical_constants

e = physical_constants["elementary charge"][0]
E_h = physical_constants["Hartree energy"][0]

hamiltonian *= E_h / e
```

## Constructing the Transport Hamiltonian and Orbital Overlap Matrices

Just like with electronic structure data from Wannier90, the CP2K
matrices can be converted to `quatrex`'s HDF5 format and used directly
in transport calculations, where the upscaling from unit cell to
transport matrix (`device.construct_from_unit_cell = true`) is handled
by `quatrex` (see the README for the `w90/carbon-nanotube` example for
details).

Alternatively, also following the same procedure as in the
`w90/carbon-nanotube` example, the transport Hamiltonian and orbital
overlap matrices can be manually upscaled by calling into the
`quatrex.device.inputs` API.
