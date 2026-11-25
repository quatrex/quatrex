# agnr5_poisson: self-consistent Schrödinger–Poisson (quatrex) for a 5-AGNR FET

A minimal quatrex SCSP example (`formalism = "wf"` + `[scsp]`): an n-i-n double-gate
FET made of a pristine 5-AGNR. It is separate from the graphene | 5-AGNR | graphene
junction: quatrex's excess-charge electrostatics only works for semiconducting
contacts, and the graphene leads are metallic.

## Model

- `make_inputs.py`: p_z tight binding (Son, Cohen & Louie, PRL 97, 216803 (2006)):
  t = -2.7 eV, edge dimers 1.12 t, one orthogonal orbital per C, no H. Gap 0.314 eV at
  Gamma, mid-gap 0 eV. 40 cells x 4.26 Å = 170.4 Å (400 orbitals), ribbon at y = z = 10 Å
  in a 170.4 x 20 x 20 Å box with no periodicity. The full device H is written as `[0,0,0]`
  and the contacts are found in real space. It also writes the bare Coulomb matrix
  (`coulomb_matrix.h5`) used by `negf-gw-poisson` (Ohno, U = 11.26 eV, V_NN = 7.54 eV).
- `scsp/quatrex_config.toml`: source x < 55 Å and drain x > 115 Å doped with 5e19 cm^-3
  donors (per box volume, 0.085 e per cell). quatrex sets E_F = 0.243 eV from the doping
  (CB 0.157 eV). Channel: 6 nm, intrinsic, with gate plates at z = 4 and 16 Å and
  eps_r = 3.9 between them (vacuum elsewhere). V_D = 0.1 V (`voltage` of the right
  contact, mu_D = E_F - 0.1). Gate Dirichlet value phi = -V_G (no work function).
  FE mesh 2 Å, 0.7 Å at the atoms. Energy grid -1..1 eV at 1 meV.
  Under-relaxation 0.5 (adaptive) and tol 1e-3 eV. 1D density model.

## Run (on tortin4, where the venv lives)

    ./run.sh               # scsp/ (V_G = 0): quatrex mesh + quatrex run, 50 ranks, about 45 s
    ./sweep.sh             # vg<V_G>/ for V_G = -0.4 .. 0.2 V
    ../../.venv/bin/python plot.py   # sweep.png, sweep.json

`env.sh` sets the OpenMPI + Intel runtime paths that mpi4py needs. `run.sh` sources it.

## Results (sweep.png / sweep.json)

| V_G (V) | iterations | I_D (µA) | CB top - E_F (eV) |
|---|---|---|---|
| -0.4 | not conv. (80) | 0.81 | +0.245 |
| -0.3 | 57 | 0.91 | +0.182 |
| -0.2 | 30 | 1.24 | +0.128 |
| -0.1 | 27 | 1.89 | +0.084 |
|  0.0 | 25 | 3.03 | +0.040 |
| +0.1 | 24 | 4.86 | -0.015 |
| +0.2 | 24 | 6.55 | -0.070 |

- `current_lr` (A, negative for electrons going L -> R) matches Landauer
  2e/h ∫T(f_S - f_D) dE from `transmission_lr` to 1e-8.
- The ON/OFF ratio is only about 8. With a 0.31 eV gap and a 6 nm channel, at negative V_G the
  channel VB rises to E_F (holes up to 0.035 e per cell). The current then flows by
  band-to-band tunnelling, so the device is ambipolar.
- V_G = -0.4 V: the residual stops falling at 1-3 meV and stays there (holes in the
  channel well, sharp resonances on the 1 meV grid). DIIS (`max_history = 5`) does the same.
  Over the last 20 of 80 iterations the potential changes by 1.4 meV. The last iterate is plotted.
- Point-charge (Mulliken) electrostatics: the donor background is spread evenly over
  all atoms, but the band-edge states have a node on the central dimer line. Those two atoms
  per cell therefore carry net positive charge and a -0.18 eV potential spike. With
  atoms_mesh_size at 1.5 Å instead of 0.7 Å the spikes stay the same size (-0.19 vs
  -0.18 eV in cell 5), and the per-cell minima move by at most 0.07 eV. The spikes come
  from the model, not the mesh. `plot.py` therefore takes the
  band edges as <psi_edge|V|psi_edge> (first order), not as the cell mean.
- `electrostatics.initial_guess = "constraints"` fails for wf contacts in this quatrex
  version (`QTBMContact` has no `origin_atom_indices`), so `"zero"` is used.

## negf-gw-poisson: NEGF-GW with Poisson

The same n-i-n double-gate FET with `formalism = "negf"`, GW and pseudo-phonon
scattering (50 meV, 25 meV deformation potential), V_G = 0, V_D = 0.1 V. Unlike
`scsp/`, the dielectric is 3.9 everywhere, as in the GW Coulomb matrix (`epsilon_r = 4`).

    python make_inputs.py
    cd negf-gw-poisson
    quatrex mesh quatrex_config.toml --off-screen
    mpirun -n 64 quatrex run quatrex_config.toml
    python plot.py    # ldos_charge_potential.png, .pdf

About 17 min on 64 ranks (one thread each, EPYC 7742), 1.7 GB of outputs.

### Coulomb matrix

V_ij = U / sqrt(1 + (4 pi eps0 U d_ij / e^2)^2), the Ohno interpolation between the
on-site U of C p_z and the point-charge limit. The diagonal is U, not zero as in the
carbon nanotube example. It is bare: quatrex divides by `epsilon_r` and drops pairs
further apart than `interaction_cutoff` along x. With the Fock term the on-site U shifts
the mid-gap by about -U/(2 epsilon_r) = -1.41 eV, hence the mid-gap of -1.43 eV.

Cutoff (homogeneous ribbon, V = 0, 10 meV grid):

| interaction_cutoff | SCBA |
|---|---|
| 4.5 Å | converges: self-energy update 6e-4 eV after 24 iterations, weight 1.999 per orbital (exact: 2) |
| 8 Å | does not converge: update levels off at about 0.35 eV, weight 1.88, negative densities (down to -22 per eV and orbital) at deep valence states |
| 16 Å | 11 iterations only, stopped: update rising again at 0.24 eV |

Why the larger cutoffs fail is not known. The truncated V(q) is not positive definite (lowest
eigenvalue -5.3 eV at 4.5 Å, -3.0 eV at 8 Å), but that is worse for the cutoff that
converges. Interactions beyond one unit cell are therefore missing, and the gap depends
on the cutoff (0.42 eV with 4.5 Å, 0.6 eV in the unconverged 8 Å run). The 8 Å run also
failed once on a 5 meV grid with a singular matrix in the Lyapunov solver.

### Contact levels

Band edge tracking is off in SCSP runs, so the contact levels are those of the homogeneous
doped ribbon with GW (standalone SCBA, V = 0, 5 meV grid, 30 iterations, update ~1e-4 eV).
Charge above mid-gap in a middle cell:

| E_F (eV) | -1.15 | -1.118 (interpolated) | -1.11 | -1.07 |
|---|---|---|---|---|
| electrons per cell | 0.0590 | 0.0852 | 0.0909 | 0.1143 |

Donors: 0.0852 e per cell. The band edges are read off the spectral function: VB -1.64 eV,
CB -1.23 eV, mid-gap -1.43 eV (gap 0.41 eV, 0.31 eV in TB). The edges shift by about 10 to
20 meV between these fills, so they are not exact. On the 10 meV grid the charge at
E_F = -1.15 is 0.0580 (2 % lower). The levels need recalibration if the doping,
`epsilon_r`, cutoffs or grid change.

### Results

- Converges in 22 steps (2 SCBA iterations each, 2 meV tolerance); self-energy update 7e-3 eV.
- Leads: potential -0.02 to -0.01 V in the source and -0.09 V at the drain end (-0.10 V
  ideal), charge within 4 % of the 0.0852 donors in the inner lead cells, and 0.02 e per
  cell in the channel centre. Barrier (highest cell mean minus source) 0.12 V.
- Left and right contact currents agree to 1e-4.
- Transient negative hole densities in SCBA iterations 5 and 6 (-0.29 and -1.58 per eV
  and orbital); none below -1e-3 in any other iteration.

### Limitations

- The TB parameters were fitted to LDA and GW adds exchange on top of them (double counting).
- `energy_renormalization_terms` is the quatrex default, i.e. without the Hermitian
  part of the polarization. With it, the 8 Å run diverged (update 26 eV at one iteration).
- One orbital per atom: no sigma screening other than `epsilon_r`.
- Grid convergence was checked only between 10 and 5 meV, for the homogeneous ribbon.
