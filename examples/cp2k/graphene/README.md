# Graphene

<!-- TODO: Document the CP2K calculation behind the inputs. -->

## `qtbm-poisson`: Back-gated graphene with doped leads

Self-consistent Schrödinger-Poisson (SCSP) simulation of a graphene
sheet (49.2 Å long in transport direction $x$, periodic along $y$) with
n-doped leads and a local back gate.

```text
   z
   ^      left lead (n)        channel          right lead (n)
   |  0 ======================================================  graphene
   |         vacuum gap (1 Å, eps_r = 1)
   | -1 ------------------------------------------------------
   |                 oxide (eps_r = 3.9)
   |-10 ---------------[ gate, V_g = -0.5 V ]----------------
   +--------------------------------------------------------> x
       0             14                 34                49.2  (Å)
```

Run the example with

```bash
quatrex mesh quatrex_config.toml --off-screen  # writes ../inputs/device.msh
mpirun -n 8 quatrex run quatrex_config.toml
```

### Graphene-specific settings

- **Density model.** `[electrostatics] density_model = "graphene"` uses
  an ambipolar Dirac-cone density model for the inner Newton iterations
  of the Poisson solver. Electrons and holes are treated simultaneously,
  so the channel can be inverted (n-p-n) by the gate.
- **Charge neutrality level.** The Dirac point of this Hamiltonian is at
  $E_D \approx 0.631$ eV, which is set as the contacts'
  `mid_gap_energy`. Excess electrons/holes are counted above/below it.
- **Doping.** Doping concentrations are given in cm$^{-3}$ and converted
  to a charge per atom using the volume of the cell in `structure.xyz`.
  For a sheet density $n_{2D}$ use
  $N_D = n_{2D} / L_z$ with the cell height $L_z = 6.709$ Å, e.g.
  $5 \times 10^{12}$ cm$^{-2}$ $\rightarrow 7.45 \times 10^{19}$
  cm$^{-3}$. The contact `lattice_vectors` must use the same height so
  that the contact Fermi levels are computed for the same doping.
- **Contact Fermi levels.** They are computed from the lead doping and
  the contact band structure (no `fermi_level` given), so that the leads
  are charge neutral for the transverse k-point sampling actually used
  in the transport calculation.
- **Gates.** The Dirichlet value at the gate is $\phi_\mathrm{gate} =
  -V_g$, i.e. the gate voltage is measured relative to flat-band
  conditions (no `work_function`/`electron_affinity` alignment).
- **Energy window.** It covers $E_D \pm 0.6$ eV, which has to contain the
  Dirac point plus the full range of the local potential and the
  chemical potentials (with a margin of several $k_B T$).

### Transverse k-points

The cell is only 8.5 Å wide in $y$, so the transverse k-point grid
determines the density of states that the transport calculation sees:

- Grids without $\Gamma$ (even number of points) miss the Dirac point
  and leave an artificial gap.
- With $\Gamma$ included, the $k_y = 0$ metallic subband dominates for
  small grids. For a fixed $E_F - E_D = 0.257$ eV the lead excess charge
  is $1.9\times10^{-2}$, $6.2\times10^{-3}$ e/atom for 1 and 3 $k_y$
  points, compared to $1.3\times10^{-3}$ e/atom for 2D graphene.
  Reaching the 2D limit requires on the order of 50 or more $k_y$
  points.

This example uses `kpoint_grid = [1, 3, 1]` to stay cheap. Since the
contact Fermi levels are computed for this sampling, the leads are still
charge neutral ($E_F - E_D \approx 55$ meV instead of 257 meV for 2D
graphene at $5 \times 10^{12}$ cm$^{-2}$), but the quantum capacitance
of the channel is larger than in real graphene. Increase the number of
$k_y$ points for quantitative results.

### Density model comparison

The density model only enters the inner Newton iterations of the
Poisson solver; the charge itself always comes from the transport
calculation. It therefore affects how fast the SCSP loop converges, not
the converged result. Running this example with the default
`density_model = "single-band"` (`density_model_dim = 2`) instead of
`"graphene"` gives (8 MPI ranks, about 2.3 min per SCSP iteration):

| | `"graphene"` | `"single-band"` |
|---|---|---|
| SCSP iterations to converge | 8 | 14 |
| Potential at the channel centre | 0.1480 V | 0.1469 V |
| Potential range at the atoms | 0.040 – 0.152 V | 0.039 – 0.152 V |
| Total excess charge | −0.0719 e | −0.0704 e |

The converged potentials differ by at most 1.3 mV at the atoms, of the
order of the SCSP tolerance (`convergence_tol = 1e-3` V on the unmixed
Poisson update), and the total charge by 2 %. The
single-band model converges more slowly (the mixed potential update
decreases by about ×0.75 per iteration instead of ×0.55), because its
density derivative does not match the quantum capacitance of the
Dirac cone. `max_iterations = 20` leaves room for either model.

## `negf-poisson`: The same device with NEGF

Ballistic version of `qtbm-poisson` with `formalism = "negf"`, for
cross-checking the two formalisms and as a starting point for SCSP runs
with scattering. The device, the doping, the gate, the bias, the k-point
grid and the energy grid are the same. Run the example with

```bash
mpirun -n 30 quatrex run quatrex_config.toml
```

It needs the mesh `../inputs/device.msh`, see above. With 30 ranks and
two threads each, it takes about 4 min per SCSP iteration (35 min in
total) and up to about 100 GB of memory, most of it while the contact
Fermi levels are computed. Every SCSP iteration also writes the spectral
electron and hole densities (about 24 MB).

### NEGF-specific settings

- **Ballistic SCBA.** The SCBA requires at least one interaction. A
  pseudo-phonon with vanishing coupling makes all self-energies zero, so
  that one SCBA iteration is sufficient.
- **Blocks.** `block_size = 416` corresponds to four slices of 8 atoms
  with 13 orbitals each. The contact cell is therefore four times longer
  than in `qtbm-poisson`, and 75 instead of 300 `num_kpoints_transport`
  give the same sampling density.
- **Right contact.** It has to be the exact mirror of the last block of
  the device. In each slice, the orbitals of the lower half (in $y$)
  come first, so the transverse lattice vector is reversed. It is the
  same lead as in `qtbm-poisson`.
- **Lead potential.** `lead_potential = "unit-cell"` repeats the
  potential of the outermost contact unit cell in the lead.

### Lead potential

The potential varies by almost 20 mV between the atoms at the left end
of the device ($x = 0$), also between atoms that are equivalent in the
lead. In the lead, this opens a gap of about 3 – 4 meV at the Dirac
point, which is of the order of the spacing of the energy grid (5 meV).
If the edge of the gap crosses a point of the grid from one SCSP
iteration to the next, the occupation at that energy jumps and the SCSP
loop oscillates between two states.

| `lead_potential` | `"device"` | `"unit-cell"` | `"average"` | QTBM |
|---|---|---|---|---|
| Last SCSP iteration | 19 (not converged) | 8 | 8 | 8 |
| Final potential update | 2.94 mV | 0.68 mV | 0.65 mV | 0.66 mV |
| Total excess charge | −0.070 / −0.074 e | −0.0719 e | −0.0724 e | −0.0719 e |
| Mean deviation from QTBM at the atoms | −0.5 / −0.1 mV | −0.04 mV | −0.9 mV | – |
| Largest deviation from QTBM at an atom | 1.6 / 1.2 mV | 0.4 mV | 2.1 mV | – |

- `"device"` (the default) is not strictly periodic, and the edge of the
  gap crosses the grid point at 0.681 eV, where the current alternates
  between zero and a finite value.
- `"unit-cell"` agrees best with QTBM. It does not remove the gap, but
  0.681 eV stays inside of it, so the loop converges. This can be
  different for other biases, gate voltages or energy grids.
- `"average"` removes the gap, but shifts the potential by about 1 mV.

The iteration count is the index of the last SCSP iteration, starting
from 0. The two values for `"device"` are those of the last two
iterations, i.e. of the two states that the loop alternates between.
