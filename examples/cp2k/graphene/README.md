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
