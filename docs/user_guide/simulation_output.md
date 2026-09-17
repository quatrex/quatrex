# Simulation Output

## NEGF Simulation Output

### Charge carrier densities

The charge carrier densities are computed from the lesser and greater
Green's functions and saved as `numpy` arrays (`electron_density.npy` /
`hole_density.npy`) in the
[`output_directory`](parameters/quatrex/#output_directory). These
quantities are both orbital-resolved and have shapes of `(num_energies,
*num_kpoints, num_orbitals)`. `*num_kpoints` is only present if the
simulation is performed with a transverse k-point grid (see
[`kpoint_grid`](parameters/device/#kpoint_grid)).

$$
\rho^{e/h}_i(E, \mathbf{k}_\perp) = \frac{2_{\mathrm{spin}}}{2\pi}
\mathrm{Im}\left\{ G^{\lessgtr}_{ii}(E, \mathbf{k}_\perp) \right\}
$$

### Local density of states (LDOS)

Similar to the charge carrier densities, the local density of states
(LDOS) is computed from the retarded Green's function and saved as a
`numpy` array (`ldos.npy`) in the
[`output_directory`](parameters/quatrex/#output_directory). The LDOS is
also orbital-resolved and has a shape of `(num_energies, *num_kpoints,
num_orbitals)`.

$$
g_i(E, \mathbf{k}_\perp) = \frac{2_{\mathrm{spin}}}{2\pi}
\mathrm{Im}\left\{ G^{R}_{ii}(E, \mathbf{k}_\perp) \right\}
$$

### Spectral current

Two types of spectral device current can be computed with `quatrex`:

- The Meir-Wingreen current, which is computed from the lesser and
  greater Green's functions and the self-energies, and saved as a
  `numpy` array (`current_meir_wingreen.npy`). This quantity is resolved
  per transport cell and has a shape of `(num_energies, *num_kpoints,
  num_transport_cells + 1)`. It includes the contribution from the
  reservoirs into / out of the device (hence the `+ 1`). It is only
  output if [`compute_current`](parameters/solver/#compute_current) flag
  is set to `true`. In block-distributed simulations, only the contact
  currents are computed, while the remainder will be set to NaN.

$$
j_{n-1 \to n}(E, \mathbf{k}_\perp) = \mathrm{tr}\left[
\mathbf{\widetilde{\Sigma}}^{>}_{nn}(E, \mathbf{k}_\perp)
\mathbf{G}^{<}_{nn}(E, \mathbf{k}_\perp) - \mathbf{G}^{>}_{nn}(E,
\mathbf{k}_\perp) \mathbf{\widetilde{\Sigma}}^{<}_{nn}(E,
\mathbf{k}_\perp) \right]
$$

- The spectral current computed from the commutator of the Hamiltonian
  and the lesser Green's function (quantum Liouville equation). This
  quantity is also resolved per transport cell and has a shape of
  `(num_energies, *num_kpoints, num_transport_cells - 1)`. This only
  includes the current flowing between the transport cells (not the
  reservoirs).

$$
j_{n \to n+1}(E, \mathbf{k}_\perp) = \mathbf{H}_{n, n+1} \odot
\mathbf{G}^{<}_{n+1, n}(E, \mathbf{k}_\perp) - \mathbf{G}^{<}_{n,
n+1}(E, \mathbf{k}_\perp) \odot \mathbf{H}_{n+1, n}
$$

## QTBM Simulation Output

### Transmission function

The transmission function is the main output of a QTBM simulation. It is
written for every combination of leads and saved as a `numpy` array
`transmission_<xy>.npy`, where `<xy>` indicates the direction of
transport denoted by the contact name initials (e.g., `lr` for two
contacts named `"left"` and `"right"`). The transmission function has a
shape of `(*num_kpoints, num_energies)`.

### Current

The current output from a QTBM simulation is already integrated over
energy and the transverse k-points and saved as `current_<xy>.npy`,
where `<xy>` is again the direction of transport denoted by the
contacts.

### Local density of states (LDOS) per contact

The LDOS per contact contains the contribution of each contact to the
total LDOS. These are saved as `numpy` arrays `dos_<x>.npy`, where `<x>`
is the contact name. They are orbital-resolved and have a shape of
`(*num_kpoints, num_orbitals, num_energies)`.

## Self-consistent Schrödinger-Poisson Simulation Output

Besides the regular transport outputs, self-consistent
Schrödinger-Poisson simulations will produce orbital-centered potential
(`potential.npy`) and "real-space" excess charge density
(`real_space_charge_density.npy`) and potential
(`real_space_potential.npy`) files for each iteration of the
self-consistent loop. Real-space here means that the quantities are
given on the finite-element mesh used for the Poisson solver.

## Profiling and Timing Information

Every simulation run will produce a `quatrex_times.out` file in the
directory where `quatrex` was invoked. This file contains timing
information for different parts of the simulation. The file contents
vary depending on the simulation type and configuration, but they typically look something like this:

```log
SCBA: Sparsity Pattern : 0.0006s
SCBA: Sparsity Pattern all : 0.0007s
SCBA: Sparsity Pattern : 0.0006s
SCBA: Sparsity Pattern all : 0.0006s
      ElectronSolver: Assemble : 0.2277s
      ElectronSolver: Assemble all : 0.2277s
      ElectronSolver: Band edges : 0.0236s
      ElectronSolver: Band edges all : 0.0236s
      ElectronSolver: OBC : 4.7030s
      ElectronSolver: OBC all : 4.7030s
      ElectronSolver: Solve : 1.1370s
      ElectronSolver: Solve all : 1.1370s
      ElectronSolver: Filter : 0.0017s
      ElectronSolver: Filter all : 0.0017s
    ElectronSolver : 6.0935s
    ElectronSolver all : 6.0935s
    SCBA: G observables : 0.0244s
    SCBA: G observables all : 0.0244s
    SCBA: stack->nnz transpose : 0.1931s
    SCBA: stack->nnz transpose all : 0.1931s
...
```

The indentation indicates the hierarchy of the different parts of the
simulation, i.e., the line `ElectronSolver: Assemble : 0.2277s` is
accounted for in the total time of `ElectronSolver : 6.0935s`. The word
`all` means that the timing occurs after synchronization across all MPI
processes, while the lines without `all` indicate the timing for only
rank 0.

