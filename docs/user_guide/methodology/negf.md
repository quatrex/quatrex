# The Non-Equilibrium Green's Function Formalism

To include scattering processes in quantum transport simulations, we
need to go beyond the coherent transport formalism of the
[QTBM](qtbm.md) and use the *non-equilibrium Green's function (NEGF)
formalism*. This formalism allows us to include interactions with
phonons, photons, and other electrons on the same theoretical footing.

Because the scattering self-energies depend on the Green's functions
(and vice versa), the two must be solved together. This is done using
the self-consistent Born approximation (SCBA):

$$
\mathbf{G}_0\left[ \mathbf{\Sigma} = 0 \right] \rightarrow
\mathbf{\Sigma}_0\left[ \mathbf{G}_0\right] \rightarrow
\mathbf{G}_1\left[ \mathbf{\Sigma}_0 \right] \rightarrow
\mathbf{\Sigma}_1\left[ \mathbf{G}_1\right] \rightarrow
\mathbf{G}_2\left[ \mathbf{\Sigma}_1 \right] \rightarrow \cdots
$$

We iterate between computing the Green's functions from the current
self-energies, and computing the self-energies from the current Green's
functions, until the two are consistent with one another. Convergence of
this cycle ensures that energy and momentum are conserved throughout the
device.

The rest of this page briefly works through the pieces of that loop: The
[Dyson and Keldysh equations](#dyson-and-keldysh-equations) that define
the Green's functions, the general structure of the [interaction
terms](#scattering-terms), and finally the [interactions `quatrex`
currently supports](#interacting-systems).

## Dyson and Keldysh Equations

Under non-equilibrium conditions, the occupation of states is no longer
simply given by the Fermi-Dirac distribution, and is instead encoded in
the *lesser* and *greater* Green's functions, $\mathbf{G}^{<}(E,
\mathbf{k})$ and $\mathbf{G}^{>}(E, \mathbf{k})$, which respectively
describe occupied and unoccupied states at energy $E$ and momentum
$\mathbf{k}$. To compute these Green's functions, we first need to
compute the retarded and advanced Green's functions, $\mathbf{G}^{R}(E,
\mathbf{k})$ and $\mathbf{G}^{A}(E, \mathbf{k})$, which describe the
density of states. The retarded Green's function $\mathbf{G}^R(E,
\mathbf{k})$ is computed from the Dyson equation

$$
\left[E\mathbf{S}(\mathbf{k}) - \mathbf{H}(\mathbf{k}) -
\mathbf{\Sigma}^R(E, \mathbf{k})\right] \mathbf{G}^R(E, \mathbf{k}) =
\mathbf{I},
$$

where $\mathbf{H}(\mathbf{k})$ and $\mathbf{S}(\mathbf{k})$ are the
device Hamiltonian and overlap matrices at a specific
$\mathbf{k}$-point, $\mathbf{I}$ is the identity, and
$\mathbf{\Sigma}^R(E, \mathbf{k})$ is the retarded self-energy
describing all [scattering processes](#interacting-systems), as well as
the [open boundaries](obc.md). The advanced Green's function is given as
$\mathbf{G}^A(E, \mathbf{k}) = [\mathbf{G}^R(E, \mathbf{k})]^{\dagger}$.

The lesser and greater Green's functions are then computed from the
Keldysh equation

$$
\mathbf{G}^{\lessgtr}(E, \mathbf{k}) = \mathbf{G}^R(E, \mathbf{k})
\mathbf{\Sigma}^{\lessgtr}(E, \mathbf{k}) \mathbf{G}^A(E, \mathbf{k}),
$$

where $\mathbf{\Sigma}^{\lessgtr}(E, \mathbf{k})$ are the lesser and
greater self-energies associated with the same scattering processes and
contacts. The lesser/greater contact for electrons are computed from the
retarded contact self-energy as described in the [OBC
section](obc.md#lessergreater-open-boundary-self-energy).

In practice, the Dyson and Keldysh equations are solved in a single
step, rather than one after the other, by recasting them as a single
system of equations, which is solved using our implementation of the
recursive Green's function (RGF) algorithm.

## Scattering Terms

All lesser/greater self-energies, Green's functions, and polarizations
have skew-hermitian symmetry, i.e., $\mathbf{B}^{\lessgtr} =
-[\mathbf{B}^{\lessgtr}]^{\dagger}$.
The retarded self-energies and polarizations contain both a
skew-hermitian part that describes the scattering rate and a Hermitian
part that describes the energy renormalization due to the interaction.
The Hermitian part can be computed from the skew-Hermitian part using
the Kramers-Kronig relation, which is implemented in `quatrex` through a
Hilbert transform. The retarded scattering self-energy is given by

$$
\mathbf{\Sigma}^{R}(E, \mathbf{k}) = \frac{1}{2} \left
[\mathbf{\Sigma}^{>}(E, \mathbf{k}) - \mathbf{\Sigma}^{<}(E,
\mathbf{k})\right] + \frac{1}{2\pi i} \mathcal{P} \int dE'
\frac{\mathbf{\Sigma}^{>}(E', \mathbf{k}) - \mathbf{\Sigma}^{<}(E',
\mathbf{k})}{E - E'}
$$

where $\mathcal{P}$ denotes the [Cauchy principal
value](https://en.wikipedia.org/wiki/Cauchy_principal_value) of the
integral. The retarded polarization for the screened Coulomb interaction
is computed analogously.

!!! info "Screened Coulomb Interaction"
    Whether the principal value integral is actually evaluated is
    controlled by the
    [`include_energy_renormalization`](../parameters/coulomb_screening.md#include_energy_renormalization)
    parameter.

With $\mathbf{\Sigma}^{\lessgtr}(E, \mathbf{k})$ and
$\mathbf{\Sigma}^{R}(E, \mathbf{k})$ in hand, the loop closes: They
enter the Dyson and Keldysh equations above to give updated Green's
functions, from which new self-energies are computed. The following
sections describe how $\mathbf{\Sigma}^{\lessgtr}(E, \mathbf{k})$ itself
is obtained for the interactions `quatrex` currently supports.

## Interacting Systems

### Screened Coulomb Interaction

The Coulomb interaction is the longitudinal part of the electromagnetic
interaction. It has an analytic (bare) form $\mathbf{V}$ in vacuum.
Interaction with the electron ensemble in the device modifies this bare
Coulomb interaction into the *screened* Coulomb interaction
$\mathbf{W}$, obtained from its own Dyson/Keldysh pair:

$$
\begin{align}
\left[ \mathbf{I} - \mathbf{V}(\mathbf{k})\mathbf{P}^R(E, \mathbf{k})
\right] \mathbf{W}^R(E, \mathbf{k}) = \mathbf{V}(\mathbf{k}) \\
\mathbf{W}^{\lessgtr}(E, \mathbf{k}) = \mathbf{W}^R(E, \mathbf{k})
\mathbf{P}^{\lessgtr}(E, \mathbf{k}) \mathbf{W}^A(E, \mathbf{k})
\end{align}
$$

where $\mathbf{P}^{\lessgtr}(E, \mathbf{k})$ is the longitudinal
polarization function, constructed as

$$
\mathbf{P}^{\lessgtr}_{ij}(E, \mathbf{k}) = \frac{i}{2\pi \hbar} \iint
\mathbf{G}^{\lessgtr}_{ij}(E', \mathbf{q})
\left[\mathbf{G}^{\gtrless}_{ij}(E' - E, \mathbf{q} -
\mathbf{k})\right]^{\dagger} \;dE'd\mathbf{q}.
$$

The retarded polarization is obtained from the lesser/greater parts
through the Kramers-Kronig relation, as described
[above](#scattering-terms).

The screened Coulomb interaction $\mathbf{W}^{\lessgtr}(E, \mathbf{k})$
then enters the GW self-energy though the following convolution with the
Green's functions:

$$
\mathbf{\Sigma}^{\lessgtr, GW}_{ij}(E, \mathbf{k}) = \frac{i}{2\pi
\hbar} \iint \mathbf{G}^{\lessgtr}_{ij}(E - E', \mathbf{k} - \mathbf{q})
\mathbf{W}^{\lessgtr}_{ij}(E',\mathbf{q}) \;dE'd\mathbf{q}
$$

Because $\mathbf{G}^{R,\lessgtr}(E, \mathbf{k}), E \in [E_0, E_N]$ and
$\mathbf{W}^{R,\lessgtr}(E, \mathbf{k}), E \in [0, E_N - E_0]$, the
integration above is done in two parts: First the convolution over the
positive energies, followed by the convolution over the negative
energies where the energy reversal symmetry
$\mathbf{W}^{\lessgtr}_{ij}(E, \mathbf{k}) =
-\left[\mathbf{W}^{\gtrless}_{ij}(-E, \mathbf{k})\right]^{\dagger}$ is
used. To compute the convolution efficiently, `quatrex` uses a fast
Fourier transform (FFT) to convert the convolution into a multiplication
in the time domain, and then transforms back to the energy domain.

In addition, to the GW self-energy, the screened Coulomb interaction
also adds the retarded Fock self-energy, which is computed as

$$
\mathbf{\Sigma}^{R, F}_{ij}(\mathbf{k}) =  \frac{i}{2\pi \hbar} \iint
\mathbf{V}_{ij}(\mathbf{q}) \, \mathbf{G}_{ij}^<(E', \mathbf{k} -
\mathbf{q}) \;dE'd\mathbf{q}
$$

In practice, computing $\mathbf{W}^R$ explicitly is unnecessary: to
obtain the GW self-energy, `quatrex` only needs $\mathbf{W}^{\lessgtr}$,
and it solves for this directly by rewriting the Keldysh equation for
$\mathbf{W}$ in terms of an effective lesser/greater polarization
$\mathbf{L}^{\lessgtr}$ and retarded "screening" $\mathbf{X}^R$:

$$
\mathbf{W}^{\lessgtr}(E, \mathbf{k}) = \underbrace{\left[ \mathbf{I} -
\mathbf{V}(\mathbf{k})\mathbf{P}^R(E, \mathbf{k})
\right]^{-1}}_{\mathbf{X}^{R}(E, \mathbf{k})}
\;\underbrace{\mathbf{V}(\mathbf{k}) \mathbf{P}^{\lessgtr}(E,
\mathbf{k}) \mathbf{V}^{\dagger}(\mathbf{k})}_{\mathbf{L}^{\lessgtr}(E,
\mathbf{k})} \; \underbrace{\left[ \mathbf{I} -
\mathbf{V}(\mathbf{k})\mathbf{P}^R(E, \mathbf{k})
\right]^{-\dagger}}_{\left[\mathbf{X}^{R}(E,
\mathbf{k})\right]^{\dagger}}
$$

!!! note "Bare Coulomb Interaction"
    Note that the bare Coulomb interaction $\mathbf{V}$ can be
    constructed approximately by assuming point charges at the atomic
    positions, or by properly evaluating the Coulomb integrals over the
    basis functions. We are working to integrate both options into
    `quatrex`.

#### Spillover Corrections

Due to the assumption of infinite leads, the system matrix $\mathbf{I} -
\mathbf{V}(\mathbf{k})\mathbf{P}^R(E, \mathbf{k})$ and the right hand
side $\mathbf{V}(\mathbf{k}) \mathbf{P}^{\lessgtr}(E, \mathbf{k})
\mathbf{V}^{\dagger}(\mathbf{k})$ need to be corrected to account for
$\mathbf{V}(\mathbf{k})$ and $\mathbf{P}^{R,\lessgtr}(E, \mathbf{k})$
continuing beyond the system boundaries. This is done by adding the
extra terms coming from "outside" the system to the system matrix and
right hand side in the Keldysh equation.

!!! info "Spillover corrections with unit cell periodicity"
    If the device is periodic in smaller cells, the spillover
    corrections is computed only with the outermost unit cell and its
    hopping matrices. Currently, this is configured through the
    [`block_sections`](../parameters/obc.md#block_sections) parameter in
    the Coulomb Screening solver. This will be refactored and in the
    future, the periodicity will end up either being automatically
    detected or set through a new parameter.

#### Open Boundary Conditions

Same as for the electronic Green's functions, open boundary conditions
are applied to the screened Coulomb interaction by adding the open
boundary normalization to the system matrix. This is detailed in the
[OBC pages](obc.md). Besides retarded boundary conditions,
lesser/greater open boundary conditions are also applied in the
right-hand side of the Keldysh equation, which is explained in the
[Lyapunov](lyapunov.md) sections.

!!! note "OBC with unit cell periodicity"
    As for the spillover corrections, if the device is periodic in
    smaller cells, the open boundary conditions are computed from the
    outermost unit cell and its corresponding hopping matrices.
    Currently, this is configured through the
    [`block_sections`](../parameters/obc.md#block_sections) parameter in
    the Coulomb Screening solver. This will be refactored and in the
    future, the periodicity will end up either being automatically
    detected or set through a new parameter.

### Phonons

Electron-phonon scattering is treated in a more simple picture.
`quatrex` currently implements a
`"pseudo-scattering"` electron-phonon coupling model with a
single optical phonon mode as well as a
`"long-wavelength"` model with multiple modes.

In the `"pseudo-scattering"` model, the implemented phonon self-energy is

$$
\mathbf{\Sigma}^{\lessgtr}_{ph}(E, \mathbf{k}) =
D_{ph}^{2}\!\left[
(N_{ph}+1)\,\mathbf{G}^{\lessgtr}(E-\hbar\omega_{ph}, \mathbf{k})
+ N_{ph}\,\mathbf{G}^{\lessgtr}(E+\hbar\omega_{ph}, \mathbf{k})
\right].
$$

where \(D_{ph}\) is the electron-phonon deformation potential,
$\hbar\omega_{ph}$ is the phonon energy and $N_{ph} =
\left[\exp\left(\hbar\omega_{ph}/k_BT\right) - 1\right]^{-1}$ is its
Bose-Einstein occupancy at temperature $T$.
This model only computes the diagonal entries of
$\mathbf{\Sigma}^{\lessgtr}_{ph}(E, \mathbf{k})$.

In the `"long-wavelength"` model, the self-energy is computed as

$$
\Sigma^\gtrless(E)
= \sum_{E_\mathrm{ph}}
V_{E_\mathrm{ph}}^\mathrm{em}
G^\gtrless(E\mp E_\mathrm{ph})
+ V_{E_\mathrm{ph}}^\mathrm{abs}
G^\gtrless(E\pm E_\mathrm{ph}),
$$

where the prefactors $V_{E_\mathrm{ph}}^\mathrm{em}$ and
$V_{E_\mathrm{ph}}^\mathrm{abs}$ are derived from a provided phonon dispersion.
Here the off-diagonal elements are computed as well.
