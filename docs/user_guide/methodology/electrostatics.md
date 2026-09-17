# Electrostatics

Solving the Poisson equation self-consistently with the open-boundary
Schrödinger equation is a key part of `quatrex`'s quantum transport
formalism. By inducing a non-equilibrium charge density in the device,
the electrostatic potential will change, which in turn changes the
charge density. This feedback loop is solved self-consistently until
convergence.

## Excess Charge Approach

!!! danger "No Electrostatics Support for Metallic Systems"
    The excess charge approach to electrostatics and the automatic
    computation of [contact Fermi levels](#contact-chemical-potentials)
    are ***only applicable in semiconductors***. In principle a full
    charge model could be used (at least for simulations without
    scattering) but this is not implemented in `quatrex`.

We employ an excess charge approach in the electrostatic model
implemented in `quatrex`. The ***excess charge density***
$\rho(\mathbf{r})=n(\mathbf{r}) - p(\mathbf{r})$ is computed from the
occupied/depleted electronic states around a charge neutrality level
$E_{CNL}(\mathbf{r})$. For NEGF we can write

$$
n(\mathbf{r}) \equiv - 2_\mathrm{spin} \int_{E_{CNL}(\mathbf{r})}^\infty
\frac{dE}{2\pi} \operatorname{Im} \left\{ G^{<}(\mathbf{r}, \mathbf{r},
E) \right\}
$$

$$
p(\mathbf{r}) \equiv 2_\mathrm{spin}
\int_{-\infty}^{E_{CNL}(\mathbf{r})}  \frac{dE}{2\pi}
\operatorname{Im}\left\{ G^{>}(\mathbf{r}, \mathbf{r}, E) \right\}
$$

<!-- NOTE: Included as snippet for dynamic coloring -->
--8<-- "docs/assets/images/electrostatics/excess_charge.svg"

/// figure-caption | #excess-charge
Illustration of excess electron and hole charge densities around the
band gap of a device with a potential drop along it's transport
direction $\mathbf{r}$. The charge neutrality level
$E_{CNL}(\mathbf{r})$ is shown in the middle of the band gap in green.
///

For the wave function transport formalism we get a very similar formula,
where instead of the lesser/greater Green's functions we have the
contact density of states multiplied by the respective Fermi-Dirac
occupancies.

In either case, the electrostatic potential $\phi(\mathbf{r})$ is then
obtained via the Poisson equation

$$
\nabla \cdot \left (\varepsilon(\mathbf{r}) \nabla
\phi(\mathbf{r})\right) = -\rho(\mathbf{r}),
$$

where $\varepsilon(\mathbf{r})$ is the dielectric permittivity. We
iterate between computing transport and electrostatics until reaching
convergence.

!!! tip "Mixing Schemes for Self-Consistent Schrödinger-Poisson Runs"
    To accelerate the convergence of the Schrödinger-Poisson loop one
    can choose from (adaptive) under-relaxation and DIIS-based mixing
    schemes via the [mixing parameters](../parameters/scsp/#mixer)

## Non-linear Poisson Equation

Since the excess charge density is a non-linear functional of the
electrostatic potential, $\rho[\phi(\mathbf{r})]$, the Poisson
problem we solve has a nonlinear (or more precisely a *semilinear*) form

$$
\nabla \cdot \left (\varepsilon(\mathbf{r}) \nabla
\phi(\mathbf{r})\right) = -\rho[\phi(\mathbf{r})].
$$

To tackle this nonlinearity, we employ a **Root-Finding approach** (i.e.
Newton-Raphson, or also *predictor-corrector* scheme in this context).
We look for the roots of the functional

$$
F[\phi(\mathbf{r})] \equiv \nabla \cdot \left (\varepsilon(\mathbf{r}) \nabla
\phi(\mathbf{r})\right) + \rho[\phi(\mathbf{r})]
$$

via the recursion relation

$$
\begin{aligned}
\phi_{n+1}(\mathbf{r}) &= \phi_{n}(\mathbf{r}) - \left( \frac{\delta
F[\phi_{n}(\mathbf{r})]}{\delta \phi_{n}(\mathbf{r})} \right) ^{-1}
F[\phi_{n}(\mathbf{r})] \\
&= \phi_{n}(\mathbf{r}) - \underbrace{ \left( \nabla \cdot
\left(\varepsilon(\mathbf{r}) \nabla (\cdot) \right) + \frac{\delta
\rho[\phi_{n}(\mathbf{r})]}{\delta \phi_{n}(\mathbf{r})}\right) ^{-1}
F[\phi_{n}(\mathbf{r})] }_{ \Delta \phi_{n}(\mathbf{r}) }.
\end{aligned}
$$

!!! info "Electrostatics Solving Scheme"
    The solution scheme for the electrostatics can be controlled through
    the [`solving_scheme`](../parameters/electrostatics/#solving_scheme)
    parameter. However, the `"direct"` method (no inner Newton-Raphson)
    is typically unstable and is intended mainly for testing purposes.

At every step of this scheme we need to evaluate
$\rho[\phi_n(\mathbf{r})]$ and $\frac{\delta
\rho[\phi_{n}(\mathbf{r})]}{\delta \phi_{n}(\mathbf{r})}$. Since
evaluating this accurately using the coupled transport formalism would
be *very* costly, we opt to model the charge density's response to
potential variations by making a few approximations.

### Density Response Models

By approximating the electronic structure around our charge neutrality
level by *single parabolic bands* (one each for electrons and for holes)
we can write

$$
n(\mathbf{r}) = N_{ND}(\mathbf{r}) \mathcal{F}_{k}(\eta_{n}(\mathbf{r})).
$$

Here $N_{ND}(\mathbf{r})$ is the *effective density of states* of an
$N$-dimensional system, $\eta_{n}(\mathbf{r})$ is a reduced
electrochemical potential describing the distance between the band onset
and the CNL, and $\mathcal{F}_{k}(\eta)$ is the *complete Fermi-Dirac
integral* of order $k = N / 2 - 1$:

$$
\mathcal{F}_{k}(\eta) = \frac{1}{\Gamma(k+1)} \int_{0}^{\infty}
\frac{x^{k}}{1 + \exp(x - \eta)} dx
$$

See also the very useful *"Notes on Fermi-Dirac
Integrals"*[^fermi-integrals].

[^fermi-integrals]: R. Kim et al., *Notes on Fermi-Dirac Integrals*.
    https://arxiv.org/abs/0811.0116

<!-- NOTE: Included as snippet for dynamic coloring -->
--8<-- "docs/assets/images/electrostatics/density_model.svg"

/// figure-caption | #density-model
Illustration showing the relationship between the charge neutrality
level, electrostatic potential, and the charge density. The electronic
density of states is approximated by a single effective parabolic band,
and the charge density is computed by evaluating a Fermi-Dirac integral
of order $k$ (depends on system dimensionality).
///

Evaluating the charge density and its derivative with respect to the
potential in this picture boils down to evaluating Fermi-Dirac
integrals:

$$
\rho[\phi(\mathbf{r})] \sim \mathcal{F}_{k}(\eta[\phi(\mathbf{r})])
$$

$$
\frac{\delta \rho[\phi(\mathbf{r})]}{\delta \phi(\mathbf{r})} \sim
\mathcal{F}_{k-1}(\eta[\phi(\mathbf{r})])
$$

where we can use the fact that the derivative of a Fermi-Dirac integral
of order $k$ is another Fermi-Dirac integral of order $k-1$.

We also need to determine a CNL that is consistent with this model.
Since the since the CNL directly depends on the potential, we can do so
by inverting the relationship between density and potential:

$$
\rho[\phi(\mathbf{r})] \sim \mathcal{F}_{k}(\eta[\phi(\mathbf{r})])
\longleftrightarrow E_{CNL}(\mathbf{r}) \sim
\mathcal{F}_{k}^{-1}(u[\rho(\mathbf{r})])
$$

While in two-dimensional systems ($N=2$) the Fermi-Dirac integral, its
derivative, and its inverse have an analytic form, they have to be
evaluated numerically for density models with other dimensionalities.

!!! tip "Density Response Model Dimensionality"
    The
    [`density_model_dim`](../parameters/electrostatics/#density_model_dim)
    parameter controls the dimensionality of the density model. Note
    that this does not have to be the same as the actual dimensionality
    of the system, e.g., a 2D model can work very well for 1D systems.

!!! info "Fermi-Dirac Integral Evaluation"
    In `quatrex`, Fermi-Dirac integrals of orders $k=0$ and $k=-1$ are
    evaluated via their analytic form, while they are computed by
    quadrature for other orders. Evaluating inverses of Fermi-Dirac
    integrals without analytic representations would normally be more
    costly optimization problems, but in `quatrex` efficient piecewise
    rational approximation schemes for the relevant orders $k=1/2$
    [^plus-half] and $k=-1/2$ [^minus-half] are implemented.

[^plus-half]: T. Fukushima, *Precise and fast computation of inverse
    Fermi-Dirac integral of order 1/2 by minimax rational function
    approximation*. https://doi.org/10.1016/j.amc.2015.03.015

[^minus-half]: T. Fukushima, *Analytical computation of inverse
    Fermi-Dirac integral of order -1/2 by piecewise rational function
    approximation*. https://doi.org/10.13140/RG.2.2.15176.88325

## Contact Chemical Potentials

In quantum transport simulations, we exert control over the system
through its contacts. In self-consistent Schrödinger-Poisson (SCSP)
runs, determining appropriate and physically consistent contact chemical
potentials is therefore very important.

In `quatrex`, the most appropriate way of configuring contacts for SCSP
runs, is to set a [contact's bias
voltage](../parameters/contact.md#voltage) $V_b$ and a [mid-gap
energy](../parameters/contact.md#mid_gap_energy) (this could, for
instance, be the contacts DFT Fermi level) and then have `quatrex`
compute the contact's Fermi level $E_F$ and in turn its chemical
potential $\mu = E_F - V_b$ from this.

??? info "Automatic Fermi Level Calculation"
    In `quatrex`, a contact's Fermi level is automaticaly determined
    from its band structure, $\epsilon_{\mathbf{k}}$, given by

    $$
    \mathbf{h}(\mathbf{k}) \psi_{\mathbf{k}} = \epsilon_{\mathbf{k}}
    \mathbf{s}(\mathbf{k}) \psi_{\mathbf{k}}
    $$

    where $\mathbf{h}(\mathbf{k})$ and $\mathbf{s}(\mathbf{k})$ are the
    contact Hamiltonian and overlap matrices, on a grid of
    $\mathbf{k}$-points both in transport direction and in the
    transverse periodic directions. The band structure is then separated
    into conduction and valence states
    $\epsilon_\mathbf{k}^C$/$\epsilon_\mathbf{k}^V$ by comparing
    $\epsilon_\mathbf{k}$ to the provided mid-gap energy.

    The Fermi level is computed by minimizing

    $$
    E_F = \underset{E}{\arg\min} \left( \frac{2_{spin}}{2\pi V}\int
    d\mathbf{k} \left[
    f_{FD}(\epsilon^C_{\mathbf{k}} - E )
    - f_{FD}(E - \epsilon^V_{\mathbf{k}}) \right] - N_{A/D} \right)
    $$

    where $f_{FD}$ is the Fermi-Dirac occupancy, $V$ the contact cell
    volume, and $N_{A/D}$ the doping density in the contact.

## Connecting Real-Space and Localized Orbital Basis Sets

In the transport part of an SCSP run, we operate on quantities expressed
in a basis of localized orbitals. The electrostatics, on the other hand
are treated in (discretized) real-space. Taking the Green's function as
an example, the transformation from real-space to a localized orbital
basis $\{\psi_i\}$ is

$$
G_{mn}(E) = \int d^{3}\mathbf{r} \int  d^{3}\mathbf{r'}
\, \psi^{*}_{m}(\mathbf{r})G(\mathbf{r}, \mathbf{r'};
E) \psi_{n}(\mathbf{r'})
$$

and the corresponding inverse projection is

$$
G(\mathbf{r}, \mathbf{r'}; E) = \sum_{m,n} \hat{\psi}_{m}(\mathbf{r})
G_{mn}(E) \hat{\psi}^{*}_{n}(\mathbf{r'}),
$$

where $\hat{\psi}_{m} = \sum_k \left(\mathbf{S}^{-1}\right)_{mk}
\psi_{k}$ are the dual basis states for the general case of an
non-orthonormal basis.

!!! warning "Proper Real-Space Projections Not Implemented Yet"
    The real-space projections of the Green's functions and the charge
    density are not yet implemented in `quatrex`. For now, we use a
    Mulliken charge projection scheme to compute the charge density in
    real-space from the localized orbital basis.

### Mulliken Charge Projection

We employ a Mulliken charge analysis to construct the real-space charge
density. In the case of non-orthonormal basis sets this can be found by
matrix multiplication of the density matrix $\boldsymbol{\rho}$ with the
overlap matrix $\mathbf{S}$.

$$
n(\mathbf{r}) = \sum_m \left(\boldsymbol{\rho}\mathbf{S}\right)_{mm}
\delta(\mathbf{r} - \mathbf{R}_{m})
$$

### Finite Element Discretization

In `quatrex` we discretize the Laplacian using linear (first-order)
tetrahedral finite elements as implemented in
[`scikit-fem`](https://scikit-fem.readthedocs.io/en/latest/). By default
natural boundary conditions are used. Parameters concerning regions in
the real-space simulation domain and the meshing process (using
[`gmsh`](https://gmsh.info/)) are set in the
[`[device.geometry]`](../parameters/geometry/) section of the
configuration.

The `structure.xyz` input file containing the atom/orbital coordinates
that make up the device, serves as a reference point for all defined
geometry regions. This file also informs the periodicity in transverse
directions through the `pbc="..."` entry on the second line in the
extended `.xyz` file format. Setting `pbc="F F F"`, the real-space mesh
will not enforce any periodicity, while `pbc="T F F"` will lead to
periodic boundary conditions being enforced along $x$-direction.

Each [3D region](../parameters/volume_properties/) can define properties
such as relative permittivity, and doping concentrations, while [2D
regions](../parameters/surface_properties) in the simulation domain act
as gates/ground planes with work functions and set voltages. Orbital
sites are embedded into the mesh to accomodate the Mulliken charge
projection.

!!! tip "`quatrex mesh` Command"
    The [`quatrex mesh`](../cli/#quatrex-mesh) command has to be invoked
    for the simulation configuration before starting an SCSP run. Using
    `gmsh`, this command sets up the real space simulation domain and
    meshes it, accounting for all defined regions and the periodicity in
    transverse directions. This command will also visualize the defined
    simulation domain (either interactive or off-screen, controlled via
    the `--off-screen` flag).

### Dirichlet Boundary Conditions

We model electrostatic control from gates in the simulation by imposing
Dirichlet boundary conditions. The actual Dirichlet boundary condition
entering `quatrex`'s' Poisson solver is not directly the
[`voltage`](../parameters/surface_properties/#voltage) parameter
$V_{\mathrm{gate}}$ set for that surface. Instead, we also have to
consider the electrostatic alignment of semiconductor channel and the
metallic gate contact. The actual Dirichlet boundary condition entering
the Poisson problem is

$$
\phi_\mathrm{gate} = -V_\mathrm{gate} + \Phi_\mathrm{gate} -
\chi_\mathrm{channel} - \Delta E_{\mathrm{CB}-F},
$$

where $\Phi_\mathrm{gate}$ is the metal's
[`work_function`](../parameters/surface_properties/#work_function),
$\chi_\mathrm{channel}$ is the semiconductor channel's
[`electron_affinity`](..parameters/electrostatics/#electron_affinity),
and $\Delta E_{\mathrm{CB}-F} = E_\mathrm{CB} - E_\mathrm{F}$ is the
distance between Fermi level and conduction band in the semiconductor.

### Multifreedom Constraints for Periodic Boundaries

Since `gmsh` supports the creation of periodic meshes, we can make use
of multifreedom constraints (MFC)[^mfc] to enforce periodic boundary
conditions straightforwardly. We construct a map from one surface to its
image and then assemble an MFC transformation matrix that couples the
image degrees of freedom to the authoritative ones.

[^mfc]: Section 8 in C. A. Felippa, *Introduction to Finite Element
    Methods*.
