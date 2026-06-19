# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
import re
import subprocess
import tomllib
import warnings
from pathlib import Path
from typing import Literal

import numba as nb
import numpy as np
from mpi4py.MPI import COMM_WORLD as mpi_comm_world
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from qttools import xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO, DSDBSparse
from qttools.profiling import Profiler
from quatrex.electrostatics.geometry_config import GeometryConfig, parse_geometry_config

profiler = Profiler()


class SCSPConfig(BaseModel):
    """Parameters controlling the self-consistent Schrödinger-Poisson loop.

    For more information on the self-consistent Schrödinger-Poisson
    loop, see the [section on
    electrostatics](../methodology/electrostatics.md) in the user guide.

    """

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    """The minimum number of Schrödinger-Poisson iterations to perform."""

    max_iterations: PositiveInt = 100
    """The maximum number of Schrödinger-Poisson iterations to perform."""

    convergence_tol: PositiveFloat = 1e-3
    r"""The convergence tolerance for the potential in the
    Schrödinger-Poisson loop.

    This is defined as the infinity norm of the difference between the
    potential in the current iteration and the previous iteration.

    $$
        \lVert V_{n} - V_{n-1} \rVert_{\infty} < \texttt{convergence_tol}
    $$

    """

    # Parameters for potential mixing.
    mixer: Literal["under-relaxation", "diis"] = "under-relaxation"
    """The mixing scheme to use for the self-consistent solution of the
    Poisson equation.

    - `"under-relaxation"`: Simple under-relaxation scheme where the new
      potential is a weighted average of the previous potential and the
      newly computed potential. The weight is given by the
      `mixing_factor` parameter.
    - `"diis"`: Direct inversion in the iterative subspace (DIIS) method
      which constructs the new potential as a linear combination of the
      previous potentials and the newly computed potential. The
      coefficients of the linear combination are determined by
      minimizing the residuals of the previous potentials.

    """

    mixing_factor: PositiveFloat = Field(default=0.75, le=1.0)
    """Under-relaxation factor for the under-relaxation mixer. Should be
    between 0 and 1.

    """

    adaptive_mixing: bool = False
    """Whether to adaptively adjust the mixing factor based on the
    convergence behavior.

    If `True`, the mixing factor is adjusted based on the convergence
    behavior. If the residual between two potential iterations is larger
    than the previous iteration, the mixing factor is reduced by 50%. If
    it is smaller, the mixing factor is increased by 10%.

    """

    max_history: PositiveInt = 3
    """Maximum number of previous potentials and residuals to store for
    the DIIS extrapolation.

    Only used if `mixer` is set to "diis".

    """

    epsilon: PositiveFloat = 1e-5
    """Regularization parameter for the least-squares problem in the
    DIIS method to ensure numerical stability.

    Only used if `mixer` is set to "diis".

    """

    extrapolation_interval: PositiveInt = 1
    """Number of iterations between DIIS extrapolation steps.

    For example, if set to 3, the mixer will perform two
    under-relaxation steps followed by a DIIS extrapolation step, and
    then repeat this cycle. If set to 1 (the default), the Pulay mixing
    is performed at every iteration.

    Only used if `mixer` is set to "diis".

    """


class QTBMConfig(BaseModel):
    """Parameters for the quantum transmitting boundary method (QTBM).

    !!! note
        Only used in simulations where `formalism = "wf"`.

    """

    model_config = ConfigDict(extra="forbid")

    max_batch_size: PositiveInt = 10
    """The maximum number of energies that are batched together when
    computing open boundary conditions (OBCs) in the QTBM solver.

    This can be used to reduce the memory footprint of the QTBM solver,
    at the cost of increased computation time.

    """

    low_rank_obc: bool = False
    """Whether to use reduced rank for the boundary self-energies.
    
    If set to True, boundary self-energies are moved to the
    right-hand-side of the linear system, which greatly reduces fill-in
    during factorization.

    The system matrix becomes Hermitian or even real symmetric in
    gamma-only simulations. Therefore, the `low_rank_obc` parameter can
    only be used in combination with direct solvers that can exploit the
    symmetry, i.e., `direct_solver="cudss"` on GPU,
    `direct_solver="pardiso"` on CPU, and `direct_solver="thomas"` on
    both CPU and GPU.

    """


class SCBAConfig(BaseModel):
    """Parameters for the self-consistent Born approximation (SCBA)
    loop.

    This is the main loop that computes the self-energies and Green's
    functions in simulations where `formalism = "negf"`.

    See the [section on NEGF](../methodology/negf.md)
    in the user guide for more information on the SCBA loop.

    """

    model_config = ConfigDict(extra="forbid")

    min_iterations: PositiveInt = 1
    """The minimum number of SCBA iterations to perform.

    This must be greater than or equal to 1.

    !!! warning
        This parameter currently has no effect.

    """

    max_iterations: PositiveInt = 100
    """The maximum number of SCBA iterations to perform."""

    convergence_tol: PositiveFloat = 1e-5
    """The convergence tolerance for the SCBA iterations.

    !!! warning
        This parameter currently has no effect.

    """

    mixing_factor: PositiveFloat = Field(default=0.1, le=1.0)
    r"""The under-relaxation factor for the SCBA iterations.

    The new self-energy is computed as a weighted average of the
    previous self-energy and the new self-energy.

    $$
    \mathbf{\Sigma}^{(n)} = (1 - \text{mixing_factor})
    \mathbf{\Sigma}^{(n-1)} + \text{mixing_factor}
    \mathbf{\Sigma}^{\text{new}}
    $$

    """

    output_interval: PositiveInt = 1
    """The interval at which to output observables during the SCBA iterations.

    !!! warning
        This parameter currently has no effect.

    """

    coulomb_screening: bool = False
    """Whether to include screened Coulomb interactions."""

    photon: bool = False
    """Whether to include electron-photon interactions."""

    phonon: bool = False
    """Whether to include electron-phonon interactions."""

    symmetric: bool = False
    """Whether to exploit symmetry in NEGF calculations.

    All lesser and greater quantitiese are skew-Hermitian, allowing us
    to only store and compute the upper triangular part of the matrices.

    The retarded quantities can be decomposed into a Hermitian and
    skew-Hermitian part, which also allows memory and computation
    savings.

    This can reduce the memory footprint and computation time by a
    significant factor, especially for large systems.

    """

    align_self_energy_to_complex_axes: bool = True
    r"""Whether to discard certain parts of the self-energy.

    This is an approximation that affects the self-energy in the
    following way:

    - The real parts of the lesser/greater self-energy are discarded.
    - The imaginary part of the retarded self-energy from any previous
      computation is discarded.

    This happens before the anti-Hermitian part of the retarded
    self-energy is computed from the lesser and greater parts as

    $$
    \mathbf{\Sigma}^R_{AH} = \frac{1}{2i} ( \mathbf{\Sigma}^> -
    \mathbf{\Sigma}^< )
    $$

    """


class ElectrostaticsConfig(BaseModel):
    """Parameters for the electrostatics calculations."""

    model_config = ConfigDict(extra="forbid")

    orbital_basis: Literal["point-charge"] = "point-charge"
    """The orbital basis to use to transform between the real-space and
    orbital-space representations of the potential and charge density.

    Currently, only the "point-charge" basis is supported. Each orbital
    is represented as a point charge located at the corresponding atomic
    position.

    """

    solving_scheme: Literal["root-finding", "direct"] = "root-finding"
    """The scheme to solve the non-linear Poisson equation.

    - `"root-finding"`: Solves the Poisson equation using an iterative
      predictor-corrector scheme where the charge density response is
      computed from the potential using a density model and the Poisson
      equation is solved iteratively until convergence.
    - `"direct"`: Solves the Poisson equation directly using a linear
      solver. Due to the non-linearity of the Schrödinger-Poisson
      problem, this scheme is not recommended and should only be used
      with very cautious mixing and a good initial guess.

    """

    max_iterations: PositiveInt = 20
    """The maximum number of inner iterations for the root-finding scheme.

    Only used if `solving_scheme` is set to "root-finding".

    """

    convergence_tol: PositiveFloat = 1e-3
    """The convergence tolerance for the root-finding scheme.

    This is defined as the infinity norm of the potential update in the
    root-finding scheme.

    Only used if `solving_scheme` is set to "root-finding".

    """

    density_model: Literal["single-band", "omen"] = "single-band"
    """The density model to use for the root-finding scheme.

    - `"single-band"`: Uses a simple single-band density model where the
      charge density is computed from the potential using a single-band
      approximation.
    - `"omen"`: Uses the density model from the OMEN code. This is
      almost identical to the single-band model with `density_model_dim
      = 2`. However, it uses slightly different physical constants,
      i.e., not the CODATA values used everywhere else in `quatrex`.

    Only used if `solving_scheme` is set to "root-finding".

    """

    density_model_dim: Literal[1, 2, 3] = 2
    """The dimensionality of the system to use for the single-band
    density model.

    The density model does not have to match the actual dimensionality
    of the system. For example, a 2D density model might actually work
    best for systems of all dimensionalities.

    Only used if `solving_scheme` is set to "root-finding" and
    `density_model` is set to "single-band".

    """

    initial_guess: Literal["zero", "constraints", "file"] = "zero"
    """The strategy to generate the initial guess for the potential.

    - `"zero"`: Uses a zero potential as the initial guess.
    - `"constraints"`: Solves a linear Poisson equation with the
        potential constraints to generate the initial guess. This is
        expected to work best at regimes close to equilibrium where the
        potential does not vary too much.
    - `"file"`: Loads the initial guess from a file. The file should be
        located in the `input_dir` and named `potential.npy`.

    """

    default_epsilon_r: PositiveFloat = 1.0
    """The default relative permittivity to use for the Poisson solver.

    This is used as a fallback for regions that do not have a specified
    relative permittivity.

    """

    electron_affinity: float | None = None
    """The electron affinity of the semiconductor channel.

    This is used to align the voltage levels of any gates to the
    semiconductor channel levels in SCSP runs. If not set, the voltages
    of the gates are taken as absolute values without any alignment,
    i.e. they are directly used as the Dirichlet boundary conditions for
    the Poisson equation.

    """


class MemoizerConfig(BaseModel):
    """Parameters for memoizing wrappers.

    The memoizers store and reuse previously computed results
    to speed up the fixed-point iterations in OBC and Lyapunov solvers.

    See the [section on open boundary conditions](../methodology/obc.md)
    in the user guide for more information on the

    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "force", "force-after-first", "off"] = "auto"
    """The memoization mode determines when to reuse cached results.

    - `"auto"`: Automatically decides whether to use memoization based
      on the specified tolerances. This will only lead to iteration
      savings when all ranks agree to memoize. This incurs a small
      amount of communication overhead.
    - `"force"`: Always use memoization on all ranks.
    - `"force-after-first"`: Use memoization after the first SCBA
      iteration.
    - `"off"`: Never use memoization.

    """

    num_ref_iterations: PositiveInt = Field(default=2, ge=2)
    """The number of fixed-point iterations to perform.

    This must be greater than or equal to 2. The first iteration is used
    to estimate the residuals, and a second fixed-point iteration is
    performed to get the residuals after the memoization is applied.

    """

    relative_tol: PositiveFloat = 2e-1
    """The relative tolerance on fixed-point residuals for memoization.

    !!! note
        Only used if `mode` is set to `"auto"`.

    """

    absolute_tol: PositiveFloat = 1e-6
    """The absolute tolerance on fixed-point residuals for memoization.

    !!! note
        Only used if `mode` is set to `"auto"`.

    """

    warning_threshold: PositiveFloat = 1e-1
    """The threshold for issuing a memoization warning.

    If the memoized functions residual is above this value after the
    fixed-point iterations, a warning is issued. This is to alert the
    user that the memoization may not be accurate enough and that the
    results may be unreliable.

    """

    agreement_threshold: float = Field(default=0.999, ge=0, le=1)
    """The threshold for agreement between ranks for memoization.

    The default value of 0.999 means 99.9% of the ranks must agree to
    use memoization.

    !!! note
        Only used if `mode` is set to `"auto"`.

    """


class SolverConfig(BaseModel):
    """Options for the system solver."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["rgf", "inv"] = "rgf"
    """The algorithm to use for the system solver.

    - `"rgf"`: Uses the recursive Green's function (RGF) algorithm to
      compute the Green's functions. This is the default.

    - `"inv"`: Uses a direct matrix inversion to compute the Green's
      functions. This is mainly useful for debugging and testing, as it
      is not efficient for realistically sized systems.

    """

    max_batch_size: PositiveInt = 100
    """The maximum number of energies that are batched together when
    computing the Green's functions in the system solver.

    This can be used to reduce the memory footprint of the system
    solver, at the cost of increased computation time.

    """

    compute_current: bool | None = None
    """Whether to compute the current via the Meir-Wingreen formula.

    This is only supported for the `"rgf"` algorithm. If not set, it is
    automatically determined based on the algorithm. (i.e. `True` for
    `"rgf"` and `False` for `"inv"`)

    If `True`, the current is computed between each layer and from/to
    the leads. This way of computing the current is usually preferable
    as it is independet of any interaction cutoffs, since it is computed
    from the temporarily densified Green's functions and self-energies.

    !!! note
        This is parameter is only used in the electron solver. The
        Coulomb screening solver does not compute currents, so this
        parameter is ignored for the Coulomb screening solver.

    """

    direct_solver: Literal[
        "superlu",
        "mumps",
        "cudss",
        "pardiso",
        "thomas",
        "auto",
    ] = "auto"
    """The direct solver to use in `wf` simulations.

    If set to `"auto"`, the solver is automatically chosen based on the
    matrix type and the available direct solver libraries.

    In runs with `low_rank_obc = true`, the system matrix will be
    Hermitian or even real and symmetric in gamma-only simulations. In
    those cases, libraries that can exploit the symmetry are preferred,
    i.e., cuDSS on GPU and PARDISO on CPU.

    On GPU, SuperLU is the only fallback option if cuDSS is not
    available. On CPU, if PARDISO is not available, the fallback options
    are MUMPS and then SuperLU.

    The Thomas solver involves a straight-forward tiling of the system
    matrix into blocks without reordering. It is therefore important
    that the Hamiltonian is ordered in a way that results in a
    block-tridiagonal structure.

    """

    @model_validator(mode="after")
    def set_compute_current(self) -> Self:
        """Sets the `compute_current` parameter based on the algorithm."""
        if self.compute_current is None:
            if self.algorithm == "rgf":
                self.compute_current = True
            else:
                self.compute_current = False

        if self.compute_current and self.algorithm != "rgf":
            raise ValueError(
                "Current computation is only supported for the RGF algorithm."
            )

        return self


class OBCConfig(BaseModel):
    r"""Options for open-boundary conditions (OBCs).

    The OBC solvers compute the surface Green's functions of the
    contacts. The retarded surface Green's function satisfies the
    following recursion relation:

    $$
    \mathbf{g}^R = \left[\mathbf{m}_{0} - \mathbf{m}_{-1} \mathbf{g}^R
    \mathbf{m}_{+1} \right]^{-1},
    $$

    where $\mathbf{m}_{0}$ is the contact Hamiltonian,
    $\mathbf{m}_{-1}$ is the coupling from the device to the contact,
    and $\mathbf{m}_{+1}$ is the coupling from the contact to the
    device. In the NEGF framework, the system matrix $\mathbf{m}$
    includes scattering self-energies.

    More information on the boundary conditions can be found in the user
    guide [section on open boundary conditions](../methodology/obc.md).

    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["sancho-rubio", "spectral"] = "spectral"
    """The algorithm to use when solving the OBC recursion relation.

    - `"sancho-rubio"`: Uses the Sancho-Rubio iterative
      scheme[^sancho-rubio] to compute the surface Green's functions.
      This method achieves exponential convergence compared to the
      linear convergence of fixed-point iterations.

    - `"spectral"`: Uses the specified `nevp_solver` to compute
      eigenpairs of the polynomial contact eigenvalue problem and uses
      them to construct the surface Green's functions. This is generally
      more efficient method when combined with a contour integral NEVP
      solver (`"beyn"`), but can require more parameter tuning.

    [^sancho-rubio]: M. P. Lopez Sancho, et al., 1985 J. Phys. F: Met.
        Phys. 15 851, https://doi.org/10.1088/0305-4608/15/4/009

    """

    nevp_solver: Literal["beyn", "full"] = "beyn"
    r"""The NEVP solver to use for the spectral OBC algorithm.

    The contact eigenvalue problem is a polynomial eigenvalue problem of
    the form:

    $$
    \sum \limits_{n=-b}^{+b} \lambda^{n} \hat{\mathbf{m}}_{n} \mathbf{v}
    = 0,
    $$

    where $b$ is the number of `block_sections`, and
    $\hat{\mathbf{m}}_{n}$ are potentially reduced coupling matrices.

    From selected eigenvalues $\lambda = e^{i k}$ and eigenvectors
    $\mathbf{v}$, the surface Green's functions can be constructed.

    - `"beyn"`: Uses the Beyn's contour integral method[^beyn] to solve
      the NEVP and find the eigenpairs within a specified contour in the
      complex plane. Also see the `r_o`, `r_i`, `m_0`, and
      `num_quad_points` parameters for configuration of the contour
      integral method.

    - `"full"`: Uses a full dense eigensolver to solve for all
      eigenvalues, linearizing the original polynomial problem. This
      results in a doubled problem size which is also not reduced by
      block sectioning or exploiting periodicity.

    !!! note
        Only used if `algorithm` is set to `"spectral"`.

    [^beyn]: W.-J. Beyn, An integral method for solving nonlinear
        eigenvalue problems, Linear Algebra and its Applications, 2012,
        https://doi.org/10.1016/j.laa.2011.03.030.

    """

    block_sections: PositiveInt = 1
    """The number of unit cell blocks along transport direction.

    !!! note
        This is automatically determined in QTBM calculations. Thus it
        only has an effect in NEGF calculations.

    In NEGF calculations, one needs to define block-sizes that lead to a
    block-tridiagonal tiling of the system matrix. These *transport
    blocks* are sometimes constructed from multiple unit cells.

    With the `block_sections` parameter, one can specify how many unit
    cells are merged into a single transport block. This is then used
    when `nevp_solver` is set to `"beyn"` to reduce the size of the
    contact NEVP.

    For example, if the transport cell is constructed from two unit
    cells along the transport direction, setting `block_sections = 2`
    will halve the size of the NEVP. The contact transport blocks need
    to be sorted accordingly.

    """

    min_decay: PositiveFloat = 1e-3
    r"""The minimum rate by which a mode must decay to be considered
    evanescent.

    The decay rate is computed as $\|\mathrm{Im}(k)\|$ where $k$ is the
    complex wavevector of the mode.

    This is used to classify the modes obtained from the spectral OBC
    solver into propagating modes and evanescent modes. Modes with decay
    rates below this threshold are considered propagating.

    """

    max_decay: PositiveFloat | None = None
    r"""The maximum rate a mode can decay while still being considered
    relevant for the surface Green's functions.

    The decay rate is computed as $\|\mathrm{Im}(k)\|$ where $k$ is the
    complex wavevector of the mode.

    Very rapidly decaying modes do not contribute to the surface Green's
    functions and can be neglected. These modes should be filtered out
    as including them can lead to numerical instabilities.

    If `max_decay` is not set, it is computed from the outer contour
    radius as `1.5 * log(r_o)`.

    """

    num_ref_iterations: PositiveInt = 2
    """The number of fixed-point iterations used to refine the surface
    Green's functions.

    This is needed to improve the accuracy of the surface Green's
    functions, especially if not enough eigenpairs are considered.

    !!! note
        Only used if `algorithm` is set to `"spectral"`.

    """

    min_propagation: PositiveFloat = 1e-2
    r"""The minimum group velocity propagation/decay ratio for a mode to
    be considered.

    This ratio is determined by dividing the real part of the group
    velocity by the imaginary part of the group velocity:

    $$
    \mathrm{Re}(\frac{dE}{dk}) / \mathrm{Im}(\frac{dE}{dk}).
    $$

    """

    residual_tolerance: PositiveFloat = 1e-3
    r"""The tolerance on the residual of an eigenpair.

    The eigenpair residuals are computed as by inserting the eigenvalues
    and eigenvectors back into the polynomial eigenvalue problem.

    $$
    \text{residual} = \lvert \sum \limits_{n=-b}^{b} \lambda^{b}
    \mathbf{M}_{n} \vec{v} \rvert.
    $$

    Modes exceeding this tolerance are considered spurious and are
    discarded.

    !!! note
        Only used if `algorithm` is set to `"spectral"`.

    """

    residual_normalization: bool = True
    """Whether to consider relative residuals instead of absolute
    residuals when filtering eigenpairs.

    This is useful to avoid that large eigenvalues will have larger
    absolute residuals than small eigenvalues.

    """

    warning_threshold: PositiveFloat = 1e-1
    r"""The threshold for issuing a warning about the surface Green's
    functions recursion residual.

    This residual is computed as

    $$
    \lvert \mathbf{g}^R - \left[\mathbf{M}_{0} - \mathbf{M}_{-1}
    \mathbf{g}^R \mathbf{M}_{+1} \right]^{-1} \rvert / \lvert
    \mathbf{g}^R \rvert
    $$

    !!! note
        This parameter is only used if the `formalism = "wf"`.
        Otherwise, the memoizer is responsible for residual checking and
        issuing warnings.

    """

    eta_decay: PositiveFloat = 1e-12
    """Small value to separate very slowly decaying modes from perfectly
    propagating ones.

    Modes that are very close to the unit circle could get misclassified
    via the `min_decay` and `min_propagation` conditions, i.e., when
    their decay rate is smaller than `min_decay` but their
    propagation/decay ratio is not pronounced enough. Modes with decay
    rates smaller than this value are considered as perfectly
    propagating modes, even if the propagation/decay ratio is not above
    the `min_propagation` threshold.

    """

    # Parameters for iterative OBC algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the Sancho-Rubio method.

    A warning is issued if the method does not converge within this
    number of iterations.

    """

    convergence_tol: PositiveFloat = 1e-6
    """The convergence tolerance for the Sancho-Rubio method.

    This is the Frobenius norm of the update matrices `alpha` and `beta`
    in the Sancho-Rubio method. Note that the norm is taken over the
    entire energy batch.

    """

    # Parameters for subspace NEVP solvers.
    r_o: PositiveFloat = Field(default=10.0, gt=1)
    """The outer radius of the contour in the complex plane for the
    contour nevp methods (`"beyn"`).

    This parameter should not be too large to avoid having too many
    eigenpairs inside the contour. It should also not be too small to
    avoid missing important eigenpairs. If an eigenpair is very close to
    the contour, it can lead to numerical instabilities.

    """

    r_i: PositiveFloat = Field(default=0.8, gt=0, lt=1)
    """The inner radius of the contour in the complex plane for the
    contour methods.

    This must be less than one to capture propagating modes, but should
    not be too small to avoid including too many decaying modes.

    """

    m_0: PositiveInt = 10
    """The subspace guess in the contour methods.

    The guess has to be larger than the expected number of eigenvalues
    inside the contour. If too small, the method will fail. If too
    large, the method will be less efficient.

    """

    num_quad_points: PositiveInt = 20
    """The number of quadrature points for the contour integrals."""

    memoizer: MemoizerConfig = MemoizerConfig()
    """Options for memoizing the surface Green's functions."""

    @model_validator(mode="after")
    def set_max_decay(self) -> Self:
        """Sets the max decay if not already set."""
        if self.max_decay is None:
            self.max_decay = 1.5 * np.log(self.r_o)

        return self

    @model_validator(mode="after")
    def scale_contour_radii(self) -> Self:
        """Scales the contour radii based on block_sections."""
        self.r_o **= 1 / self.block_sections
        self.r_i **= 1 / self.block_sections

        return self


class LyapunovConfig(BaseModel):
    r"""Parameters for solving the (discrete-time) Lyapunov equation.

    The discrete-time Lyapunov equation (also called Stein equation)
    arises in the computation of lesser boundary conditions.

    This is a matrix equation of the form

    $$
    \mathbf{A} \mathbf{X} \mathbf{A}^{\dagger} - \mathbf{X} =
    -\mathbf{Q}
    $$

    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["spectral", "doubling"] = "spectral"
    r"""The Lyapunov solver algorithm to be used.

    - `"spectral"`: Uses eigenvalue decomposition to solve the Lyapunov
      equation. This method is somewhat expensive since a full
      eigendecomposition is required.

    - `"doubling"`: Uses iterative doubling to solve the Lyapunov
      equation. This method should converge exponentially, but is
      theoretically unstable if $\mathbf{A}$ has eigenvalues outside the
      unit circle. It is therefore generally recommended to use
      `"spectral"` in conjuntion with the memoizer, which will only call
      the actual Lyapunov solver when the residuals are above the
      specified tolerances.

    """

    reduce_sparsity: bool = True
    r"""Whether to exploit the sparsity of $\mathbf{A}$ to accelerate
    the Lyapunov solver.

    This is done by removing zero rows and columns from $\mathbf{A}$,
    solving the reduced Lyapunov equation, and then expanding the
    solution back to the original system's size.

    """

    assume_constant_sparsity: bool = False
    r"""Whether to assume that the sparsity pattern of $\mathbf{A}$
    remains constant between calls to the Lyapunov solver. This is only
    relevant when the Lyapunov solver is called during the SCBA
    iterations. In practice, this should always be the case.

    If set to `True`, the sparsity pattern is only computed once during
    the first SCBA iteration and reused for subsequent iterations.

    """

    # Parameters for iterative Lyapunov algorithms.
    max_iterations: PositiveInt = 100
    """The maximum number of iterations for the `"doubling"` algorithm."""

    relative_tol: PositiveFloat = 1e-4
    """The relative convergence tolerance for the `"doubling"` algorithm."""

    absolute_tol: PositiveFloat = 1e-8
    """The absolute tolerance for the `"doubling"` algorithm."""

    # Parameter for spectral Lyapunov solver.
    num_ref_iterations: PositiveInt = Field(default=2, ge=1)
    """The number of fixed-point iterations used to refine the solution
    of the spectral Lyapunov solver.

    """

    memoizer: MemoizerConfig = MemoizerConfig()
    """Options for memoizing the solution of the Lyapunov equation."""


class ContactConfig(BaseModel):
    """Configuration for a contact.

    !!! warning

        Many contact parameters are currently only used in the
        `"wf"` formalism.

    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """A unique name for the contact."""

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The origin of the contact region in Å.

    This is used to automatically determine the orbitals that belong to
    this contact.

    !!! warning

        This parameter is currently only used in the `"wf"` formalism.


    """

    lattice_vectors: list[list[float]] = Field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    """The lattice vectors of the contact cell in Å.

    In `"wf"` simulations this is used to automatically determine the
    orbitals that belong to this contact.

    The volume of the contact cell is also used to determine the Fermi
    level of the contact from its doping density and the density of
    states of its band structure.

    """

    direction: Literal["a", "b", "c"] | None = None
    """The direction from contact to the device.

    This is used to find periodic images of the contact in transport
    direction.

    !!! warning

        This parameter is currently only used in the `"wf"` formalism.

    """

    fermi_level: float | None = None
    """The Fermi level of the contact.

    If not set, the Fermi level is automatically determined from the
    band structure of the contact via the `mid_gap_energy` parameter.

    When set explicitly, this may lead to physically inconsistent
    results, especially in the context of Schrödinger-Poisson
    simulations.

    """

    mid_gap_energy: float | None = None
    """An energy lying somewhere in the band gap of the contact.

    This is used to separate conduction from valence band states, which
    is necessary to automatically determine a contact's Fermi level, and
    to compute the excess carrier density that is used in computing the
    electrostatic potential in the Poisson solver.

    This is also necessary when band-edge tracking is enabled in
    `"negf"` simulations, since the band edges, and their initial
    distance to the Fermi level, are determined via the mid-gap energy.

    """

    num_kpoints_transport: int = 50
    """Number of k-points to use for contact band structure calculation.

    This is used when automatically determining the Fermi level of the
    contact from its band structure. The k-point grid along the
    transverse directions are determined from the `kpoint_grid`
    parameter.

    """

    temperature: NonNegativeFloat = 300.0  # K
    """The temperature of the contact."""

    voltage: float = 0.0
    """The voltage applied to the contact.

    At least one contact needs to be grounded (i.e. have zero voltage)
    to serve as a reference for the other voltages.

    The voltage and the Fermi level of the contact are used to determine
    its chemical potential.

    """

    @model_validator(mode="after")
    def to_array(self) -> Self:
        """Transforms origin and size to arrays."""
        self.origin = np.array(self.origin, dtype=float)
        self.lattice_vectors = np.array(self.lattice_vectors, dtype=float)
        return self


class ElectronConfig(BaseModel):
    """Options for the electronic subsystem solver."""

    model_config = ConfigDict(extra="forbid")

    solver: SolverConfig = SolverConfig()
    """Parameters concerning the system solver."""

    obc: OBCConfig = OBCConfig()
    """Parameters concerning the open boundary conditions."""

    lyapunov: LyapunovConfig = LyapunovConfig()
    """Parameters concerning the Lyapunov solver.

    !!! warning
        The Lyapunov solver is not used in the electronic subsystem
        solver.

    """

    eta_obc: NonNegativeFloat = 0  # eV
    """Small imaginary value to add to the energy when computing the
    OBCs.

    Including this small broadening can help stabilize the convergence
    of the iterative sancho-rubio OBC solver near van Hove
    singularities.

    """

    eta: NonNegativeFloat = 1e-12  # eV
    """Small imaginary value to add to the energy when computing the
    Green's functions.

    """

    left_contact: ContactConfig | None = None
    """Configuration for the left contact.

    This must be provided for any `"negf"` simulation.

    !!! note

        In `"wf"` simulations, the left and right contacts are not used.

    """

    right_contact: ContactConfig | None = None
    """Configuration for the right contact.

    This must be provided for any `"negf"` simulation.

    !!! note

        In `"wf"` simulations, the left and right contacts are not used.

    """

    band_edge_tracking: bool = False
    """Whether to track the band edges during the SCBA iterations.

    This is setting is only useful if the considered interactions result
    in energy renormalization in the electronic subsystem, which is
    primarily the screened Coulomb interaction.

    If set to `True`, the band edges are tracked during the SCBA
    iterations by computing the eigenvalues of the Hamiltonian
    renormalized with the current self-energy.

    The Fermi levels are then set to be a fixed distance from the band
    edges, which is determined by the initial Fermi level and band
    edges. For example, if the initial Fermi level is 0.5 eV above the
    conduction band edge, the Fermi level is always set to be 0.5 eV
    above the conduction band edge during all SCBA iterations.

    """

    energy_window_min: float | None = None
    """The minimum energy of the energy grid used for electronic
    quantities."""

    energy_window_max: float | None = None
    """The maximum energy of the energy grid used for electronic
    quantities."""

    energy_window_num: PositiveInt | None = None
    """The number of energy points in the energy grid used for electronic
    quantities.

    Either `energy_window_num` or `energy_window_num_per_rank` can be
    set to determine the total number of energy points.

    """
    energy_window_num_per_rank: PositiveInt | None = None
    """The number of energy points per rank in the energy grid used for
    electronic quantities.

    Either `energy_window_num` or `energy_window_num_per_rank` can be
    set to determine the total number of energy points.

    """

    flatband: bool | None = None
    """Whether the system is in flatband conditions.

    If not set, it is automatically determined from the left and
    right Fermi levels. If the Fermi levels are equal, it is assumed
    to be in flatband conditions.

    """

    dos_peak_limit: PositiveFloat = 100.0
    """The maximum derivative of the density of states (DOS) with
    respect to energy.

    At energy points where the DOS derivative exceeds this value, the
    electronic quantities are set to zero to stabilize the convergence
    of the SCBA iterations.

    This is especially a problem during the first few SCBA iterations
    when the self-energies are not yet fully developed and can lead to
    very sharp features in the DOS.

    """

    filtering_iteration_limit: PositiveInt = 1
    """The maximum number of SCBA iterations during which the DOS peak
    filtering is applied.

    This is because the DOS peak filtering is mainly needed during the
    first few SCBA iterations when the self-energies are not yet fully
    developed and can lead to very sharp features in the DOS.

    """

    max_batch_size: PositiveInt | None = None
    """The maximum number of energies to batch together in the solution
    of the electronic subsystem.

    This controls how many energies are treated together when computing boundary
    conditions and electron Green's functions. If not set, all energies are
    computed at once.

    This can help mitigate memory bottlenecks.

    """

    @model_validator(mode="after")
    def check_mid_gap_energy_band_edge_tracking(self) -> Self:
        """Checks that the mid-gap-energy is set if band edge tracking is enabled."""
        if self.band_edge_tracking:
            if (
                self.left_contact is not None
                and self.left_contact.mid_gap_energy is None
            ):
                raise ValueError(
                    "When band edge tracking is enabled, the `mid_gap_energy` of the left contact must be set."
                )
            if (
                self.right_contact is not None
                and self.right_contact.mid_gap_energy is None
            ):
                raise ValueError(
                    "When band edge tracking is enabled, the `mid_gap_energy` of the right contact must be set."
                )

        return self

    @model_validator(mode="after")
    def verify_energies(self) -> Self:
        """Verifies the energy window settings."""

        if (
            self.energy_window_min is not None
            or self.energy_window_max is not None
            or self.energy_window_num is not None
            or self.energy_window_num_per_rank is not None
        ):

            if (self.energy_window_min is None) and (self.energy_window_max is None):
                raise ValueError(
                    "When the energy grid is not read from file, should set both `energy_window_min` and `energy_window_max`."
                )

            if (
                self.energy_window_num is not None
                and self.energy_window_num_per_rank is not None
            ):
                raise ValueError(
                    "Should **exclusively** set electron `energy_window_num` or `energy_window_num_per_rank` in the config."
                )

        return self


class CoulombScreeningConfig(BaseModel):
    """Options for the Coulomb screening solver."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom
    """The cutoff distance for the screened Coulomb interaction
    self-energy.

    Self-energy matrix elements corresponding to pairs of orbitals that
    are further apart than this distance are not computed. A higher
    cutoff can lead to more accurate results, but also increases the
    computation time. The optimal value depends on the system and the
    desired accuracy.

    """

    solver: SolverConfig = SolverConfig()
    """Parameters concernig the system solver."""

    obc: OBCConfig = OBCConfig()
    """Parameters concerning the open boundary conditions."""

    lyapunov: LyapunovConfig = LyapunovConfig()
    """Parameters concerning the Lyapunov solver."""

    temperature: PositiveFloat = 300.0  # K
    """The temperature of the system.

    !!! warning
        The temperature in the Coulomb screening solver is not used. The
        (contact) particle densities are computed via the Lyapunov
        solver.

    """

    epsilon_r: PositiveFloat = 1.0
    """The relative permittivity of the system.

    The Coulomb matrix is scaled by this value. It is primarily useful
    as a way to scale the strength of the Coulomb interaction and to
    better fit the model to experimental results.

    """

    left_temperature: PositiveFloat | None = None
    """The temperature of the left contact.

    If not set, it is assumed to be the same as `temperature`.

    """
    right_temperature: PositiveFloat | None = None
    """The temperature of the right contact.

    If not set, it is assumed to be the same as `temperature`.

    """

    # How many blocks should be merged into a single block.
    num_connected_blocks: Literal["auto"] | PositiveInt = "auto"
    r"""The number of connected blocks to merge into a single block.

    The computation of the effective lesser/greater polarization
    involves a "sandwich" multiplication (congruence transform) of the
    form

    $$
    \mathbf{L}^{\lessgtr} = \mathbf{V} \mathbf{P}^{\lessgtr} \mathbf{V},
    $$

    where $\mathbf{V}$ is the Coulomb matrix and $\mathbf{P}^{\lessgtr}$
    is the lesser/greater polarization. Since all of these matrices are
    banded, the resulting effective polarization $\mathbf{L}^{\lessgtr}$
    can have a much larger bandwidth.

    The block-tridiagonal tiling of the system matrix used in the OBC is
    therefore larger than the transport blocks used in the electron
    solver. The `num_connected_blocks` parameter determines how many of
    the original transport blocks are merged into a single block for the
    Coulomb screening solver. If set to `"auto"`, the number of
    connected blocks is automatically determined based on the
    `interaction_cutoff` and the geometry of the system.

    """

    dos_peak_limit: PositiveFloat = 100.0
    """The maximum derivative of the density of states (DOS) with
    respect to energy.

    At energy points where the DOS derivative exceeds this value, the
    Coulomb screening quantities are set to zero to stabilize the
    convergence of the SCBA iterations.

    """

    filtering_iteration_limit: PositiveInt = 1
    """The maximum number of SCBA iterations during which the DOS peak
    filtering is applied.

    This is because the DOS peak filtering is mainly needed during the
    first few SCBA iterations when the self-energies are not yet fully
    developed and can lead to very sharp features in the DOS.

    """

    align_polarization_to_complex_axes: bool = True
    r"""Whether to discard certain parts of the polarization.

    This affects the polarization in the following way:

    - The real parts of the lesser/greater polarization are discarded.
    - The imaginary part of the retarded polarization from anyprevious
      computation is zeroed.

    This happens before the anti-Hermitian part of the retarded
    polarization is computed from the lesser and greater parts as

    $$
    \mathbf{P}^R_{AH} = \frac{1}{2i} ( \mathbf{P}^> - \mathbf{P}^< )
    $$

    """

    include_energy_renormalization: Literal["self-energy", "polarization", "both"] = (
        "self-energy"
    )
    r"""Whether to compute the Hermitian part of the retarded
    polarization and/or self-energy.

    Possible values are `"self-energy"`, `"polarization"`, and `"both"`.

    The full retarded interaction quantities are general complex-valued
    matrices, where the Hermitian part is computed from the
    skew-Hermitian part using the Kramers-Kronig relations:

    $$
    \mathbf{X}^{R} = \frac{1}{2} (\mathbf{X}^{>} - \mathbf{X}^{<}) +
    \frac{1}{2\pi} \mathrm{p.v.} \int_{-\infty}^{\infty}  dE' \,
    \frac{\mathbf{X}^{>} - \mathbf{X}^{<}}{E^{'} - E}
    $$

    The Hermitian part only leads to only a shift in the energy, so it
    is often neglected:

    $$
    \mathbf{X}^{R} \approx \frac{1}{2} (\mathbf{X}^{>} - \mathbf{X}^{<})
    $$

    The default is to only include the skew-Hermitian part in the
    Coulomb screening self-energy and not in the polarization.

    The Hermitian part is computed using a Hilbert transform. For the
    Coulomb screening self-energy, this Hilbert transform can lead to
    errors at the edges of the energy window. The
    `apply_hilbert_correction` option can be used to apply a correction
    to the Hilbert transform to mitigate these errors.

    """

    apply_hilbert_correction: bool = False
    """Whether to apply the corrections for the edges of the energy
    window to the Hilbert transform when computing the retarded
    self-energy.

    Computing the correction is slightly more expensive.

    """

    @model_validator(mode="after")
    def check_hilbert_correction_applicable(self) -> Self:
        """Checks if the Hilbert correction can be applied."""
        if (
            self.apply_hilbert_correction
            and self.include_energy_renormalization not in ["self-energy", "both"]
        ):
            raise ValueError(
                "Hilbert correction can only be applied if the real part of the self-energy is included."
            )

        return self

    max_batch_size: PositiveInt | None = None
    """The maximum number of energies to batch together in the solution
    of the screened Coulomb interaction.

    This controls how many energies are treated together when computing boundary
    conditions and screened Coulomb interactions. If not set, all energies are
    computed at once.

    This can help mitigate memory bottlenecks.

    """


class PhotonConfig(BaseModel):
    """Parameters for photons and electron-photon interactions.

    !!! warning
        The photon solver is not implemented yet. The parameters in this
        section are not used and may be subject to change in the future.

    """

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom

    solver: SolverConfig = SolverConfig()
    obc: OBCConfig = OBCConfig()
    lyapunov: LyapunovConfig = LyapunovConfig()


class PhononConfig(BaseModel):
    """Parameters for phonons and electron-phonon interactions."""

    model_config = ConfigDict(extra="forbid")

    interaction_cutoff: PositiveFloat = 10.0  # Angstrom
    """The cutoff distance for the electron-phonon interaction
    self-energy.

    !!! note
        Currently, only the `"pseudo-scattering"` model / deformation
        potential interaction is implemented, which does not produce
        any self-energy matrix elements besides the diagonal ones.

    """

    solver: SolverConfig = SolverConfig()
    """Parameters concerning the system solver."""

    obc: OBCConfig = OBCConfig()
    """Parameters concerning the open boundary conditions."""

    lyapunov: LyapunovConfig = LyapunovConfig()
    """Parameters concerning the Lyapunov solver."""

    model: Literal["pseudo-scattering", "negf"] = "pseudo-scattering"
    r"""Which model to use for the electron-phonon interaction.

    Currently, only a monochromatic `"pseudo-scattering"` model is
    implemented.

    In this model, the electron-phonon interaction is modeled as

    $$
    \Sigma^{\lessgtr}(E) = D^2 \left[ (N_{ph} + 1) G^{\lessgtr}(E - \hbar
    \omega) + N_{ph} G^{\lessgtr}(E + \hbar \omega) \right],
    $$

    where $D$ is the `deformation_potential`, $\hbar \omega$ is the
    `phonon_energy`, and $N_{ph}$ is the phonon occupation number given
    by the Bose-Einstein distribution at the specified `temperature`.

    """

    phonon_energy: NonNegativeFloat | None = None
    """The energy of the phonon mode in eV."""

    deformation_potential: NonNegativeFloat | None = None
    """The deformation potential of the phonon mode in eV."""

    temperature: PositiveFloat = 300.0  # K
    """The temperature of the system in Kelvin."""

    @model_validator(mode="after")
    def check_phonon_energy_or_deformation_potential(self):
        """Check if 'phonon_energy' and 'deformation_potential' are set."""
        if self.model == "pseudo-scattering" and (
            self.phonon_energy is None or self.deformation_potential is None
        ):
            raise ValueError("'phonon_energy' and 'deformation_potential' must be set.")

        return self


class OutputConfig(BaseModel):
    """Options for the output of `quatrex` calculations.

    !!! warning
        The output options are not yet fully implemented and may be
        subject to change in the future. They are currently not used in
        QTBM calculations.

    """

    model_config = ConfigDict(extra="forbid")

    # Only the spectral currents are saved by default.
    device_currents: bool = True
    """Whether to save the device currents.

    This will output both the spectral device current between transport
    cells computed from the lesser Green's function and, if configured,
    the Meir-Wingreen device current.

    """

    potential: bool = False
    """Whether to save the potential.

    !!! warning
        This option is unused.

    """

    electron_ldos: bool = False
    """Whether to save the spectral electron local density of states
    (LDOS).

    This will output an energy and orbital resolved LDOS computed from
    the retarded Green's function.

    """

    electron_density: bool = False
    """Whether to save the electron density.

    This will output the energy-resolved electron density computed from
    the lesser Green's function.

    """
    hole_density: bool = False
    """Whether to save the hole density.

    This will output the energy-resolved hole density computed from the
    greater Green's function.

    """

    polarization_density: bool = False
    """Whether to save the polarization density.

    This will output the energy-resolved polarization densities computed
    from the lesser and greater polarizations.

    !!! note
        This is primarily a debugging option.

    """

    coulomb_screening_density: bool = False
    """Whether to save the Coulomb screening density.

    This will output the energy-resolved Coulomb screening densities
    computed from the lesser and greater screened Coulomb interactions.

    !!! note
        This is primarily a debugging option.

    """

    self_energy_density: bool = False
    """Whether to save the self-energy density.

    This will output the energy-resolved self-energy densities computed
    from the lesser, greater, and retarded self-energies.

    """

    profiling_path: Path | None = None
    """The file to save the timing results to.

    The timing results are saved in the format specified by
    `profiling_save_format`.

    If `save_profiling_results` is `True`, and the `profiling_path` is
    not set, the file name is inferred from the SLURM output file if
    running in a SLURM context. Otherwise, the default name
    `quatrex_times.out` is used.

    """

    save_profiling_results: bool = False
    """Whether to save the timing results to a file."""

    profiling_save_format: Literal["pickle", "json"] = "json"
    """The format to save the timing results in.

    The timing results are saved in either `pickle` or `json` format.
    The default is `json`. `pickle`-serialized files will contain a
    dictionary with the timing results.

    """

    @model_validator(mode="after")
    def set_profiling_parameters(self) -> Self:
        if self.profiling_path is None:
            self.profiling_path = Path("quatrex_times.out")
            if "SLURM_JOB_ID" in os.environ:
                try:
                    jid = os.environ.get("SLURM_JOB_ID")
                    if not jid:
                        raise ValueError("SLURM_JOB_ID is not set.")
                    info = subprocess.check_output(
                        ["scontrol", "show", "job", jid]
                    ).decode()

                    slurm_out = re.search(r"StdOut=(\S+)", info).group(1)
                    slurm_out_base, _ = os.path.splitext(slurm_out)

                    if os.path.exists(slurm_out):
                        self.profiling_path = Path(
                            slurm_out_base + "_quatrex_times.out"
                        )

                except Exception:
                    pass

        assert self.profiling_path is not None, "profiling_path should be set here."

        return self


class DeviceConfig(BaseModel):
    """Configuration for the simulated device.

    !!! warning
        The contacts configuration in this table is only used in QTBM
        calculations, since we allow more than two contacts in QTBM.

    """

    model_config = ConfigDict(extra="forbid")

    construct_from_unit_cell: bool = False
    """Whether to construct a device from its unit cell geometry and
    electronic structure.

    If this is set to `True`, the Hamiltonian read from the input file
    is assumed to be the tight-binding-like Hamiltonian of a single unit
    cell. The simulated device structure is then constructed by
    repeating the unit cell along the transport direction, as specified
    by `num_transport_cells`, and including the neighboring cells as
    configured by `neighbor_cell_cutoff`.

    """

    geometry: GeometryConfig
    """The geometry configuration of the device.

    This contains a defintion of all regions in the device, such as
    doping, material constants and gates.

    """

    # --- Device geometry ---------------------------------------------
    neighbor_cell_cutoff: (
        tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt] | None
    ) = None
    """The number of neighbor cells to consider along each lattice
    direction.

    If set to `None`, all neighbor cells present in the Hamiltonian
    input file are considered. A `neighbor_cell_cutoff` of zero means
    that only the unit cell itself is considered.

    Along the transport direction, at least one neighboring cell must be
    included if `construct_from_unit_cell` is `True`. If
    `construct_from_unit_cell` is `False`, including neighboring cells
    in transport direction is not allowed, since the device should
    already be upscaled in that case.

    If more neighbor cells are requested than present in the input
    Hamiltonian, a `ValueError` is raised.

    """

    num_transport_cells: PositiveInt = 1
    """The number of transport cells to include in the simulation.

    !!! note

        This parameter is only used if `construct_from_unit_cell` is
        `True`.

    """

    transport_direction: Literal["x", "y", "z"]
    """The direction along which the transport occurs.

    !!! note
        Currently, only axis-aligned transport directions are supported.

    """

    block_size: PositiveInt | list[PositiveInt] | None = None
    """The block size to use for the device Hamiltonian.

    This block size is used in NEGF calculations, where it determines
    the block-tridiagonal tiling of all quantities.

    If a single integer is given, a constant block size is assumed.
    Alternatively, a list of block sizes can be given to specify the
    size of each block along transport direction.

    The `block_size` parameter cannot be used in conjunction with
    `construct_from_unit_cell=True` since the block sizes are determined
    from the unit cell and the `neighbor_cell_cutoff` in that case.

    If `construct_from_unit_cell=False` in NEGF simulations, the block
    size must be given.

    """

    contacts: list[ContactConfig] = Field(default_factory=list)
    """The contacts of the device.

    !!! warning
        The contacts configuration in this table is only used in QTBM
        calculations, since we allow more than two contacts in QTBM.

    """

    num_orbitals_per_atom: dict[str, int] = {"X": 1}
    """The number of orbitals per atom type.

    This mapping is used to connect the atomistic geometry with the
    corresponding operator matrix elements.

    Currently, this is primarily used when configuring contacts via
    their real-space extents in QTBM calculations. It is also used to
    map a given potential vector to the corresponding orbitals in the
    Hamiltonian.

    The keys can be any string, that matches the atom types in the
    structure file. The default is a single atom type "X" with one
    orbital per atom, which is useful when dealing with Wannier orbitals
    that are not atom-centered.

    """

    kpoint_grid: tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 1, 1)
    """The kpoint grid on which to compute transport quantities.

    This is a Monkhorst-Pack grid, which is used to sample the Brillouin
    zone transverse to the transport direction. The k-point grid is
    specified as a tuple of three integers, which correspond to the
    number of k-points along the x, y, and z directions, respectively.
    The k-point grid must be 1 along the transport direction, since the
    periodicity along that direction is broken.

    """
    kpoint_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The kpoint shift to apply to the Monkhorst-Pack grid."""

    @model_validator(mode="after")
    def to_tuple(self) -> Self:
        """Transforms list to tuple."""
        if self.neighbor_cell_cutoff is not None:
            self.neighbor_cell_cutoff = tuple(self.neighbor_cell_cutoff)
        self.kpoint_grid = tuple(self.kpoint_grid)
        return self

    @model_validator(mode="after")
    def check_kpoint_grid(self) -> Self:
        """Checks that the k-point grid is 1 along the transport direction."""

        ind = "xyz".index(self.transport_direction)
        if self.kpoint_grid[ind] != 1:
            raise ValueError(
                f"Along the transport direction ('{self.transport_direction}'), the k-point grid must be 1."
            )

        return self

    @model_validator(mode="after")
    def check_connecting_cells(self) -> Self:
        """Checks that num_connecting_cells is not zero in transport direction."""

        if self.neighbor_cell_cutoff is None:
            return self

        ind = "xyz".index(self.transport_direction)
        if not self.construct_from_unit_cell:
            if self.neighbor_cell_cutoff[ind] != 0:
                raise ValueError(
                    f"Along the transport direction ('{self.transport_direction}'),"
                    "no neighboring cells should be included if `construct_from_unit_cell` is False."
                )
        else:
            if self.neighbor_cell_cutoff[ind] < 1:
                raise ValueError(
                    f"At least one neighboring cell in transport direction "
                    f"('{self.transport_direction}') must be included."
                )

        return self


class LyapunovComputeConfig(BaseModel):
    """Configuration concerning the solution of the Lyapunov equation."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"
    """Backend to use for computing eigenvalues.

    The spectral Lyapunov solver requires the computation of eigenvalues
    of a general dense matrix. This parameter determines whether to use
    NumPy, CuPy, or NVMath for this computation. The default is NumPy.

    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory when transferring data in the
    spectral Lyapunov solver."""


class NEVPConfig(BaseModel):
    """Configurations concerning the solution of NEVPs."""

    model_config = ConfigDict(extra="forbid")

    eig_compute_location: Literal["numpy", "cupy", "nvmath"] = "numpy"
    """Backend to use for computing eigenvalues.

    This parameter determines whether to use NumPy, CuPy, or NVMath for
    computing eigenvalues in the NEVP solvers. The default is NumPy.

    """

    # Parameters for contour NEVP solvers.
    project_compute_location: Literal["numpy", "cupy"] = "numpy"
    """Backend to use for computing the projection matrices.

    When using contour-based NEVP solvers, one needs to project the
    non-linear system onto a linear subspace. This can either be done
    using QR decomposition or by computing a singular value
    decomposition (SVD), which is controlled by the `use_qr` parameter.

    The `project_compute_location` parameter determines whether to use
    NumPy or CuPy for this computation. The default is NumPy.

    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory when transferring data in the NEVP
    solvers."""

    use_qr: bool = False
    """Whether to use QR decomposition or SVD for the projection.

    When using contour-based NEVP solvers, one needs to project the
    non-linear system onto a linear subspace. This can either be done
    using QR decomposition or by computing a singular value
    decomposition (SVD). The `use_qr` parameter determines which method
    to use. The default is to use SVD, but QR decomposition can be
    significantly faster than SVD.

    """

    contour_batch_size: PositiveInt | None = None
    """The batch size to use for the contour NEVP solvers.

    The contour NEVP solvers require performing quadrature of an
    operator over a contour in the complex plane. Since this can lead to
    memory bottlenecks, the quadrature can be performed in batches. The
    `contour_batch_size` parameter determines the number of quadrature
    points to use in each batch. If set to `None`, the entire quadrature
    is performed in a single batch.

    """

    num_threads_contour: PositiveInt = 1024
    """The number of GPU threads to use for computing the operator
    inverses in the contour NEVP solvers.

    Only used if the GPU is available and the contour NEVP solvers are
    used.

    """

    # Parameters for full NEVP solvers.
    reduce_sparsity: bool = False
    """Whether to reduce the sparsity of the matrices in the full NEVP
    solver.

    The matrices arising in the full NEVP solver can contain some zero
    rows and columns, which can be removed to reduce the size of the
    eigenvalue problem.

    """


class BandEdgeConfig(BaseModel):
    """Parameters concerning the eigenvalue-based band-edge tracking."""

    model_config = ConfigDict(extra="forbid")

    use_eigvalsh: bool = True
    r"""Whether to use eigvalsh when computing the band edges.

    The non-linear eigenvalue problem
    
    $$
    \left[\mathbf{H} + \mathbf{\Sigma}^R(E)\right] \boldsymbol{\psi} = E
    \boldsymbol{\psi},
    $$

    which needs to be solved to compute the band edges is in principle a
    general eigenvalue problem. However, since we only care about real
    eigenvalues and the energy renormalization due to the Hermitian part
    of $\Sigma^R$, we can just solve the Hermitian part of the problem
    using `eigvalsh`. This is significantly faster than solving the full
    non-linear eigenvalue problem, but it is an approximation if
    scattering is included.

    Only relevant if `band_edge_tracking = True`.

    """

    eigvalsh_compute_location: Literal["numpy", "cupy"] = "numpy"
    """Location where to compute the eigenvalues.

    The eigenvalues can be computed either on the CPU using NumPy or on
    the GPU using CuPy. The default is to use NumPy.

    Only relevant if `band_edge_tracking = True`.

    """

    use_pinned_memory: bool = True
    """Whether to use pinned memory when transferring data in the
    band-edge tracking computation.

    Only relevant if `band_edge_tracking = True`.

    """

    block_sections: PositiveInt = 1
    """The number of block sections to use when computing the band
    edges."""

    @field_validator("use_eigvalsh", mode="after")
    @classmethod
    def check_use_eigvalsh(cls, value) -> bool:
        if not value:
            raise NotImplementedError(
                "Only use_eigvalsh=True is supported at the moment."
            )
        return value

    @field_validator("eigvalsh_compute_location", mode="after")
    @classmethod
    def check_eigvalsh_location(cls, value) -> Literal["numpy", "cupy"]:
        if value == "cupy" and xp.__name__ != "cupy":
            warnings.warn(
                "eigvalsh_compute_location is set to 'cupy' but cupy is not available. Falling back to 'numpy'.",
                UserWarning,
            )
            return "numpy"
        elif value == "numpy" and xp.__name__ == "cupy":
            warnings.warn(
                "eigvalsh_compute_location is set to 'numpy' but cupy is available. Consider setting it to 'cupy' for better performance.",
                UserWarning,
            )

        return value


class ConvolveConfig(BaseModel):
    """Parameters concerning the FFT convolution."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: should be calculate from the number of energy points, ranks,
    # and nnz.
    batch_size: PositiveInt | None = None
    """The batch size to use for the FFT convolution.
    
    Since the performing FFT can lead to memory bottlenecks, the
    convolution can be performed in batches. The `batch_size` parameter
    determines the number of matrix elements to compute in each batch.
    If set to `None`, the entire convolution is performed in a single
    batch.

    """


class CommConfig(BaseModel):
    """Parameters concerning the communication backends.

    The communication backend in `quatrex` has two subcommicator groups:
    One between energy points and one between matrix blocks.

    For both `block` and `stack` subcommunicators, the following
    communication operations can be performed:

    - `all_to_all`
    - `all_gather`
    - `all_reduce`
    - `bcast`
    - `send_recv`

    The communication backend can be set to either `"host_mpi"`,
    `"device_mpi"`, or `"nccl"` for each of these operations.

    """

    model_config = ConfigDict(extra="forbid")

    block_comm_size: PositiveInt = 1
    """The number of ranks over which to disctribute matrix blocks.
    
    SCBA supports spatial domain distribution. The matrix blocks can be
    distributed over multiple ranks, which can be useful for extremely
    large systems. The `block_comm_size` parameter determines the number
    of ranks over which to distribute the matrix blocks.

    If set to 1 (the default), the matrix blocks are not distributed
    over multiple ranks.

    """

    block_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block all-to-all."""
    block_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block all-gather."""
    block_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block all-reduce."""
    block_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block broadcast."""
    block_send_recv: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for block send-receive."""

    stack_all_to_all: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack all-to-all."""
    stack_all_gather: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack all-gather."""
    stack_all_reduce: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack all-reduce."""
    stack_bcast: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack broadcast."""
    stack_send_recv: Literal["host_mpi", "device_mpi", "nccl"] | None = None
    """Communication backend to use for stack send-receive."""


class ComputeConfig(BaseModel):
    """Top level configuration for all performance and compute options."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dsdbsparse_type: DSDBSparse = DSDBCOO
    """The type of sparse matrix to use for the DSDBSparse matrices.

    !!! warning
        Currently, only `DSDBCOO` is supported. A CSR type had been
        implemented, but it is no longer fully supported.

    """
    numba_threading_layer: Literal["workqueue", "omp", "tbb"] = "workqueue"
    """The threading layer to use for Numba.
    
    We recommend using the default `"workqueue"` threading layer in
    numba, as we have had issues with correctly limiting the number of
    threads when using the `"omp"` threading layer.

    """

    threadpool_api: Literal["blas", "openmp", "tbb"] | None = None
    """
    !!! warning
        The `threadpool_api` parameter is currently not used.

    """

    numba_num_threads: PositiveInt | None = None
    """The number of threads to use for Numba."""

    blas_num_threads: PositiveInt | Literal["sequential_blas_under_openmp"] | None = (
        None
    )
    """The number of threads to use for BLAS."""

    convolve: ConvolveConfig = ConvolveConfig()
    """Parameters concerning the FFT convolution in scattering interactions."""
    nevp: NEVPConfig = NEVPConfig()
    """Parameters concerning the solution of non-linear eigenvalue problems."""
    lyapunov: LyapunovComputeConfig = LyapunovComputeConfig()
    """Parameters concerning the solution of Lyapunov equations."""
    band_edge: BandEdgeConfig = BandEdgeConfig()
    """Parameters concerning the eigenvalue-based band-edge tracking."""
    comm: CommConfig = CommConfig()
    """Parameters concerning the communication backends."""

    @field_validator("dsdbsparse_type", mode="before")
    @classmethod
    def set_dsdbsparse(cls, value) -> DSDBSparse:
        """Converts the string value to the corresponding DSDBSparse object."""
        if value == "DSDBCOO":
            return DSDBCOO
        raise ValueError(f"Invalid value '{value}' for dbsparse")


class QuatrexConfig(BaseModel):
    """Top-level simulation configuration."""

    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    device: DeviceConfig
    """The device configuration."""

    formalism: Literal["wf", "negf"]
    """The transport formalism to use.

    There are two supported formalisms:

    - `"wf"`: Wavefunction formalism
    - `"negf"`: Non-equilibrium Green's function formalism

    !!! warning "Inconsistent input formats"

        Currently, the input formats for the two formalisms are not
        consistent.

    """
    scsp: SCSPConfig | None = None
    """Parameters for the self-consistent Schrödinger-Poisson loop."""

    scba: SCBAConfig = SCBAConfig()
    """Parameters for the self-consistent Born approximation loop."""

    qtbm: QTBMConfig = QTBMConfig()
    """Parameters for the quantum transmitting boundary method."""

    electrostatics: ElectrostaticsConfig = ElectrostaticsConfig()
    """Parameters for the electrostatics calculations."""

    electron: ElectronConfig
    """Parameters for the electronic system."""

    phonon: PhononConfig | None = None
    """Parameters for the phonon system."""
    coulomb_screening: CoulombScreeningConfig | None = None
    """Parameters for the Coulomb screening."""
    photon: PhotonConfig | None = None
    """Parameters for the photon system."""

    # --- Directory paths ----------------------------------------------
    config_dir: Path
    simulation_dir: Path = Path("./quatrex/")
    """The directory where the simulation is run."""
    input_dir: Path | None = None
    """The directory where the input files are located."""
    output_dir: Path | None = None
    """The directory where the output files are saved."""

    # --- Output options -----------------------------------------------
    outputs: OutputConfig = OutputConfig()
    """Parameters for the output of `quatrex` calculations."""

    # --- Compute options ----------------------------------------------
    compute: ComputeConfig = ComputeConfig()
    """Parameters for the performance and compute options."""

    @model_validator(mode="after")
    def resolve_config_path(self) -> Self:
        """Resolves the config directory path."""
        self.config_dir = Path(self.config_dir).resolve()
        return self

    @model_validator(mode="after")
    def resolve_simulation_dir(self):
        """Resolves the simulation directory path."""
        self.simulation_dir = (self.config_dir / self.simulation_dir).resolve()
        return self

    @model_validator(mode="after")
    def set_output_dir(self):
        """Resolves the simulation directory path."""
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            if self.output_dir.is_absolute():
                self.output_dir = self.output_dir.resolve()
                return self

            self.output_dir = (self.config_dir / self.output_dir).resolve()
            return self

        self.output_dir = self.simulation_dir / "outputs/"
        return self

    @model_validator(mode="after")
    def set_input_dir(self) -> Path:
        """Returns the input directory path."""
        if self.input_dir is not None:
            self.input_dir = Path(self.input_dir)
            if self.input_dir.is_absolute():
                self.input_dir = self.input_dir.resolve()
                return self

            self.input_dir = (self.config_dir / self.input_dir).resolve()
            return self

        self.input_dir = self.simulation_dir / "inputs/"
        return self

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Validates the input file paths."""

        if (
            self.electron.energy_window_min is None
            and self.electron.energy_window_max is None
            and self.electron.energy_window_num is None
            and self.electron.energy_window_num_per_rank is None
        ):
            if not (self.input_dir / "electron_energies.npy").resolve().is_file():
                raise ValueError(
                    f"Energy grid not specified and file '{(self.input_dir / 'electron_energies.npy').resolve()}' does not exist."
                )

        # TODO: extend this to other paths, not only energies

        return self

    @model_validator(mode="after")
    def check_device_block_size(self) -> Self:
        """Checks that block size is consistent with other parameters."""

        if self.formalism == "wf":
            # NOTE: Block sizes are not used in the wavefunction
            # formalism.
            return self

        if self.device.construct_from_unit_cell and self.device.block_size is not None:
            raise ValueError(
                "block_size cannot be used in conjunction with construct_from_unit_cell=True."
            )

        if not self.device.construct_from_unit_cell and self.device.block_size is None:
            raise ValueError(
                "block_size must be given when construct_from_unit_cell=False."
            )

        return self

    @model_validator(mode="after")
    def check_device_contact_voltages(self) -> Self:
        """Checks that at least one contact exists and is grounded."""
        # TODO: Contacts should be unified between the two formalisms.
        if self.formalism == "negf":
            if (
                self.electron.left_contact is None
                or self.electron.right_contact is None
            ):
                raise ValueError("Both left and right contacts must be defined.")
            contacts = [self.electron.left_contact, self.electron.right_contact]
        elif self.formalism == "wf":
            contacts = self.device.contacts
        else:
            raise ValueError(f"Invalid formalism '{self.formalism}'.")

        if len(contacts) < 2:
            raise ValueError("At least two contacts must be defined.")

        if not any(contact.voltage == 0 for contact in contacts):
            raise ValueError(
                "At least one contact must be grounded (i.e. have zero voltage)."
            )

        return self

    @model_validator(mode="after")
    def check_either_fermi_or_midgap(self) -> Self:
        """Checks that either the Fermi level or the mid-gap energy is set."""
        if self.formalism == "negf":
            contacts = [self.electron.left_contact, self.electron.right_contact]
        elif self.formalism == "wf":
            contacts = self.device.contacts
        else:
            raise ValueError(f"Invalid formalism '{self.formalism}'.")

        for contact in contacts:
            if contact.fermi_level is None and contact.mid_gap_energy is None:
                raise ValueError(
                    "Either `fermi_level` or `mid_gap_energy` must be set."
                )

            if (
                contact.fermi_level is not None
                and contact.mid_gap_energy is not None
                and not (self.electron.band_edge_tracking or self.scsp is not None)
            ):
                raise ValueError(
                    "Both `fermi_level` and `mid_gap_energy` cannot be set "
                    "simultaneously, unless band edge tracking is active "
                    "or the Schrödinger-Poisson solver is enabled."
                )

        return self

    @model_validator(mode="after")
    def check_contact_direction(self) -> Self:
        """Checks that the contact direction is set in "wf" formalism."""

        if self.formalism == "negf":
            # NOTE: The contact direction is not used in the NEGF
            # formalism.
            return self

        for contact in self.device.contacts:
            if contact.direction is None:
                raise ValueError(
                    "The `direction` parameter of each contact must be "
                    "set in the 'wf' formalism."
                )

        return self


def parse_config(config_file: Path) -> QuatrexConfig:
    """Reads the TOML config file.

    Only rank 0 process reads the config file. It is then broadcasted to
    the other processes. Each process then parses the config into a
    `QuatrexConfig` object.

    Parameters
    ----------
    config_file : Path
        Path to the TOML configuration file.

    Returns
    -------
    QuatrexConfig
        The parsed configuration object.

    """
    config = None
    if mpi_comm_world.rank == 0:
        config_file = Path(config_file).resolve()

        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        if "simulation_dir" in config:
            simulation_dir = config["simulation_dir"]
            if not os.path.isabs(simulation_dir):
                parent_dir = os.path.dirname(os.path.abspath(config_file))
                simulation_dir = Path(os.path.join(parent_dir, simulation_dir))
                config["simulation_dir"] = simulation_dir

        config["config_dir"] = config_file.parent

    config = mpi_comm_world.bcast(config, root=0)

    # Resolve the geometry config.
    config["device"]["geometry"] = parse_geometry_config(config["device"])

    return QuatrexConfig(**config)


def _setup_profiler(config: QuatrexConfig) -> None:
    """Sets up the profiler based on the given configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object containing the profiling settings.

    """

    if not config.outputs.profiling_path.is_absolute():
        config.outputs.profiling_path = (
            config.config_dir / config.outputs.profiling_path
        ).resolve()

    # Saving will strip the extension
    profiler.set_parameters(
        print_path=config.outputs.profiling_path,
        save_path=config.outputs.profiling_path,
        save_format=config.outputs.profiling_save_format,
    )


def _setup_comm(comm_config: CommConfig) -> None:
    """Sets up the communication backend.

    Parameters
    ----------
    comm_config : CommConfig
        The communication configuration containing the communication settings.

    """
    default_backend = "host_mpi" if xp.__name__ == "cupy" else "device_mpi"

    block_comm_config = {
        "all_to_all": comm_config.block_all_to_all or default_backend,
        "all_gather": comm_config.block_all_gather or default_backend,
        "all_reduce": comm_config.block_all_reduce or default_backend,
        "bcast": comm_config.block_bcast or default_backend,
        "send_recv": comm_config.block_send_recv or default_backend,
    }

    stack_comm_config = {
        "all_to_all": comm_config.stack_all_to_all or default_backend,
        "all_gather": comm_config.stack_all_gather or default_backend,
        "all_reduce": comm_config.stack_all_reduce or default_backend,
        "bcast": comm_config.stack_bcast or default_backend,
        "send_recv": comm_config.stack_send_recv or default_backend,
    }

    comm.configure(
        block_comm_size=comm_config.block_comm_size,
        block_comm_config=block_comm_config,
        stack_comm_config=stack_comm_config,
        override=True,
    )


def _setup_threading(compute_config: ComputeConfig):
    """Sets up the threading layer.

    Parameters
    ----------
    compute_config : ComputeConfig
        The compute configuration containing the threading settings.

    """

    # TODO: set the number of threads automatically based on the available cores
    # problems is that we do not know yet how many energy points there will be
    # has to be after unifying the configs
    # NOTE: here we could now do this tuening
    if compute_config.numba_num_threads is None:
        compute_config.numba_num_threads = 1
    if compute_config.blas_num_threads is None:
        compute_config.blas_num_threads = 1

    nb.set_num_threads(compute_config.numba_num_threads)
    nb.config.THREADING_LAYER = compute_config.numba_threading_layer

    if compute_config.numba_num_threads == 1 and compute_config.blas_num_threads in [
        "sequential_blas_under_openmp",
        1,
    ]:
        if comm.rank == 0:
            warnings.warn(
                "The CPU code will run sequentially which may impact performance.",
                UserWarning,
            )


def setup_context(config: QuatrexConfig) -> None:
    """Sets up the simulation context based on the given configuration.

    This includes setting up the profiler, the communication backend,
    and the threading layer.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object containing the settings for the
        simulation context.

    """
    _setup_profiler(config)
    _setup_comm(config.compute.comm)
    _setup_threading(config.compute)
