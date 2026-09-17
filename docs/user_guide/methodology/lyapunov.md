# Lyapunov Problem

As mentioned in [`obc`](obc.md), the Lyapunov Equation

$$
\begin{equation}
\mathbf{w}^{\lessgtr} = \mathbf{q}^{\lessgtr} −
\mathbf{a}\mathbf{w}^{\lessgtr}\mathbf{a}^\dagger
\label{eq:lyapunov}
\end{equation}
$$

needs to be solved to compute the lesser and greater surface Green's
functions in systems where the fluctuation-dissipation theorem cannot be
applied directly.

<!--
TODO: include derivation of the Lyapunov equation from the derivation of
RGF with a non-identity right hand side matrix.
-->

## Sparsity Reduction

The Lyapunov problem can be reduced in size by exploiting the sparsity
of the matrices $\mathbf{a}$. Either zero columns or rows of the matrix
$\mathbf{a}$ can be removed. This can lead to significant speedups for
large systems with sparse matrices. The sparsity reduction is controlled
through the parameter
[`reduce_sparsity`](../parameters/lyapunov.md#reduce_sparsity). By
default, it is enabled, but it is assumed that the sparsity of the
matrix $\mathbf{a}$ can change throughout the simulation.

!!! danger "Constant Sparsity Assumption"
    Assuming constant sparsity is currently never valid. Thus, the
    parameter
    [`assume_constant_sparsity`](../parameters/lyapunov.md#assume_constant_sparsity)
    should not be set to `true`. This is a feature that will be further
    developed in the future, but currently it is not supported.

## Solution Approaches

Similar to the solution of the fixed point problem for the retarded
boundary conditions, both iterative and direct methods can be used to
solve the Lyapunov equation. Similar considerations apply to the choice
of method. The iterative method can be more memory efficient, but can
also suffer from convergence issues. Thus, the choice of method depends
on the well-posedness of the problem and the available computational
resources. For the Lyapunov problem, convergence properties are known in
the literature. Iterative methods are stable when the magnitudes of all
eigenvalues of the matrix $\mathbf{a}$ are below one [^1].

!!! info "Algorithm Selection"
    The method for the Lyapunov problem can be set through the parameter
    [`algorithm`](../parameters/lyapunov.md#algorithm) inside
    [`lyapunov`](../parameters/lyapunov.md).

[^1]: Poloni, Federico. "Iterative and doubling algorithms for
    Riccati‐type matrix equations: A comparative introduction."
    GAMM‐Mitteilungen 43.4 (2020): e202000018.

### Iterative

#### Fixed-Point Iterations

The linearly convergent fixed-point iteration method is the simplest
iterative method to solve the Lyapunov problem.

$$
\begin{equation}
\mathbf{w}^{\lessgtr}_{n+1} = \mathbf{q}^{\lessgtr} −
\mathbf{a}\mathbf{w}^{\lessgtr}_{n}\mathbf{a}^\dagger
\label{eq:lyapunov_iterative}
\end{equation}
$$

The convergence of the method depends on the spectral radius of the
matrix $\mathbf{a}$, which is defined as the largest absolute value of
its eigenvalues. If the spectral radius is greater than or equal to one,
the method may diverge.

Simple fixed-point iterations are not exposed to the user, but are used
as a refinement step in both the direct method and the memoizer. From
experience, the iterative methods can converge well for the Lyapunov
problem, except that spurious energies can lead to divergence. Thus, the
iterative methods are not recommended for general use.

#### Squared Smith

Like Sancho-Rubio, an exponentially convergent iterative method can be
derived for this recursion relation. This doubling method is also called
*squared Smith method* and is described in [^1]. As for fixed-point
iterations, this method convergence depends on the spectral radius of
the matrix $\mathbf{a}$.

### Direct

#### Spectral Method

Solving the Lyapunov problem directly can be done by eigenvalue
decomposing the matrix $\mathbf{a}$ and then solving the Lyapunov
problem in the eigenbasis. We call this the `"spectral"` method.

??? info "Derivation of the Spectral Method"

    The derivation of the method is as follows:

    $$
    \mathbf{w}^{\lessgtr}_{n+1} = \mathbf{q}^{\lessgtr} −
    \mathbf{a}\mathbf{w}^{\lessgtr}_{n}\mathbf{a}^\dagger \quad
    \xrightarrow{\mathbf{a} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1}}
    \quad \mathbf{w}^{\lessgtr}_{n+1} = \mathbf{q}^{\lessgtr} − \mathbf{V}
    \mathbf{\Lambda} \mathbf{V}^{-1}\mathbf{w}^{\lessgtr}_{n}
    \mathbf{V}^{-\dagger} \mathbf{\Lambda} \mathbf{V}^\dagger
    $$

    Next, we define the transformed matrices $\hat{\mathbf{w}} \equiv
    \mathbf{V}^{-1} \mathbf{w}^{\lessgtr} \mathbf{V}^{-\dagger}$ and
    $\hat{\mathbf{q}} \equiv \mathbf{V}^{-1} \mathbf{q}^{\lessgtr}
    \mathbf{V}^{-\dagger}$, which leads to

    $$
    \hat{\mathbf{w}} = \hat{\mathbf{q}} − \mathbf{\Lambda} \hat{\mathbf{w}}
    \mathbf{\Lambda}.
    $$

    This equation can be solved element-wise as

    $$
    \hat{\mathbf{w}}_{ij} = \frac{\hat{\mathbf{q}}_{ij}}{1 - \lambda_i
    \lambda_j^*}
    $$

    and the original matrix $\mathbf{w}^{\lessgtr}$ can be reconstructed as

    $$
    \mathbf{w}^{\lessgtr} = \mathbf{V} \hat{\mathbf{w}} \mathbf{V}^\dagger.
    $$

The method is efficient, but requires the eigenvalue decomposition of
the matrix $\mathbf{a}$ which can be computationally expensive for large
matrices. The matrix $\mathbf{a}$ has generally no symmetry properties,
thus LAPACK `geev` has to be used.

!!! info "Optimizing the performance of the eigenvalue solver"
    NVIDIA has an optimized routine for solving general eigenvalue
    problems. To use this routine, the
    [`eig_compute_location`](../parameters/nevp.md#eig_compute_location)
    parameter should be set to `"cupy"`. Note that this option will be
    refactored and in the future, the best option will be determined
    automatically.

As we observed some stability issues with this method, we still do a
fixed-point iteration refinement step after the spectral method. The
spectral method is currently the default method for the Lyapunov
problem, but potentially the Schur method can be more stable.

### Memoization

See [`obc`](obc.md) for a detailed description of the memoization
method. The memoization method can be used to solve the Lyapunov problem
as well and its implementation is shared with the memoization method for
the retarded boundary conditions.
