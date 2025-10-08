The `quatrex` codebase is divided into two sub-packages:

- The [`qttools`](qttools/README.md) package contains all general-purpose
  numerical tools, including data structures, non-linear eigenvalue
  solvers, MPI communication utilities, etc.
- The [`quatrex`](quatrex/README.md) package builds on top of this, implementing
  the actual physical models by making use of the tools provided by
  `qttools`.
