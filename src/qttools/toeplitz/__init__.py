# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the methods to acclerate operations with Toeplitz matrices.

For expansion of blocks, we have to differentiate between the following
cases:
- The expansion of summed up k-point matrices where contacts can be
phi-circulant in transverse direction and toeplitz in transport.
- The expansion of single real space matrices where contacts are not
phi-circulant but still toeplitz in transport direction.
- The expansion for the full matrix (real or k-space) of the system,
which shares the same structure as the previsous cases.

Furtheremore, we can differentiate between how the inputs are provided
and the shape of the output. We currently have a mixed functionality
where the inputs can be full blocks which are sliced or `dict` of the
already slicesd unit cells.

TODO: Refactor the code to have a more consistent interface and unify
the different cases.

NOTE: All of the current functions assume correct order of the inputs
and will not check that the inputs are consistent with the expected
structure.

"""
