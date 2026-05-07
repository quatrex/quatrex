# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Different mixing schemes for self-consistent loops."""
from abc import ABC, abstractmethod

import numpy as np

from qttools import NDArray


class Mixer(ABC):
    """Abstract base class for mixing schemes."""

    @abstractmethod
    def mix(self, previous_value: NDArray, incoming_value: NDArray) -> NDArray:
        """Mixes the incoming value with the previous value."""
        ...


class UnderRelaxation(Mixer):
    """Simple under-relaxation mixer.

    Parameters
    ----------
    alpha : float
        Under-relaxation factor. Should be in (0, 1].

    """

    def __init__(self, alpha: float = 0.5):
        """Initializes the under-relaxation mixer."""

        if not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def mix(self, previous_value: NDArray, incoming_value: NDArray) -> NDArray:
        """Mixes the incoming value with the previous value.

        Parameters
        ----------
        previous_value : NDArray
            The value from the previous iteration (e.g., the potential
            from the previous iteration).
        incoming_value : NDArray
            The newly computed value (e.g., the potential computed from
            the current charge density).

        Returns
        -------
        NDArray
            The mixed value.

        """
        return self.alpha * incoming_value + (1 - self.alpha) * previous_value


class DIIS(Mixer):
    """DIIS mixing scheme.

    This implementation is version of "periodic" Pulay mixing algorithm
    [^1]. It alternates between simple under-relaxation steps and DIIS
    extrapolation steps to improve stability, especially in the early
    iterations when the residuals may be large and the history is not
    yet well-established.

    The mixer stores a history of previous values and their
    corresponding residuals, and uses this history to compute an optimal
    linear combination of the previous values to minimize the residual.
    The `epsilon` parameter is used for Tikhonov regularization to
    ensure numerical stability when solving the least-squares problem.

    [^1]: A. S. Banerjee, P. Suryanarayana, and J. E. Pask, "Periodic
    Pulay method for robust and efficient convergence acceleration of
    self-consistent field iterations", Chem. Phys. Lett., 2016.

    Parameters
    ----------
    max_history : int, optional
        Maximum number of previous values and residuals to store for the
        DIIS extrapolation.
    epsilon : float, optional
        Regularization parameter for the least-squares problem to ensure
        numerical stability.
    alpha : float, optional
        Under-relaxation factor to use in the early iterations or in
        between DIIS extrapolation steps. Should be in (0, 1].
    extrapolation_interval : int, optional
        Number of iterations between DIIS extrapolation steps. For
        example, if set to 3, the mixer will perform two
        under-relaxation steps followed by a DIIS extrapolation step,
        and then repeat this cycle. If set to 1 (the default), the Pulay
        mixing is performed at every iteration.

    """

    def __init__(
        self,
        max_history: int = 5,
        epsilon: float = 1e-5,
        alpha: float = 0.5,
        extrapolation_interval: int = 1,
    ):
        """Initializes the DIIS mixer."""
        self.max_history = max_history
        self.epsilon = epsilon
        self.alpha = alpha
        self.extrapolation_interval = extrapolation_interval

        self.call_count = 0

        # NOTE: If this ends up causing memory issues, we can
        # preallocate these arrays and use a circular buffer approach.
        self.history = []
        self.residuals = []

    def mix(self, previous_value: NDArray, incoming_value: NDArray) -> NDArray:
        """Mixes the incoming value with the previous value.

        Parameters
        ----------
        previous_value : NDArray
            The value from the previous iteration (e.g., the potential
            from the previous iteration).
        incoming_value : NDArray
            The newly computed value (e.g., the potential computed from
            the current charge density).

        Returns
        -------
        mixed_value : NDArray
            The mixed value.

        """

        # Update history.
        if len(self.history) == self.max_history:
            self.history.pop(0)
            self.residuals.pop(0)

        self.history.append(incoming_value)

        residual = incoming_value - previous_value
        self.residuals.append(residual)

        num_entries = len(self.history)
        self.call_count += 1

        # If we don't have enough entries, fall back to under-relaxation.
        # TODO: We could pass a parameter for the under-relaxation factor
        # here, or even do a dynamic adjustment based on the norm of the
        # residuals.
        if num_entries < 2 or self.call_count % self.extrapolation_interval != 0:
            return UnderRelaxation(self.alpha).mix(previous_value, incoming_value)

        # Construct the matrix for the least-squares problem.
        B = np.zeros((num_entries + 1, num_entries + 1))
        B[:num_entries, :num_entries] = np.dot(
            np.array(self.residuals).reshape(num_entries, -1),
            np.array(self.residuals).reshape(num_entries, -1).T,
        )

        # Tikhonov regularization to ensure numerical stability. (Trick
        # to avoid issues with near-singular matrices.)
        B[:num_entries, :num_entries] += self.epsilon * np.eye(num_entries)

        # Lagrange multiplier to enforce the constraint that the
        # coefficients sum to 1.
        B[-1, :num_entries] = 1
        B[:num_entries, -1] = 1
        B[-1, -1] = 0

        # Right-hand side for the least-squares problem.
        rhs = np.zeros(num_entries + 1)
        rhs[-1] = 1

        # Solve the least-squares problem to find the optimal coefficients.
        coeffs, *__ = np.linalg.lstsq(B, rhs, rcond=None)
        coeffs = coeffs[:-1]  # Exclude the last constrained coefficient.

        # Compute the mixed value.
        mixed_value = np.einsum("i,ik->k", coeffs, self.history)

        # NOTE: Optionally, we could even do another under-relaxation
        # step here to further stabilize the mixing.
        # mixed_value = UnderRelaxation(self.alpha).mix(previous_value, mixed_value)
        return mixed_value
