# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Different mixing schemes for the self-consistent solution of the
Poisson equation."""
from abc import ABC, abstractmethod

import numpy as np

from qttools import NDArray


class Mixer(ABC):
    """Abstract base class for mixing schemes."""

    @abstractmethod
    def mix(self, previous_value: NDArray, incoming_value: NDArray) -> NDArray:
        """Mixes the incoming value with the previous value."""
        ...


class Underrelaxation(Mixer):
    """Simple underrelaxation mixer.

    Parameters
    ----------
    alpha : float
        Underrelaxation factor. Should be between 0 and 1.

    """

    def __init__(self, alpha: float = 0.5):
        """Initializes the underrelaxation mixer."""

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

    This implementation is a straightforward version of the DIIS
    algorithm. It stores a history of previous values and their
    corresponding residuals, and uses this history to compute an optimal
    linear combination of the previous values to minimize the residual.
    The `epsilon` parameter is used for Tikhonov regularization to
    ensure numerical stability when solving the least-squares problem.

    Parameters
    ----------
    max_history : int
        Maximum number of previous values and residuals to store for the
        DIIS extrapolation.
    epsilon : float
        Regularization parameter for the least-squares problem to ensure
        numerical stability.

    """

    def __init__(self, max_history: int = 5, epsilon: float = 1e-10):
        """Initializes the DIIS mixer."""
        self.max_history = max_history
        self.epsilon = epsilon

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
        NDArray
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

        # If we don't have enough entries, fall back to underrelaxation.
        # TODO: We could pass a parameter for the underrelaxation factor
        # here, or even do a dynamic adjustment based on the norm of the
        # residuals.
        if num_entries < 2:
            return Underrelaxation().mix(incoming_value, previous_value)

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

        # NOTE: Optionally, we could even do another underrelaxation
        # step here to further stabilize the mixing.
        # mixed_value = Underrelaxation(0.8).mix(mixed_value, incoming_value)
        return mixed_value
