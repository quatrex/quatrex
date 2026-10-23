# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import re
from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np
import opt_einsum as oe
from scipy.constants import physical_constants
from scipy.special import comb

from qttools import NDArray
from quatrex.pre_processing.orbitals.orbitals import RealSpaceOrbitals

bohr = physical_constants["Bohr radius"][0] * 1e10  # Å


@dataclass
class ParameterSet:
    """Represents a CP2K parameter set.

    Each basis set can contain multiple parameter sets, each
    corresponding to a specific angular momentum range and set of
    exponents and contraction coefficients.

    """

    n: int
    """Principal quantum number."""
    l_min: int
    """Minimum angular momentum quantum number."""
    l_max: int
    """Maximum angular momentum quantum number."""
    num_exponents: int
    """Number of exponents in the parameter set."""
    num_contractions: list[int]
    """Number of contractions for each l."""
    exponents: NDArray
    """Exponents for the Gaussians. These are read in (Bohr^-2) and converted to Å^-2."""
    coefficients: list[NDArray]
    """Contraction coefficients for each l."""

    @classmethod
    def from_lines(cls, lines: list[str]) -> "ParameterSet":
        """Creates a ParameterSet object from a list of lines.

        Parameters
        ----------
        lines : list of str
            The lines specifying the set of parameters.

        Returns
        -------
        parameter_set : ParameterSet
            A ParameterSet object containing the parsed information.

        """
        n, l_min, l_max, num_exponents = map(int, lines[0].split()[:4])
        num_contractions = list(map(int, lines[0].split()[4 : 4 + l_max - l_min + 1]))

        values = np.genfromtxt(lines[1:])
        exponents = values[:, 0] / bohr**2  # Convert from Bohr^-2 to Å^-2
        coefficients = np.array_split(
            values[:, 1:], np.cumsum(num_contractions)[:-1], axis=1
        )

        return cls(
            n=n,
            l_min=l_min,
            l_max=l_max,
            num_exponents=num_exponents,
            num_contractions=num_contractions,
            exponents=exponents,
            coefficients=coefficients,
        )


@dataclass
class BasisSet:
    """Represents a CP2K basis set.

    A basis set for a given element can contain multiple parameter sets,
    each corresponding to a specific angular momentum range and set of
    exponents and contraction coefficients.

    """

    element: str
    """The chemical element for which the basis set is defined."""
    name: str
    """The name of the basis set."""
    num_parameter_sets: int
    """The number of parameter sets in this basis set."""
    parameter_sets: list[ParameterSet]
    """List of parameter sets for this basis set."""

    @classmethod
    def from_string(cls, basis_set_string: str) -> "BasisSet":
        """Creates a BasisSet object from a string representation.

        Parameters
        ----------
        basis_set_string : str
            The string representation of the basis set.

        Returns
        -------
        basis_set : BasisSet
            A BasisSet object containing the parsed information.

        """
        lines = basis_set_string.strip().splitlines()
        element, *name = lines[0].split()
        num_parameter_sets = int(lines[1])

        # Parse each parameter set.
        parameter_sets = []
        offset = 2
        for __ in range(num_parameter_sets):
            num_exponents = int(lines[offset].split()[3])
            parameter_sets.append(
                ParameterSet.from_lines(lines[offset : offset + num_exponents + 1])
            )
            offset += num_exponents + 1

        return cls(
            element=element,
            name=" ".join(name),
            num_parameter_sets=num_parameter_sets,
            parameter_sets=parameter_sets,
        )


@dataclass
class CP2KBasis:
    """Represents a CP2K basis set file.

    These are files like `BASIS_MOLOPT`, `BASIS_MOLOPT_UCL`, etc.

    """

    basis_sets: list[BasisSet]

    def get_basis_set(self, element: str, name: str) -> BasisSet:
        """Retrieves a basis set for a given element and name.

        Parameters
        ----------
        element : str
            The chemical element for which to retrieve the basis set.
        name : str
            The name of the basis set. This can be a partial name, and
            the method will return the first matching basis set.

        Returns
        -------
        basis_set : BasisSet
            The requested basis set.

        Raises
        ------
        ValueError
            If the specified basis set is not found.

        """
        for basis_set in self.basis_sets:
            if basis_set.element == element and name in basis_set.name:
                return basis_set

        raise ValueError(f"Basis set '{name}' for element '{element}' not found.")

    @classmethod
    def from_file(cls, path: str | Path) -> "CP2KBasis":
        """Read a CP2K basis set file and return a BasisFile object.

        These are files like `BASIS_MOLOPT`, `BASIS_MOLOPT_UCL`, etc.

        Parameters
        ----------
        file : str or Path
            Path to the basis set file.

        Returns
        -------
        basis_file : BasisFile
            A BasisFile object containing the parsed basis sets.

        """
        path = Path(path)
        with open(path, "r") as f:
            lines = f.readlines()

        # Remove comments and empty lines.
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != "" and line[0] != "#"]

        # Split everything per basis set.
        basis_sets = re.split("(^[A-Z])", "\n".join(lines), flags=re.MULTILINE)
        basis_sets = [
            el + params for el, params in zip(basis_sets[1::2], basis_sets[2::2])
        ]

        # Parse each basis set.
        basis_sets = [BasisSet.from_string(basis_set) for basis_set in basis_sets]
        return cls(basis_sets=basis_sets)


def tuples_summing_to(total: int):
    """Generates all tuples of 3 integers summing to a given total.

     Parameters
     ----------
     total : int
         The sum for which to generate the tuples.

     Yields
     ------
     tuple : tuple of int
         A tuple of 3 integers that sum to the specified total.

     Examples
     --------
     Generate all tuples summing to 2:
    >>> list(get_tuples_summing_to(2))
     [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]

    """

    for i in range(total + 1):
        for j in range(total - i + 1):
            yield i, j, total - i - j


def cartesian_to_spherical_coeff(m: int, l_x: int, l_y: int, l_z: int) -> complex:
    """Computes the contribution from cart. to pure sph. harmonic GTO.

    This implements formula (12) in [^1].

    [^1]: H. B. Schlegel and M. J. Frisch, "Transformation between
    Cartesian and pure spherical harmonic Gaussians", Int J of Quantum
    Chemistry, 1995.

    Parameters
    ----------
    m : int
        Magnetic quantum number.
    l_x : int
        Angular momentum in the x-direction.
    l_y : int
        Angular momentum in the y-direction.
    l_z : int
        Angular momentum in the z-direction.

    Returns
    -------
    coeff : complex
        Coefficient for the contribution from the Cartesian to the pure
        spherical harmonic Gaussian-type orbital.

    """
    l = l_x + l_y + l_z

    j = (l_x + l_y - abs(m)) / 2

    if j < 0 or j != int(j):
        return 0.0
    else:
        j = int(j)

    numerator = (
        factorial(2 * l_x)
        * factorial(2 * l_y)
        * factorial(2 * l_z)
        * factorial(l)
        * factorial(l - abs(m))
    )
    denominator = (
        factorial(2 * l)
        * factorial(l_x)
        * factorial(l_y)
        * factorial(l_z)
        * factorial(l + abs(m))
    )
    first_term = np.emath.sqrt(numerator / denominator)

    second_term_prefactor = 1 / (2**l * factorial(l))
    second_term_sum = 0
    for i in range((l - abs(m)) // 2 + 1):
        temp_0 = comb(l, i) * comb(i, j)
        temp_1 = (-1) ** i * factorial(2 * l - 2 * i) / factorial(l - abs(m) - 2 * i)
        second_term_sum += temp_0 * temp_1
    second_term = second_term_prefactor * second_term_sum

    third_term = sum(
        (comb(j, k) * comb(abs(m), l_x - 2 * k))
        * (-1) ** float(np.sign(m) * (abs(m) - l_x + 2 * k) / 2)
        for k in range(j + 1)
    )

    return first_term * second_term * third_term


def primitive_gto(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    l_x: int,
    l_y: int,
    l_z: int,
    alpha_i: float,
) -> NDArray | float:
    """Evaluates a primitive Gaussian-type orbital.

    Parameters
    ----------
    x : NDArray
        The x-coordinates at which to evaluate the orbital.
    y : NDArray
        The y-coordinates at which to evaluate the orbital.
    z : NDArray
        The z-coordinates at which to evaluate the orbital.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.
    alpha_i : float
        The exponent of the Gaussian.

    Returns
    -------
    primitive_gto : ndarray or float
        The evaluated primitive Gaussian-type orbital.

    """
    return x**l_x * y**l_y * z**l_z * np.exp(-alpha_i * (x**2 + y**2 + z**2))


def primitive_gto_gradient(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    l_x: int,
    l_y: int,
    l_z: int,
    alpha_i: float,
) -> NDArray:
    """Evaluates the gradient of a primitive Gaussian-type orbital.

    Parameters
    ----------
    x : NDArray
        The x-coordinates at which to evaluate the orbital.
    y : NDArray
        The y-coordinates at which to evaluate the orbital.
    z : NDArray
        The z-coordinates at which to evaluate the orbital.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.
    alpha_i : float
        The exponent of the Gaussian.

    Returns
    -------
    gradient : NDArray
        The gradient of the primitive GTO, which is a 3-element array
        containing the derivatives with respect to x, y, and z.

    """
    exp_factor = np.exp(-alpha_i * (x**2 + y**2 + z**2))

    grad_x = (
        l_x * x ** (l_x - 1) * y**l_y * z**l_z * exp_factor
        - 2 * alpha_i * x ** (l_x + 1) * y**l_y * z**l_z * exp_factor
    )

    grad_y = (
        l_y * x**l_x * y ** (l_y - 1) * z**l_z * exp_factor
        - 2 * alpha_i * x**l_x * y ** (l_y + 1) * z**l_z * exp_factor
    )

    grad_z = (
        l_z * x**l_x * y**l_y * z ** (l_z - 1) * exp_factor
        - 2 * alpha_i * x**l_x * y**l_y * z ** (l_z + 1) * exp_factor
    )

    return np.array([grad_x, grad_y, grad_z])


def primitive_gto_hessian(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    l_x: int,
    l_y: int,
    l_z: int,
    alpha_i: float,
) -> NDArray:
    """Evaluates the Hessian tensor of a primitive Gaussian-type orbital.

    Parameters
    ----------
    x : NDArray
        The x-coordinates at which to evaluate the orbital.
    y : NDArray
        The y-coordinates at which to evaluate the orbital.
    z : NDArray
        The z-coordinates at which to evaluate the orbital.
    l_x : int
        The angular momentum in the x-direction.
    l_y : int
        The angular momentum in the y-direction.
    l_z : int
        The angular momentum in the z-direction.
    alpha_i : float
        The exponent of the Gaussian.

    Returns
    -------
    hessian : NDArray
        The Hessian tensor of the primitive GTO, which is a 3x3 matrix
        containing the second derivatives with respect to x, y, and z.

    """
    exp_factor = np.exp(-alpha_i * (x**2 + y**2 + z**2))

    def second_derivative(
        coordinate: NDArray,
        angular_momentum: int,
        other_polynomial: NDArray,
    ) -> NDArray:
        result = np.zeros_like(exp_factor)
        if angular_momentum > 1:
            result += (
                angular_momentum
                * (angular_momentum - 1)
                * coordinate ** (angular_momentum - 2)
            )
        result -= (
            2 * alpha_i * (2 * angular_momentum + 1) * coordinate**angular_momentum
        )
        result += 4 * alpha_i**2 * coordinate ** (angular_momentum + 2)
        return result * other_polynomial * exp_factor

    x_polynomial = y**l_y * z**l_z
    y_polynomial = x**l_x * z**l_z
    z_polynomial = x**l_x * y**l_y
    hessian_xx = second_derivative(x, l_x, x_polynomial)
    hessian_yy = second_derivative(y, l_y, y_polynomial)
    hessian_zz = second_derivative(z, l_z, z_polynomial)
    hessian_xy = (
        ((l_x * x ** (l_x - 1) if l_x > 0 else 0) - 2 * alpha_i * x ** (l_x + 1))
        * ((l_y * y ** (l_y - 1) if l_y > 0 else 0) - 2 * alpha_i * y ** (l_y + 1))
        * z**l_z
        * exp_factor
    )
    hessian_xz = (
        ((l_x * x ** (l_x - 1) if l_x > 0 else 0) - 2 * alpha_i * x ** (l_x + 1))
        * ((l_z * z ** (l_z - 1) if l_z > 0 else 0) - 2 * alpha_i * z ** (l_z + 1))
        * y**l_y
        * exp_factor
    )
    hessian_yz = (
        ((l_y * y ** (l_y - 1) if l_y > 0 else 0) - 2 * alpha_i * y ** (l_y + 1))
        * ((l_z * z ** (l_z - 1) if l_z > 0 else 0) - 2 * alpha_i * z ** (l_z + 1))
        * x**l_x
        * exp_factor
    )

    return np.array(
        [
            [hessian_xx, hessian_xy, hessian_xz],
            [hessian_xy, hessian_yy, hessian_yz],
            [hessian_xz, hessian_yz, hessian_zz],
        ]
    )


class CP2KOrbitals(RealSpaceOrbitals):
    """Class to evaluate CP2K orbitals based on a given parameter set.

    Parameters
    ----------
    parameter_set : ParameterSet
        The parameter set containing the exponents and contraction coefficients
        for the orbitals.

    Attributes
    ----------
    parameter_set : ParameterSet
        The parameter set containing the exponents and contraction
        coefficients for the orbitals.
    gto_normalizations : dict[tuple[int], NDArray]
        A dictionary mapping each unique (l_x, l_y, l_z) tuple to its
        corresponding set of normalization factors for the corresponding
        primitive GTOs.
    contraction_normalizations : list[NDArray]
        A list of arrays containing the normalization factors for the
        contracted GTOs for each angular momentum l.
    cartesian_to_spherical_transforms : list[NDArray]
        A list of transformation matrices that convert a set of
        Cartesian GTOs to spherical GTOs for each angular momentum l
        between l_min and l_max.

    """

    def __init__(self, parameter_set: ParameterSet):

        self.parameter_set = parameter_set

        # These are the unique l-tuples that have to be considered to
        # construct the full set of basis functions.
        unique_l_tuples = {
            t
            for l in range(parameter_set.l_min, parameter_set.l_max + 1)
            for t in tuples_summing_to(l)
        }

        unique_l_tuples = sorted(unique_l_tuples)

        self.gto_normalizations = self._init_gto_normalizations(
            self.parameter_set, unique_l_tuples
        )
        self.contraction_normalizations = self._init_contraction_normalizations(
            self.parameter_set
        )
        self.cartesian_to_spherical_transforms = (
            self._init_cartesian_to_spherical_transforms(self.parameter_set)
        )

    @staticmethod
    def _init_gto_normalizations(
        parameter_set: ParameterSet, unique_l_tuples: list[tuple[int]]
    ) -> dict[tuple[int], NDArray]:
        """Compute the individual GTO normalization factors.

        These factors normalize the individual primitive GTOs based on
        their angular momentum components (l_x, l_y, l_z) and the
        exponents provided in the parameter set.

        Parameters
        ----------
        parameter_set : ParameterSet
            The parameter set containing the exponents.
        unique_l_tuples : list[tuple[int]]
            A list of unique tuples representing the angular momentum
            components (l_x, l_y, l_z) for which the normalization
            factors need to be computed.

        Returns
        -------
        dict[tuple[int], NDArray]
            A dictionary mapping each unique (l_x, l_y, l_z) tuple to
            its corresponding set of GTO normalization factors.

        """
        normalization = {}
        for l_x, l_y, l_z in unique_l_tuples:
            factors = (2 * parameter_set.exponents / np.pi) ** (3 / 4)
            factors *= np.sqrt((8 * parameter_set.exponents) ** (l_x + l_y + l_z))
            factors *= np.sqrt(factorial(l_x) * factorial(l_y) * factorial(l_z))
            factors /= np.sqrt(
                factorial(2 * l_x) * factorial(2 * l_y) * factorial(2 * l_z)
            )

            normalization[l_x, l_y, l_z] = factors

        return normalization

    @staticmethod
    def _init_contraction_normalizations(parameter_set: ParameterSet) -> list[NDArray]:
        """Compute the contraction normalization factors.

        These factors normalize the contracted GTOs based on the
        contraction coefficients and the exponents provided in the
        parameter set.

        Parameters
        ----------
        parameter_set : ParameterSet
            The parameter set containing the exponents and contraction
            coefficients.

        Returns
        -------
        list[NDArray]
            A list of arrays, where each one contains the contraction
            normalization factors for a specific angular momentum l.

        """
        normalization = []
        for l_ind, l in enumerate(range(parameter_set.l_min, parameter_set.l_max + 1)):
            factors = np.zeros((parameter_set.num_contractions[l_ind],))
            for contraction_ind, coefficients in enumerate(
                parameter_set.coefficients[l_ind].T
            ):
                factor = sum(
                    c_i * c_j * (2 * np.sqrt(a_i * a_j) / (a_i + a_j)) ** (l + 1.5)
                    for a_i, c_i in zip(parameter_set.exponents, coefficients)
                    for a_j, c_j in zip(parameter_set.exponents, coefficients)
                )
                factors[contraction_ind] = 1 / np.sqrt(factor)

            normalization.append(factors)

        return normalization

    @staticmethod
    def _init_cartesian_to_spherical_transforms(
        parameter_set: ParameterSet,
    ) -> list[NDArray]:
        """Compute the transformation matrices from Cartesian to spherical GTOs.

        These matrices convert a set of Cartesian GTOs to spherical GTOs
        for each angular momentum l between l_min and l_max.

        The method is described in the paper by Schlegel and Frisch[^1].

        [^1]: H. B. Schlegel and M. J. Frisch, "Transformation between
        Cartesian and pure spherical harmonic Gaussians", Int J of
        Quantum Chemistry, 1995.

        Parameters
        ----------
        parameter_set : ParameterSet
            The parameter set containing the angular momentum range.

        Returns
        -------
        list[NDArray]
            A list of transformation matrices, where each matrix
            corresponds to a specific angular momentum l and converts
            Cartesian GTOs to spherical GTOs.

        """

        transforms = []
        for l in range(parameter_set.l_min, parameter_set.l_max + 1):
            l_tuples = list(tuples_summing_to(l))
            transform = np.zeros((2 * l + 1, len(l_tuples)))
            for m_ind, m in enumerate(range(-l, l + 1)):
                coeff = np.array(
                    [cartesian_to_spherical_coeff(m, *l_tuple) for l_tuple in l_tuples]
                )
                if m != 0:
                    coeff_m = np.array(
                        [
                            cartesian_to_spherical_coeff(-m, *l_tuple)
                            for l_tuple in l_tuples
                        ]
                    )
                    if m > 0:
                        coeff = (coeff_m + coeff) / np.sqrt(2)
                    else:
                        coeff = (coeff_m - coeff) / (1j * np.sqrt(2))

                transform[m_ind] = coeff.real

            transforms.append(transform)

        return transforms

    def _evaluate(
        self,
        positions: NDArray,
        primitive: Callable[[NDArray, NDArray, NDArray, int, int, int, float], NDArray],
    ) -> NDArray:
        """Evaluate a primitive function at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the function. Should be
            of shape (N, 3), N being the number of points.
        primitive : Callable
            A callable that takes in x, y, z coordinates, angular
            momentum components (l_x, l_y, l_z), and an exponent alpha,
            and returns the evaluated primitive function.

        Returns
        -------
        values : NDArray
            The evaluated values of the function at the given positions.

        """
        values = []

        for i_l, l in enumerate(
            range(self.parameter_set.l_min, self.parameter_set.l_max + 1)
        ):
            l_tuples = list(tuples_summing_to(l))
            # Evaluate primitives.
            primitive_values = np.array(
                [
                    primitive(*positions.T, l_x, l_y, l_z, alpha)
                    for l_x, l_y, l_z in l_tuples
                    for alpha in self.parameter_set.exponents
                ]
            )
            primitive_values = primitive_values.reshape(
                len(l_tuples),
                self.parameter_set.num_exponents,
                *primitive_values.shape[1:],
            )

            # Normalize primitives.
            normalizations = np.vstack(
                [self.gto_normalizations[l_tuple] for l_tuple in l_tuples]
            )
            # Expand the normalization array to match the shape of
            # primitive_values for broadcasting.
            num_value_dims = primitive_values.ndim - normalizations.ndim
            normalizations = normalizations[(...,) + (np.newaxis,) * num_value_dims]
            normalized = primitive_values * normalizations

            # Perform a set of contractions for the current angular
            # momentum.
            for i_c, coefficients in enumerate(self.parameter_set.coefficients[i_l].T):
                contraction_normalization = self.contraction_normalizations[i_l][i_c]
                contracted = (
                    oe.contract("j,ij...->i...", coefficients, normalized)
                    * contraction_normalization
                )

                # Transform from Cartesian to spherical harmonics.
                values.append(
                    oe.contract(
                        "mj,j...->m...",
                        self.cartesian_to_spherical_transforms[i_l],
                        contracted,
                    )
                )

        return np.vstack(values)

    def evaluate_orbitals(self, positions: NDArray) -> NDArray:
        """Evaluate the CP2K orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbitals. Should be
            of shape (N, 3), N being the number of points.

        Returns
        -------
        orbitals : NDArray
            The evaluated orbitals at the given positions. The shape of
            the output will be (M, N), where M is the total number of
            orbitals and N is the number of points.

        """
        return self._evaluate(positions, primitive_gto)

    def evaluate_orbital_gradients(self, positions: NDArray) -> NDArray:
        """Evaluate gradients of CP2K orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital gradients. Should be
            of shape (N, 3), N being the number of points.

        Returns
        -------
        gradients : NDArray
            The evaluated orbital gradients at the given positions. The
            shape of the output will be (M, N, 3), where M is the total
            number of orbitals, 3 corresponds to the x, y, z components
            of the gradient, and N is the number of points.

        """
        return self._evaluate(positions, primitive_gto_gradient).swapaxes(1, 2)

    def evaluate_orbital_hessians(self, positions: NDArray) -> NDArray:
        """Evaluate Hessians of CP2K orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital Hessians. Should be
            of shape (N, 3), N being the number of points.

        Returns
        -------
        hessians : NDArray
            The evaluated orbital Hessians at the given positions. The
            shape of the output will be (M, N, 3, 3), where M is the
            total number of orbitals, 3x3 corresponds to the Hessian
            matrix, and N is the number of points.

        """
        return self._evaluate(positions, primitive_gto_hessian).swapaxes(1, 3)
