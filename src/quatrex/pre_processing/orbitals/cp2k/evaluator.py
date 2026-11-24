# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Methods for evaluating CP2K orbitals and their derivatives."""

from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from typing import Self

import ase
import numpy as np
import opt_einsum as oe
import pyvista as pv
from scipy import sparse
from scipy.constants import physical_constants
from scipy.special import comb

from qttools import NDArray
from quatrex.core.config import QuatrexConfig
from quatrex.pre_processing.orbitals.cp2k.basis import (
    CP2KBasisData,
    CP2KBasisSet,
    CP2KParameterSet,
)
from quatrex.pre_processing.orbitals.orbitals import (
    DEFAULT_REMESHING_OPTIONS,
    DeviceOrbitalProjector,
    OrbitalEvaluator,
    RemeshingOptions,
)

bohr = physical_constants["Bohr radius"][0] * 1e10  # Å


def _tuples_summing_to(total: int):
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


def _cartesian_to_spherical_coeff(m: int, l_x: int, l_y: int, l_z: int) -> complex:
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


def _primitive_gto(
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


def _primitive_gto_gradient(
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


def _primitive_gto_hessian(
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


@dataclass(frozen=True)
class ProcessedCP2KParameterSet(CP2KParameterSet):
    """A processed version of the CP2KParameterSet.

    This additionally includes precomputed normalization factors and
    transformation matrices.

    """

    gto_normalizations: dict[tuple[int, int, int], NDArray]
    """
    A dictionary mapping each unique (l_x, l_y, l_z) tuple to its
    corresponding set of normalization factors for the corresponding
    primitive GTOs.
    """
    contraction_normalizations: list[NDArray]
    """
    A list of arrays containing the normalization factors for the
    contracted GTOs for each angular momentum l.
    """
    cartesian_to_spherical_transforms: list[NDArray]
    """
    A list of transformation matrices that convert a set of Cartesian
    GTOs to spherical GTOs for each angular momentum l between l_min and
    l_max.
    """

    @staticmethod
    def _init_gto_normalizations(
        parameter_set: CP2KParameterSet, unique_l_tuples: list[tuple[int]]
    ) -> dict[tuple[int], NDArray]:
        """Compute the individual GTO normalization factors.

        These factors normalize the individual primitive GTOs based on
        their angular momentum components (l_x, l_y, l_z) and the
        exponents provided in the parameter set.

        Parameters
        ----------
        parameter_set : CP2KParameterSet
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
    def _init_contraction_normalizations(
        parameter_set: CP2KParameterSet,
    ) -> list[NDArray]:
        """Compute the contraction normalization factors.

        These factors normalize the contracted GTOs based on the
        contraction coefficients and the exponents provided in the
        parameter set.

        Parameters
        ----------
        parameter_set : CP2KParameterSet
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
        parameter_set: CP2KParameterSet,
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
        parameter_set : CP2KParameterSet
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
            l_tuples = list(_tuples_summing_to(l))
            transform = np.zeros((2 * l + 1, len(l_tuples)))
            for m_ind, m in enumerate(range(-l, l + 1)):
                coeff = np.array(
                    [_cartesian_to_spherical_coeff(m, *l_tuple) for l_tuple in l_tuples]
                )
                if m != 0:
                    coeff_m = np.array(
                        [
                            _cartesian_to_spherical_coeff(-m, *l_tuple)
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

    @classmethod
    def from_parameter_set(cls, parameter_set: CP2KParameterSet) -> Self:
        """Creates a ProcessedParameterSet from a CP2KParameterSet.

        Parameters
        ----------
        parameter_set : CP2KParameterSet
            The CP2K parameter set to process.

        Returns
        -------
        processed_set : ProcessedParameterSet
            A ProcessedParameterSet containing the precomputed
            normalization factors and transformation matrices.

        """
        # These are the unique l-tuples that have to be considered to
        # construct the full set of basis functions.
        unique_l_tuples = {
            t
            for l in range(parameter_set.l_min, parameter_set.l_max + 1)
            for t in _tuples_summing_to(l)
        }
        unique_l_tuples = sorted(unique_l_tuples)

        gto_normalizations = cls._init_gto_normalizations(
            parameter_set, unique_l_tuples
        )
        contraction_normalizations = cls._init_contraction_normalizations(parameter_set)
        cartesian_to_spherical_transforms = cls._init_cartesian_to_spherical_transforms(
            parameter_set
        )

        return cls(
            n=parameter_set.n,
            l_min=parameter_set.l_min,
            l_max=parameter_set.l_max,
            num_exponents=parameter_set.num_exponents,
            num_contractions=parameter_set.num_contractions,
            exponents=parameter_set.exponents,
            coefficients=parameter_set.coefficients,
            gto_normalizations=gto_normalizations,
            contraction_normalizations=contraction_normalizations,
            cartesian_to_spherical_transforms=cartesian_to_spherical_transforms,
        )


class CP2KOrbitalEvaluator(OrbitalEvaluator):
    """Class to evaluate CP2K orbitals based on a given basis set.

    Parameters
    ----------
    basis_set : CP2KBasisSet
        A CP2K basis set for some element, containing (possibly
        multiple) parameter sets.


    """

    def __init__(self, basis_set: CP2KBasisSet):
        """Initialize the CP2KOrbitalEvaluator with a given basis set."""

        self.parameter_sets = [
            ProcessedCP2KParameterSet.from_parameter_set(parameter_set)
            for parameter_set in basis_set.parameter_sets
        ]

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

        for parameter_set in self.parameter_sets:
            for i_l, l in enumerate(
                range(parameter_set.l_min, parameter_set.l_max + 1)
            ):
                l_tuples = list(_tuples_summing_to(l))
                # Evaluate primitives.
                primitive_values = np.array(
                    [
                        primitive(*positions.T, l_x, l_y, l_z, alpha)
                        for l_x, l_y, l_z in l_tuples
                        for alpha in parameter_set.exponents
                    ]
                )
                primitive_values = primitive_values.reshape(
                    len(l_tuples),
                    parameter_set.num_exponents,
                    *primitive_values.shape[1:],
                )

                # Normalize primitives.
                normalizations = np.vstack(
                    [parameter_set.gto_normalizations[l_tuple] for l_tuple in l_tuples]
                )
                # Expand the normalization array to match the shape of
                # primitive_values for broadcasting.
                num_value_dims = primitive_values.ndim - normalizations.ndim
                normalizations = normalizations[(...,) + (np.newaxis,) * num_value_dims]
                normalized = primitive_values * normalizations

                # Perform a set of contractions for the current angular
                # momentum.
                for i_c, coefficients in enumerate(parameter_set.coefficients[i_l].T):
                    contraction_normalization = (
                        parameter_set.contraction_normalizations[i_l][i_c]
                    )
                    contracted = (
                        oe.contract("j,ij...->i...", coefficients, normalized)
                        * contraction_normalization
                    )

                    # Transform from Cartesian to spherical harmonics.
                    values.append(
                        oe.contract(
                            "mj,j...->m...",
                            parameter_set.cartesian_to_spherical_transforms[i_l],
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
        return self._evaluate(positions, _primitive_gto)

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
        return self._evaluate(positions, _primitive_gto_gradient).swapaxes(1, 2)

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
        return self._evaluate(positions, _primitive_gto_hessian).swapaxes(1, 3)


class AtomCenteredOrbitalEvaluator(OrbitalEvaluator):
    def __init__(
        self,
        evaluators: dict[str, OrbitalEvaluator],
        structure: ase.Atoms,
    ):
        """Initialize the AtomCenteredOrbitalEvaluator.

        Parameters
        ----------
        evaluators : dict[str, OrbitalEvaluator]
            A dictionary mapping atom types to their corresponding
            orbital evaluators.
        structure : ase.Atoms
            The atomic structure for which the orbitals are to be
            evaluated.

        """
        self.evaluators = evaluators
        self.structure = structure

    def _evaluate(self, positions: NDArray, method: str) -> NDArray:
        """Evaluate the orbitals, gradients, or Hessians at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate. Should be of shape (N,
            3), N being the number of points.
        method : str
            The method to use for evaluation. Can be
            'evaluate_orbitals', 'evaluate_orbital_gradients', or
            'evaluate_orbital_hessians'.

        Returns
        -------
        result : NDArray
            The evaluated values at the given positions.

        """
        values = []
        for position, symbol in zip(
            self.structure.positions, self.structure.get_chemical_symbols()
        ):

            values.append(
                getattr(self.evaluators[symbol], method)(positions - position)
            )

        return np.concatenate(values, axis=0)

    def evaluate_orbitals(self, positions: NDArray) -> NDArray:
        return self._evaluate(positions, "evaluate_orbitals")

    def evaluate_orbital_gradients(self, positions: NDArray) -> NDArray:
        return self._evaluate(positions, "evaluate_orbital_gradients")

    def evaluate_orbital_hessians(self, positions: NDArray) -> NDArray:
        return self._evaluate(positions, "evaluate_orbital_hessians")


class CP2KDeviceOrbitalProjector(DeviceOrbitalProjector):

    def __init__(self, config: QuatrexConfig):
        super().__init__(config)

    @staticmethod
    def _init_orbital_evaluator(
        config: QuatrexConfig, structure: ase.Atoms
    ) -> dict[str, CP2KOrbitalEvaluator]:
        """Initialize the real-space orbitals based on the configuration.

        Parameters
        ----------
        config : QuatrexConfig
            The main quatrex configuration.
        structure : ase.Atoms
            The atomic structure for which the orbitals are to be
            initialized.

        Returns
        -------
        orbitals : dict[str, CP2KOrbitals]
            A dictionary mapping atom types to their corresponding CP2K
            orbitals.

        """
        basis = CP2KBasisData.from_file(config.pre_process.cp2k_basis_file)
        unique_atom_types = list(set(structure.get_chemical_symbols()))

        orbital_evaluators = {}
        for atom_type in unique_atom_types:
            basis_set = basis.get_basis_set(
                atom_type, config.pre_process.cp2k_basis_type
            )
            orbital_evaluators[atom_type] = CP2KOrbitalEvaluator(basis_set)

        return AtomCenteredOrbitalEvaluator(orbital_evaluators, structure)

    def get_initial_mesh(
        self, options: RemeshingOptions | None = None
    ) -> pv.UnstructuredGrid:
        """Get an initial mesh guess for the orbital projections.

        Parameters
        ----------
        options : RemeshingOptions, optional
            Options concerning the remeshing for orbital projection. If
            not provided, the default options will be used.

        Returns
        -------
        initial_mesh : pv.UnstructuredGrid
            The initial mesh to be used for remeshing.

        """
        options = options or DEFAULT_REMESHING_OPTIONS
        coordinates = []
        for i, periodic in enumerate(self.structure.pbc):
            padding = options.padding if not periodic else 0.0
            cell_length = self.structure.cell[i, i]
            x_i = np.linspace(
                -padding,
                cell_length + padding,
                int((cell_length + 2 * padding) // options.initial_spacing),
            )
            coordinates.append(x_i)

        meshgrid = np.meshgrid(*coordinates, indexing="ij")
        initial_mesh = pv.StructuredGrid(*meshgrid).delaunay_3d()

        return initial_mesh

    def assemble_projections(
        self, mesh_nodes: NDArray
    ) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix]]:
        """Evaluate the Basis functions on the device mesh.

        Parameters
        ----------
        mesh_nodes : NDArray
            The coordinates of the mesh nodes where the basis functions
            are to be evaluated. Should be of shape (N, 3), N being the
            number of mesh nodes.

        Returns
        -------
        projections : sparse.csr_matrix
            A sparse matrix containing the evaluated basis functions on
            the mesh. Each row corresponds to a point in the device
            mesh, and each column corresponds to a basis function.
        gradient_matrices : list[sparse.csr_matrix]
            A list of three sparse matrices, each corresponding to the
            gradient components (x, y, z) of the evaluated basis
            functions on the mesh. Each matrix has the same shape as
            `projections`, with each row corresponding to a point in the
            device mesh and each column corresponding to a basis
            function.

        """
        projections = self.orbital_evaluator.evaluate_orbitals(mesh_nodes)
        gradient_projections = self.orbital_evaluator.evaluate_orbital_gradients(
            mesh_nodes
        )

        # Construct three sparse matrices for the gradient components.
        gradient_matrices = []
        for i in range(3):
            gradient_matrices.append(sparse.csr_matrix(gradient_projections[:, :, i].T))

        return sparse.csr_matrix(projections.T), gradient_matrices
