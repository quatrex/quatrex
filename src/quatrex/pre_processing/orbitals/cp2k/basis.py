# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""This module provides classes and functions for parsing and handling CP2K basis set files."""


import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from scipy.constants import physical_constants

from qttools import NDArray

bohr = physical_constants["Bohr radius"][0] * 1e10  # Å


@dataclass(frozen=True)
class CP2KParameterSet:
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
    def from_lines(cls, lines: list[str]) -> Self:
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


@dataclass(frozen=True)
class CP2KBasisSet:
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
    parameter_sets: list[CP2KParameterSet]
    """List of parameter sets for this basis set."""

    @classmethod
    def from_string(cls, basis_set_string: str) -> Self:
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
                CP2KParameterSet.from_lines(lines[offset : offset + num_exponents + 1])
            )
            offset += num_exponents + 1

        return cls(
            element=element,
            name=" ".join(name),
            num_parameter_sets=num_parameter_sets,
            parameter_sets=parameter_sets,
        )


@dataclass(frozen=True)
class CP2KBasisData:
    """Represents a CP2K basis set file.

    These are files like `BASIS_MOLOPT`, `BASIS_MOLOPT_UCL`, etc.

    """

    basis_sets: list[CP2KBasisSet]

    def get_basis_set(self, element: str, name: str) -> CP2KBasisSet:
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
    def from_file(cls, path: str | Path) -> Self:
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
        basis_sets = [CP2KBasisSet.from_string(basis_set) for basis_set in basis_sets]
        return cls(basis_sets=basis_sets)
