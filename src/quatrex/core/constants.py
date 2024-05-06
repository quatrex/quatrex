# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from scipy.constants import physical_constants

c_0 = 1e10 * physical_constants["speed of light in vacuum"][0]  # angstrom / s
e = physical_constants["elementary charge"][0]  # C
hbar = physical_constants["reduced Planck constant in eV s"][0]  # eV s
h = physical_constants["Planck constant in eV s"][0]  # eV s
alpha = physical_constants["fine-structure constant"][0]  # dimensionless
epsilon_0 = e**2 / (2 * alpha * h * c_0)
mu_0 = 2 * alpha * h / (c_0 * e**2)
k_B = physical_constants["Boltzmann constant in eV/K"][0]  # eV / K
