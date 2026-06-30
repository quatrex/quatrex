# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import meshio
import numpy as np
import skfem
from scipy import sparse
from skfem.helpers import dot, grad, mul

from qttools import NDArray


def assemble_stiffness_matrix(
    basis: skfem.Basis, epsilon_r: NDArray | None = None
) -> sparse.csr_matrix:
    """Assemble the stiffness matrix for a tetrahedral mesh.

    Parameters
    ----------
    basis : skfem.Basis
        The basis for the tetrahedral mesh, which contains the
        necessary information about the mesh and the finite element
        space.
    epsilon_r : NDArray, optional
        The relative permittivity tensor at the quadrature points. If
        not provided, it is assumed to be the identity tensor (i.e.,
        free space).

    Returns
    -------
    sparse.csr_matrix
        The assembled stiffness matrix.

    """
    if epsilon_r is not None:
        # Validate the provided epsilon_r.
        if epsilon_r.shape[-1] != basis.N:
            raise ValueError(
                f"Expected epsilon_r to have shape (..., {basis.N}), "
                f"but got {epsilon_r.shape}."
            )

        if epsilon_r.ndim == 3:
            if epsilon_r.shape[:-1] != (3, 3):
                raise ValueError(
                    f"Expected epsilon_r to have shape (3, 3, ...), "
                    f"but got {epsilon_r.shape}."
                )
        elif epsilon_r.ndim < 3:
            if epsilon_r.ndim == 2 and epsilon_r.shape[0] != 3:
                raise ValueError(
                    f"Expected epsilon_r to have shape (3, ...), "
                    f"but got {epsilon_r.shape}."
                )

            # NOTE: Broadcasting works like this for both 1D and 2D
            # cases, since the last dimension is the number of DOFs.
            epsilon_r = epsilon_r * np.stack(basis.N * [np.eye(3)], axis=-1)

        else:
            raise ValueError(f"epsilon_r must be 1D, 2D, or 3D, got {epsilon_r.ndim}D.")

    else:
        epsilon_r = np.stack(basis.N * [np.eye(3)], axis=-1)

    # Interpolate the permittivity tensor to the quadrature points.
    epsilon_r_quad = np.zeros((3, 3, *basis.basis[0][0].shape), dtype=np.float64)
    for i, j in np.ndindex(3, 3):
        epsilon_r_quad[i, j] = basis.interpolate(epsilon_r[i, j])

    @skfem.BilinearForm
    def laplace(u, v, w):
        return dot(mul(w.epsilon_r, grad(u)), grad(v))

    return laplace.assemble(basis, epsilon_r=epsilon_r_quad)


def initialize_tetrahedral_mesh(
    mesh: meshio.Mesh,
) -> tuple[skfem.MeshTet, skfem.Basis]:
    """Initialize a tetrahedral mesh and its basis.

    Parameters
    ----------
    mesh : meshio.Mesh
        The tetrahedral mesh containing the nodes and elements.

    Returns
    -------
    mesh : skfem.MeshTet
        The initialized tetrahedral mesh.
    basis : skfem.Basis
        The basis for the tetrahedral mesh.

    """

    elements = mesh.cells_dict["tetra"]
    nodes = mesh.points

    mesh = skfem.MeshTet(nodes.T, elements.T)
    basis = skfem.Basis(mesh, skfem.ElementTetP1())

    return mesh, basis


def assemble_mfc_transform(
    mesh: skfem.MeshTet,
    basis: skfem.Basis,
    cell_vectors: NDArray,
    pbc: tuple,
):
    """Assemble the transform to impose periodic boundary conditions.

    This function identifies pairs of nodes that are periodic images of
    each other and constructs a transformation matrix that maps the
    degrees of freedom of the mesh to account for these periodicities.

    This multifreedom constraint (MFC) transform enforces that the
    degrees of freedom on periodic boundaries are coupled together.

    Parameters
    ----------
    mesh : skfem.MeshTet
        The tetrahedral mesh containing the nodes and elements.
    basis : skfem.Basis
        The basis for the tetrahedral mesh.
    cell_vectors : NDArray
        The vectors defining the unit cell of the mesh.
    pbc : tuple
        A tuple indicating which boundaries are periodic. Each element
        should be a boolean, where `True` indicates that the
        corresponding boundary is periodic.

    Returns
    -------
    sparse.csr_matrix
        The transformation matrix that maps the degrees of freedom of
        the mesh.

    """
    shape = (basis.N, basis.N)

    if not any(pbc):
        # If no periodic boundary conditions are specified, return the
        # identity matrix.
        return sparse.eye(basis.N, format="csr")

    nodes_fractional = np.linalg.solve(cell_vectors.T, basis.doflocs).T

    periodic_pairs = find_periodic_pairs(
        pbc=pbc, boundary_inds=mesh.boundary_nodes(), nodes_fractional=nodes_fractional
    )

    actual_inds, image_inds = zip(*periodic_pairs)

    actual_inds = np.array(actual_inds, dtype=np.int32)
    image_inds = np.array(image_inds, dtype=np.int32)
    unique_actual_inds = np.unique(actual_inds)

    unconstrained_inds = np.setdiff1d(
        np.arange(basis.N, dtype=np.int32),
        np.union1d(unique_actual_inds, image_inds),
        assume_unique=True,
    )

    transformation_matrix = (
        # Multifreedom mapping.
        sparse.coo_matrix(
            (np.ones(len(actual_inds)), (actual_inds, image_inds)), shape=shape
        )
        # Unconstrained nodes map to themselves.
        + sparse.coo_matrix(
            (
                np.ones(len(unconstrained_inds)),
                (unconstrained_inds, unconstrained_inds),
            ),
            shape=shape,
        )
        # Non-image nodes map to themselves.
        + sparse.coo_matrix(
            (
                np.ones(len(unique_actual_inds)),
                (unique_actual_inds, unique_actual_inds),
            ),
            shape=shape,
        )
    )
    return transformation_matrix.tocsr()


def find_periodic_pairs(
    pbc: tuple,
    boundary_inds: NDArray,
    nodes_fractional: NDArray,
    match_tol: float = 1e-4,
) -> list[tuple[int, int]]:
    """Find pairs of nodes that are periodic images of each other.

    This function identifies pairs of nodes that are periodic images of
    each other based on their fractional coordinates and the periodic
    boundary conditions (PBC) specified.

    Parameters
    ----------
    pbc : tuple
        A tuple indicating which boundaries are periodic. Each element
        should be a boolean, where `True` indicates that the
        corresponding boundary is periodic.
    boundary_inds : NDArray
        An array of indices of the nodes that are on the boundaries of
        the mesh.
    nodes_fractional : NDArray
        An array of fractional coordinates of the nodes in the mesh.
    match_tol : float, optional
        The tolerance for determining whether two nodes are periodic
        images of each other. Default is 1e-4.

    Returns
    -------
    periodic_pairs : list[tuple[int, int]]
        A list of tuples, where each tuple contains the indices of a
        pair of nodes that are periodic images of each other.

    """
    pbc = np.array(pbc, dtype=bool)

    periodic_pairs = []
    unique_actual_inds = set()
    for i in boundary_inds:
        if i in unique_actual_inds:
            # Skip nodes that have already been processed.
            continue

        boundary_mask = (
            np.abs(np.mod(np.round(nodes_fractional[i], decimals=5), 1)) < match_tol
        )
        if not (pbc[boundary_mask]).all():
            # If the node is on a non-periodic boundary, skip it.
            continue

        if boundary_mask.sum() == 0:
            # If the node is not on any periodic boundary, skip it.
            continue

        distances = np.abs(nodes_fractional[i] - nodes_fractional[boundary_inds])
        distances = np.round(distances, decimals=5)
        paired_nodes = (np.mod(distances, 1) < match_tol).all(axis=1).nonzero()[0]

        if paired_nodes.shape[0] != 2 ** boundary_mask.sum():
            raise ValueError(
                f"Expected {2 ** boundary_mask.sum()} paired nodes, "
                f"but found {paired_nodes.shape[0]} for node {i}."
            )

        paired_nodes = boundary_inds[paired_nodes]

        # Sort the paired nodes by coordinates to ensure consistent
        # pairing.
        x, y, z = nodes_fractional[paired_nodes].T
        inds = np.lexsort((z, y, x))
        paired_nodes = paired_nodes[inds]

        unique_actual_inds.add(paired_nodes[0])
        for j in paired_nodes:
            if j in unique_actual_inds:
                continue

            # Ensure we only add each pair once.
            periodic_pairs.append((paired_nodes[0], j))
            unique_actual_inds.add(j)

    return periodic_pairs
