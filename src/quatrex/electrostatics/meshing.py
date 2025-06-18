import os
import random
import string

import gmsh
import meshio
import numpy as np
from qttools import NDArray


def _embed_points(
    cell_vectors: NDArray,
    pbc: tuple,
    points: NDArray,
    size_points: float,
    cell: int,
):
    """Embeds points into the mesh, handling periodic boundary conditions.

    Parameters
    ----------
    cell_vectors : NDArray
        The cell vectors defining the parallelepiped cell, shape (3, 3).
    pbc : tuple
        The periodic boundary conditions in each direction.
    points : NDArray
        The points to embed in the mesh, shape (n, 3).
    size_points : float
        The mesh size for the points.
    cell : int
        The tag of the cell in which to embed the points.
    Returns
    -------
    point_tags : set
        The tags of the embedded points.
    """

    points_fractional = np.linalg.solve(cell_vectors.T, points.T).T

    # Add the points.
    point_tags = set()
    for point, point_fractional in zip(points, points_fractional):
        embedded = False
        tag_1 = None
        # Check if the point is on any periodic boundary.
        for i, periodic in enumerate(pbc):
            gmsh.model.occ.synchronize()

            if point_fractional[i] % 1.0 == 0.0:
                if not periodic:
                    raise ValueError(f"Point {point} lies on a non-periodic boundary.")
                if embedded:
                    raise NotImplementedError(
                        f"Point {point} lies on multiple periodic boundaries."
                    )

                point_fractional_1 = np.copy(point_fractional)
                point_fractional_1[i] = 0.0
                point_1 = np.dot(cell_vectors.T, point_fractional_1)
                tag_1 = gmsh.model.occ.add_point(*point_1, meshSize=size_points)
                point_tags.add(tag_1)

                # Point lies on a periodic boundary. We need to add its
                # periodic image.
                point_fractional_2 = np.copy(point_fractional)
                point_fractional_2[i] = 1.0
                point_2 = np.dot(cell_vectors.T, point_fractional_2)
                tag_2 = gmsh.model.occ.add_point(*point_2, meshSize=size_points)
                point_tags.add(tag_2)

                gmsh.model.occ.synchronize()

                # Embed the points on opposing faces.
                gmsh.model.mesh.embed(0, [tag_1], 2, i * 2 + 1)
                gmsh.model.mesh.embed(0, [tag_2], 2, i * 2 + 2)

                embedded = True

                gmsh.model.occ.synchronize()

        if embedded:
            continue

        # Otherwise, embed the point in the cell.
        tag = gmsh.model.occ.add_point(*point)
        point_tags.add(tag)
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.embed(0, [tag], 3, cell)

    gmsh.model.occ.synchronize()

    return point_tags


def _configure_mesh_size_field(
    size: float,
    size_points: float,
    point_tags: set,
):
    """Configures the mesh size field in GMSH.

    Parameters
    ----------
    size : float
        The mesh size for the cell.
    size_points : float
        The mesh size for the points.
    point_tags : set
        The tags of the points to be used in the mesh size field.
    """

    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "PointsList", list(point_tags))
    gmsh.model.mesh.field.setNumbers(distance_field, "Sampling", [100])

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", size_points)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", size)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 1)  # 1 Å
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 3)  # 3 Å

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])

    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


def _apply_periodic_boundary_conditions(cell_vectors: NDArray, pbc: tuple):
    """Applies periodic boundary conditions to the mesh in GMSH.

    Parameters
    ----------
    cell_vectors : NDArray
        The cell vectors defining the parallelepiped cell, shape (3, 3).

    pbc : tuple
        The periodic boundary conditions in each direction.
        Should be a tuple of three boolean values indicating whether
        periodic boundary conditions are applied in the x, y, and z
        directions, respectively.
    """
    for i, (lattice_vector, periodic) in enumerate(zip(cell_vectors, pbc)):
        if not periodic:
            continue
        translation = np.eye(4)
        translation[:3, -1] = lattice_vector
        gmsh.model.mesh.set_periodic(2, [i * 2 + 2], [i * 2 + 1], translation.flatten())


def generate_mesh(
    cell_vectors: NDArray,
    pbc: tuple | None = None,
    size: float = 1.0,
    points: NDArray | None = None,
    size_points: float | None = None,
) -> meshio.Mesh:
    """Meshes a parallelepiped cell with optional points and periodic
    boundary conditions using GMSH.

    This function initializes GMSH, creates a parallelepiped cell
    defined by the provided cell vectors, and embeds points into the
    mesh. It applies periodic boundary conditions if specified. The mesh
    is generated and returned as a `meshio.Mesh` object.

    Parameters
    ----------
    cell_vectors : NDArray
        The cell vectors defining the parallelepiped cell, shape (3, 3).
    pbc : tuple, optional
        The periodic boundary conditions in each direction, by default
        None.
    size : float, optional
        The mesh size, by default 1.0.
    points : ArrayLike, optional
        The points to embed in the mesh, shape (n, 3), by default None.
    size_points : float, optional
        The mesh size for the points, by default None. If None, it will
        be set to `size / 10`.

    Returns
    -------
    meshio.Mesh
        The mesh.

    """
    if points is None:
        points = np.array([]).reshape(0, 3)

    if pbc is None:
        pbc = (False, False, False)

    if size_points is None:
        size_points = size / 10.0

    # Intialize GMSH.
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 4)
    gmsh.model.add(".tmp")

    cell = gmsh.model.occ.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, tag=1)
    transform = np.eye(4)
    transform[:3, :3] = cell_vectors.T
    gmsh.model.occ.affine_transform(
        [(3, cell)],
        transform.flatten(),
    )
    gmsh.model.occ.synchronize()

    point_tags = _embed_points(
        cell_vectors=cell_vectors,
        pbc=pbc,
        points=points,
        size_points=size_points,
        cell=cell,
    )

    _configure_mesh_size_field(
        size=size, size_points=size_points, point_tags=point_tags
    )

    _apply_periodic_boundary_conditions(cell_vectors, pbc)

    # Generate the mesh.
    gmsh.model.mesh.generate(dim=3)
    gmsh.model.mesh.optimize(method="", niter=3)

    # Transfer the mesh to meshio.
    filename = (
        "."
        + "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        + ".msh"
    )
    gmsh.write(filename)
    gmsh.finalize()

    mesh = meshio.read(filename)

    os.remove(filename)
    return mesh
