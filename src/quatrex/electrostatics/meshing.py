# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
from hashlib import sha256
from tempfile import NamedTemporaryFile

import ase.io
import gmsh
import matplotlib as mpl
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista as pv

from qttools import NDArray
from quatrex import __version__
from quatrex.core.config import QuatrexConfig
from quatrex.electrostatics.geometry_config import (
    Box,
    Cylinder,
    Shape,
    ShapeDifference,
    ShapeUnion,
    VolumeProperties,
)

GMSH_GEOMETRY_TOLERANCE = 1e-5


def _add_box(shape: Shape) -> tuple[int, int]:
    """Adds a box to the GMSH model.

    Parameters
    ----------
    shape : Shape
        The shape definition for the box.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the added box.

    """
    lower, upper = np.array(shape.bounds)

    x, y, z = lower
    dx, dy, dz = upper - lower

    return (3, gmsh.model.occ.add_box(x, y, z, dx, dy, dz))


def _add_cylinder(shape: Shape) -> tuple[int, int]:
    """Adds a cylinder to the GMSH model

    Parameters
    ----------
    shape : Shape
        The shape definition for the cylinder.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the added cylinder.

    """
    axis = np.array(shape.axis)
    axis /= np.linalg.norm(axis)
    v = axis * shape.length

    x, y, z = shape.center - v / 2
    dx, dy, dz = v

    return (3, gmsh.model.occ.add_cylinder(x, y, z, dx, dy, dz, shape.radius))


def _compute_rotation_matrix(shape: Shape) -> NDArray:
    """Computes the rotation matrix.

    Thsi function computes the rotation matrix needed to align a
    rectangle with the specified normal and tangent.

    Parameters
    ----------
    shape : Shape
        The shape definition containing the normal and tangent vectors.

    Returns
    -------
    NDArray
        The rotation matrix as a 3x3 array.

    """
    normal = np.array(shape.normal)
    normal /= np.linalg.norm(normal)
    tangent = np.array(shape.tangent)
    tangent /= np.linalg.norm(tangent)

    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    return np.column_stack((tangent, bitangent, normal))


def _add_rectangle(shape: Shape) -> tuple[int, int]:
    """Adds a rectangle to the GMSH model.

    OCC only allows axis-aligned rectangles, so we create the rectangle
    in the XY plane and then apply an affine transformation to align it
    with the specified normal and tangent.

    Parameters
    ----------
    shape : Shape
        The shape definition for the rectangle.
    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the added rectangle.

    """

    rectangle_tag = gmsh.model.occ.add_rectangle(
        -shape.length / 2, -shape.width / 2, 0, shape.length, shape.width
    )
    transform = np.eye(4)
    transform[:3, :3] = _compute_rotation_matrix(shape)
    transform[:3, 3] = shape.center

    gmsh.model.occ.affine_transform([(2, rectangle_tag)], transform.flatten())

    return (2, rectangle_tag)


def _add_tube(shape: Shape) -> tuple[int, int]:
    """Adds a tube to the GMSH model.

    This consists of extruding a circle along the specified axis. The
    circle is created in the plane perpendicular to the axis, and then
    extruded along the axis to create the tube.

    Parameters
    ----------
    shape : Shape
        The shape definition for the tube.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the added tube.

    """

    axis = np.array(shape.axis)
    axis /= np.linalg.norm(axis)

    v = axis * shape.length
    x, y, z = shape.center - v / 2

    circle_tag = gmsh.model.occ.add_circle(x, y, z, shape.radius, zAxis=axis)

    dx, dy, dz = v
    tube_dim_tags = gmsh.model.occ.extrude(
        [(1, circle_tag)], dx, dy, dz, recombine=False
    )

    # Get the tag of the extruded surface (the tube wall), i.e. the
    # entity with dimension 2.
    tube_dim_tag = next(dim_tag for dim_tag in tube_dim_tags if dim_tag[0] == 2)

    return tube_dim_tag


# Mapping from shape to function for adding the shape to the GMSH model.
_add_basic_entity_functions = {
    "box": _add_box,
    "cylinder": _add_cylinder,
    "rectangle": _add_rectangle,
    "tube": _add_tube,
}


def _add_entity(shape: Shape) -> tuple[int, int]:
    """Adds a shape entity to the GMSH model.

    Parameters
    ----------
    shape : Shape
        The shape definition for the entity.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the added entity.

    """
    if shape.type in _add_basic_entity_functions:
        return _add_basic_entity_functions[shape.type](shape)
    if shape.type == "difference":
        return _add_difference(shape)
    if shape.type == "union":
        return _add_union(shape)

    raise ValueError(f"Unsupported shape type {shape.type}")


def _add_difference(shape: Shape) -> tuple[int, int]:
    """Adds a difference of two shapes to the GMSH model.

    This function computes the difference of two shapes by adding them
    to the GMSH model and then using the `cut` operation to compute the
    difference.

    Parameters
    ----------
    shape : Shape
        The shape definition for the difference, which should have a
        `base` and `subtract` field containing the shapes to be used in
        the difference operation.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the resulting shape.

    """
    base_dim_tag = _add_entity(shape.base)
    subtract_dim_tag = _add_entity(shape.subtract)
    out_dim_tags, out_map = gmsh.model.occ.cut([base_dim_tag], [subtract_dim_tag])
    return out_dim_tags[0]


def _add_union(shape: Shape) -> tuple[int, int]:
    """Adds a union of shapes to the GMSH model.

    This function computes the union of multiple shapes by adding them
    to the GMSH model and then using the `fuse` operation to compute the
    union.

    Parameters
    ----------
    shape : Shape
        The shape definition for the union, which should have a `shapes`
        field containing a list of shapes to be used in the union
        operation.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the resulting shape.

    """

    shape_dim_tags = [_add_entity(shape) for shape in shape.shapes]
    out_dim_tags, out_map = gmsh.model.occ.fuse(
        # Fuse the first shape with the union of the remaining shapes.
        shape_dim_tags[:1],
        shape_dim_tags[1:],
    )
    return out_dim_tags[0]


def _inside_box(point: NDArray, box: Box) -> bool:
    """Checks if a point is inside a box defined by its bounds.

    Parameters
    ----------
    point : array-like
        The coordinates of the point to check.
    box : Box
        The box defined by its bounds, which should be a tuple of two
        points representing the lower and upper corners of the box.

    Returns
    -------
    bool
        True if the point is inside the box, False otherwise.

    """
    point = np.asarray(point)
    (x_min, y_min, z_min), (x_max, y_max, z_max) = box.bounds
    inside = (
        (x_min <= point[..., 0])
        & (point[..., 0] <= x_max)
        & (y_min <= point[..., 1])
        & (point[..., 1] <= y_max)
        & (z_min <= point[..., 2])
        & (point[..., 2] <= z_max)
    )

    return inside


def _inside_cylinder(point: NDArray, cylinder: Cylinder) -> bool:
    """Checks if a point is inside a cylinder defined by its center, axis, length, and radius.

    Parameters
    ----------
    point : array-like
        The coordinates of the point to check.
    cylinder : Cylinder
        The cylinder defined by its center, axis, length, and radius.

    Returns
    -------
    bool
        True if the point is inside the cylinder, False otherwise.

    """
    point = np.asarray(point)
    single_point = point.ndim == 1
    if single_point:
        point = point[None, :]

    center = np.array(cylinder.center)
    axis = np.array(cylinder.axis)
    axis /= np.linalg.norm(axis)
    length = cylinder.length
    radius = cylinder.radius

    # Compute the vector from the center of the cylinder to the point.
    v = point - center

    # Project the vector onto the axis of the cylinder to get the
    # component along the axis.
    v_parallel = np.dot(v, axis)[:, None] * axis

    # The component perpendicular to the axis is then given by:
    v_perpendicular = v - v_parallel

    # Check if the point is within the length and radius of the
    # cylinder.
    inside = (np.linalg.norm(v_parallel, axis=1) <= length / 2) & (
        np.linalg.norm(v_perpendicular, axis=1) <= radius
    )
    return inside[0] if single_point else inside


_inside_entity_functions = {
    "box": _inside_box,
    "cylinder": _inside_cylinder,
}


def _inside_union(point: NDArray, shape_union: ShapeUnion) -> bool:
    """Checks if a point is inside a union of shapes.

    Parameters
    ----------
    point : array-like
        The coordinates of the point to check.
    shape_union : ShapeUnion
        The union of shapes defined by a list of shapes.

    Returns
    -------
    bool
        True if the point is inside any of the shapes in the union, False otherwise.

    """
    inside = np.zeros(point.shape[0], dtype=bool)
    for shape in shape_union.shapes:
        inside_shape = _inside_entity_functions[shape.type](point, shape)
        inside |= inside_shape

    return inside


def _inside_difference(point: NDArray, shape_difference: ShapeDifference) -> bool:
    """Checks if a point is inside a difference of two shapes.

    Parameters
    ----------
    point : array-like
        The coordinates of the point to check.
    shape_difference : ShapeDifference
        The difference of two shapes defined by a `base` shape and a
        `subtract` shape.

    Returns
    -------
    bool
        True if the point is inside the `base` shape and not inside the
        `subtract` shape, False otherwise.

    """
    inside_base = _inside_entity_functions[shape_difference.base.type](
        point, shape_difference.base
    )
    inside_subtract = _inside_entity_functions[shape_difference.subtract.type](
        point, shape_difference.subtract
    )
    inside = inside_base & ~inside_subtract
    return inside


def inside_shape(point: NDArray, shape: Shape) -> bool:
    """Checks if a point is inside a shape.

    Parameters
    ----------
    point : array-like
        The coordinates of the point to check.
    shape : Shape
        The shape definition, which can be a basic shape, a union of
        shapes, or a difference of shapes.

    Returns
    -------
    bool
        True if the point is inside the shape, False otherwise.

    """
    if shape.type in _inside_entity_functions:
        return _inside_entity_functions[shape.type](point, shape)
    if shape.type == "union":
        return _inside_union(point, shape)
    if shape.type == "difference":
        return _inside_difference(point, shape)

    raise ValueError(f"Unsupported shape type {shape.type}")


class DeviceMesh:
    """Class for generating a mesh for the device geometry defined in the configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration for the meshing, containing the geometry and
        meshing parameters.

    """

    def __init__(self, config: QuatrexConfig):
        """Initializes the DeviceMesh with the given configuration."""
        self.config = config
        self.structure = ase.io.read(config.input_dir / "structure.xyz")

        # TODO: Set up size fields based on the configuration.
        self.default_mesh_size = config.device.geometry.default_mesh_size

        self.region_mesh_sizes = {
            region.name: region.mesh_size for region in config.device.geometry.regions
        }
        self.region_mesh_sizes["atoms"] = config.device.geometry.atoms_mesh_size

        self._mesh = None
        self._region_node_inds = None

    @property
    def mesh(self) -> meshio.Mesh:
        """Returns the generated mesh.

        Returns
        -------
        meshio.Mesh
            The generated mesh.

        """
        if self._mesh is None:
            raise ValueError(
                "Mesh has not been generated yet. Call generate_mesh() first."
            )
        return self._mesh

    @property
    def region_node_inds(self) -> dict[str, NDArray]:
        """Returns the mapping from nodes to physical regions.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping region names to arrays of node indices.

        """
        if self._region_node_inds is None:
            raise ValueError(
                "Region node indices have not been computed yet. Call generate_mesh() first."
            )
        return self._region_node_inds

    def _add_structure_cell(self):
        """Adds the structure cell to the GMSH model.

        This function adds a box representing the structure cell to the
        GMSH model. The box is defined by the cell vectors of the
        structure, and an affine transformation is applied to align the
        box with the cell vectors. The resulting box is added to the
        GMSH model and synchronized.

        """
        transform = np.eye(4)
        transform[:3, :3] = self.structure.cell.T

        structure_cell = gmsh.model.occ.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.affine_transform(
            [(3, structure_cell)],
            transform.flatten(),
        )
        gmsh.model.occ.synchronize()

        return (3, structure_cell)

    def _add_bounding_box(self) -> tuple[int, int]:
        """Adds a bounding box to the GMSH model.

        This determines the bounding box of all entities in the GMSH
        model and adds a bounding box.

        Returns
        -------
        tuple[int, int]
            A tuple containing the dimension and tag of the added
            bounding box.

        """
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.get_bounding_box(-1, -1)
        bounding_box = gmsh.model.occ.add_box(
            x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min
        )
        gmsh.model.occ.synchronize()

        return (3, bounding_box)

    def _embed_atoms(self, dim_tags: tuple[int, int]) -> list[tuple[int, int]]:
        """Embeds the atoms into the given gmsh entities.

        Parameters
        ----------
        dim_tags : tuple[int, int]
            A list of tuples containing the dimension and tag of the
            entities in which to embed the atoms.

        Returns
        -------
        list[tuple[int, int]]
            A list of tuples containing the dimension and tag of the
            embedded atoms.

        """

        atom_tags = []
        for position in self.structure.positions:
            tag = gmsh.model.occ.add_point(*position)
            atom_tags.append(tag)

        gmsh.model.occ.synchronize()

        for dim, cell in dim_tags:
            gmsh.model.mesh.embed(0, atom_tags, 3, cell)

        return [(0, tag) for tag in atom_tags]

    def _enforce_mesh_periodicity(self, dim_tags: tuple[int, int]) -> None:
        """Enforces periodic boundary conditions on the mesh.

        This function identifies pairs of surfaces corresponding to
        periodic boundaries and applies the appropriate periodicity
        constraints in GMSH.

        Parameters
        ----------
        dim_tags : tuple[int, int]
            A list of tuples containing the dimension and tag of the
            entities in the GMSH model, used to identify the surfaces
            for applying periodicity.

        """

        surface_dim_tags = gmsh.model.get_boundary(dim_tags)

        surface_pairs = [[], [], []]
        for dim, tag in surface_dim_tags:
            x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.occ.get_bounding_box(
                dim, tag
            )
            # Determine the plane of the surface by checking which
            # coordinate is constant in the bounding box.
            if np.isclose(x_min, x_max, atol=GMSH_GEOMETRY_TOLERANCE):
                surface_pairs[0].append([tag])
            elif np.isclose(y_min, y_max, atol=GMSH_GEOMETRY_TOLERANCE):
                surface_pairs[1].append([tag])
            elif np.isclose(z_min, z_max, atol=GMSH_GEOMETRY_TOLERANCE):
                surface_pairs[2].append([tag])
            else:
                raise ValueError(f"Surface {tag} of bounding box is not axis-aligned")

        for i, (lattice_vector, periodic, surface_pair) in enumerate(
            zip(self.structure.cell, self.structure.pbc, surface_pairs)
        ):
            if not periodic:
                continue

            if len(surface_pair) != 2:
                raise ValueError(
                    "Periodic boundary surfaces must remain unfragmented. "
                    f"Expected 2 surfaces along axis {i}, found {len(surface_pair)}. "
                    "Check geometry/Dirichlet definitions to ensure they do not "
                    "intersect periodic boundary surfaces."
                )

            print(f"    Enforcing periodicity along lattice vector {lattice_vector}...")

            # Check which sits at a smaller coordinate value to
            # determine the direction of the translation.
            bounding_box_0 = gmsh.model.occ.get_bounding_box(2, surface_pair[0][0])
            bounding_box_1 = gmsh.model.occ.get_bounding_box(2, surface_pair[1][0])
            if bounding_box_0[i] > bounding_box_1[i]:
                surface_pair = [surface_pair[0], surface_pair[1]]
            else:
                surface_pair = [surface_pair[1], surface_pair[0]]

            translation = np.eye(4)
            translation[:3, -1] = lattice_vector
            gmsh.model.mesh.set_periodic(2, *surface_pair, translation.flatten())

    def _configure_mesh_size_fields(
        self, region_dim_tags: dict[str, list[tuple[int, int]]]
    ) -> None:
        """Configures the mesh size fields in GMSH.

        This function sets up the mesh size fields in GMSH based on the
        configuration parameters. It creates a distance field from the
        embedded atoms and a threshold field to control the mesh size
        based on the distance from the atoms.

        """
        region_fields = []
        for name, dim_tags in region_dim_tags.items():

            if name == "atoms":
                # The atoms are not just a constant size field, but we
                # want to have a finer mesh close to the atoms. We use a
                # distance field from the atoms and a threshold field to
                # control the mesh size based on the distance from the
                # atoms.
                atoms_mesh_size = (
                    self.region_mesh_sizes.get("atoms") or self.default_mesh_size
                )

                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.set_numbers(
                    distance_field, "PointsList", [tag for __, tag in dim_tags]
                )
                gmsh.model.mesh.field.set_number(distance_field, "Sampling", 100)

                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.set_number(
                    threshold_field, "InField", distance_field
                )
                gmsh.model.mesh.field.set_number(
                    threshold_field, "SizeMin", atoms_mesh_size
                )
                gmsh.model.mesh.field.set_number(
                    threshold_field, "SizeMax", self.default_mesh_size
                )
                gmsh.model.mesh.field.set_number(threshold_field, "DistMin", 1.0)
                gmsh.model.mesh.field.set_number(threshold_field, "DistMax", 3.0)
                region_fields.append(threshold_field)

                continue

            mesh_size = self.region_mesh_sizes.get(name) or self.default_mesh_size

            const_field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.set_string(const_field, "F", str(mesh_size))
            restrict_field = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.set_number(restrict_field, "InField", const_field)

            # Check the dimension of the tags to set the correct parameter name.
            dim, __ = dim_tags[0]
            if dim == 3:
                gmsh.model.mesh.field.set_numbers(
                    restrict_field, "VolumesList", [tag for __, tag in dim_tags]
                )
            elif dim == 2:
                gmsh.model.mesh.field.set_numbers(
                    restrict_field, "SurfacesList", [tag for __, tag in dim_tags]
                )
            else:
                gmsh.model.mesh.field.set_numbers(
                    restrict_field, "PointsList", [tag for __, tag in dim_tags]
                )

            region_fields.append(restrict_field)

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.set_numbers(min_field, "FieldsList", region_fields)
        gmsh.model.mesh.field.set_as_background_mesh(min_field)

    def _map_nodes_to_physical_regions(self):
        """Maps the nodes in the mesh to the physical regions defined in GMSH.

        This function creates a mapping from the nodes in the mesh to the
        physical regions defined in GMSH. It identifies which nodes belong
        to which physical groups based on the cell sets defined in the
        mesh.

        """
        region_node_inds = {}
        for region_name in self.region_mesh_sizes.keys():

            node_inds = []
            for cell_block, cell_inds in zip(
                self.mesh.cells, self.mesh.cell_sets[region_name]
            ):
                if not cell_inds.size:
                    continue

                node_inds.append(np.unique(cell_block.data[cell_inds]))

            if not node_inds:
                region_node_inds[region_name] = np.array([], dtype=int)
                continue

            region_node_inds[region_name] = np.unique(np.concatenate(node_inds))

        return region_node_inds

    def generate_mesh(self) -> meshio.Mesh:
        """Generates the mesh for the device geometry.

        This function generates a mesh for the device geometry defined in
        the configuration. It uses GMSH to create the mesh and returns it
        as a `meshio.Mesh` object.

        Returns
        -------
        meshio.Mesh
            The generated mesh.

        """
        print("Generating mesh with GMSH...")

        # Initialize gmsh and create a temporary model.
        gmsh.initialize()
        gmsh.option.set_number("General.Terminal", 0)
        gmsh.option.set_number("General.Verbosity", 5)
        gmsh.option.set_number("Geometry.Tolerance", GMSH_GEOMETRY_TOLERANCE)

        gmsh.logger.start()
        gmsh.model.add(".tmp")

        # Add device geometry entities.
        print("    Adding device geometry regions...")
        region_dim_tags = {}
        for region in self.config.device.geometry.regions:
            if isinstance(region.properties, VolumeProperties):
                # We do not add volumes as physical groups directly.
                # Instead, we will find the nodes in the volumes after
                # meshing and assign them to physical groups based on
                # that. This is because the volumes are not meshed as
                # separate entities,
                continue

            dim_tag = _add_entity(region.shape)
            region_dim_tags[region.name] = dim_tag

        print("    Adding structure cell...")
        cell_dim_tags = [self._add_structure_cell()]

        print("    Adding bounding box...")
        bounding_box_dim_tags = [self._add_bounding_box()]

        # Remove the cell again, since we only needed it to define the
        # bounding, but we do not want it to be part of the final
        # geometry.
        gmsh.model.occ.remove(cell_dim_tags, recursive=True)
        gmsh.model.occ.synchronize()

        # Only fragment if there is any geometry defined, otherwise we
        # just have the bounding box.
        if region_dim_tags:
            # Fragment and remap the entities to ensure a consistent mesh.
            out_dim_tags, out_map = gmsh.model.occ.fragment(
                bounding_box_dim_tags,
                list(region_dim_tags.values()),
            )
            region_dim_tags = dict(zip(region_dim_tags.keys(), out_map[1:]))
            bounding_box_dim_tags = out_map[0]

        region_dim_tags["bounding-box"] = bounding_box_dim_tags

        print("    Embedding atoms...")
        atom_dim_tags = self._embed_atoms(region_dim_tags["bounding-box"])
        region_dim_tags["atoms"] = atom_dim_tags

        self._enforce_mesh_periodicity(region_dim_tags["bounding-box"])

        self._configure_mesh_size_fields(region_dim_tags)

        # Add physical groups for the regions.
        for name, dim_tags in region_dim_tags.items():
            dim, __ = dim_tags[0]
            gmsh.model.add_physical_group(dim, [tag for __, tag in dim_tags], name=name)

        # Generate the mesh.
        print("    Meshing...")
        gmsh.model.mesh.generate(dim=3)
        gmsh.model.mesh.remove_duplicate_nodes()
        gmsh.model.mesh.remove_duplicate_elements()

        # Set up physical groups for the volume regions based on which
        # nodes are inside the corresponding shapes. We do this after
        # meshing, since the volume regions may not be meshed as
        # separate entities, but we can still identify which nodes
        # belong to which regions based on their coordinates.
        node_tags, node_coords, __ = gmsh.model.mesh.get_nodes()
        node_coords = node_coords.reshape(-1, 3)
        for region in self.config.device.geometry.regions:
            if not isinstance(region.properties, VolumeProperties):
                continue

            inside_mask = inside_shape(node_coords, region.shape)
            region_node_tags = node_tags[inside_mask]

            discrete_entity_tag = gmsh.model.add_discrete_entity(0)
            gmsh.model.mesh.add_nodes(
                0, discrete_entity_tag, [], node_coords[inside_mask].flatten()
            )
            max_tag = gmsh.model.mesh.get_max_node_tag()
            element_tags = np.arange(len(region_node_tags)) + max_tag + 1_000_000
            gmsh.model.mesh.add_elements(
                0, discrete_entity_tag, [15], [element_tags], [region_node_tags]
            )
            gmsh.model.add_physical_group(0, [discrete_entity_tag], name=region.name)

        gmsh.model.mesh.optimize(method="", niter=10)

        gmsh_logs = gmsh.logger.get()
        gmsh.logger.stop()

        if not os.path.exists(self.config.output_dir):
            os.mkdir(self.config.output_dir)

        with open(self.config.output_dir / "gmsh.log", "w") as f:
            f.write("\n".join(gmsh_logs))

        # Transfer the mesh to meshio.
        with NamedTemporaryFile(suffix=".msh") as file:
            gmsh.write(file.name)
            gmsh.finalize()
            mesh = meshio.read(file.name)
        print("Done!")

        # Save the mesh to the output directory.
        print(f"Saving mesh to {self.config.output_dir / 'device.msh'}...")
        mesh.write(
            self.config.output_dir / "device.msh",
            file_format="gmsh",
            binary=False,
        )

        # Add some provenance information to the mesh file as comments.
        with open(self.config.output_dir / "device.msh", "a") as f:
            f.write(
                "\n".join(
                    [
                        "$Comments",
                        f"Generated with quatrex v{__version__}.",
                        f"{sha256(str(self.config.device.geometry).encode()).hexdigest()}",
                        "$EndComments",
                        "",
                    ]
                )
            )

        # Store the mesh and the mapping from nodes to physical regions
        # for later use.
        self._mesh = mesh
        self._region_node_inds = self._map_nodes_to_physical_regions()

        return mesh

    def visualize(self, off_screen: bool = False) -> None:
        """Plots the generated mesh.

        This function visualizes the generated mesh using PyVista. It colors
        the mesh based on the physical groups defined for the regions in the
        GMSH model.

        Parameters
        ----------
        off_screen : bool, optional
            Whether to use off-screen rendering, by default False.

        """
        region_names = list(self.region_mesh_sizes.keys())

        # Create stable color mapping.
        cmap = plt.get_cmap("tab10")
        region_colors = {
            name: mpl.colors.to_hex(cmap(i % cmap.N))
            for i, name in enumerate(region_names)
        }

        ugrid = pv.UnstructuredGrid(
            {pv.CellType.TETRA: self.mesh.cells_dict["tetra"]}, self.mesh.points
        )

        pl = pv.Plotter(off_screen=off_screen)

        pl.add_mesh(
            ugrid.extract_all_edges(),
            show_edges=True,
            color="black",
            opacity=0.1,
        )

        for name, inds in self.region_node_inds.items():
            points = self.mesh.points[inds]
            pl.add_points(
                points,
                color=region_colors[name],
                opacity=0.35,
                show_edges=True,
                label=name,
            )

        pl.add_legend(loc="lower right")
        pl.show_grid()
        pl.add_axes()
        pl.enable_parallel_projection()

        if off_screen:
            pl.screenshot("device_mesh.png")
            return

        pl.show()

    @classmethod
    def from_config(cls, config: QuatrexConfig) -> "DeviceMesh":
        """Creates a DeviceMesh instance from the configuration file.

        This function reads the mesh from the output directory and
        creates a DeviceMesh instance with the loaded mesh.

        Parameters
        ----------
        config : QuatrexConfig
            The configuration for the meshing, containing the geometry
            and meshing parameters.

        Returns
        -------
        DeviceMesh
            An instance of DeviceMesh with the loaded mesh.

        """
        # Check if the mesh file is up to date with the configuration by
        # comparing the hash of the configuration with the hash stored
        # in the mesh file comments.
        with open(config.output_dir / "device.msh", "r") as f:
            lines = f.readlines()
            comments_start = lines.index("$Comments\n")
            comments_end = lines.index("$EndComments\n")
            comments = lines[comments_start + 1 : comments_end]
            config_hash = next(
                line.strip() for line in comments if len(line.strip()) == 64
            )
            if config_hash != sha256(str(config.device.geometry).encode()).hexdigest():
                # Warn the user that the mesh file is not up to date with the configuration.
                print(
                    "Warning: The mesh file is not up to date with the configuration. "
                    "Please regenerate the mesh by running `quatrex mesh`."
                    "The existing mesh file will be used, but it may not reflect the current configuration."
                )

        device_mesher = cls(config)
        device_mesher._mesh = meshio.read(config.output_dir / "device.msh")
        device_mesher._region_node_inds = device_mesher._map_nodes_to_physical_regions()
        return device_mesher
