# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the methods to generate a mesh for the device geometry."""

from hashlib import sha256
from tempfile import NamedTemporaryFile

import ase.io

try:
    import gmsh

    gmsh_available = True

except ImportError:
    gmsh_available = False

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista as pv

from qttools import NDArray
from qttools.comm import comm
from quatrex import __version__
from quatrex.core.config import QuatrexConfig
from quatrex.electrostatics.geometry_config import (
    Box,
    Cylinder,
    Disk,
    Line,
    Point,
    Rectangle,
    Shape,
    ShapeDifference,
    ShapeUnion,
    Tube,
)

GMSH_GEOMETRY_TOLERANCE = 1e-5

# Regular OCC bounding box is not accurate enough for the conformal
# meshing we want in the simulation domain.
GMSH_OCC_USE_STL_BOUNDS = True

_dim_to_cell_type = {
    1: (pv.CellType.LINE, "line"),
    2: (pv.CellType.TRIANGLE, "triangle"),
    3: (pv.CellType.TETRA, "tetra"),
}


def shapes_with_volume(dim: int) -> list:
    """Returns the shape types that have a volume for the given
    dimension.

    Parameters
    ----------
    dim : int
        The dimension for which to get the shape types.

    Returns
    -------
    list[str]
        A list of shape types that have a volume for the given
        dimension.

    """
    if dim == 3:
        return (Box, Cylinder, ShapeUnion, ShapeDifference)
    if dim == 2:
        return (Rectangle, ShapeUnion, ShapeDifference)
    if dim == 1:
        return (Line, ShapeUnion, ShapeDifference)

    raise ValueError(f"Unsupported dimension {dim}")


def _add_box(box: Box) -> tuple[int, int]:
    """Adds a box to the GMSH model.

    Parameters
    ----------
    box : Box
        The shape definition for the box.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added box.

    """
    lower, upper = np.array(box.bounds)

    x, y, z = lower
    dx, dy, dz = upper - lower

    return (3, gmsh.model.occ.add_box(x, y, z, dx, dy, dz))


def _add_cylinder(cylinder: Cylinder) -> tuple[int, int]:
    """Adds a cylinder to the GMSH model

    Parameters
    ----------
    cylinder : Cylinder
        The shape definition for the cylinder.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added cylinder.

    """
    axis = np.array(cylinder.axis)
    axis /= np.linalg.norm(axis)
    v = axis * cylinder.length

    x, y, z = cylinder.center - v / 2
    dx, dy, dz = v

    return (3, gmsh.model.occ.add_cylinder(x, y, z, dx, dy, dz, cylinder.radius))


def _add_tube(tube: Tube) -> tuple[int, int]:
    """Adds a tube to the GMSH model.

    This consists of extruding a circle along the specified axis. The
    circle is created in the plane perpendicular to the axis, and then
    extruded along the axis to create the tube.

    Parameters
    ----------
    tube : Tube
        The shape definition for the tube.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added tube.

    """

    axis = np.array(tube.axis)
    axis /= np.linalg.norm(axis)

    v = axis * tube.length
    x, y, z = tube.center - v / 2

    circle_tag = gmsh.model.occ.add_circle(x, y, z, tube.radius, zAxis=axis)

    dx, dy, dz = v
    tube_dim_tags = gmsh.model.occ.extrude(
        [(1, circle_tag)], dx, dy, dz, recombine=False
    )

    # Get the tag of the extruded surface (the tube wall), i.e. the
    # entity with dimension 2.
    tube_dim_tag = next(dim_tag for dim_tag in tube_dim_tags if dim_tag[0] == 2)

    return tube_dim_tag


def _compute_rotation_matrix(
    normal: tuple[float, float, float],
    tangent: tuple[float, float, float] = (1, 0, 0),
) -> NDArray:
    """Computes the rotation matrix.

    This function computes the rotation matrix needed to align a shape
    with the specified normal and tangent.

    Parameters
    ----------
    normal : tuple[float, float, float]
        The normal vector to which the shape should be aligned.
    tangent : tuple[float, float, float], optional
        The tangent vector to which the shape should be aligned. The
        default is (1, 0, 0).

    Returns
    -------
    NDArray
        The rotation matrix as a 3x3 array.

    """
    normal = np.array(normal)
    normal /= np.linalg.norm(normal)
    tangent = np.array(tangent)
    tangent /= np.linalg.norm(tangent)

    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    return np.column_stack((tangent, bitangent, normal))


def _add_rectangle(rectangle: Rectangle) -> tuple[int, int]:
    """Adds a rectangle to the GMSH model.

    OCC only allows axis-aligned rectangles, so we create the rectangle
    in the XY plane and then apply an affine transformation to align it
    with the specified normal and tangent.

    Parameters
    ----------
    rectangle : Rectangle
        The shape definition for the rectangle.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added rectangle.

    """
    if rectangle.bounds is not None:
        (x_min, y_min), (x_max, y_max) = np.array(rectangle.bounds)
        rectangle_tag = gmsh.model.occ.add_rectangle(
            x_min, y_min, 0, x_max - x_min, y_max - y_min
        )
        return (2, rectangle_tag)

    rectangle_tag = gmsh.model.occ.add_rectangle(
        -rectangle.length / 2,
        -rectangle.width / 2,
        0,
        rectangle.length,
        rectangle.width,
    )

    transform = np.eye(4)
    transform[:3, :3] = _compute_rotation_matrix(rectangle.normal, rectangle.tangent)
    transform[:3, 3] = rectangle.center

    gmsh.model.occ.affine_transform([(2, rectangle_tag)], transform.flatten())

    return (2, rectangle_tag)


def _add_disk(disk: Disk) -> tuple[int, int]:
    """Adds a disk to the GMSH model.

    OCC only allows axis-aligned disks, so we create the disk in the XY
    plane and then apply an affine transformation to align it with the
    specified normal.

    Parameters
    ----------
    disk : Disk
        The shape definition for the disk.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added disk.

    """
    if disk.normal is None:
        # disk.center will only have 2 coordinates, so we need to add a
        # 0 for the z-coordinate.
        return (2, gmsh.model.occ.add_disk(*disk.center, 0, disk.radius, disk.radius))

    disk_tag = gmsh.model.occ.add_disk(*disk.center, disk.radius, disk.radius)

    transform = np.eye(4)
    transform[:3, :3] = _compute_rotation_matrix(disk.normal)
    transform[:3, 3] = disk.center

    gmsh.model.occ.affine_transform([(2, disk_tag)], transform.flatten())

    return (2, disk_tag)


def _add_line(line: Line) -> tuple[int, int]:
    """Adds a line to the GMSH model.

    Parameters
    ----------
    line : Line
        The shape definition for the line.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added line.

    """
    if isinstance(line.start, float) and isinstance(line.end, float):
        start = (line.start, 0, 0)
        end = (line.end, 0, 0)
    elif len(line.start) == 2 and len(line.end) == 2:
        start = (*line.start, 0)
        end = (*line.end, 0)
    else:
        start = line.start
        end = line.end

    start_tag = gmsh.model.occ.add_point(*start)
    end_tag = gmsh.model.occ.add_point(*end)
    return (1, gmsh.model.occ.add_line(start_tag, end_tag))


def _add_point(point: Point) -> tuple[int, int]:
    """Adds a point to the GMSH model.

    Parameters
    ----------
    point : Point
        The shape definition for the point.

    Returns
    -------
    dim_tag : tuple[int, int]
        A tuple containing the dimension and tag of the added point.

    """
    # Pad the position with zeros if it has less than 3 coordinates.
    if isinstance(point.position, float):
        position = (point.position, 0, 0)
    elif len(point.position) == 2:
        position = (*point.position, 0)
    else:
        position = point.position

    return (0, gmsh.model.occ.add_point(*position))


# Mapping from shape to function for adding the shape to the GMSH model.
_add_basic_entity_functions = {
    # 3D primitives:
    "box": _add_box,
    "cylinder": _add_cylinder,
    "tube": _add_tube,
    # 2D primitives:
    "rectangle": _add_rectangle,
    "disk": _add_disk,
    # 1D primitives:
    "line": _add_line,
    # 0D primitives:
    "point": _add_point,
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


def _add_difference(shape: ShapeDifference) -> tuple[int, int]:
    """Adds a difference of two shapes to the GMSH model.

    This function computes the difference of two shapes by adding them
    to the GMSH model and then using the `cut` operation to compute the
    difference.

    Parameters
    ----------
    shape : ShapeDifference
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
    out_dim_tags, __ = gmsh.model.occ.cut([base_dim_tag], [subtract_dim_tag])
    return out_dim_tags[0]


def _add_union(shape: ShapeUnion) -> tuple[int, int]:
    """Adds a union of shapes to the GMSH model.

    This function computes the union of multiple shapes by adding them
    to the GMSH model and then using the `fuse` operation to compute the
    union.

    Parameters
    ----------
    shape : ShapeUnion
        The shape definition for the union, which should have a `shapes`
        field containing a list of shapes to be used in the union
        operation.

    Returns
    -------
    tuple[int, int]
        A tuple containing the dimension and tag of the resulting shape.

    """

    shape_dim_tags = [_add_entity(shape) for shape in shape.shapes]
    out_dim_tags, __ = gmsh.model.occ.fuse(
        # Fuse the first shape with the union of the remaining shapes.
        shape_dim_tags[:1],
        shape_dim_tags[1:],
    )
    return out_dim_tags[0]


class DeviceMesh:
    """Class for generating a mesh for the device geometry defined in
    the configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration for the meshing, containing the geometry and
        meshing parameters.

    """

    def __init__(self, config: QuatrexConfig):
        """Initializes the DeviceMesh with the given configuration."""
        self.config = config

        structure = None
        if config.material_model == "atomistic":
            try:
                structure = ase.io.read(config.input_dir / "structure.xyz")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Structure file 'structure.xyz' not found in {config.input_dir}. "
                    "Please ensure the structure file is present for atomistic simulations."
                )
        self.structure = structure

        self.dimensionality = config.device.geometry.dimensionality

        self.default_mesh_size = config.device.geometry.default_mesh_size

        self.region_mesh_sizes = {
            region.name: region.mesh_size for region in config.device.geometry.regions
        }
        if self.structure is not None:
            self.region_mesh_sizes["atoms"] = config.device.geometry.atoms_mesh_size

        self._mesh = None
        self._region_node_inds = None
        self._region_element_inds = None

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

    @property
    def region_element_inds(self) -> dict[str, NDArray]:
        """Returns the mapping from elements to physical regions.

        Returns
        -------
        dict[str, NDArray]
            A dictionary mapping region names to arrays of element indices.

        """
        if self._region_element_inds is None:
            raise ValueError(
                "Region node indices have not been computed yet. Call generate_mesh() first."
            )
        return self._region_element_inds

    def _add_structure_cell(self) -> tuple[int, int]:
        """Adds the structure cell to the GMSH model.

        This function adds a box representing the structure cell to the
        GMSH model. The box is defined by the cell vectors of the
        structure, and an affine transformation is applied to align the
        box with the cell vectors. The resulting box is added to the
        GMSH model.

        Returns
        -------
        tuple[int, int]
            A tuple containing the dimension and tag of the added
            structure cell.

        """
        if self.structure is None:
            raise ValueError("Structure is not defined. Cannot add structure cell.")

        transform = np.eye(4)
        transform[:3, :3] = self.structure.cell.T

        structure_cell = gmsh.model.occ.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.affine_transform(
            [(3, structure_cell)],
            transform.flatten(),
        )

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

        return (3, bounding_box)

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

        Parameters
        ----------
        region_dim_tags : dict[str, list[tuple[int, int]]]
            A dictionary mapping region names to lists of tuples
            containing the dimension and tag of the entities in the GMSH
            model.

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

    def _map_nodes_to_physical_regions(self) -> dict[str, NDArray]:
        """Maps the nodes in the mesh to the physical regions defined in GMSH.

        This function creates a mapping from the nodes in the mesh to
        the physical regions defined in GMSH. It identifies which nodes
        belong to which physical groups based on the cell sets defined
        in the mesh.

        Returns
        -------
        dict[str, NDArray]
            A dictionary mapping region names to arrays of node indices.

        """
        region_node_inds = {}
        for region_name in self.region_mesh_sizes:
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

    def _map_elements_to_physical_regions(self) -> dict[str, NDArray]:
        """Maps elements in the mesh to the physical regions defined in GMSH.

        This function creates a mapping from the elements in the mesh to
        the physical regions defined in GMSH. It identifies which elements
        belong to which physical groups based on the cell sets defined
        in the mesh.

        Returns
        -------
        dict[str, NDArray]
            A dictionary mapping region names to arrays of element indices.

        """

        __, cell_type = _dim_to_cell_type[self.dimensionality]
        sets_dict = self.mesh.cell_sets_dict or {}

        region_element_inds = {}
        for region_name in self.region_mesh_sizes:
            idx = sets_dict.get(region_name, {}).get(cell_type)
            if idx is None or len(idx) == 0:
                region_element_inds[region_name] = np.array([], dtype=int)
            else:
                region_element_inds[region_name] = np.unique(np.asarray(idx, dtype=int))

        return region_element_inds

    def generate_mesh(self) -> meshio.Mesh:
        """Generates the mesh for the device geometry.

        This function generates a mesh for the device geometry defined
        in the configuration. It uses GMSH to create the mesh and
        returns it as a `meshio.Mesh` object.

        Returns
        -------
        meshio.Mesh
            The generated mesh.

        """
        if not gmsh_available:
            raise ImportError(
                "GMSH is not available. Make sure GMSH is installed and "
                "the Python API is accessible."
            )

        print("Generating mesh with GMSH...")

        # Initialize gmsh and create a temporary model.
        gmsh.initialize()
        gmsh.option.set_number("General.Terminal", 0)
        gmsh.option.set_number("General.Verbosity", 5)
        gmsh.option.set_number("Geometry.Tolerance", GMSH_GEOMETRY_TOLERANCE)
        gmsh.option.set_number("Geometry.OCCBoundsUseStl", GMSH_OCC_USE_STL_BOUNDS)

        gmsh.logger.start()
        gmsh.model.add(".tmp")

        # Add device geometry entities.
        print("    Adding device geometry regions...")
        region_dim_tags = {}
        for region in self.config.device.geometry.regions:
            dim_tag = _add_entity(region.shape)
            region_dim_tags[region.name] = [dim_tag]

        if self.structure is not None:
            print("    Adding structure cell...")
            region_dim_tags["cell"] = [self._add_structure_cell()]

            print("    Adding atoms...")
            atom_dim_tags = []
            for position in self.structure.positions:
                tag = gmsh.model.occ.add_point(*position)
                atom_dim_tags.append((0, tag))

            region_dim_tags["atoms"] = atom_dim_tags

        gmsh.model.occ.synchronize()

        print("    Adding bounding box...")
        bounding_box_dim_tags = [self._add_bounding_box()]

        # Only fragment if there is any geometry defined, otherwise we
        # just have the bounding box.
        if region_dim_tags:
            num_dim_tags = [len(dim_tags) for dim_tags in region_dim_tags.values()]
            dim_tag_offsets = [
                sum(num_dim_tags[:i]) for i in range(len(num_dim_tags) + 1)
            ]

            __, out_map = gmsh.model.occ.fragment(
                bounding_box_dim_tags,
                list(itertools.chain.from_iterable(region_dim_tags.values())),
            )
            bounding_box_dim_tags = out_map.pop(0)
            for region, (start, stop) in zip(
                region_dim_tags, itertools.pairwise(dim_tag_offsets)
            ):
                region_dim_tags[region] = list(
                    itertools.chain.from_iterable(out_map[start:stop])
                )

            used_dim_tags = {
                dim_tag for out in region_dim_tags.values() for dim_tag in out
            }
            bounding_box_dim_tags = [
                dim_tag
                for dim_tag in bounding_box_dim_tags
                if dim_tag not in used_dim_tags
            ]

        if bounding_box_dim_tags:
            region_dim_tags["bounding-box"] = bounding_box_dim_tags

        gmsh.model.occ.synchronize()

        if self.structure is not None and any(self.structure.pbc):
            self._enforce_mesh_periodicity(region_dim_tags["cell"])

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

        gmsh.model.mesh.optimize(method="", niter=10)

        gmsh_logs = gmsh.logger.get()
        gmsh.logger.stop()

        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.input_dir / "gmsh.log", "w") as f:
            f.write("\n".join(gmsh_logs))

        # Transfer the mesh to meshio.
        with NamedTemporaryFile(suffix=".msh") as file:
            gmsh.write(file.name)
            gmsh.finalize()
            mesh = meshio.read(file.name)
        print("Done!")

        # Save the mesh to the input directory.
        print(f"Saving mesh to {self.config.input_dir / 'device.msh'}...")
        mesh.write(
            self.config.input_dir / "device.msh",
            file_format="gmsh",
            binary=False,
        )

        # Add some provenance information to the mesh file as comments.
        with open(self.config.input_dir / "device.msh", "a") as f:
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

        # Store the mesh and the mapping from nodes/elements to physical regions
        # for later use.
        self._mesh = mesh
        self._region_node_inds = self._map_nodes_to_physical_regions()
        self._region_element_inds = self._map_elements_to_physical_regions()

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

        cell_type, cells_dict_key = _dim_to_cell_type[self.dimensionality]

        ugrid = pv.UnstructuredGrid(
            {cell_type: self.mesh.cells_dict[cells_dict_key]}, self.mesh.points
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
                opacity=0.5,
                show_edges=True,
                label=name,
                name=name,
            )

        pl.add_legend(loc="lower right")
        pl.show_grid()
        pl.add_axes()
        pl.enable_parallel_projection()

        if off_screen:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            pl.screenshot(self.config.output_dir / "device_mesh.png")
            return

        # Add a small widget to switch visibility of the different
        # regions.
        def toggle_visibility(region_name: str):
            actor = pl.renderer.actors[region_name]
            actor.SetVisibility(not actor.GetVisibility())
            pl.render()

        for name in region_names:
            pl.add_checkbox_button_widget(
                lambda state, name=name: toggle_visibility(name),
                value=True,
                position=(10, 30 * len(region_names) - 30 * region_names.index(name)),
                size=20,
                color_on=region_colors[name],
                color_off="gray",
            )

        pl.show()

    @classmethod
    def from_config(cls, config: QuatrexConfig) -> "DeviceMesh":
        """Creates a DeviceMesh instance from the configuration file.

        This function reads the mesh from the input directory and
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
        with open(config.input_dir / "device.msh", "r") as f:
            lines = f.readlines()
            comments_start = lines.index("$Comments\n")
            comments_end = lines.index("$EndComments\n")
            comments = lines[comments_start + 1 : comments_end]
            config_hash = next(
                line.strip() for line in comments if len(line.strip()) == 64
            )
            if config_hash != sha256(str(config.device.geometry).encode()).hexdigest():
                # Warn the user that the mesh file is not up to date
                # with the configuration.
                if comm.rank == 0:
                    print(
                        "Warning: The mesh file is not up to date with the configuration. "
                        "Please regenerate the mesh by running `quatrex mesh`."
                        "The existing mesh file will be used, but it may not reflect the current configuration.",
                        flush=True,
                    )

        device_mesher = cls(config)
        device_mesher._mesh = meshio.read(config.input_dir / "device.msh")
        device_mesher._region_node_inds = device_mesher._map_nodes_to_physical_regions()
        device_mesher._region_element_inds = (
            device_mesher._map_elements_to_physical_regions()
        )
        return device_mesher
