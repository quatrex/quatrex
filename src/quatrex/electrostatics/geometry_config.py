# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the geometry configuration classes."""

from typing import Annotated, Any, Literal, Self, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

from qttools import NDArray


class Box(BaseModel):
    """An axis-aligned box defined by its minimum and maximum corners."""

    type: Literal["box"] = Field("box", frozen=True)

    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    """The bounds of the box defined by its minimum and maximum corners."""

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Validates that the bounds are correctly defined."""
        min_corner, max_corner = self.bounds
        if any(min_c > max_c for min_c, max_c in zip(min_corner, max_corner)):
            raise ValueError(
                "Minimum corner must be less than or equal to maximum corner."
            )
        return self


class Cylinder(BaseModel):
    """A solid cylinder.

    Defined by a center point, an axis vector, a radius, and a length.

    """

    type: Literal["cylinder"] = Field("cylinder", frozen=True)

    center: tuple[float, float, float]
    """The center point of the cylinder."""
    radius: PositiveFloat
    """The radius of the cylinder."""
    axis: tuple[float, float, float]
    """The axis along which the cylinder extends."""
    length: PositiveFloat
    """The length of the cylinder along the given axis."""


class Tube(BaseModel):
    """A cylindrical surface.

    Defined by a center point, an axis vector, a radius, and a length.

    """

    type: Literal["tube"] = Field("tube", frozen=True)

    center: tuple[float, float, float]
    """The center point of the tube."""
    axis: tuple[float, float, float]
    """The axis along which the tube extends."""
    radius: PositiveFloat
    """The radius of the tube."""
    length: PositiveFloat
    """The length of the tube along the given axis."""


class Rectangle(BaseModel):
    """A rectangular surface.

    Needs a normal for orientation and a tangent vector to define the
    direction of the length and width. The rectangle is centered at the
    given center point.

    """

    type: Literal["rectangle"] = Field("rectangle", frozen=True)

    center: tuple[float, float, float] | None = None
    """The center point of the rectangle."""
    normal: tuple[float, float, float] | None = None
    """The normal vector of the rectangle."""
    tangent: tuple[float, float, float] | None = None
    """The tangent vector of the rectangle.

    The length of the rectangle is along the tangent direction and the
    width is along the cross product of the normal and tangent.

    """
    length: PositiveFloat | None = None
    """The length of the rectangle along the tangent direction."""
    width: PositiveFloat | None = None
    """The width of the rectangle.

    The width is along the cross product of the normal and tangent.

    """
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None

    @model_validator(mode="after")
    def _either_bounds_or_center_normal_tangent(self) -> Self:
        """Validates that either bounds or center/normal/tangent are provided."""

        bounds_defined = self.bounds is not None
        center_normal_tangent_defined = [
            attr is not None
            for attr in (
                self.center,
                self.normal,
                self.tangent,
                self.length,
                self.width,
            )
        ]

        if bounds_defined == any(center_normal_tangent_defined):
            raise ValueError(
                "Either only the 'bounds' or  the parameters 'center', "
                "'normal', 'tangent', 'length', and 'width' must be "
                "provided, but not both."
            )
        if not all(center_normal_tangent_defined) and not bounds_defined:
            raise ValueError(
                "If 'bounds' is not provided, then all of the parameters "
                "'center', 'normal', 'tangent', 'length', and 'width' must "
                "be provided."
            )
        return self

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Validates that the bounds are correctly defined."""
        min_corner, max_corner = self.bounds
        if any(min_c > max_c for min_c, max_c in zip(min_corner, max_corner)):
            raise ValueError(
                "Minimum corner must be less than or equal to maximum corner."
            )
        return self


class Disk(BaseModel):
    """A disk surface.

    Defined by a center point, a normal vector, and a radius.

    """

    type: Literal["disk"] = Field("disk", frozen=True)

    normal: tuple[float, float, float] | None = None
    """The normal vector of the disk."""
    center: tuple[float, ...]
    """The center point of the disk."""
    radius: PositiveFloat
    """The radius of the disk."""

    @model_validator(mode="after")
    def _validate_center_dimension(self) -> Self:
        if self.normal is not None and len(self.center) != len(self.normal):
            raise ValueError(
                "The dimension of the center point must match the dimension of the normal vector."
            )
        return self


class Line(BaseModel):
    """A line segment defined by its two endpoints."""

    type: Literal["line"] = Field("line", frozen=True)

    start: float | tuple[float, ...]
    """The starting point of the line segment."""
    end: float | tuple[float, ...]
    """The endpoint of the line segment."""

    @model_validator(mode="after")
    def _validate_endpoints(self) -> Self:
        """Validates that the bounds are correctly defined."""
        if isinstance(self.start, float) and isinstance(self.end, float):
            return self  # 1D case is valid

        if len(self.start) != len(self.end):
            raise ValueError("All endpoints must have the same dimension.")
        return self


class Point(BaseModel):
    """A point in space."""

    type: Literal["point"] = Field("point", frozen=True)

    position: float | tuple[float, ...]
    """The coordinates of the point."""


class ShapeUnion(BaseModel):
    """A union of multiple shapes."""

    type: Literal["union"] = "union"

    shapes: list["Shape"]
    """The shapes that are part of the union."""


class ShapeDifference(BaseModel):
    """A difference of two shapes."""

    type: Literal["difference"] = "difference"

    base: "Shape"
    """The base shape from which another shape gets subtracted."""
    subtract: "Shape"
    """The shape that is subtracted from the base shape."""


class _ShapeUnionReference(BaseModel):
    """A union that can include inline shapes and named references."""

    type: Literal["union"] = "union"

    shapes: list[Union["_ConfigShape", str]]


class _ShapeDifferenceReference(BaseModel):
    """A difference where operands can be inline shapes or references."""

    type: Literal["difference"] = "difference"

    base: Union["_ConfigShape", str]
    subtract: Union["_ConfigShape", str]


# The main Shape type used in Region definitions, allowing for direct
# shapes and unions/differences.
Shape = Annotated[
    Box
    | Cylinder
    | Tube
    | Rectangle
    | Disk
    | Line
    | Point
    | ShapeUnion
    | ShapeDifference,
    Field(discriminator="type"),
]

# _ConfigShape is used in the raw config parsing stage, allowing for
# named references before resolution.
_ConfigShape = Annotated[
    Box
    | Cylinder
    | Tube
    | Rectangle
    | Disk
    | Line
    | Point
    | _ShapeUnionReference
    | _ShapeDifferenceReference,
    Field(discriminator="type"),
]


# Rebuild models to resolve forward references for recursive types.
ShapeUnion.model_rebuild()
ShapeDifference.model_rebuild()
_ShapeUnionReference.model_rebuild()
_ShapeDifferenceReference.model_rebuild()


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


class Properties(BaseModel):
    """Properties associated with a geometry region."""

    model_config = ConfigDict(extra="forbid")

    epsilon_r: float | None = None
    """The relative permittivity of the material in this region.

    This is used for solving the Poisson equation.

    """
    donor_concentration: float = 0.0
    """The donor concentration in this region in cm^-3.

    This is used for solving the Poisson equation and to determine
    the contact Fermi levels.

    """
    acceptor_concentration: float = 0.0
    """The acceptor concentration in this region in cm^-3.

    This is used for solving the Poisson equation and to determine
    the contact Fermi levels.

    """
    effective_mass: float | tuple[float, float, float] | None = None
    """The effective mass of the material in this region."""
    band_offset: float | None = None
    """The conduction band offset of the material in this region."""
    work_function: float | None = None
    """The work function of the geometry region."""
    voltage: float | None = None
    """The voltage applied at the geometry region."""


class Region(BaseModel):
    """Physical region in the geometry.

    This is the main model that gets used in the simulation, with all
    shape references resolved to actual shape definitions.

    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """A unique name for the region."""
    shape: Shape
    """The geometry of the region."""
    properties: Properties | None = None
    """The properties associated with the region."""
    mesh_size: PositiveFloat | None = None
    """The target mesh size for this region.

    This is an optional hint for the meshing process to control the
    local mesh density. If not provided, a default mesh size will be
    used.

    """


class _ConfigRegion(BaseModel):
    """A region allowing a shape object or a named shape reference.

    Properties stay untyped here so shape resolution can decide whether
    they should be interpreted as volume or surface properties.

    """

    name: str
    shape: _ConfigShape | str
    properties: Any | None = None
    mesh_size: PositiveFloat | None = None


_shape_type_to_valid_dimensions = {
    "box": (3,),
    "cylinder": (3,),
    "tube": (3,),
    "rectangle": (2, 3),
    "disk": (2, 3),
    "line": (1, 2, 3),
    "point": (1, 2, 3),
}


class GeometryConfig(BaseModel):
    """Final resolved geometry configuration for the device."""

    model_config = ConfigDict(extra="forbid")

    regions: list[Region] | None = None
    """The list of regions in the device geometry."""

    atoms_mesh_size: PositiveFloat | None = None
    """The target mesh size for atomistic regions.

    This is an optional hint for the meshing process to control the local
    mesh density in atomistic regions. If not provided, a default mesh
    size will be used.

    """
    default_mesh_size: PositiveFloat = 5.0
    """The default target mesh size.
    
    Used for regions without a specified mesh size.

    """
    dimensionality: Literal[1, 2, 3] = 3
    """The dimensionality of the device geometry.

    This is used to determine the types of shapes that are valid in the
    geometry configuration. For example, in a 2D device, only 2D shapes
    (and lower-dimensional shapes) like rectangles (and lines) are
    allowed, while in a 3D device, 3D shapes like boxes are allowed.

    Dimensionalities lower than 3 imply that the material is homogeneous
    in the missing dimensions.

    """

    # TODO: Consider adding a `from_file` field to indicate if the
    # geometry is provided in a file. This could be useful for future
    # extensions where geometry configurations might be loaded from
    # external sources.
    # from_file: bool = False
    # """Whether the geometry is provided in a file."""

    @model_validator(mode="after")
    def _validate_region_dimensionalities(self) -> Self:
        """Validates that the shapes of all regions are compatible with
        the specified dimensionality of the device.

        Raises
        ------
        ValueError
            If any region's shape is not compatible with the device's
            dimensionality.

        """
        if self.regions is None:
            return self

        for region in self.regions:
            valid_dims = _shape_type_to_valid_dimensions.get(region.shape.type)
            if valid_dims and self.dimensionality not in valid_dims:
                raise ValueError(
                    f"Region '{region.name}' has shape type '{region.shape.type}' "
                    f"which is not compatible with the device dimensionality "
                    f"{self.dimensionality}D."
                )
        return self


class _RawGeometryConfig(BaseModel):
    """Raw config before resolving named shape references."""

    model_config = ConfigDict(extra="forbid")

    regions: list[_ConfigRegion] | None = None
    atoms_mesh_size: float | None = None
    default_mesh_size: float = 5.0
    dimensionality: Literal[1, 2, 3] = 3


def _resolve_config_shape(
    shape: _ConfigShape | str,
    shape_defs: dict[str, _ConfigShape],
    stack: tuple[str, ...] = (),
) -> Shape:
    """Resolves a _ConfigShape or shape reference to a Shape.

    This function recursively resolves any shape definitions,
    including unions and differences, while also checking for cyclic
    references.

    Parameters
    ----------
    shape : _ConfigShape | str
        The shape to resolve, which may include references.
    shape_defs : dict[str, _ConfigShape]
        A dictionary of shape definitions from the config.
    stack : tuple[str, ...], optional
        A tuple representing the current stack of references being
        resolved, used for cycle detection (default is an empty tuple).

    Returns
    -------
    Shape
        The fully resolved shape with all references replaced by actual
        shape definitions.

    """
    if isinstance(
        shape,
        (
            Box,
            Cylinder,
            Tube,
            Rectangle,
            Disk,
            Line,
            Point,
            ShapeUnion,
            ShapeDifference,
        ),
    ):
        return shape

    if isinstance(shape, str):
        # Handling a shape reference.
        if shape not in shape_defs:
            raise ValueError(f"Unknown shape reference '{shape}'")

        if shape in stack:
            cycle = " -> ".join((*stack, shape))
            raise ValueError(f"Cyclic shape reference detected: {cycle}")

        return _resolve_config_shape(shape_defs[shape], shape_defs, (*stack, shape))

    if isinstance(shape, _ShapeUnionReference):
        resolved_shapes = [
            _resolve_config_shape(child, shape_defs, stack) for child in shape.shapes
        ]
        return ShapeUnion(type="union", shapes=resolved_shapes)

    if isinstance(shape, _ShapeDifferenceReference):
        resolved_base = _resolve_config_shape(shape.base, shape_defs, stack)
        resolved_subtract = _resolve_config_shape(shape.subtract, shape_defs, stack)
        return ShapeDifference(
            type="difference", base=resolved_base, subtract=resolved_subtract
        )

    raise ValueError(f"Unsupported shape type '{shape.type}'")


def _assemble_shape_definitions(
    raw_device_config: _RawGeometryConfig,
) -> dict[str, _ConfigShape]:
    """Builds reusable shape definitions from explicit and region-local shapes.

    Inline region shapes are exposed by region name, allowing other
    regions to reference them without requiring top-level shape entries.

    """
    shape_defs = {}

    for region in raw_device_config.regions or []:
        if isinstance(region.shape, str):
            continue

        if region.name in shape_defs:
            raise ValueError(
                f"Duplicate reusable shape name '{region.name}': already defined in device.shapes or another region"
            )

        shape_defs[region.name] = region.shape

    return shape_defs


def parse_geometry_config(raw_device_config: dict[str, Any]) -> GeometryConfig:
    """Parses the raw geometry config and resolves all shape references."""

    # Check if there is a geometry config at all.
    raw_geometry_config = raw_device_config.get("geometry")
    if not raw_geometry_config:
        return GeometryConfig(regions=[])

    raw_geometry_config = _RawGeometryConfig(**raw_geometry_config)
    shape_defs = _assemble_shape_definitions(raw_geometry_config)

    resolved_regions: list[Region] = []
    for region in raw_geometry_config.regions or []:
        resolved_shape = _resolve_config_shape(region.shape, shape_defs)

        resolved_regions.append(
            Region(
                name=region.name,
                shape=resolved_shape,
                properties=region.properties,
                mesh_size=region.mesh_size,
            )
        )

    return GeometryConfig(
        regions=resolved_regions,
        default_mesh_size=raw_geometry_config.default_mesh_size,
        atoms_mesh_size=raw_geometry_config.atoms_mesh_size,
        dimensionality=raw_geometry_config.dimensionality,
    )
