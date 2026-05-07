from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Box(BaseModel):
    """An axis-aligned box defined by its minimum and maximum corners."""

    type: Literal["box"] = "box"

    bounds: list[list[float]]
    """The bounds of the box defined by its minimum and maximum corners."""


class Cylinder(BaseModel):
    """A solid cylinder.

    Defined by a center point, an axis vector, a radius, and a length.

    """

    type: Literal["cylinder"] = "cylinder"

    center: list[float]
    """The center point of the cylinder."""
    radius: float
    """The radius of the cylinder."""
    axis: list[float]
    """The axis along which the cylinder extends."""
    length: float
    """The length of the cylinder along the given axis."""


class Rectangle(BaseModel):
    """A rectangular surface.

    Needs a normal for orientation and a tangent vector to define the
    direction of the length and width. The rectangle is centered at the
    given center point.

    """

    type: Literal["rectangle"] = "rectangle"

    center: list[float]
    """The center point of the rectangle."""
    normal: list[float]
    """The normal vector of the rectangle."""
    tangent: list[float]
    """The tangent vector of the rectangle.

    The length of the rectangle is along the tangent direction and the
    width is along the cross product of the normal and tangent.

    """
    length: float
    """The length of the rectangle along the tangent direction."""
    width: float
    """The width of the rectangle.

    The width is along the cross product of the normal and tangent.

    """


class Tube(BaseModel):
    """A cylindrical surface.

    Defined by a center point, an axis vector, a radius, and a length.

    """

    type: Literal["tube"] = "tube"

    center: list[float]
    """The center point of the tube."""
    axis: list[float]
    """The axis along which the tube extends."""
    radius: float
    """The radius of the tube."""
    length: float
    """The length of the tube along the given axis."""


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
    Union[Box, Cylinder, Rectangle, Tube, ShapeUnion, ShapeDifference],
    Field(discriminator="type"),
]

# _ConfigShape is used in the raw config parsing stage, allowing for
# named references before resolution.
_ConfigShape = Annotated[
    Union[
        Box,
        Cylinder,
        Rectangle,
        Tube,
        _ShapeUnionReference,
        _ShapeDifferenceReference,
    ],
    Field(discriminator="type"),
]


# Rebuild models to resolve forward references for recursive types.
ShapeUnion.model_rebuild()
ShapeDifference.model_rebuild()
_ShapeUnionReference.model_rebuild()
_ShapeDifferenceReference.model_rebuild()


class VolumeProperties(BaseModel):
    """Properties associated with a volumetric region."""

    model_config = ConfigDict(extra="forbid")

    epsilon_r: float | None = None
    """The relative permittivity of the material in this region.

    This is used for solving the Poisson equation.

    """
    donor_concentration: float | None = None
    """The donor concentration in this region.

    This is used for solving the Poisson equation and to determine
    the contact Fermi levels.

    """
    acceptor_concentration: float | None = None
    """The acceptor concentration in this region.

    This is used for solving the Poisson equation and to determine
    the contact Fermi levels.

    """


class SurfaceProperties(BaseModel):
    """Properties associated with a surface region."""

    model_config = ConfigDict(extra="forbid")

    work_function: float
    """The work function of the surface."""

    voltage: float
    """The voltage applied at the surface."""


Properties = Union[VolumeProperties, SurfaceProperties]


def _expected_properties_model(shape: Shape) -> type[BaseModel]:
    """Returns the properties model implied by a resolved shape."""
    if isinstance(shape, (Rectangle, Tube)):
        return SurfaceProperties
    return VolumeProperties


def _coerce_properties_for_shape(shape: Shape, value: Any) -> Properties:
    """Validates and coerces the properties for a given shape."""
    expected_model = _expected_properties_model(shape)

    if isinstance(value, expected_model):
        return value

    if isinstance(value, (VolumeProperties, SurfaceProperties)):
        raise ValueError(
            f"properties type {type(value).__name__} does not match shape type {type(shape).__name__}"
        )

    if not isinstance(value, dict):
        raise TypeError(
            f"properties for shape type {type(shape).__name__} must be a mapping"
        )

    return expected_model.model_validate(value)


class Region(BaseModel):
    """Physical region in the geometry.

    This is the main model that gets used in the simulation, with all
    shape references resolved to actual shape definitions.

    """

    name: str
    """A unique name for the region."""
    shape: Shape
    """The geometry of the region."""
    properties: Any
    """The properties associated with the region.

    The type of the properties is determined by the shape of the region.

    """
    mesh_size: float | None = None
    """The target mesh size for this region.

    This is an optional hint for the meshing process to control the
    local mesh density. If not provided, a default mesh size will be
    used.

    """

    @field_validator("properties", mode="before")
    @classmethod
    def _validate_properties_against_shape(cls, value: Any, info) -> Any:
        shape = info.data.get("shape")
        if shape is None:
            return value
        return _coerce_properties_for_shape(shape, value)


class _ConfigRegion(BaseModel):
    """A region allowing a shape object or a named shape reference.

    Properties stay untyped here so shape resolution can decide whether
    they should be interpreted as volume or surface properties.

    """

    name: str
    shape: Union[_ConfigShape, str]
    properties: Any
    mesh_size: float | None = None


class GeometryConfig(BaseModel):
    """Final resolved geometry configuration for the device."""

    regions: list[Region] | None = None
    """The list of regions in the device geometry."""

    atoms_mesh_size: float | None = None
    """The target mesh size for atomistic regions.

    This is an optional hint for the meshing process to control the local
    mesh density in atomistic regions. If not provided, a default mesh
    size will be used.

    """
    default_mesh_size: float = 5.0
    """The default target mesh size.
    
    Used for regions without a specified mesh size.

    """


class _RawGeometryConfig(BaseModel):
    """Raw config before resolving named shape references."""

    regions: list[_ConfigRegion] | None = None
    atoms_mesh_size: float | None = None
    default_mesh_size: float = 5.0


def _resolve_shape_reference(
    ref: str,
    shape_defs: dict[str, _ConfigShape],
    stack: tuple[str, ...],
) -> Shape:
    """Resolves a shape reference to an actual Shape.

    This function checks for cyclic references by maintaining a stack of
    currently resolving references. If a reference is encountered that
    is already in the stack, a ValueError is raised.

    Parameters
    ----------
    ref : str
        The name of the shape reference to resolve.
    shape_defs : dict[str, _ConfigShape]
        A dictionary of shape definitions from the config.
    stack : tuple[str, ...]
        A tuple representing the current stack of references being
        resolved, used for cycle detection.

    Returns
    -------
    Shape
        The resolved shape corresponding to the reference.

    """
    if ref not in shape_defs:
        raise ValueError(f"Unknown shape reference '{ref}'")

    if ref in stack:
        cycle = " -> ".join((*stack, ref))
        raise ValueError(f"Cyclic shape reference detected: {cycle}")

    return _resolve_config_shape(shape_defs[ref], shape_defs, (*stack, ref))


def _resolve_config_shape(
    shape: _ConfigShape,
    shape_defs: dict[str, _ConfigShape],
    stack: tuple[str, ...] = (),
) -> Shape:
    """Resolves a _ConfigShape to a Shape.

    This function recursively resolves any shape definitions,
    including unions and differences, while also checking for cyclic
    references.

    Parameters
    ----------
    shape : _ConfigShape
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

    if isinstance(shape, (Box, Cylinder, Rectangle, Tube)):
        return shape

    if isinstance(shape, _ShapeUnionReference):
        resolved_shapes: list[Shape] = []
        for child in shape.shapes:
            if isinstance(child, str):
                resolved_shapes.append(
                    _resolve_shape_reference(child, shape_defs, stack)
                )
            else:
                resolved_shapes.append(_resolve_config_shape(child, shape_defs, stack))
        return ShapeUnion(type="union", shapes=resolved_shapes)

    if isinstance(shape, _ShapeDifferenceReference):
        if isinstance(shape.base, str):
            resolved_base = _resolve_shape_reference(shape.base, shape_defs, stack)
        else:
            resolved_base = _resolve_config_shape(shape.base, shape_defs, stack)

        if isinstance(shape.subtract, str):
            resolved_subtract = _resolve_shape_reference(
                shape.subtract, shape_defs, stack
            )
        else:
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
        if isinstance(region.shape, str):
            resolved_shape = _resolve_shape_reference(region.shape, shape_defs, ())
        else:
            resolved_shape = _resolve_config_shape(region.shape, shape_defs)

        resolved_regions.append(
            Region(
                name=region.name,
                shape=resolved_shape,
                properties=_coerce_properties_for_shape(
                    resolved_shape, region.properties
                ),
                mesh_size=region.mesh_size,
            )
        )

    return GeometryConfig(
        regions=resolved_regions,
        default_mesh_size=raw_geometry_config.default_mesh_size,
        atoms_mesh_size=raw_geometry_config.atoms_mesh_size,
    )
