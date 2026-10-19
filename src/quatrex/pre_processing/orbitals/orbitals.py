# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import mmgpy as mmg
import numpy as np
import pyvista as pv

from qttools import NDArray


def _resample_hessian_metric(
    grid: pv.StructuredGrid,
    mesh: pv.UnstructuredGrid,
    target_error: float = 1e-2,
    **create_metric_kwargs: dict,
) -> NDArray:
    """Compute the Hessian metric from a pyvista StructuredGrid sampled
    on a tetrahedral mesh.

    Parameters
    ----------
    grid : pv.StructuredGrid
        The input grid containing scalar data and the computed Hessian
        in the field "hessian".
    mesh : pv.UnstructuredGrid
        The tetrahedral mesh generated from the subsampled grid. The
        Hessian data from the StructuredGrid is resampled onto this
        mesh.
    target_error : float, optional
        The target error for the metric computation.
    **create_metric_kwargs : dict

    Returns
    -------
    metric : NDArray
        The computed Hessian metric as an Nx6 array, where N is the
        number of points in the subsampled grid. The metric is stored in
        the order [xx, xy, xz, yy, yz, zz].

    """
    if grid.point_data.get("hessian") is None:
        raise ValueError(
            "The input grid does not contain Hessian data. "
            "Please compute the Hessian before calling this function."
        )
    hessian = mesh.sample(grid)["hessian"].reshape(-1, 3, 3, order="F")
    metric = mmg.metrics.create_metric_from_hessian(
        mmg.metrics.matrix_to_tensor(hessian),
        target_error=target_error,
        **create_metric_kwargs,
    )
    return metric


def assemble_metric(
    grids: list[pv.StructuredGrid],
    mesh: pv.UnstructuredGrid,
    target_error: float = 1e-2,
    **create_metric_kwargs: dict,
) -> NDArray:
    """Compute the intersection of Hessian metrics from a list of
    pyvista StructuredGrids.

    Parameters
    ----------
    grids : list of pv.StructuredGrid
        The input grids containing scalar data and the computed
        Hessians in the field "hessian".
    mesh : pv.UnstructuredGrid
        The tetrahedral mesh on which the Hessian data from the
        structured grids is resampled.
    target_error : float, optional
        The target error for the metric computation.

    Returns
    -------
    intersection : NDArray
        The intersected Hessian metric as an Nx6 array, where N is the
        number of points in the subsampled grids. The metric is stored
        in the order xx, xy, xz, yy, yz, zz.

    """
    intersection = _resample_hessian_metric(
        grids[0],
        mesh,
        target_error=target_error,
        **create_metric_kwargs,
    )

    for grid in grids[1:]:
        metric = _resample_hessian_metric(
            grid,
            mesh,
            target_error=target_error,
            **create_metric_kwargs,
        )
        intersection = mmg.metrics.intersect_metrics(intersection, metric)

    return intersection


def sample_orbitals_on_mesh(
    grids: list[pv.StructuredGrid],
    mesh: pv.UnstructuredGrid,
    translation: NDArray | None = None,
) -> list[pv.UnstructuredGrid]:
    """Sample the orbitals on a given tetrahedral mesh.

    Parameters
    ----------
    grids : list of pv.StructuredGrid
        The input grids containing scalar data.
    mesh : pv.UnstructuredGrid
        The tetrahedral mesh on which the orbitals are sampled.
    translation : NDArray, optional
        A translation vector to apply to the grids before sampling. If
        None, no translation is applied.

    Returns
    -------
    orbital_meshes : list of pv.UnstructuredGrid
        The sampled orbitals on the tetrahedral mesh.

    """
    orbital_meshes = []
    for grid in grids:
        orbital_mesh = mesh.copy(deep=False)
        if translation is not None:
            grid = grid.translate(xyz=translation)
        orbital_mesh["data"] = mesh.sample(grid)["data"]
        orbital_meshes.append(orbital_mesh)

    return orbital_meshes


def remesh_orbitals_on_grid(
    grids: list[pv.StructuredGrid],
    mesh: pv.UnstructuredGrid,
    num_iterations: int = 3,
    target_error: float = 1e-2,
    **create_metric_kwargs: dict,
) -> tuple[pv.UnstructuredGrid, list[dict]]:
    """Remesh the orbitals on a given grid.

    Parameters
    ----------
    grids : list of pv.StructuredGrid
        The input grids containing scalar data and the computed Hessians
        in the field "hessian".
    mesh : pv.UnstructuredGrid
        The tetrahedral mesh on which the Hessian data from the
        structured grids is resampled.
    num_iterations : int, optional
        The number of iterations to run the remeshing process.
    target_error : float, optional
        The target error for the metric computation.
    **create_metric_kwargs : dict
        Additional keyword arguments to pass to the metric creation
        function.

    Returns
    -------
    mesh : pv.UnstructuredGrid
        The remeshed tetrahedral mesh after the specified number of
        iterations.
    remesh_info : list of dict
        A list of dictionaries containing information about each
        remeshing iteration, including the number of vertices and
        elements in the remeshed mesh.

    """
    mesh = mesh.copy()
    remesh_info = []
    for __ in range(num_iterations):
        mmg_mesh = mmg.from_pyvista(mesh)
        metric = assemble_metric(
            grids,
            mesh,
            target_error=target_error,
            **create_metric_kwargs,
        )
        mmg_mesh.set_field("tensor", metric)
        # TODO: Decide on sensible parameters for remeshing.
        info = mmg_mesh.remesh(hgrad=3, hausd=0.01 * mesh.length)
        remesh_info.append(info)
        mesh = mmg.to_pyvista(mmg_mesh)

    return mesh, remesh_info


def compute_l2_errors(
    grids: list[pv.StructuredGrid],
    meshes: list[pv.UnstructuredGrid],
) -> NDArray:
    """Compute the relative L2 errors of the orbitals on the remeshed grid.

    Parameters
    ----------
    grids : list of pv.StructuredGrid
        The input grids containing scalar data.
    meshes : list of pv.UnstructuredGrid
        The remeshed tetrahedral meshes.

    Returns
    -------
    relative_l2_errors : NDArray
        The relative L2 errors of the orbitals on the remeshed grid.

    """
    relative_l2_errors = []
    for grid, mesh in zip(grids, meshes):
        grid = grid.copy()
        sampled = grid.sample(mesh)["data"]
        grid["error"] = np.abs(grid["data"] - sampled) ** 2
        grid["squared_modulus"] = np.abs(grid["data"]) ** 2
        integrated = grid.integrate_data()
        relative_l2_errors.append(integrated["error"] / integrated["squared_modulus"])

    return np.array(relative_l2_errors)
