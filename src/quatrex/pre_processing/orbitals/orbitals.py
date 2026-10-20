# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from abc import ABC, abstractmethod
from collections.abc import Callable

import mmgpy as mmg
import numpy as np
import pyvista as pv

from qttools import NDArray


class RealSpaceOrbitals(ABC):
    """Abstract base class for real-space orbitals.

    This class defines the interface for evaluating orbitals, their
    gradients, and Hessians at given positions in real space. Subclasses
    must implement these methods to provide specific functionality for
    different types of orbitals, such as those from CP2K or Wannier90.

    """

    @abstractmethod
    def evaluate_orbitals(self, positions: NDArray) -> NDArray:
        """Evaluate the values of the orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital values.

        Returns
        -------
        orbitals : NDArray
            The evaluated orbitals at the given positions. The shape of
            the output will be (M, N), where M is the total number of
            orbitals and N is the number of points.

        """
        ...

    @abstractmethod
    def evaluate_orbital_gradients(self, positions: NDArray) -> NDArray:
        """Evaluate gradients of the orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital gradients.

        Returns
        -------
        gradients : NDArray
            The evaluated orbital gradients at the given positions. The
            shape of the output will be (M, N, 3), where M is the total
            number of orbitals, N is the number of points, and 3
            corresponds to the x, y, z components of the gradient.

        """
        ...

    @abstractmethod
    def evaluate_orbital_hessians(self, positions: NDArray) -> NDArray:
        """Evaluate Hessians of the orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital Hessians.

        Returns
        -------
        hessians : NDArray
            The evaluated orbital Hessians at the given positions. The
            shape of the output will be (M, N, 3, 3), where M is the
            total number of orbitals, N is the number of points 3x3
            corresponds to the Hessian matrix.

        """
        ...


def intersect_tensor_metrics(metrics_a: NDArray, metrics_b: NDArray) -> NDArray:
    """Compute the intersection of two tensor metrics.

    The two metrics have shape (..., 3, 3) and are assumed to be
    symmetric positive definite.

    This is a batched reimplementation of what mmgpy also does. There it
    however loops over all the DOFs, which is a bit slower. The
    shape-agnostic batching also allows intersecting two sets of metrics
    with different numbers of DOFs, which is useful to efficiently
    reduce the Hessian metrics of multiple orbitals to a single metric.

    Parameters
    ----------
    metrics_a : NDArray
        The first set of tensor metrics with shape (..., 3, 3).
    metrics_b : NDArray
        The paired set of tensor metrics with shape (..., 3, 3).

    Returns
    -------
    metrics : NDArray
        The intersected tensor metrics with shape (..., 3, 3).

    """
    ws, vs = np.linalg.eigh(metrics_a)

    # Ensure that the eigenvalues are positive to avoid numerical issues.
    eps = (
        np.finfo(metrics_a.dtype).eps * np.max(np.abs(ws), axis=-1, keepdims=True) * 100
    )
    ws = np.maximum(ws, eps)

    # Move metrics_b into the eigenbases of metrics_a.
    sqrt_a = np.einsum("...ik,...k,...jk->...ij", vs, np.sqrt(ws), vs)
    inv_sqrt_a = np.einsum("...ik,...k,...jk->...ij", vs, 1.0 / np.sqrt(ws), vs)
    transformed_metric = np.einsum(
        "...ij,...jk,...kl->...il", inv_sqrt_a, metrics_b, inv_sqrt_a
    )

    ws_transformed, vs_transformed = np.linalg.eigh(transformed_metric)
    ws_transformed = np.maximum(ws_transformed, 1.0)

    middle = np.einsum(
        "...ik,...k,...jk->...ij", vs_transformed, ws_transformed, vs_transformed
    )
    result = np.einsum("...ij,...jk,...lk->...il", sqrt_a, middle, sqrt_a)

    return result.reshape(metrics_a.shape)


def reduce_metrics(metrics: NDArray) -> NDArray:
    """Reduce a set of metrics to a single metric.

    This function computes the intersection of a set of metrics by
    sequentially intersecting sets of metrics until only one metric
    remains. The input metrics are assumed to be symmetric positive
    definite and have shape (N, ..., 3, 3).

    This is a parallel tree-reduction version of
    mmgpy.metrics.intersect_metrics, which would have to be called over
    and over again on two metrics. This is not guaranteed to give the
    exact same result as the sequential intersection, but it is much
    faster

    Parameters
    ----------
    metrics : NDArray
        The input metrics with shape (N, ..., 3, 3).

    Returns
    -------
    metric : NDArray
        The reduced metric with shape (..., 3, 3).


    """
    while metrics.shape[0] > 1:
        num_metrics = metrics.shape[0]
        if num_metrics % 2:
            # In the case of an odd number of metrics, carry the last
            # one over to the next step in the reduction.
            carry, metrics = metrics[-1:], metrics[:-1]
        else:
            # In the case of an even number of metrics, no need to
            # carry.
            carry = metrics[:0]
        metrics = np.concatenate(
            [intersect_tensor_metrics(metrics[0::2], metrics[1::2]), carry]
        )
    return metrics[0]


def metrics_from_hessians(
    hessians: NDArray,
    target_error: float = 1e-2,
    hmin: float | None = None,
    hmax: float | None = None,
) -> NDArray:
    """Compute the Hessian metric from a set of Hessian matrices.

    This implements the same metric computation as
    mmgpy.metrics.create_metric_from_hessian, but in a batched and
    shape-agnostic way and without converting from and to mmgpy's tensor
    representation.

    Parameters
    ----------
    hessians : NDArray
        The input Hessian matrices with shape (..., 3, 3).
    target_error : float, optional
        The target error for the metric computation.
    hmin : float, optional
        The minimum edge length for the metric. If provided, the
        eigenvalues of the metric will be clipped to ensure that the
        minimum edge length is respected.
    hmax : float, optional
        The maximum edge length for the metric. If provided, the
        eigenvalues of the metric will be clipped to ensure that the
        maximum edge length is respected.

    Returns
    -------
    metrics : NDArray
        The computed Hessian metrics with shape (..., 3, 3).

    """
    # This is apparently related to the interpolation error of a linear
    # function on a tetrahedron. The given target error is not a L2
    # error, but rather some kind of interpolation error.
    c = 1.0 / 8.0

    ws, vs = np.linalg.eigh(hessians)
    metric_ws = c * np.abs(ws) / target_error

    # Clamp the eigenvalues of the metric to ensure that the minimum and
    # maximum edge lengths are respected.
    w_min = max(1e-12, 1.0 / hmin**2) if hmin is not None else 1e-12
    w_max = 1.0 / hmax**2 if hmax is not None else None

    np.clip(metric_ws, min=w_min, max=w_max, out=metric_ws)

    metrics = np.einsum("...ij,...j,...kj->...ik", vs, metric_ws, vs)
    return metrics


def assemble_metric(
    hessians: NDArray,
    target_error: float = 1e-2,
    **create_metric_kwargs: dict,
) -> NDArray:
    """Compute a global Hessian metric from a set of Hessian matrices.

    The input Hessians are assumed to have shape (N, ..., 3, 3), where N
    is the number of orbitals, and the rest are the DOFs where the
    Hessians are defined.

    Parameters
    ----------
    hessians : NDArray
        The input Hessian matrices with shape (N, ..., 3, 3), where
        N is the number of orbitals.
    target_error : float, optional
        The target error for the metric computation.
    **create_metric_kwargs : dict
        Additional keyword arguments to pass to the metric creation
        function.

    Returns
    -------
    metric : NDArray
        The computed Hessian metric as an Nx6 array, where N is the
        number of points in the subsampled grids. The metric is stored
        in the order xx, xy, xz, yy, yz, zz.

    """
    metrics = metrics_from_hessians(
        hessians,
        target_error=target_error,
        **create_metric_kwargs,
    )

    intersection = reduce_metrics(metrics)

    # Now convert to the mmgpy tensor representation, which is a 6D
    # vector with the order xx, xy, xz, yy, yz, zz. This is the same as
    # mmgpy.metrics.matrix_to_tensor, but batched.
    i, j = np.triu_indices(3)
    return intersection[..., i, j]


def remesh_for_orbitals(
    mesh: pv.UnstructuredGrid,
    compute_hessians: Callable[[pv.UnstructuredGrid], NDArray],
    num_iterations: int = 3,
    target_error: float = 1e-2,
    **create_metric_kwargs: dict,
) -> tuple[pv.UnstructuredGrid, list[dict]]:
    """Remesh the orbitals on a given grid.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        The tetrahedral mesh on which the Hessian data from the
        structured grids is re-evaluated.
    compute_hessians : Callable[[pv.UnstructuredGrid], NDArray]
        A function that computes the Hessian of the orbitals on a given mesh.
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
        hessians = compute_hessians(mesh)
        metric = assemble_metric(
            hessians,
            target_error=target_error,
            **create_metric_kwargs,
        )
        mmg_mesh = mmg.from_pyvista(mesh)
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
