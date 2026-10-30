# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Assemblese effective-mass Hamiltonian from a device mesh."""

import warnings

import meshio
import numpy as np
import skfem
from scipy import sparse
from skfem.helpers import dot, grad, mul

from qttools.utils.hdf5_utils import save_hdf5_dict
from quatrex.core.config import QuatrexConfig
from quatrex.core.constants import hbar2_over_2me
from quatrex.electrostatics.meshing import DeviceMesh

_DIM_TO_CELL_TYPE = {
    1: "line",
    2: "triangle",
    3: "tetra",
}

_GAMMA_KEY = "[0,0,0]"


def _initialize_skfem_basis(
    mesh: meshio.Mesh,
    dim: int,
) -> tuple[skfem.Mesh, skfem.Basis]:
    """Builds a P1 skfem mesh/basis from a meshio mesh (1D/2D/3D)."""
    if dim not in _DIM_TO_CELL_TYPE:
        raise ValueError(f"Unsupported dimensionality {dim}. Expected 1, 2, or 3.")

    cell_type = _DIM_TO_CELL_TYPE[dim]
    if cell_type not in mesh.cells_dict:
        raise ValueError(
            f"Mesh does not contain '{cell_type}' cells required for "
            f"dimensionality={dim}. Available: {list(mesh.cells_dict.keys())}."
        )

    points = np.asarray(mesh.points, dtype=float)
    elements = np.asarray(mesh.cells_dict[cell_type], dtype=int)

    if dim == 1:
        sk_mesh = skfem.MeshLine(points[:, :1].T, elements.T)
        basis = skfem.Basis(sk_mesh, skfem.ElementLineP1())
    elif dim == 2:
        sk_mesh = skfem.MeshTri(points[:, :2].T, elements.T)
        basis = skfem.Basis(sk_mesh, skfem.ElementTriP1())
    else:
        sk_mesh = skfem.MeshTet(points[:, :3].T, elements.T)
        basis = skfem.Basis(sk_mesh, skfem.ElementTetP1())

    return sk_mesh, basis


def _as_tensor(value, dim: int, *, name: str = "tensor") -> np.ndarray:
    """Map scalar / diagonal / full matrix input to a (dim, dim) tensor.

    A length-3 vector with dim < 3 keeps the first dim entries (e.g. a
    3-component m* on a 2D device).

    """
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        if arr == 0.0:
            raise ValueError(f"{name} cannot be zero.")
        return float(arr) * np.eye(dim)

    if arr.ndim == 1:
        if arr.shape == (3,) and dim < 3:
            arr = arr[:dim]
        if arr.shape != (dim,):
            raise ValueError(
                f"{name} diagonal vector must have length {dim} "
                f"(or 3 for dim<3), got {arr.shape}."
            )
        if np.any(arr == 0.0):
            raise ValueError(f"{name} diagonal entries cannot be zero.")
        return np.diag(arr)

    if arr.ndim == 2:
        if arr.shape != (dim, dim):
            raise ValueError(
                f"{name} tensor must have shape ({dim}, {dim}), got {arr.shape}."
            )
        return arr

    raise ValueError(
        f"Invalid {name} input: expected scalar, length-{dim} vector, "
        f"or ({dim}, {dim}) matrix."
    )


def _inverse_mass_tensor(m_star, dim: int) -> np.ndarray:
    """Inverse effective-mass tensor."""
    return np.linalg.inv(_as_tensor(m_star, dim, name="effective mass"))


def _element_indices_for_region(
    mesh: meshio.Mesh,
    region_name: str,
    cell_type: str,
    dim: int,
) -> np.ndarray:
    """Element indices for a named physical group."""
    sets_dict = mesh.cell_sets_dict or {}
    if region_name in sets_dict and cell_type in sets_dict[region_name]:
        idx = np.asarray(sets_dict[region_name][cell_type], dtype=int)
        if idx.size:
            return np.unique(idx)

    cell_data = mesh.cell_data_dict or {}
    physical = cell_data.get("gmsh:physical", {})
    tags = physical.get(cell_type)
    if tags is None or mesh.field_data is None or region_name not in mesh.field_data:
        return np.array([], dtype=int)

    tag_meta = mesh.field_data[region_name]
    # field_data[name] = [physical_tag, geometric_dimension]
    if int(tag_meta[1]) != dim:
        return np.array([], dtype=int)

    region_tag = int(tag_meta[0])
    tags = np.asarray(tags, dtype=int)
    return np.flatnonzero(tags == region_tag).astype(int)


def _build_material_quadtrature_fields(
    basis: skfem.Basis,
    mesh: meshio.Mesh,
    config: QuatrexConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Map effective_mass / band_offset onto quadrature points

    Regions withouth effective_mass are skipped. Unassigned volume elemnts fall back to m*=1 and band_offset=0.

    """

    dim = basis.mesh.dim()
    nelems = basis.mesh.nelements
    nqp = basis.X.shape[1]
    cell_type = _DIM_TO_CELL_TYPE[dim]

    inv_m_qp = np.zeros((dim, dim, nelems, nqp), dtype=float)
    v_qp = np.zeros((nelems, nqp), dtype=float)
    assigned = np.zeros(nelems, dtype=bool)

    for region in config.device.geometry.regions or []:
        if region.properties is None or region.properties.effective_mass is None:
            continue

        elem_idx = _element_indices_for_region(mesh, region.name, cell_type, dim)
        if elem_idx.size == 0:
            warnings.warn(
                f"Region '{region.name}' has effective_mass but no '{cell_type}' "
                f"elements in the mesh physical groups; skipping.",
                stacklevel=2,
            )
            continue

        inv_m = _inverse_mass_tensor(region.properties.effective_mass, dim)
        band_offset = (
            0.0
            if region.properties.band_offset is None
            else float(region.properties.band_offset)
        )

        inv_m_qp[:, :, elem_idx, :] = inv_m[:, :, None, None]
        v_qp[elem_idx, :] = band_offset
        assigned[elem_idx] = True

    if not np.all(assigned):
        n_unassigned = int(np.sum(~assigned))
        warnings.warn(
            f"{n_unassigned} volume elements have no effective_mass; "
            "using m*=1 and band_offset=0.",
            stacklevel=2,
        )
        unassigned = np.flatnonzero(~assigned)
        inv_m_qp[:, :, unassigned, :] = np.eye(dim)[:, :, None, None]

    return inv_m_qp, v_qp


def _assemble_bdd_hamiltonian(
    basis: skfem.Basis, inv_m_qp: np.ndarray, v_qp: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Assemble full BDD Haimltoninan and overlap matrix."""

    @skfem.BilinearForm
    def kinetic_and_pot(u, v, w):
        gu, gv = grad(u), grad(v)
        # 1D: promote grad to shape (1, ...)
        if gu.ndim == 2:
            gu = gu[None, :, :]
            gv = gv[None, :, :]
        kinetic = hbar2_over_2me * dot(mul(w["inv_m"], gu), gv)
        pot = w["V"] * u * v
        return kinetic + pot

    @skfem.BilinearForm
    def mass(u, v, w):
        return u * v

    hamiltonian = kinetic_and_pot.assemble(basis, inv_m=inv_m_qp, V=v_qp)
    overlap = mass.assemble(basis)
    return hamiltonian.tocsr(), overlap.tocsr()


def assemble_hamiltonian(config: QuatrexConfig) -> None:
    """Assemble and save effective-mass hamiltoninan.h5 / overlap.h5"""

    dim = config.device.geometry.dimensionality
    device_mesh = DeviceMesh.from_config(config)

    sk_mesh, basis = _initialize_skfem_basis(device_mesh.mesh, dim)
    inv_m_qp, v_qp = _build_material_quadtrature_fields(basis, device_mesh.mesh, config)
    hamiltonian, overlap = _assemble_bdd_hamiltonian(basis, inv_m_qp, v_qp)

    input_dir = config.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    save_hdf5_dict(str(input_dir / "hamiltonian.h5"), {_GAMMA_KEY: hamiltonian})
    save_hdf5_dict(str(input_dir / "overlap.h5"), {_GAMMA_KEY: overlap})

    print(
        f"Saved effective-mass Hamiltonian ({hamiltonian.shape[0]} DOFs to)"
        f"{input_dir / 'hamiltonian.h5'} and overlap to"
        f"{input_dir / 'overlap.h5'}.",
        flush=True,
    )
