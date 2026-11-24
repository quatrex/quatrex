# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Methods for evaluating Wannier90 orbitals and their derivatives."""

import warnings
from pathlib import Path
from typing import Self

import ase
import h5py
import numpy as np
import opt_einsum as oe
import pyvista as pv
from scipy import sparse

from qttools import NDArray
from quatrex.core.config import QuatrexConfig
from quatrex.pre_processing.orbitals.orbitals import (
    DEFAULT_REMESHING_OPTIONS,
    DeviceOrbitalProjector,
    OrbitalEvaluator,
    RemeshingOptions,
)
from quatrex.pre_processing.orbitals.wannier90.io import (
    CHKData,
    UNKReader,
    load_unk_data,
)


def _prepare_supercell_meshgrid(
    supercell: tuple[int, int, int], grid_shape: tuple[int, int, int]
) -> tuple[list[NDArray], list[NDArray]]:
    """Prepare the supercell meshgrid for evaluating Wannier functions.

    Parameters
    ----------
    supercell : tuple of int
        The size of the supercell in terms of home cells to evaluate the
        Wannier functions.
    grid_shape : tuple of int
        The shape of the grid in the home cell.

    Returns
    -------
    meshgrid : list of NDArray
        List of 3D arrays representing the meshgrid coordinates in the
        supercell.
    meshgrid_wrapped : list of NDArray
        List of 3D arrays representing the wrapped meshgrid coordinates
        in the home cell.

    """
    # Prepare the real-space supercell grid, centering the the whole
    # thing around the home cell, as done in plot_wannier.F90.
    lower_bounds = [-(s // 2) * n - 1 for s, n in zip(supercell, grid_shape)]
    upper_bounds = [((s + 1) // 2) * n - 1 for s, n in zip(supercell, grid_shape)]
    meshgrid_inds = (
        np.arange(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)
    )
    meshgrid = np.array(np.meshgrid(*meshgrid_inds, indexing="ij"))

    # Wrap meshgrid indices into the home cell to then use indexing
    # hacks when upscaling to the supercell.
    meshgrid_wrapped = [ni % n for ni, n in zip(meshgrid, grid_shape)]
    return meshgrid, meshgrid_wrapped


def _assemble_wannier_functions(
    chk: CHKData,
    u_nk: NDArray,
    meshgrid: list[NDArray],
    meshgrid_wrapped: list[NDArray],
    grid_shape: tuple[int, int, int],
) -> NDArray:
    """Construct the real-space orbitals from CHK and UNK data.

    Parameters
    ----------
    chk : CHKData
        CHKData class containing information like the unitary
        transformation matrices and the disentanglement information.
    u_nk : NDArray
        A complex128 array of shape (num_bands, num_kpoints,
        *grid_shape) containing the cell-periodic Bloch functions for
        all bands and k-points.
    meshgrid : list of NDArray
        List of 3D arrays representing the meshgrid coordinates in the
        supercell.
    meshgrid_wrapped : list of NDArray
        List of 3D arrays representing the wrapped meshgrid coordinates
        in the home cell.
    grid_shape : tuple of int
        The shape of the grid in the home cell.

    Returns
    -------
    wannier_functions : NDArray
        A complex128 array of shape (num_wann, *meshgrid[0].shape )
        containing the Wannier functions evaluated on the supercell
        meshgrid.
    """

    wannier_functions = np.zeros(
        (chk.num_wann, *meshgrid[0].shape),
        dtype=np.complex128,
    )

    for ik in range(chk.num_kpoints):
        # Extract cell-periodic Bloch functions for all bands at the
        # current k-point.
        u_m = u_nk[:, ik].copy()
        if chk.disentangled:
            # If we disentangled, we need to select the bands that are
            # included in the Wannierization and apply the
            # disentanglement unitary transformation.
            included_bands = np.flatnonzero(chk.disentangle_window[:, ik])
            u_mn_dis = chk.u_mnk_opt[: chk.num_bands_disentangled[ik], :, ik]
            u_m = oe.contract("mn,m...->n...", u_mn_dis, u_m[included_bands])

        # Apply the unitary transformation to get the mixed Bloch
        # functions in the home cell.
        u_mn = chk.u_mnk[..., ik]
        u_n = oe.contract("mn,m...->n...", u_mn, u_m)

        # Extend the Bloch functions to the supercell by indexing into
        # the home-cell-sized array using the wrapped meshgrid indices.
        u_n_supercell = u_n[:, *meshgrid_wrapped]

        # Construct the phase factor for the current k-point and
        # meshgrid indices from the supercell meshgrid.
        exponent = (2j * np.pi) * oe.contract(
            "i,i...->...", chk.kpoints[ik] / grid_shape, meshgrid
        )

        # Multiply cell-periodic Bloch functions by the phase factor and
        # sum over k-points to get the Wannier functions in the
        # supercell.
        wannier_functions += u_n_supercell * np.exp(exponent)[None, ...]

    wannier_functions /= chk.num_kpoints
    return wannier_functions


def _fix_global_phase(wannier_functions: NDArray) -> NDArray:
    """Fix the global phase of the Wannier functions.

    This rotates the overall phase of each Wannier function so that
    the value at the point of maximum modulus is real and positive.

    Parameters
    ----------
    wannier_functions : NDArray
        A complex128 array of shape (num_wann, *grid_shape)
        containing the Wannier functions.

    Returns
    -------
    NDArray
        A complex128 array of the same shape as `wannier_functions`,
        but with the global phase of each Wannier function adjusted
        so that the value at the point of maximum modulus is real
        and positive.

    """
    out = np.empty_like(wannier_functions)
    for iw in range(wannier_functions.shape[0]):
        w = wannier_functions[iw]
        idx = np.unravel_index(np.argmax(np.abs(w)), w.shape)
        wmod = w[idx]
        wmod = wmod / abs(wmod)
        out[iw] = w / wmod
    return out


def _reality_check(
    wannier_functions: NDArray,
    threshold: float = 0.01,
) -> None:
    """Check that the Wannier functions are real.

    Issues a warning if the imaginary part of any Wannier function is
    larger than 0.1% of the real part at any grid point where the real
    part is above the specified threshold.

    Parameters
    ----------
    wannier_functions : NDArray
        A complex128 array of shape (num_wann, *grid_shape)
        containing the Wannier functions.
    threshold : float, optional
        The threshold for the real part of the Wannier function.
        Only grid points where |Re| >= threshold are considered in
        the reality check. Default is 0.01.

    """
    ratios = []
    for iw in range(wannier_functions.shape[0]):
        w = wannier_functions[iw]
        mask = np.abs(w.real) >= threshold
        if mask.any():
            ratios.append(float(np.max(np.abs(w.imag[mask]) / np.abs(w.real[mask]))))
        else:
            ratios.append(float("nan"))

    if any(ratio > 1e-3 for ratio in ratios if not np.isnan(ratio)):
        warnings.warn(
            "Wannier functions are not sufficiently real. "
            "Ratios of max |Im|/|Re| for each Wannier function: "
            f"{ratios}."
        )


def _compute_self_overlaps(wannier_grids: list[pv.StructuredGrid]) -> NDArray:
    """Compute the self-overlap of the Wannier functions.

    Parameters
    ----------
    wannier_grids : list of pv.StructuredGrid
        A list of pyvista StructuredGrid objects, each representing
        the real-space grid of a Wannier function.

    Returns
    -------
    NDArray
        A 1D array containing the self-overlap of each Wannier
        function, computed as the sum of the squared values of the
        function over the grid.

    """
    result = []
    for wf in wannier_grids:
        integrand = wf.copy()
        integrand["squared_modulus"] = np.abs(wf["data"]) ** 2

        result.append(integrand.integrate_data()["squared_modulus"])

    return np.array(result)


class Wannier90OrbitalEvaluator(OrbitalEvaluator):
    """A class to represent the orbitals data from Wannier90.

    This class crunches the data from both the UNK files and the CHK
    file, and provides an interface for evaluating Wannier functions on
    a real-space grid, as well as their gradients and Hessians.

    Parameters
    ----------
    unk_data : NDArray
        A complex128 array of shape (num_bands, num_kpoints,
        *grid_shape) containing the cell-periodic Bloch functions for
        all bands and k-points. Must be provided if the grids are to be
        computed from the UNK files and the CHK data. If the grids are
        provided directly, this can be None.
    chk_data : CHKFile, optional
        An instance of the CHKFile class with information about the
        Wannier functions, including the unitary transformation matrices
        and the disentanglement information. Has to be provided if the
        grids are to be computed from the UNK data.
    supercell : tuple of int, optional
        The size of the supercell in terms of home cells to evaluate
        the Wannier functions. Default is (1, 1, 1), which evaluates
        the Wannier functions only in the home cell.
    grids : list of pv.StructuredGrid, optional
        A list of pyvista StructuredGrid objects, each representing
        the real-space grid of a Wannier function. If provided, this
        will be used directly instead of computing the grids from the
        UNK and CHK data.

    """

    def __init__(
        self,
        u_nk: NDArray | None = None,
        chk_data: CHKData | None = None,
        supercell: tuple[int, int, int] = (1, 1, 1),
        grids: list[pv.StructuredGrid] | None = None,
    ):
        """Initialize the Wannier90Orbitals class."""
        if grids is not None:
            self._grids = grids
            return

        if u_nk is None or chk_data is None:
            raise ValueError(
                "Both u_nk and chk_data must be provided if grids are not given."
            )

        # Try to see whether norm-conserving pseudos were used to
        # generate the UNK files. The self-overlap of the periodic part
        # of the Bloch functions should be constant over the bands.
        self_overlap = oe.contract("nkxyz,nkxyz->nk", u_nk, np.conj(u_nk)).mean(axis=1)
        if not np.allclose(self_overlap, self_overlap[0], rtol=1e-3):
            warnings.warn(
                "Self-overlap of Bloch functions is not constant over bands. "
                "This suggests that the UNK files were generated with "
                "pseudopotentials that are not norm-conserving. "
                "The Wannier functions will be normalized regardless!"
            )

        self._grids = self._init_orbital_grids(
            u_nk=u_nk, chk_data=chk_data, supercell=supercell
        )

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        supercell: tuple[int, int, int] = (1, 1, 1),
    ) -> Self:
        """Instantiate orbital evaluator from a directory.

        This method loads the UNK and CHK files from the specified
        directory and sets up the Wannier90OrbitalEvaluator.

        Parameters
        ----------
        path : str or Path
            Path to the directory containing the UNK and CHK files.

        Returns
        -------
        Wannier90OrbitalEvaluator
            An instance of the Wannier90OrbitalEvaluator.

        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"'{path}' is not a directory")

        # Find the chk file in the directory assuming there's only one.
        chk_files = list(path.glob("*.chk"))
        if not chk_files:
            raise FileNotFoundError("No chk file found in the directory")
        if len(chk_files) > 1:
            raise ValueError("Multiple chk files found in the directory")
        chk_data = CHKData.from_file(chk_files[0])

        # Check that CHK and UNK files are consistent in terms of number
        # of k-points and bands.
        if not (len(list(path.glob("UNK?????.1"))) == chk_data.num_kpoints):
            raise ValueError(
                "Number of UNK files does not match number of k-points in CHK file."
            )
        with UNKReader(path / "UNK00001.1") as unk_reader:
            if not (unk_reader.num_bands == chk_data.num_bands):
                raise ValueError(
                    "Number of bands in UNK files does not match number of "
                    "bands in CHK file."
                )

        u_nk = load_unk_data(path)

        return cls(u_nk=u_nk, chk_data=chk_data, supercell=supercell)

    @staticmethod
    def _init_orbital_grids(
        u_nk: NDArray,
        chk_data: CHKData,
        supercell: tuple[int, int, int] = (1, 1, 1),
    ) -> list[pv.StructuredGrid]:
        """Evaluate the Wannier functions on a real-space grid.

        Similar to what plot_wannier.F90 does, but returns a list of
        pyvista StructuredGrid objects instead of writing to .xsf/.cube
        files. The grid is evaluated over a supercell.

        Parameters
        ----------
        u_nk : NDArray
            A complex128 array of shape (num_bands, num_kpoints,
            *grid_shape) containing the cell-periodic Bloch functions
            for all bands and k-points.
        chk : CHKData
            An instance of the CHKData class containing information
            about the Wannier functions, including the unitary
            transformation matrices and the disentanglement information.
        supercell : tuple of int, optional
            The size of the supercell in terms of home cells to evaluate
            the Wannier functions. Default is (1, 1, 1), which evaluates
            the Wannier functions only in the home cell.

        Returns
        -------
        wannier_grids : list of pv.StructuredGrid
            A list of pyvista StructuredGrid objects, each representing
            the real-space grid of a Wannier function evaluated over the
            specified supercell. All grids have the same shape and
            spacing, and contain the Wannier function values, gradients,
            and Hessians as point data.

        """
        grid_shape = np.array(u_nk.shape[2:])
        meshgrid, meshgrid_wrapped = _prepare_supercell_meshgrid(supercell, grid_shape)

        wannier_functions = _assemble_wannier_functions(
            chk=chk_data,
            u_nk=u_nk,
            meshgrid=meshgrid,
            meshgrid_wrapped=meshgrid_wrapped,
            grid_shape=grid_shape,
        )
        wannier_functions = _fix_global_phase(wannier_functions)
        _reality_check(wannier_functions)

        fractional_coords = np.stack(
            [ni / n for ni, n in zip(meshgrid, grid_shape)], axis=-1
        )
        cartesian_coords = fractional_coords @ chk_data.real_lattice
        grid = pv.StructuredGrid(
            cartesian_coords[..., 0],
            cartesian_coords[..., 1],
            cartesian_coords[..., 2],
        )

        grids = []
        for w in wannier_functions:
            grid = grid.copy(deep=False)
            grid["data"] = w.real.ravel(order="F")
            grids.append(grid)

        self_overlaps = _compute_self_overlaps(grids)
        # Check that the Wannier functions all share the same
        # self-overlap, which should be the case if the supercell is
        # large enough and the UNK files were generated with
        # norm-conserving pseudopotentials.
        # NOTE: It is not checked whether the self-overlap is 1, because
        # sometimes things are given in different units.
        if not np.allclose(self_overlaps, self_overlaps[0], rtol=1e-3):
            warnings.warn(
                "Self-overlap of Wannier functions is not constant over orbitals. "
                "This suggests that the supercell is too small or that the UNK files were "
                "generated with pseudopotentials that are not norm-conserving. "
                "The Wannier functions will be normalized regardless!"
            )

        # Normalize the Wannier functions to have unit self-overlap.
        for grid, self_overlap in zip(grids, self_overlaps):
            grid["data"] /= np.sqrt(self_overlap)

        # Compute gradients and Hessians of the Wannier functions on the
        # grid. Useful for remeshing.
        for i, grid in enumerate(grids):
            grid = grid.compute_derivative(gradient="gradient", scalars="data")
            grid = grid.compute_derivative(gradient="hessian", scalars="gradient")
            grids[i] = grid

        return grids

    def save_hdf5(self, filename: str | Path) -> None:
        """Save the Wannier90 orbital evaluator to an HDF5 file.

        Parameters
        ----------
        filename : str or Path
            The name of the HDF5 file to save the Wannier90 orbital
            evaluator to.

        """
        filename = Path(filename)

        with h5py.File(filename, "w") as f:
            f["dimensions"] = self._grids[0].dimensions
            f["points"] = self._grids[0].points
            f["data"] = np.array([grid["data"] for grid in self._grids])

    @classmethod
    def from_hdf5_file(cls, filename: str | Path) -> Self:
        """Load the Wannier90 orbital evaluator from an HDF5 file.

        Parameters
        ----------
        filename : str or Path
            The name of the HDF5 file to load the Wannier90 orbital
            evaluator from.

        """
        filename = Path(filename)

        with h5py.File(filename, "r") as f:
            dimensions = f["dimensions"][:]
            points = f["points"][:]
            data = f["data"][:]

            grids = []
            for i in range(data.shape[0]):
                grid = pv.StructuredGrid()
                grid.dimensions = dimensions
                grid.points = points
                grid["data"] = data[i]
                grid = grid.compute_derivative(gradient="gradient", scalars="data")
                grid = grid.compute_derivative(gradient="hessian", scalars="gradient")
                grids.append(grid)

        return cls(grids=grids)

    def _evaluate(self, positions: NDArray, field: str) -> NDArray:
        """Evaluate the Wannier functions at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the Wannier functions.
            Should be of shape (N, 3), N being the number of points.
        field : str
            The field name in the StructuredGrids to sample. This can be
            "data" for the Wannier function values, "gradient" for their
            gradients, or "hessian" for the Hessians.

        Returns
        -------
        values : NDArray
            The values sampled from the Wannier functions at the given
            positions. The shape of the output will be (M, N), where M
            is the total number of Wannier functions and N is the number
            of points.

        """
        points = pv.PolyData(positions)
        values = np.array([points.sample(grid)[field] for grid in self._grids])

        return values

    def evaluate_orbitals(self, positions: NDArray) -> NDArray:
        """Evaluate the Wannier orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbitals. Should be
            of shape (N, 3), N being the number of points.

        Returns
        -------
        orbitals : NDArray
            The evaluated orbitals at the given positions. The shape of
            the output will be (M, N), where M is the total number of
            orbitals and N is the number of points.

        """
        return self._evaluate(positions, field="data")

    def evaluate_orbital_gradients(self, positions: NDArray) -> NDArray:
        """Evaluate the gradients of the Wannier orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital gradients.
            Should be of shape (N, 3), N being the number of points.

        Returns
        -------
        gradients : NDArray
            The evaluated orbital gradients at the given positions. The
            shape of the output will be (M, N, 3), where M is the total
            number of orbitals, N is the number of points, and 3
            corresponds to the x, y, z components of the gradient.

        """
        return self._evaluate(positions, field="gradient").reshape(
            -1, positions.shape[0], 3, order="F"
        )

    def evaluate_orbital_hessians(self, positions: NDArray) -> NDArray:
        """Evaluate the Hessians of the Wannier orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital Hessians.
            Should be of shape (N, 3), N being the number of points.

        Returns
        -------
        hessians : NDArray
            The evaluated orbital Hessians at the given positions. The
            shape of the output will be (M, N, 3, 3), where M is the
            total number of orbitals, N is the number of points, and
            3x3 corresponds to the Hessian matrix components.

        """
        return self._evaluate(positions, field="hessian").reshape(
            -1, positions.shape[0], 3, 3, order="F"
        )


class Wannier90DeviceOrbitalProjector(DeviceOrbitalProjector):

    def __init__(self, config: QuatrexConfig):
        """Initialize the Wannier90OrbitalProjections class."""
        super().__init__(config)

    @staticmethod
    def _init_orbital_evaluator(
        config: QuatrexConfig, structure: ase.Atoms
    ) -> Wannier90OrbitalEvaluator:
        """Initialize the real-space orbitals based on the configuration.

        Parameters
        ----------
        config : QuatrexConfig
            The main quatrex configuration.
        structure : ase.Atoms
            The atomic structure for which the orbitals are to be
            initialized.

        Returns
        -------
        orbital_evaluator : Wannier90OrbitalEvaluator
            An instance of the Wannier90OrbitalEvaluator class, which
            can be used to evaluate the Wannier orbitals on a real-space
            grid.

        """
        if not config.device.construct_from_unit_cell:
            raise ValueError(
                "Wannier90 orbital projections require the device to be "
                "constructed from a unit cell. Please set "
                "`device.construct_from_unit_cell = True` in the configuration."
            )
        # Try to recover the orbital evaluator from a saved HDF5 file.
        try:
            orbital_evaluator = Wannier90OrbitalEvaluator.from_hdf5_file(
                config.input_dir / "wannier90_evaluator.h5"
            )
        except FileNotFoundError:
            warnings.warn(
                "No saved Wannier90 orbital evaluator found. "
                "Attempting to load from the specified Wannier90 directory."
            )
            orbital_evaluator = Wannier90OrbitalEvaluator.from_directory(
                path=config.pre_process.wannier90_dir,
                supercell=config.pre_process.wannier90_supercell,
            )
            orbital_evaluator.save_hdf5(config.input_dir / "wannier90_evaluator.h5")

        return orbital_evaluator

    def get_initial_mesh(
        self, options: RemeshingOptions | None = None
    ) -> pv.UnstructuredGrid:
        """Get an initial mesh guess for the orbital projections.

        Parameters
        ----------
        options : RemeshingOptions, optional
            Options concerning the remeshing for orbital projection. If
            not provided, the default options will be used.

        Returns
        -------
        initial_mesh : pv.UnstructuredGrid
            The initial mesh to be used for remeshing.

        """
        options = options or DEFAULT_REMESHING_OPTIONS

        # Use the first grid for remeshing. All grids should have the
        # same shape and spacing.
        grid: pv.StructuredGrid = self.orbital_evaluator._grids[0]

        # Downsample to the initial grid spacing.
        spacing = np.array(grid.bounds_size) / np.array(grid.dimensions)
        rate = (options.initial_spacing / spacing).astype(int)

        # Get the grid dimensions.
        nx, ny, nz = grid.dimensions
        initial_mesh = grid.extract_subset(
            (0, nx - 1, 0, ny - 1, 0, nz - 1), rate=rate
        ).triangulate()

        return initial_mesh

    def assemble_projections(
        self, mesh_nodes: NDArray
    ) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix]]:
        """Evaluate the Wannier orbitals on the given device mesh.

        Parameters
        ----------
        mesh_nodes : NDArray
            A 2D array of shape (N, 3) containing the coordinates of the
            mesh nodes in real space, where N is the number of nodes.

        Returns
        -------
        projections : sparse.csr_matrix
            A sparse matrix containing the evaluated Wannier orbitals on
            the device mesh. Each row corresponds to a point in the
            device mesh, and each column corresponds to a Wannier
            orbital.
        gradient_matrices : list of sparse.csr_matrix
            A list of three sparse matrices, each corresponding to the
            gradient of the Wannier orbitals in the x, y, and z
            directions, respectively. Each matrix has the same shape as
            `projections`, with rows corresponding to mesh nodes and
            columns corresponding to Wannier orbitals.

        """
        transport_ind = "abc".index(self.config.device.transport_direction)
        num_repetitions = (
            self.config.device.num_transport_cells
            * self.config.device.neighbor_cell_cutoff[transport_ind]
        )
        offsets = [
            i * self.structure.get_cell()[transport_ind] for i in range(num_repetitions)
        ]

        projections = [
            sparse.csr_matrix(
                self.orbital_evaluator.evaluate_orbitals(mesh_nodes - offset)
            )
            for offset in offsets
        ]

        gradient_projections = [
            self.orbital_evaluator.evaluate_orbital_gradients(mesh_nodes - offset)
            for offset in offsets
        ]
        gradient_matrices = []
        for i in range(3):
            matrix = sparse.vstack(
                [
                    sparse.csr_matrix(gradient[..., i])
                    for gradient in gradient_projections
                ],
                format="csr",
            )
            gradient_matrices.append(matrix)

        return sparse.vstack(projections, format="csr"), gradient_matrices
