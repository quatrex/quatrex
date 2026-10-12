# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""This module contains the necessary functions to process real-space
orbital data from Wannier90"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pyvista as pv

from qttools import NDArray

RECORD_MARKER_DTYPE = "<u4"
INT_DTYPE = "<i4"
COMPLEX_DTYPE = "<c16"  # complex(kind=dp): double precision complex
REAL_DTYPE = "<f8"  # double precision real


class FortranUnformattedFile:
    """A small reader for Unformatted Fortran Files.

    Paramameters
    ----------
    path : str or Path
        Path to the Fortran unformatted file to read.

    """

    def __init__(self, path: str | Path):
        """Initialize the FortranUnformattedFile reader."""
        self.path = Path(path)
        self._f = self.path.open("rb")

    def read_record(self) -> bytes:
        """Read a single Fortran unformatted record.

        Returns
        -------
        data : bytes
            The raw data of the record.

        Raises
        ------
        EOFError
            If the end of the file is reached unexpectedly.
        OSError
            If there is a problem reading the file or if the record length
            doesn't match the expected size.

        """
        marker_dtype = np.dtype(RECORD_MARKER_DTYPE)
        head = np.fromfile(self._f, dtype=marker_dtype, count=1)
        if head.size == 0:
            raise EOFError("Unexpected end of file")
        nbytes = int(head[0])
        data = self._f.read(nbytes)
        tail = np.fromfile(self._f, dtype=marker_dtype, count=1)
        if tail.size == 0 or int(tail[0]) != nbytes:
            raise OSError(
                "Record length mismatch. Check fortran unformatted file endianness and "
                "RECORD_MARKER_DTYPE (4 vs 8 bytes)."
            )
        return data

    def seek(self, offset: int) -> int:
        """Seek to a specific byte offset in the file."""
        return self._f.seek(offset)

    def tell(self) -> int:
        """Return the current byte offset in the file."""
        return self._f.tell()

    def close(self):
        """Close the file."""
        self._f.close()

    def __enter__(self) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, *exct):
        """Exit the runtime context and close the file."""
        self.close()


class UNKFile:
    """Reader for wannier90's UNK file format.

    Each UNK file corresponds to a single k-point and spin channel, and
    contains the periodic part of the Bloch functions on a real-space
    grid for all bands at that k-point.

    These files are written in unformatted Fortran with 4-byte record
    markers. The first record contains the grid dimensions and metadata,
    while subsequent records contain the complex values of the periodic
    part of the Bloch functions for each band in Fortran order over the
    grid.

    The UNK files are written for the full Monkhorst-Pack grid used in
    wannierization, so no IBZ unfolding is needed, which makes this
    better than reading from something like a WAVECAR.

    Parameters
    ----------
    path : str or Path
        Path to the UNK file to read.

    """

    def __init__(self, path: str | Path):
        """Initialize the UNKFile reader."""
        self.path = Path(path)

        self.unformatted = True
        try:
            self._f = FortranUnformattedFile(self.path)
            head = np.frombuffer(self._f.read_record(), dtype=INT_DTYPE)

        except (OSError, EOFError):
            if self._f is not None:
                self._f.close()

            # Try again and see if we can read it as a formatted file.
            self.unformatted = False
            try:
                self._f = self.path.open("r")
                head = self._f.readline().split()
            except (OSError, UnicodeDecodeError) as e:
                raise OSError(
                    f"Failed to read UNK file '{self.path}' as either "
                    f"unformatted or formatted. Error: {e}"
                )

        self._header_end = self._f.tell()

        *self.grid_shape, self.kpoint_ind, self.num_bands = map(int, head)
        self._num_points = np.prod(self.grid_shape)
        # Number of bytes per data record. Data bytes sandwiched between
        # 4-byte record markers, so add 8 bytes for the markers.
        self._num_record_bytes = 4 + self._num_points * 16 + 4

    def _read_bands_unformatted(self) -> NDArray:
        """Read the Bloch function from an unformatted UNK file.

        Returns
        -------
        NDArray
            A complex128 array of shape (num_bands, *grid_shape)
            containing the periodic part of the Bloch functions for all
            bands.

        Raises
        ------
        OSError
            If there is a problem reading the file or if the number of
            points read does not match the expected number of grid
            points.

        """
        self._f.seek(self._header_end)
        bands = np.empty((self.num_bands, *self.grid_shape), dtype=np.complex128)
        for band_ind in range(self.num_bands):
            raw_bytes = self._f.read_record()
            data = np.frombuffer(raw_bytes, dtype=COMPLEX_DTYPE)
            if data.size != self._num_points:
                raise OSError(
                    f"Expected {self._num_points} points, got {data.size}; check "
                    "COMPLEX_DTYPE (single vs double precision) or "
                    "RECORD_MARKER_DTYPE at the top of this file."
                )
            # The first grid index varies fastest, meaning data stored
            # in Fortran order.
            bands[band_ind] = data.reshape(*self.grid_shape, order="F")
        return bands

    def _read_bands_formatted(self) -> NDArray:
        """Read the Bloch function from a formatted UNK file.

        Returns
        -------
        NDArray
            A complex128 array of shape (num_bands, *grid_shape)
            containing the periodic part of the Bloch functions for all
            bands.

        Raises
        ------
        OSError
            If there is a problem reading the file or if the number of
            points read does not match the expected number of grid points.

        """
        self._f.seek(0)
        self._f.readline()  # Read header line.
        raw_bytes = np.loadtxt(self._f)
        data = raw_bytes[:, 0] + 1j * raw_bytes[:, 1]
        # The first grid index varies fastest, meaning data stored in
        # Fortran order.
        data = data.reshape(self.num_bands, *self.grid_shape[::-1])
        return data.transpose(0, 3, 2, 1)  # Reorder to (band, x, y, z).

    def read_bands(self) -> NDArray:
        """Read all bands from the UNK file.

        Returns
        -------
        NDArray
            A complex128 array of shape (num_bands, *grid_shape)
            containing the periodic part of the Bloch functions for all
            bands.

        """
        if self.unformatted:
            return self._read_bands_unformatted()

        return self._read_bands_formatted()

    def close(self):
        self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc):
        self.close()


class UNKFiles:
    """A collection of UNK files.

    This class provides a convenient way to manage and access multiple
    UNK files corresponding to different k-points.

    Parameters
    ----------
    path : str or Path
        The directory where the UNK files are located.

    """

    def __init__(self, path: str | Path):
        """Initialize the UNKFiles collection."""
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError(f"'{path}' is not a directory")

        self.num_kpoints = len(list(self.path.glob("UNK?????.1")))

        with UNKFile(self.path / "UNK00001.1") as unk:
            self.grid_shape = unk.grid_shape
            self.num_bands = unk.num_bands

    def load(self) -> NDArray:
        """Load all data stored in the UNK files into a single array.

        Returns
        -------
        u_nk : NDArray
            A complex128 array of shape (num_bands, num_kpts,
            *grid_shape) containing the periodic part of the Bloch
            functions for all bands and k-points.

        """
        u_nk = np.empty(
            (self.num_bands, self.num_kpoints, *self.grid_shape),
            dtype=np.complex128,
        )
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self._load_kpoint, range(1, self.num_kpoints + 1))
        for kpoint_ind, bands in enumerate(results, start=1):
            u_nk[:, kpoint_ind - 1] = bands
        # for kpoint_ind in track(
        #     range(1, self.num_kpoints + 1),
        #     description="Loading UNK files",
        # ):
        #     with UNKFile(self.path / f"UNK{kpoint_ind:05d}.1") as unk_file:
        #         u_nk[:, kpoint_ind - 1] = unk_file.read_bands()

        return u_nk

    def _load_kpoint(self, kpoint_ind: int) -> NDArray:
        """Load the data for a specific k-point from the UNK files.

        Parameters
        ----------
        kpoint_ind : int
            The index of the k-point to load (1-based).

        Returns
        -------
        NDArray
            A complex128 array of shape (num_bands, *grid_shape)
            containing the periodic part of the Bloch functions for all
            bands at the specified k-point.

        """
        with UNKFile(self.path / f"UNK{kpoint_ind:05d}.1") as unk_file:
            return unk_file.read_bands()


@dataclass
class CHKFile:
    """A class to represent contents of Wannier90 checkpoint files."""

    header: str
    """Header string containing metadata or comments."""
    num_bands: int
    """Number of bands in the system."""
    num_exclude_bands: int
    """Number of bands to exclude from Wannierization."""
    exclude_bands: NDArray
    """Array of excluded band indices."""
    real_lattice: NDArray
    """Real-space lattice vectors."""
    reciprocal_lattice: NDArray
    """Reciprocal-space lattice vectors."""
    num_kpoints: int
    """Number of k-points."""
    kpoint_grid: tuple[int, int, int]
    """Monkhorst-Pack grid dimensions."""
    kpoints: NDArray
    """Array of reciprocal-space k-point coordinates."""
    num_neighbors: int
    """Number of nearest neighbors considered."""
    num_wann: int
    """Number of Wannier functions."""
    checkpoint: str
    """Checkpoint string."""

    disentangled: bool
    """Whether disentanglement was performed."""
    omega_invariant: float | None
    """Invariant part of the spread functional."""
    disentangle_window: NDArray | None
    """Array masking the bands included in the disentangle window."""
    num_bands_disentangled: NDArray | None
    """Number of bands in the disentanglement window."""
    u_mnk_opt: NDArray | None
    """Unitary disentanglement matrix."""

    u_mnk: NDArray
    """Unitary transformation matrix for the Wannierization."""
    m_mn: NDArray
    """Overlap matrix between Wannier functions and Bloch states."""
    wannier_centres: NDArray
    """Wannier function centers in real space."""
    wannier_spreads: NDArray
    """Wannier function spreads."""

    @staticmethod
    def _read_complex_array(raw_bytes: bytes) -> NDArray:
        """Convert raw bytes to a complex128 numpy array."""
        arr = np.frombuffer(raw_bytes, dtype=REAL_DTYPE)
        return arr[0::2] + 1j * arr[1::2]

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Read a CHK file and return a CHKFile instance.

        Parameters
        ----------
        path : str or Path
            Path to the CHK file to read.

        Returns
        -------
        CHKFile
            An instance of the CHKFile class containing the data from
            the CHK file.

        """
        path = Path(path)

        with FortranUnformattedFile(path) as f:
            header = f.read_record().decode("ascii", errors="replace").strip()
            num_bands = int(np.frombuffer(f.read_record(), dtype=INT_DTYPE)[0])
            num_exclude_bands = int(np.frombuffer(f.read_record(), dtype=INT_DTYPE)[0])

            exclude_bytes = f.read_record()
            if num_exclude_bands > 0:
                exclude_bands = np.frombuffer(exclude_bytes, dtype=INT_DTYPE).copy()
            else:
                exclude_bands = np.array([], dtype=int)

            real_lattice = (
                np.frombuffer(f.read_record(), dtype=REAL_DTYPE)
                .reshape(3, 3, order="F")
                .copy()
            )
            reciprocal_lattice = (
                np.frombuffer(f.read_record(), dtype=REAL_DTYPE)
                .reshape(3, 3, order="F")
                .copy()
            )

            num_kpoints = int(np.frombuffer(f.read_record(), dtype=INT_DTYPE)[0])

            kpoint_grid = np.frombuffer(f.read_record(), dtype=INT_DTYPE).copy()

            kpoints = (
                np.frombuffer(f.read_record(), dtype=REAL_DTYPE)
                .reshape(3, num_kpoints, order="F")
                .T.copy()
            )

            num_neighbors = int(np.frombuffer(f.read_record(), dtype=INT_DTYPE)[0])

            num_wann = int(np.frombuffer(f.read_record(), dtype=INT_DTYPE)[0])

            checkpoint = f.read_record().decode("ascii", errors="replace").strip()

            disentangled = bool(int(np.frombuffer(f.read_record(), dtype=INT_DTYPE)[0]))

            if disentangled:
                omega_invariant = float(
                    np.frombuffer(f.read_record(), dtype=REAL_DTYPE)[0]
                )

                disentangle_window = (
                    np.frombuffer(f.read_record(), dtype=INT_DTYPE)
                    .reshape(num_bands, num_kpoints, order="F")
                    .astype(bool)
                    .copy()
                )

                num_bands_disentangled = np.frombuffer(
                    f.read_record(), dtype=INT_DTYPE
                ).copy()

                u_mnk_opt = (
                    CHKFile._read_complex_array(f.read_record())
                    .reshape(num_bands, num_wann, num_kpoints, order="F")
                    .copy()
                )

            u_mnk = (
                CHKFile._read_complex_array(f.read_record())
                .reshape(num_wann, num_wann, num_kpoints, order="F")
                .copy()
            )

            m_mn = (
                CHKFile._read_complex_array(f.read_record())
                .reshape(num_wann, num_wann, num_neighbors, num_kpoints, order="F")
                .copy()
            )

            wannier_centres = (
                np.frombuffer(f.read_record(), dtype=REAL_DTYPE)
                .reshape(3, num_wann, order="F")
                .T.copy()
            )

            wannier_spreads = np.frombuffer(f.read_record(), dtype=REAL_DTYPE).copy()

        return cls(
            header=header,
            num_bands=num_bands,
            num_exclude_bands=num_exclude_bands,
            exclude_bands=exclude_bands,
            real_lattice=real_lattice,
            reciprocal_lattice=reciprocal_lattice,
            num_kpoints=num_kpoints,
            kpoint_grid=tuple(kpoint_grid),
            kpoints=kpoints,
            num_neighbors=num_neighbors,
            num_wann=num_wann,
            checkpoint=checkpoint,
            disentangled=disentangled,
            omega_invariant=omega_invariant if disentangled else None,
            disentangle_window=disentangle_window if disentangled else None,
            num_bands_disentangled=num_bands_disentangled if disentangled else None,
            u_mnk_opt=u_mnk_opt if disentangled else None,
            u_mnk=u_mnk,
            m_mn=m_mn,
            wannier_centres=wannier_centres,
            wannier_spreads=wannier_spreads,
        )


class Wannier90Orbitals:
    """A class to represent the orbitals data from Wannier90.

    This class encapsulates the data from both the UNK files and the CHK
    file, providing a convenient interface to access the orbitals and
    related information.

    Parameters
    ----------
    u_nk : NDArray
        A complex128 array of shape (num_bands, num_kpoints,
        *grid_shape) containing the periodic part of the Bloch functions
        for all bands and k-points.
    chk_file : CHKFile
        An instance of the CHKFile class containing the CHK file data.

    """

    def __init__(self, u_nk: NDArray, chk_file: CHKFile):
        """Initialize the Wannier90Orbitals class."""
        # Try to see whether norm-conserving pseudos were used to
        # generate the UNK files. If so, the self-overlap of the
        # periodic part of the Bloch functions should be constant over
        # the bands.
        self_overlap = np.einsum("nkxyz,nkxyz->nk", u_nk, np.conj(u_nk)).mean(axis=1)
        if not np.allclose(self_overlap, self_overlap[0], rtol=1e-3):
            print(
                "Self-overlap of Bloch functions is not constant over bands."
                "This suggests that the UNK files were generated with "
                "pseudopotentials that are not norm-conserving."
                "The Wannier functions will be normalized regardless!"
            )

        self._u_nk = u_nk
        self._chk = chk_file

    def assemble_orbitals(
        self, supercell: tuple[int, int, int] = (1, 1, 1)
    ) -> list[pv.StructuredGrid]:
        """Evaluate the Wannier functions on a real-space grid.

        Similar to what plot_wannier.F90 does, but returns a list of
        pyvista StructuredGrid objects instead of writing to .xsf/.cube
        files. The grid is evaluated over a supercell.

        Parameters
        ----------
        supercell : tuple of int, optional
            The size of the supercell in terms of home cells to evaluate
            the Wannier functions. Default is (1, 1, 1), which evaluates
            the Wannier functions only in the home cell.

        Returns
        -------
        wannier_grids : list of pv.StructuredGrid
            A list of pyvista StructuredGrid objects, each representing
            the real-space grid of a Wannier function evaluated over the
            specified supercell.

        """
        num_kpoints = self._chk.num_kpoints
        num_wann = self._chk.num_wann
        disentangled = self._chk.disentangled

        grid_shape = np.array(self._u_nk.shape[2:])

        # Prepare the real-space supercell grid, centering the the whole
        # thing around the home cell.
        lower_bounds = [-(s // 2) * n - 1 for s, n in zip(supercell, grid_shape)]
        upper_bounds = [((s + 1) // 2) * n - 1 for s, n in zip(supercell, grid_shape)]
        meshgrid_inds = (
            np.arange(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)
        )
        meshgrid = np.array(np.meshgrid(*meshgrid_inds, indexing="ij"))

        # Wrap meshgrid indices into the home cell to then use indexing
        # hacks when upscaling to the supercell.
        meshgrid_wrapped = [ni % n for ni, n in zip(meshgrid, grid_shape)]

        wannier_functions = np.zeros(
            (num_wann, *meshgrid[0].shape), dtype=np.complex128
        )

        for ik in range(num_kpoints):
            # Extract cell-periodic Bloch functions for all bands at the
            # current k-point.
            u_m = self._u_nk[:, ik].copy()
            if disentangled:
                # If we disentangled, we need to select the bands that
                # are included in the Wannierization and apply the
                # disentanglement unitary transformation.
                included_bands = np.flatnonzero(self._chk.disentangle_window[:, ik])
                u_mn_dis = self._chk.u_mnk_opt[
                    : self._chk.num_bands_disentangled[ik], :, ik
                ]
                u_m = np.einsum("mn,m...->n...", u_mn_dis, u_m[included_bands])

            # Apply the unitary transformation to get the mixed Bloch
            # functions in the home cell.
            u_mn = self._chk.u_mnk[..., ik]
            u_n = np.einsum("mn,m...->n...", u_mn, u_m)

            # Extend the Bloch functions to the supercell by indexing
            # into the home-cell-sized array using the wrapped meshgrid
            # indices.
            u_n_supercell = u_n[:, *meshgrid_wrapped]

            # Construct the phase factor for the current k-point and
            # meshgrid indices from the supercell meshgrid.
            exponent = (2j * np.pi) * np.einsum(
                "i,i...->...", self._chk.kpoints[ik] / grid_shape, meshgrid
            )

            # Multiply cell-periodic Bloch functions by the phase factor
            # and sum over k-points to get the Wannier functions in the
            # supercell.
            wannier_functions += u_n_supercell * np.exp(exponent)[None, ...]

        wannier_functions /= num_kpoints

        wannier_functions = self._fix_global_phase(wannier_functions)
        reality_ratios = self._reality_check(wannier_functions)
        if any(ratio > 1e-3 for ratio in reality_ratios):
            print(
                f"Warning: Reality check failed for some orbitals. Ratios: {reality_ratios}"
            )

        fractional_coords = np.stack(
            [ni / n for ni, n in zip(meshgrid, grid_shape)], axis=-1
        )
        cartesian_coords = fractional_coords @ self._chk.real_lattice

        grid = pv.StructuredGrid(
            cartesian_coords[..., 0],
            cartesian_coords[..., 1],
            cartesian_coords[..., 2],
        )

        grids = []
        for w in wannier_functions:
            g = grid.copy(deep=False)
            g["data"] = w.real.ravel(order="F")
            grids.append(g)

        self_overlap = self._compute_self_overlap(grids)
        # Check that the Wannier functions all share the same
        # self-overlap, which should be the case if the supercell is
        # large enough and the UNK files were generated with
        # norm-conserving pseudopotentials.
        # NOTE: It is not checked whether the self-overlap is 1, because
        # sometimes things are given in different units.
        if not np.allclose(self_overlap, self_overlap[0], rtol=1e-3):
            print(
                "Warning: Self-overlap of Wannier functions is not constant over orbitals."
                "This suggests that the supercell is too small or that the UNK files were "
                "generated with pseudopotentials that are not norm-conserving."
                "The Wannier functions will be normalized regardless!"
            )

        # Normalize the Wannier functions to have unit self-overlap.
        for g, overlap in zip(grids, self_overlap):
            g["data"] /= np.sqrt(overlap)

        return grids

    @staticmethod
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

    @staticmethod
    def _reality_check(
        wannier_functions: NDArray, threshold: float = 0.01
    ) -> list[float]:
        """Check that the Wannier functions are real.

        Parameters
        ----------
        wannier_functions : NDArray
            A complex128 array of shape (num_wann, *grid_shape)
            containing the Wannier functions.
        threshold : float, optional
            The threshold for the real part of the Wannier function.
            Only grid points where |Re| >= threshold are considered in
            the reality check. Default is 0.01.

        Returns
        -------
        list of float
            A list of maximum |Im|/|Re| ratios for each Wannier
            function.

        """
        ratios = []
        for iw in range(wannier_functions.shape[0]):
            w = wannier_functions[iw]
            mask = np.abs(w.real) >= threshold
            if mask.any():
                ratios.append(
                    float(np.max(np.abs(w.imag[mask]) / np.abs(w.real[mask])))
                )
            else:
                ratios.append(float("nan"))
        return ratios

    @staticmethod
    def _compute_self_overlap(wannier_grids: list[pv.StructuredGrid]) -> NDArray:
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

    @classmethod
    def from_directory(cls, path: str | Path) -> Self:
        """Load orbitals data from a directory containing UNK and CHK files.

        Parameters
        ----------
        path : str or Path
            Path to the directory containing the UNK and CHK files.

        Returns
        -------
        Wannier90Orbitals
            An instance of the Wannier90Orbitals class containing the
            loaded data.

        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"'{path}' is not a directory")

        unk_files = UNKFiles(path)

        # Find the chk file in the directory assuming there's only one.
        chk_files = list(path.glob("*.chk"))
        if not chk_files:
            raise FileNotFoundError("No chk file found in the directory")
        if len(chk_files) > 1:
            raise ValueError("Multiple chk files found in the directory")
        chk_file = CHKFile.from_file(chk_files[0])

        if not (unk_files.num_kpoints == chk_file.num_kpoints):
            raise ValueError(
                "Number of k-points in UNK files does not match number of k-points in CHK file"
            )

        return cls(u_nk=unk_files.load(), chk_file=chk_file)


def _parse_xsf_datagrid(lines: list[str]) -> dict:
    """Parses the DATAGRID_xD section of a `seedname.xsf` file.

    Parameters
    ----------
    lines : list of str
        Lines corresponding to the DATAGRID_xD section of the
        `seedname.xsf` file.

    Returns
    -------
    datagrid : dict
        A dictionary containing the parsed data from the DATAGRID_xD
        section of the `seedname.xsf` file.

    """
    lines = [line.strip() for line in lines if not line.startswith("END_")]

    dim = int(lines[0].split("_")[2][0])
    shape = tuple(map(int, lines[1].strip().split()))
    origin = np.array(list(map(float, lines[2].strip().split())))
    cell = np.zeros((dim, 3))
    for i in range(dim):
        cell[i] = list(map(float, lines[3 + i].strip().split()))

    data = np.array(list(map(float, " ".join(lines[3 + dim :]).split())))
    datagrid = {
        "shape": shape,
        "origin": origin,
        "cell": cell,
        "data": data,
    }
    return datagrid


def read_xsf(path: Path) -> dict:
    """Parses the contents of a `seedname.xsf` file.
    `
        This is a utility function to read the contents of a `seedname.xsf`
        file and return a dictionary containing the parsed data, mainly to
        compare against what wannier90 writes out in its .xsf files for
        verification purposes.

        Parameters
        ----------
        path : os.PathLike
            Path to `seedname.xsf`.

        Returns
        -------
        xsf : dict
            A dictionary containing the parsed data from the `seedname.xsf`
            file.

        Notes
        -----
        The `seedname.xsf` file is a text file containing the coordinates
        of the atoms in the unit cell, as well as the unit cell vectors.

        See the `official documentation
        <http://www.xcrysden.org/doc/XSF.html>`_ for more information.

    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Remove comment lines.
    lines = [
        line for line in lines if not line.strip().startswith("#") and line.strip()
    ]

    supported = {"CRYSTAL", "BEGIN_"}  # "END" does not need to be checked.
    unsupported = {"ATOMS", "MOLECULE", "POLYMER", "SLAB", "ANIMSTEPS"}
    keywords = supported | unsupported

    section_inds = []
    for ind, line in enumerate(lines):
        if any(name in line for name in keywords):
            section_inds += [ind]
        if any(name in line for name in unsupported):
            raise NotImplementedError(f"{line.strip()} not supported.")

    sections = [lines[i:j] for i, j in zip(section_inds, section_inds[1:] + [None])]

    xsf = {}
    for ind, section in enumerate(sections):
        if "BEGIN_BLOCK_DATAGRID" not in section[0]:
            continue

        key = section[1].strip()
        datagrid = _parse_xsf_datagrid(sections[ind + 1])
        xsf[key] = datagrid

    return xsf
