# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Methods for parsing Wannier90 files related to the real-space
orbital representations, such as UNK, CHK, and XCrySDen files."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from qttools import NDArray


class FortranRecordReader:
    """A small reader for Unformatted Fortran Files.

    These files are written with 4-byte record markers, which are used
    to indicate the length of each record.

    This is very similar to the FortranFile reader in scipy.io, with
    some stuff tacked on to make it easier to read things like strings
    and arrays of different types, but also expose seek and tell
    functionality.

    Paramameters
    ----------
    path : str or Path
        Path to the Fortran unformatted file to read.

    """

    record_marker_dtype = np.dtype("<u4")
    int_dtype = np.dtype("<i4")
    real_dtype = np.dtype("<f8")
    complex_dtype = np.dtype("<c16")

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
        head = np.fromfile(self._f, dtype=self.record_marker_dtype, count=1)
        if head.size == 0:
            raise EOFError("Unexpected end of file")
        nbytes = int(head[0])
        data = self._f.read(nbytes)
        tail = np.fromfile(self._f, dtype=self.record_marker_dtype, count=1)
        if tail.size == 0 or int(tail[0]) != nbytes:
            raise OSError(
                "Record length mismatch. Check fortran unformatted file "
                "endianness and record marker size "
                f"(assuming {self.record_marker_dtype.itemsize} bytes)."
            )
        return data

    def read_string(self) -> str:
        """Read a string from a Fortran unformatted record.

        Returns
        -------
        string : str
            The string read from the record.

        """
        return self.read_record().decode("ascii", errors="replace").strip()

    def _read_record_as_array(self, dtype: np.dtype, shape: tuple | None) -> NDArray:
        """Read a Fortran unformatted record and return it as a numpy array.

        Parameters
        ----------
        dtype : np.dtype
            The data type of the array to be returned.

        Returns
        -------
        arr : NDArray
            A numpy array of the specified dtype read from the record.

        """
        arr = np.frombuffer(self.read_record(), dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape, order="F")
        return arr

    def read_ints(self, shape: tuple | None = None) -> int:
        """Read integers from a Fortran unformatted record.

        Parameters
        ----------
        shape : tuple, optional
            The shape to which the array should be reshaped. If None,
            the array will be returned as a fortran-ordered 1D array.

        Returns
        -------
        ints : NDArray
            A numpy array of integers read from the record.

        """
        return self._read_record_as_array(self.int_dtype, shape)

    def read_reals(self, shape: tuple | None = None) -> NDArray:
        """Read real numbers from a Fortran unformatted record.

        Parameters
        ----------
        shape : tuple, optional
            The shape to which the array should be reshaped. If None,
            the array will be returned as a fortran-ordered 1D array.

        Returns
        -------
        reals : NDArray
            A numpy array of real numbers read from the record.

        """
        return self._read_record_as_array(self.real_dtype, shape)

    def read_complex(self, shape: tuple | None = None) -> NDArray:
        """Read complex numbers from a Fortran unformatted record.

        Parameters
        ----------
        shape : tuple, optional
            The shape to which the array should be reshaped. If None,
            the array will be returned as a fortran-ordered 1D array.

        Returns
        -------
        complex : NDArray
            A numpy array of complex numbers read from the record.

        """
        return self._read_record_as_array(self.complex_dtype, shape)

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


class UNKReader:
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
    better than reading from something like a WAVECAR. Another advantage
    is that this format is the universal format for wannier90, so this
    can be used with any DFT code that interfaces with wannier90.

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
            self._f = FortranRecordReader(self.path)
            head = self._f.read_ints()

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
            data = self._f.read_complex(shape=self.grid_shape)
            if data.size != self._num_points:
                raise OSError(f"Expected {self._num_points} points, got {data.size}.")
            bands[band_ind] = data
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
        # Read header line.
        self._f.readline()
        raw_bytes = np.loadtxt(self._f)
        data = raw_bytes[:, 0] + 1j * raw_bytes[:, 1]
        # Fortran order, so the first grid index varies fastest.
        data = data.reshape(self.num_bands, *self.grid_shape[::-1])
        # Reorder to (band, x, y, z).
        return data.transpose(0, 3, 2, 1)

    def read_bands(self) -> NDArray:
        """Read all bands from the UNK file.

        Returns
        -------
        bands : NDArray
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


def load_unk_data(
    path: str | Path,
    num_kpoints: int | None = None,
    num_bands: int | None = None,
    grid_shape: tuple[int, int, int] | None = None,
) -> NDArray:
    """Load all UNK file data from a directory.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing the UNK files.

    Returns
    -------
    u_nk : NDArray
        A complex64 array of shape (num_bands, num_kpoints, *grid_shape)
        containing the periodic part of the Bloch functions for all
        bands at all k-points.

    """

    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"'{path}' is not a directory")

    num_kpoints = len(list(path.glob("UNK?????.1")))

    with UNKReader(path / "UNK00001.1") as unk_reader:
        grid_shape = unk_reader.grid_shape
        num_bands = unk_reader.num_bands

    t0 = time.perf_counter()
    print(f"Loading {num_kpoints} UNK files...")
    u_nk = np.empty(
        (num_bands, num_kpoints, *grid_shape),
        dtype=np.complex64,
    )
    for kpoint_ind in range(1, num_kpoints + 1):
        with UNKReader(path / f"UNK{kpoint_ind:05d}.1") as unk_reader:
            u_nk[:, kpoint_ind - 1] = unk_reader.read_bands()

    t1 = time.perf_counter()
    print(f"Loaded {num_kpoints} UNK files in {t1 - t0:.2f} seconds.")

    return u_nk


@dataclass(frozen=True)
class CHKData:
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

        with FortranRecordReader(path) as f:
            header = f.read_string()
            num_bands = f.read_ints().item()
            num_exclude_bands = f.read_ints().item()
            exclude_bands = f.read_ints()
            real_lattice = f.read_reals((3, 3))
            reciprocal_lattice = f.read_reals((3, 3))
            num_kpoints = f.read_ints().item()
            kpoint_grid = f.read_ints()
            kpoints = f.read_reals((3, num_kpoints)).T
            num_neighbors = f.read_ints().item()
            num_wann = f.read_ints().item()
            checkpoint = f.read_string()

            disentangled = bool(f.read_ints())
            if disentangled:
                omega_invariant = f.read_reals().item()
                disentangle_window = f.read_ints((num_bands, num_kpoints)).astype(bool)
                num_bands_disentangled = f.read_ints()
                u_mnk_opt = f.read_complex((num_bands, num_wann, num_kpoints))

            u_mnk = f.read_complex((num_wann, num_wann, num_kpoints))
            m_mn = f.read_complex((num_wann, num_wann, num_neighbors, num_kpoints))
            wannier_centres = f.read_reals((3, num_wann)).T
            wannier_spreads = f.read_reals()

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
    The `seedname.xsf` file is a text file containing the coordinates of
    the atoms in the unit cell, as well as the unit cell vectors.

    See the official documentation at
    http://www.xcrysden.org/doc/XSF.html for more information.

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
