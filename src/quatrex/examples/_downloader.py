# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import atexit
import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# 8KB chunk size for reading files.
CHUNK_SIZE = 8192


class FileCache:
    def __init__(self):
        REPO_ROOT = Path(__file__).resolve().parents[3]
        EXAMPLES_DIR = REPO_ROOT / "examples"
        self._file = EXAMPLES_DIR / ".file_cache.npy"

        # Load existing cache if present
        try:
            self.paths = np.load(self._file, allow_pickle=True).item()
        except (FileNotFoundError, EOFError):
            self.paths = {}

        # Register save on exit
        atexit.register(self.save)

    def save(self):
        np.save(
            self._file,
            self.paths,
            allow_pickle=True,
        )


_cache = FileCache()


def _compute_sha256(filename: Path) -> str:
    """Computes SHA256 checksum of a file.

    Parameters
    ----------
    filename : Path
        File to compute checksum for.

    Returns
    -------
    str
        SHA256 checksum as a hex string.

    """
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(
    url: str,
    target_dir: Path,
    sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Downloads and extracts an archive from a URL.

    Supports .zip and .tar.gz archives. If the file is not an archive,
    it is simply downloaded.

    Parameters
    ----------
    url : str
        URL of the file to download.
    target_dir : Path
        Directory to extract the contents into.
    sha256 : str | None
        Optional SHA256 checksum to verify the download.
    force : bool
        If True, forces re-download even if the file already exists.

    Returns
    -------
    str
        Path to the directory containing the extracted contents.

    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target_dir / Path(url).name

    if url not in _cache.paths.keys():
        _cache.paths[url] = str(tmp_path)

    # Download file if not already present.
    if not tmp_path.exists():
        # create symlink if file is already downloaded
        if url in _cache.paths.keys() and not force:
            cached_path = Path(_cache.paths[url])
            if cached_path.exists():
                print(f"Linking {cached_path} to {tmp_path} ...")
                tmp_path.symlink_to(cached_path)
            else:
                print(f"Cached file {cached_path} not found, downloading {url} ...")
                urllib.request.urlretrieve(url, tmp_path)
        else:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, tmp_path)

    # If checksum is given, verify it.
    if sha256:
        digest = _compute_sha256(tmp_path)
        if digest != sha256:
            raise RuntimeError(f"Checksum mismatch for {tmp_path}")

    # Extract if the file is an archive.
    if tmp_path.suffix == ".zip":
        print(f"Extracting {tmp_path} ...")
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(target_dir)
        tmp_path.unlink()

    elif tmp_path.suffixes[-2:] == [".tar", ".gz"] or tmp_path.suffix == ".tgz":
        print(f"Extracting {tmp_path} ...")
        with tarfile.open(tmp_path, "r:gz") as tf:
            tf.extractall(target_dir)
        tmp_path.unlink()
    else:
        # Not an archive, nothing to extract.
        pass

    return target_dir
