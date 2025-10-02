# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack_repo.builtin.build_systems.python import PythonPackage

from spack.package import *


class PyQuatrex(PythonPackage):
    """Quantum transport at the exascale and beyond."""

    homepage = "https://quatrex.github.io/quatrex/"
    url = "https://github.com/quatrex/quatrex.git"
    git = "https://github.com/quatrex/quatrex.git"

    maintainers("vetschn", "almaeder", "alexnick83")

    license("BSD-3-Clause")

    version("dev", branch="dev")
    version("main", branch="main")

    depends_on("python@3.13:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")

    depends_on("py-numpy@1.23.2:", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-numba", type=("build", "run"))
    depends_on("py-mpi4py", type=("build", "run"))
    depends_on("py-pydantic", type=("build", "run"))
    depends_on("py-matplotlib", type=("build", "run"))
    depends_on("py-threadpoolctl", type=("build", "run"))
    depends_on("py-typer", type=("build", "run"))
