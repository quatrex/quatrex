"""Plot potential and charge of the graphene SCSP run.

Usage:
    python plot_scsp.py [simulation_dir]

Reads from <simulation_dir>/outputs:
    potential_{i}.npy                   atom-resolved potential fed into
                                        transport at SCSP iteration i
    real_space_charge_density_{k}.npy   charge on the FEM mesh nodes from
                                        the transport run of iteration k-1
    real_space_potential_{k}.npy        Poisson solution on the mesh nodes

and writes <simulation_dir>/outputs/scsp_summary.png.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Device layout (Å), see quatrex_config.toml.
LEAD_LEFT_END = 14.0
LEAD_RIGHT_START = 34.0
GATE_X = (23.997 - 10.0, 23.997 + 10.0)
GATE_Z = -10.0
OXIDE_TOP_Z = -1.0


def load_series(outputs: Path, prefix: str) -> dict[int, np.ndarray]:
    """Loads `<prefix>_<int>.npy` files, keyed by the integer."""
    pattern = re.compile(rf"^{prefix}_(\d+)\.npy$")
    series = {}
    for file in outputs.iterdir():
        match = pattern.match(file.name)
        if match:
            series[int(match.group(1))] = np.load(file)
    return dict(sorted(series.items()))


def load_atoms(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    num_atoms = int(lines[0])
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in lines[2 : 2 + num_atoms]]
    )


def region_nodes(mesh: meshio.Mesh, name: str) -> np.ndarray:
    """Node indices of a physical region, as in quatrex's DeviceMesher.

    For "atoms" this reproduces `ElectrostaticSolver.atom_inds`, whose
    order matches the atom order in structure.xyz.
    """
    inds = [
        np.unique(block.data[cell_inds])
        for block, cell_inds in zip(mesh.cells, mesh.cell_sets[name])
        if cell_inds.size
    ]
    return np.unique(np.concatenate(inds))


def column_average(x: np.ndarray, values: np.ndarray, decimals: int = 2):
    """Averages atom values over atoms sharing the same x position."""
    keys = np.round(x, decimals)
    xs = np.unique(keys)
    return xs, np.array([values[keys == k].mean() for k in xs])


def shade_leads(ax):
    for lo, hi in [(ax.get_xlim()[0], LEAD_LEFT_END), (LEAD_RIGHT_START, ax.get_xlim()[1])]:
        ax.axvspan(lo, hi, color="0.9", zorder=0)
    ax.axvspan(*GATE_X, ymin=0, ymax=0.04, color="tab:orange", zorder=0)


def main():
    sim_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    outputs = sim_dir / "outputs"
    inputs = sim_dir / "../inputs"

    atoms = load_atoms(inputs / "structure.xyz")
    mesh = meshio.read(inputs / "device.msh")
    atom_nodes = region_nodes(mesh, "atoms")
    dist = np.linalg.norm(mesh.points[atom_nodes] - atoms, axis=1)
    if atom_nodes.size != len(atoms) or dist.max() > 1e-3:
        print(f"Warning: atom nodes do not match structure.xyz "
              f"(max distance {dist.max():.2e} Å)")
    # The .msh also holds duplicate, dangling vertex nodes; interpolate
    # only on the nodes of the volume mesh.
    volume_nodes = np.unique(
        np.concatenate([c.data.ravel() for c in mesh.cells if c.type == "tetra"])
    )

    potentials = load_series(outputs, "potential")
    charges = load_series(outputs, "real_space_charge_density")
    rs_potentials = load_series(outputs, "real_space_potential")
    if not potentials:
        sys.exit(f"No potential_*.npy in {outputs}")
    final_path = outputs / "potential_final.npy"
    converged = final_path.exists()

    print(f"{len(potentials)} potentials, {len(charges)} charge densities, "
          f"converged: {converged}")

    x = atoms[:, 0]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ax_pot, ax_rho, ax_conv, ax_map = axes.ravel()

    # (a) Atom potential along x, per iteration.
    n_pot = len(potentials)
    for j, (i, pot) in enumerate(potentials.items()):
        xs, ps = column_average(x, pot)
        last = j == n_pot - 1
        ax_pot.plot(xs, ps, color=cmap(j / max(n_pot - 1, 1)),
                    lw=2.2 if last else 1, alpha=1 if last else 0.6,
                    label=f"iter {i}" if (j == 0 or last) else None)
    ax_pot.set(xlabel="x (Å)", ylabel="potential (V)",
               title="(a) Potential at the atoms (column average)")
    shade_leads(ax_pot)
    ax_pot.legend()

    # (b) Atom charge along x, per iteration.
    n_rho = len(charges)
    for j, (k, rho) in enumerate(charges.items()):
        xs, qs = column_average(x, rho[atom_nodes])
        last = j == n_rho - 1
        ax_rho.plot(xs, qs, color=cmap(j / max(n_rho - 1, 1)),
                    lw=2.2 if last else 1, alpha=1 if last else 0.6,
                    label=f"iter {k - 1}" if (j == 0 or last) else None)
    ax_rho.axhline(0, color="k", lw=0.5)
    ax_rho.set(xlabel="x (Å)", ylabel="n - p (e / atom)",
               title="(b) Excess charge at the atoms (column average)")
    if charges:
        shade_leads(ax_rho)
        ax_rho.legend()

    # (c) Convergence: potential update and total excess charge.
    pot_list = list(potentials.values())
    dphi = [np.max(np.abs(b - a)) for a, b in zip(pot_list, pot_list[1:])]
    ax_conv.semilogy(range(1, len(dphi) + 1), dphi, "o-", label=r"max $|\Delta\phi|$ (mixed)")
    if rs_potentials:
        resid = [np.max(np.abs(rs[atom_nodes] - potentials[k - 1]))
                 for k, rs in rs_potentials.items() if k - 1 in potentials]
        ax_conv.semilogy(range(len(resid)), resid, "s--",
                         label=r"max $|\phi_\mathrm{Poisson} - \phi_\mathrm{in}|$")
    ax_conv.set(xlabel="SCSP iteration", ylabel="V",
                title="(c) Convergence" + ("" if converged else " (not converged)"))
    ax_conv.legend(loc="upper right")
    if charges:
        ax_q = ax_conv.twinx()
        total = [rho[atom_nodes].sum() for rho in charges.values()]
        ax_q.plot([k - 1 for k in charges], total, "^:", color="tab:red")
        ax_q.set_ylabel("total excess charge (e)", color="tab:red")

    # (d) Final Poisson potential in the x-z plane through the cell centre.
    if rs_potentials:
        k_last, phi = list(rs_potentials.items())[-1]
        interp = LinearNDInterpolator(mesh.points[volume_nodes], phi[volume_nodes])
        y0 = 0.5 * (mesh.points[:, 1].min() + mesh.points[:, 1].max())
        gx, gz = np.meshgrid(
            np.linspace(mesh.points[:, 0].min(), mesh.points[:, 0].max(), 300),
            np.linspace(mesh.points[:, 2].min(), mesh.points[:, 2].max(), 200),
        )
        gphi = interp(gx, np.full_like(gx, y0), gz)
        im = ax_map.pcolormesh(gx, gz, gphi, shading="auto", cmap="RdBu_r")
        ax_map.contour(gx, gz, gphi, levels=15, colors="k", linewidths=0.4)
        fig.colorbar(im, ax=ax_map, label="potential (V)")
        ax_map.axhline(0, color="k", lw=1.5)
        ax_map.axhline(OXIDE_TOP_Z, color="w", lw=0.8, ls="--")
        ax_map.plot(GATE_X, [GATE_Z, GATE_Z], color="tab:orange", lw=4)
        ax_map.set(xlabel="x (Å)", ylabel="z (Å)",
                   title=f"(d) Poisson potential, y = {y0:.2f} Å (iter {k_last - 1})")

    fig.suptitle(f"Graphene SCSP: {sim_dir.resolve().name}")
    out = outputs / "scsp_summary.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
