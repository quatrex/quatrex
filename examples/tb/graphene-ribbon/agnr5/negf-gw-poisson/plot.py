"""LDOS, excess charge and potential of the negf-gw-poisson run (last SCSP step).

Usage: python plot.py [outputs_dir]  ->  ldos_charge_potential.png / .pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
NCELL, NORB, A = 40, 10, 4.26
VBM, CBM, EF, VD = -1.64, -1.23, -1.118, 0.1  # homogeneous GW ribbon, see README
XS, XD, GATE = 55.0, 115.0, (55.0, 115.0)
DONORS = 5e19 * 1e-24 * A * 20 * 20  # e per cell
L = NCELL * A

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def last(name):
    return max(int(f.stem.split("_")[-1]) for f in OUT.glob(f"{name}_*.npy") if f.stem.split("_")[-1].isdigit())


m, n = last("electron_density"), last("potential")
E = np.load(OUT / "electron_energies.npy").real
dE = E[1] - E[0]
spec = np.load(OUT / f"electron_density_{m}.npy").real + np.load(OUT / f"hole_density_{m}.npy").real
pot = np.load(OUT / f"potential_{n}.npy").real
q = np.load(OUT / f"real_space_charge_density_{last('real_space_charge_density')}.npy")
q = q[q != 0]
assert q.size == NCELL * NORB
x = np.array([float(l.split()[1]) for l in open(Path(__file__).parent / "../inputs/structure.xyz").readlines()[2:]])

ldos = spec.reshape(len(E), NCELL, NORB).sum(2)
qc, pc = q.reshape(NCELL, NORB).sum(1), pot.reshape(NCELL, NORB).mean(1)
edges = np.arange(NCELL + 1) * A
mid = edges[:-1] + A / 2

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "axes.facecolor": SURF,
    "figure.facecolor": SURF, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
cmap = LinearSegmentedColormap.from_list("seq", [SURF, "#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"])

fig, ax = plt.subplots(3, 1, figsize=(8, 10.5), sharex=True, gridspec_kw=dict(height_ratios=[2.6, 1, 1], hspace=0.22))
fig.subplots_adjust(left=0.1, right=0.87, top=0.925, bottom=0.06)

# LDOS
w = (E >= -2.3) & (E <= -0.5)
ee = np.r_[E[w] - dE / 2, E[w][-1] + dE / 2]
im = ax[0].pcolormesh(edges, ee, ldos[w], vmin=0, vmax=1.5, cmap=cmap, rasterized=True)
for lev, lab, dy, va in ((CBM, "flat-ribbon CBM + mean potential", -0.08, "top"),
                         (VBM, "flat-ribbon VBM + mean potential", 0.06, "bottom")):
    ax[0].step(edges, np.r_[pc, pc[-1]] + lev, where="post", color=INK, lw=1.1, ls=(0, (4, 2)))
    ax[0].text(1, pc[0] + lev + dy, lab, fontsize=7.5, va=va, color=INK)
for x0, x1, mu in ((0, XS, EF), (XD, L, EF - VD)):
    ax[0].plot([x0, x1], [mu, mu], color=ORANGE, lw=2)
ax[0].text(1, EF + 0.02, f"$E_F$ source {EF:.3f}", fontsize=8, va="bottom")
ax[0].text(L - 1, EF - VD + 0.02, f"$E_F$ drain {EF - VD:.3f}", fontsize=8, ha="right", va="bottom")
for xt, lab in ((XS / 2, "n-doped lead"), ((XS + XD) / 2, "gated channel (undoped)"), ((XD + L) / 2, "n-doped lead")):
    ax[0].text(xt, -0.53, lab, ha="center", va="top", fontsize=8, color=INK2)
ax[0].set(ylim=(-2.3, -0.5), ylabel="Energy (eV)", title="LDOS, last SCBA iteration (GW + phonon)")
box = ax[0].get_position()
cb = fig.colorbar(im, cax=fig.add_axes([0.885, box.y0, 0.018, box.height]), extend="max")
cb.set_label("States / (eV · unit cell)", color=INK2)
cb.outline.set_visible(False)

# charge
ax[1].bar(mid, qc, width=A - 0.5, color=BLUE, label="Electrons − holes")
ax[1].plot([0, XS], [DONORS] * 2, color=ORANGE, lw=2, label=f"Donors ({DONORS:.4f})")
ax[1].plot([XD, L], [DONORS] * 2, color=ORANGE, lw=2)
ax[1].set(ylim=(0, 0.115), ylabel="Charge (per unit cell)", title="Excess electrons per unit cell")
ax[1].legend(frameon=False, fontsize=8, loc="upper center", labelcolor=INK2)

# potential
ax[2].scatter(x, pot, s=7, color=BLUE, alpha=0.45, lw=0, label="Atoms")
ax[2].plot(mid, pc, color=ORANGE, lw=2, label="Unit-cell mean")
ylo = min(pot.min(), pc.min()) - 0.05
ax[2].plot(GATE, [ylo] * 2, color=INK, lw=3, solid_capstyle="butt")
ax[2].text(sum(GATE) / 2, ylo + 0.012, "gates (60 Å)", ha="center", va="bottom", fontsize=8, color=INK2)
ax[2].set(ylim=(ylo - 0.03, pot.max() + 0.05), xlim=(0, L), ylabel="Potential energy (eV)",
          xlabel="Position along the ribbon (Å)", title="Electrostatic potential energy, last SCSP step")
ax[2].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    for b in (XS, XD):
        a.axvline(b, color=MUTED, lw=0.8, ls=":")
for a in ax[1:]:
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
fig.suptitle("5-AGNR FET, NEGF-GW-Poisson: V$_G$ = 0 V, V$_D$ = 0.1 V", x=0.1, y=0.975, ha="left", fontsize=11, fontweight="bold")
fig.savefig("ldos_charge_potential.png", dpi=160)
fig.savefig("ldos_charge_potential.pdf")
print("barrier (cell mean)", (pc.max() - pc[:3].mean()).round(4), "at x =", mid[pc.argmax()].round(1),
      "| charge", qc.sum().round(3), "| atom potential range", pot.min().round(3), pot.max().round(3))
