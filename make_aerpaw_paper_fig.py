# -*- coding: utf-8 -*-
"""Paper-grade figure for the real-measured AERPAW ds32 channel validation.

Outputs paper/figs/aerpaw_measured.png (double-column, 300 dpi):
  (a) measured path loss vs 3D distance for the four flights, with the
      38.901 ULA-Los model curve and the free-space reference;
  (b) residual (measured - model) histogram with the fitted Gaussian.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import channel_3gpp as ch3g

F_HZ = 3.4e9
# Historical hard-coded path from the original ds32 run.  Kept ONLY as a
# documented default; override with --csv-dir (see aerpaw_data/README.md).
# AUTHOR TODO: the AERPAW CSVs live in the 17 GB Dryad package
# (doi:10.5061/dryad.2ngf1vj4r) and are not bundled in this repository.
DEFAULT_CSV_DIR = r"E:/AERPAW/ds32/data"
OUT = Path(__file__).resolve().parent / "paper" / "figs" / "aerpaw_measured.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
})

COLORS = {"f10": "#1f77b4", "f11": "#ff7f0e",
          "f12": "#2ca02c", "f13": "#d62728"}
LINEAR = ("f10", "f11")     # repeated linear tracks
ZIGZAG = ("f12", "f13")     # repeated zigzag tracks


def model_pl(d3d, h_uav, h_tx=12.0):
    n = len(d3d)
    d2 = np.sqrt(np.maximum(d3d ** 2 - (h_uav - h_tx) ** 2, 1.0))
    uav = np.stack([np.zeros(n), d2, np.full(n, h_uav)], axis=1)
    user = np.stack([np.zeros(n), np.zeros(n), np.full(n, h_tx)], axis=1)
    return ch3g.path_loss_db(F_HZ, uav, user, np.ones(n, dtype=bool))


def main():
    ap = argparse.ArgumentParser(
        description="Paper-grade AERPAW measured-channel figure (fig:aerpaw).")
    ap.add_argument(
        "--csv-dir", type=str, default=DEFAULT_CSV_DIR,
        help=("directory containing f10.csv..f13.csv (AERPAW ds32). "
              f"Default is the original local path ({DEFAULT_CSV_DIR}); "
              "AUTHOR TODO: point this at your extracted Dryad CSVs."))
    args = ap.parse_args()
    csv_dir = Path(args.csv_dir)
    for name in sorted(COLORS):
        if not (csv_dir / f"{name}.csv").exists():
            raise SystemExit(
                f"Missing {csv_dir / (name + '.csv')}. Download the 17 GB Dryad "
                "package (doi:10.5061/dryad.2ngf1vj4r), extract "
                "f10.csv..f13.csv, then pass --csv-dir <extracted_dir>. "
                "See aerpaw_data/README.md.")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))

    # ---------------- panel (a): PL vs distance ----------------
    ax = axes[0]
    d_all, pl_all, h_mean = [], [], []
    for name in sorted(COLORS):
        d = np.genfromtxt(csv_dir / f"{name}.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
        d3d = np.asarray(d["uav2ground_dist"], float)
        pl = np.asarray(d["path_loss_db"], float)
        h = np.asarray(d["rx_altitude_agl_m"], float).mean()
        d_all.append(d3d); pl_all.append(pl); h_mean.append(h)
        step = 4          # decimate for legibility
        lab = (f"{name} (linear)" if name in LINEAR else f"{name} (zigzag)")
        ax.scatter(d3d[::step], pl[::step], s=1.0, alpha=0.18, rasterized=True,
                   color=COLORS[name], label=lab)
    d_all = np.concatenate(d_all); pl_all = np.concatenate(pl_all)
    h_bar = float(np.mean(h_mean))

    dd = np.linspace(55.0, d_all.max() * 1.02, 400)
    ax.plot(dd, model_pl(dd, h_bar), color="k", lw=1.5,
            label="38.901 ULA-Los (model)")
    ax.plot(dd, 20 * np.log10(4 * np.pi * dd * F_HZ / 299792458.0),
            color="0.45", lw=0.9, ls="--", label="free space")
    ax.set_xscale("log")
    ax.set_xlim(50, d_all.max() * 1.05)
    ax.set_xlabel("3D distance $d_{3\\mathrm{D}}$ [m]")
    ax.set_ylabel("Path loss [dB]")
    ax.set_title("(a) Measured path loss, AERPAW 3.4 GHz campaign")
    ax.legend(loc="upper left", frameon=True, framealpha=0.85,
              handletextpad=0.4, borderpad=0.3)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    # ---------------- panel (b): residual histogram ----------------
    ax = axes[1]
    h_mean_arr = np.concatenate([
        np.full(len(np.asarray(np.genfromtxt(csv_dir / f"{n}.csv", delimiter=",",
                names=True, dtype=None, encoding="utf-8")["uav2ground_dist"],
                float)),
                np.asarray(np.genfromtxt(csv_dir / f"{n}.csv", delimiter=",",
                names=True, dtype=None, encoding="utf-8")["rx_altitude_agl_m"],
                float).mean())
        for n in sorted(COLORS)])
    # model PL per sample with per-flight mean altitude
    d2 = np.sqrt(np.maximum(d_all ** 2 - (h_mean_arr - 12.0) ** 2, 1.0))
    uav = np.stack([np.zeros(len(d2)), d2, h_mean_arr], axis=1)
    user = np.stack([np.zeros(len(d2)), np.zeros(len(d2)),
                     np.full(len(d2), 12.0)], axis=1)
    pl_model = ch3g.path_loss_db(F_HZ, uav, user, np.ones(len(d2), dtype=bool))
    err = pl_all - pl_model
    bias = float(err.mean()); rmse = float(np.sqrt(np.mean(err ** 2)))
    sd = float(err.std())

    ax.hist(err, bins=90, range=(-25, 25), density=True,
            color="#9ecae1", edgecolor="#3182bd", lw=0.2)
    xs = np.linspace(-25, 25, 500)
    ax.plot(xs, np.exp(-xs ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
            "r--", lw=1.1, label=f"$\\mathcal{{N}}(0,\\,{sd:.2f})$")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Measured $-$ model path loss [dB]")
    ax.set_ylabel("Density")
    ax.set_title("(b) Residual over 136,987 samples")
    ax.text(0.03, 0.96, f"bias $={bias:+.2f}$ dB\nRMSE $={rmse:.2f}$ dB",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlim(-25, 25)

    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"bias={bias:+.3f} dB  rmse={rmse:.3f} dB  std={sd:.3f}  n={len(err)}")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
