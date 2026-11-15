# -*- coding: utf-8 -*-
"""Per-flight and binned analysis of the real AERPAW ds32 validation.

Companion to validate_aerpaw.py: breaks the 136,987-point validation down by
flight, distance bin, and a one-slope (log-distance) fit, so the numbers that
go into the paper can be reported honestly.
"""
import argparse
import json
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
OUT_JSON = Path(__file__).resolve().parent / "aerpaw_validation_detail.json"
OUT_PNG = Path(__file__).resolve().parent / "aerpaw_validation_detail.png"

COLS = ("path_loss_db", "uav2ground_dist", "rx_altitude_agl_m",
        "tx_altitude_agl_m")


def load(csv):
    d = np.genfromtxt(csv, delimiter=",", names=True, dtype=None,
                      encoding="utf-8")
    return d


def model_pl(d3d, h_uav, h_tx):
    n = len(d3d)
    d2 = np.sqrt(np.maximum(d3d ** 2 - (h_uav - h_tx) ** 2, 1.0))
    uav = np.stack([np.zeros(n), d2, h_uav], axis=1)
    user = np.stack([np.zeros(n), np.zeros(n), np.full(n, h_tx)], axis=1)
    los = np.ones(n, dtype=bool)
    pl_los = ch3g.path_loss_db(F_HZ, uav, user, los)
    pl_nlos = ch3g.path_loss_db(F_HZ, uav, user, ~los)
    return pl_los, pl_nlos


def main():
    ap = argparse.ArgumentParser(
        description="Per-flight / binned analysis of the real AERPAW ds32 validation.")
    ap.add_argument(
        "--csv-dir", type=str, default=DEFAULT_CSV_DIR,
        help=("directory containing f10.csv..f13.csv (AERPAW ds32). "
              f"Default is the original local path ({DEFAULT_CSV_DIR}); "
              "AUTHOR TODO: point this at your extracted Dryad CSVs."))
    args = ap.parse_args()
    csv_dir = Path(args.csv_dir)

    flights = {}
    for csv in sorted(csv_dir.glob("f1*.csv")):
        d = load(csv)
        flights[csv.stem] = d
    if not flights:
        raise SystemExit(
            f"No f1*.csv found in {csv_dir}. Download the 17 GB Dryad package "
            "(doi:10.5061/dryad.2ngf1vj4r), extract f10.csv..f13.csv, then pass "
            "--csv-dir <extracted_dir>. See aerpaw_data/README.md.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    summary = {}
    all_d, all_pl, all_model = [], [], []

    for name, d in flights.items():
        pl = np.asarray(d["path_loss_db"], float)
        d3d = np.asarray(d["uav2ground_dist"], float)
        h_u = np.asarray(d["rx_altitude_agl_m"], float)
        h_t = np.asarray(d["tx_altitude_agl_m"], float)
        pl_los, pl_nlos = model_pl(d3d, h_u, h_t)
        err = pl - pl_los
        # one-slope fit: PL = A + 10 n log10(d3d)
        x = np.log10(d3d)
        n_fit, A = np.polyfit(x, pl, 1)
        summary[name] = {
            "n": int(len(pl)),
            "d3d_min": float(d3d.min()), "d3d_max": float(d3d.max()),
            "h_uav_mean": float(h_u.mean()),
            "pl_mean": float(pl.mean()),
            "bias_db": float(err.mean()),
            "rmse_db": float(np.sqrt(np.mean(err ** 2))),
            "err_std_db": float(err.std()),
            "one_slope_n": float(n_fit),
            "one_slope_A": float(A),
        }
        all_d.append(d3d); all_pl.append(pl); all_model.append(pl_los)
        print(f"{name}: n={len(pl):6d}  d3d=[{d3d.min():6.1f},{d3d.max():7.1f}] m  "
              f"h={h_u.mean():5.1f} m  bias={err.mean():+6.2f} dB  "
              f"RMSE={np.sqrt(np.mean(err**2)):5.2f} dB  std={err.std():5.2f}  "
              f"one-slope n={n_fit:.2f}")

    all_d = np.concatenate(all_d); all_pl = np.concatenate(all_pl)
    all_model = np.concatenate(all_model)
    err = all_pl - all_model
    summary["ALL"] = {
        "n": int(len(err)),
        "bias_db": float(err.mean()),
        "rmse_db": float(np.sqrt(np.mean(err ** 2))),
        "err_std_db": float(err.std()),
        "err_p5": float(np.percentile(err, 5)),
        "err_p95": float(np.percentile(err, 95)),
    }
    # binned bias vs distance
    edges = np.percentile(all_d, np.linspace(0, 100, 11))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (all_d >= lo) & (all_d < hi)
        if m.sum() > 50:
            bins.append({"d_mid": float(np.sqrt(lo * hi)),
                         "d_lo": float(lo), "d_hi": float(hi),
                         "n": int(m.sum()),
                         "bias_db": float(err[m].mean()),
                         "rmse_db": float(np.sqrt(np.mean(err[m] ** 2)))})
    summary["distance_bins"] = bins
    for b in bins:
        print(f"  d~{b['d_mid']:7.1f} m (n={b['n']:6d}): bias={b['bias_db']:+6.2f} dB, "
              f"RMSE={b['rmse_db']:5.2f} dB")

    # panel (a): measured PL vs distance with model curves
    for name, d in flights.items():
        d3d = np.asarray(d["uav2ground_dist"], float)
        pl = np.asarray(d["path_loss_db"], float)
        ax.scatter(d3d, pl, s=1.2, alpha=0.15, rasterized=True, label=name)
    dd = np.linspace(max(all_d.min(), 5), all_d.max(), 300)
    h_u = np.full_like(dd, np.mean([np.asarray(f["rx_altitude_agl_m"], float).mean()
                                    for f in flights.values()]))
    pl_los, _ = model_pl(dd, h_u, 12.0)
    fspl = 20 * np.log10(4 * np.pi * dd * F_HZ / 299792458.0)
    ax.plot(dd, pl_los, "k-", lw=1.6, label="38.901 ULA-Los (model)")
    ax.plot(dd, fspl, "k--", lw=1.0, label="free space")
    ax.set_xscale("log")
    ax.set_xlabel("3D distance [m]"); ax.set_ylabel("path loss [dB]")
    ax.set_title("(a) Measured vs. model path loss (AERPAW ds32)")
    ax.legend(fontsize=7, markerscale=8); ax.grid(alpha=0.3, which="both")

    # panel (b): error histogram + binned bias
    ax = axes[1]
    ax.hist(err, bins=80, range=(-25, 25), color="#9ecae1",
            edgecolor="#3182bd", lw=0.3, density=True)
    xs = np.linspace(-25, 25, 400)
    ax.plot(xs, 1 / (summary["ALL"]["err_std_db"] * np.sqrt(2 * np.pi)) *
            np.exp(-xs ** 2 / (2 * summary["ALL"]["err_std_db"] ** 2)),
            "r--", lw=1.2, label=f"N(0, {summary['ALL']['err_std_db']:.2f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("measured $-$ model path loss [dB]")
    ax.set_ylabel("density")
    ax.set_title(f"(b) Residual: bias={summary['ALL']['bias_db']:+.2f} dB, "
                 f"RMSE={summary['ALL']['rmse_db']:.2f} dB, n={summary['ALL']['n']}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[write] {OUT_JSON.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
