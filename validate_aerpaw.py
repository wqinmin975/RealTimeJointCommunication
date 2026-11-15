"""Validate the TR 38.901 ULA / ULA-Los aerial channel model against *real*
UAV air-to-ground (A2G) measurement data -- the AERPAW rural campaign.

Why AERPAW
----------
The recommended real-measured benchmark is the AERPAW "Air-to-Ground Channel
Sounding and Path Loss Measurements at 3.4 GHz" dataset (Dryad DOI
10.5061/dryad.2ngf1vj4r; Gurses & Sichitiu, VTC 2024; Raouf et al., npj
Wireless 2026).  It is a *real* SDR channel-sounder campaign: a UAV receiver
at ~60 m AGL, a fixed ground transmitter at 3.4 GHz, rural environment, with
processed CSV files that contain, per sample:

    measurement_id, campaign_id, timestamp_utc, time_s,
    rx_latitude, rx_longitude, rx_altitude_agl_m,
    tx_latitude, tx_longitude, tx_altitude_agl_m,
    heading_deg, uav2ground_dist, path_loss_db

This script ingests those CSVs and answers one question for the *large-scale*
path loss (the small-scale Rician K was already validated against the official
5G Toolbox ``nrCDLChannel`` in ``calibrate_official_fading.py``):

    How well does our 38.901 ULA-Los path-loss model (run at 3.4 GHz) predict
    the *measured* A2G path loss, and is the predicted LOS/NLOS split, bias and
    CDF consistent with the measurements?

Because the CSVs ship inside a 17 GB ``data.tar.gz`` on Dryad (and the
individual file endpoint is session-protected), the raw pull is done once,
offline; see ``aerpaw_data/README.md`` for the exact fetch + extract steps.
The script then runs unchanged on the extracted ``f10.csv``..``f13.csv``.

A ``--demo`` mode synthesises an AERPAW-like sample (60 m AGL, 3.4 GHz, the
model's own PL + a few dB of shadow/measurement noise) so the pipeline is
exercised end-to-end *without* the raw data.  The demo is a self-consistency
check only -- it is NOT the real-measured validation.

Outputs (measured run): ``aerpaw_validation.json`` (metrics) and
``aerpaw_validation.png`` (model-vs-measured PL scatter + CDF).  A ``--demo``
run writes to ``aerpaw_validation_demo.json`` / ``aerpaw_validation_demo.png``
instead, and is refused if asked to write to the measured artifact names, so a
synthetic self-consistency check can never overwrite the real validation.
Every JSON output carries an explicit ``data_mode`` marker
(``"measured"`` vs ``"synthetic-demo"``).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import channel_3gpp as ch3g

# AERPAW campaign operating point (rural, Lake Wheeler testbed)
AERPAW_F_HZ = 3.4e9
AERPAW_TX_ALT_M = 0.0          # ground transmitter AGL
AERPAW_UAV_ALT_M = 60.0        # nominal UAV altitude AGL
SF_SIGMA_LOS = 4.0
SF_SIGMA_NLOS = 7.8

# --- output filename isolation -------------------------------------------
# The measured-validation artifacts must never be overwritten by a --demo
# (synthetic) run.  Demo runs default to *_demo.* and refuse the real names.
REAL_OUT_JSON = "aerpaw_validation.json"
REAL_OUT_PNG = "aerpaw_validation.png"
DEMO_OUT_JSON = "aerpaw_validation_demo.json"
DEMO_OUT_PNG = "aerpaw_validation_demo.png"

# Expected CSV column names (see header above)
COL_PL = "path_loss_db"
COL_D3D = "uav2ground_dist"
COL_RX_ALT = "rx_altitude_agl_m"
COL_TX_ALT = "tx_altitude_agl_m"


def load_aerpaw_csv(path):
    """Load one AERPAW processed CSV into a numpy structured array."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    return data


def geometry_from_csv(data):
    """Recover (uav, user) coordinate arrays (meters) from the AERPAW columns.

    We only need the 3D distance and the altitude difference, so we place the
    transmitter at the origin on the ground and the UAV at horizontal distance
    ``d2D = sqrt(d3D^2 - dh^2)`` directly above the transmitter height.
    """
    d3d = np.asarray(data[COL_D3D], float)
    h_uav = np.asarray(data[COL_RX_ALT], float)
    h_tx = np.asarray(data[COL_TX_ALT], float)
    dh = h_uav - h_tx
    d2d = np.sqrt(np.maximum(d3d ** 2 - dh ** 2, 1.0))
    uav = np.stack([np.zeros_like(d2d), d2d, h_uav], axis=1)   # (N, 3) meters
    user = np.stack([np.zeros_like(d2d), np.zeros_like(d2d), h_tx], axis=1)
    return uav, user


def model_path_loss(uav, user, f_hz=AERPAW_F_HZ):
    """Return the model's LOS and NLOS path loss, plus the LOS probability."""
    los_mask = np.ones(len(uav), dtype=bool)        # branch A
    pl_los = ch3g.path_loss_db(f_hz, uav, user, los_mask)
    pl_nlos = ch3g.path_loss_db(f_hz, uav, user, ~los_mask)
    d2 = ch3g.d_2d(uav, user)
    h_uav = uav[:, 2]
    p_los = ch3g.los_probability(d2, h_uav)
    return pl_los, pl_nlos, p_los


def validate(measured_pl, uav, user, f_hz=AERPAW_F_HZ):
    """Compute model-vs-measured path-loss metrics.

    Three complementary views:
      * expected-PL RMSE  : measured vs the LOS-probability-weighted PL;
      * LOS-subset RMSE   : measured vs PL_LOS for the samples the model deems LOS;
      * NLOS-subset RMSE  : measured vs PL_NLOS for the samples deemed NLOS.
    The LOS/NLOS assignment for the subsets uses the model's own P_LOS with a
    0.5 threshold, which is the standard operational classifier.
    """
    pl_los, pl_nlos, p_los = model_path_loss(uav, user, f_hz)
    is_los = p_los >= 0.5

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    expected = p_los * pl_los + (1.0 - p_los) * pl_nlos
    rmse_exp = _rmse(measured_pl, expected)
    bias_exp = float(np.mean(measured_pl - expected))
    rmse_los = _rmse(measured_pl[is_los], pl_los[is_los]) if is_los.any() else float("nan")
    rmse_nlos = _rmse(measured_pl[~is_los], pl_nlos[~is_los]) if (~is_los).any() else float("nan")

    return {
        "n_samples": int(len(measured_pl)),
        "rmse_expected_db": rmse_exp,
        "bias_expected_db": bias_exp,
        "rmse_los_db": rmse_los,
        "rmse_nlos_db": rmse_nlos,
        "frac_model_los": float(is_los.mean()),
    }, (measured_pl, pl_los, pl_nlos, p_los, is_los)


def synthetic_aerpaw_like(n=2000, seed=42):
    """Self-consistency demo: build AERPAW-like geometry and *true* PL from the
    model (+ a few dB shadow/measurement noise).  NOT a real-measured dataset."""
    rng = np.random.default_rng(seed)
    # Wider altitude/distance spread than the nominal 60 m so the demo also
    # exercises the NLOS branch (at a fixed 60 m AGL the model is ~all LOS).
    h_uav = rng.normal(AERPAW_UAV_ALT_M, 25.0, n).clip(15, 150)
    d2d = rng.uniform(40, 500, n)
    d3d = np.sqrt(d2d ** 2 + h_uav ** 2)
    uav = np.stack([np.zeros(n), d2d, h_uav], axis=1)
    user = np.zeros((n, 3))
    pl_los, pl_nlos, p_los = model_path_loss(uav, user, AERPAW_F_HZ)
    is_los = rng.random(n) < p_los
    true_pl = np.where(is_los, pl_los, pl_nlos)
    # shadow + measurement noise
    sf = np.where(is_los, rng.normal(0, SF_SIGMA_LOS, n), rng.normal(0, SF_SIGMA_NLOS, n))
    meas = true_pl + sf + rng.normal(0, 1.0, n)
    return meas, uav, user


def _json_safe(value):
    """Replace non-finite floats with ``None`` so json.dumps emits strict JSON.

    ``validate()`` legitimately returns ``float("nan")`` for a subset RMSE when
    that subset is empty (e.g. no NLOS samples).  A bare ``NaN`` token is NOT
    valid JSON; mapping it to ``null`` keeps the artifact readable by strict
    parsers (``json.loads`` with default ``parse_constant`` rejects NaN).
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=str, default=None,
                    help="directory containing f10.csv..f13.csv (AERPAW).")
    ap.add_argument("--demo", action="store_true",
                    help="run the SYNTHETIC self-consistency demo without raw "
                         "data; writes to *_demo.* and never to the measured "
                         "artifacts.")
    ap.add_argument("--f-hz", type=float, default=AERPAW_F_HZ)
    ap.add_argument("--out-json", type=str, default=None,
                    help="override output JSON name (default depends on mode).")
    ap.add_argument("--out-png", type=str, default=None,
                    help="override output PNG name (default depends on mode).")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    # Resolve output names by mode.  A --demo run must never overwrite the
    # measured-validation artifact, even if the user passes an explicit name.
    if args.demo:
        cand_json = args.out_json or DEMO_OUT_JSON
        cand_png = args.out_png or DEMO_OUT_PNG
        if (Path(cand_json).name == REAL_OUT_JSON
                or Path(cand_png).name == REAL_OUT_PNG):
            raise SystemExit(
                "[validate_aerpaw] --demo refuses to write to the measured "
                f"artifact names ({REAL_OUT_JSON} / {REAL_OUT_PNG}). "
                "Omit --out-json/--out-png (defaults to *_demo.*) or choose "
                "another name. Synthetic numbers must stay separable from the "
                "measured AERPAW validation.")
        out_json, out_png = out_dir / cand_json, out_dir / cand_png
        data_mode = "synthetic-demo"
    else:
        out_json = out_dir / (args.out_json or REAL_OUT_JSON)
        out_png = out_dir / (args.out_png or REAL_OUT_PNG)
        data_mode = "measured"

    if args.demo:
        print("[validate_aerpaw] DEMO mode (synthetic, self-consistency only)")
        measured_pl, uav, user = synthetic_aerpaw_like()
        source = "synthetic-aerpaw-like (demo)"
    else:
        if not args.csv_dir:
            raise SystemExit("Provide --csv-dir (AERPAW CSVs) or --demo.")
        csv_dir = Path(args.csv_dir)
        csvs = sorted(csv_dir.glob("f1*.csv"))
        if not csvs:
            raise SystemExit(f"No f1*.csv found in {csv_dir}")
        chunks = [load_aerpaw_csv(c) for c in csvs]
        # concatenate by stacking the needed columns
        cols = (COL_PL, COL_D3D, COL_RX_ALT, COL_TX_ALT)
        big = np.concatenate([c[list(cols)] for c in chunks])
        uav, user = geometry_from_csv(big)
        measured_pl = np.asarray(big[COL_PL], float)
        source = "; ".join(c.name for c in csvs)

    metrics, aux = validate(measured_pl, uav, user, args.f_hz)
    metrics["source"] = source
    metrics["f_hz"] = args.f_hz
    # Explicit provenance marker: a synthetic demo must never be mistaken for
    # the real measured AERPAW validation (Dryad doi:10.5061/dryad.2ngf1vj4r).
    metrics["data_mode"] = data_mode
    metrics["is_real_measurement"] = (data_mode == "measured")
    if data_mode == "synthetic-demo":
        metrics["notice"] = (
            "SYNTHETIC self-consistency demo -- NOT the measured AERPAW "
            "validation. Do NOT cite these numbers as the measured bias/RMSE "
            "(136,987 samples, bias -0.02 dB, RMSE 5.4 dB).")

    # ---- figure: scatter (measured vs LOS/NLOS branch) + CDF ----------------
    measured, pl_los, pl_nlos, p_los, is_los = aux
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    ax.scatter(pl_los[is_los], measured[is_los], s=4, alpha=0.3,
               color="#1f77b4", label="model-LOS subset")
    ax.scatter(pl_nlos[~is_los], measured[~is_los], s=4, alpha=0.3,
               color="#d62728", label="model-NLOS subset")
    lim = [min(measured.min(), pl_los.min()) - 2, max(measured.max(), pl_nlos.max()) + 2]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("model path loss [dB]"); ax.set_ylabel("measured path loss [dB]")
    ax.set_title("(a) Measured vs model PL"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for lab, x in [("measured", measured), ("model-LOS", pl_los),
                   ("model-NLOS", pl_nlos)]:
        xs = np.sort(x); cdf = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, cdf, lw=1.3, label=lab)
    ax.set_xlabel("path loss [dB]"); ax.set_ylabel("empirical CDF")
    ax.set_title("(b) Path-loss CDF"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    out_json.write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False),
        encoding="utf-8")
    if data_mode == "synthetic-demo":
        print("  WARNING: SYNTHETIC demo data -- not measured, not citable.")
    print(f"[validate_aerpaw] source : {source}")
    print(f"  n={metrics['n_samples']}  frac_model_LOS={metrics['frac_model_los']:.2f}")
    print(f"  RMSE(expected)={metrics['rmse_expected_db']:.2f} dB  "
          f"bias={metrics['bias_expected_db']:+.2f} dB")
    print(f"  RMSE(LOS)={metrics['rmse_los_db']:.2f} dB  "
          f"RMSE(NLOS)={metrics['rmse_nlos_db']:.2f} dB")
    print(f"[write] {out_json.name}, {out_png.name}  (data_mode={data_mode})")


if __name__ == "__main__":
    main()
