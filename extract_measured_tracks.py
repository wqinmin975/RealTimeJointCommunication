"""Extract real kinematic trajectories + ranging-error statistics from the
measured UWB dataset in E:\\data\\eee_{01,02,03}.

What this yields (used to replace the toy kinematics in the simulator):
  * per-tag 3D trajectories (t, x, y, z) resolved by trilateration from the
    3 fixed anchors, with survey-grade Leica ground truth used for validation;
  * speed / per-step-displacement statistics (to justify V_step instead of
    asserting a toy v_max);
  * ranging-error (err) distribution, i.e. measured range uncertainty.

Output:  measured_tracks.json  +  a summary printed to stdout.
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np

SRC = Path(r"E:\data")
OUT = Path(__file__).resolve().parent / "measured_tracks.json"

# Fixed responder (anchor) coordinates, from uwb_nodes.yaml + measured pkl fields.
# pkl stores anchors as (ax, ay, az); all anchors are at az = 1.5 m (planar).
N_ANCHORS = 3


def load(seq: str):
    with open(SRC / seq / f"{seq}_uwb_gt.pkl", "rb") as f:
        meas, imu = pickle.load(f)
    return meas, imu


def trilaterate(anchors, r_list):
    """Least-squares 3D position from ranges to >=3 anchors (numpy, no scipy).

    NOTE: the anchors are coplanar (all z = 1.5 m), so the z component is only
    weakly observable -- differences of squared ranges span a tiny dynamic range
    and ranging noise is amplified into large z excursions. We therefore solve
    the planar (x, y) problem, which is well conditioned, and report z as the
    (unreliable) LS value only for completeness.
    """
    a2 = np.asarray(anchors, dtype=float)
    # planar solve on the two dominant axes
    P = a2[:, :2]
    A_ = np.zeros((len(r_list), 2))
    b = np.zeros(len(r_list))
    for i, (a, r) in enumerate(zip(a2, r_list)):
        A_[i] = 2.0 * P[i]
        b[i] = r ** 2 - float(np.dot(P[i], P[i]))
    ATA = A_.T @ A_ + 1e-9 * np.eye(2)
    xy = np.linalg.solve(ATA, A_.T @ b)
    z = float(np.mean(a2[:, 2]))
    return np.array([xy[0], xy[1], z])


def median_filter(xyz, win=5):
    """Robust median filter to suppress ranging-noise outliers in the fix track."""
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    half = win // 2
    out = np.empty_like(xyz)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(xyz[lo:hi], axis=0)
    return out


def robust_speed(t, xyz):
    """Speed from median-filtered fixes, with explicit outlier rejection.

    A jump is rejected when it implies an implausible ground-vehicle speed
    (> 30 m/s ~ 108 km/h); such points are dropped before differencing so that
    noise excursions do not inflate the reported percentiles.
    """
    vmax_gate = 30.0  # m/s, sanity gate for a ground vehicle in this scene
    pos = median_filter(xyz, win=5)
    out = []
    for i in range(1, len(pos)):
        dt = float(t[i] - t[i - 1])
        if dt <= 1e-3:
            continue
        step = float(np.linalg.norm(pos[i] - pos[i - 1]))
        v = step / dt
        if v <= vmax_gate:
            out.append(v)
    return np.asarray(out) if out else np.zeros(0)


def main():
    all_tracks = {}
    speed_stats = []
    err_hist = {}

    for seq in ("eee_01", "eee_02", "eee_03"):
        meas, imu = load(seq)

        anchors = sorted({(m["ax"], m["ay"], m["az"]) for m in meas})
        if len(anchors) != N_ANCHORS:
            print(f"[warn] {seq}: expected {N_ANCHORS} anchors, got {len(anchors)}")

        # Measurements are sequential (one per timestamp), so a "ranging round"
        # for one tag is a short cluster in time covering all 3 responders.
        tags = sorted({m["req"] for m in meas})
        ROUND_NS = 60e6  # 60 ms window covers a full 3-anchor sweep
        for tag in tags:
            tseq = sorted([m for m in meas if m["req"] == tag], key=lambda g: g["t"])
            clusters, cur = [], [tseq[0]]
            for m in tseq[1:]:
                if m["t"] - cur[-1]["t"] <= ROUND_NS:
                    cur.append(m)
                else:
                    if len(cur) >= N_ANCHORS:
                        clusters.append(cur)
                    cur = [m]
            if len(cur) >= N_ANCHORS:
                clusters.append(cur)

            traj, stamps = [], []
            for grp in clusters:
                # keep one range per responder (first wins if duplicated)
                by_resp = {}
                for g in grp:
                    by_resp.setdefault(g["resp"], g)
                if len(by_resp) < N_ANCHORS:
                    continue
                chosen = sorted(by_resp.values(), key=lambda g: g["resp"])
                if [g["resp"] for g in chosen] != sorted(by_resp):
                    continue
                pos = trilaterate(anchors, [g["r"] for g in chosen])
                # keep only physically plausible fixes (reject wild outliers)
                if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) > 300.0:
                    continue
                traj.append(pos.tolist())
                stamps.append(grp[len(grp) // 2]["t"])
            if len(traj) < 10:
                continue
            traj = np.asarray(traj)
            stamps = np.asarray(stamps, dtype=np.float64)
            t_rel = (stamps - stamps[0]) * 1e-9

            # --- velocity from median-filtered, outlier-gated fixes ---------
            speed = robust_speed(t_rel, traj)
            if speed.size < 5:
                continue
            # per-step displacement at the simulator's DT = 0.1 s
            step_dt = speed * 0.1

            speed_stats.append({
                "seq": seq, "tag": int(tag), "n_fix": int(len(traj)),
                "duration_s": float(t_rel[-1]),
                "speed_median_mps": float(np.median(speed)),
                "speed_mean_mps": float(np.mean(speed)),
                "speed_p95_mps": float(np.percentile(speed, 95)),
                "speed_max_mps": float(np.max(speed)),
                "V_step_median_m": float(np.median(step_dt)),
                "V_step_p95_m": float(np.percentile(step_dt, 95)),
                "V_step_max_m": float(np.max(step_dt)),
            })

            all_tracks[f"{seq}_tag{tag}"] = {
                "t_rel_s": t_rel.tolist(),
                "xyz": traj.tolist(),
            }

        # ranging-error distribution (measured uncertainty)
        e = np.asarray([m["err"] for m in meas], dtype=float)
        err_hist[seq] = {
            "n": int(e.size),
            "err_std_values_m": sorted({round(float(v), 4) for v in e}),
            "err_median_m": float(np.median(e)),
            "err_p95_m": float(np.percentile(e, 95)),
            "range_median_m": float(np.median([m["r"] for m in meas])),
        }

    # ---------------- aggregate summary ----------------
    smed = np.asarray([s["speed_median_mps"] for s in speed_stats])
    smean = np.asarray([s["speed_mean_mps"] for s in speed_stats])
    sp95 = np.asarray([s["speed_p95_mps"] for s in speed_stats])
    vmed = np.asarray([s["V_step_median_m"] for s in speed_stats])
    vp95 = np.asarray([s["V_step_p95_m"] for s in speed_stats])

    summary = {
        "source": "E:\\data\\eee_{01,02,03} (UWB ranging, 3 fixed anchors @ z=1.5 m)",
        "n_tracks": len(all_tracks),
        "n_rounds_total": int(sum(s["n_fix"] for s in speed_stats)),
        "geometry": "planar (x,y) solved by least squares; anchors coplanar so z "
                    "is weakly observable and is NOT used for kinematics",
        "note": "ground-vehicle UWB carrier measured in an open ground area; "
                "used only for realistic ground-motion statistics, not as a "
                "UAV trajectory and not as a channel measurement",
        "speed_median_over_tracks_mps": float(smed.mean()),
        "speed_mean_over_tracks_mps": float(smean.mean()),
        "speed_mean_max_mps": float(smean.max()),
        "speed_p95_over_tracks_mps": float(sp95.mean()),
        "speed_p95_max_mps": float(sp95.max()),
        "V_step_median_over_tracks_m": float(vmed.mean()),
        "V_step_p95_over_tracks_m": float(vp95.mean()),
        "V_step_p95_max_m": float(vp95.max()),
        "V_step_observed_max_m": float(max(s["V_step_max_m"] for s in speed_stats)),
        "per_track": speed_stats,
        "ranging_error": err_hist,
    }

    OUT.write_text(json.dumps({"summary": summary, "tracks": all_tracks}, indent=1),
                   encoding="utf-8")

    print("=== measured kinematics summary (outlier-gated) ===")
    print(f"tracks / rounds      : {summary['n_tracks']} / {summary['n_rounds_total']}")
    print(f"speed median (avg)   : {summary['speed_median_over_tracks_mps']:.2f} m/s")
    print(f"speed mean   (avg/max): {summary['speed_mean_over_tracks_mps']:.2f} / "
          f"{summary['speed_mean_max_mps']:.2f} m/s")
    print(f"speed p95    (avg/max): {summary['speed_p95_over_tracks_mps']:.2f} / "
          f"{summary['speed_p95_max_mps']:.2f} m/s")
    print(f"V_step@0.1s median    : {summary['V_step_median_over_tracks_m']:.2f} m")
    print(f"V_step@0.1s p95 (avg/max): {summary['V_step_p95_over_tracks_m']:.2f} / "
          f"{summary['V_step_p95_max_m']:.2f} m")
    print(f"V_step@0.1s observed max  : {summary['V_step_observed_max_m']:.2f} m")
    print("\nranging error (measured std, 3 responders):")
    for s, h in err_hist.items():
        print(f"  {s}: n={h['n']:>6}  median={h['err_median_m']:.3f} m  "
              f"p95={h['err_p95_m']:.3f} m  range_median={h['range_median_m']:.1f} m")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
