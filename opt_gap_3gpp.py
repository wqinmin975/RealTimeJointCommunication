"""Optimality-gap analysis on the *paper's own* 3GPP TR 38.901 channel.

Why this script replaces ``opt_gap.py``
----------------------------------------
``opt_gap.py`` produced the gap numbers reported in Table V, but it imported its
entire model from ``hrl_airborne_fm_v5`` -- a prototype whose channel is the
*idealised free-space* law ``G0/(d3^2+eps)``.  That contradicts Section II-A of
the manuscript, which states that the UMa/ULA-Los aerial model of TR 38.901 is
adopted explicitly *instead of* an idealised free-space law, and it is why the
acceleration drift measured there (+0.36%) did not reproduce in the ablation
(+20% on the 3GPP channel).  This script re-runs the same experiment inside
``sim_analysis_3gpp`` so that the gap, the ablation, and the channel are one
and the same setup: identical users, identical demand ramp, identical measured
speed profile, identical solver code path.

Two gaps are reported and must not be conflated:

  (1) acceleration cost   = J(v5) vs J(v4)      -- the price of 3 rounds + DPP
                                                 warm-start, i.e. THIS paper's
                                                 contribution.
  (2) surrogate gap       = J(v4) vs J(ref@30)  -- the sub-optimality that the
                                                 SCA-BCD convexification carries
                                                 on its own, at any round count.

Only (1) is attributable to the acceleration suite.  (2) would persist for any
solver built on the same convexification.

Outputs: ``opt_gap_3gpp.csv`` and the paper figures ``opt_gap.png`` /
``opt_gap_bar.png``.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim_analysis_3gpp as S

HERE = Path(__file__).resolve().parent
N_STEPS = 40
REF_ROUNDS = 30


def build_scenario():
    """Reproduce the ablation scenario exactly (same seed, users, ramp, tracks)."""
    np.random.seed(123)
    users = S.make_users()
    demands = [np.clip(S.DEMAND_LO + (S.DEMAND_HI - S.DEMAND_LO) * t / 39.0
                       + np.random.randn(S.N_USER) * 2.0,
                       S.DEMAND_LO, S.DEMAND_HI)
               for t in range(N_STEPS)]
    speed = S.load_measured_speed_profile(N_STEPS)
    return users, np.array(demands), speed


def gap_pct(a, b):
    """Relative gap of a w.r.t. b, in percent (signed)."""
    return (a - b) / np.abs(b) * 100.0


def stat(x):
    return dict(mean=float(x.mean()),
                rms=float(np.sqrt(np.mean(x ** 2))),
                mae=float(np.mean(np.abs(x))),
                maxabs=float(np.max(np.abs(x))))


def main():
    users, demands, speed = build_scenario()

    variants = [
        ("v4", dict(bcd=True, adaptive=True, velocity=True, mpc=True,
                    accel=False, rounds=5)),
        ("v5", dict(bcd=True, adaptive=True, velocity=True, mpc=True,
                    accel=True, rounds=3)),
        ("ref", dict(bcd=True, adaptive=True, velocity=True, mpc=True,
                     accel=False, rounds=REF_ROUNDS)),
    ]

    J, P, lat_ms, acc = {}, {}, {}, {}
    for name, flags in variants:
        print(f"[gap] running {name} (rounds={flags['rounds']}, "
              f"accel={flags['accel']}) ...", flush=True)
        m = S.run_variant(name, flags, users, demands, speed)
        J[name] = np.asarray(m["Js"], float)
        lat_ms[name] = m["mean_lat"]
        acc[name] = m["mean_acc"]
        print(f"        mean J = {J[name].mean():.4f}   "
              f"mean latency = {m['mean_lat']:.2f} ms   "
              f"mean acc = {m['mean_acc']:.3f}", flush=True)

    g_acc = gap_pct(J["v5"], J["v4"])       # (1) acceleration cost
    g_v4 = gap_pct(J["v4"], J["ref"])      # (2) surrogate gap, baseline
    g_v5 = gap_pct(J["v5"], J["ref"])      # total for the accelerated solver

    s_acc, s_v4, s_v5 = stat(g_acc), stat(g_v4), stat(g_v5)

    print("\n================ OPTIMALITY GAP (3GPP TR 38.901) ================")
    print(f"  mean J  v4  (5 rounds)  : {J['v4'].mean():.4f}")
    print(f"  mean J  v5  (3 rounds)  : {J['v5'].mean():.4f}")
    print(f"  mean J  ref ({REF_ROUNDS} rounds) : {J['ref'].mean():.4f}")
    print(f"  (1) acceleration cost  v5 vs v4 : mean {s_acc['mean']:+.2f}%  "
          f"RMS {s_acc['rms']:.2f}%  MAE {s_acc['mae']:.2f}%  "
          f"max|.| {s_acc['maxabs']:.2f}%")
    print(f"  (2) surrogate gap  v4 vs ref   : mean {s_v4['mean']:+.2f}%  "
          f"MAE {s_v4['mae']:.2f}%  max|.| {s_v4['maxabs']:.2f}%")
    print(f"      surrogate gap  v5 vs ref   : mean {s_v5['mean']:+.2f}%  "
          f"MAE {s_v5['mae']:.2f}%  max|.| {s_v5['maxabs']:.2f}%")
    print(f"  latency  v4 {lat_ms['v4']:.1f} ms  ->  v5 {lat_ms['v5']:.1f} ms  "
          f"({lat_ms['v4']/max(lat_ms['v5'],1e-9):.1f}x)")
    print("=================================================================")

    out = HERE / "opt_gap_3gpp.csv"
    np.savetxt(out,
               np.column_stack([np.arange(N_STEPS), J["v4"], J["v5"], J["ref"],
                                g_v4, g_v5, g_acc]),
               delimiter=",", comments="",
               header="step,J_v4,J_v5,J_ref,gap_v4_pct,gap_v5_pct,acc_gap_v5_v4_pct",
               fmt="%.6f")
    print(f"[saved] {out}")

    summary = dict(
        channel="3GPP TR 38.901 UMa/ULA-Los (sim_analysis_3gpp)",
        note=("Replaces opt_gap.py, which ran on the prototype's idealised "
              "free-space channel; the two are not comparable."),
        n_steps=N_STEPS, ref_rounds=REF_ROUNDS,
        mean_J=dict(v4=float(J["v4"].mean()), v5=float(J["v5"].mean()),
                    ref=float(J["ref"].mean())),
        mean_lat_ms=dict(v4=float(lat_ms["v4"]), v5=float(lat_ms["v5"])),
        mean_acc=dict(v4=float(acc["v4"]), v5=float(acc["v5"])),
        acceleration_cost_pct=s_acc,
        surrogate_gap_v4_pct=s_v4,
        surrogate_gap_v5_pct=s_v5,
    )
    (HERE / "opt_gap_3gpp.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {HERE / 'opt_gap_3gpp.json'}")

    # ---- figures (paper paths) ----
    plot_dir = HERE / "paper" / "figs"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(J["ref"], "k-", lw=2, label=f"reference (SCA, {REF_ROUNDS} rounds)")
    ax.plot(J["v4"], "C0--", lw=1.4, label="v4 baseline (5 rounds)")
    ax.plot(J["v5"], "C3:", lw=1.4, label="v5 accelerated (3 rounds)")
    ax.set_xlabel("MPC step"); ax.set_ylabel(r"objective $J$")
    ax.set_title("(a) Objective per step (3GPP TR 38.901)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(g_v4, "C0--", lw=1.4, label=r"(2) surrogate gap: v4 vs ref")
    ax.plot(g_v5, "C3:", lw=1.4, label=r"total: v5 vs ref")
    ax.plot(g_acc, "C2-", lw=1.4, label=r"(1) acceleration cost: v5 vs v4")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("MPC step"); ax.set_ylabel("relative gap [\\%]")
    ax.set_title("(b) Two-layer gap decomposition")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "opt_gap.png", dpi=150)
    plt.close(fig)
    print(f"[saved] {plot_dir / 'opt_gap.png'}")

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
    names = ["v4 baseline\n(5 rounds)", "v5 accelerated\n(3 rounds)",
             f"reference\n({REF_ROUNDS} rounds)"]
    means = [J["v4"].mean(), J["v5"].mean(), J["ref"].mean()]
    bars = ax2.bar(names, means, color=["C0", "C3", "k"])
    for b, m in zip(bars, means):
        ax2.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel(r"mean objective $J$")
    ax2.set_title("Mean objective on the 3GPP channel")
    ax2.grid(alpha=.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(plot_dir / "opt_gap_bar.png", dpi=150)
    plt.close(fig2)
    print(f"[saved] {plot_dir / 'opt_gap_bar.png'}")


if __name__ == "__main__":
    main()
