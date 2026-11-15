"""Fading-characterisation figure for the TR 38.901 upgrade.

fig: paper/figs/fading.png
  (a) Realised per-step rate under the full 3GPP model (correlated shadow
      fading + CDL fast fading + geometry) for v5, showing the per-step
      fluctuation that forces re-optimisation.
  (b) Small-scale fading sample with the ULA-Los K-factor, plus the
      spatially-correlated shadow-fading realisation.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import channel_38901_fading as chfade
import channel_3gpp as ch3g
import sim_analysis_3gpp as S

OUT = Path("paper/figs/fading.png")

# ---- (a) realised rate under the full model ----------------------------
np.random.seed(123)
users = S.make_users()
demands = [np.clip(S.DEMAND_LO + (S.DEMAND_HI - S.DEMAND_LO) * t / 39.0
                   + np.random.randn(S.N_USER) * 2.0, S.DEMAND_LO, S.DEMAND_HI)
           for t in range(40)]
prof = S.load_measured_speed_profile(40)
flags = dict(bcd=True, adaptive=True, velocity=True, mpc=True,
             accel=True, rounds=3)
m = S.run_variant("v5", flags, users, demands, prof)
rates = np.asarray(m["rates"])
steps = np.arange(len(rates))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

ax = axes[0]
ax.plot(steps, rates, "-o", ms=3.2, color="C0", label="realised rate $R_k$")
ax.plot(steps, [S.DEMAND_LO + (S.DEMAND_HI - S.DEMAND_LO) * t / 39.0
                for t in range(40)], "--", color="C3", lw=1.6,
        label="demand $d_k$")
ax.set_xlabel("decision step $t$ ($\\Delta t$ = 0.1 s)")
ax.set_ylabel("rate (Mb/s)")
ax.set_title(f"(a) Rate under correlated shadow + fast fading\n"
             f"mean {m['mean_rate']:.1f}, sd {m['rate_std']:.1f} Mb/s "
             f"(cv {m['rate_cv']*100:.1f}%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# ---- (b) fading components ---------------------------------------------
ax = axes[1]
h = chfade.fast_fading_series(seed=7, n_snapshots=300,
                              k_factor_db=chfade.ULA_LOS_K_FACTOR_DB,
                              max_doppler_hz=250.0)
st = chfade.fading_statistics(h)
p_db = 10.0 * np.log10(np.maximum(np.abs(h) ** 2, 1e-8))
ax.plot(20 * np.log10(np.abs(h)), "-", lw=1.0, color="C2",
        label=f"fast fading $|h|$ (K={st['k_factor_db']:.1f} dB)")

traj = np.stack([np.linspace(0, 400, 200), np.zeros(200)], axis=1)
sf = chfade.correlated_shadow_fading(traj, chfade.SF_SIGMA_LOS_DB,
                                     chfade.SHADOW_CORR_DISTANCE_M, seed=5)
ax.plot(np.linspace(0, 199, 200), sf, "-", lw=1.4, color="C0",
        label=f"shadow fading, $\\sigma$={chfade.SF_SIGMA_LOS_DB:.0f} dB (corr.)")
ax.set_xlabel("sample / step")
ax.set_ylabel("amplitude (dB)")
ax.set_title("(b) TR 38.901 fading components\n"
             f"fading depth {st['fading_depth_db']:.1f} dB, "
             f"lag-1 corr {np.corrcoef(sf[:-1], sf[1:])[0,1]:.2f}")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"[plot] {OUT}")

cal = {
    "model": "3GPP TR 38.901 ULA-Los (large-scale + small-scale)",
    "table_7_8_3_1": {
        "delay_spread_ns": chfade.ULA_LOS_DELAY_SPREAD_S * 1e9,
        "angular_spread_deg": chfade.ULA_LOS_ANG_SPREAD_DEG,
        "xpr_dB": chfade.ULA_LOS_R_XPR_DB,
        "k_factor_dB": chfade.ULA_LOS_K_FACTOR_DB,
        "clusters": chfade.NUM_CLUSTERS,
    },
    "shadow_fading": {
        "sigma_los_dB": chfade.SF_SIGMA_LOS_DB,
        "sigma_nlos_dB": chfade.SF_SIGMA_NLOS_DB,
        "corr_distance_m": chfade.SHADOW_CORR_DISTANCE_M,
        "measured_std_dB": float(sf.std()),
        "measured_lag1_corr": float(np.corrcoef(sf[:-1], sf[1:])[0, 1]),
    },
    "fast_fading": {
        "measured_k_dB": st["k_factor_db"],
        "fading_depth_dB": st["fading_depth_db"],
        "cv": st["cv"],
    },
    "realised_rate_Mb_s": {
        "mean": m["mean_rate"], "sd": m["rate_std"], "cv": m["rate_cv"],
        "min": m["rate_min"], "max": m["rate_max"],
    },
}
with open("fading_calib.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=1)
print("[calib] fading_calib.json")
print(json.dumps(cal, indent=1))
