"""Produce the channel-calibration figure for the revised manuscript.

fig: paper/figs/channel_calib.png
  (a) 3GPP TR 38.901 ULA-Los path loss vs 3-D range, one curve per UAV
      altitude, with the free-space reference the previous model used.
  (b) Achievable rate vs range at the operating power/bandwidth, with the
      demand ramp drawn on top -- showing that the rate constraint is active
      over part of the operating region (it was inactive under G0 = 1e-4).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import channel_3gpp as ch3g
import sim_analysis_3gpp as S

OUT = Path("paper/figs/channel_calib.png")
F = ch3g.F_HZ
B_LINK_MHZ = S.B_TOTAL / S.N_UAV

d3 = np.linspace(30.0, 400.0, 400)
alts = (50, 100, 150)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))

# ---------------- (a) path loss ----------------
ax = axes[0]
for h in alts:
    pl = []
    for d in d3:
        uav = np.array([[0.0, 0.0, float(h)]])
        # keep 3-D range = d by placing the user on the ground accordingly
        horiz = np.sqrt(max(d ** 2 - h ** 2, 0.0))
        usr = np.array([[horiz, 0.0, 0.0]])
        pl.append(float(ch3g.path_loss_db(F, uav, usr, np.array([[True]]))[0, 0]))
    ax.plot(d3, pl, "-", lw=1.8, label=f"ULA-Los, $h$={h} m")

# free-space reference at the same carrier (the previously assumed law)
fs = 20.0 * np.log10(4.0 * np.pi * d3 * F / ch3g.F_C)
ax.plot(d3, fs, "k--", lw=1.4, label="free space (previous model)")
ax.set_xlabel("3-D range $d_{3D}$ (m)")
ax.set_ylabel("path loss (dB)")
ax.set_title("(a) 3GPP TR 38.901 ULA-Los path loss")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# ---------------- (b) achievable rate vs demand ----------------
ax = axes[1]
p_w = 0.25
for h in alts:
    R = []
    for d in d3:
        uav = np.array([[0.0, 0.0, float(h)]])
        horiz = np.sqrt(max(d ** 2 - h ** 2, 0.0))
        usr = np.array([[horiz, 0.0, 0.0]])
        g = float(ch3g.gain_from_db(
            ch3g.path_loss_db(F, uav, usr, np.array([[True]]))[0, 0],
            d_ref=None, f_hz=F, b_ref_hz=S.B_PER_LINK_HZ))
        R.append(B_LINK_MHZ * np.log2(1.0 + p_w * g))
    ax.plot(d3, R, "-", lw=1.8, label=f"$h$={h} m")

# demand ramp mapped onto the range axis for reference
ax.axhline(S.DEMAND_LO, color="C2", ls=":", lw=1.4)
ax.axhline(S.DEMAND_HI, color="C2", ls="--", lw=1.4)
ax.text(310, S.DEMAND_LO + 0.6, "demand min", color="C2", fontsize=7.5, ha="right")
ax.text(310, S.DEMAND_HI + 0.6, "demand max", color="C2", fontsize=7.5, ha="right")
ax.set_xlabel("3-D range $d_{3D}$ (m)")
ax.set_ylabel("achievable rate (Mb/s)")
ax.set_title(f"(b) rate at $p$={p_w} W, $B$={B_LINK_MHZ:.2f} MHz/link")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"[plot] {OUT}")

# ---------------- machine-readable calibration for the paper ----------------
def pl_at(h, d):
    """LOS path loss at 3-D range d for a UAV at altitude h (user on ground)."""
    horiz = np.sqrt(max(d ** 2 - h ** 2, 0.0))
    uav = np.array([[0.0, 0.0, float(h)]])
    usr = np.array([[horiz, 0.0, 0.0]])
    return float(ch3g.path_loss_db(F, uav, usr, np.array([[True]]))[0, 0])


def rate_at(h, d, p_w):
    horiz = np.sqrt(max(d ** 2 - h ** 2, 0.0))
    uav = np.array([[0.0, 0.0, float(h)]])
    usr = np.array([[horiz, 0.0, 0.0]])
    g = float(ch3g.gain_from_db(pl_at(h, d), d_ref=None, f_hz=F,
                                b_ref_hz=S.B_PER_LINK_HZ))
    return B_LINK_MHZ * np.log2(1.0 + p_w * g)


cal = {
    "model": "3GPP TR 38.901 ULA / ULA-Los",
    "carrier_Hz": F,
    "G_tx_dBi": ch3g.G_TX_DBI,
    "G_rx_dBi": ch3g.G_RX_DBI,
    "shadow_sigma_dB": S.SHADOW_SIGMA,
    "B_total_MHz": S.B_TOTAL,
    "B_per_link_MHz": B_LINK_MHZ,
    "path_loss_dB": {
        f"h={h}": {f"d3D_{int(d)}m": round(pl_at(h, d), 2) for d in (100, 200, 300)}
        for h in alts
    },
    "achievable_rate_Mb_s": {
        f"h={h}": {f"d3D_{int(d)}m": round(rate_at(h, d, 0.25), 2) for d in (100, 200, 300)}
        for h in alts
    },
    "demand_ramp_Mb_s": [S.DEMAND_LO, S.DEMAND_HI],
    "P_range_W": [S.P_MIN, S.P_MAX],
    "P_LOS": {
        f"h={h}": {
            "d2D_20m": round(float(ch3g.los_probability(np.array([20.0]), h)[0]), 3),
            "d2D_100m": round(float(ch3g.los_probability(np.array([100.0]), h)[0]), 3),
            "d2D_200m": round(float(ch3g.los_probability(np.array([200.0]), h)[0]), 3),
        } for h in alts
    },
}
with open("channel_calib.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=1)
print("[calib] channel_calib.json")
print("path loss (dB):", json.dumps(cal["path_loss_dB"]))
print("rate @0.25W (Mb/s):", json.dumps(cal["achievable_rate_Mb_s"]))
