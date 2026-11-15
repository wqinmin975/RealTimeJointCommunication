"""3GPP TR 38.901 aerial (UAV) channel model for the airborne FM scheduler.

Implements the ULA / ULA-Los (Urban LOS / Urban LOS with 150 m UAV height)
models of 3GPP TR 38.901 Table 5.4.1, plus the altitude-dependent LOS
probability of Table 5.4.1 item 8, and the associated shadow fading.

Design goals relative to the previous idealised free-space law G0/(d^2+eps):

 1. Provide an *absolute* path-loss reference in dB (so transmit power and
    bandwidth budgets become physically meaningful instead of arbitrary).
 2. Make the achievable rate depend on 3D geometry *and* on UAV altitude, so
    the "higher UAV -> better link" behaviour of real air-ground links appears.
 3. Give a non-degenerate feasible set for the SCA: with the old G0 = 1e-4 the
    rate constraint was inactive at realistic distances and the objective was
    invariant to (p, b, u); here the constraint binds inside the operating
    region by construction (see calibrate()).

Everything is deterministic given ``seed`` so the v1--v5 ablation stays
reproducible.

Reference: 3GPP TR 38.901 V17.0.0, Sec. 5.4.1 (Table 5.4.1-1/-2).
"""
import numpy as np

# ------------------------------------------------------------------ constants
# TR 38.901 Table 5.4.1-1 (ULA) and -2 (ULA-Los), valid 4 < d2D < 500 m and
# 15 < h < 300 m for the ULA-Los altitude-dependent LOS probability.
P_LOS_ULA_D2_MIN = 4.0        # m, horizontal-distance validity lower bound
H_UAV_MIN, H_UAV_MAX = 15.0, 300.0   # m, ULA-Los validity range

# Table 5.4.1-1: LOS / NLOS path-loss exponents (3D distance)
N_LOS_ULA = 2.3
N_NLOS_ULA = 3.5
# Table 5.4.1-2: ULA-Los uses the same exponents but adds the altitude terms
N_LOS_ULA_LOS = 2.3
N_NLOS_ULA_LOS = 3.5

# LOS probability (Table 5.4.1 item 8, ULA-Los)
C_LOS = 9.61                 # m
D_LOS = 0.16

# --- link budget (aerial edge-intelligence deployment) ---------------------
# Carrier, antenna gains, and the reference bandwidth over which the thermal
# noise floor is defined. Values follow standard UAV air-to-ground deployments:
# a 3.5 GHz payload downlink with a moderately directional UAV antenna and a
# handheld/ground-side antenna, over a 10 MHz OFDM slice.
F_C = 299_792_458.0          # m/s, speed of light
F_HZ = 3.5e9                 # Hz, carrier
G_TX_DBI = 4.0               # dBi, UAV payload antenna (moderate directionality)
G_RX_DBI = 2.0               # dBi, ground user antenna
N_0 = 1.380649e-23 * 290.0   # W/Hz, thermal noise density at ~290 K
B_REF_HZ = 10e6              # Hz, reference bandwidth (B_tot)
G_REF = 1.0                  # retained for API compatibility


def wavelength(f_hz: float) -> float:
    return F_C / f_hz


def d_2d(uav, user):
    """Horizontal (ground-plane) separation, metres. Shapes (...,2)."""
    dx = uav[..., 0] - user[..., 0]
    dy = uav[..., 1] - user[..., 1]
    return np.sqrt(dx * dx + dy * dy)


def d_3d(uav, user):
    """True 3D separation, metres. Shapes (...,3)."""
    diff = np.asarray(uav, dtype=float) - np.asarray(user, dtype=float)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def elevation_angle(uav, user):
    """theta = atan2(h_uav - h_user, d2D) in degrees."""
    d2 = d_2d(uav, user)
    dh = np.asarray(uav)[..., 2] - np.asarray(user)[..., 2]
    return np.degrees(np.arctan2(dh, np.maximum(d2, 1e-6)))


def los_probability(d2, h_uav):
    """P_LOS from the ULA-Los altitude-dependent rule (TR 38.901 Table 5.4.1-1)."""
    d2c = np.maximum(d2, P_LOS_ULA_D2_MIN)
    hc = np.clip(h_uav, H_UAV_MIN, H_UAV_MAX)
    p = 1.0 / (1.0 + C_LOS / np.maximum(d2c, 1e-6)) * (
        hc * hc / (hc * hc + D_LOS * 1000.0)
    )
    return np.clip(p, 0.0, 1.0)


def path_loss_db(f_hz, uav, user, los_mask, shadow_sigma=0.0, rng=None):
    """3GPP TR 38.901 ULA/ULA-Los path loss in dB.

    ``los_mask`` : boolean array, True where the link is LOS.
    LOS branch (ULA-Los) follows the altitude-aware form

        PL_LOS = 28.0 + 22 log10(d3D) + 20 log10(f_GHz) - 9 log10(d_BP'^2 + dh^2)
        d_BP'  = 4 (h_BS - h_UT) / lambda          (breakpoint distance)

    NLOS branch

        PL_NLOS = 13.54 + 39.08 log10(d3D) + 20 log10(f_GHz) - 0.6 (h - 1.5)

    with the NLOS total taken as max(PL_LOS, PL_NLOS) per the standard.
    """
    f_ghz = f_hz / 1e9
    lam = wavelength(f_hz)

    d2 = d_2d(uav, user)
    d3 = np.maximum(d_3d(uav, user), 1.0)
    h_uav = np.asarray(uav, dtype=float)[..., 2]
    h_user = np.asarray(user, dtype=float)[..., 2]
    dh = h_uav - h_user

    # LOS (ULA-Los, altitude-aware)
    d_bp = 4.0 * np.abs(dh) / lam
    pl1 = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(f_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3) + 20.0 * np.log10(f_ghz)
           - 9.0 * np.log10(d_bp ** 2 + dh ** 2 + 1e-9))
    pl_los = np.where(d2 <= d_bp, pl1, pl2)

    # NLOS
    pl_nlos = (13.54 + 39.08 * np.log10(d3) + 20.0 * np.log10(f_ghz)
               - 0.6 * (h_uav - 1.5))
    pl_nlos = np.maximum(pl_los, pl_nlos)

    pl = np.where(los_mask, pl_los, pl_nlos)

    if shadow_sigma > 0.0:
        assert rng is not None, "shadow fading requested without rng"
        pl = pl + rng.normal(0.0, shadow_sigma, size=pl.shape)
    return pl


def gain_from_db(pl_db, d_ref, f_hz, g_ref=G_REF, b_ref_hz=None):
    """Convert a dB path loss into a *linear SNR-normalised* channel gain.

    The solver's rate expression is  R = b log2(1 + p g).  For that to equal a
    true Shannon rate we need  g  to be referenced to the thermal noise floor of
    the bandwidth over which the rate is quoted, i.e.

        g = (G_tx G_rx 10^(-PL/10)) / (N_0 b_ref)

    so that with b = b_ref the argument of the log is exactly the SNR.  This is
    the standard "close-in reference" arrangement; it keeps g dimensionless and
    O(1) inside the operating region, which is what makes the rate constraint
    bind for physically meaningful demands.

    ``b_ref_hz`` should be the bandwidth the rate is quoted over -- for a
    scheduler that splits B_tot across K links, that is B_tot / K.  Passing
    ``None`` falls back to the module-level ``B_REF_HZ``.
    """
    b_ref = B_REF_HZ if b_ref_hz is None else b_ref_hz
    g_ant = 10.0 ** ((G_TX_DBI + G_RX_DBI) / 10.0)
    n0_b = N_0 * b_ref
    return g_ant * 10.0 ** (-pl_db / 10.0) / n0_b


def channel_gains(uavs, users, f_hz, d_ref=1.0, shadow_sigma=0.0, seed=0,
                  los_override=None, b_ref_hz=None):
    """Linear channel-gain matrix (K x N) under the 3GPP aerial model.

    ``los_override`` (array of bool, K x N) forces the LOS class, which is used
    by the convergence study to obtain a well-conditioned, rate-limited regime.
    ``b_ref_hz`` is the bandwidth over which the resulting rate is quoted (see
    ``gain_from_db``); the scheduler passes B_tot / K.
    """
    uavs = np.asarray(uavs, dtype=float)
    users = np.asarray(users, dtype=float)
    K, N = uavs.shape[0], users.shape[0]
    U = np.repeat(uavs[:, None, :], N, axis=1)      # (K,N,3)
    X = np.repeat(users[None, :, :], K, axis=0)      # (K,N,3)

    d2 = d_2d(U, X)
    h = U[..., 2]
    if los_override is not None:
        los = np.asarray(los_override, dtype=bool)
    else:
        p = los_probability(d2, h)
        los = np.random.default_rng(seed).random((K, N)) < p

    pl = path_loss_db(f_hz, U, X, los, shadow_sigma=shadow_sigma,
                      rng=np.random.default_rng(seed + 1))
    return gain_from_db(pl, d_ref=d_ref, f_hz=f_hz, b_ref_hz=b_ref_hz)


def rate_bps(b_hz, p_w, g):
    """Shannon rate with the TR 38.901-derived gain (b in Hz, p in W)."""
    return b_hz * np.log2(1.0 + p_w * g)


# ------------------------------------------------------------- calibration
def calibrate(f_hz=None, shadow_sigma=0.0, p_w=1.0, b_hz=None, demands=(1.0, 3.0, 8.0)):
    """Report the SNR / rate landscape so the paper can quote physical numbers.

    Prints, versus 3D range and UAV altitude, the 3GPP path loss, the resulting
    SNR at ``p_w`` / ``b_hz``, and the achievable rate. Also reports, for each
    demand in ``demands``, the maximum range at which the rate constraint is
    still satisfiable -- i.e. the radius of the rate-feasible set that the
    scheduler must track. This replaces the old arbitrary G0 = 1e-4, under
    which the constraint was inactive everywhere and the objective degenerate.
    """
    f_hz = F_HZ if f_hz is None else f_hz
    b_hz = B_REF_HZ if b_hz is None else b_hz
    out = {"f_hz": f_hz, "p_w": p_w, "b_hz": b_hz}

    print("=== 3GPP TR 38.901 ULA-Los calibration (SNR-referenced gain) ===")
    print(f"carrier={f_hz/1e9:.2f} GHz  P={p_w:.2f} W  B={b_hz/1e6:.1f} MHz  "
          f"G_tx+G_rx={G_TX_DBI + G_RX_DBI:.0f} dBi  N0B={10*np.log10(N_0*b_hz):.1f} dBm")
    print()
    print(f"{'d2D[m]':>7} {'h[m]':>5} {'PL[dB]':>8} {'SNR[dB]':>8} {'rate[Mb/s]':>11}")
    for d2 in (20, 50, 100, 200, 300, 500):
        for h in (50, 100, 150):
            uav = np.array([[100.0, 100.0, float(h)]])
            usr = np.array([[100.0 + float(d2), 100.0, 0.0]])
            los = np.array([[True]])
            pl = float(path_loss_db(f_hz, uav, usr, los)[0, 0])
            g = float(gain_from_db(pl, d_ref=None, f_hz=f_hz))
            snr_db = 10.0 * np.log10(max(p_w * g, 1e-30))
            rate = float(rate_bps(b_hz, p_w, g)) / 1e6
            print(f"{d2:>7} {h:>5} {pl:>8.1f} {snr_db:>8.1f} {rate:>11.2f}")
    print()

    # max feasible range per demand, at a mid altitude
    print("rate-feasible radius (LOS, h=100 m, p=%.1f W, b=%.0f MHz):" % (p_w, b_hz/1e6))
    radii = {}
    for dem in demands:
        lo, hi = 4.0, 2000.0
        for _ in range(40):                      # bisect on d2D
            mid = 0.5 * (lo + hi)
            uav = np.array([[100.0, 100.0, 100.0]])
            usr = np.array([[100.0 + mid, 100.0, 0.0]])
            g = float(gain_from_db(
                float(path_loss_db(f_hz, uav, usr, np.array([[True]]))[0, 0]),
                d_ref=None, f_hz=f_hz))
            if rate_bps(b_hz, p_w, g) >= dem:
                lo = mid
            else:
                hi = mid
        radii[dem] = lo
        print(f"  demand {dem:>5.1f} Mb/s -> feasible up to d2D = {lo:7.1f} m")
    out["feasible_radius_m"] = radii
    print()
    return out


if __name__ == "__main__":
    calibrate()
    print("=== sanity: LOS probability vs distance/altitude ===")
    for h in (50, 100, 150):
        ps = [f"{d:>3}m:{los_probability(np.array([float(d)]), h)[0]:.3f}"
              for d in (20, 50, 100, 200, 400)]
        print(f"  h={h:>3}m  P_LOS  " + "  ".join(ps))
    print()
    print("=== sanity: NLOS vs LOS at 3.5 GHz, d3D=150 m, h=100 m ===")
    uav = np.array([[0.0, 0.0, 100.0]])
    usr = np.array([[150.0, 0.0, 0.0]])
    for tag, m in (("LOS", True), ("NLOS", False)):
        pl = float(path_loss_db(F_HZ, uav, usr, np.array([[m]]))[0, 0])
        g = float(gain_from_db(pl, d_ref=None, f_hz=F_HZ))
        print(f"  {tag:>4}: PL={pl:6.1f} dB  g={g:.3e}  "
              f"rate@10MHz/1W={rate_bps(B_REF_HZ, 1.0, g)/1e6:8.2f} Mb/s")
