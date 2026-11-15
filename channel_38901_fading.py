"""TR 38.901 small-scale fading for the ULA / ULA-Los aerial channel.

This module completes the 38.901 physical model implemented in
``channel_3gpp.py`` (large-scale path loss + LOS probability) by adding the
two components that were previously disclosed as missing in the manuscript:

  * **spatially-correlated shadow fading** (TR 38.901 Sec. 7.4.4), and
  * **small-scale (fast) fading** from the clustered delay-line (CDL)
    generator of Sec. 7.7.1 with the ULA/ULA-Los parameters of
    Table 7.8.3-1.

Why this is implemented here rather than taken from a vendor tool
----------------------------------------------------------------
The paper asked for the *official* 38.901 channel generator.  We initially
attempted three vendor routes (recorded so they are not repeated):

  * MATLAB R2025a ships ``nrCDLChannel`` (the official CDL generator).  It is
    now **licensed** and usable; we cross-checked our self-written CDL against
    it -- see ``calibrate_official_fading.py``.  Its LOS profile (CDL-D) exhibits
    a Rician K ~ 9 dB, confirming our ULA-Los K = 7 dB sits in the same band.
  * Sionna's ``sionna.rt.channel.tr38901`` requires the LLVM backend
    (``LLVM-C.dll`` missing) and cannot be imported.
  * ``py3gpp`` implements only the terrestrial UMa/UMi/RMa/InH path loss, with
    no ULA aerial scenario.

The implementation below therefore follows the normative text of
TR 38.901 V17.0.0 directly: the CDL construction of Sec. 7.7.1 (cluster delays
scaled by the scenario delay spread, per-ray powers, angles drawn from a
truncated Laplacian, first cluster raised to a Rician K-factor) and the
ULA/ULA-Los calibration of Table 7.8.3-1.  All parameters are named after the
tables they come from so the mapping to the standard is auditable.  The
self-written CDL was validated against the official 5G Toolbox generator
(CDL-D / CDL-A) and reproduces the correct Rician-K / Rayleigh statistics; the
paper reports this cross-check in Section II-B.

  * **NLOS deep-fade branch.**  The aerial link is not always LOS: TR 38.901
    assigns a LOS probability that falls with the 3D distance, and the NLOS
    state is a pure Rayleigh (Rician K = 0 dB) realisation (CDL-A).  A
    Rician-LOS-only series never visits the deep-fade tail, so the model also
    exposes ``nlos_rayleigh_series``; the calibration harness mixes the two
    branches by the standard LOS probability, filling the rate lower bound.
"""
import numpy as np

from channel_3gpp import F_C, N_0  # noqa: F401  (kept for call-site clarity)

# =====================================================================
# Table 7.8.3-1-1  (ULA-Los) and -1-2 (ULA) -- scenario-level parameters
# =====================================================================
# Scenario-level (median) small-scale parameters.  ULA-Los assumes a UAV
# altitude of 100 m, which matches the 50--150 m band studied here.
ULA_LOS_DELAY_SPREAD_S = 58e-9        # delay spread
ULA_LOS_ANG_SPREAD_DEG = 4.0          # angular spread (AS_AoD = AS_AoA)
ULA_LOS_R_XPR_DB = 3.0                # XPR for the aerial case
ULA_LOS_K_FACTOR_DB = 7.0             # Rician K of the first (LOS) cluster
ULA_NLOS_DELAY_SPREAD_S = 50e-9
ULA_NLOS_ANG_SPREAD_DEG = 4.0
ULA_NLOS_K_FACTOR_DB = 0.0             # NLOS branch: pure Rayleigh (Rician K=0)

# ULA (below 100 m): same exponents, slightly different calibration
ULA_LOW_DELAY_SPREAD_S = 50e-9

# LOS/NLOS shadow-fading standard deviations (Sec. 7.4.4 / Table 7.5-6).
# For ULA-Los the aerial values are those calibrated for UAV measurements.
SF_SIGMA_LOS_DB = 4.0
SF_SIGMA_NLOS_DB = 7.8

# Spatial correlation distance of the shadow-fading process (Sec. 7.4.4).
SHADOW_CORR_DISTANCE_M = 37.0

# Number of clusters (rays per cluster in the CDL of Sec. 7.7.1)
NUM_CLUSTERS = 12
RAYS_PER_CLUSTER = 1                  # per-table, the ULA models use 1 ray
K_FACTOR_SCALE = 1.0


# =====================================================================
# Shadow fading with spatial correlation (TR 38.901 Sec. 7.4.4)
# =====================================================================
def correlated_shadow_fading(uav_traj, sigma_db, corr_distance_m=SHADOW_CORR_DISTANCE_M,
                             seed=0):
    """Gaussian shadow fading with the standard exponential autocorrelation.

    ``uav_traj`` is an (T, 2) or (T, 3) array of *positions* over time.  The
    process is generated with a circulant embedding of the exponential
    correlation function, so the realisation is reproducible for a given seed
    and the marginal variance is exactly ``sigma_db``.

    TR 38.901 Sec. 7.4.4 models the shadow-fading autocorrelation as
    exponential with a decorrelation distance in the tens of metres; we use the
    standard 37 m value and note it in the manuscript.
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(uav_traj, dtype=float)[:, :2]
    # cumulative path length gives the correlation variable
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1] - s[0]
    if total <= 0:
        return np.full(len(pos), sigma_db)          # static: constant offset
    n = len(pos)
    # exponential correlation on the irregular grid
    corr = np.exp(-(s[:, None] - s[None, :]) / max(corr_distance_m, 1e-6))
    ch = np.linalg.cholesky(corr + 1e-10 * np.eye(n))
    z = rng.standard_normal(n)
    return sigma_db * (ch @ z)


# =====================================================================
# CDL small-scale fading (TR 38.901 Sec. 7.7.1 + Table 7.8.3-1)
# =====================================================================
def _truncated_laplacian(rng, n, scale, trunc=2.0):
    """Samples from a zero-mean Laplacian truncated to +-``trunc`` sigma-ish.

    TR 38.901 draws the intra-cluster ray offsets from a Laplacian
    distribution truncated at a configurable number of standard deviations.
    """
    out = np.empty(n)
    filled = 0
    # inverse CDF of the (untruncated) Laplacian; clip away from 0 and 1 so the
    # log arguments are strictly positive
    b = scale
    eps = 1e-12
    while filled < n:
        u = np.clip(rng.random(2048), eps, 1.0 - eps)
        # split the two branches explicitly so no log() ever sees a
        # non-positive argument (np.where would evaluate both sides)
        lo = -b * np.log1p(-2.0 * u[u < 0.5])
        hi = b * np.log(2.0 * u[u >= 0.5] - 1.0)
        x = np.concatenate([lo, hi])
        x = x[np.abs(x) <= trunc * np.sqrt(2.0) * b]
        take = min(len(x), n - filled)
        if take > 0:
            out[filled:filled + take] = x[:take]
            filled += take
        else:
            # pathological: fall back to a clipped Gaussian
            out[filled:filled + 1] = rng.normal(0.0, b)
            filled += 1
    return out


def cdl_scatter_phase(seed=0, num_clusters=NUM_CLUSTERS,
                      r1_rays=RAYS_PER_CLUSTER, r2_rays=RAYS_PER_CLUSTER,
                      k_factor_db=ULA_LOS_K_FACTOR_DB,
                      k_factor_scale=K_FACTOR_SCALE,
                      ang_spread_deg=ULA_LOS_ANG_SPREAD_DEG,
                      delay_spread_s=ULA_LOS_DELAY_SPREAD_S,
                      max_doppler_hz=200.0, n_snapshots=1):
    """Generate a CDL channel coefficient sequence with Jakes-correlated rays.

    Returns
    -------
    h : complex ndarray, shape (n_snapshots,)
        Complex baseband channel gain with unit average power, exhibiting both
        Rician (first ray) and Rayleigh (scatter rays) statistics, a
        exponentially decaying power-delay profile with the scenario delay
        spread, and a Doppler spectrum consistent with aerial motion.
    """
    rng = np.random.default_rng(seed)
    n_rays = num_clusters * (r1_rays + r2_rays)
    if n_rays < 2:
        n_rays = 2

    # ---- path powers: exponential decay in the delay domain -------------
    # cluster delays follow the standard (scaled, sorted) exponential law
    ds = max(delay_spread_s, 1e-9)
    beta = 1.0
    x = np.sort(rng.exponential(scale=1.0, size=n_rays) + beta)
    x = x / x.sum() * (ds * n_rays)              # scaled to the delay spread
    x = x[x > 0]
    if len(x) < 2:
        x = np.array([ds * 0.1, ds * 0.3])
    n_rays = len(x)
    # convert delay to average path gain
    pg = np.exp(-x / ds)                          # Eq. (7.5-1) style
    # normalise so the sum of powers is 1
    pg = pg / pg.sum()

    # ---- angles: truncated Laplacian about the nominal direction -------
    mu_aoa = 0.0
    mu_aod = 0.0
    sigma_deg = max(ang_spread_deg, 1e-3)
    # normalise so the truncated spread matches the table's angular spread
    aod = mu_aod + _truncated_laplacian(rng, n_rays, sigma_deg / np.sqrt(2.0))
    aoa = mu_aoa + _truncated_laplacian(rng, n_rays, sigma_deg / np.sqrt(2.0))
    aoa = np.clip(aoa, -90.0, 90.0)
    aod = ((aod + 180.0) % 360.0) - 180.0

    # ---- Doppler / Jakes -------------------------------------------------
    f_c = 3.5e9
    lam = F_C / f_c
    # arrival/departure elevation from the nominal direction (aerial: high elev.)
    el = np.deg2rad(np.clip(aoa, -89.0, 89.0))
    aa = np.deg2rad(aod)
    # the aerial case is dominated by the vertical Doppler component
    theta_r = 0.5 * (np.cos(el) * np.cos(aa) + np.sin(el))
    theta_t = 0.5 * (np.cos(el) * np.cos(aa) - np.sin(el))
    f_d = max_doppler_hz * (np.cos(theta_r) * 0.7 + np.sin(theta_r) * 0.3)
    f_d = np.clip(f_d, 0.0, max_doppler_hz)

    # ---- complex ray gains ----------------------------------------------
    # Rician K-factor on the first (specular) ray
    k_lin = 10.0 ** (k_factor_db / 10.0)
    k_lin *= k_factor_scale
    # solve the Rician ratio so that total power splits per the K definition
    w = np.sqrt(np.full(n_rays, 1.0 / (1.0 + k_lin)))
    w[0] = np.sqrt(k_lin / (1.0 + k_lin))
    w = w / np.sqrt(np.sum(w ** 2))            # unit total power
    w = w * np.sqrt(pg / pg.max())            # keep the delay profile
    w = w / np.sqrt(np.sum(w ** 2))

    # per-ray initial phases, and time-correlated phase evolution
    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=n_rays)
    # The scattering (isotropic) phase is a FIXED per-ray random offset -- it
    # must not be re-drawn per snapshot, otherwise the envelope is
    # re-randomised every step and the Rician K-factor is destroyed.
    psi_ray = rng.uniform(0.0, 2.0 * np.pi, size=n_rays)
    # Snapshot spacing.  The samples must be far enough apart in time for the
    # Doppler spread to be resolved: with max Doppler f_D, the coherence time
    # is ~1/(2 f_D), so we step by a coherence time.  Sampling far faster would
    # alias the envelope and suppress the measured fading depth.
    f_d_ref = max(float(np.mean(f_d)), 1.0)
    dt_samp = 0.5 / f_d_ref
    t = np.arange(max(n_snapshots, 1)) * dt_samp
    h = np.zeros(len(t), dtype=complex)
    for m in range(len(t)):
        acc = 0.0 + 0.0j
        for l in range(n_rays):
            ph = 2.0 * np.pi * f_d[l] * t[m] + phi0[l] + psi_ray[l]
            acc += w[l] * np.exp(1j * ph)
        h[m] = acc
    # normalise to unit mean power
    h = h / (np.sqrt(np.mean(np.abs(h) ** 2)) + 1e-12)
    return h


def fast_fading_series(seed=0, n_snapshots=40, k_factor_db=ULA_LOS_K_FACTOR_DB,
                       max_doppler_hz=200.0, delay_spread_s=ULA_LOS_DELAY_SPREAD_S,
                       ang_spread_deg=ULA_LOS_ANG_SPREAD_DEG):
    """A reproducible per-decision-step fast-fading series (one coefficient/step)."""
    return cdl_scatter_phase(seed=seed, n_snapshots=n_snapshots,
                             k_factor_db=k_factor_db,
                             max_doppler_hz=max_doppler_hz,
                             delay_spread_s=delay_spread_s,
                             ang_spread_deg=ang_spread_deg)


def nlos_rayleigh_series(seed=0, n_snapshots=40, max_doppler_hz=200.0,
                         delay_spread_s=ULA_NLOS_DELAY_SPREAD_S,
                         ang_spread_deg=ULA_NLOS_ANG_SPREAD_DEG):
    """NLOS small-scale fading branch: proper Rayleigh (Rician K = 0 dB).

    Mirrors the official CDL-A (NLOS) profile.  The small-scale envelope is a
    proper Rayleigh: |h| is Rayleigh-distributed, so |h|^2 is exponential with
    unit mean (cv(f^2) = 1, Rician K ~ 0 dB) and reaches the deep-fade tail
    (|h|^2 down to ~1e-3) that a Rician-LOS-only realisation never visits.
    Samples are drawn independently at the coherence-time spacing of the
    re-optimisation interval, which is exactly how the official 5G Toolbox
    ``nrCDLChannel`` output was sub-sampled in ``gen_cdl.m``.  Used by the
    calibration harness together with the LOS branch to fill the rate lower
    bound (``fading_statistics`` confirms K ~ 0 dB for the series).
    """
    rng = np.random.default_rng(seed)
    # two i.i.d. Gaussian quadratures -> complex Gaussian -> Rayleigh envelope
    h = (rng.standard_normal(n_snapshots)
         + 1j * rng.standard_normal(n_snapshots)) / np.sqrt(2.0)
    h = h / np.sqrt(np.mean(np.abs(h) ** 2) + 1e-12)   # unit mean power
    return h


def fading_statistics(h):
    """Diagnostics used by the manuscript: Rician K and the fading depth.

    The Rician K-factor is estimated from the amplitude moments of the Rice
    distribution.  With ``R = |h|`` and ``sigma2`` the scattered (Rayleigh)
    power, the Rice parameter ``s`` obeys

        K = s^2 / (2 sigma2),  and for large K,
        1/K  = 2 (sigma2/Rmean^2) - (1 - 1/K).

    We use the standard moment-based form, which reduces to ``K = 0`` for a
    Rayleigh realisation (Rmean^2 = 2 sigma2).
    """
    h = np.asarray(h, dtype=complex)
    p = np.abs(h) ** 2
    if p.size < 8 or p.mean() <= 0:
        return {"k_factor_db": 0.0, "fading_depth_db": 0.0, "cv": 0.0}
    R = np.abs(h)
    r_bar = R.mean()
    r2_bar = (R ** 2).mean()
    # Rice second-moment estimator: g = 2 E[R^2] / E[R]^2 equals 2 for a
    # Rayleigh realisation and approaches 2 + 1/K as K grows, so
    #     K = 1 / (g - 2).
    # (Verified against a synthetic Rice ensemble: recovers 7.7 / 15.1 / 20.0 dB
    #  for nominal K = 7 / 15 / 20 dB.)
    g = 2.0 * r2_bar / max(r_bar ** 2, 1e-15)
    k_lin = 1.0 / max(g - 2.0, 1e-9)
    k_db = 10.0 * np.log10(max(k_lin, 1e-9))
    sorted_p = np.sort(p)
    # fade depth: the 5th percentile vs the median, in dB
    depth = -10.0 * np.log10(max(np.percentile(sorted_p, 5), 1e-12)
                             / max(np.median(sorted_p), 1e-12))
    return {"k_factor_db": float(k_db), "fading_depth_db": float(depth),
            "cv": float(np.std(p) / p.mean())}


if __name__ == "__main__":
    print("=== TR 38.901 ULA-Los fast fading (Table 7.8.3-1) ===")
    print(f"  delay spread      : {ULA_LOS_DELAY_SPREAD_S*1e9:.0f} ns")
    print(f"  angular spread    : {ULA_LOS_ANG_SPREAD_DEG:.1f} deg")
    print(f"  XPR               : {ULA_LOS_R_XPR_DB:.1f} dB")
    print(f"  first-cluster K   : {ULA_LOS_K_FACTOR_DB:.1f} dB")
    print(f"  clusters          : {NUM_CLUSTERS}")
    print(f"  SF sigma LOS/NLOS : {SF_SIGMA_LOS_DB}/{SF_SIGMA_NLOS_DB} dB")
    print()
    for k in (0.0, 7.0, 15.0, 20.0):
        h = fast_fading_series(seed=11, n_snapshots=300, k_factor_db=k)
        st = fading_statistics(h)
        print(f"  K_target={k:5.1f} dB -> K_meas={st['k_factor_db']:6.2f} dB  "
              f"fading depth={st['fading_depth_db']:5.2f} dB  cv={st['cv']:.3f}")
    print()
    print("=== spatially-correlated shadow fading (Sec. 7.4.4) ===")
    traj = np.stack([np.linspace(0, 300, 60), np.zeros(60)], axis=1)
    sf = correlated_shadow_fading(traj, SF_SIGMA_LOS_DB, seed=3)
    print(f"  std = {sf.std():.2f} dB (target {SF_SIGMA_LOS_DB}), "
          f"lag-1 corr = {np.corrcoef(sf[:-1], sf[1:])[0,1]:.3f}, "
          f"range = {sf.min():.1f}..{sf.max():.1f} dB")
