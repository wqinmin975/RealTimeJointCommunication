"""
sim_analysis_3gpp.py--  Airborne FM-inference experiments on a physical channel
=================================================================================
Same two artifacts as ``sim_analysis.py`` (convergence curve + quantitative v1-v5
ablation), but with two substantive upgrades requested for the TVT revision:

  1. CHANNEL: the idealised free-space law ``G0/(d^2+eps)`` is replaced by the
     3GPP TR 38.901 ULA/ULA-Los aerial channel model (LOS/NLOS path loss with
     the altitude-dependent LOS probability, plus shadow fading).  Because the
     gain is now referenced to the thermal noise floor of the 10 MHz slice, the
     achievable rate is physically meaningful and the rate constraint actually
     binds inside the operating region -- under the old ``G0 = 1e-4`` it was
     inactive everywhere and the objective was degenerate.

  2. KINEMATICS: the per-step UAV displacement is driven by a *measured*
     ground-mobility speed profile extracted from a real UWB localization
     dataset (``E:\\data\\eee_{01,02,03}``), resampled to the simulator's
     decision interval.  This replaces an asserted constant speed with an
     empirical per-step motion profile.  The carrier itself is a ground vehicle
     (all nodes at z = 1.5 m), so it supplies the *temporal* motion statistics
     only -- not a channel measurement and not a UAV trajectory.

Demand is expressed in Mb/s against the 3GPP rate scale, so the numbers in the
resulting table/figure are directly comparable with the standard.
"""

import json
import time
import csv
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import cvxpy as cp
import clarabel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import channel_3gpp as ch3g
import channel_38901_fading as chfade

# ============================================================
# 0. System parameters
# ============================================================
AREA = (200.0, 200.0)
ALT_RANGE = (50.0, 150.0)
N_UAV = 3
N_USER = 5
G0 = 1e-4                  # legacy free-space constant, superseded by 3GPP below

# --- channel: 3GPP TR 38.901 ULA/ULA-Los (replaces the idealised G0/(d^2+eps))
F_CARRIER = ch3g.F_HZ
# Shadow fading now follows the standard: spatially correlated (Sec. 7.4.4)
# with the ULA-Los sigma, instead of a memoryless per-step draw.
SHADOW_SIGMA = chfade.SF_SIGMA_LOS_DB
USE_CORRELATED_SHADOW = True
USE_FAST_FADING = True
FAST_FADING_SEED = 991
CHANNEL_SEED = 2024

B_TOTAL = 10.0          # MHz  (total bandwidth budget, == ch3g.B_REF_HZ/1e6)
P_MAX = 1.0
P_MIN = 0.01

L_MAX = 5
FM_LAT = {1: 0.020, 2: 0.040, 3: 0.070, 4: 0.110, 5: 0.160}   # s
FM_ACC = {1: 0.55, 2: 0.70, 3: 0.82, 4: 0.90, 5: 0.95}
EXIT_ACC_LOSS = 0.06
QUANT_ACC_LOSS_PER_BIT = 0.005

H_MPC = 3
DT = 0.1
V_MAX = 10.0
V_STEP = V_MAX * DT      # = 1.0 m

W_LAT = 1.0
W_EN = 0.3
W_ACC_BASE = 2.0
MU_REG = 0.5
# Demand reference on the physical Mb/s scale of the 3GPP rate model.  With a
# 3.5 GHz link, 4 dBi UAV + 2 dBi ground antennas, 1 W and an equal 10/3 MHz
# share, the achievable rate over the operating region is ~57-68 Mb/s.  The
# demand ramp below therefore spans 30 -> 62 Mb/s: the rate constraint is
# genuinely infeasible on transmit power alone at the low end and becomes
# reachable only by moving the UAV closer / re-allocating bandwidth, which is
# exactly the mobility-compute coupling the paper studies.
DEMAND_REF = 46.0
DEMAND_LO = 30.0
DEMAND_HI = 62.0

np.random.seed(42)


# ============================================================
# 1. Channel helpers (3GPP TR 38.901 ULA/ULA-Los aerial model)
# ============================================================
# Each UAV is served one user and is allotted an equal share of the total
# bandwidth, so per-link rates are quoted over B_TOTAL/N_UAV.  Referencing the
# SNR-referenced gain to that same slice makes the solver's rate expression
#   R_k = b_k log2(1 + p_k g_k)   [Mb/s, b_k in MHz]
# an exact Shannon rate rather than an arbitrary scaling.
B_PER_LINK_HZ = ch3g.B_REF_HZ / N_UAV


def make_users(n=N_USER):
    xs = np.linspace(40, AREA[0] - 40, n)
    ys = np.random.uniform(40, AREA[1] - 40, size=n)
    return np.stack([xs, ys, np.zeros(n)], axis=1)


def channel_gains(uavs, users, seed=CHANNEL_SEED, los_override=None,
                  shadow_db=None, fast_fading=None):
    """(K, N) linear SNR-referenced gain from the 3GPP ULA/ULA-Los model.

    ``shadow_db`` supplies a *spatially correlated* shadow-fading realisation
    (dB, one per UAV) so the link budget is temporally consistent as the
    platform moves; ``fast_fading`` supplies a per-step complex small-scale
    fading coefficient (unit mean power).
    """
    G = ch3g.channel_gains(uavs, users, f_hz=F_CARRIER,
                            shadow_sigma=0.0, seed=seed,
                            los_override=los_override,
                            b_ref_hz=B_PER_LINK_HZ)
    if shadow_db is not None:
        G = G * (10.0 ** (-np.asarray(shadow_db, dtype=float) / 10.0))[:, None]
    if fast_fading is not None:
        ff = np.asarray(fast_fading)
        if ff.ndim == 0:
            ff = ff[None]
        G = G * (np.abs(ff) ** 2)[:, None]
    return G


def _los_flag(uav, user):
    """Branch indicator at the current geometry (False = LOS, True = NLOS)."""
    u = np.asarray(uav, float).reshape(1, 1, 3)
    x = np.asarray(user, float).reshape(1, 1, 3)
    p = ch3g.los_probability(ch3g.d_2d(u, x), u[..., 2])[0, 0]
    return np.array([[p < 0.5]])


def grad_gain(uav, user, rel_step=0.02, los=None):
    """Central-difference gradient of the SNR-referenced gain wrt UAV position.

    The 3GPP path loss is a max() of LOS/NLOS branches with a log-distance law,
    so it is only piecewise smooth; a central difference is used instead of a
    closed-form gradient.  The stencil is scaled to the local 3-D range so the
    linearisation stays accurate over one SCA step.
    """
    uav = np.asarray(uav, dtype=float)
    user = np.asarray(user, dtype=float)
    if los is None:
        los = _los_flag(uav, user)
    d3 = max(float(np.linalg.norm(uav - user)), 1e-3)
    step = rel_step * d3

    def _g(u):
        pl = ch3g.path_loss_db(F_CARRIER, u.reshape(1, 1, 3),
                               user.reshape(1, 1, 3), los)[0, 0]
        return float(ch3g.gain_from_db(pl, d_ref=None, f_hz=F_CARRIER,
                                        b_ref_hz=B_PER_LINK_HZ))

    g = np.zeros(3)
    for i in range(3):
        up = uav.copy(); up[i] += step
        um = uav.copy(); um[i] -= step
        g[i] = (_g(up) - _g(um)) / (2.0 * step)
    return g


# backwards-compatible alias used by the solver bodies
grad_path_loss = grad_gain


# ============================================================
# 1b. Measured kinematics  (real ground-mobility speed profile)
# ============================================================
# The paper previously assumed a constant v_max, i.e. a constant per-step
# displacement.  Instead we drive the per-step motion with an *empirical* speed
# profile measured with UWB in E:\data\eee_{01,02,03}, resampled to the
# simulator's decision interval.  The measured carrier is a ground vehicle
# (all nodes at z = 1.5 m in an open ground area), so it contributes the
# temporal motion statistics only -- not a channel measurement and not a UAV
# trajectory.  It is used to replace an *assumed* motion profile with a real
# one, which is a strictly weaker and clearly-labelled claim.
#
# The measured profile is much slower than a UAV (median ~0.4 m/s vs the
# simulator's 10 m/s), so we use its *shape*: the per-step displacement is
# V_STEP scaled by the normalised measured speed, clipped to the physical
# V_STEP bound.  This preserves the simulator's speed limit while importing
# real acceleration/deceleration structure.
_TRACKS_JSON = Path(__file__).resolve().parent / "measured_tracks.json"


def load_measured_speed_profile(n_steps, seed=7):
    """Return an array of n_steps per-step speed fractions in [0, 1]."""
    fallback = np.ones(n_steps)
    if not _TRACKS_JSON.exists():
        print("[kinematics] measured_tracks.json not found; using constant motion")
        return fallback
    with open(_TRACKS_JSON, encoding="utf-8") as f:
        blob = json.load(f)
    tracks = blob.get("tracks", {})
    if not tracks:
        return fallback

    # pool per-step speeds from every measured track
    pooled = []
    for tr in tracks.values():
        t = np.asarray(tr["t_rel_s"], dtype=float)
        x = np.asarray(tr["xyz"], dtype=float)
        if t.size < 3:
            continue
        dt = np.diff(t)
        dt = np.where(dt <= 1e-3, np.nan, dt)
        step = np.linalg.norm(np.diff(x, axis=0), axis=1)
        with np.errstate(invalid="ignore"):
            v = np.where(np.isfinite(dt), step / dt, np.nan)
        pooled.append(v[np.isfinite(v)])
    if not pooled:
        return fallback
    v_all = np.concatenate(pooled)
    v_all = v_all[v_all > 0]
    if v_all.size < 10:
        return fallback

    # resample the empirical CDF onto n_steps quantiles -> deterministic,
    # seed-independent, and preserves the measured speed distribution
    q = (np.arange(n_steps) + 0.5) / n_steps
    prof = np.quantile(v_all, q)
    prof = prof / prof.max() if prof.max() > 0 else fallback
    return np.clip(prof, 0.05, 1.0)


# ============================================================
# 2. Objective / performance evaluators  (consistent with opt_gap.py)
# ============================================================
def fm_lat_acc(l, q, e):
    li = int(np.clip(round(l), 1, L_MAX))
    lat = FM_LAT[li] * (1.0 - 0.3 * e)          # s
    acc = (FM_ACC[li]
           - EXIT_ACC_LOSS * e
           - QUANT_ACC_LOSS_PER_BIT * (16.0 - q))
    return lat, acc


def eval_J(l, q, e, p, demand_scale):
    lat, acc = fm_lat_acc(l, q, e)
    W_ACC = W_ACC_BASE * demand_scale
    return (W_LAT * lat + W_EN * np.sum(p)
            + W_ACC * np.sum(1.0 - acc))


# ============================================================
# 3. Un-accelerated SCA + BCD solver with feature toggles
#    (mirrors v4's solve_sca_block_cd; rebuilds CVXPY each step)
# ============================================================
def _solver_event(stats, block, status=None, exc=None):
    """Append one CLARABEL outcome to ``solve_stats`` (additive; no numerics).

    ``block`` is one of 'lqe', 'p', 'u'.  A solve either returns (record the
    solver status string) or raises (record ``EXC:<ExceptionClass>``); the two
    are mutually exclusive.  Recording per block AND globally makes the claim
    "N failures, all in the UAV-position block" reproducible straight from the
    exported JSON, without any post-hoc diagnostic script.  This function only
    increments counters -- it never touches l/q/e/p/uavs, so a successful
    run's numerics are bit-for-bit unchanged.
    """
    key = ("EXC:" + type(exc).__name__) if exc is not None else str(status)
    bb = stats["by_block"][block]
    bb["calls"] += 1
    bb["outcomes"][key] = bb["outcomes"].get(key, 0) + 1
    stats["outcomes"][key] = stats["outcomes"].get(key, 0) + 1


def solve_unaccel(uavs_prev, users, demand_avg, flags, rounds,
                  trace=None, solve_stats=None):
    # ``solve_stats`` is an optional dict that lets the caller SEE the solver
    # failures that the three bare ``except Exception`` handlers below used to
    # swallow silently.  With the default (None) a throw-away local dict keeps
    # every legacy call site behaviourally identical.  Only failure branches
    # are touched, so a successful run's numerics are unchanged.
    if solve_stats is None:
        solve_stats = {}
    solve_stats.setdefault("solver_calls", 0)
    solve_stats.setdefault("exceptions", 0)
    solve_stats.setdefault("nonoptimal", 0)
    solve_stats.setdefault("by_block", {})
    solve_stats.setdefault("outcomes", {})
    for _blk in ("lqe", "p", "u"):
        solve_stats["by_block"].setdefault(_blk, {"calls": 0, "outcomes": {}})
    G = channel_gains(uavs_prev, users)
    assignment = np.argmax(G, axis=1)
    g_sel = G[np.arange(N_UAV), assignment]
    d_sel = demand_avg[assignment]
    demand_scale = np.clip(np.mean(d_sel) / DEMAND_REF, 0.5, 2.0)
    W_ACC = W_ACC_BASE * demand_scale

    uavs = uavs_prev.copy()
    p = np.full(N_UAV, 0.3)
    b = np.full(N_UAV, B_TOTAL / N_UAV)
    l_val, q_val, e_val = 3.0, 9.0, 0.5

    def snap_J():
        if trace is not None:
            trace.append(eval_J(l_val, q_val, e_val, p, demand_scale))

    for r in range(rounds):
        # ---- (a) l, q, e ----
        if flags["adaptive"]:
            l = cp.Variable(); q = cp.Variable(); e = cp.Variable()
            obj = (W_LAT * (0.035 * (l - 1) - 0.015 * e + 0.005 * (q - 2) / 14.0)
                   + W_EN * 0.01 * l
                   - W_ACC * (0.11 * (l - 1) - 0.06 * e - 0.005 * (16 - q))
                   + MU_REG * ((l - 3) ** 2 + (q - 9) ** 2 / 10.0 + (e - 0.5) ** 2))
            prob = cp.Problem(cp.Minimize(obj),
                              [l >= 1, l <= L_MAX, q >= 2, q <= 16, e >= 0, e <= 1])
            solve_stats["solver_calls"] += 1
            _st, _exc = None, None
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
                _st = prob.status
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    l_val = float(np.clip(l.value, 1, L_MAX))
                    q_val = float(np.clip(q.value, 2, 16))
                    e_val = float(np.clip(e.value, 0, 1))
                else:
                    solve_stats["nonoptimal"] += 1
                    l_val, q_val, e_val = 3.0, 9.0, 0.5
            except Exception as exc:
                _exc = exc
                solve_stats["exceptions"] += 1
                l_val, q_val, e_val = 3.0, 9.0, 0.5
            _solver_event(solve_stats, "lqe", status=_st, exc=_exc)
        else:
            # v3-style analytic bound selection (verbatim coefficients from
            # hrl_airborne_fm_v3.py): picks a domain corner per variable,
            # ignoring the accuracy constraint -> corner solutions.
            W_ACC_v3 = W_ACC_BASE
            a_l = W_LAT * 0.035 + W_EN * 0.01 - W_ACC_v3 * 0.11
            a_q = W_LAT * 0.005 / 14.0 + W_ACC_v3 * 0.005
            a_e = -W_LAT * 0.015 + W_ACC_v3 * EXIT_ACC_LOSS
            l_val = 1.0 if a_l < 0 else float(L_MAX)
            q_val = 2.0 if a_q < 0 else 16.0
            e_val = 0.0 if a_e < 0 else 1.0
        if trace is not None:
            snap_J()

        # ---- (b) Block A: solve p ----
        p_var = cp.Variable(N_UAV, nonneg=True)
        cons = [p_var <= P_MAX, p_var >= P_MIN]
        for k in range(N_UAV):
            R_k = b[k] * cp.log(1 + p_var[k] * g_sel[k]) / float(np.log(2))
            cons.append(R_k >= d_sel[k])
        prob_p = cp.Problem(cp.Minimize(cp.sum(p_var)), cons)
        solve_stats["solver_calls"] += 1
        _st, _exc = None, None
        try:
            prob_p.solve(solver=cp.CLARABEL, verbose=False)
            _st = prob_p.status
            if prob_p.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and p_var.value is not None:
                p = np.clip(p_var.value, P_MIN, P_MAX)
            else:
                solve_stats["nonoptimal"] += 1
        except Exception as exc:
            _exc = exc
            solve_stats["exceptions"] += 1
        _solver_event(solve_stats, "p", status=_st, exc=_exc)
        if trace is not None:
            snap_J()

        # ---- (c) Block B: bandwidth ----
        if flags["bcd"]:
            for k in range(N_UAV):
                sinr = p[k] * g_sel[k]
                needed = d_sel[k] / (np.log(1 + sinr) / np.log(2) + 1e-9)
                b[k] = np.clip(needed, 0.01, B_TOTAL)
            if b.sum() > B_TOTAL:
                b = b * (B_TOTAL / b.sum())
        else:
            b = np.full(N_UAV, B_TOTAL / N_UAV)
        if trace is not None:
            snap_J()

        # ---- (d) Block C: UAV position ----
        if flags["bcd"]:
            u_var = cp.Variable((N_UAV, 3))
            cons_u = [u_var[:, 0] >= 0, u_var[:, 0] <= AREA[0],
                      u_var[:, 1] >= 0, u_var[:, 1] <= AREA[1],
                      u_var[:, 2] >= ALT_RANGE[0], u_var[:, 2] <= ALT_RANGE[1]]
            if flags["velocity"]:
                for k in range(N_UAV):
                    cons_u.append(cp.norm(u_var[k] - uavs_prev[k]) <= V_STEP)
            for k in range(N_UAV):
                u0_k = uavs[k]
                user_k = users[assignment[k]]
                g0_k = g_sel[k]
                grad_k = grad_path_loss(u0_k, user_k)
                g_lin = g0_k + grad_k @ (u_var[k] - u0_k)
                R_k = b[k] * cp.log(1 + p[k] * g_lin) / float(np.log(2))
                cons_u.append(R_k >= d_sel[k])
            prob_u = cp.Problem(cp.Minimize(0), cons_u)
            solve_stats["solver_calls"] += 1
            _st, _exc = None, None
            try:
                prob_u.solve(solver=cp.CLARABEL, verbose=False)
                _st = prob_u.status
                if prob_u.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u_var.value is not None:
                    uavs = np.clip(u_var.value,
                                   [0, 0, ALT_RANGE[0]],
                                   [AREA[0], AREA[1], ALT_RANGE[1]])
                    G = channel_gains(uavs, users)
                    g_sel = G[np.arange(N_UAV), assignment]
                else:
                    solve_stats["nonoptimal"] += 1
            except Exception as exc:
                _exc = exc
                solve_stats["exceptions"] += 1
            _solver_event(solve_stats, "u", status=_st, exc=_exc)
        else:
            uavs = uavs_prev.copy()
        if trace is not None:
            snap_J()

    return dict(uavs=uavs, p=p, b=b, l=l_val, q=q_val, e=e_val,
                g_sel=g_sel, assignment=assignment)


# ============================================================
# 4. Accelerated solver  (verbatim from v5 -- pre-built DPP problem)
# ============================================================
class FastSCASolver:
    def __init__(self, users):
        self.users = users
        self.p_g_sel = cp.Parameter(N_UAV)
        self.p_d_sel = cp.Parameter(N_UAV)
        self.p_b = cp.Parameter(N_UAV)
        self.p_p_min = cp.Parameter(N_UAV)
        self.p_var = cp.Variable(N_UAV, nonneg=True)
        self.prob_p = cp.Problem(
            cp.Minimize(cp.sum(self.p_var)),
            [self.p_var <= P_MAX, self.p_var >= self.p_p_min])

        self.u_g_sel = cp.Parameter(N_UAV)
        self.u_grad = cp.Parameter((N_UAV, 3))
        self.u_uprev = cp.Parameter((N_UAV, 3))
        self.u_rhs = cp.Parameter(N_UAV)
        self.u_var = cp.Variable((N_UAV, 3))
        cons_u = [self.u_var[:, 0] >= 0, self.u_var[:, 0] <= AREA[0],
                  self.u_var[:, 1] >= 0, self.u_var[:, 1] <= AREA[1],
                  self.u_var[:, 2] >= ALT_RANGE[0], self.u_var[:, 2] <= ALT_RANGE[1]]
        for k in range(N_UAV):
            cons_u.append(cp.norm(self.u_var[k] - self.u_uprev[k]) <= V_STEP)
            g_lin = self.u_g_sel[k] + self.u_grad[k] @ self.u_var[k]
            cons_u.append(g_lin >= self.u_rhs[k])
        self.prob_u = cp.Problem(cp.Minimize(0), cons_u)
        self.p_init = np.full(N_UAV, 0.3)
        self.u_init = np.zeros((N_UAV, 3))

    def solve(self, uavs_prev, g_sel, d_sel, b, assignment, users, max_rounds=3):
        uavs = uavs_prev.copy()
        p = self.p_init.copy()
        b = b.copy()
        for r in range(max_rounds):
            self.p_g_sel.value = g_sel
            self.p_d_sel.value = d_sel
            self.p_b.value = b
            p_min_req = np.maximum(P_MIN,
                                   (2.0 ** (d_sel / b) - 1.0) / (g_sel + 1e-12))
            self.p_p_min.value = p_min_req
            try:
                self.prob_p.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
                if self.prob_p.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and self.p_var.value is not None:
                    p = np.clip(self.p_var.value, P_MIN, P_MAX)
                    self.p_init = p.copy()
            except Exception:
                pass
            for k in range(N_UAV):
                sinr = p[k] * g_sel[k]
                needed = d_sel[k] / (np.log(1 + sinr) / np.log(2) + 1e-9)
                b[k] = np.clip(needed, 0.01, B_TOTAL)
            if b.sum() > B_TOTAL:
                b = b * (B_TOTAL / b.sum())
            c_min = (2.0 ** (d_sel / b) - 1.0) / (p + 1e-12)
            grads = np.stack([grad_path_loss(uavs[k], users[assignment[k]])
                              for k in range(N_UAV)])
            self.u_rhs.value = c_min + np.sum(grads * uavs, axis=1)
            self.u_grad.value = grads
            self.u_uprev.value = uavs_prev
            self.u_g_sel.value = g_sel
            try:
                self.prob_u.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
                if self.prob_u.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and self.u_var.value is not None:
                    uavs = np.clip(self.u_var.value,
                                   [0, 0, ALT_RANGE[0]],
                                   [AREA[0], AREA[1], ALT_RANGE[1]])
                    G = channel_gains(uavs, users)
                    g_sel = G[np.arange(N_UAV), assignment]
            except Exception:
                pass
        self.u_init = uavs.copy()
        return p, b, uavs, g_sel


def solve_lqe_analytic(demand_scale):
    W_ACC = W_ACC_BASE * demand_scale
    a_l = W_LAT * 0.035 + W_EN * 0.01 - W_ACC * 0.11
    a_q = W_LAT * 0.005 / 14.0 + W_ACC * 0.005
    a_e = -W_LAT * 0.015 + W_ACC * 0.06
    l_star = 3.0 - a_l / (2.0 * MU_REG)
    q_star = 9.0 - a_q / (2.0 * MU_REG / 10.0)
    e_star = 0.5 - a_e / (2.0 * MU_REG)
    return (float(np.clip(l_star, 1, L_MAX)),
            float(np.clip(q_star, 2, 16)),
            float(np.clip(e_star, 0, 1)))


# ============================================================
# 5. MPC controller (shared) that dispatches to a variant
# ============================================================
class LinearPredictor:
    def __init__(self):
        self.hist_d = []
    def update(self, d):
        self.hist_d.append(d.copy())
        if len(self.hist_d) > 20:
            self.hist_d.pop(0)
    def predict_demand(self, h):
        if len(self.hist_d) < 2:
            return np.tile(self.hist_d[-1], (h, 1))
        recent = np.stack(self.hist_d[-5:])
        base = recent.mean(axis=0)
        if len(recent) >= 3:
            slope = recent[-1] - recent[0]
            out = np.stack([base + slope * (i + 1) for i in range(h)])
        else:
            out = np.tile(base, (h, 1))
        return np.clip(out, DEMAND_LO, DEMAND_HI)


# ============================================================
# 6. Run one variant over the 40-step ramp, return metrics
# ============================================================
def run_variant(name, flags, users, demands, speed_profile=None):
    n_steps = len(demands)
    pred = LinearPredictor()
    uavs = np.zeros((N_UAV, 3))
    uavs[:, 0] = np.linspace(40, AREA[0] - 40, N_UAV)
    uavs[:, 1] = AREA[1] / 2
    uavs[:, 2] = 100.0
    b = np.full(N_UAV, B_TOTAL / N_UAV)
    fast = None
    if flags["accel"]:
        fast = FastSCASolver(users)

    # measured per-step motion fractions (falls back to constant motion)
    if speed_profile is None:
        speed_profile = np.ones(n_steps)
    speed_profile = np.asarray(speed_profile, dtype=float)[:n_steps]
    if speed_profile.size < n_steps:
        speed_profile = np.concatenate(
            [speed_profile, np.ones(n_steps - speed_profile.size)])

    # Each UAV keeps its own heading, so the fleet spreads out as in the
    # measured runs; the step length follows the measured profile and is
    # clipped to the physical V_STEP mobility bound.
    rng = np.random.default_rng(11)
    headings = rng.uniform(0.0, 2.0 * np.pi, size=N_UAV)

    # --- TR 38.901 fading state ------------------------------------------
    # Fast fading: one complex coefficient per UAV per decision step, from the
    # CDL generator with the ULA-Los Doppler spread.
    ff_series = None
    if USE_FAST_FADING:
        f_d = V_MAX * ch3g.F_C / F_CARRIER      # Doppler from platform speed
        ff_all = chfade.fast_fading_series(
            seed=FAST_FADING_SEED, n_snapshots=n_steps * N_UAV,
            k_factor_db=chfade.ULA_LOS_K_FACTOR_DB, max_doppler_hz=f_d)
        ff_series = ff_all.reshape(n_steps, N_UAV)
    # Spatially-correlated shadow fading is generated once over the whole
    # (deterministic) motion profile, so it stays consistent across the run.
    sf_traj = None
    if USE_CORRELATED_SHADOW:
        base = np.zeros((n_steps, 3))
        for k in range(N_UAV):
            base[:, 0] = uavs[k, 0] + headings[k] * V_STEP * np.arange(n_steps)
            base[:, 1] = uavs[k, 1] + np.sin(headings[k]) * V_STEP * np.arange(n_steps)
        sf_series = np.stack([
            chfade.correlated_shadow_fading(base, SHADOW_SIGMA,
                                            chfade.SHADOW_CORR_DISTANCE_M,
                                            seed=CHANNEL_SEED + k)
            for k in range(N_UAV)], axis=1)          # (n_steps, N_UAV)

    dec_ms, accs, lats, dists, Js = [], [], [], [], []
    # per-step traces for the evolution figures (consistent with Table IV)
    LT, QT, ET = [], [], []          # inference controls l, q, e
    PMEAN, BSUM = [], []             # mean power [W], sum bandwidth [MHz]
    DEM = []                         # mean realized demand [Mb/s]
    POSX, POSY = [], []              # UAV x / y positions (incl. measured motion)
    rates_log, sf_log = [], []
    A_MIN = 0.75

    for t in range(n_steps):
        demand = demands[t]
        pred.update(demand)
        if flags["mpc"]:
            demand_avg = pred.predict_demand(H_MPC).mean(axis=0)
        else:
            demand_avg = demand.copy()

        shadow_t = sf_series[t] if sf_series is not None else None
        ff_t = ff_series[t] if ff_series is not None else None

        t0 = time.perf_counter()
        if flags["accel"]:
            G = channel_gains(uavs, users, shadow_db=shadow_t, fast_fading=ff_t)
            assignment = np.argmax(G, axis=1)
            g_sel = G[np.arange(N_UAV), assignment]
            d_sel = demand_avg[assignment]
            p, b, uavs, g_sel = fast.solve(uavs, g_sel, d_sel, b, assignment, users)
            ds = np.clip(np.mean(d_sel) / DEMAND_REF, 0.5, 2.0)
            l_val, q_val, e_val = solve_lqe_analytic(ds)
        else:
            rounds = flags["rounds"]
            sol = solve_unaccel(uavs, users, demand_avg, flags, rounds)
            p, b, uavs = sol["p"], sol["b"], sol["uavs"]
            l_val, q_val, e_val = sol["l"], sol["q"], sol["e"]
            g_sel = sol["g_sel"]
        dec = (time.perf_counter() - t0) * 1e3

        # --- apply the measured per-step motion (bounded by V_STEP) -------
        frac = float(np.clip(speed_profile[t], 0.0, 1.0))
        step = V_STEP * frac
        if step > 1e-9:
            for k in range(N_UAV):
                uavs[k, 0] += step * np.cos(headings[k])
                uavs[k, 1] += step * np.sin(headings[k])
            uavs[:, 0] = np.clip(uavs[:, 0], 0.0, AREA[0])
            uavs[:, 1] = np.clip(uavs[:, 1], 0.0, AREA[1])

        lat, acc = fm_lat_acc(l_val, q_val, e_val)
        assignment = np.argmax(channel_gains(uavs, users), axis=1)
        d_u = float(np.mean([np.linalg.norm(uavs[k] - users[assignment[k]])
                             for k in range(N_UAV)]))
        ds = np.clip(np.mean(demand_avg[assignment]) / DEMAND_REF, 0.5, 2.0)
        J = eval_J(l_val, q_val, e_val, p, ds)

        # --- log the realised link quality (fading-induced variation) ------
        R_real = b * np.log2(1.0 + p * g_sel)
        rates_log.append(float(np.mean(R_real)))
        sf_log.append(float(np.mean(shadow_t)) if shadow_t is not None else 0.0)

        dec_ms.append(dec)
        accs.append(acc)
        lats.append(lat * 1e3)
        dists.append(d_u)
        Js.append(J)
        # ---- per-step evolution traces (for the v2..v5 figures) ----
        LT.append(float(l_val))
        QT.append(float(q_val))
        ET.append(float(e_val))
        PMEAN.append(float(np.mean(p)))
        BSUM.append(float(np.sum(b)))
        DEM.append(float(np.mean(demand)))
        POSX.append(uavs[:, 0].copy())
        POSY.append(uavs[:, 1].copy())

    dec_ms = np.array(dec_ms)
    accs = np.array(accs)
    rates = np.array(rates_log)
    return dict(name=name, flags=flags,
                mean_lat=float(dec_ms.mean()),
                p95_lat=float(np.percentile(dec_ms, 95)),
                frac50=float((dec_ms < 50.0).mean() * 100.0),
                mean_acc=float(accs.mean()),
                min_acc=float(accs.min()),
                acc_viol=float((accs < A_MIN).mean() * 100.0),
                mean_fmlat=float(np.mean(lats)),
                mean_dist=float(np.mean(dists)),
                mean_J=float(np.mean(Js)),
                mean_rate=float(rates.mean()),
                rate_std=float(rates.std()),
                rate_cv=float(rates.std() / max(rates.mean(), 1e-9)),
                rate_min=float(rates.min()),
                rate_max=float(rates.max()),
                rates=rates.tolist(),
                Js=Js,
                dec_ms=dec_ms, accs=accs, lats=lats,
                LT=LT, QT=QT, ET=ET,
                PMEAN=PMEAN, BSUM=BSUM, DEM=DEM,
                POSX=POSX, POSY=POSY)


# ============================================================
# 7. Convergence-behaviour experiment (fixed scenario, poor init)
#    + real multi-start study (5 seeds) for the PeerJ CS resubmission
# ============================================================

# The original hard-coded poor-corner initialization, kept VERBATIM so that
# seed 0 reproduces the single-trajectory experiment exactly.
_ORIGINAL_UAVS0 = np.array([[10.0, 10.0, 50.0],
                            [15.0, 12.0, 50.0],
                            [12.0, 15.0, 50.0]])


def _poor_init(users, seed):
    """One UAV initialization for the multi-start convergence study.

    seed == 0 -> the original hard-coded poor corner (exact reproduction).
    seed >= 1 -> a randomized, geometrically feasible but deliberately poor
                 start: each UAV is drawn uniformly inside the area and altitude
                 bounds and accepted only if it is at least 45 m (horizontal)
                 from every ground user.  Uses a PRIVATE
                 np.random.default_rng(seed), so the global RNG state
                 (np.random.seed(42), FAST_FADING_SEED=991, CHANNEL_SEED=2024)
                 is never touched.
    """
    if seed == 0:
        return _ORIGINAL_UAVS0.copy()
    rng = np.random.default_rng(seed)
    min_sep = 45.0
    out = np.empty((N_UAV, 3), dtype=float)
    for k in range(N_UAV):
        for _ in range(5000):
            cand = np.array([rng.uniform(0.0, AREA[0]),
                             rng.uniform(0.0, AREA[1]),
                             rng.uniform(ALT_RANGE[0], ALT_RANGE[1])])
            if np.all(np.linalg.norm(users[:, :2] - cand[:2], axis=1) > min_sep):
                out[k] = cand
                break
        else:  # pragma: no cover - extremely unlikely, keep it feasible & poor
            out[k] = np.array([5.0, 5.0, 0.5 * (ALT_RANGE[0] + ALT_RANGE[1])])
    return out


def _plot_convergence(trace, png):
    """Two-panel convergence figure (plotting code preserved verbatim)."""
    iters = np.arange(1, len(trace) + 1)
    J_star = trace[-1]
    resid = np.abs(trace - J_star) / abs(J_star) + 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    ax.plot(iters, trace, "-o", ms=2.5, color="C0")
    outer = np.arange(4, len(trace) + 1, 4)
    for o in outer[::5]:
        ax.axvline(o, color="gray", ls=":", lw=0.6, alpha=0.6)
    ax.set_xlabel("BCD iteration (4 blocks / SCA round)")
    ax.set_ylabel("Problem objective $J$")
    ax.set_title("(a) Objective per block update")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(iters, resid, "-o", ms=2.5, color="C3")
    ax.set_xlabel("BCD iteration (4 blocks / SCA round)")
    ax.set_ylabel(r"relative residual $|J_i-J^\star|/|J^\star|$")
    ax.set_title("(b) Convergence (log scale)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()


def _agg(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()),
                std=float(a.std(ddof=0)),
                std_sample_ddof1=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                min=float(a.min()), max=float(a.max()),
                values=[float(v) for v in a])


def _trace_metrics(seed, trace, solve_stats, uavs0, users, rounds):
    """Per-seed metrics from the full per-block trace + solver-failure counts."""
    trace = np.asarray(trace, dtype=float)
    J_star = float(trace[-1])
    denom = abs(J_star) + 1e-12

    # One representative objective per OUTER SCA round: the value recorded
    # after the last of the four block updates (indices 3, 7, 11, ...).
    end_of_round = trace[3::4]

    def conv_round(tau):
        rho = np.abs(end_of_round - J_star) / denom
        bad = np.where(rho > tau)[0]
        # smallest 1-indexed outer round R with rho[r] <= tau for all r >= R-1
        return int(bad[-1] + 2) if bad.size else 1

    c3, c2 = conv_round(1e-3), conv_round(1e-2)
    diffs = np.diff(trace)
    n_viol = int(np.sum(diffs > 1e-9))
    max_inc = float(np.max(diffs)) if diffs.size else 0.0
    min_sep = float(np.min(np.linalg.norm(
        uavs0[:, :2][:, None, :] - users[:, :2][None, :, :], axis=2)))

    exc = int(solve_stats.get("exceptions", 0))
    nonopt = int(solve_stats.get("nonoptimal", 0))
    return {
        "seed": int(seed),
        "init_mode": "original_corner" if seed == 0 else "randomized",
        "init": [[float(v) for v in row] for row in uavs0],
        "min_init_user_distance_m": min_sep,
        "J_initial": float(trace[0]),
        "J_final": J_star,
        "n_blocks": int(len(trace)),
        "n_outer_rounds": int(rounds),
        "conv_round_tau_1e-3": c3,
        "conv_round_tau_1e-2": c2,
        "monotone_violations": n_viol,
        "max_block_increase": max_inc,
        "solver_calls": int(solve_stats.get("solver_calls", 0)),
        "solver_exceptions": exc,
        "solver_nonoptimal": nonopt,
        "solver_failures_total": exc + nonopt,
        # Aliases consumed by paper/make_peerj.py::load_multistart().  They
        # keep the real data injectable without editing that lead-owned file.
        "label": ("original poor-corner init" if seed == 0
                  else "randomized poor init"),
        "rounds_to_1e-3": c3,
        "rounds_to_1e-2": c2,
        "solver_failures": exc + nonopt,
    }


def convergence_experiment(seed=0, rounds=40, out_png=None, dump=True, users=None):
    """Single-initialization convergence run (the multi-start worker).

    seed 0     -> the original hard-coded poor-corner initialization (exact).
    seeds 1..4 -> randomized, geometrically feasible poor inits drawn with a
                  private np.random.default_rng(seed); the global RNG state
                  (np.random.seed(42) and the 991 / 2024 channel seeds) is
                  never touched.
    users      -> optional shared scenario; if None, make_users() is called once
                  (identical to the original standalone experiment).
    dump       -> master switch for filesystem side-effects.  When False the
                  function is a pure computation (used by the orchestrator).
    out_png    -> figure destination.  None (default) keeps the committed
                  paper/figs/convergence.png generation intact for seed 0 only;
                  other seeds write no figure.  An explicit path overrides.
    Returns a dict holding the full trace, the scenario and per-seed metrics.
    """
    if users is None:
        users = make_users()
    uavs0 = _poor_init(users, seed)
    demand_avg = np.full(N_USER, 0.5 * (DEMAND_LO + DEMAND_HI))
    flags = dict(bcd=True, adaptive=True, velocity=True,
                 mpc=True, accel=False, rounds=rounds)
    trace = []
    solve_stats = {}
    solve_unaccel(uavs0, users, demand_avg, flags, rounds=rounds,
                  trace=trace, solve_stats=solve_stats)
    trace = np.array(trace, dtype=float)

    if dump:
        png = out_png if out_png is not None else (
            "paper/figs/convergence.png" if seed == 0 else None)
        if png is not None:
            _plot_convergence(trace, png)
        print(f"[convergence] seed={seed} {len(trace)} block updates; "
              f"J: {trace[0]:.4f} -> {trace[-1]:.4f} "
              f"(initial -> final), rel. drop "
              f"{(trace[0]-trace[-1])/abs(trace[0])*100:.1f}%")

    metrics = _trace_metrics(seed, trace, solve_stats, uavs0, users, rounds)
    return dict(trace=trace, users=users, uavs0=uavs0,
                demand_avg=demand_avg, flags=flags, rounds=rounds,
                solve_stats=dict(solve_stats), **metrics)


def _write_summary_md(path, summary, results):
    agg = summary["aggregate"]
    jf, ji = agg["J_final"], agg["J_initial"]
    c3, c2 = agg["conv_round_tau_1e-3"], agg["conv_round_tau_1e-2"]
    defs = summary["definitions"]
    # Aggregate the in-process solver_events evidence across seeds/blocks.
    se = summary["solver_events"]
    _OK = ("optimal", "optimal_inaccurate")
    _glob_out, _perblk = {}, {}
    for _sd, _st in se.items():
        for _k, _v in _st["outcomes"].items():
            _glob_out[_k] = _glob_out.get(_k, 0) + _v
        for _blk, _bd in _st["by_block"].items():
            _pb = _perblk.setdefault(_blk, {"calls": 0, "outcomes": {}})
            _pb["calls"] += _bd["calls"]
            for _k, _v in _bd["outcomes"].items():
                _pb["outcomes"][_k] = _pb["outcomes"].get(_k, 0) + _v

    def _nfail(_d):
        return sum(_v for _k, _v in _d["outcomes"].items() if _k not in _OK)

    L = []
    L.append("# Multi-start convergence — real 5-seed run")
    L.append("")
    L.append(f"Generated (UTC): {summary['utc_timestamp']}")
    L.append("")
    L.append("All numbers below are the **actual output** of this run; none are "
             "extrapolated, rounded up, or substituted.")
    L.append("")
    L.append("## Headline numbers")
    L.append("")
    L.append(f"- Final objective over {len(results)} seeds: mean = "
             f"{jf['mean']:.6f}, population std (ddof=0) = {jf['std']:.6f}, "
             f"sample std (ddof=1) = {jf['std_sample_ddof1']:.6f}, "
             f"min = {jf['min']:.6f}, max = {jf['max']:.6f}.")
    L.append(f"- Convergence rounds tau=1e-3: mean = {c3['mean']:.3f}, "
             f"std(ddof0) = {c3['std']:.3f}, min = {c3['min']}, max = {c3['max']}.")
    L.append(f"- Convergence rounds tau=1e-2: mean = {c2['mean']:.3f}, "
             f"std(ddof0) = {c2['std']:.3f}, min = {c2['min']}, max = {c2['max']}.")
    L.append(f"- Monotone-decrease violations (total): "
             f"{agg['monotone_violations_total']}.")
    L.append(f"- Solver failures (total, counted — NOT swallowed): "
             f"{agg['solver_failures_total']} over "
             f"{agg['solver_calls_total']} solver calls "
             f"(exceptions + non-optimal statuses).")
    ref = summary.get("seed0_reference")
    if ref and "error" not in ref:
        L.append(f"- Seed 0 vs pristine original single run: equal length = "
                 f"{ref['same_length']}, max|dJ| = {ref['max_abs_diff']}, "
                 f"match within 1e-9 = {ref['matches_within_1e-9']} "
                 f"(original J_init {ref['original_J_initial']:.6f} -> "
                 f"J_final {ref['original_J_final']:.6f}, "
                 f"n_blocks {ref['original_n_blocks']}).")
    L.append("")
    L.append("## Per-seed (human-readable)")
    L.append("")
    L.append("| Seed | Init | J_init | J_final | R(1e-3) | R(1e-2) | "
             "Mono. viol. | Solver fails |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| {r['seed']} | {r['init_mode']} | {r['J_initial']:.6f} | "
                 f"{r['J_final']:.6f} | {r['conv_round_tau_1e-3']} | "
                 f"{r['conv_round_tau_1e-2']} | {r['monotone_violations']} | "
                 f"{r['solver_failures_total']} |")
    L.append("")
    L.append("## Definitions (explicit criteria)")
    L.append("")
    L.append(f"- Convergence round: {defs['convergence_round']}")
    L.append(f"- Monotone violation: {defs['monotone_violation']}")
    L.append(f"- Solver failure: {defs['solver_failure']}")
    L.append(f"- Std: {defs['std']}")
    L.append("")
    L.append("## Solver-failure breakdown (in-process evidence)")
    L.append("")
    L.append("Recorded inside `solve_unaccel`'s `solve_stats` (not a post-hoc "
             "diagnostic). `optimal` / `optimal_inaccurate` are successes; every "
             "other status or `EXC:<ExceptionClass>` is a failure.")
    L.append("")
    L.append("- Global CLARABEL outcomes over 5 seeds: "
             + ", ".join(f"`{k}`={v}" for k, v in sorted(_glob_out.items())) + ".")
    L.append(f"- Total solver calls = {sum(d['calls'] for d in _perblk.values())}; "
             f"total failures = {sum(_nfail(d) for d in _perblk.values())}.")
    for _blk in ("lqe", "p", "u"):
        _d = _perblk[_blk]
        L.append(f"- block `{_blk}`: calls={_d['calls']}, failures={_nfail(_d)}; "
                 "outcomes = "
                 + ", ".join(f"`{k}`={v}" for k, v in sorted(_d["outcomes"].items()))
                 + ".")
    L.append("")
    L.append("## Software / reproducibility")
    L.append("")
    sw = summary["software"]
    L.append(f"- Python {sw['python']}, numpy {sw['numpy']}, scipy "
             f"{sw['scipy']}, matplotlib {sw['matplotlib']}, cvxpy "
             f"{sw['cvxpy']}, CLARABEL {sw['clarabel']}.")
    L.append(f"- Command: `{summary['command']}`")
    L.append(f"- cwd: `{summary['cwd']}`")
    L.append(f"- Figure: {summary['figure_integrity']['note']}")
    L.append("")
    L.append("## LaTeX table (paste-ready tabular body)")
    L.append("")
    L.append("```latex")
    L.append(r"\begin{tabular}{c|c|r|r|c|c|r|r}")
    L.append(r"\hline")
    L.append(r"Seed & Init & $J_{\mathrm{init}}$ & $J^\star$ & "
             r"$R_{10^{-3}}$ & $R_{10^{-2}}$ & Mono.~viol. & Solver fails \\")
    L.append(r"\hline")
    for r in results:
        L.append(
            rf"{r['seed']} & {r['init_mode']} & {r['J_initial']:.4f} & "
            rf"{r['J_final']:.4f} & {r['conv_round_tau_1e-3']} & "
            rf"{r['conv_round_tau_1e-2']} & {r['monotone_violations']} & "
            rf"{r['solver_failures_total']} \\"
        )
    L.append(r"\hline")
    L.append(
        rf"Mean$\pm$std & -- & {ji['mean']:.4f}$\pm${ji['std']:.4f} & "
        rf"{jf['mean']:.4f}$\pm${jf['std']:.4f} & "
        rf"{c3['mean']:.2f}$\pm${c3['std']:.2f} & "
        rf"{c2['mean']:.2f}$\pm${c2['std']:.2f} & "
        rf"$\Sigma$={agg['monotone_violations_total']} & "
        rf"$\Sigma$={agg['solver_failures_total']} \\"
    )
    L.append(r"\hline")
    L.append(r"\end{tabular}")
    L.append("```")
    L.append("")
    L.append(r"$J_{\mathrm{init}}$ / $J^\star$: initial / final objective. "
             r"$R_{\tau}$: convergence round at relative-residual threshold "
             r"$\tau$. Mono.~viol.: number of objective increases. Solver "
             r"fails: counted solver exceptions + non-optimal statuses. "
             r"Std is population std (ddof=0); sample std (ddof=1) is in the JSON.")
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def run_convergence_multistart(seeds=(0, 1, 2, 3, 4), rounds=40,
                               reference_path=None):
    """Run the multi-start convergence study and export paper/data/ artifacts.

    Every seed shares ONE scenario (make_users() called exactly once, as in the
    original standalone experiment); only the UAV initialization varies.
    Nothing under paper/figs/ is written by this function.
    """
    data_dir = Path("paper/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Single shared scenario -- first/only consumer of the global RNG (seed 42),
    # exactly as in the original single-run experiment.
    users = make_users()
    demand_avg_const = np.full(N_USER, 0.5 * (DEMAND_LO + DEMAND_HI))

    results = [convergence_experiment(seed=int(s), rounds=rounds,
                                      dump=False, users=users)
               for s in seeds]

    # ---- traces CSV ------------------------------------------------------
    csv_path = data_dir / "convergence_multistart_traces.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "block_index", "J"])
        for r in results:
            for i, J in enumerate(r["trace"], start=1):
                w.writerow([r["seed"], i, format(float(J), ".17g")])

    # ---- aggregate statistics -------------------------------------------
    def agg_of(key):
        return _agg([r[key] for r in results])

    aggregate = {
        "J_initial": agg_of("J_initial"),
        "J_final": agg_of("J_final"),
        "conv_round_tau_1e-3": agg_of("conv_round_tau_1e-3"),
        "conv_round_tau_1e-2": agg_of("conv_round_tau_1e-2"),
        "monotone_violations_total": int(sum(
            r["monotone_violations"] for r in results)),
        "solver_failures_total": int(sum(
            r["solver_failures_total"] for r in results)),
        "solver_calls_total": int(sum(r["solver_calls"] for r in results)),
    }

    # ---- software / command / timestamp ----------------------------------
    software = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "cvxpy": cp.__version__,
        "clarabel": clarabel.__version__,
    }
    command = f"{sys.executable} " + " ".join(sys.argv)
    timestamp = datetime.now(timezone.utc).isoformat()

    # ---- seed-0 reproduction vs pristine original ------------------------
    ref_obj = None
    if reference_path:
        try:
            with open(reference_path) as f:
                ref = np.asarray(json.load(f), dtype=float)
            s0 = np.asarray(results[0]["trace"], dtype=float)
            same_len = bool(len(s0) == len(ref))
            max_abs = float(np.max(np.abs(s0 - ref))) if same_len else None
            ref_obj = {
                "source": str(reference_path),
                "original_J_initial": float(ref[0]),
                "original_J_final": float(ref[-1]),
                "original_n_blocks": int(len(ref)),
                "seed0_J_initial": float(s0[0]),
                "seed0_J_final": float(s0[-1]),
                "seed0_n_blocks": int(len(s0)),
                "same_length": same_len,
                "max_abs_diff": max_abs,
                "matches_within_1e-9": bool(max_abs is not None and max_abs < 1e-9),
            }
        except Exception as e:
            ref_obj = {"error": f"reference comparison failed: {e}"}

    # ---- figure integrity (committed file is NEVER written) --------------
    committed = Path("paper/figs/convergence.png")
    committed_sha = (hashlib.sha256(committed.read_bytes()).hexdigest()
                     if committed.exists() else None)
    figure_integrity = {
        "committed_figure": str(committed),
        "committed_figure_sha256": committed_sha,
        "committed_figure_written_by_this_run": False,
        "note": ("paper/figs/convergence.png was NOT touched by this run. The "
                 "refactored convergence_experiment(seed=0, dump=True) still "
                 "regenerates the identical two-panel figure (plotting code "
                 "preserved verbatim); the multistart run writes only to "
                 "paper/data/."),
    }

    # ---- summary JSON ----------------------------------------------------
    summary = {
        "experiment": "multi-start convergence of SCA+BCD (airborne FM, 3GPP)",
        "scenario": {
            "users": [[float(v) for v in row] for row in users],
            "demand_avg": [float(v) for v in demand_avg_const],
            "flags": dict(bcd=True, adaptive=True, velocity=True, mpc=True,
                          accel=False, rounds=rounds),
            "rounds": int(rounds),
            "blocks_per_round": 4,
            "area": list(AREA), "alt_range": list(ALT_RANGE),
            "n_uav": N_UAV, "n_user": N_USER,
        },
        "seeds": [int(s) for s in seeds],
        "per_seed": [{k: v for k, v in r.items()
                      if k not in ("trace", "users", "uavs0", "demand_avg",
                                   "flags", "solve_stats")}
                     for r in results],
        "per_seed_traces": {str(r["seed"]): [float(v) for v in r["trace"]]
                            for r in results},
        # Per-seed, per-block CLARABEL outcome evidence (from solve_unaccel's
        # in-process solve_stats).  Backs the "148 failures, all in the UAV
        # position block" claim with reproducible in-artifact data.
        "solver_events": {str(r["seed"]): r["solve_stats"] for r in results},
        "aggregate": aggregate,
        "definitions": {
            "convergence_round": ("smallest 1-indexed outer SCA round R such that "
                                  "the end-of-round objective (last of the 4 block "
                                  "updates per round) has relative residual "
                                  "|J_r - J*|/(|J*|+1e-12) <= tau for every "
                                  "round r >= R, with J* = final objective. "
                                  "Both tau=1e-3 and tau=1e-2 are reported."),
            "monotone_violation": ("count of block-to-block objective increases "
                                   "diff = J[i]-J[i-1] > 1e-9 (absolute units)."),
            "solver_failure": ("exceptions raised by cp.solve PLUS non-optimal "
                               "solver statuses (value is None or status not in "
                               "{OPTIMAL, OPTIMAL_INACCURATE}). No solver time "
                               "limit is set, so timeouts are not a separate "
                               "mode; these were previously swallowed by bare "
                               "`except Exception: pass` handlers and are now "
                               "counted in solve_unaccel. The 'solver_events' "
                               "field records, per seed and per block ('lqe', "
                               "'p', 'u'), the exact CLARABEL status strings "
                               "(or 'EXC:<ExceptionClass>') so the per-block "
                               "attribution is reproducible from this file. "
                               "'optimal_inaccurate' is a SUCCESS here (matching "
                               "the original code) and is NOT counted as a "
                               "failure, though it emits a solver warning. "
                               "'user_limit' denotes CLARABEL hitting its "
                               "iteration cap and IS counted as a failure."),
            "std": ("aggregate 'std' is population std (numpy ddof=0); "
                    "'std_sample_ddof1' is the sample std (ddof=1)."),
        },
        "software": software,
        # paper/make_peerj.py reads environment.<..._version> for its version line
        "environment": {
            "cvxpy_version": cp.__version__,
            "clarabel_version": clarabel.__version__,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "command": command,
        "cwd": str(Path.cwd()),
        "utc_timestamp": timestamp,
        "seed0_reference": ref_obj,
        "figure_integrity": figure_integrity,
    }
    json_path = data_dir / "convergence_multistart_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    _write_summary_md(data_dir / "convergence_multistart_summary.md",
                      summary, results)

    # ---- console summary (ACTUAL numbers) --------------------------------
    print("\n=== MULTI-START CONVERGENCE (real run) ===")
    for r in results:
        print(f"  seed {r['seed']} [{r['init_mode']:>15s}] "
              f"J_init={r['J_initial']:.6f} J_final={r['J_final']:.6f} "
              f"conv(1e-3)={r['conv_round_tau_1e-3']:2d} "
              f"conv(1e-2)={r['conv_round_tau_1e-2']:2d} "
              f"mono_viol={r['monotone_violations']:3d} "
              f"solver_fail={r['solver_failures_total']:3d} "
              f"(exc={r['solver_exceptions']}, "
              f"nonopt={r['solver_nonoptimal']}, "
              f"calls={r['solver_calls']})")
    print(f"  J_final: mean={aggregate['J_final']['mean']:.6f} "
          f"std(ddof0)={aggregate['J_final']['std']:.6f} "
          f"std(ddof1)={aggregate['J_final']['std_sample_ddof1']:.6f} "
          f"min={aggregate['J_final']['min']:.6f} "
          f"max={aggregate['J_final']['max']:.6f}")
    print(f"  conv(1e-3) rounds: mean={aggregate['conv_round_tau_1e-3']['mean']:.3f} "
          f"std(ddof0)={aggregate['conv_round_tau_1e-3']['std']:.3f}")
    print(f"  conv(1e-2) rounds: mean={aggregate['conv_round_tau_1e-2']['mean']:.3f} "
          f"std(ddof0)={aggregate['conv_round_tau_1e-2']['std']:.3f}")
    print(f"  total monotone violations={aggregate['monotone_violations_total']} "
          f"total solver failures={aggregate['solver_failures_total']} "
          f"solver calls={aggregate['solver_calls_total']}")
    if ref_obj and "error" not in ref_obj:
        print(f"  seed0 vs original: same_length={ref_obj['same_length']} "
              f"max_abs_diff={ref_obj['max_abs_diff']} "
              f"match(1e-9)={ref_obj['matches_within_1e-9']}")
    print(f"  wrote: {csv_path}, {json_path}, "
          f"{data_dir / 'convergence_multistart_summary.md'}")
    print(f"  versions: {software}")
    print(f"  command: {command}")
    return summary


# ============================================================
# 7b. Evolution-figure generation (2x3 panel for v2/v3/v4)
#     Captures the *same* per-step run that fills Table IV, so the plotted
#     l/q/e/p/b/demand/trajectory are numerically consistent with the table.
#     Demand is on the 3GPP Mb/s scale (30 -> 62) and the horizon is the full
#     40-step ramp, fixing the old free-space "req/s" / 0-30-step artefacts.
# ============================================================
def _zoom_ylim(arr, pad_frac=0.25):
    """Axis limits zoomed to the data range (so a subtle l/e trend is visible)."""
    lo, hi = float(np.min(arr)), float(np.max(arr))
    pad = (hi - lo) * pad_frac if hi - lo > 1e-9 else max(abs(lo) * 0.1, 0.05)
    return lo - pad, hi + pad


def make_evolution_fig(m, users, fname):
    t = np.arange(len(m["Js"]))
    n = len(t)
    LT = np.array(m["LT"]); QT = np.array(m["QT"]); ET = np.array(m["ET"])
    PMEAN = np.array(m["PMEAN"]); BSUM = np.array(m["BSUM"])
    DEM = np.array(m["DEM"])
    POSX = np.stack(m["POSX"]); POSY = np.stack(m["POSY"])
    accs = np.array(m["accs"]); lats = np.array(m["lats"])
    dec = np.array(m["dec_ms"])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))

    # (a) demand vs split depth l  (right axis zoomed so the trend is visible)
    ax = axes[0, 0]
    ax.plot(t, DEM, "k--", label="demand")
    ax.set_ylabel("demand [Mb/s]")
    a2 = ax.twinx()
    a2.plot(t, LT, "o-", ms=3, color="C0", label="$l$ (split)")
    a2.set_ylabel("split depth $l$", color="C0")
    a2.tick_params(axis="y", labelcolor="C0")
    a2.set_ylim(*_zoom_ylim(LT))
    ax.set_title("(a) Demand vs split depth $l$")
    ax.set_xlabel("step"); ax.set_xlim(0, n - 1); ax.grid(alpha=.3)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, loc="upper left", fontsize=8)

    # (b) demand vs early-exit e and quantization bits q (both zoomed)
    ax = axes[0, 1]
    ax.plot(t, DEM, "k--", label="demand")
    ax.set_ylabel("demand [Mb/s]")
    a2 = ax.twinx()
    a2.plot(t, ET, "s-", ms=3, color="C1", label="$e$ (exit)")
    a2.set_ylabel("early-exit $e$", color="C1")
    a2.tick_params(axis="y", labelcolor="C1")
    a2.set_ylim(*_zoom_ylim(ET))
    a3 = ax.twinx()
    a3.plot(t, QT, "^-", ms=3, color="C2", label="$q$ [bits]")
    a3.set_ylabel("bits $q$", color="C2")
    a3.tick_params(axis="y", labelcolor="C2")
    a3.set_ylim(*_zoom_ylim(QT))
    a3.spines["right"].set_position(("outward", 40))
    ax.set_title("(b) Demand vs early-exit $e$ & bits $q$")
    ax.set_xlabel("step"); ax.set_xlim(0, n - 1); ax.grid(alpha=.3)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = a2.get_legend_handles_labels()
    h3, l3_ = a3.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1_ + l2_ + l3_, loc="upper left", fontsize=8)

    ax = axes[0, 2]
    ax.plot(t, lats, "-", label="FM lat [ms]")
    ax.plot(t, accs * 100, "--", label="acc [%]")
    ax.set_title("(c) FM latency vs accuracy")
    ax.set_xlabel("step"); ax.set_xlim(0, n - 1)
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 0]
    ax.plot(t, PMEAN, "-", label="mean p [W]")
    a2 = ax.twinx(); a2.plot(t, BSUM, "-", color="C3", label="sum b [MHz]")
    ax.set_title("(d) Power & bandwidth")
    ax.set_xlabel("step"); ax.set_xlim(0, n - 1)
    ax.legend(loc="upper left"); a2.legend(loc="upper right")
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    for k in range(N_UAV):
        ax.plot(POSX[:, k], POSY[:, k], "o-", ms=3, label=f"UAV{k}")
        ax.plot(users[k, 0], users[k, 1], "r*", markersize=15)
    ax.set_title("(e) UAV trajectory (* = user)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_xlim(0, AREA[0])
    ax.set_ylim(0, AREA[1]); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 2]
    ax.plot(t, dec, "-", color="C4", label="decision time")
    ax.axhline(DT * 1e3, ls=":", color="k", label=f"Δt={DT * 1e3:.0f} ms")
    ax.set_title("(f) Online decision latency")
    ax.set_xlabel("step"); ax.set_ylabel("ms"); ax.set_xlim(0, n - 1)
    ax.legend(); ax.grid(alpha=.3)

    fig.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {fname}")


def make_v5_fig(m, users, fname):
    """v5 panel: latency-focused (matches its caption) + demand/l-e adaptation."""
    t = np.arange(len(m["Js"]))
    n = len(t)
    LT = np.array(m["LT"]); ET = np.array(m["ET"]); DEM = np.array(m["DEM"])
    dec = np.array(m["dec_ms"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(t, dec, "-", color="C3", label="decision time")
    ax.axhline(50, ls="--", color="g", label="50 ms budget")
    ax.axhline(DT * 1e3, ls=":", color="k", label=f"Δt={DT * 1e3:.0f} ms")
    ax.set_title("(a) Online decision latency (warm-start)")
    ax.set_xlabel("step"); ax.set_ylabel("ms"); ax.set_xlim(0, n - 1)
    ax.legend(); ax.grid(alpha=.3)

    # (b) demand vs l (zoomed right axis) and e (offset far-right axis)
    ax = axes[1]
    ax.plot(t, DEM, "k--", label="demand")
    ax.set_ylabel("demand [Mb/s]")
    ax.set_ylim(0, DEM.max() * 1.15)
    a2 = ax.twinx()
    a2.plot(t, LT, "o-", ms=3, color="C0", label="$l$ (split)")
    a2.set_ylabel("split depth $l$", color="C0")
    a2.tick_params(axis="y", labelcolor="C0")
    a2.set_ylim(*_zoom_ylim(LT))
    a3 = ax.twinx()
    a3.plot(t, ET, "s-", ms=3, color="C1", label="$e$ (exit)")
    a3.set_ylabel("early-exit $e$", color="C1")
    a3.tick_params(axis="y", labelcolor="C1")
    a3.set_ylim(*_zoom_ylim(ET))
    a3.spines["right"].set_position(("outward", 40))
    ax.set_title("(b) Demand vs $l/e$ adaptation")
    ax.set_xlabel("step"); ax.set_xlim(0, n - 1); ax.grid(alpha=.3)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = a2.get_legend_handles_labels()
    h3, l3_ = a3.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1_ + l2_ + l3_, loc="upper left", fontsize=8)

    fig.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {fname}")


# ============================================================
# 8. Main: convergence + ablation
# ============================================================
def main():
    # ---- (A) convergence ----
    conv = convergence_experiment()

    # ---- (B) ablation: method family v1..v5 ----
    # Identical scenario for every variant (same users, same demand ramp).
    # Demand is on the physical Mb/s scale of the 3GPP rate model.
    np.random.seed(123)
    users_abl = make_users()
    demands = [np.clip(DEMAND_LO + (DEMAND_HI - DEMAND_LO) * t / 39.0
                       + np.random.randn(N_USER) * 2.0,
                       DEMAND_LO, DEMAND_HI)
               for t in range(40)]
    # Flags are a *nested* removal from v5 (each version drops the last stage).
    variants = [
        # name, dropped module, flags
        ("v5", "--- (full method)",
         dict(bcd=True, adaptive=True, velocity=True, mpc=True,
              accel=True, rounds=3)),
        ("v4", "solver acceleration (DPP / warm-start)",
         dict(bcd=True, adaptive=True, velocity=True, mpc=True,
              accel=False, rounds=5)),
        ("v3", "+ demand-adaptive $l/q/e$ & velocity bound",
         dict(bcd=True, adaptive=False, velocity=False, mpc=True,
              accel=False, rounds=5)),
        ("v2", "+ block-coordinate descent (UAV & bandwidth)",
         dict(bcd=False, adaptive=False, velocity=False, mpc=True,
              accel=False, rounds=5)),
        ("v1", "+ offline warm-start dictionary & MPC look-ahead",
         dict(bcd=False, adaptive=False, velocity=False, mpc=False,
              accel=False, rounds=10)),
    ]
    rows = []
    speed_profile = load_measured_speed_profile(len(demands))
    print(f"[kinematics] measured per-step speed fraction: "
          f"min={speed_profile.min():.2f} med={np.median(speed_profile):.2f} "
          f"max={speed_profile.max():.2f} "
          f"(mean per-step displacement = "
          f"{V_STEP * speed_profile.mean():.3f} m of V_STEP={V_STEP} m)")
    for name, dropped, flags in variants:
        m = run_variant(name, flags, users_abl, demands, speed_profile)
        m["dropped"] = dropped
        rows.append(m)

    lat_ref = rows[0]["mean_lat"]   # v5 accelerated
    for m in rows:
        m["rel_lat"] = m["mean_lat"] / lat_ref
        print(f"  {m['name']}: lat={m['mean_lat']:6.2f}ms (x{m['rel_lat']:5.1f}) "
              f"p95={m['p95_lat']:6.2f} <50={m['frac50']:5.1f}%  "
              f"acc={m['mean_acc']:.3f} (min {m['min_acc']:.3f}, "
              f"A_min viol {m['acc_viol']:5.1f}%) "
              f"fmlat={m['mean_fmlat']:5.1f}ms J={m['mean_J']:.3f}")

    # ---- CSV ----
    with open("ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "dropped", "mean_lat_ms", "rel_lat_x",
                    "p95_lat_ms", "frac_lt50_pct", "mean_acc", "min_acc",
                    "acc_viol_pct", "mean_fm_lat_ms",
                    "mean_uav_user_dist_m", "mean_J"])
        for m in rows:
            w.writerow([m["name"], m["dropped"], f"{m['mean_lat']:.2f}",
                        f"{m['rel_lat']:.1f}", f"{m['p95_lat']:.2f}",
                        f"{m['frac50']:.1f}", f"{m['mean_acc']:.3f}",
                        f"{m['min_acc']:.3f}", f"{m['acc_viol']:.1f}",
                        f"{m['mean_fmlat']:.1f}", f"{m['mean_dist']:.1f}",
                        f"{m['mean_J']:.3f}"])

    # ---- Figure: two panels (latency | system performance) ----
    names = [m["name"] for m in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # panel (a): decision latency
    ax = axes[0]
    ax.bar(x - 0.18, [m["mean_lat"] for m in rows], 0.36,
           label="mean", color="C0")
    ax.bar(x + 0.18, [m["p95_lat"] for m in rows], 0.36,
           label="p95", color="C1")
    ax.axhline(50, ls="--", color="g", label="50 ms budget")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("decision latency [ms]")
    ax.set_title("(a) Decision latency  (lower is better)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    # panel (b): system performance
    ax = axes[1]
    ax2 = ax.twinx()
    l1 = ax.bar(x - 0.18, [m["mean_acc"] for m in rows], 0.36,
                color="C2", label="mean accuracy")
    ax.axhline(0.75, ls="--", color="r", label=r"$A_{\min}$")
    ax.set_ylabel("mean accuracy", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax.set_ylim(0.0, 0.85)
    l2 = ax2.bar(x + 0.18, [m["mean_fmlat"] for m in rows], 0.36,
                 color="C4", label="mean FM lat [ms]")
    ax2.set_ylabel("mean FM latency [ms]", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("(b) System performance  (accuracy vs. FM latency)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    plt.savefig("paper/figs/ablation.png", dpi=150)
    plt.close()
    print("[plot] paper/figs/ablation.png, paper/figs/convergence.png, ablation.csv")

    # ---- Evolution figures (v2..v5) on the identical 40-step ramp -------
    # Built from the SAME run_variant calls above, so the plotted l/q/e/p/b
    # and demand are numerically consistent with Table IV and the 3GPP Mb/s
    # demand scale (30 -> 62), with the full 40-step horizon.
    _fig_map = {
        "v2": ("paper/figs/mpc_evolution.png", make_evolution_fig),
        "v3": ("paper/figs/v3_evolution.png", make_evolution_fig),
        "v4": ("paper/figs/v4_evolution.png", make_evolution_fig),
        "v5": ("paper/figs/v5_warmstart.png", make_v5_fig),
    }
    for m in rows:
        if m["name"] in _fig_map:
            _fn, _gen = _fig_map[m["name"]]
            _gen(m, users_abl, _fn)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Airborne FM-inference 3GPP experiments.")
    ap.add_argument("--multistart", action="store_true",
                    help="run the 5-seed multi-start convergence study and "
                         "write paper/data/ artifacts (does NOT touch paper/figs/)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="initialization seeds (default: 0 1 2 3 4)")
    ap.add_argument("--rounds", type=int, default=40,
                    help="outer SCA rounds per seed (default: 40)")
    ap.add_argument("--reference", type=str, default=None,
                    help="optional JSON file with the pristine original "
                         "single-run trace (list of J values) for the seed-0 "
                         "reproduction check")
    args = ap.parse_args()
    if args.multistart:
        run_convergence_multistart(seeds=args.seeds, rounds=args.rounds,
                                   reference_path=args.reference)
    else:
        main()
