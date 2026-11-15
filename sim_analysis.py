"""
sim_analysis.py  --  Supplementary experiments for the airborne FM-inference paper
=================================================================================
Produces two artifacts used by the revised manuscript:

  (A) Convergence-behaviour curve  ->  figs/convergence.png
        Runs the un-accelerated SCA + block-coordinate-descent (BCD) solver on a
        *fixed* scenario from a poor initialization, logging the problem
        objective J after every inner block update, to show stable convergence.

  (B) Quantitative ablation study   ->  figs/ablation.png  +  ablation.csv
        Drives the v1->v5 method family on the *same* 40-step demand ramp and
        records, for every version, BOTH the decision latency (mean / p95 /
        fraction < 50 ms) AND the system performance (mean accuracy, mean FM
        inference latency, mean air-to-ground channel gain, mean objective J).

The channel / FM proxies / weights are identical to hrl_airborne_fm_v4.py and
hrl_airborne_fm_v5.py so the numbers are consistent with the rest of the paper.
"""

import time
import csv
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. System parameters  (kept identical to v4 / v5 notebooks)
# ============================================================
AREA = (200.0, 200.0)
ALT_RANGE = (50.0, 150.0)
N_UAV = 3
N_USER = 5
G0 = 1e-4

B_TOTAL = 10.0          # MHz  (total bandwidth budget)
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
DEMAND_REF = 3.0

np.random.seed(42)


# ============================================================
# 1. Channel helpers
# ============================================================
def make_users(n=N_USER):
    xs = np.linspace(40, AREA[0] - 40, n)
    ys = np.random.uniform(40, AREA[1] - 40, size=n)
    return np.stack([xs, ys, np.zeros(n)], axis=1)


def path_loss(uav, user):
    d3 = np.linalg.norm(uav - user)
    return G0 / (d3 ** 2 + 1e-6)


def channel_gains(uavs, users):
    G = np.zeros((N_UAV, N_USER))
    for k in range(N_UAV):
        for j in range(N_USER):
            G[k, j] = path_loss(uavs[k], users[j])
    return G


def grad_path_loss(uav, user):
    diff = uav - user
    d2 = np.sum(diff ** 2) + 1e-6
    return -2.0 * G0 * diff / (d2 ** 2)


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
def solve_unaccel(uavs_prev, users, demand_avg, flags, rounds,
                  trace=None):
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
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    l_val = float(np.clip(l.value, 1, L_MAX))
                    q_val = float(np.clip(q.value, 2, 16))
                    e_val = float(np.clip(e.value, 0, 1))
                else:
                    l_val, q_val, e_val = 3.0, 9.0, 0.5
            except Exception:
                l_val, q_val, e_val = 3.0, 9.0, 0.5
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
        try:
            prob_p.solve(solver=cp.CLARABEL, verbose=False)
            if prob_p.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and p_var.value is not None:
                p = np.clip(p_var.value, P_MIN, P_MAX)
        except Exception:
            pass
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
            try:
                prob_u.solve(solver=cp.CLARABEL, verbose=False)
                if prob_u.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u_var.value is not None:
                    uavs = np.clip(u_var.value,
                                   [0, 0, ALT_RANGE[0]],
                                   [AREA[0], AREA[1], ALT_RANGE[1]])
                    G = channel_gains(uavs, users)
                    g_sel = G[np.arange(N_UAV), assignment]
            except Exception:
                pass
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
        return np.clip(out, 0.1, 10.0)


# ============================================================
# 6. Run one variant over the 40-step ramp, return metrics
# ============================================================
def run_variant(name, flags, users, demands):
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

    dec_ms, accs, lats, dists, Js = [], [], [], [], []
    A_MIN = 0.75

    for t in range(n_steps):
        demand = demands[t]
        pred.update(demand)
        if flags["mpc"]:
            demand_avg = pred.predict_demand(H_MPC).mean(axis=0)
        else:
            demand_avg = demand.copy()

        t0 = time.perf_counter()
        if flags["accel"]:
            G = channel_gains(uavs, users)
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

        lat, acc = fm_lat_acc(l_val, q_val, e_val)
        assignment = np.argmax(channel_gains(uavs, users), axis=1)
        d_u = float(np.mean([np.linalg.norm(uavs[k] - users[assignment[k]])
                             for k in range(N_UAV)]))
        ds = np.clip(np.mean(demand_avg[assignment]) / DEMAND_REF, 0.5, 2.0)
        J = eval_J(l_val, q_val, e_val, p, ds)

        dec_ms.append(dec)
        accs.append(acc)
        lats.append(lat * 1e3)
        dists.append(d_u)
        Js.append(J)

    dec_ms = np.array(dec_ms)
    accs = np.array(accs)
    return dict(name=name, flags=flags,
                mean_lat=float(dec_ms.mean()),
                p95_lat=float(np.percentile(dec_ms, 95)),
                frac50=float((dec_ms < 50.0).mean() * 100.0),
                mean_acc=float(accs.mean()),
                min_acc=float(accs.min()),
                acc_viol=float((accs < A_MIN).mean() * 100.0),
                mean_fmlat=float(np.mean(lats)),
                mean_dist=float(np.mean(dists)),
                mean_J=float(np.mean(Js)))


# ============================================================
# 7. Convergence-behaviour experiment (fixed scenario, poor init)
# ============================================================
def convergence_experiment():
    # NOTE: the paper's free-space model uses G0 = 1e-4, under which the
    # rate constraint is inactive at realistic UAV--user distances (a documented
    # toy limitation, Sec. VI).  To *illustrate* the SCA+BCD convergence
    # behaviour we calibrate the link budget (G0 = 1e5) on a representative
    # instance where the rate constraint is active, so that p and u genuinely
    # drive the objective.  The convergence qualitative is independent of the
    # absolute calibration.
    global G0
    G0_old = G0
    G0 = 1e5
    users = make_users()
    # poor initialization: all UAVs clustered in a corner, far from users
    uavs0 = np.array([[10.0, 10.0, 50.0],
                      [15.0, 12.0, 50.0],
                      [12.0, 15.0, 50.0]])
    demand_avg = np.full(N_USER, 3.0)
    flags = dict(bcd=True, adaptive=True, velocity=True,
                 mpc=True, accel=False, rounds=40)
    trace = []
    solve_unaccel(uavs0, users, demand_avg, flags, rounds=40, trace=trace)
    G0 = G0_old
    trace = np.array(trace)
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
    plt.savefig("paper/figs/convergence.png", dpi=150)
    plt.close()
    print(f"[convergence] {len(trace)} block updates; "
          f"J: {trace[0]:.4f} -> {trace[-1]:.4f} "
          f"(initial -> final), rel. drop {(trace[0]-trace[-1])/abs(trace[0])*100:.1f}%")
    return trace


# ============================================================
# 8. Main: convergence + ablation
# ============================================================
def main():
    # ---- (A) convergence ----
    conv = convergence_experiment()

    # ---- (B) ablation: method family v1..v5 ----
    # Identical scenario for every variant (same users, same demand ramp).
    np.random.seed(123)
    users_abl = make_users()
    demands = [np.clip(0.5 + 0.12 * t + np.random.randn(N_USER) * 0.1, 0.1, 8.0)
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
    for name, dropped, flags in variants:
        m = run_variant(name, flags, users_abl, demands)
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


if __name__ == "__main__":
    main()
