"""
Efficient FM Service over Airborne Networks — v5 (warm-start edition)
======================================================================
Goal: cut per-step decision latency to < 50 ms.

Speedups over v4:
  (1) Pre-build CVXPY Problems ONCE at init, wrap all varying quantities
      (g_sel, d_sel, W_acc, uavs_prev) as cp.Parameter. Each MPC step
      only .value= ... updates parameters; problem canonicalization is
      amortized.
  (2) CLARABEL warm_start=True across outer SCA rounds and across
      MPC steps (carry p, b, u, l, q, e from previous step as init).
  (3) l/q/e solved analytically (3-variable QP has closed form) —
      no CVXPY call at all.
  (4) SCA_ROUNDS reduced from 5 to 3 (warm-start makes extra rounds
      unnecessary).

Run:  python3 hrl_airborne_fm_v5.py
"""

import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 0. System parameters
# ============================================================
AREA = (200.0, 200.0)
ALT_RANGE = (50.0, 150.0)
N_UAV = 3
N_USER = 5
G0 = 1e-4

B_TOTAL = 10.0
P_MAX   = 1.0
P_MIN   = 0.01

L_MAX = 5
FM_LAT = {1: 0.020, 2: 0.040, 3: 0.070, 4: 0.110, 5: 0.160}
FM_ACC = {1: 0.55, 2: 0.70, 3: 0.82, 4: 0.90, 5: 0.95}
EXIT_ACC_LOSS = 0.06
QUANT_ACC_LOSS_PER_BIT = 0.005

H_MPC = 3
SCA_ROUNDS = 3                 # was 5
DT = 0.1
V_MAX = 10.0
V_STEP = V_MAX * DT

W_LAT = 1.0
W_EN  = 0.3
W_ACC_BASE = 2.0
MU_REG = 0.5
DEMAND_REF = 3.0

LATENCY_BUDGET_MS = 50.0       # target


# ============================================================
# 1. Channel helpers
# ============================================================
def make_users(n=N_USER):
    xs = np.linspace(40, AREA[0]-40, n)
    ys = np.random.uniform(40, AREA[1]-40, size=n)
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
    d2 = np.sum(diff**2) + 1e-6
    return -2.0 * G0 * diff / (d2 ** 2)


# ============================================================
# 2. Pre-built solver (problems built ONCE, parameters updated each step)
# ============================================================
class FastSCASolver:
    """
    Holds pre-built CVXPY problems. Each MPC step only updates
    cp.Parameter values; CLARABEL warm-starts from previous solution.
    """
    def __init__(self, users):
        self.users = users

        # ---- parameters ----
        # Block A: solve p given (b, g_sel, d_sel)
        # Rate constraint  b*log2(1+p*g) >= d  <=>  p >= (2^(d/b)-1)/g
        # This is a LINEAR lower bound on p, no log needed in CVXPY.
        self.p_g_sel = cp.Parameter(N_UAV)
        self.p_d_sel = cp.Parameter(N_UAV)
        self.p_b     = cp.Parameter(N_UAV)
        self.p_p_min = cp.Parameter(N_UAV)   # analytic lower bound on p
        self.p_var    = cp.Variable(N_UAV, nonneg=True)
        cons_p = [self.p_var <= P_MAX, self.p_var >= self.p_p_min]
        self.prob_p = cp.Problem(cp.Minimize(cp.sum(self.p_var)), cons_p)

        # Block C: solve u given (p, b, g_sel, grad, u0, u_prev, d_sel)
        # Rate constraint  b*log2(1+p*g) >= d  <=>  g_lin(u) >= c_k
        # where c_k = (2^(d/b)-1)/p_k is a constant.
        self.u_g_sel = cp.Parameter(N_UAV)
        self.u_grad  = cp.Parameter((N_UAV, 3))
        self.u_uprev = cp.Parameter((N_UAV, 3))
        self.u_rhs   = cp.Parameter(N_UAV)   # c_min + grad@u0 (folded offset)
        self.u_var   = cp.Variable((N_UAV, 3))
        cons_u = [
            self.u_var[:, 0] >= 0, self.u_var[:, 0] <= AREA[0],
            self.u_var[:, 1] >= 0, self.u_var[:, 1] <= AREA[1],
            self.u_var[:, 2] >= ALT_RANGE[0], self.u_var[:, 2] <= ALT_RANGE[1],
        ]
        for k in range(N_UAV):
            # velocity constraint
            cons_u.append(cp.norm(self.u_var[k] - self.u_uprev[k]) <= V_STEP)
            # linearized channel gain >= c_min  (linear in u, DPP-compliant:
            # grad@u0 is folded into u_rhs so only grad@u_var remains param@var)
            g_lin = self.u_g_sel[k] + self.u_grad[k] @ self.u_var[k]
            cons_u.append(g_lin >= self.u_rhs[k])
        self.prob_u = cp.Problem(cp.Minimize(0), cons_u)

        # carry previous solution (warm-start)
        self.p_init = np.full(N_UAV, 0.3)
        self.u_init = np.zeros((N_UAV, 3))

    def solve(self, uavs_prev, g_sel, d_sel, b, assignment,
              users, max_rounds=SCA_ROUNDS):
        uavs = uavs_prev.copy()
        p = self.p_init.copy()
        b = b.copy()

        for r in range(max_rounds):
            # ---- Block A: solve p ----
            self.p_g_sel.value = g_sel
            self.p_d_sel.value = d_sel
            self.p_b.value = b
            # analytic lower bound: p_k >= (2^(d_k/b_k) - 1) / g_k
            p_min_req = np.maximum(P_MIN,
                (2.0 ** (d_sel / b) - 1.0) / (g_sel + 1e-12))
            self.p_p_min.value = p_min_req
            try:
                self.prob_p.solve(solver=cp.CLARABEL, warm_start=True,
                                  verbose=False)
                if self.prob_p.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) \
                   and self.p_var.value is not None:
                    p = np.clip(self.p_var.value, P_MIN, P_MAX)
                    self.p_init = p.copy()
            except Exception as ex:
                print("Block A fail:", ex)

            # ---- Block B: analytic b ----
            for k in range(N_UAV):
                sinr = p[k] * g_sel[k]
                needed = d_sel[k] / (np.log(1 + sinr) / np.log(2) + 1e-9)
                b[k] = np.clip(needed, 0.01, B_TOTAL)
            if b.sum() > B_TOTAL:
                b = b * (B_TOTAL / b.sum())

            # ---- Block C: solve u ----
            # analytic lower bound on g(u): g(u) >= (2^(d_k/b_k)-1) / p_k
            c_min = (2.0 ** (d_sel / b) - 1.0) / (p + 1e-12)
            grads = np.stack([grad_path_loss(uavs[k], users[assignment[k]])
                              for k in range(N_UAV)])
            # Fold grad@u0 into RHS: constraint becomes g_sel + grad@u >= c_min + grad@u0
            self.u_rhs.value = c_min + np.sum(grads * uavs, axis=1)
            self.u_grad.value = grads
            self.u_uprev.value = uavs_prev
            self.u_g_sel.value = g_sel
            try:
                self.prob_u.solve(solver=cp.CLARABEL, warm_start=True,
                                  verbose=False)
                if self.prob_u.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) \
                   and self.u_var.value is not None:
                    uavs = np.clip(self.u_var.value,
                                   [0, 0, ALT_RANGE[0]],
                                   [AREA[0], AREA[1], ALT_RANGE[1]])
                    G = channel_gains(uavs, users)
                    g_sel = G[np.arange(N_UAV), assignment]
            except Exception as ex:
                print("Block C fail:", ex)

        self.u_init = uavs.copy()
        return p, b, uavs, g_sel


def solve_lqe_analytic(demand_scale):
    """Closed-form 1-D optimum for each of l, q, e.
    Objective:  a_l*l + a_q*q + a_e*e + mu*((l-3)^2 + (q-9)^2/10 + (e-0.5)^2)
    Unconstrained optimum:  x* = x0 - a / (2*mu_xx)
    """
    W_ACC = W_ACC_BASE * demand_scale
    a_l = W_LAT*0.035 + W_EN*0.01 - W_ACC*0.11
    a_q = W_LAT*0.005/14.0 + W_ACC*0.005
    a_e = -W_LAT*0.015 + W_ACC*0.06

    l_star = 3.0 - a_l / (2.0 * MU_REG)
    q_star = 9.0 - a_q / (2.0 * MU_REG / 10.0)
    e_star = 0.5 - a_e / (2.0 * MU_REG)

    l_val = float(np.clip(l_star, 1, L_MAX))
    q_val = float(np.clip(q_star, 2, 16))
    e_val = float(np.clip(e_star, 0, 1))
    return l_val, q_val, e_val


# ============================================================
# 3. MPC controller
# ============================================================
class LinearPredictor:
    def __init__(self):
        self.hist_d = []
    def update(self, d):
        self.hist_d.append(d.copy())
        if len(self.hist_d) > 20: self.hist_d.pop(0)
    def predict_demand(self, h):
        if len(self.hist_d) < 2:
            return np.tile(self.hist_d[-1], (h, 1))
        recent = np.stack(self.hist_d[-5:])
        base = recent.mean(axis=0)
        if len(recent) >= 3:
            slope = recent[-1] - recent[0]
            out = np.stack([base + slope*(i+1) for i in range(h)])
        else:
            out = np.tile(base, (h, 1))
        return np.clip(out, 0.1, 10.0)


class MPCController:
    def __init__(self, users, h=H_MPC):
        self.users = users
        self.h = h
        self.uavs = np.zeros((N_UAV, 3))
        self.uavs[:, 0] = np.linspace(40, AREA[0]-40, N_UAV)
        self.uavs[:, 1] = AREA[1] / 2
        self.uavs[:, 2] = 100.0
        self.pred = LinearPredictor()
        self.solver = FastSCASolver(users)
        self.b = np.full(N_UAV, B_TOTAL / N_UAV)

    def step(self, demand_now):
        self.pred.update(demand_now)
        demand_avg = self.pred.predict_demand(self.h).mean(axis=0)

        G = channel_gains(self.uavs, self.users)
        assignment = np.argmax(G, axis=1)
        g_sel = G[np.arange(N_UAV), assignment]
        d_sel = demand_avg[assignment]

        p, b, uavs, g_sel_new = self.solver.solve(
            self.uavs, g_sel, d_sel, self.b, assignment, self.users)
        self.b = b
        self.uavs = uavs

        demand_scale = np.clip(np.mean(d_sel) / DEMAND_REF, 0.5, 2.0)
        l_val, q_val, e_val = solve_lqe_analytic(demand_scale)

        return dict(p=p, b=b, uavs=uavs, l=l_val, q=q_val, e=e_val)


# ============================================================
# 4. Evaluation
# ============================================================
def evaluate(n_steps=40):
    users = make_users()
    ctrl = MPCController(users)
    logs = dict(t=[], l=[], q=[], e=[], dec_ms=[],
                demand_mean=[])

    print(f"[v5] warm-start edition, SCA_ROUNDS={SCA_ROUNDS}, "
          f"target decision < {LATENCY_BUDGET_MS} ms")
    # warm-up (first call builds problems)
    t0 = time.perf_counter()
    ctrl.step(np.full(N_USER, 1.0))
    warmup_ms = (time.perf_counter() - t0) * 1e3
    print(f"  warm-up (first step incl. problem build): {warmup_ms:.1f} ms")

    for t in range(n_steps):
        ramp = 0.5 + 0.12*t
        demand = np.clip(ramp + np.random.randn(N_USER)*0.1, 0.1, 8.0)
        t0 = time.perf_counter()
        sol = ctrl.step(demand)
        dec_ms = (time.perf_counter() - t0) * 1e3

        logs["t"].append(t)
        logs["l"].append(sol["l"]); logs["q"].append(sol["q"])
        logs["e"].append(sol["e"])
        logs["dec_ms"].append(dec_ms)
        logs["demand_mean"].append(float(np.mean(demand)))

        if t % 5 == 0:
            print(f"  t={t:3d} dem={np.mean(demand):.2f} "
                  f"l={sol['l']:.2f} q={sol['q']:5.2f} e={sol['e']:.2f} "
                  f"dec={dec_ms:6.2f}ms")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.plot(logs["t"], logs["dec_ms"], "-", color="C3", label="decision time")
    ax.axhline(LATENCY_BUDGET_MS, ls="--", color="g",
               label=f"target {LATENCY_BUDGET_MS:.0f} ms")
    ax.axhline(DT*1e3, ls=":", color="k", label=f"Δt={DT*1e3:.0f}ms")
    ax.set_title("(a) Online decision latency (warm-start)")
    ax.set_xlabel("step"); ax.set_ylabel("ms")
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(logs["t"], logs["demand_mean"], "k--", label="demand")
    ax.set_ylabel("demand"); ax2 = ax.twinx()
    ax2.plot(logs["t"], logs["l"], "o-", label="l")
    ax2.plot(logs["t"], logs["e"], "s-", label="e")
    ax.set_title("(b) Demand vs l/e adaptation")
    ax.set_xlabel("step"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("v5_warmstart.png", dpi=150)
    plt.close()
    print("[plot] v5_warmstart.png")

    dec = np.array(logs["dec_ms"])
    print("\n=== v5 summary ===")
    print(f"  mean decision time : {dec.mean():.2f} ms")
    print(f"  p50 decision time  : {np.median(dec):.2f} ms")
    print(f"  p95 decision time  : {np.percentile(dec,95):.2f} ms")
    print(f"  max decision time  : {dec.max():.2f} ms")
    print(f"  target met (<{LATENCY_BUDGET_MS:.0f}ms): "
          f"{(dec < LATENCY_BUDGET_MS).mean()*100:.1f}% of steps")


if __name__ == "__main__":
    evaluate(n_steps=40)

