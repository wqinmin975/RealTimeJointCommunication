"""
Efficient FM Service over Airborne Networks — v3
================================================
Upgrades over v2:
  (1) Block-coordinate descent for (p, b, u):
        Block A: fix b, u -> solve p via CLARABEL  (concave rate in p)
        Block B: fix p, u -> analytically set b_k* = d_k / log(1+p_k*g_k)
        Block C: fix p, b -> solve u via CLARABEL
                 (linearize  g(u) = g0/d^2  around current u0;
                  tangent is a lower bound because 1/d^2 is convex)
  (2) l, q, e optimized as a tiny LP inside each outer SCA round.

Run:  python3 hrl_airborne_fm_v3.py
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
NOISE = 1e-9

B_TOTAL = 10.0                  # MHz
P_MAX   = 1.0
P_MIN   = 0.01

L_MAX = 5
FM_LAT = {1: 0.020, 2: 0.040, 3: 0.070, 4: 0.110, 5: 0.160}
FM_ACC = {1: 0.55, 2: 0.70, 3: 0.82, 4: 0.90, 5: 0.95}
EXIT_ACC_LOSS = 0.06
QUANT_ACC_LOSS_PER_BIT = 0.005

H_MPC = 3
SCA_ROUNDS = 5
DT = 0.1

W_LAT = 1.0
W_EN  = 0.3
W_ACC = 2.0


# ============================================================
# 1. Scenario / channel
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
    """gradient of g(u) = G0/d^2 w.r.t. uav position.
    dg/du = -2*G0*(uav-user) / d^4
    """
    diff = uav - user
    d2 = np.sum(diff**2) + 1e-6
    return -2.0 * G0 * diff / (d2 ** 2)


# ============================================================
# 2. Block-coordinate SCA solver
# ============================================================
def solve_sca_block_cd(uavs_init, users, demand_per_user,
                       max_rounds=SCA_ROUNDS):
    """
    Block-coordinate descent:
      outer loop (r = 1..max_rounds):
        (a) solve l, q, e (small LP)
        (b) Block A: fix b, u -> solve p (concave rate in p)
        (c) Block B: fix p, u -> analytic b
        (d) Block C: fix p, b -> solve u (linearized 1/d^2)
    """
    # ---- init ----
    G = channel_gains(uavs_init, users)
    assignment = np.argmax(G, axis=1)
    g_sel = G[np.arange(N_UAV), assignment]
    d_sel = demand_per_user[assignment]

    uavs = uavs_init.copy()
    p = np.full(N_UAV, 0.3)
    b = np.full(N_UAV, B_TOTAL / N_UAV)
    l_val, q_val, e_val = 3.0, 8.0, 0.0

    for r in range(max_rounds):
        # ---- (a) l, q, e as tiny LP ----
        # Objective linear in (l,q,e):
        #   min  a_l*l + a_q*q + a_e*e
        # coefficients derived analytically from objective.
        a_l = W_LAT * 0.035 + W_EN * 0.01 - W_ACC * 0.11
        a_q = W_LAT * 0.005/14.0 + W_ACC * 0.005
        a_e = -W_LAT * 0.015 + W_ACC * EXIT_ACC_LOSS
        # pick bounds
        l_val = 1.0 if a_l < 0 else float(L_MAX)
        q_val = 2.0 if a_q < 0 else 16.0
        e_val = 0.0 if a_e < 0 else 1.0

        # ---- (b) Block A: solve p with fixed b, u ----
        p_var = cp.Variable(N_UAV, nonneg=True)
        cons = [p_var <= P_MAX, p_var >= P_MIN]
        for k in range(N_UAV):
            R_k = b[k] * cp.log(1 + p_var[k] * g_sel[k]) / float(np.log(2))
            cons.append(R_k >= d_sel[k])
        obj_p = cp.Minimize(cp.sum(p_var))
        prob_p = cp.Problem(obj_p, cons)
        try:
            prob_p.solve(solver=cp.CLARABEL, verbose=False)
            if prob_p.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and p_var.value is not None:
                p = np.clip(p_var.value, P_MIN, P_MAX)
        except Exception as ex:
            print("Block A fail:", ex)

        # ---- (c) Block B: analytic b with fixed p, u ----
        # b_k >= d_sel[k] / log2(1 + p_k * g_sel[k])
        for k in range(N_UAV):
            sinr = p[k] * g_sel[k]
            needed = d_sel[k] / (np.log(1 + sinr) / np.log(2) + 1e-9)
            b[k] = np.clip(needed, 0.01, B_TOTAL)
        # renormalize to fit B_TOTAL
        if b.sum() > B_TOTAL:
            b = b * (B_TOTAL / b.sum())

        # ---- (d) Block C: solve u with linearized 1/d^2 ----
        u_var = cp.Variable((N_UAV, 3))
        cons_u = [
            u_var[:, 0] >= 0, u_var[:, 0] <= AREA[0],
            u_var[:, 1] >= 0, u_var[:, 1] <= AREA[1],
            u_var[:, 2] >= ALT_RANGE[0], u_var[:, 2] <= ALT_RANGE[1],
        ]
        # channel lower bound (convex tangent):
        #   g_lin(u) = g(u0) + grad^T (u - u0)  <=  g(u)
        rate_terms = 0
        for k in range(N_UAV):
            u0_k = uavs[k]
            user_k = users[assignment[k]]
            g0_k = g_sel[k]
            grad_k = grad_path_loss(u0_k, user_k)         # (3,)
            g_lin = g0_k + grad_k @ (u_var[k] - u0_k)      # affine
            R_k = b[k] * cp.log(1 + p[k] * g_lin) / float(np.log(2))
            cons_u.append(R_k >= d_sel[k])
            # objective: maximize sum of squared distances to keep UAVs spread
            # (optional; here just feasibility)
        prob_u = cp.Problem(cp.Minimize(0), cons_u)
        try:
            prob_u.solve(solver=cp.CLARABEL, verbose=False)
            if prob_u.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u_var.value is not None:
                uavs = np.clip(u_var.value,
                               [0, 0, ALT_RANGE[0]],
                               [AREA[0], AREA[1], ALT_RANGE[1]])
                # recompute g_sel at new uavs
                G = channel_gains(uavs, users)
                g_sel = G[np.arange(N_UAV), assignment]
        except Exception as ex:
            print("Block C fail:", ex)

    return dict(
        uavs=uavs, p=p, b=b,
        l=l_val, q=q_val, e=e_val,
        assignment=assignment,
    )


# ============================================================
# 3. MPC controller (same as v2)
# ============================================================
class LinearPredictor:
    def __init__(self):
        self.hist_d = []
        self.hist_u = []
    def update(self, d, u):
        self.hist_d.append(d.copy()); self.hist_u.append(u.copy())
        if len(self.hist_d) > 20:
            self.hist_d.pop(0); self.hist_u.pop(0)
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

    def step(self, demand_now):
        self.pred.update(demand_now, self.uavs)
        demand_avg = self.pred.predict_demand(self.h).mean(axis=0)
        sol = solve_sca_block_cd(self.uavs, self.users, demand_avg)
        self.uavs = sol["uavs"]
        return sol


# ============================================================
# 4. Evaluation
# ============================================================
def evaluate(n_steps=30):
    users = make_users()
    ctrl = MPCController(users)
    logs = dict(t=[], obj=[], l=[], q=[], e=[],
                lat_ms=[], acc=[], dec_ms=[],
                p_mean=[], b_sum=[],
                x_uav=[], y_uav=[])

    print(f"[v3] MPC steps={n_steps}, horizon={H_MPC}, SCA rounds={SCA_ROUNDS}")
    for t in range(n_steps):
        demand = np.clip(0.5 + 0.05*t + np.random.randn(N_USER)*0.1, 0.1, 5.0)
        t0 = time.perf_counter()
        sol = ctrl.step(demand)
        dec_ms = (time.perf_counter() - t0) * 1e3

        l, q, e = sol["l"], sol["q"], sol["e"]
        fm_lat = FM_LAT[int(np.clip(round(l),1,L_MAX))] * (1 - 0.3*e)
        acc = FM_ACC[int(np.clip(round(l),1,L_MAX))] \
              - EXIT_ACC_LOSS*e - QUANT_ACC_LOSS_PER_BIT*(16-q)

        logs["t"].append(t)
        logs["l"].append(l); logs["q"].append(q); logs["e"].append(e)
        logs["lat_ms"].append(fm_lat*1e3)
        logs["acc"].append(acc)
        logs["dec_ms"].append(dec_ms)
        logs["p_mean"].append(float(np.mean(sol["p"])))
        logs["b_sum"].append(float(np.sum(sol["b"])))
        logs["x_uav"].append(sol["uavs"][:, 0].copy())
        logs["y_uav"].append(sol["uavs"][:, 1].copy())

        if t % 5 == 0:
            print(f"  t={t:3d} l={l:.1f} q={q:5.1f} e={e:.2f} "
                  f"fm_lat={fm_lat*1e3:5.1f}ms acc={acc:.3f} "
                  f"p_mean={np.mean(sol['p']):.3f}W "
                  f"b_sum={np.sum(sol['b']):.2f}MHz "
                  f"dec={dec_ms:6.2f}ms")

    # ---- plots ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (a) l & e
    ax = axes[0,0]
    ax.plot(logs["t"], logs["l"], "o-", label="split layer l")
    ax.plot(logs["t"], logs["e"], "s-", label="early-exit e")
    ax.set_title("(a) Split layer & early exit")
    ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=.3)

    # (b) q
    ax = axes[0,1]
    ax.plot(logs["t"], logs["q"], "^-", color="C2")
    ax.set_title("(b) Dynamic quantization [bits]")
    ax.set_xlabel("step"); ax.grid(alpha=.3)

    # (c) latency & accuracy
    ax = axes[0,2]
    ax.plot(logs["t"], logs["lat_ms"], "-", label="FM lat [ms]")
    ax.plot(logs["t"], np.array(logs["acc"])*100, "--", label="accuracy [%]")
    ax.set_title("(c) Latency vs accuracy")
    ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=.3)

    # (d) power & bandwidth
    ax = axes[1,0]
    ax.plot(logs["t"], logs["p_mean"], "-", label="mean p [W]")
    ax2 = ax.twinx()
    ax2.plot(logs["t"], logs["b_sum"], "-", color="C3", label="sum b [MHz]")
    ax.set_title("(d) Power & bandwidth")
    ax.set_xlabel("step"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(alpha=.3)

    # (e) UAV trajectory
    ax = axes[1,1]
    for k in range(N_UAV):
        xs = [x[k] for x in logs["x_uav"]]
        ys = [y[k] for y in logs["y_uav"]]
        ax.plot(xs, ys, "o-", label=f"UAV{k}")
        ax.plot(users[k,0], users[k,1], "r*", markersize=15)
    ax.set_title("(e) UAV 2D trajectory (* = user)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(); ax.grid(alpha=.3)
    ax.set_xlim(0, AREA[0]); ax.set_ylim(0, AREA[1])

    # (f) decision latency
    ax = axes[1,2]
    ax.plot(logs["t"], logs["dec_ms"], "-", color="C4", label="decision time")
    ax.axhline(DT*1e3, ls=":", color="k", label=f"Δt={DT*1e3:.0f}ms")
    ax.set_title("(f) Online decision latency")
    ax.set_xlabel("step"); ax.set_ylabel("ms")
    ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("v3_evolution.png", dpi=150)
    plt.close()
    print("[plot] v3_evolution.png")

    print("\n=== v3 summary ===")
    print(f"  mean decision time : {np.mean(logs['dec_ms']):.2f} ms")
    print(f"  mean split layer   : {np.mean(logs['l']):.2f}")
    print(f"  mean quant bits    : {np.mean(logs['q']):.2f}")
    print(f"  mean early-exit    : {np.mean(logs['e']):.2f}")
    print(f"  mean accuracy      : {np.mean(logs['acc']):.3f}")
    print(f"  mean power         : {np.mean(logs['p_mean']):.3f} W")
    print(f"  mean sum bandwidth : {np.mean(logs['b_sum']):.2f} MHz")


if __name__ == "__main__":
    evaluate(n_steps=30)
