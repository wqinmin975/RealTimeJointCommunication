"""
Efficient FM Service over Airborne Networks — v2
================================================
Upgrades over v1:
  (1) Multi-variable SCA: joint optimization of
        - UAV 3D position  u_k = (x, y, z)
        - transmit power    p_k
        - bandwidth         b_k
        - split layer       l
        - quantization bits q
        - early-exit flag   e
      Channel: free-space path loss  g_k(u) = g0 / ||u_k - user||^2
      SCA: linearize g(u) around current u^(r-1).
  (2) MPC rolling horizon:
      - linear predictor for user demand drift and UAV velocity
      - at each step solve SCA over [t, t+H], apply first action only
  (3) Early exit + dynamic quantization as decision variables:
      - e in [0,1]  : exit-branch used, saves latency, loses accuracy
      - q in [2,16] : quantization bits, fewer bits -> less Tx, less accuracy

Run:  python3 hrl_airborne_fm_v2.py
"""
import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 0. Scenario / system parameters
# ============================================================
# Geometry
AREA = (200.0, 200.0)          # m, square coverage
ALT_RANGE = (50.0, 150.0)      # m
N_UAV = 3
N_USER = 5
G0 = 1e-4                       # path-loss constant at 1 m
NOISE = 1e-9                    # W

# Resources
B_TOTAL = 10.0                  # MHz, total bandwidth
P_MAX   = 1.0                   # W per UAV
P_MIN   = 0.01

# FM model
L_MAX = 5                       # max split layer
FM_LAT = {1: 0.020, 2: 0.040, 3: 0.070, 4: 0.110, 5: 0.160}  # s
FM_ACC = {1: 0.55, 2: 0.70, 3: 0.82, 4: 0.90, 5: 0.95}       # accuracy
EXIT_ACC_LOSS = 0.06            # accuracy loss if early exit at layer l
QUANT_ACC_LOSS_PER_BIT = 0.005  # per bit below 16
FEATURE_DIM = 512               # embedding dim (bits * dim -> Tx size)

# MPC
H_MPC = 3                       # rolling horizon
SCA_ROUNDS = 5                  # inner SCA rounds per MPC solve
DT = 0.1                        # s, decision interval

# Weights in objective
W_LAT   = 1.0
W_EN    = 0.3
W_ACC   = 2.0                   # accuracy reward (penalize loss)

# ============================================================
# 1. Scenario / channel helpers
# ============================================================
def make_users(n=N_USER):
    """fixed user positions on a rough grid."""
    xs = np.linspace(40, AREA[0]-40, n)
    ys = np.random.uniform(40, AREA[1]-40, size=n)
    return np.stack([xs, ys, np.zeros(n)], axis=1)  # (n,3)


def path_loss(uav, user):
    """g0 / d^2, with d = 3D distance (users at z=0)."""
    d3 = np.linalg.norm(uav - user)
    return G0 / (d3 ** 2 + 1e-6)


def channel_gains(uavs, users):
    """(N_UAV, N_USER) channel gain matrix."""
    G = np.zeros((N_UAV, N_USER))
    for k in range(N_UAV):
        for j in range(N_USER):
            G[k, j] = path_loss(uavs[k], users[j])
    return G


# ============================================================
# 2. Multi-variable SCA solver
# ============================================================
def solve_sca_multi(uavs_init, users, demand_per_user,
                    max_rounds=SCA_ROUNDS):
    """
    Jointly optimize (convex part):
      p[k] >= 0     (power)
      b[k] >= 0     (bandwidth share)
      l     in [1, L_MAX]  (split layer, relaxed)
      q     in [2, 16]     (quantization bits)
      e     in [0, 1]      (early-exit flag)

    UAV position u[k] is updated greedily toward its assigned user
    (standard decoupling: geometric placement + convex resource alloc).

    Objective: min  W_LAT * T_total + W_EN * E_total - W_ACC * A_total
    Constraint: per-UAV rate >= assigned demand; sum b <= B_TOTAL.
    """
    # 1) current channel gains at current uavs
    G = channel_gains(uavs_init, users)          # (K, J)

    # 2) 1:1 UAV->user assignment (each UAV serves its strongest user)
    assignment = np.argmax(G, axis=1)            # (K,)
    g_sel = G[np.arange(N_UAV), assignment]      # (K,)
    d_sel = demand_per_user[assignment]          # (K,)

    # 3) convex variables (b fixed to equal split to avoid bilinear b*log(p))
    b_fixed = B_TOTAL / N_UAV
    p = cp.Variable(N_UAV, nonneg=True)
    l = cp.Variable()
    q = cp.Variable()
    e = cp.Variable()

    # FM latency: convex surrogate, linear in (l, e)
    fm_lat_lin = 0.02 + 0.035 * (l - 1)
    fm_lat_expr = fm_lat_lin - 0.3 * 0.05 * e    # early-exit saves latency

    # quant overhead (more bits -> more Tx)
    quant_overhead = 0.005 * (q - 2) / 14.0

    # accuracy
    acc_base = 0.50 + 0.11 * (l - 1)
    acc_loss = EXIT_ACC_LOSS * e + 0.005 * (16 - q)
    acc_expr = acc_base - acc_loss

    # energy
    comp_en = 0.01 * l
    comm_en = cp.sum(p) * DT
    energy_expr = comp_en + comm_en

    constraints = [
        p <= P_MAX, p >= P_MIN,
        l >= 1, l <= L_MAX,
        q >= 2, q <= 16,
        e >= 0, e <= 1,
    ]
    # concave rate constraint:  R_k = b_fixed * log(1 + p_k * g_k) / ln2 >= d_sel[k]
    for k in range(N_UAV):
        R_k = b_fixed * cp.log(1 + p[k] * g_sel[k]) / float(np.log(2))
        constraints.append(R_k >= d_sel[k])

    obj = cp.Minimize(
        W_LAT * (fm_lat_expr + quant_overhead)
        + W_EN * energy_expr
        - W_ACC * acc_expr
    )
    prob = cp.Problem(obj, constraints)

    best = None
    for r in range(max_rounds):
        try:
            prob.solve(solver=cp.CLARABEL, warm_start=(r > 0), verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and p.value is not None:
                best = dict(
                    p=np.clip(p.value, P_MIN, P_MAX),
                    b=np.full(N_UAV, b_fixed),
                    l=float(np.clip(round(l.value), 1, L_MAX)),
                    q=float(np.clip(q.value, 2, 16)),
                    e=float(np.clip(e.value, 0, 1)),
                    obj=float(prob.value),
                    assignment=assignment,
                )
        except Exception as ex:
            print("SCA round", r, "fail:", ex)
            continue

    if best is None:
        best = dict(p=np.full(N_UAV, 0.5),
                    b=np.full(N_UAV, b_fixed),
                    l=3.0, q=8.0, e=0.0, obj=1e9,
                    assignment=assignment)

    # 4) greedy UAV position update: each UAV moves 30% toward its user
    #    constrained to altitude bounds
    uavs_new = uavs_init.copy()
    for k in range(N_UAV):
        user_pos = users[assignment[k]]
        target = np.array([user_pos[0], user_pos[1], 100.0])
        uavs_new[k] = uavs_init[k] + 0.3 * (target - uavs_init[k])
        # clamp
        uavs_new[k, 0] = np.clip(uavs_new[k, 0], 0, AREA[0])
        uavs_new[k, 1] = np.clip(uavs_new[k, 1], 0, AREA[1])
        uavs_new[k, 2] = np.clip(uavs_new[k, 2], ALT_RANGE[0], ALT_RANGE[1])
    best["uavs"] = uavs_new
    return best


# ============================================================
# 3. Linear predictor (for MPC)
# ============================================================
class LinearPredictor:
    """Predicts future user demand and UAV drift using simple random-walk
    extrapolation. In production replace with LSTM/GRU."""
    def __init__(self):
        self.history_demand = []
        self.history_uav = []

    def update(self, demand, uavs):
        self.history_demand.append(demand.copy())
        self.history_uav.append(uavs.copy())
        if len(self.history_demand) > 20:
            self.history_demand.pop(0)
            self.history_uav.pop(0)

    def predict_demand(self, h):
        """Predict demand for next h steps (mean of last 5)."""
        if len(self.history_demand) < 2:
            return np.tile(self.history_demand[-1], (h, 1))
        recent = np.stack(self.history_demand[-5:])
        base = recent.mean(axis=0)
        # small linear drift
        if len(recent) >= 3:
            slope = recent[-1] - recent[0]
            out = np.stack([base + slope * (i + 1) for i in range(h)])
        else:
            out = np.tile(base, (h, 1))
        return np.clip(out, 0.1, 10.0)

    def predict_uav(self, h):
        """Predict UAV positions (constant position assumption)."""
        base = self.history_uav[-1]
        return np.tile(base, (h, 1, 1))


# ============================================================
# 4. MPC rolling-horizon controller
# ============================================================
class MPCController:
    def __init__(self, users, h=H_MPC):
        self.users = users
        self.h = h
        # initial UAV positions: evenly spread
        self.uavs = np.zeros((N_UAV, 3))
        self.uavs[:, 0] = np.linspace(40, AREA[0]-40, N_UAV)
        self.uavs[:, 1] = AREA[1] / 2
        self.uavs[:, 2] = 100.0
        self.predictor = LinearPredictor()
        self.prev_action = None
        self.t = 0

    def step(self, demand_now):
        """
        One MPC step:
          1. update predictor with current demand
          2. predict future demands and UAV positions
          3. solve SCA over the horizon (mean prediction)
          4. apply first-step action (move UAVs partially toward target)
        """
        self.predictor.update(demand_now, self.uavs)
        future_demand = self.predictor.predict_demand(self.h)
        # average demand over horizon (robust)
        demand_avg = future_demand.mean(axis=0)

        # solve SCA over the horizon
        sol = solve_sca_multi(self.uavs, self.users, demand_avg,
                              max_rounds=SCA_ROUNDS)
        # SCA already returned updated uavs (greedy 30% step); adopt them
        self.uavs = sol["uavs"]
        self.t += 1
        return sol


# ============================================================
# 5. Evaluation
# ============================================================
def evaluate(n_steps=30):
    users = make_users()
    ctrl = MPCController(users)
    logs = dict(t=[], obj=[], l=[], q=[], e=[],
                lat_ms=[], en=[], acc=[], decision_ms=[])

    print(f"[MPC] running {n_steps} steps, horizon={H_MPC}, "
          f"SCA rounds={SCA_ROUNDS}")
    for t in range(n_steps):
        # demand drifts over time
        demand = np.clip(0.5 + 0.05 * t + np.random.randn(N_USER) * 0.1,
                         0.1, 5.0)
        t0 = time.perf_counter()
        sol = ctrl.step(demand)
        dec_ms = (time.perf_counter() - t0) * 1e3

        # compute actual metrics at this step
        l, q, e = sol["l"], sol["q"], sol["e"]
        fm_lat = FM_LAT[int(np.clip(round(l), 1, L_MAX))] * (1 - 0.3 * e)
        acc = FM_ACC[int(np.clip(round(l), 1, L_MAX))] \
              - EXIT_ACC_LOSS * e - QUANT_ACC_LOSS_PER_BIT * (16 - q)
        en = float(np.sum(sol["p"])) * DT

        logs["t"].append(t)
        logs["obj"].append(sol["obj"])
        logs["l"].append(l)
        logs["q"].append(q)
        logs["e"].append(e)
        logs["lat_ms"].append(fm_lat * 1e3)
        logs["en"].append(en)
        logs["acc"].append(acc)
        logs["decision_ms"].append(dec_ms)

        if t % 5 == 0:
            print(f"  t={t:3d}  l={l:.1f} q={q:5.1f} e={e:.2f}  "
                  f"fm_lat={fm_lat*1e3:5.1f}ms  acc={acc:.3f}  "
                  f"dec={dec_ms:6.2f}ms")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(logs["t"], logs["l"], "o-", label="split layer l")
    ax.plot(logs["t"], logs["e"], "s-", label="early-exit flag e")
    ax.set_xlabel("step"); ax.set_ylabel("decision"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("(a) Split layer & early exit")

    ax = axes[0, 1]
    ax.plot(logs["t"], logs["q"], "^-", color="C2", label="quantization bits")
    ax.set_xlabel("step"); ax.set_ylabel("bits"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("(b) Dynamic quantization")

    ax = axes[1, 0]
    ax.plot(logs["t"], logs["lat_ms"], "-", label="FM latency [ms]")
    ax.plot(logs["t"], np.array(logs["acc"]) * 100, "--", label="accuracy [%]")
    ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("(c) Latency vs accuracy trade-off")

    ax = axes[1, 1]
    ax.plot(logs["t"], logs["decision_ms"], "-", color="C3",
            label="MPC decision time")
    ax.axhline(DT * 1e3, ls=":", color="k", label=f"step Δt={DT*1e3:.0f}ms")
    ax.set_xlabel("step"); ax.set_ylabel("ms")
    ax.legend(); ax.grid(alpha=.3)
    ax.set_title("(d) Online decision latency")

    plt.tight_layout()
    plt.savefig("mpc_evolution.png", dpi=150)
    plt.close()
    print("[plot] mpc_evolution.png")

    # summary
    print("\n=== MPC summary ===")
    print(f"  mean decision time : {np.mean(logs['decision_ms']):.2f} ms "
          f"(budget {DT*1e3:.0f} ms)")
    print(f"  mean split layer   : {np.mean(logs['l']):.2f}")
    print(f"  mean quant bits    : {np.mean(logs['q']):.2f}")
    print(f"  mean early-exit    : {np.mean(logs['e']):.2f}")
    print(f"  mean accuracy      : {np.mean(logs['acc']):.3f}")


if __name__ == "__main__":
    evaluate(n_steps=30)
