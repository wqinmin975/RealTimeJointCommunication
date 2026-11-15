"""
Optimality-gap analysis for the airborne FM scheduler.
=====================================================
Compares the per-step *objective value* of
  * v4  -- un-accelerated SCA baseline (5 outer rounds, 3 CLARABEL / round)
  * v5  -- accelerated method        (3 outer rounds, warm-start, <50 ms)
against a *full-convergence SCA optimum* used as the reference.

The reference is the SAME deployed method (feasibility-only UAV block, exactly
as v4/v5) run with many outer SCA rounds (30) and several UAV restarts per
step; it therefore represents the best achievable objective of the deployed
algorithm family -- the "full un-accelerated SCA optimum" cited in the paper's
Conclusion.  Because the per-round l/q/e QP and the p lower bound are solved to
global optimality, the only remaining combinatorial knob is the UAV->user
assignment (which sets the required rate d_sel and hence l,q,e); the restarts
probe that.  This isolates:
  (1) acceleration gap  = (J_v5 - J_v4)        -- price of 3 rounds + warm-start
  (2) convergence gap   = (J_v4 - J_ref)       -- price of using 5 vs 30 rounds
The total sub-optimality of v5 vs the reference is (1)+(2).

Objective (problem (1) in main.tex):
  J = W_lat * L_FM(l,q,e) + W_en * sum_k p_k + W_acc(d) * (1 - A_FM(l,q,e))
with W_acc = W_ACC_BASE * demand_scale, demand_scale = clip(mean(d_sel)/D_REF, .5, 2).

Run:  python opt_gap.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hrl_airborne_fm_v5 as v5
import hrl_airborne_fm_v4 as v4

# ---- pull shared constants from v5 (identical to v4) ----
G0=v5.G0; N_UAV=v5.N_UAV; N_USER=v5.N_USER; B_TOTAL=v5.B_TOTAL
P_MAX=v5.P_MAX; P_MIN=v5.P_MIN; L_MAX=v5.L_MAX
FM_LAT=v5.FM_LAT; FM_ACC=v5.FM_ACC; EXIT=v5.EXIT_ACC_LOSS; QLOSS=v5.QUANT_ACC_LOSS_PER_BIT
W_LAT=v5.W_LAT; W_EN=v5.W_EN; W_ACC_BASE=v5.W_ACC_BASE; DEMAND_REF=v5.DEMAND_REF
AREA=v5.AREA; ALT=v5.ALT_RANGE; V_STEP=v5.V_STEP; H=v5.H_MPC

N_STEPS = 40
SEED = 123


def make_users(n=N_USER, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.linspace(40, AREA[0]-40, n)
    ys = rng.uniform(40, AREA[1]-40, size=n)
    return np.stack([xs, ys, np.zeros(n)], axis=1)


def eval_J(l, q, e, p, demand_scale):
    """Problem (1) objective at a realised solution."""
    W_acc = W_ACC_BASE * demand_scale
    lat = FM_LAT[int(np.clip(round(l), 1, L_MAX))] * (1.0 - 0.3*e)
    acc = (FM_ACC[int(np.clip(round(l), 1, L_MAX))]
           - EXIT*e - QLOSS*(16.0 - q))
    pwr = float(np.sum(p))
    J = W_LAT*lat + W_EN*pwr + W_acc*(1.0 - acc)
    return J, lat, acc, pwr


def recompute_pb(uavs, users, demand_avg):
    """Given a UAV layout, return assignment, g_sel, d_sel, l,q,e, p, b, demand_scale."""
    G = v5.channel_gains(uavs, users)
    assignment = np.argmax(G, axis=1)
    g_sel = G[np.arange(N_UAV), assignment]
    d_sel = demand_avg[assignment]
    demand_scale = float(np.clip(np.mean(d_sel)/DEMAND_REF, 0.5, 2.0))
    l, q, e = v5.solve_lqe_analytic(demand_scale)
    p = np.full(N_UAV, 0.3); b = np.full(N_UAV, B_TOTAL/N_UAV)
    for _ in range(6):
        sinr = p*g_sel
        needed = d_sel/(np.log(1.0+sinr)/np.log(2.0)+1e-9)
        b = np.clip(needed, 0.01, B_TOTAL)
        if b.sum() > B_TOTAL:
            b = b*(B_TOTAL/b.sum())
        p_req = np.maximum(P_MIN, (2.0**(d_sel/b)-1.0)/(g_sel+1e-12))
        p = np.clip(p_req, P_MIN, P_MAX)
    return assignment, g_sel, d_sel, demand_scale, l, q, e, p, b


# ---------------------------------------------------------------------------
# 1. v4 (un-accelerated, 5 rounds) and v5 (accelerated, 3 rounds)
# ---------------------------------------------------------------------------
def demand_avg_at(demands, t):
    """Reconstruct the linear-predictor demand average at step t (identical
    for v4, v5 and the reference, since all consume the same demand sequence)."""
    pred = v5.LinearPredictor()
    for tt in range(t+1):
        pred.update(demands[tt])
    return pred.predict_demand(H).mean(0)


def eval_solution(users, da, sol):
    """Problem-(1) objective at a realised solution."""
    G = v5.channel_gains(sol["uavs"], users)
    assign = np.argmax(G, axis=1)
    d_sel = da[assign]
    ds = float(np.clip(np.mean(d_sel)/DEMAND_REF, 0.5, 2.0))
    return eval_J(sol["l"], sol["q"], sol["e"], sol["p"], ds)


def run_v4(users, demands):
    ctrl = v4.MPCController(users, h=H)
    Js, Ps = [], []
    for t in range(N_STEPS):
        sol = ctrl.step(demands[t])
        da = demand_avg_at(demands, t)
        J, _, _, pw = eval_solution(users, da, sol)
        Js.append(J); Ps.append(pw)
    return np.array(Js), np.array(Ps)


def run_v5(users, demands):
    ctrl = v5.MPCController(users, h=H)
    Js, Ps = [], []
    for t in range(N_STEPS):
        sol = ctrl.step(demands[t])
        da = demand_avg_at(demands, t)
        J, _, _, pw = eval_solution(users, da, sol)
        Js.append(J); Ps.append(pw)
    return np.array(Js), np.array(Ps)


# ---------------------------------------------------------------------------
# 2. Reference: full-convergence SCA optimum (the SAME deployed method,
#    feasibility-only UAV block, run with many outer rounds + restarts).
#    This is exactly the "full un-accelerated SCA optimum" referred to in
#    the Conclusion.  The only free combinatorial knob left in the deployed
#    method is the UAV->user assignment (which sets d_sel and hence l,q,e),
#    so the restarts probe that; the l/q/e QP and the p lower bound are
#    solved to global optimality inside every round.
# ---------------------------------------------------------------------------
def solve_ref(users, uavs_prev, demand_avg, rounds=30, restarts=6):
    best = None; bestJ = 1e9
    for rs in range(restarts):
        if rs == 0:
            u0 = uavs_prev.copy()
        else:
            rng = np.random.RandomState(9000 + rs)
            u0 = uavs_prev + rng.uniform(-15, 15, size=uavs_prev.shape)
            u0[:, 2] = np.clip(u0[:, 2], ALT[0], ALT[1])
        sol = v4.solve_sca_block_cd(u0, users, demand_avg, max_rounds=rounds)
        J, _, _, pw = eval_solution(users, demand_avg, sol)
        if J < bestJ:
            bestJ = J
            best = dict(J=J, pw=pw, uavs=sol["uavs"])
    return best


def run_ref(users, demands):
    uavs_prev = np.zeros((N_UAV, 3))
    uavs_prev[:, 0] = np.linspace(40, AREA[0]-40, N_UAV)
    uavs_prev[:, 1] = AREA[1]/2
    uavs_prev[:, 2] = 100.0
    Js, Ps = [], []
    for t in range(N_STEPS):
        da = demand_avg_at(demands, t)
        ref = solve_ref(users, uavs_prev, da, rounds=30, restarts=6)
        Js.append(ref["J"]); Ps.append(ref["pw"])
        uavs_prev = ref["uavs"].copy()
    return np.array(Js), np.array(Ps)


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    users = make_users(seed=0)
    demands = []
    for t in range(N_STEPS):
        ramp = 0.5 + 0.12*t
        d = np.clip(ramp + np.random.randn(N_USER)*0.1, 0.1, 8.0)
        demands.append(d)
    demands = np.array(demands)

    print("Running v4 (baseline) ...")
    J_v4, P_v4 = run_v4(users, demands)
    print("Running v5 (accelerated) ...")
    J_v5, P_v5 = run_v5(users, demands)
    print("Running reference (power-aware full SCA) ...")
    J_ref, P_ref = run_ref(users, demands)

    def gap(a, b):   # relative gap of a w.r.t. b, in %
        return (a - b) / np.abs(b) * 100.0

    g_v4  = gap(J_v4,  J_ref)
    g_v5  = gap(J_v5,  J_ref)
    g_acc = gap(J_v5,  J_v4)      # acceleration-only gap v5 vs v4

    def stat(x):
        return (x.mean(), x.std(), np.sqrt(np.mean(x**2)),
                np.mean(np.abs(x)), np.max(np.abs(x)))

    print("\n================ OPTIMALITY GAP SUMMARY ================")
    print(f"  mean J  v4 (baseline)      : {J_v4.mean():.4f}")
    print(f"  mean J  v5 (accelerated)   : {J_v5.mean():.4f}")
    print(f"  mean J  reference (30 rnd) : {J_ref.mean():.4f}")
    print(f"  mean sum-p  (all)          : {P_v4.mean():.4f} W  (toy channel: pinned)")
    m, s, rms, mae, mx = stat(g_acc)
    print(f"  accel. drift  v5 vs v4    : mean {m:+.3f}%  RMS {rms:.3f}%  "
          f"MAE {mae:.3f}%  max|*| {mx:.3f}%")
    m, s, rms, mae, mx = stat(g_v4)
    print(f"  gap v4  vs 30-rnd ref     : mean {m:+.3f}%  RMS {rms:.3f}%  "
          f"MAE {mae:.3f}%  max|*| {mx:.3f}%")
    m, s, rms, mae, mx = stat(g_v5)
    print(f"  gap v5  vs 30-rnd ref     : mean {m:+.3f}%  RMS {rms:.3f}%  "
          f"MAE {mae:.3f}%  max|*| {mx:.3f}%")
    print("=======================================================")

    # ---- save CSV ----
    out = np.column_stack([np.arange(N_STEPS), J_v4, J_v5, J_ref,
                           P_v4, P_v5, P_ref, g_v4, g_v5, g_acc])
    hdr = ("step,J_v4,J_v5,J_ref,P_v4,P_v5,P_ref,"
           "gap_v4_pct,gap_v5_pct,acc_gap_v5_v4_pct")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_gap.csv")
    np.savetxt(csv_path, out, delimiter=",", header=hdr, fmt="%.5f", comments="")
    print(f"[saved] {csv_path}")

    # ---- plots ----
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figs")
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(J_ref, "k-",  lw=2,   label="reference (SCA optimum)")
    ax.plot(J_v4,  "C0--", lw=1.4, label="v4 baseline (5 rnd)")
    ax.plot(J_v5,  "C3:",  lw=1.4, label="v5 accelerated (3 rnd)")
    ax.set_xlabel("MPC step"); ax.set_ylabel("objective $J$")
    ax.set_title("(a) Objective value vs step")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(g_v4, "C0--", lw=1.4, label="v4 gap vs optimum")
    ax.plot(g_v5, "C3:",  lw=1.4, label="v5 gap vs optimum")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("MPC step"); ax.set_ylabel("optimality gap [\\%]")
    ax.set_title("(b) Optimality gap (vs SCA optimum)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    plt.tight_layout()
    fig_path = os.path.join(plot_dir, "opt_gap.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[saved] {fig_path}")

    # ---- bar: mean objective + mean gap ----
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
    methods = ["v4 baseline", "v5 accelerated", "SCA optimum"]
    means = [J_v4.mean(), J_v5.mean(), J_ref.mean()]
    bars = ax2.bar(methods, means, color=["C0", "C3", "k"])
    for b, m in zip(bars, means):
        ax2.text(b.get_x()+b.get_width()/2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("mean objective $J$")
    ax2.set_title("Mean objective: accelerated vs optimum")
    ax2.grid(alpha=.3, axis="y")
    plt.tight_layout()
    bar_path = os.path.join(plot_dir, "opt_gap_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[saved] {bar_path}")

    return dict(J_v4=J_v4, J_v5=J_v5, J_ref=J_ref, P_v4=P_v4, P_v5=P_v5, P_ref=P_ref,
               g_v4=g_v4, g_v5=g_v5, g_acc=g_acc)


if __name__ == "__main__":
    main()
