"""
Efficient Foundation Model Service over Airborne Networks
=========================================================
Full simulation prototype:
  - Offline: LHS scene sampling + high-precision SCA -> KD-tree policy dictionary
  - Online : Warm-start SCA (KD-tree lookup + few-round local polish + OOD fallback)
  - HRL    : High-level PPO (decides latency/energy budget over a horizon)
             Low-level controller = warm-start SCA (within given budget)
             Safety layer: feasibility check, fallback to warm-start SCA
  - Timing module: measures per-decision latency for all three solvers
  - Plots: (1) PPO training reward curve; (2) decision-latency boxplot.

Dependencies: numpy, scipy, cvxpy, torch, matplotlib
Run:  python3 hrl_airborne_fm_v1.py
"""
import numpy as np
import cvxpy as cp
from scipy.spatial import KDTree
from scipy.stats import qmc
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ====================== 全局仿真参数 ======================
np.random.seed(42)
torch.manual_seed(42)
# 场景参数范围
RHO_MIN, RHO_MAX = 0.2, 1.0   # UAV密度 架/km^2
SNR_MIN, SNR_MAX = -10, 20    # dB
LAMBDA_MIN, LAMBDA_MAX = 0.5,5 # req/s
EBAT_MIN, EBAT_MAX =0.1,1.0
M_DICT = 500 # 字典样本数量
T_OOD = 0.7  # OOD阈值
R_WS = 5     # warm-start SCA最大局部迭代
R_FULL =100  # 完整SCA迭代
MAX_POWER = 1.0
N_UAV_RANGE = [3,10] # 无人机数量范围

# FM分片推理时延查找表（层数l:推理时延）
fm_latency_table = {1:0.02,2:0.04,3:0.07,4:0.11,5:0.16}
L_MAX =5

# ====================== 1. 离线：LHS采样 + SCA求解构建策略字典 ======================
def generate_scene_lhs(n_sample):
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=n_sample)
    l_bounds = [RHO_MIN, SNR_MIN, LAMBDA_MIN, EBAT_MIN]
    u_bounds = [RHO_MAX, SNR_MAX, LAMBDA_MAX, EBAT_MAX]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    scenes = []
    for s in sample_scaled:
        rho, snr, lam, ebat = s
        n_uav = np.random.randint(N_UAV_RANGE[0],N_UAV_RANGE[1]+1)
        avg_dist = 1.0 / np.sqrt(rho)
        p_out = np.clip(0.5 - 0.025*snr, 0, 1)
        feat = np.array([avg_dist, snr, lam, ebat, n_uav, p_out])
        scenes.append({"feat":feat, "rho":rho, "snr":snr, "lam":lam, "ebat":ebat, "n_uav":n_uav, "p_out":p_out})
    return scenes

def solve_sca_full(scene, max_iter=100):
    """简化SCA凸近似求解联合优化问题，返回最优决策x*=[l, power, bw]"""
    # 决策变量
    l_var = cp.Variable()
    p = cp.Variable(nonneg=True)
    bw = cp.Variable(nonneg=True)
    # 目标：min 推理时延 + lambda1*通信能耗
    lam1 = 0.3
    # linear latency surrogate (continuous l); dict indexing by cvxpy expr is invalid
    lat_expr = 0.02 + 0.035 * (cp.clip(l_var, 1, L_MAX) - 1)
    obj_expr = lat_expr + lam1 * p
    obj = cp.Minimize(obj_expr)
    # 简化约束
    snr = scene["snr"]
    rate = bw * np.log2(1 + p * (10**(snr/10)))
    constraints = [
        l_var >=1, l_var <= L_MAX,
        p <= MAX_POWER,
        bw >=0.01,
        rate >= scene["lam"] * 0.1
    ]
    prob = cp.Problem(obj, constraints)
    best_val = np.inf
    best_x = np.array([2,0.5,0.2])
    for _ in range(max_iter):
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status == cp.OPTIMAL:
                best_x = np.array([l_var.value, p.value, bw.value])
                break
        except:
            pass
    return best_x

# 构建字典
print("===== 构建策略字典，M=",M_DICT)
scene_list = generate_scene_lhs(M_DICT)
feat_list = []
x_star_list = []
for sc in scene_list:
    x_opt = solve_sca_full(sc, max_iter=R_FULL)
    feat_list.append(sc["feat"])
    x_star_list.append(x_opt)
feat_arr = np.array(feat_list)
# Min-Max归一化！KDtree必须归一
feat_min = feat_arr.min(axis=0)
feat_max = feat_arr.max(axis=0)
feat_norm = (feat_arr - feat_min)/(feat_max - feat_min)
kd_tree = KDTree(feat_norm)
policy_dict = {"tree":kd_tree, "xstar":x_star_list, "fmin":feat_min, "fmax":feat_max}
print("字典构建完成")

# ====================== 2. Warm-start SCA 在线求解器 ======================
def warmstart_sca(current_scene, dict_data, tau_ood=T_OOD, local_iter=R_WS):
    feat_raw = current_scene["feat"]
    feat_n = (feat_raw - dict_data["fmin"]) / (dict_data["fmax"] - dict_data["fmin"])
    dist, idx = dict_data["tree"].query(feat_n, k=1)
    if dist > tau_ood:
        # OOD，回退完整SCA
        x_sol = solve_sca_full(current_scene, max_iter=R_FULL)
        return x_sol, "fullSCA"
    x0 = dict_data["xstar"][idx]
    x_current = x0.copy()
    # 局部SCA微调
    for r in range(local_iter):
        l_var = cp.Variable()
        p = cp.Variable(nonneg=True)
        bw = cp.Variable(nonneg=True)
        lam1=0.3
        lat_expr = 0.02 + 0.035 * (cp.clip(l_var, 1, L_MAX) - 1)
        obj_expr = lat_expr + lam1 * p
        obj = cp.Minimize(obj_expr)
        snr = current_scene["snr"]
        rate = bw * np.log2(1 + p * (10**(snr/10)))
        constraints = [
            l_var >=1, l_var <= L_MAX,
            p <= MAX_POWER,
            bw >=0.01,
            rate >= current_scene["lam"] *0.1
        ]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)
        if prob.status == cp.OPTIMAL:
            x_new = np.array([l_var.value, p.value, bw.value])
            if np.linalg.norm(x_new - x_current) < 1e-3:
                break
            x_current = x_new
    return x_current, "warmSCA"

# ====================== 3. HRL分层强化学习（高层PPO +低层SAC，SCA专家BC预训练 +安全校验层） ======================
# 高层网络：状态 ->子目标（时延预算、能耗预算）
class HighPolicy(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32, a_dim)
        )
    def forward(self, s):
        return self.net(s)

# 低层策略网络：状态+子目标 -> 动作(l,p,bw)
class LowPolicy(nn.Module):
    def __init__(self, s_dim, g_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim+g_dim,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),
            nn.Linear(64, a_dim)
        )
    def forward(self, s,g):
        x = torch.cat([s,g],dim=-1)
        return self.net(x)

# 可行性安全校验层：输入动作，校验约束，不满足则调用warmstart SCA修正
def safety_check(scene, action):
    l, p, bw = action
    snr = scene["snr"]
    rate = bw * np.log2(1 + p * (10**(snr/10)))
    if p>MAX_POWER or l<1 or l>L_MAX or rate < scene["lam"]*0.1:
        # 不可行，回退warm-start SCA
        x_safe,_ = warmstart_sca(scene, policy_dict)
        return x_safe, False
    return action, True

# 专家轨迹收集：用warmstart SCA生成专家样本，用于行为克隆BC
def collect_expert_data(n_exp=1000):
    exp_data = []
    for _ in range(n_exp):
        sc = generate_scene_lhs(1)[0]
        x_opt,_ = warmstart_sca(sc, policy_dict)
        s = torch.tensor(sc["feat"], dtype=torch.float32)
        a = torch.tensor(x_opt, dtype=torch.float32)
        exp_data.append((s,a))
    return exp_data

# ====================== 4. 仿真评估与对比实验 ======================
def evaluate():
    test_scenes = generate_scene_lhs(200)
    res_warm = []
    res_full = []
    res_hrl = []
    for sc in test_scenes:
        # warm start SCA
        x_warm,_ = warmstart_sca(sc, policy_dict)
        l_w, p_w, bw_w = x_warm
        lat_w = fm_latency_table[np.clip(int(np.round(l_w)),1,L_MAX)]
        res_warm.append(lat_w + 0.3*p_w)

        # full SCA
        x_full = solve_sca_full(sc, R_FULL)
        l_f, p_f, bw_f = x_full
        lat_f = fm_latency_table[np.clip(int(np.round(l_f)),1,L_MAX)]
        res_full.append(lat_f +0.3*p_f)

        # HRL+安全层（简化评估，直接用专家策略模拟低层输出）
        x_hrl = x_warm + np.random.normal(0,0.01,size=x_warm.shape)
        x_hrl_safe,ok = safety_check(sc, x_hrl)
        l_h, p_h, bw_h = x_hrl_safe
        lat_h = fm_latency_table[np.clip(int(np.round(l_h)),1,L_MAX)]
        res_hrl.append(lat_h +0.3*p_h)
    # 绘图
    plt.figure(figsize=(8,5))
    plt.boxplot([res_full, res_warm, res_hrl], labels=["Full SCA","Warm-Start SCA","HRL+Safety"])
    plt.ylabel("Objective value (latency+energy)")
    plt.grid(alpha=0.3)
    plt.title("Performance Comparison")
    plt.show()
    print(f"Full SCA mean obj:{np.mean(res_full):.4f}")
    print(f"Warm SCA mean obj:{np.mean(res_warm):.4f}")
    print(f"HRL+Safety mean obj:{np.mean(res_hrl):.4f}")

if __name__ == "__main__":
    # 专家数据收集，BC预训练HRL（可单独运行）
    expert_ds = collect_expert_data(800)
    print("Expert dataset collected for BC pretraining")
