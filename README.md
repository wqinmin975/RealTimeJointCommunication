> ⚠️ **STATUS — 本文档为 HISTORICAL 原型文档，请勿作为本稿 pipeline 依据。**
> 下文把 `hrl_airborne_fm_v1`–`v5`（尤其 v5「主版本」）、`sim_analysis.py`、`opt_gap.py`
> 描述成论文主方法，这是**过时的玩具原型口径**。当前投稿（PeerJ Computer Science）的
> **权威复现入口是 `repro/README.md`**：`sim_analysis_3gpp.py` 才是生成全部图表与表格的
> 权威脚本（3GPP TR 38.901 信道 + 实测 UWB 运动轮廓）；`sim_analysis.py` 与 v1–v5 玩具版本
> 一律为 **HISTORICAL 原型，不是论文 pipeline**。`validate_aerpaw.py --demo` 是 SYNTHETIC
> 演示，其数值不得与真实 AERPAW 实测（Dryad `10.5061/dryad.2ngf1vj4r`）混淆。
> 详见 `repro/README.md`、`repro/provenance.md`。

---

# Airborne FM Service — 机载网络高效大模型推理服务仿真

> 把论文 *Efficient Foundation Model Service over Airborne Networks* 的方法论整理成可运行的程序。
> 离线用 **LHS 场景采样 + 高精度 SCA** 构建策略字典（warm-start），在线用 **MPC / 块坐标下降 SCA** 做实时联合资源调度（功率 p、带宽 b、UAV 3D 位置 u、切分层 l、量化比特 q、早退标志 e）。

本目录由设计文档 `UVA通信.docx` 整理而来，对应文档末尾的项目结构：五版演进脚本 + 本说明。
**v5 为主版本（论文 Method 主方法）**，v1–v4 退为 ablation / 对照基线。

---

## 1. 快速开始

### 依赖安装

```bash
pip install numpy scipy cvxpy matplotlib
# v1（含分层 PPO 骨架）额外需要：
pip install torch gym
```

> 求解器统一使用 **CLARABEL**（cvxpy ≥ 1.5 默认自带）。文档早期草稿用过 ECOS / 整数变量，已在 v1 中修正为连续松弛 + CLARABEL。

### 运行

```bash
python3 hrl_airborne_fm_v1.py   # 离线字典 + warm-start SCA + HRL/PPO 骨架（+ torch/gym）
python3 hrl_airborne_fm_v2.py   # 多变量 SCA + MPC + early-exit + 动态量化 → mpc_evolution.png
python3 hrl_airborne_fm_v3.py   # 块坐标下降 (p,b,u) + UAV 位置 1/d^2 线性化 → v3_evolution.png
python3 hrl_airborne_fm_v4.py   # 需求自适应 l/q/e QP + 速度约束 → v4_evolution.png
python3 hrl_airborne_fm_v5.py   # 【主版本】warm-start 加速：预构建问题 + CLARABEL warm_start → v5_warmstart.png
```

v2/v3/v4/v5 为纯 `cvxpy` 实现，开箱即跑；v1 依赖 `torch`/`gym`（仅 HRL 部分），离线字典与 warm-start SCA 核心不依赖它们。

---

## 2. 五版演进对照表

| 版本 | 角色 | 新增内容 | 关键代码位置 | 输出图 |
|------|------|----------|--------------|--------|
| **v1** | Ablation | LHS 场景采样 + 高精度 SCA 离线求解 → KD-tree 策略字典；warm-start SCA 在线求解（含 OOD 回退）；HRL 双层 PPO + 安全校验层骨架 | `generate_scene_lhs` / `solve_sca_full` / `build_policy_dict`（LHS+SCA+KD-tree）；`warmstart_sca`（KD-tree 检索 + 局部 SCA + OOD 兜底）；`AirborneFMEnv` + `Actor`/`Critic`/`PPO`（Gaussian 策略、clip、GAE）；`safety_check`（不可行回退 warm-start） | `ppo_reward.png`、`decision_latency.png`（HRL 版） |
| **v2** | Ablation | 多变量 SCA（联合 u/p/b/l/q/e）；MPC 滚动时域（`LinearPredictor` + `MPCController`）；early-exit + 动态量化进目标函数 | `solve_sca_multi`（固定 b 均分，避免 b·log 双线性）；`MPCController.step`（只采纳第一步） | `mpc_evolution.png`（4 子图） |
| **v3** | Ablation | 块坐标下降 Block A/B/C（b 变优化变量）；UAV 位置 `1/d²` 切平面线性化（凸函数的下界） | `solve_sca_block_cd`：Block A(`p` via CLARABEL) / Block B(解析 `b*`) / Block C(`u` via CLARABEL) | `v3_evolution.png`（6 子图） |
| **v4** | Ablation | l/q/e 写进 CVXPY QP（带二次正则，需求自适应权重 `W_acc(d)`）；UAV 速度约束 `‖u_t−u_{t-1}‖≤v_max·Δt` | `solve_sca_block_cd` 中 (a) 段 l/q/e 改为 QP；Block C 约束加 `cp.norm(u_var[k]-uavs_prev[k])<=V_STEP` | `v4_evolution.png`（demand 爬坡下 l/q/e 动态适配） |
| **v5** | **主版本** | warm-start 加速版：①问题在 `__init__` 预构建一次，所有变化量（`g_sel,d_sel,W_acc,uavs_prev`）包成 `cp.Parameter`，每步只更新 `.value`；②CLARABEL `warm_start=True` 跨 SCA 轮与 MPC 步；③l/q/e 解析闭式解（3 变量 QP 不需 CVXPY）；④`SCA_ROUNDS 5→3` | `FastSCASolver`（预构建 `prob_p`/`prob_u` + warm-start）；`solve_lqe_analytic`（l/q/e 闭式）；`MPCController.step`（`Timer` 计时） | `v5_warmstart.png`（决策时延 vs demand 爬坡） |

### 2.1 主版本 vs Ablation 角色结构

| 角色 | 文件 | 用途 |
|------|------|------|
| **主版本** | `hrl_airborne_fm_v5.py` + `v5_warmstart.png` | 论文 Method 主方法，决策时延 ~22 ms |
| Ablation | v1 warm-start 字典 + PPO | 消融「无 warm-start」 |
| Ablation | v2 MPC + early exit + 量化 | 消融「无 MPC / 无 early exit」 |
| Ablation | v3 块坐标下降 + 1/d² 线性化 | 消融「固定带宽 / UAV 不进 SCA」 |
| Ablation | v4 l/q/e QP + 速度约束 | 消融「无需求自适应 / 无速度约束」 |

### 2.2 v5 vs v4 决策时延对比

> ⚠️ **HISTORICAL 口径，勿与改投稿数值并列**：下表 120 ms / 22 ms / 13 ms /
> ~5.5× / ~9× / 100% 都是 v1–v5 **玩具原型**期的「文档目标 vs 早期本机实测」，
> **不是** PeerJ CS 改投稿的数据。改投稿统一以 `ablation.csv` 与
> `paper/manuscript_peerj.tex:834–835` 为准：**v5 13.3 ms、v4 227 ms、17×、50 ms 预算**。
> 权威映射见 `repro/README.md §2` 与 `repro/provenance.md`。

| 指标 | v4（文档目标） | v5（文档目标） | v5（本机实测） |
|------|----------------|----------------|----------------|
| 平均决策时延 | ~120 ms | ~22 ms | **13 ms** |
| p95 / 最大 | — | — | 16 / 18 ms |
| <50 ms 达标率 | 部分 | 100% | **100%** |
| 相对 v4 加速 | 1× | ~5.5× | **~9×** |

> 文档笔记目标为 120 ms → 22 ms（约 5.5×）；本机实测 v5 均值 13 ms、p95 16 ms、100% 步数 <50 ms（不同机器/测量口径会有差异，22 ms 与 13 ms 同量级，证明 warm-start 加速真实成立）。
>
> **关键修复（影响结论可信度）**：原稿 `prob_u` 约束 `g_sel + grad@(u-u0) >= c_min` 中的 `grad@u0` 是 *参数 × 参数*，破坏 DPP（Disciplined Parametrized Programming），CVXPY 每步仍重新 canonicalize，warm-start 加速**未生效**——实测决策时延约 58 ms、仅 20% 步数 <50 ms。已将 `grad@u0` 折叠进右侧常数参数 `u_rhs`，约束改为 `g_sel + grad@u >= c_min + grad@u0`（纯 *参数 × 变量*，DPP-compliant），加速才真正落地。修复后 `prob_p`/`prob_u` 均 `is_dpp()==True`，无 DPP 警告。

### 2.3 Ablation 矩阵（每版回答「去掉 X 会怎样」）

| Ablation 版本 | 被去掉的模块 | 预期退化现象 |
|---------------|--------------|--------------|
| v1 | warm-start 字典 + KD-tree 检索 | 在线每步重跑完整 SCA，时延飙升；OOD 无兜底易不可行 |
| v2 | MPC 滚动时域 / early-exit / 动态量化 | 无前瞻 → demand 突变超调；精度/时延不可联合调 |
| v3 | 块坐标下降 + 1/d² 线性化 | 带宽固定均分、UAV 不进 SCA → 速率受限、覆盖差 |
| v4 | 需求自适应 l/q/e QP / 速度约束 | 固定 l/q/e 无法随负载适配；UAV 瞬移违反物理 |

---

## 3. 各版本已知 toy 局限（写论文前要替换）

- **v1**：SCA 用简化查表 + 线性代理，仅为原型；HRL 仅骨架（BC 预训练数据由 warm-start 生成，PPO 训练曲线需要真实 reward 才能出图）。
- **v2**：目标函数过线性 → 最优解卡中间点（l≈3, q≈8, e≈0），换真实 FM 时延/精度表才会随 demand 变化；带宽固定均分（未联合分配）；预测器是线性外推（应换 LSTM/Transformer）；UAV 移动为贪心 30%（未写进 SCA 内迭代）。
- **v3**：l/q/e 用解析选界 → 总取极端值（l=1,q=16,e=1），应写进 CVXPY 联合 LP 并调 `W_acc`；Block C 只做 feasibility（应加 `min ‖u−u_desired‖²` 或 max-min 速率目标）；1:1 接入固定（应在每轮外迭代重跑 `assignment=argmax G`）；UAV 无速度约束。
- **v4**：决策时延 ~110–155 ms，超 100 ms 预算（每步 5 轮外迭代 × 3 次 CLARABEL）；修复——`SCA_ROUNDS 5→2~3`、CLARABEL `warm_start=True`、`lqe` QP 可解析求解；`v_max=10` m/s 过严（真实无人机 30–50 m/s），导致 UAV 几乎不动；Block C 无目标函数（UAV 停原地），论文应加目标。
- **v5**：warm-start 加速已落地（DPP 修复后实测 13 ms / 100% <50 ms）；但仍是 toy 原型——`FM_LAT`/`FM_ACC` 为线性代理、预测器是 `LinearPredictor`（应换 LSTM）、信道为静态自由空间路径损耗、SCA 轮数/速度上限等超参未做消融扫描。上述 v1–v4 的 toy 局限 v5 同样存在。

---

## 4. 论文章节 ↔ 代码函数映射

| 论文章节 / 创新点 | 代码模块 / 函数 |
|-------------------|-----------------|
| A.2 离线阶段：构建策略字典 | `generate_scene_lhs` + `solve_sca_full` / `build_policy_dict`（LHS + SCA + KD-tree） |
| Algorithm 1 在线：warm-start SCA + OOD | `warmstart_sca`（KD-tree 检索 + 5 轮局部 SCA + OOD 兜底） |
| 创新 2：分层 PPO | `AirborneFMEnv` + `Actor`/`Critic`/`PPO`（Gaussian 策略、clip ratio、GAE 折扣回报） |
| 创新 3：可行性校验安全层 | `safety_check`（benchmark 中不可行时回退 warm-start） |
| 在线决策时延统计 | `Timer` 上下文管理器 / `time.perf_counter()` 对三个求解器分别计时 |
| **Section IV-E 加速模块（v5 主方法）** | `FastSCASolver.__init__`（问题预构建一次 + `cp.Parameter` 包装变化量）；`cp.Problem(...).solve(warm_start=True)` 跨 SCA 轮与 MPC 步复用；`solve_lqe_analytic`（l/q/e 闭式，免 CVXPY）；`SCA_ROUNDS=3` |
| 仿真图（Performance / Latency / Evolution） | `evaluate()` + `plot_reward` / `plot_latency`（v1）；`evaluate()` 内 `plt.subplots`（v2/v3/v4）；`v5_warmstart.png`（v5，`Timer` 计时决策时延 vs demand） |

---

## 5. 下一步建议（论文实验前）

1. **warm-start CLARABEL**：用上一步 `(p,b,u)` 作 `warm_start=True` 初值，降时延。
2. **真实 FM 表**：用真实分片推理时延 / 精度损失曲线替换 `FM_LAT`/`FM_ACC` 查表与线性代理。
3. **LSTM / Transformer 预测器**：替换 `LinearPredictor`，否则审稿人易质疑。
4. **基线对比**：Full SCA vs Warm-Start SCA vs HRL+Safety vs MPC 系列，量化时延/精度/能耗/收敛步数。
5. **消融实验**：字典大小 `M=200/500/1000`、OOD 阈值 `τ_ood`、SCA 轮数 `R`、速度上限 `v_max`、预测步长 `H`；按 §2.3 的 Ablation 矩阵逐版做「去掉 X」对照，量化时延/精度/能耗/收敛步数。
6. **真实 FANET 信道**：多普勒 / 时变衰落替换静态 SNR 自由空间路径损耗。
7. **v5 加速消融参数矩阵**：扫描 `SCA_ROUNDS∈{1,2,3,5}`、`warm_start∈{True,False}`、`DPP 合规` vs 原稿 `grad@u0` 非 DPP 版，定位时延主要来自 canonicalization 还是求解；补 `v_max`、预测器类型（Linear/LSTM）对时延-精度的 Pareto。

---

## 目录结构

```
airborne_fm_sim/
├── README.md                      # 本说明
├── hrl_airborne_fm_v1.py          # warm-start SCA 字典 + PPO 骨架（需 torch/gym）
├── hrl_airborne_fm_v2.py          # + MPC + early exit + 量化
├── hrl_airborne_fm_v3.py          # + 块坐标下降 (p,b,u)
├── hrl_airborne_fm_v4.py          # + 需求自适应 l/q/e QP + 速度约束
├── hrl_airborne_fm_v5.py          # 【主版本】warm-start 加速：预构建问题 + CLARABEL warm_start
├── mpc_evolution.png              # v2 输出
├── v3_evolution.png               # v3 输出
├── v4_evolution.png               # v4 输出
└── v5_warmstart.png               # v5 输出（决策时延 vs demand 爬坡）
```
