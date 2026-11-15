# 复现包说明（Reproduction Package）

本目录是 *Efficient Foundation Model Service over Airborne Networks* 投 **PeerJ
Computer Science** 的**权威复现包**。PeerJ CS 要求投稿时即可获得复现全部结果所需的代码与
原始数据；补充文件限制为**总计 ≤ 50 MB、单文件 ≤ 30 MB、且必须是机器可读格式（CSV/JSON）**，
PDF / PPT / 图片 / 字体 / `.tex` / `.pyc` 一律不计为原始数据。

> **本文件是唯一的权威入口。** 仓库根目录的 `README.md` 是历史原型文档（把 v1–v5 玩具版本描述成
> 论文主方法），**已过时，不代表本稿的 pipeline**；详见下文 §5。

---

## 0. 一句话口径（务必先读）

- **`sim_analysis_3gpp.py` 是生成当前稿件全部图表与表格的权威脚本**（3GPP TR 38.901 信道 +
  实测 UWB 运动轮廓）。
- **`sim_analysis.py` 与 `hrl_airborne_fm_v1.py` … `v5.py`（以及 `opt_gap.py` / `opt_gap.csv`）
  全部是 HISTORICAL 玩具原型**，只在理想化自由空间信道 `G0/(d²+ε)` 上跑过；**不是论文的
  pipeline，不得作为论文结果引用**。
- **`validate_aerpaw.py --demo` 是 SYNTHETIC 自洽演示**，其数值（例如 n=2000、RMSE 5.51 dB）
  **绝不可与真实实测校验**（Dryad `10.5061/dryad.2ngf1vj4r`，136,987 个实测路径损耗样本，
  bias −0.02 dB，RMSE 5.4 dB）混淆。脚本已做输出文件名隔离与 JSON 内嵌标记，见 §3、§6。
- 本机（2026-11-12）3 个**基础** Python 均**未安装 cvxpy / clarabel**（→ 原始投稿图表那次
  运行在本机不能直接复跑）；但 exp-worker 另建隔离 venv `.venv_exp` 成功安装 cvxpy 并真实
  生成了 multi-start 复核产物（**另一次重跑、非原始环境**）。精确版本 pin 须由作者在
  **原始可运行环境** `pip freeze` 后回填，严禁用本机 / 重跑环境版本，见 §7 / §2.4。

---

## 1. 权威脚本清单（脚本 → 运行命令 → 产出图表/表格）

所有命令均在仓库根目录 `airborne_fm_sim/` 下执行。带 `paper/figs/` 的路径是稿件引用的图。

| # | 脚本 | 角色 | 精确运行命令 | 产出（图 / 表 / 数据） |
|---|------|------|--------------|------------------------|
| A1 | `sim_analysis_3gpp.py` | **权威主生成器**：3GPP TR 38.901 UMa/ULA-Los 信道 + 实测 UWB 速度轮廓上的 SCA–BCD / MPC 消融 | `python sim_analysis_3gpp.py` | `paper/figs/convergence.png`(fig:conv)、`paper/figs/ablation.png`(fig:ablation)、`paper/figs/mpc_evolution.png`(fig:v2)、`v3_evolution.png`(fig:v3)、`v4_evolution.png`(fig:v4)、`v5_warmstart.png`(fig:v5)；`ablation.csv`(tab:abl / tab:lat / tab:v4ev)；tab:params 的参数常量见 §2 |
| A1b | `sim_analysis_3gpp.py`（多起点扩展） | 5-seed multi-start 收敛统计（由 exp-worker 追加，替换原本单轨迹表述） | `python sim_analysis_3gpp.py --multistart --seeds 0 1 2 3 4 --rounds 40` | `paper/data/convergence_multistart_traces.csv`、`paper/data/convergence_multistart_summary.json`（以及文档 `…_summary.md`） |
| A2 | `channel_3gpp.py` | 3GPP TR 38.901 路径损耗 / LOS 概率 / 信道增益**库模块**（`path_loss_db`、`los_probability`、`calibrate` 等）；无独立 CLI，被 A1/A3/A4/A5 及图脚本 import | 无独立命令；通过下游脚本调用，或 `python -c "import channel_3gpp as c; print(c.calibrate())"` | 无独立文件；其标定结果落在 `channel_calib.json` |
| A3 | `opt_gap_3gpp.py` | 论文自有 3GPP 信道上的两层最优性间隙分解（替代 `opt_gap.py`） | `python opt_gap_3gpp.py` | `opt_gap_3gpp.csv`、`opt_gap_3gpp.json`；`paper/figs/opt_gap.png`(fig:gap)、`opt_gap_bar.png`(tab:gap) |
| A4 | `validate_aerpaw.py` | **真实实测** AERPAW 大尺度路径损耗校验 | `python validate_aerpaw.py --csv-dir <AERPAW解包目录>` （显式写真实产物：`python validate_aerpaw.py --csv-dir <dir> --out-json aerpaw_validation.json --out-png aerpaw_validation.png`） | `aerpaw_validation.json`、`aerpaw_validation.png`；支撑摘要 / fig:aerpaw 的 136,987 / −0.02 dB / 5.4 dB |
| A4d | `validate_aerpaw.py`（demo） | **SYNTHETIC 合成自洽演示，非实测** | `python validate_aerpaw.py --demo` | `aerpaw_validation_demo.json`、`aerpaw_validation_demo.png`（**不写** `aerpaw_validation.*`） |
| A5 | `aerpaw_validation_detail.py` | 按飞行 / 距离分箱 / 单斜率拟合的实测细分 | `python aerpaw_validation_detail.py --csv-dir <AERPAW解包目录>` | `aerpaw_validation_detail.json`、`aerpaw_validation_detail.png` |

### 1.1 产出稿件其余图表的辅助脚本（同样权威，但需外部原始数据）

| 脚本 | 运行命令 | 产出 | 外部输入 |
|------|----------|------|----------|
| `make_channel_fig.py` | `python make_channel_fig.py` | `channel_calib.json` + `paper/figs/channel_calib.png`(fig:chcalib) | 无（内部计算）；但 import `sim_analysis_3gpp` → 需 CVXPY |
| `make_fading_fig.py` | `python make_fading_fig.py` | `fading_calib.json` + `paper/figs/fading.png`(fig:fading) | 无；需 CVXPY |
| `make_aerpaw_paper_fig.py` | `python make_aerpaw_paper_fig.py --csv-dir <AERPAW解包目录>` | `paper/figs/aerpaw_measured.png`(fig:aerpaw) | AERPAW `f10.csv..f13.csv`（17 GB Dryad 包） |
| `channel_38901_fading.py` | 库模块（被 A1 import） | 无 | — |
| `calibrate_official_fading.py` | `python calibrate_official_fading.py` | `calib_official.json` + `calib_official.png` | 外部 `D:/matlab_work/cdl_data.json`（MATLAB 5G Toolbox CDL 导出），**未随包提供 / UNVERIFIED** |
| `extract_measured_tracks.py` | `python extract_measured_tracks.py` | `measured_tracks.json` | 外部 `E:\data` UWB `.pkl`，**未随包提供 / UNVERIFIED**；其产物 `measured_tracks.json` 已随 `supplement/` 提供 |

> 说明：A2 与 `channel_38901_fading.py` 是库模块而非独立生成器；`calibrate_official_fading.py`
> 与 `extract_measured_tracks.py` 的**派生结果**（`calib_official.json` / `measured_tracks.json`）
> 已作为机器可读输入随补充包提供，重算稿件数字不需要那些外部原始源。

---

## 2. 稿件关键数值 → 来源（并附四项强制 caveat）

> **行号说明**：下表 `:NNN` 沿用 repro-audit 的 2026-10-07 快照口径。exp-worker 于
> 2026-11-12 重构 `sim_analysis_3gpp.py`（959 → 1423 行），行号整体后移；**请以函数名
> 为准定位**，新旧行号对照见 `repro/provenance.md` 的「行号基准 / 漂移说明」。
> `opt_gap_3gpp.py` 未改动，其行号仍精确有效。

| 稿件数值 | 出现位置 | 权威来源（脚本 + 函数/行 + 数据文件） |
|----------|----------|--------------------------------------|
| **13.3 ms**（v5 平均决策时延，p95 14.4 ms，<50 ms 100%） | 摘要 L50；tab:abl L834；tab:lat L986；L598、L137 | `ablation.csv` 行 `v5` ← `sim_analysis_3gpp.py:main()` → `run_variant()`(:878) → CSV(:892) |
| **227 ms**（v4 平均，p95 311 ms，<50 ms 0%） | 摘要 L43；L97；tab:abl L835；tab:lat L986 | `ablation.csv` 行 `v4` ← 同上 |
| **17× / 17-fold**（227 / 13.3 ≈ 17.1） | 摘要 L51；L851；L996；L1250 | `ablation.csv` 列 `rel_lat_x`（v4=17.1） ← `sim_analysis_3gpp.py:884` |
| **50 ms**（硬实时预算 / 达标线） | 摘要 L43、L50；tab:abl 说明 | **设计约束，非测量值**；代码字面量 `50.0` 见 `sim_analysis_3gpp.py:620`（`frac50`）与 `:916`（`ax.axhline(50)`） |
| **0.06%**（凸化间隙，v4 vs 30-round ref 均值；max 0.22%） | 摘要 L53；L358；L1037；tab:gap L1066；L1109 | `opt_gap_3gpp.json: surrogate_gap_v4_pct.mean = 0.0620603…` ← `opt_gap_3gpp.py:96,99`（`gap_pct(J["v4"],J["ref"])` / `stat`） |
| **21.7%**（v5 加速代价均值；以及 v5 vs ref 总间隙均值） | L143、L872、L1038；tab:gap L1067；L1106、L1114 | `opt_gap_3gpp.json: acceleration_cost_pct.mean = 21.6637…`（= `surrogate_gap_v5_pct.mean = 21.7322…`） ← `opt_gap_3gpp.py:95,97,99`；tab:gap 注 L1071 说明是 40 步逐漂移均值 |
| **136,987 样本 / bias −0.02 dB / RMSE 5.4 dB（精确 5.39 dB）** | 摘要 L40–41；L300；fig:aerpaw 标题 L747 / L753；L766、L727 | `aerpaw_validation.json`（`n_samples`、`bias_expected_db=-0.0238956`、`rmse_expected_db=5.391573`）与 `aerpaw_validation_detail.json:ALL`（n=136987, bias −0.0239, rmse 5.3916） ← `validate_aerpaw.py:validate()`(:99) / `aerpaw_validation_detail.py:main()` |
| 仿真参数表（tab:params，L647–676） | 正文 L678–703 | `sim_analysis_3gpp.py:44–101` 的常量（`AREA/ALT_RANGE/N_UAV=3/N_USER=5/F_CARRIER=ch3g.F_HZ=3.5GHz/B_TOTAL=10MHz/P_MIN=0.01/P_MAX=1.0/V_MAX=10/H_MPC=3/DT=0.1/W_LAT=1.0/W_EN=0.3/W_ACC_BASE=2.0/MU_REG=0.5/DEMAND_LO=30/DEMAND_HI=62/seeds 2024,991`）+ `channel_3gpp.py:47–53`（`F_HZ=3.5e9`、`G_TX_DBI=4`、`G_RX_DBI=2`、`B_REF_HZ=10e6`）+ `channel_38901_fading.py:65–72`（σ_SF=4 dB、corr=37 m、K=7 dB、12 clusters） |

### 2.1 ⚠ CAVEAT 1 — 两次独立计时运行，不得静默统一（需作者裁决）

- `opt_gap_3gpp.json` 的计时 **`mean_lat_ms`: v4 = 225.7240400009323 / v5 = 14.610509999329224 ms**
  来自 **另一个 `run_variant` 计时过程**：`opt_gap_3gpp.py:87`（`lat_ms[name]=m["mean_lat"]`，L89）。
- `ablation.csv` 的 **227 / 13.3 ms** 来自 **另一个 `run_variant` 计时过程**：
  `sim_analysis_3gpp.py:878`。
- 这**是两次彼此独立的运行**，不是同一测量的两个修订版。稿件引用的 227/13.3/17× 取自
  `ablation.csv`；`opt_gap_3gpp.json` 里的 225.724/14.6105 **没有被稿件当作计时引用**，且两者
  算出的加速比也不同（227/13.3 ≈ **17.1×** vs 225.724/14.6105 ≈ **15.45×**）。
- **要求：不得把这两组数字合并 / 互换 / 取平均。** 请作者明确决定稿件采用哪一组口径，并在
  Data Availability 或方法中说明 wall-clock 计时随机器/运行波动；目标函数 `J` 由固定种子决定、
  两次运行一致（v4≈0.610、v5≈0.732、ref≈0.610），而时延是 wall-clock、本就不可跨运行比较。

### 2.2 ⚠ CAVEAT 2 — `ablation.csv` 溯源为「高置信度但未证实」

`ablation.csv` 的 mtime 为 **2026-10-07 08:44:45**，比 `paper/figs/ablation.png`（08:43:08）
晚约 1.5 分钟，尽管写 CSV 的代码（`sim_analysis_3gpp.py:892`）在画图代码（`:938`）之前执行；
且 `sim_analysis.py`（历史原型）与 `sim_analysis_3gpp.py` 写出的 CSV 头**逐字节相同**。因此只能标注：
**「按内容高置信度为 3GPP 版本；生成过程 / 时间 UNVERIFIED」**，**不得断言已被确证**。
（对照：`paper/figs/convergence.png` 溯源是可靠的——08:42–08:43 批次中的演化图 mtime
08:43:09–08:43:14 为 `sim_analysis_3gpp.py:main()` 独有。）

### 2.3 ⚠ CAVEAT 3 — demo 数值不可冒充实测

见 §0 与 §6。真实实测数字只有 `aerpaw_validation.json` / `aerpaw_validation_detail.json` 两套；
`--demo` 产物以 `_demo` 后缀隔离且 JSON 内 `data_mode="synthetic-demo"`。

### 2.4 ⚠ CAVEAT 4 — multi-start 产物是「另一次重跑」，不是原始图表字节

`paper/data/convergence_multistart_*.{csv,json}` 由 exp-worker 于 **2026-11-12** 在独立虚拟环境
`.venv_exp` 中**真实生成并落盘、严格校验**（该环境精确版本记录在该 JSON 的
`environment` / `software` 块，**不是** 2026-10-07 生成投稿图表的那次运行，也不得当作
原始投稿环境的版本依据）。**这些产物是已验证的真实文件（非 UNVERIFIED / 待生成），
但仍属另一次重跑。**
`convergence.png` 原始那次运行的逐点字节从未落盘（UNVERIFIED）；**但单轨迹曲线的
机器可读底层数据现已可恢复**：=`convergence_multistart_traces.csv` 的 `seed=0` 全部
160 行 / `summary.json` 的 `per_seed_traces["0"]`，经未改动原文件独立复跑**逐位验证**
（`max|ΔJ|=0.0`）。**仓库内不存在单独的 `convergence.csv` / `solver_status.csv` /
`solver_events.csv`**，勿寻找、勿列为待生成。引用时表述为「后续多起点收敛复核」，
不得声称是 `convergence.png` 的原始底层字节；该脚本 `--multistart` 明确不改写
`paper/figs/`。详见 `provenance.md` §D（含函数名冻结清单）。

**与稿件口径的三处冲突（AUTHOR TODO / 由 lead 裁决，见 §10）**：

- 该 5-seed 复核披露 **600 次求解调用中 148 次（24.7%）失败**（每 seed 29–31/120），
  **100% 集中在 UAV 位置可行性块**（CLARABEL SolverError / infeasible /
  infeasible_inaccurate / user_limit），正是原代码 `except Exception: pass` 静默吞掉的；
  **失败时 UAV 位置不更新**；另有 **22 次 optimal_inaccurate 被原代码按成功处理**。
  逐种子计数见 `summary.json` 的 `per_seed[].solver_failures_total / solver_exceptions /
  solver_nonoptimal / solver_calls`。因此**本复现包不作「干净收敛 / 单调收敛」断言**。
- 稿件 `manuscript_peerj.tex` 的 **L623–626 / L1137–1138 / L1293–1294** 仍写
  「the curve is monotone …」「multi-start statistics remain future work」，与已完成的
  5-seed 复核相矛盾，须由 lead 更新（本包不改动稿件）。
- 稿件 **0.06% / 21.7% 间隙**来自 `opt_gap_3gpp.py` 的 `REF_ROUNDS=30`、**每个 variant
  单次运行、无重启**，属 **single-trajectory**，**不得描述为 multi-start 支持**；且其
  `run_variant` 路径同样存在静默回退、solver status 未单独落盘 → UNVERIFIED。

---

## 3. `--demo` 与真实实测的隔离（已实现并实测验证）

- 真实运行默认写 `aerpaw_validation.json` / `.png`；**`--demo` 默认写
  `aerpaw_validation_demo.json` / `.png`**。
- `--demo` 若被显式要求写到真实产物名（`--out-json aerpaw_validation.json` 或
  `--out-png aerpaw_validation.png`），**直接报错退出**，绝不覆盖。
- 每个 JSON 输出都带 `data_mode`（`"measured"` / `"synthetic-demo"`）与
  `is_real_measurement`（bool）；demo 另带 `notice` 明确「SYNTHETIC … NOT measured … do NOT cite」。
- 所有 JSON 用 `allow_nan=False` 严格序列化，非有限浮点映射为 JSON `null`。
- 本机实测：`python validate_aerpaw.py --demo` 只生成 `*_demo.*`，真实
  `aerpaw_validation.json` 的 SHA-256 前后不变；两个 guard 测试均按预期 exit=1。

---

## 4. 硬编码外部路径 → `--csv-dir`（已修复）

`aerpaw_validation_detail.py` 与 `make_aerpaw_paper_fig.py` 原先硬编码
`E:/AERPAW/ds32/data`，现改为 `--csv-dir` 参数，**旧值仅保留为 documented default**
（`DEFAULT_CSV_DIR`），并附 AUTHOR TODO。`validate_aerpaw.py` 本就有 `--csv-dir`。
`calibrate_official_fading.py`（`D:/matlab_work/cdl_data.json`）与
`extract_measured_tracks.py`（`E:\data`）依赖外部原始源，**未改**，但其派生产物已随
`supplement/` 提供（见 §1.1）；原始源标记 UNVERIFIED。

---

## 5. HISTORICAL 原型（不是论文 pipeline，勿引用）

| 文件 | 说明 |
|------|------|
| `sim_analysis.py` | 历史原型，理想化自由空间信道；**也写** `ablation.csv` 与 `paper/figs/{convergence,ablation}.png`，且 CSV 头与 3GPP 版逐字节相同 → 这正是 §2.2 溯源不确定性的来源 |
| `hrl_airborne_fm_v1.py` … `v5.py` | v1–v5 玩具演进原型（v5 = warm-start 玩具版）；**不是**论文 Method |
| `opt_gap.py` / `opt_gap.csv` | 历史原型间隙分析，跑在自由空间信道上；`opt_gap_3gpp.py` 文档明确「the two are not comparable」 |
| 根目录 `mpc_evolution.png` / `v3_evolution.png` / `v4_evolution.png` / `v5_warmstart.png` / `calib_official.png` / `v3_evolution.png` 等 | 原型/重复输出；稿件只用 `paper/figs/` |
| 根目录 `README.md` | 历史文档，把 v1–v5 描述成主方法，已过时 |

---

## 6. 原始数据与 AERPAW（作者必须自行完成）

- 真实 AERPAW 数据集：**Dryad `10.5061/dryad.2ngf1vj4r`**，
  *Air-to-Ground Channel Sounding and Path Loss Measurements at 3.4 GHz*。
  处理后 CSV `f10.csv..f13.csv` 封装在 **17 GB `data.tar.gz`** 内，单文件端点需会话 cookie，
  **超出补充文件 50 MB / 30 MB 限制**，因此**不随补充包提供**；获取/解包步骤见
  `aerpaw_data/README.md`。
- 补充包只提供**聚合结果**（`aerpaw_validation*.json`、`channel_calib.json`、
  `fading_calib.json`、`calib_official.json`、`opt_gap_3gpp.*`、`ablation.csv`、
  `measured_tracks.json`）与多起点收敛 traces。
- **UWB 障碍已降级**：`sim_analysis_3gpp.py:load_measured_speed_profile()`（`:194–231`）
  **只读仓库内 `measured_tracks.json`**（203 KB，随包提供），`:196–199` 有常数速度兜底；
  **主仿真不读 `E:\data`**。`E:/data/eee_{01,02,03}` 仅 `extract_measured_tracks.py:20`
  在「离线抽取」阶段需要，用于重建该 JSON。口径：
  *derived speed statistics ship with the repository; `E:\data` needed only to re-run the
  extraction script*。**AUTHOR TODO**：`measured_tracks.json` 的 `summary.source` 是裸路径
  `E:\data\eee_{01,02,03}`，**无 DOI / 许可声明**。
- **UWB tracks 计数核对（AUTHOR TODO，不静默调和）**：审计意见提到「shipped JSON 为
  4 tracks / 1195 行，与稿件 6 tracks 不符」。**实核：仓库内 `measured_tracks.json` 为
  12,134 行 / 203,363 B，`summary.n_tracks=6`、`n_rounds_total=1992`、
  `ranging_error.*.err_median_m=0.055`，`tracks` 含 6 个键**（eee_01/02/03 × tag200/201），
  与 `manuscript_access.tex:863–864`、`manuscript_peerj.tex:786–787` 的「6 tracks,
  1992 ranging rounds, 0.055 m median」**逐项一致**；全项目未找到 4 tracks / 1195 行的产物。
  故**不存在 4-vs-6 不一致**；若审计所见为另一版本 / 局部视图，请作者复核。仍为
  AUTHOR TODO：`summary.source` 裸路径无 DOI / 许可；若 lead 从新版 scope 段移除这些计数，
  请同步说明依据（实测数据是 6 tracks）。
- **AUTHOR TODO**：下载 17 GB 原始包、在 Zenodo/GitHub 存档、决定是否把逐条 CSV 做成
  有文档的机器可读子集（Data S1）。逐条 `f1*.csv` 约 20 MB，技术上可入补充包（总计仍
  ≤50 MB / 单文件 ≤30 MB），但**再分发许可待作者确认，不得擅自复制 / 公开**；本包未包含
  任何逐条原始 CSV。

---

## 7. 环境与依赖（AUTHOR TODO：精确版本待钉）

- `requirements.txt` / `environment.yml` 只列出可从代码 import 验证的包：`numpy`、
  `scipy`（现被顶层 import，仅用于记录版本）、`cvxpy`（含 CLARABEL）、`clarabel`（现被
  顶层 import，随 cvxpy 附带）、`matplotlib`。`scipy` / `clarabel` 都是 cvxpy 的传递依赖，
  `pip install numpy cvxpy matplotlib` 即足够，无需单列 pip 条目（见两文件注释）。
- **精确版本号一律留作 AUTHOR TODO，不得编造。** 本机（typesetting / 复现策展用机）测到的
  版本**不是**产出论文数据的那套环境；本机**基础解释器**的 cvxpy / CLARABEL / osqp /
  scs / ecos 全部缺失，**但不能据此断言「依赖 CVXPY 的产物一律不可生成」**：exp-worker
  另建隔离虚拟环境 `.venv_exp` 成功安装了 cvxpy（含 CLARABEL），**真实生成并严格校验**
  了 multi-start 复核产物（见 §2.4 / `provenance.md` §D）——那是**另一次重跑，非原始
  投稿环境**。**严禁把本机 / 重跑环境版本号写入任何交付物当作 pin**，那会导致复现失败。
- **AUTHOR TODO（阻塞）**：请作者在**生成投稿图表的原始可运行环境**（必然装有 cvxpy）中
  `pip freeze` / `conda list`，回填 Python、numpy、scipy、matplotlib、CVXPY、CLARABEL、OS
  的精确版本，并确认 CLARABEL 是否随所钉 CVXPY 版本附带。
- 建议最小安装（**版本待作者确认**）：`pip install "cvxpy>=1.5"` 并确认 CLARABEL 随附后
  `pip freeze`。`cvxpy>=1.5` 仅是仓库历史 README 中的提示性下限 / 占位，非本包断言的 pin。
- **代码存档路径**：`.py` 源码通过**仓库 + Zenodo DOI** 存档（见
  `data_availability_draft.md`），**不在 `supplement/` 内重复**；`supplement/` 只放 CSV/JSON
  输入与聚合结果。两者共同满足 PeerJ CS「投稿时即可获得全部代码与原始数据」。
- **改完 ≠ 已复现（分层结论）**：repro-exec 的脚本改动只做了 `py_compile` / `--help`
  语法校验与 `validate_aerpaw.py --demo`（不依赖 CVXPY）的运行时验证。
  - **multi-start 复核产物**（`convergence_multistart_*`）已由 exp-worker 在 `.venv_exp`
    真实落盘并严格校验 → **非 UNVERIFIED**，但属另一次重跑（§2.4）。
  - **原始投稿图表那次运行**（2026-10-07）的产物与精确环境仍 **UNVERIFIED**，须由
    作者在**原始可运行环境**跑通并 `pip freeze` 回填后才算复现；本包不把「改完」写成
    「已复现原始结果」。

---

## 8. `supplement/` 内容与体积自检

`repro/supplement/` 只放 **CSV/JSON 机器可读输入与聚合结果**；清单、逐文件来源与排除项见
`repro/SUPPLEMENT_MANIFEST.md`。**排除**：PDF、字体（`.pfb/.tfm/.map/.fd`）、PNG/JPG、
`.tex/.cls/.sty`、`.pyc`、`.zip`、`.md`、17 GB 原始 AERPAW CSV。

体积自检（作者增删文件后请重跑）：

```bash
python - <<'PY'
from pathlib import Path
d = Path("repro/supplement")
files = sorted(p for p in d.iterdir() if p.is_file())
total = 0
for p in files:
    s = p.stat().st_size
    total += s
    assert s <= 30 * 1024 * 1024, f"单文件超限: {p.name} {s}"
    print(f"{s:>10d} B  {p.name}")
assert total <= 50 * 1024 * 1024, f"总大小超限: {total}"
print(f"TOTAL {total} B = {total/1024/1024:.3f} MiB  (<=50 MB OK)")
allowed = {".csv", ".json"}
for p in files:
    assert p.suffix.lower() in allowed, f"禁止的文件类型: {p.name}"
print("文件类型检查 OK (仅 .csv/.json)")
PY
```

---

## 9. 目录索引

```
repro/
├── README.md                      # 本文件（权威入口）
├── requirements.txt               # 依赖（版本 AUTHOR TODO）
├── environment.yml                # conda 环境（版本 AUTHOR TODO）
├── CITATION.cff                   # 引用模板（AUTHOR TODO）
├── LICENSE                        # 许可证模板（AUTHOR TODO）
├── provenance.md                  # 数值/图表 → 脚本+函数+输入+环境+commit 映射表
├── data_availability_draft.md     # Data and Code Availability 段落草稿（DOI 待填）
├── SUPPLEMENT_MANIFEST.md         # supplement/ 逐文件清单与排除项
└── supplement/                    # 机器可读 CSV/JSON（≤50 MB / ≤30 MB）
```

---

## 10. AUTHOR TODO 总表（本复现包内所有显式占位）

- `requirements.txt` / `environment.yml`：精确版本 pin。
- `CITATION.cff`：`authors[].family-names/given-names/orcid`、`version`、`date-released`、
  `title`、`doi`、`url`、`license`、`copyright-holder`、`year`。
- `LICENSE`：版权持有人、年份、许可证选择（代码许可 vs 数据许可）。
- `data_availability_draft.md`：Zenodo DOI、GitHub 仓库 URL、Data S1 编号与仓库内路径。
- 正文 `manuscript_peerj.tex` 的各类 `[AUTHOR TODO]`（funder-role、Competing Interests、
  Author Contributions、Ethics、Data and Code Availability）。
- 下载 Dryad 17 GB 原始包；决定并存档逐条 CSV 子集（**再分发许可待确认**）；钉软件版本
  （**在原始可运行环境 `pip freeze`，不得用本机版本**）；确认 ORCID / 机构。
- 裁决 §2.1 的两次独立计时口径；确认 §2.2 的 `ablation.csv` 溯源表述。
- `measured_tracks.json` 的 `summary.source` 补 DOI / 许可；复核 §6 记录的 6-tracks 计数
  （与稿件一致）；若从稿件移除这些计数，说明依据。
- **lead 裁决稿件冲突**：`manuscript_peerj.tex` L623–626 / L1137–1138 / L1293–1294 的
  「monotone / multi-start remain future work」须与已完成的 5-seed 复核及 24.7% solver
  失败披露对齐；0.06% / 21.7% 标注为 single-trajectory、不得写成 multi-start 支持。
- 回填 commit hash / release tag（仓库当前不是 git 仓库）。`paper/data/` 已定稿、两份
  `supplement/` 副本逐字节一致、无需刷新；**唯一条件触发**：若 lead 要求把「148 次失败
  逐块细分」写进 JSON/md，exp-worker 重跑（改变哈希/UTC）并先通知，再按
  `SUPPLEMENT_MANIFEST.md §2` 的两条 `cp` 刷新并复核（当前非待办）。
