# Provenance 映射表 — 稿件数值 / 图表 → 脚本 + 函数 + 输入 + 环境 + commit

对应 PeerJ Computer Science **改投稿** `paper/manuscript_peerj.tex`（只读核对，未改动）。
图/表编号取自该文件的 `\label{}`；行号为该文件当前行号（如作者后续改稿会漂移，请按标签定位）。

> **稿件版本唯一性**：本表所有 `manuscript_peerj.tex:NNN` 均指 **PeerJ CS 改投稿**。
> 仓库另有两份旧稿 `paper/manuscript_access.tex`（IEEE Access）与 `paper/manuscript.tex`，
> 同一组表述在它们里面是**另一套行号**——例如「6 tracks / 1992 rounds / 0.055 m」在
> `manuscript_access.tex:863–864`、`manuscript_peerj.tex:786–787`、`manuscript.tex:662–663`；
> 「multi-start remain future work」在 `manuscript_access.tex:703`、`manuscript_peerj.tex:626`、
> `manuscript.tex:558`。**三份稿件行号不得混用**，一律以改投稿 `manuscript_peerj.tex` 为准。

## 图例与全局前提

- **状态**：`TRACEABLE` = 可追溯到具体脚本/函数/数据文件；`PARTIAL` = 部分可追溯；
  `UNVERIFIED` = 无法追溯 / 未证实，**不得当作已确证事实引用**。
- **commit 列**：本仓库当前**不是 git 仓库**（已核实 `git rev-parse` 失败），故全部记为
  `N/A (no git repo)`。AUTHOR TODO：存档到 Zenodo/GitHub 后回填 release tag / commit hash。
- **环境列**：生成投稿图表的那次运行（mtime 批次 **2026-10-07 08:42–08:44**）的
  Python / CVXPY / CLARABEL / OS 精确版本**均未记录 → UNVERIFIED / AUTHOR TODO**。
  exp-worker 于 **2026-11-12** 在独立虚拟环境 `.venv_exp` 做的 multi-start 复核是
  **另一次重跑**（其精确版本记录在该产物 JSON 的 `environment` / `software` 块），
  **不得当作原始投稿环境、不得当作版本 pin**。本机（typesetting 用机）测到的版本同样不是
  原始运行环境（且 cvxpy/CLARABEL 缺失），**严禁写入任何交付物当作 pin**。AUTHOR TODO：
  由作者在**原始可运行环境** `pip freeze` 回填。
- **两次独立计时运行**：见 §A；**`ablation.csv` 溯源保留意见**：见 §B；
  **demo 与实测隔离**：见 §C；**multi-start 产物非原始图表字节 / solver 失败披露**：见 §D；
  **UWB 溯源与计数**：见 §E。

### ⚠ 行号基准 / 漂移说明（务必先读）

本表所有 `:NNN` 行号沿用 **repro-audit 与任务背景的 2026-10-07 快照口径**（当时的
`sim_analysis_3gpp.py` 共 959 行）。**2026-11-12 exp-worker 为 multi-start 复核重构了该脚本
（959 → 1423 行），行号整体后移**：因此**一律以「函数名」为准定位**，行号仅作辅助。
另 `validate_aerpaw.py` / `aerpaw_validation_detail.py` 因本次复现包改动也整体后移。
`opt_gap_3gpp.py` **未改动**，其 `:87/89/95/96/99/113/117/138` 仍精确有效。

| 定位点 | 2026-10-07 快照 | 2026-11-12 当前 | 备注 |
|--------|-----------------|-----------------|------|
| 系统参数常量段 | `sim_analysis_3gpp.py:44–101` | `:49–106` | 内容未变，整体上移 5 行 |
| `solve_unaccel()` | `:257` | `:262` | exp-worker 追加 `solve_stats=None` 参数 |
| `run_variant()` | `:490` | `:514` | — |
| `convergence_experiment()` | `:643–688` | `:795–` | 签名改为 `seed=0, rounds=40, out_png=None, dump=True, users=None`（`users` 供 5 seed 共享场景、保证 seed0 逐位复现） |
| `main()` | `:838` | `:1283` | — |
| `main()` 内 `run_variant()` 调用 | `:878` | `:1322`（循环），`:1323` | — |
| `rel_lat` 计算 | `:882–884` | `:1327–1329` | — |
| `ablation.csv` 写入 | `:892` | `:1337` | — |
| 50 ms `ax.axhline` | `:916` | `:1361`（另 `:1245` 在演化图） | — |
| `ablation.png` 保存 | `:938` | `:1383` | — |
| `frac50=(dec_ms<50.0)` | `:620` | `:644`（在 `run_variant()` 内） | — |
| `validate()` | `validate_aerpaw.py:99` | `validate_aerpaw.py:113` | — |
| JSON 写出 | `validate_aerpaw.py:214` | `validate_aerpaw.py:276` | 改为 `_json_safe` + `allow_nan=False` |
| `main()` | `aerpaw_validation_detail.py:44` | `:49` | 新增 `--csv-dir` |
| `ALL` 汇总 / JSON 写出 | `:85` / `:143` | `:105` / `:163` | — |

---

## §A. 两次独立计时运行（强制 caveat，需作者裁决）

| 数据文件 | 计时数值 | 产生它的 `run_variant` 调用 | 是否被稿件引用为计时 |
|----------|----------|-----------------------------|----------------------|
| `ablation.csv` | v5 **13.3 ms**、v4 **227 ms**、v4 `rel_lat_x` **17.1×** | `sim_analysis_3gpp.py:878`（`main()` 内 5 个 variant 循环 → `:892` 写 CSV） | **是**（摘要、tab:abl、tab:lat） |
| `opt_gap_3gpp.json` | `mean_lat_ms.v4 = 225.7240400009323`、`v5 = 14.610509999329224` | `opt_gap_3gpp.py:87`（`lat_ms[name] = m["mean_lat"]`，L89） | **否** |

- 这**是两次彼此独立的运行 / 两个独立进程**（两组独立的 `run_variant` 调用，CLARABEL
  canonicalization / DPP-cache / warm-start 状态不同），属**运行间计时波动**；
  **NOT a data conflict，NOT two revisions of one measurement**。
- 两者的加速比不同：227/13.3 ≈ **17.07×** vs 225.724/14.6105 ≈ **15.45×**
  （`opt_gap_3gpp.py:113` 打印的是后者）。
- 目标函数 `J` 由固定种子决定、两次一致（v4≈0.6101066、v5≈0.7322259、ref≈0.6096518）；
  时延是 wall-clock、随机器 / 运行波动，**不可跨运行比较**。
- **全项目 grep 佐证**：`225.724` / `14.6105` 仅命中 `opt_gap_3gpp.json:12–13`，
  **任何 `.tex` 零引用** → **稿件 227 / 13.3 / 17× 无需改动**。
- **要求：do not mix, do not treat as revisions；不得静默统一 / 取平均 / 互换；
  DO NOT change manuscript numbers。** AUTHOR TODO：明确稿件采用哪一组口径，并在方法与
  Data Availability 中说明计时口径；`opt_gap_3gpp.json` 已有 `note` 字段，建议同时在该 note
  或补充包交代计时来源，避免审稿人拿补充数据包对比稿件数值生疑。

## §B. `ablation.csv` 溯源保留意见（不得断言确证）

- **内容高置信**：`ablation.csv` 的 v5 行（13.3 / 14.4 / 100 / 0.761 / 0.732）与 v4 行
  （227 / 311 / 0 / 0.762 / 0.610）与 `manuscript_peerj.tex:834–835`（tab:abl）**逐项吻合**，
  可判定为 3GPP 版本内容。
- **但生成过程 / 时间未证实**：mtime **2026-10-07 08:44:45.886414**，比整批 08:43 演化图
  （`paper/figs/ablation.png` 08:43:08.377358 等）**晚约 97 秒**；写 CSV 的代码点
  `sim_analysis_3gpp.py:892` 却在 `ablation.png` 的 `savefig`（`:938`）**之前**执行；
  `sim_analysis.py`（HISTORICAL 原型）与 `sim_analysis_3gpp.py` 写出的 **12 列 CSV 头
  逐字节相同**；且**无运行日志**。
- **统一口径（请使用原文）**：**"high confidence 3GPP version by content; generating
  process/time unverified"**。**不得声称已确证**由 `sim_analysis_3gpp.py` 的某次特定运行生成，
  也不得把「内容吻合」升级为「生成过程已证实」。
- 对照：`paper/figs/convergence.png` 溯源**可靠**——mtime 08:42:47 落在
  `sim_analysis_3gpp.py`(08:42:32) 与演化图批次(08:43:08–08:43:14) 之间，且这些演化图
  **仅**由 `sim_analysis_3gpp.py:main()` 生成。

## §C. demo vs 实测隔离

- 真实实测：`aerpaw_validation.json`（`data_mode="measured"`，
  `is_real_measurement=true`，`source="f10.csv; f11.csv; f12.csv; f13.csv"`）。
- SYNTHETIC demo：`aerpaw_validation_demo.*`（`data_mode="synthetic-demo"`、
  `is_real_measurement=false`、带 `notice`）。本机实测 demo 输出 n=2000 /
  RMSE 5.51 dB / bias −0.10 dB —— **与 136,987 / −0.02 dB / 5.4 dB 完全无关，不可引用**。
- `aerpaw_validation.json` 的 `rmse_nlos_db` 原为裸 `NaN`，已改为 JSON `null`
  （因 `frac_model_los=1.0`、无 NLOS 子集，RMSE 未定义），严格 `json.loads` 通过。

## §D. multi-start 产物（另一次重跑，非原始图表字节）

- `paper/data/convergence_multistart_traces.csv`（列 `seed,block_index,J`，**800 数据行 =
  5 seed × 160 块**，每 seed 的 block_index 连续 1..160）与
  `paper/data/convergence_multistart_summary.json`（含 `per_seed` / `per_seed_traces`
  / `aggregate` / `definitions` / `software` / `environment` / `command` /
  `utc_timestamp` / `seed0_reference` / `figure_integrity`）由 exp-worker 于
  2026-11-12 在隔离虚拟环境 `.venv_exp` 中**真实生成并落盘**（该环境成功安装
  cvxpy，`installed_solvers()` 含 CLARABEL；**不是** 2026-10-07 的原始投稿图表运行）。
  **这些产物已严格校验、真实存在，不是 UNVERIFIED / 待生成**；但仍属**另一次重跑**，
  不得当作原始投稿环境或原始图表字节。
- **`convergence.png` 的机器可读底层数据（重要，口径以此为准）**：原始运行
  （2026-10-07）当初只画图、未把逐 block trace 写盘；**但单轨迹（seed=0）的完整 160 点
  J 序列现已可恢复**，位置在：`convergence_multistart_traces.csv` 的 `seed=0` 全部 160 行，
  及 `summary.json` 的 `per_seed_traces["0"]`。exp-worker 用**未修改的原始文件**独立复跑
  验证 seed-0 **逐位相同**（`same_length=True`、`max|ΔJ|=0.0`，
  J 从 0.8081407143280384 → 0.5471407219285491）。因此 fig:conv 的底层数据应记为
  **「= multi-seed CSV 的 seed 0 行 / `per_seed_traces['0']`」**，**不要**写成存在一个
  单独的 `convergence.csv`。注意这是**重跑复现**，不是原始运行的字节。
- `summary.json` 自报 `figure_integrity.committed_figure_written_by_this_run=false`，
  即该重跑未改写 `paper/figs/`；`ablation.csv` mtime 仍 2026-10-07 08:44:45、
  `paper/figs/*` 未动。引用时应表述为「后续多起点收敛复核」。
- **solver 失败披露（exp-worker 实核）**：5 seed × 120 次 = **600 次求解调用中 148 次
  （24.7%）失败**（每 seed 29–31 次），**100% 集中在 UAV 位置可行性块**
  （CLARABEL SolverError / infeasible / infeasible_inaccurate / user_limit），
  失败时 UAV 位置不更新；另有 **22 次 optimal_inaccurate 被原代码按成功处理**。
  这些正是 `try/except Exception: pass` 静默吞掉的。计数见
  `summary.json` 的 `aggregate.solver_failures_total / solver_calls_total` 与
  `per_seed[].solver_failures_total / solver_exceptions / solver_nonoptimal /
  solver_calls`；逐块细分脚本 `.venv_exp/_scratch/diagnose_fallbacks_per_block.py`。
  **因此本复现包不作「干净收敛 / 单调收敛」断言**；稿件
  `manuscript_peerj.tex:623–626` 的 "the curve is monotone …" 及
  `:626 / :1137–1138 / :1293–1294` 的 "multi-start statistics remain future work"
  与已完成的 5-seed 复核冲突，**须由 lead 更新**（本包不改动稿件）。
- **0.06% / 21.7% 是 single-trajectory**：`opt_gap_3gpp.py` 的 `REF_ROUNDS=30` 让每个
  variant **各跑一次、无重启**；这两个间隙数字**不得描述为 multi-start 支持**。
  且 `run_variant` 路径同样存在静默回退、solver status 未单独落盘 → **该路径（非
  multi-start 复核）**的 solver 成功/失败谱 **UNVERIFIED / AUTHOR TODO**。
- **不存在的「虚名产物」（勿寻找、勿列为待生成）**：`convergence.csv`、
  `solver_status.csv`、`solver_events.csv` 及 `SOLVER_EVENTS` 符号**全项目均不存在**；
  repro-exec 因发现 exp-worker 并发编辑同一函数而主动停手，未实现它们。
  **等效物就是已落盘的 `paper/data/convergence_multistart_*`**：trace 落盘由
  `convergence_multistart_traces.csv` 承担，CLARABEL 回退可见性由 exp-worker 的
  `solve_stats` instrumentation 把逐种子失败数计入 `summary.json` 承担。
  本包不引用这些虚名文件、不把「改完」写成「已复现」。
- **函数名冻结（exp-worker 确认，按函数名定位稳定）**：`paper/data/` 已定稿，不再编辑。
  - 改动：`solve_unaccel`（+`solve_stats`）、`convergence_experiment`
    （+`seed/rounds/out_png/dump/users`）、`main()`（**本身未改动**，仅行号后移；
    `__main__` 新增 argparse 分支）。
  - 新增：`_ORIGINAL_UAVS0`（常量）、`_poor_init`、`_plot_convergence`、`_agg`、
    `_trace_metrics`、`_write_summary_md`、`run_convergence_multistart`。
  - 未改：`run_variant`、`FastSCASolver`、`solve_lqe_analytic`、`make_evolution_fig`、
    `make_v5_fig`、`make_users`、`channel_gains` 等。
  - 唯一可能的后续变动：若 lead 要求把「148 次全部在 position 块」的逐块细分写进
    JSON/md，exp-worker 会重跑（改变哈希与 UTC）并**先通知本包重新同步副本**；
    在此之前 `repro/supplement/` 两份副本即定稿。

## §E. UWB 运动统计：障碍降级、溯源与计数核对

- **不是运行时障碍**：`sim_analysis_3gpp.py:load_measured_speed_profile()`（`:194–231`）
  **只读仓库内 `measured_tracks.json`**（203,363 B，随 `supplement/` 提供），`:196–199`
  有常数速度兜底；**主仿真不读 `E:\data`**。口径：*derived speed statistics ship with the
  repository; `E:\data` needed only to re-run the extraction script*
  （`extract_measured_tracks.py:20` 的离线抽取阶段）。`aerpaw_data/` 内只有 `README.md`，
  17 GB 原始包缺失仍是真实 gap，但与 UWB 运动统计无关。
- **计数核对（AUTHOR TODO，不静默调和）**：审计意见称「shipped JSON 为 4 tracks / 1195 行，
  与稿件 6 tracks 不符」。**实核：该 JSON 为 12,134 行 / 203,363 B，
  `summary.n_tracks=6`、`n_rounds_total=1992`、`ranging_error.*.err_median_m=0.055`，
  `tracks` 含 6 个键**（`eee_01/02/03 × tag200/201`），与
  `manuscript_peerj.tex:786–787`（及 `manuscript_access.tex:863–864`）的
  「6 tracks, 1992 ranging rounds, 0.055 m median」**逐项一致**；全项目未找到
  4 tracks / 1195 行产物 → **不存在 4-vs-6 不一致**。若审计所见为另一版本 / 局部视图，
  请作者复核；若 lead 从新版 scope 段移除这些计数，请说明依据（实测为 6 tracks）。
- **AUTHOR TODO**：`measured_tracks.json` 的 `summary.source` 是裸路径
  `E:\data\eee_{01,02,03} (UWB ranging, 3 fixed anchors @ z=1.5 m)`，
  **无 DOI / 许可 / 引用声明**；原始 `E:\data` 数据未随包，重分发许可待确认。

---

## 1. 定量声明映射表

| 稿件编号 / 位置 | 声明 | 脚本 | 函数 / 行 | 输入数据 | 环境 | commit | 状态 |
|-----------------|------|------|-----------|----------|------|--------|------|
| 摘要 L40–41、L300 | 136,987 实测路径损耗样本；bias −0.02 dB；RMSE 5.4 dB（精确 5.39） | `validate_aerpaw.py` + `aerpaw_validation_detail.py` | `validate()` :99–128；`main()` :85–92（ALL） | `aerpaw_validation.json`、`aerpaw_validation_detail.json`；原始 `f10..f13.csv`（Dryad `10.5061/dryad.2ngf1vj4r`，17 GB，**未随包**） | 原始批次版本 UNVERIFIED | N/A (no git repo) | **TRACEABLE**（聚合值）；原始 17 GB 源 UNVERIFIED |
| 摘要 L43、L97；tab:abl L835；tab:lat L986 | v4 平均 **227 ms**（p95 311 ms，<50 ms 0%） | `sim_analysis_3gpp.py` | `main()` → `run_variant()` :878 → CSV :892/899 | `ablation.csv` 行 `v4`；`measured_tracks.json`（速度轮廓） | UNVERIFIED / AUTHOR TODO | N/A | **PARTIAL**（数值可追溯；但见 §A/§B 两处 caveat） |
| 摘要 L50；tab:abl L834；tab:lat L986；L598、L137 | v5 平均 **13.3 ms**（p95 14.4 ms，<50 ms 100%） | `sim_analysis_3gpp.py` | 同上 | `ablation.csv` 行 `v5` | UNVERIFIED | N/A | **PARTIAL**（同上） |
| 摘要 L51；L851；L996；L1250 | **17× / 17-fold** 加速 | `sim_analysis_3gpp.py` | `main()` :882–884（`rel_lat = mean_lat/lat_ref`）→ CSV 列 `rel_lat_x` | `ablation.csv`（v4=17.1） | UNVERIFIED | N/A | **PARTIAL**（227/13.3≈17.1；见 §A：另一独立运行给出 15.45×） |
| 摘要 L43、L50；多处 | **50 ms** 硬实时预算 | `sim_analysis_3gpp.py` | `run_variant()` :620（`frac50=(dec_ms<50.0).mean()`）；`main()` :916（`ax.axhline(50)`） | 无（**设计约束，字面量 50.0**，非测量） | 不适用 | N/A | **TRACEABLE**（设计常量） |
| 摘要 L53；L358；L1037；tab:gap L1066；L1109 | 凸化间隙 **0.06%**（v4 vs 30-round ref 均值，max 0.22%） | `opt_gap_3gpp.py` | `gap_pct()` :59；:96 `g_v4`；:99 `s_v4=stat(...)` → :138 JSON | `opt_gap_3gpp.json: surrogate_gap_v4_pct.{mean=0.0620603, maxabs=0.2233844}`；`opt_gap_3gpp.csv` 列 `gap_v4_pct` | UNVERIFIED | N/A | **TRACEABLE** |
| L143、L872、L1038；tab:gap L1067；L1106、L1114、L1283 | 加速代价 **+21.7%**（均值，max 67.4%） | `opt_gap_3gpp.py` | :95 `g_acc=gap_pct(J["v5"],J["v4"])`；:99 `s_acc=stat(...)`；:97/99 `s_v5` → JSON | `opt_gap_3gpp.json: acceleration_cost_pct.mean=21.6637`（及 `surrogate_gap_v5_pct.mean=21.7322`）；`opt_gap_3gpp.csv` 列 `acc_gap_v5_v4_pct` | UNVERIFIED | N/A | **TRACEABLE**；注意稿件用同一个「21.7%」指称两个略不同的量（21.66% 与 21.73%），tab:gap 注 L1071 已说明是逐漂移均值 |
| tab:gap L1066–1068 | Mean J：v4 0.610 / v5 0.732 / ref 0.610 | `opt_gap_3gpp.py` | :88 `J[name]=m["Js"]`；:130 `mean_J` | `opt_gap_3gpp.json: mean_J` | UNVERIFIED | N/A | **TRACEABLE** |
| tab:params L647–676；正文 L678–703 | 仿真参数表（见 §3） | `sim_analysis_3gpp.py` + `channel_3gpp.py` + `channel_38901_fading.py` | 常量 :44–101 / :47–53 / :65–72 | `channel_calib.json`、`fading_calib.json` | UNVERIFIED | N/A | **TRACEABLE**（源码常量） |
| fig:conv 标题 L632 | SCA–BCD 收敛曲线 | `sim_analysis_3gpp.py` | `convergence_experiment()` :643–688 → `paper/figs/convergence.png` | 原始运行逐点字节未落盘（UNVERIFIED）；**但单轨迹曲线可由 §D 的 `convergence_multistart_traces.csv` 的 `seed=0` 全部 160 行 / `per_seed_traces["0"]` 完整恢复**（未改动原文件独立复跑逐位验证 `max\|ΔJ\|=0.0`），属重跑复现、非原始字节；**无单独 `convergence.csv`** | 重跑环境见 §D（非原始） | N/A | **PARTIAL→可恢复**（图可追溯；底层曲线经重跑复现恢复，见 §D） |
| 正文 L295（CV≈0.06）、fig:fading L321–322 | 快衰落 / 阴影标定 | `make_fading_fig.py` + `calibrate_official_fading.py` | `main()`；`calibrate_official_fading.py:205` | `fading_calib.json`、`calib_official.json`；后者原始源 `D:/matlab_work/cdl_data.json` **未随包 / UNVERIFIED** | UNVERIFIED | N/A | **PARTIAL**（派生 JSON 随包；原始源 UNVERIFIED） |

> **表内两条间隙数字的限定**：0.06% 与 21.7% 均来自 `opt_gap_3gpp.py` 的
> `REF_ROUNDS=30`、**每个 variant 单次运行、无重启** → **single-trajectory**，
> **不得描述为 multi-start 支持**；该 `run_variant` 路径同样存在静默回退、
> solver status 未落盘 → 其成功/失败谱 UNVERIFIED（详见 §D）。
> **UWB 运动统计**（`measured_tracks.json`）的溯源、非运行时障碍与 6-tracks 计数核对见 §E；
> **稿件版本 / 行号唯一性**见文首说明；**两次计时**见 §A、**ablation.csv** 见 §B。

## 2. 图表映射表

| 稿件图 / 表 | 文件 | 生成脚本 | 函数 / 行 | 底层机器可读数据 | 状态 |
|-------------|------|----------|-----------|------------------|------|
| fig:conv (L631) | `paper/figs/convergence.png` | `sim_analysis_3gpp.py` | `convergence_experiment()` :643–683 | **= multi-seed CSV 的 `seed=0` 全部 160 行 / `per_seed_traces["0"]`**（`paper/data/convergence_multistart_traces.csv` + `..._summary.json`；逐位复跑验证，重跑复现非原始字节，见 §D）；**无单独 `convergence.csv`** | PARTIAL→可恢复 |
| fig:ablation (L889) | `paper/figs/ablation.png` | `sim_analysis_3gpp.py` | `main()` :906–938 | `ablation.csv`（§B 保留意见） | PARTIAL |
| tab:abl (L828) / tab:lat (L981) / tab:v4ev (L959) | — | `sim_analysis_3gpp.py` | `main()` :892 CSV；`:617–637` `run_variant()` 指标 | `ablation.csv` | PARTIAL |
| fig:v2 (L928) | `paper/figs/mpc_evolution.png` | `sim_analysis_3gpp.py` | `make_evolution_fig()` :705 / `main()` :946–955 | `ablation.csv` 同源 `run_variant()` 轨迹（未单独落盘 → UNVERIFIED） | PARTIAL |
| fig:v3 (L935) | `paper/figs/v3_evolution.png` | 同上 | 同上 | 同上 | PARTIAL |
| fig:v4 (L942) | `paper/figs/v4_evolution.png` | 同上 | 同上 | 同上 | PARTIAL |
| fig:v5 (L948) | `paper/figs/v5_warmstart.png` | `sim_analysis_3gpp.py` | `make_v5_fig()` :790 / `main()` :950–955 | 同上 | PARTIAL |
| fig:gap (L1085) / tab:gap (L1061) | `paper/figs/opt_gap.png`、`opt_gap_bar.png` | `opt_gap_3gpp.py` | `main()` :146–182（图）、:116–139（数据） | `opt_gap_3gpp.csv`（逐 step）、`opt_gap_3gpp.json`（聚合） | **TRACEABLE** |
| fig:chcalib (L363) | `paper/figs/channel_calib.png` | `make_channel_fig.py` | `main()` :79（图）、:126–127（JSON） | `channel_calib.json` | **TRACEABLE** |
| fig:fading (L321) | `paper/figs/fading.png` | `make_fading_fig.py` | `main()` :77（图）、:106–107（JSON） | `fading_calib.json`、`calib_official.json`（原始源 UNVERIFIED） | PARTIAL |
| fig:aerpaw (L746) | `paper/figs/aerpaw_measured.png` | `make_aerpaw_paper_fig.py` | `main()` :120 | `aerpaw_validation.json`、`aerpaw_validation_detail.json`；原始 `f10..f13.csv` **未随包 / UNVERIFIED** | PARTIAL |
| （校验散点 + CDF，非稿件图） | `aerpaw_validation.png` | `validate_aerpaw.py` | `main()` :273 | `aerpaw_validation.json` | **TRACEABLE** |
| （细分图，非稿件图） | `aerpaw_validation_detail.png` | `aerpaw_validation_detail.py` | `main()` :142 | `aerpaw_validation_detail.json` | **TRACEABLE** |

## 3. 仿真参数表（tab:params）逐项溯源

| 参数 | 稿件值 | 源码位置 |
|------|--------|----------|
| 载波 `f_c` | 3.5 GHz | `channel_3gpp.py:48` `F_HZ=3.5e9`；`sim_analysis_3gpp.py:51` `F_CARRIER=ch3g.F_HZ` |
| 天线增益 | 4 / 2 dBi | `channel_3gpp.py:49–50` |
| 总带宽 / 每链路 | 10 MHz / 10/3 MHz | `channel_3gpp.py:52` `B_REF_HZ=10e6`；`sim_analysis_3gpp.py:60,101` |
| 发射功率 | [0.01, 1.0] W | `sim_analysis_3gpp.py:61–62` |
| UAV 数量 / 用户数 | 3 / 5 | `sim_analysis_3gpp.py:46–47` |
| 区域 / 高度 | 200×200 m / 50–150 m | `sim_analysis_3gpp.py:44–45` |
| 阴影 σ / 相关距离 | 4 dB / 37 m | `channel_38901_fading.py:65,69` |
| 快衰落 K / clusters | 7 dB / 12 | `channel_38901_fading.py:55,72` |
| 需求爬坡 | 30 → 62 Mb/s / 40 步 | `sim_analysis_3gpp.py:87–88`；`main()` :847–850 |
| MPC  horizon / Δt / V_step | 3 / 0.1 s / 1.0 m | `sim_analysis_3gpp.py:70–73` |
| 权重 | W_lat=1.0, W_en=0.3, W_acc_base=2.0, μ=0.5 | `sim_analysis_3gpp.py:75–78` |
| FM 代理 | L_FM={0.02,0.04,0.07,0.11,0.16}, A_FM={0.55,…,0.95} | `sim_analysis_3gpp.py:65–66` |
| 早退 / 量化惩罚 | 0.06 / 0.005 | `sim_analysis_3gpp.py:67–68` |
| 精度下限 A_min | 0.75 | `sim_analysis_3gpp.py:547` |
| 外循环轮数 | 3 / 5 / 10（v5 / v4–v2 / v1） | `sim_analysis_3gpp.py:852–869` |
| 随机种子 | channel 2024 / fading 991 | `sim_analysis_3gpp.py:57–58` |
| `M=500` / `τ_OOD=0.7` | 「not exercised」 | 稿件 L672–673 自述未使用 → **UNVERIFIED / 无对应源码常量**（作者确认） |
| 「rate ≈57–68 Mb/s」「slack <0.02 Mb/s」 | 正文 L689–695 | 由 `channel_calib.json:achievable_rate_Mb_s` 推得；逐 step slack 无落盘 → **PARTIAL / UNVERIFIED** |

## 4. 明确标记 UNVERIFIED 的项

1. 原始投稿批次（2026-10-07）的 Python / CVXPY / CLARABEL / OS 精确版本 —— 无记录。
2. `ablation.csv` 的生成过程 / 时间（§B）。
3. `convergence.png` **原始运行**的逐点字节 —— 从未落盘（UNVERIFIED）。注意：单轨迹曲线
   本身**已可恢复**于 §D 的 `seed=0` 160 行 / `per_seed_traces["0"]`，经未改动原文件重跑
   逐位验证；此处 UNVERIFIED 仅指「原始那次运行的字节」，非「曲线数据不可得」。
4. 演化图 fig:v2–v5 的逐 step 轨迹 —— 只在内存中，无独立 CSV/JSON。
5. AERPAW 17 GB 原始 `f10..f13.csv`（超出补充文件限制，作者需自行下载 / 存档）。
6. `calib_official.json` 的原始源 `D:/matlab_work/cdl_data.json`；`measured_tracks.json` 的原始源
   `E:\data` UWB `.pkl` —— 均为外部数据，未随包（但 UWB 派生的速度统计**已随包**，见 §E，
   主仿真不读 `E:\data`，**不是运行时障碍**）。
7. Zenodo DOI、GitHub 仓库 URL、ORCID、版权持有人 / 年份 / 许可 —— 全部 AUTHOR TODO。
8. `M=500` / `τ_OOD=0.7`、逐 step rate slack —— 稿件自述未使用或无落盘数据。
9. 两次独立计时运行的口径裁决（§A）。
10. `measured_tracks.json` 的 `summary.source` 无 DOI / 许可；原始 UWB 重分发许可待确认（§E）。
11. 0.06% / 21.7% 间隙所在 `run_variant` 路径的 solver status / 回退谱 —— 未单独落盘（§D）。
12. 稿件 `manuscript_peerj.tex` 的 monotone / "multi-start remain future work" 与已完成
    5-seed 复核的冲突裁决 —— 由 lead 处理，本表不改动稿件（§D）。

---

## 5. AUTHOR TODO（与本表相关）

- 回填 commit / release tag、精确软件版本（**在原始可运行环境 `pip freeze`，严禁用本机 /
  重跑环境版本**）、ORCID、机构、许可、Zenodo DOI、仓库 URL。
- 裁决 §A 两组计时（do not mix / do not treat as revisions / 不改稿件数值）；确认 §B
  `ablation.csv` 表述。§D 的单轨迹曲线机器可读底层数据**已落盘**（multi-seed CSV 的
  `seed=0` 行 / `per_seed_traces["0"]`，逐位复跑验证），**无需再补**；作者只需确认以
  「重跑复现」口径引用，并如实声明原始那次运行的字节不可得（§4 第 3 项）。
- 下载并存档 17 GB AERPAW 原始包；决定 Data S1 子集（**再分发许可待确认**）。
- 补 `measured_tracks.json` 的 `source` DOI / 许可；复核 §E 的 6-tracks 计数；若从稿件移除
  这些计数请说明依据。
- lead 更新稿件：monotone / future-work 表述对齐 5-seed 复核与 24.7% solver 失败披露；
  0.06% / 21.7% 标为 single-trajectory。
