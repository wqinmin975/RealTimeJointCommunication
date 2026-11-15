# `repro/supplement/` 清单

PeerJ CS 补充文件限制：**总计 ≤ 50 MB、单文件 ≤ 30 MB、机器可读（CSV/JSON）**。
本目录**只放机器可读 CSV/JSON 输入与聚合结果**；PDF / 字体 / 图片 / `.tex` / `.pyc` /
`.zip` / `.md` 一律排除（`.md` 文档放在 `repro/` 本层）。

**实测（2026-11-12）：11 个文件，总计 268,285 B = 0.2559 MiB**，远低于两个上限。

---

## 1. 逐文件清单（大小 / 来源脚本 / 支撑的图表）

| 文件 | 大小 (B) | 类型 | 来源脚本 + 函数/行 | 支撑的稿件图 / 表 / 数值 | 备注 |
|------|---------:|------|--------------------|--------------------------|------|
| `measured_tracks.json` | 203,363 | 输入 | `extract_measured_tracks.py:205`（外部 `E:\data` UWB `.pkl` 派生） | A1 的实测运动学 `load_measured_speed_profile()` :194 → 全部消融/收敛结果 | 原始 UWB 源**未随包 / UNVERIFIED**；此派生 JSON 即复现所需输入 |
| `ablation.csv` | 604 | 聚合结果 | `sim_analysis_3gpp.py:892`（`main()` → `run_variant()` :878） | tab:abl、tab:lat、fig:ablation；13.3 / 227 ms、17.1×、50 ms 达标率 | **溯源保留意见**：见 `provenance.md` §B（高置信度 3GPP 版 by content，生成过程/时间 UNVERIFIED） |
| `opt_gap_3gpp.csv` | 2,715 | 图源逐 step 数据 | `opt_gap_3gpp.py:117`（`np.savetxt`） | fig:gap 两条曲线；列 `step,J_v4,J_v5,J_ref,gap_v4_pct,gap_v5_pct,acc_gap_v5_v4_pct` | 40 步（41 行含表头） |
| `opt_gap_3gpp.json` | 1,009 | 聚合结果 | `opt_gap_3gpp.py:138` | tab:gap、fig:gap_bar；0.06%、21.7%、mean_J、mean_lat_ms | `mean_lat_ms` 225.724 / 14.6105 属**另一次独立计时运行**，见 `provenance.md` §A |
| `aerpaw_validation.json` | 957 | 实测聚合结果 | `validate_aerpaw.py:validate()` :99 → `main()` :276 | 摘要 / fig:aerpaw：136,987、bias −0.02 dB、RMSE 5.4 dB | **真实实测**（`data_mode="measured"`）；原裸 `NaN`→JSON `null`，见 §4 |
| `aerpaw_validation_detail.json` | 3,608 | 实测细分结果 | `aerpaw_validation_detail.py:143` | fig:aerpaw 按飞行 / 距离分箱 / 单斜率拟合；`ALL` n=136987, bias −0.0239, rmse 5.3916 | 需 17 GB Dryad 原始 CSV 才能重算 |
| `channel_calib.json` | 1,140 | 图源 | `make_channel_fig.py:126` | fig:chcalib（3.5 GHz 路径损耗 / 可达速率表） | — |
| `fading_calib.json` | 707 | 图源 | `make_fading_fig.py:106` | fig:fading（阴影 / 快衰落标定） | — |
| `calib_official.json` | 1,629 | 图源 / 标定 | `calibrate_official_fading.py:205` | 小尺度 Rician K 与官方 5G Toolbox 交叉校验 | 原始源 `D:/matlab_work/cdl_data.json` **未随包 / UNVERIFIED** |
| `convergence_multistart_traces.csv` | 21,004 | 收敛 traces | `sim_analysis_3gpp.py`（exp-worker 的 `--multistart`）→ `paper/data/` | fig:conv 的后续多起点收敛复核；列 `seed,block_index,J`（**800 数据行 = 5 seed × 160 块**，每 seed block_index 连续 1..160；+ 表头 = 801 行）。`seed=0` 全部 160 行 = 原始单轨迹曲线底层数据（见 `provenance.md` §D） | **§2**：定稿快照；来自 2026-11-12 另一次重跑 |
| `convergence_multistart_summary.json` | 31,549 | 聚合结果 | 同上 → `paper/data/` | 多起点收敛统计（J_initial/J_final、收敛轮、`solver_failures` 定义）+ 环境/命令元数据 | **§2**：**一次性 schema 对齐**（30,758 → 31,549 B）后**已定稿**，非持续编辑；非 UNVERIFIED / 待生成 |

**TOTAL = 268,285 B = 0.2559 MiB（11 files）** — 单文件最大 203,363 B（`measured_tracks.json`），
≤ 30 MB；总计 ≤ 50 MB。

---

## 2. 定稿快照一致性（multi-start 两个文件，已核验、非陈旧）

`convergence_multistart_*.{csv,json}` 是 `paper/data/` 的**快照拷贝**，取自
**2026-11-12**（`summary.json` 自报 `utc_timestamp = 2026-11-12T12:20:35.473633+00:00`）。
`paper/data/` **现已定稿、不再编辑**：此前观察到的 30,758 → 31,549 B 是 exp-worker 的
**一次性 schema 对齐**（给 `per_seed` 加 `rounds_to_1e-3` / `rounds_to_1e-2` /
`solver_failures` / `label` 兼容别名 + 顶层 `environment` 版本块，以兼容
`paper/make_peerj.py::load_multistart()`），**不是持续编辑**；该次重跑 UTC 即上述
12:20:35.473633+00:00，与快照时间一致。

**已复核（exp-worker + 本包各一次）**：两份副本与 `paper/data/` 定稿产物**逐字节一致**
（`traces.csv` SHA-256 `dbe7d31a39a48240e2c4cc7d41423e579166df5e1529a1f2c7845293a924bbd7`、
`summary.json` `db049b65f68472d7007701c6456df1340e69aec7e472a8693d8a423064938707`），
且严格校验通过：`summary.json` `json.loads(parse_constant=reject)` OK、`seeds=[0,1,2,3,4]`、
`per_seed=5`；`traces.csv` **800 数据行**、表头 `seed,block_index,J`、全部 `J` 有限。
当前**无需刷新**。这些产物**真实存在、已验证（非 UNVERIFIED / 待生成）**，但来自
**exp-worker 在独立虚拟环境 `.venv_exp` 的另一次重跑**（精确版本记录在该 JSON 的
`environment` / `software` 块，**不是** 2026-10-07 生成投稿图表的原始运行，**也不得当作
原始投稿环境的版本 pin**）。

`convergence.png` **原始那次运行**的逐点字节从未落盘（UNVERIFIED）；**但单轨迹 seed-0
曲线的机器可读底层数据现已可恢复**：= `traces.csv` 的 `seed=0` 全部 160 行 /
`summary.json` 的 `per_seed_traces["0"]`，经未改动原文件独立复跑**逐位验证**
（`max|ΔJ|=0.0`）。**仓库内不存在单独的 `convergence.csv` / `solver_status.csv` /
`solver_events.csv`**，勿寻找、勿列为待生成。

`summary.json` 含 `per_seed` / `per_seed_traces` / `aggregate` / `definitions` /
`software` / `environment` / `command` / `utc_timestamp` / `seed0_reference` /
`figure_integrity` 及上述兼容字段，可被 `paper/make_peerj.py::load_multistart()`
（`STATS_FILE = paper/data/convergence_multistart_summary.json`）直接消费（exp-worker 用
真实源码验证返回非 None）。本包自检只做严格 JSON 解析，**不对 schema 做断言**。

**函数名冻结**（exp-worker 确认、无后续改名计划，`provenance.md` 行号按函数名定位稳定）：
改动 `solve_unaccel` / `convergence_experiment` / `main`（本身未改，仅 `__main__` 加 argparse
分支）；新增 `run_convergence_multistart`、`_ORIGINAL_UAVS0`、`_poor_init`、
`_plot_convergence`、`_agg`、`_trace_metrics`、`_write_summary_md`；`run_variant` /
`FastSCASolver` 等未改。

**条件刷新触发（唯一可能的后续变动）**：截至定稿，lead **未**要求把「148 次失败全部在
UAV position 块」的逐块细分写进 JSON/md。若之后要求，exp-worker 会重跑（**改变哈希与
UTC**）并**第一时间知会本包重新同步**下面两份副本；在此之前这两份副本即定稿、无需动作。

```bash
cp paper/data/convergence_multistart_traces.csv repro/supplement/
cp paper/data/convergence_multistart_summary.json repro/supplement/
```

引用时应表述为「后续多起点收敛复核」，不得声称是 `convergence.png` 的原始底层数据字节
（见 `provenance.md` §D）。

---

## 3. 明确排除的文件（不进 supplement/）

| 排除项 | 原因 |
|--------|------|
| `opt_gap.csv` | **HISTORICAL 原型**（`opt_gap.py`，理想化自由空间信道），`opt_gap_3gpp.py` 文档明确「the two are not comparable」；不得混入论文聚合结果 |
| `convergence_multistart_summary.md` | `.md` 文档，非机器可读数据；留在 `paper/data/` |
| 所有 `*.png` / `*.pdf` | PDF/PPT/图片不计为原始数据，且超出「CSV/JSON only」口径 |
| 字体 `*.pfb/.tfm/.map/.fd`、`*.tex/.cls/.sty` | 排版资产，非数据 |
| `*.pyc` / `__pycache__/` | 编译缓存 |
| `*.zip` / `*.tar.gz` | 归档容器；尤其 AERPAW 17 GB `data.tar.gz` 超出 50 MB / 30 MB 限制 |
| AERPAW 逐条 `f10.csv..f13.csv` | **17 GB 原始包未随包**（`aerpaw_data/` 内只有 `README.md`）；由作者从 Dryad `10.5061/dryad.2ngf1vj4r` 下载，或做成有文档的机器可读子集（Data S1）。逐条 CSV 约 20 MB、技术上可入补充包且仍 ≤50 MB / ≤30 MB，但**再分发许可待作者确认，本包未擅自复制 / 公开** |
| `.py` 源码 | 代码通过**仓库 + Zenodo DOI** 存档（见 `data_availability_draft.md`），**不在 `supplement/` 内重复**；`supplement/` 只放 CSV/JSON 输入与聚合结果 |
| 根目录历史原型输出 `mpc_evolution.png` / `v*_evolution.png` 等 | 历史原型产物，稿件只用 `paper/figs/` |

---

## 4. `aerpaw_validation.json` 的修改说明（重要，避免误读为改动测量值）

- 原文件 `rmse_nlos_db` 是**裸 `NaN` token**（非法 JSON）。已改为 **JSON `null`**
  （严格 `json.loads` 通过）。原因：`frac_model_los=1.0`，无 NLOS 子集，该 RMSE 未定义。
- 新增 4 个**provenance 元数据字段**：`data_mode="measured"`、`is_real_measurement=true`、
  `measurement_source_doi="10.5061/dryad.2ngf1vj4r"`、`rmse_nlos_db_note` / `provenance_note`。
- **所有测量字段（`n_samples` / `bias_expected_db` / `rmse_expected_db` / `rmse_los_db` /
  `frac_model_los` / `source` / `f_hz`）逐值未改。**
- demo 产物 `aerpaw_validation_demo.*` 永不写入此文件名；其 JSON 带
  `data_mode="synthetic-demo"` 与 `notice`，与实测值隔离。

---

## 5. 重跑自检（作者增删文件后必跑）

见 `repro/README.md §8` 的 Python 片段：打印逐文件大小与总计，断言
`<= 50 MB total` / `<= 30 MB each`，并断言仅含 `.csv` / `.json`；
对所有 `.json` 用 `json.load(..., parse_constant=raise)` 严格解析（拒绝裸 NaN/Infinity）。
本次自检结果：**11/11 JSON 严格解析通过、3/3 CSV 解析通过、体积与类型断言全通过**。
