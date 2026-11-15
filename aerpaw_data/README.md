# AERPAW 真实 UAV 空地测量数据集 — 获取与校验说明

本目录用于承接**真实测量的 UAV 空地（A2G）信道数据**，对论文的
TR 38.901 ULA/ULA-Los 大尺度路径损耗模型做「终极实测校验」。
小尺度 Rician K 已由官方 5G Toolbox `nrCDLChannel` 交叉验证
（`calibrate_official_fading.py`）；本项目补齐**大尺度**的实测校验。

## 选用数据集

**AERPAW — Air-to-Ground Channel Sounding and Path Loss Measurements at 3.4 GHz**
（Dryad DOI: `10.5061/dryad.2ngf1vj4r`）

- 真实 SDR 信道探测 campaign：UAV 接收机 ~60 m AGL，固定地面发射机 3.4 GHz，
  乡村环境（Lake Wheeler 试验场）。
- 4 个飞行（2 条重复直线 + 2 条重复之字形），含原始 I/Q 与处理后 CSV。
- 处理后 CSV 每行包含：
  `measurement_id, campaign_id, timestamp_utc, time_s, rx_latitude,
  rx_longitude, rx_altitude_agl_m, tx_latitude, tx_longitude,
  tx_altitude_agl_m, heading_deg, uav2ground_dist, path_loss_db`

该数据集是 Nature npj Wireless 2026（Raouf et al., "Curated Wireless Datasets
for Aerial Networks Research"）统一编目的公开 A2G 实测集之一，可直接引用。

## 获取步骤（一次性，需联网机器）

> 注：沙箱环境无法直接拉取——CSV 仅封装在 17 GB 的 `data.tar.gz` 内，且
> Dryad 的单个文件端点（`/downloads/file_stream/`）需要会话 cookie。
> 因此这一步在本地/服务器完成，之后 `validate_aerpaw.py` 不变地读取解包后的 CSV。

1. 下载并解包：
   ```bash
   # 在 Dryad 数据集页（https://datadryad.org/dataset/doi:10.5061/dryad.2ngf1vj4r）
   # 申请/下载 data.tar.gz（17 GB），解包后得到 f10.csv .. f13.csv
   tar -xzf data.tar.gz
   # 将 f10.csv .. f13.csv 复制到本目录（aerpaw_data/）
   ```
2. 运行校验（自动拼接所有 `f1*.csv`）：
   ```bash
   python validate_aerpaw.py --csv-dir aerpaw_data \
       --out-json aerpaw_validation.json --out-png aerpaw_validation.png
   ```
3. 指标解读（见 `aerpaw_validation.json`）：
   - `rmse_expected_db` / `bias_expected_db`：测量值 vs 模型 LOS 概率加权预测 PL。
   - `rmse_los_db` / `rmse_nlos_db`：分别针对模型判为 LOS / NLOS 的子集。
   - `frac_model_los`：模型判为 LOS 的比例，应与实测 LOS 占比趋势一致。

## 自洽演示（无需原始数据，验证管线本身）

```bash
python validate_aerpaw.py --demo
```

该模式用模型自身生成 AERPAW 风格的合成样本（60 m AGL、3.4 GHz、叠加阴影/
测量噪声），跑通同一套指标管线。**仅用于自洽检查，不是真实实测校验。**

> **文件名隔离（复现包修复）**：`--demo` 现在只写
> `aerpaw_validation_demo.json` / `aerpaw_validation_demo.png`，**绝不覆盖**
> `aerpaw_validation.*`；若被显式要求写真实产物名会直接报错退出。输出 JSON 内
> `data_mode="synthetic-demo"`、`is_real_measurement=false` 并带 `notice`，
> 以免合成数值被误当作 136,987 样本 / −0.02 dB / 5.4 dB 的实测结果。

## 与代码其它部分的关系

- `channel_38901_fading.py`：小尺度（已加 NLOS Rayleigh 深衰落分支）。
- `calibrate_official_fading.py`：小尺度 vs 官方 5G Toolbox 交叉验证。
- `validate_aerpaw.py`：大尺度 vs 真实 AERPAW 实测（本目录）。
