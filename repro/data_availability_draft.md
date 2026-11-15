# Data and Code Availability — 草稿（可直接改写进稿件）

> 用法：把下面 §1 的英文段落替换 `paper/manuscript_peerj.tex` 中现有的
> 「Data and Code Availability」小节（当前 L1315–1330 含大段 `[AUTHOR TODO]` 块）。
> **所有 `AUTHOR_TODO_…` 占位都必须由作者填写；此处未编造任何 DOI / URL / 版本 / 许可。**
> 已核实的真实 DOI 只有 Dryad `10.5061/dryad.2ngf1vj4r`（原稿已引用，来自任务背景）。

---

## §1. 英文段落（ready to adapt）

```latex
\subsection*{Data and Code Availability}
The AERPAW air-to-ground path-loss measurements used for the large-scale channel
validation are publicly available from Dryad
(doi:10.5061/dryad.2ngf1vj4r)~\citep{gurses2026dryad}. The full 17\,GB raw
package exceeds the supplementary-file limits; the processed per-flight CSVs
(``f10.csv''--``f13.csv'') can be obtained from that deposit and are read by the
validation scripts through a \texttt{--csv-dir} argument.

All code required to reproduce the results, together with the machine-readable
inputs and aggregated results behind every figure and table, is archived in a
public repository with a permanent Zenodo DOI:
repository \url{AUTHOR_TODO_REPOSITORY_URL}, archived release
doi:AUTHOR_TODO_ZENODO_DOI. The authoritative generator is
\texttt{sim_analysis\_3gpp.py} (3GPP TR~38.901 channel with a measured UWB
mobility profile); \texttt{opt\_gap\_3gpp.py}, \texttt{channel\_3gpp.py},
\texttt{validate\_aerpaw.py} and \texttt{aerpaw\_validation\_detail.py} produce
the optimality-gap, channel-calibration and measured-validation results,
respectively. Prototype scripts (\texttt{sim\_analysis.py}, the
\texttt{hrl\_airborne\_fm\_v1}--\texttt{v5} variants and \texttt{opt\_gap.py})
are retained for historical reference only and are not part of the reported
pipeline. Exact software versions (Python, CVXPY, CLARABEL, operating system)
are pinned in \texttt{repro/requirements.txt} and
\texttt{repro/environment.yml}:
AUTHOR_TODO_VERSION_STRING. The synthetic \texttt{--demo} self-check output is
written to separately named \texttt{*\_demo.*} files and is explicitly marked
\texttt{data\_mode="synthetic-demo"}; it is not the measured validation and must
not be cited as such.

A machine-readable subset of the aggregated results is supplied as
Supplementary Data~S1 (CSV/JSON only), well within the 50\,MB total /
30\,MB per-file supplementary-file limits; its per-file provenance is listed in
\texttt{repro/SUPPLEMENT\_MANIFEST.md} and the numeric-to-script mapping in
\texttt{repro/provenance.md}. Code and data licences:
AUTHOR_TODO_CODE_LICENSE / AUTHOR_TODO_DATA_LICENSE.
```

---

## §2. 中文改写说明 / 需作者裁决项

1. **路由选择**：当前稿件 L1328 的原 TODO 写「Choose EITHER the repository/Zenodo
   route OR the Supplementary route, not both」。本草稿默认走**仓库 + Zenodo DOI**
   （PeerJ CS 要求投稿时即可获得代码与原始数据），并额外提供一个
   **Supplementary Data S1**（聚合 CSV/JSON）。若编辑部要求二选一，请删掉 Data S1
   那一句并只保留仓库/Zenodo 路径；或反过来。AUTHOR TODO：确认并据反馈删改。
2. **占位符清单**（全部不得编造）：
   - `AUTHOR_TODO_REPOSITORY_URL` — 公开仓库 URL。
   - `AUTHOR_TODO_ZENODO_DOI` — **先注册 Zenodo 存档再回填，禁止凭空写 DOI**。
   - `AUTHOR_TODO_VERSION_STRING` — 由 `pip freeze` / `conda list` 得到的精确版本串。
   - `AUTHOR_TODO_CODE_LICENSE` / `AUTHOR_TODO_DATA_LICENSE` — 见 `repro/LICENSE`。
   - `Data S1` 编号与仓库内路径（若采用补充文件路线）。
3. **必须一并处理的其它 `[AUTHOR TODO]`**（同在 `manuscript_peerj.tex`）：
   funder-role statement (L1312)、Competing Interests (L1333)、
   Author Contributions / CRediT (L1337)、Ethics statement (L1343)。
4. **口径一致性**：段落里 13.3/227 ms、17×、0.06%、21.7%、136,987/−0.02 dB/5.4 dB
   的来源见 `repro/provenance.md`；两次独立计时运行的裁决（provenance §A）完成前，
   不要在 Data Availability 里对计时做新的断言。
5. **Dryad DOI 保留**：`10.5061/dryad.2ngf1vj4r` 是真实 DOI（原稿已引用），可保留；
   但原稿 L1325 已说明「The Dryad DOI alone does NOT satisfy this requirement」，
   所以必须另有 Zenodo DOI / 仓库 / Data S1。
6. **UWB 运动统计（建议补一句，避免审稿人误以为缺数据）**：派生速度统计
   `measured_tracks.json` 已随仓库 / Data S1 提供，主仿真只读该文件；外部
   `E:\data` 原始 UWB 数据**仅**在重跑 `extract_measured_tracks.py` 抽取脚本时才需要。
7. **收敛图底层数据 / multi-start 口径（避免与补充数据包对比生疑）**：
   `convergence.png` 原始那次运行的逐点字节未落盘，但单轨迹曲线已可由
   `convergence_multistart_traces.csv` 的 `seed=0` 行 / `per_seed_traces["0"]` 逐位恢复，
   该 multi-start 产物是**另一次重跑（后续多起点收敛复核）**、非原始投稿图表字节，
   且披露 600 次求解中 148 次（24.7%）失败。若 Data Availability 提及收敛数据，
   请用「subsequent multi-start verification」口径，勿写成原始单轨迹逐点日志；
   详见 `repro/provenance.md` §D / `SUPPLEMENT_MANIFEST.md` §2。
   AUTHOR TODO：`measured_tracks.json` 的 `summary.source` 是裸路径，需补 DOI / 许可 /
   引用声明；原始 UWB 数据的再分发许可待确认。
7. **逐条 AERPAW CSV 的再分发**：`f10..f13.csv` 约 20 MB，技术上可入 Data S1
   （仍 ≤50 MB / ≤30 MB），但**再分发许可待作者确认**，不得擅自复制 / 公开；本包只放
   聚合结果。若作者决定纳入，请在段落中明确子集范围与许可。
8. **版本串来源**：`AUTHOR_TODO_VERSION_STRING` 必须在**生成投稿图表的原始可运行环境**
   （装有 cvxpy）里 `pip freeze` / `conda list` 得到；**严禁使用本机 / 排版机 / 后续重跑
   环境的版本**（本机无 cvxpy，照搬会导致复现失败）。
