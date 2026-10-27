# E9 报告：prefix defrag 进树（v7-rc4 / v7-single-rc4）与 3-D 批次重跑（N56 hp_5090，2026-10-27 起）

> E8 的目录、数据与报告（`claude/e8_3d_report.md`，`docs/cluster_v6/data/e8/`）原样保留；E9 的数据在 `docs/cluster_v6/data/e9/`，作业脚本在 `docs/cluster_v6/e9_jobs/`，本报告按你 10-27 的清单逐项写。E8 7.18 的结论是起点：默认的 atomicAdd defrag 把 pid 块按到达序摆放，8M 的 slab 每粒子多付 12 %，64M 多付 2 %；prefix 序消掉这项代价。

## 0. 概要（随作业完成更新）

- 代码（第 1 节）：host 端 exclusive scan 成为 `experiment/v7` 的 defrag 正式路径并默认开启；v7_single 用同一份 scan；核函数与 SPIR-V 不变。
- 本机验证（第 2 节）：第 4–6 项**全部通过**。K = 1 的每粒子成本与 7.18 的 prefix 结果相差 −0.16 % / −0.03 %（8M / 64M），每次 defrag 后的 pid 序审计 0 次块间倒序；K = 2（同一张卡）内建检查全过、seam OK、第 300 步场与 K = 1 的差在阶段 1 的噪声水平；defrag 间隔 100 的运行三次周期 defrag 后步长无跳变。**2-D 敏感性：1M -0.82 %，8M -5.46 %**——8M 超过 0.5 % 的门槛，E7 要不要重跑等你定（第 2.3 节）。
- 第 7 项（Section 3 基线重测，600 W）、第 8–11 项（集群）：见第 3、4 节，随进度补。

## 1. 代码改动（分支 `e8-3d` → 标签 `v7-rc4`；分支 `v7-single` → 标签 `v7-single-rc4`；提交号见第 5 节）

| 文件 | 改动 |
|---|---|
| `experiment/v7/utils/defrag_prefix.py`（新） | host 端的生产者：`upload_defrag_prefix(sim)` 在每次 defrag 提交前读回 `inside_particle_count`，只对 **own 列 voxel**（`is_own_voxel` 的范围：leading ghost 之后到 trailing ghost 之前，ghost 列计 0）按 voxel 编号做 exclusive scan，上传 `voxel_base_offset`，并把 own 粒子总数写进 `defrag_scratch_counter`；`audit_pid_order(sim)` = 诊断（`V7_DEFRAG_AUDIT=1`），读回 `position_voxel_id` 检查 pid 序。|
| `experiment/v7/utils/simulator_v7.py` | `submit_defrag_and_wait`：prefix 时先 scan 再提交（启动 defrag 与周期 defrag 都走这里，restart 也是）；`_record_defrag_cmd`：prefix 时不再把计数器填 0（host 写入的总数经原有第 6 步拷进 `alive_particle_count`，第 7 步的 install tail 复位不变，departed 区不碰）；init 行打印 `defrag_order=`。|
| `experiment/v7/utils/case_v7.py` | `use_prefix_sum_defrag` 默认 `True`（spec constant 46 的来源不变）。|
| `experiment/v7/utils/case_loader_v7.py` | 新键 `numerics.defrag_order: prefix | atomic`，默认 prefix；**旧键 `use_prefix_sum_defrag` 忽略并打印一行提示**——库里全部 100 个 case.yaml 都由生成器写了 `use_prefix_sum_defrag: false`（没有人选过它），集群共享算例库里的 yaml 也是旧的，不忽略它开关的默认值就不会生效。A/B 对照用 `defrag_order: atomic`。|
| `utils/geometry/_demo_cavity_case{,_3d,_aligned}.py` | 生成器改写 `defrag_order: prefix`。已入库的 case.yaml 不动（算例库比对要求逐字节相同）。|
| `docs/cluster_v6/scripts/run_chain_v6.py` | `[e30-config]` 行加 `defrag_order=`。|
| `experiment/v7/shaders/**` | **不动**：`defrag.comp` 的 prefix 分支本来就在，46 是特化常量，`defrag.comp.spv` 等全部 SPIR-V 与 E8 逐字节相同（部署脚本逐个 `cmp`）。|

v7-single 分支：同一份 `defrag_prefix.py` 复制进它的 `experiment/v7/utils/`，它的 rc1 版 `case_v7.py` / `case_loader_v7.py` / `simulator_v7.py` 做同样三处改动（host 端 python，SPIR-V 目录不动，MANIFEST 校验照旧），`experiment/v7_single/utils/simulator_single.py`：`record_defrag` prefix 时不填计数器、`submit_defrag_and_wait` 排空 → scan → 提交、`submit_step(with_defrag=True)`（不排空的合并提交）在 prefix 下改为"步 → 排空 → scan → defrag"（scan 要的是 defrag 前那一步的 voxel 计数）。K = 1 用到的 SPIR-V 仍与链路逐字节相同。

diff 规模（e8-3d，experiment/v7 + 生成器 + harness 行）： 7 files changed, 50 insertions(+), 11 deletions(-)。

## 2. 本机验证（GPU 1 = 无显示的 5090，575 W，e8 工作树的 E9 代码；脚本 `docs/cluster_v6/scripts/e9_local/`，数据 `docs/cluster_v6/data/e9/local/`，日志镜像 `logs/e8_local/e9_local_20261027/`）


##### item 4: K = 1, 300 steps (window 100-299), pid-order audit after the bootstrap defrag

| run | defrag order | step (ms) | ns / particle | corr+dens (ms) | force (ms) | 7.18 prefix step (ms) | diff | audit (bootstrap defrag) | drift |
|---|---|---|---|---|---|---|---|---|---|
| e9_8m_k1_audit | prefix | 46.250 | 5.781 | 23.040 | 22.147 | 46.325 | -0.16 % | 125,000 blocks = 125,000 voxels, 0 descents, 0 without voxel -> EXACT | 0 |
| e9_64m_k1_audit | prefix | 375.400 | 5.866 | 183.729 | 183.092 | 375.500 | -0.03 % | 1,000,000 blocks = 1,000,000 voxels, 0 descents, 0 without voxel -> EXACT | 0 |
| e9_8m_k1 | prefix | 46.400 | 5.800 | 23.107 | 22.230 | 46.325 | +0.16 % | - | 0 |
| e9_64m_k1 | prefix | 375.300 | 5.864 | 183.632 | 183.042 | 375.500 | -0.05 % | - | 0 |

##### item 5: K = 2 on one card (8M cube, --device-map 1,1), 300 steps

- `e9_8m_k2_trace`: 55.500 ms/step (18.02 fps, two slabs on one card), drift 0, overflow_total 0, seam check OK; audits (bootstrap defrag, both slabs, then the end-of-run defrag): EXACT (4,000,000 pids, 0 descents), EXACT (4,000,000 pids, 0 descents), EXACT (4,000,000 pids, 0 descents), EXACT (4,000,000 pids, 0 descents)
  - [chain_v7] seam 0 (col 26): L_overshoot=-0.19dx R_overshoot=-0.46dx dup=0 OK

field comparison K = 2 vs K = 1 at step 300 (`e8_field_compare.py`, phase-1 format):

K = 2 (cuts [26], devices [1, 1]) vs K = 1 at step 300: 7,077,888 fluid particles compared (ids missing in the test 0, in the reference 0); max |dv|/U = 3.943e-05, rms 2.480e-06; max |drho|/rho0 = 1.831e-07, rms 2.052e-08; max |dx| 1.156e-05 m

| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |
|---|---|---|---|---|---|---|
| 0 | 294,912 | 1.941e-05 | 4.317e-06 | 1.221e-07 | 2.693e-08 | 2.772e-06 |
| 1 | 294,912 | 1.840e-05 | 4.174e-06 | 1.221e-07 | 2.757e-08 | 2.625e-06 |
| 2 | 294,912 | 1.873e-05 | 3.511e-06 | 1.221e-07 | 2.490e-08 | 2.594e-06 |
| 3 | 294,912 | 1.648e-05 | 2.767e-06 | 1.221e-07 | 2.228e-08 | 2.209e-06 |
| 4 | 294,912 | 1.665e-05 | 2.290e-06 | 1.221e-07 | 2.069e-08 | 2.623e-06 |
| 5 | 294,912 | 1.408e-05 | 2.092e-06 | 1.221e-07 | 2.016e-08 | 2.385e-06 |
| 6 | 294,912 | 1.276e-05 | 1.967e-06 | 1.831e-07 | 1.970e-08 | 2.297e-06 |
| 7 | 294,912 | 1.755e-05 | 1.856e-06 | 1.221e-07 | 1.933e-08 | 3.398e-06 |
| 8 | 294,912 | 2.032e-05 | 1.851e-06 | 1.221e-07 | 1.913e-08 | 3.249e-06 |
| 9 | 294,912 | 2.013e-05 | 1.762e-06 | 1.221e-07 | 1.895e-08 | 3.428e-06 |
| 10 | 294,912 | 2.054e-05 | 1.744e-06 | 1.831e-07 | 1.909e-08 | 2.777e-06 |
| 11 | 294,912 | 1.438e-05 | 1.706e-06 | 1.831e-07 | 1.923e-08 | 4.831e-06 |
| 12 | 294,912 | 2.348e-05 | 1.762e-06 | 1.221e-07 | 1.910e-08 | 4.503e-06 |
| 13 | 294,912 | 2.563e-05 | 1.825e-06 | 1.221e-07 | 1.884e-08 | 4.232e-06 |
| 14 | 294,912 | 2.349e-05 | 1.990e-06 | 1.221e-07 | 1.894e-08 | 4.984e-06 |
| 15 | 294,912 | 3.288e-05 | 2.165e-06 | 1.831e-07 | 1.937e-08 | 5.254e-06 |
| interior | 2,359,296 | 3.943e-05 | 2.426e-06 | 1.831e-07 | 1.934e-08 | 1.156e-05 |

| slab | device | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 |
|---|---|---|---|---|---|---|
| 0 | 1 | 3,538,944 | 2.269e-05 | 2.127e-06 | 1.831e-07 | 1.839e-08 |
| 1 | 1 | 3,538,944 | 3.943e-05 | 2.789e-06 | 1.831e-07 | 2.245e-08 |

- `e9_8m_k2_cad100` (defrag cadence 100, periodic defrags at frames 100 and 200 inside the run): step 70.450 ms, drift 0, seam check OK, defrag events [('100', '1088.521'), ('200', '1082.605'), ('300', '1081.697')] (frame, wall ms); audits ['EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT', 'EXACT']

| window (steps) | median step period (ms) |
|---|---|
| 50-99 | 54.799 |
| 101-149 | 55.067 |
| 150-199 | 54.818 |
| 201-249 | 54.378 |
| 250-299 | 55.109 |
| first step after each defrag | f100+1: 54.866, f200+1: 54.916, f300+1: - |
- before the first periodic defrag (50-99) 54.799 ms, after the second (201-249) 54.378 ms: -0.77 % -> no jump

##### item 6: 2-D sensitivity, K = 1, 1300 steps (window 300-1299, one periodic defrag at 1000 inside; fps = steady_fps_nodefrag)

| case | particles | atomic (ms/step) | prefix (ms/step) | prefix - atomic | atomic audit | prefix audit |
|---|---|---|---|---|---|---|
| cavity2d_n1000 (1M) | 1,044,484 | 1.3621 | 1.3509 | **-0.82 %** | 642 descents | 0 descents |
| cavity2d_n2840 (8M) | 8,191,044 | 10.8929 | 10.2980 | **-5.46 %** | 5,015 descents | 0 descents |

### 2.1 第 4 项判读

两个立方体的每粒子成本与 7.18 的 prefix monkeypatch 结果一致（8M 46.25 对 46.33 ms，64M 375.4 对 375.5 ms，差 ≤ 0.16 %），说明进树的路径就是 7.18 量过的那条；每次 defrag 后 pid 序严格等于 voxel 序（0 次块间倒序，块数 = 非空 voxel 数）。host scan 本身：8M 约 4 ms、64M 约 27 ms（启动一次 + 每 1000 帧一次，不在计时窗口内）；日志里 defrag 事件的墙钟 1.08 s 是 `V7_DEFRAG_AUDIT=1` 的读回（每 slab 0.5 s），生产运行不开。

### 2.2 第 5 项判读

K = 2 两个 slab 在同一张卡上（`--device-map 1,1`）：内建检查全过（drift 0、overflow 0、seam 0 (col 26) dup=0 OK、两个 slab 每次 defrag 后都是精确 voxel 序，含三次周期 defrag）；第 300 步场对 K = 1：max |dv|/U 3.9e-5、|dρ|/ρ0 1.8e-7（阶段 1 两张卡上的 K = 2 是 5.7e-5 / 1.8e-7），逐列分布同样平坦，seam 列不突出。defrag 间隔 100：步长周期在 f100、f200 前后 54.4–55.1 ms，defrag 后第一步 54.9 ms，无跳变（−0.77 % 在逐窗口的散布内）。

### 2.3 第 6 项判读：2-D 敏感性

| 算例 | atomic (ms/步) | prefix (ms/步) | 差 | atomic 的 pid 序 |
|---|---|---|---|---|
| cavity2d_n1000（1M，1,044,484 粒子） | 1.3621 | 1.3509 | **-0.82 %** | 642 / 42,436 块倒序 |
| cavity2d_n2840（8M，8,191,044 粒子） | 10.8929 | 10.2980 | **-5.46 %** | 5,015 / 329,476 块倒序 |

（各一次，1300 步取 300–1299，窗口含第 1000 步的周期 defrag，fps 用 nodefrag 值。）1M 在门槛内，**8M 不在：prefix 快 5.5 %**。按你的规则"小于 0.5 % 就不重跑 E7"，8M 的 2-D 点超过门槛——E7 的 2-D 批次要不要在 v7-rc4 上重跑由你定；我没有排 E7 的作业。机制与 3-D 相同：2-D 8M 的 defrag 派发 2,574 个工作组，第一波覆盖大半网格。

## 3. 第 7 项：Section 3 基线重测（工作站 600 W，v7_single，1 在飞，逐核时间戳）

待做（需要把 GPU 1 的功率上限恢复到 600 W：管理员 `nvidia-smi -i 1 -pl 600`，UAC 已弹出等你点）。

## 4. 集群（第 8–11 项）

待做：标签推送 → 部署 `~/run/vulkan-demo-e9`（v7-rc4）与 `~/run/vulkan-demo-v7single-rc4` → 两卡短作业 `e9_p1`（8M K1 + 审计、8M K2 trace + 场比较、64M K2，300 步）→ 全过后正式批次 e9_B / e9_A / e9_C / e9_D（`e7_plan.py` 修订 18 / 19 / 20 = E8 修订 11 / 12 / 14 的矩阵与协议，排除 nbm13g5、nah09g1、nbj04g3、nbj04g5；单卡对照 v7-single-rc4，Sec. 5.1 的 cadence 协议）。每个作业提交与结束各报一次。

## 5. 数据与复现

- 代码：提交号与标签见本节末尾的更新。
- 本机：`docs/cluster_v6/scripts/e9_local/run_k.sh`（一次 K 槽链路运行）、`local_batch.sh`（第 4–6 项的序列）、`local_summary.py` → `docs/cluster_v6/data/e9/local/local_validation.{md,json}`、`compare/compare_8m.{md,json}`；`make_atomic_case.py`（A/B 的 atomic 变体 yaml）；树的改动由 `patch_tree_prefix.py` / `patch_tree_audit.py` 施加（两个工作树同一份），`patch_plan_e9.py`（修订 18–20），`make_deploy_scripts.py`（E9 部署脚本）。
