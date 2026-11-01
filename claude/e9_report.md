# E9 报告：prefix defrag 进树（v7-rc4 / v7-single-rc4）与 3-D 批次重跑（N56 hp_5090，2026-10-27 起）

> E8 的目录、数据与报告（`claude/e8_3d_report.md`，`docs/cluster_v6/data/e8/`）原样保留；E9 的数据在 `docs/cluster_v6/data/e9/`，作业脚本在 `docs/cluster_v6/e9_jobs/`，本报告按你 10-27 的清单逐项写。E8 7.18 的结论是起点：默认的 atomicAdd defrag 把 pid 块按到达序摆放，8M 的 slab 每粒子多付 12 %，64M 多付 2 %；prefix 序消掉这项代价。

## 0. 概要（随作业完成更新）

- 代码（第 1 节）：host 端 exclusive scan 成为 `experiment/v7` 的 defrag 正式路径并默认开启；v7_single 用同一份 scan；核函数与 SPIR-V 不变。
- 本机验证（第 2 节）：第 4–6 项**全部通过**。K = 1 的每粒子成本与 7.18 的 prefix 结果相差 −0.16 % / −0.03 %（8M / 64M），每次 defrag 后的 pid 序审计 0 次块间倒序；K = 2（同一张卡）内建检查全过、seam OK、第 300 步场与 K = 1 的差在阶段 1 的噪声水平；defrag 间隔 100 的运行三次周期 defrag 后步长无跳变。**2-D 敏感性：1M -0.82 %，8M -5.46 %**——8M 超过 0.5 % 的门槛；你决定 E7 不重跑，2-D 数据保持现状（第 2.3 节）。
- 第 7 项（第 3 节）：表 2 重测完成——2-D 8M 步长 10502 → 9969 µs（-5.1 %），3-D 9.1M 61060 → 58414 µs（-4.3 %），两个 sweep 核 −4 ~ −6 %，其余核不变；fig:baseline 的 2M–32M 点已按测量 A 重测并重画（S d2 +1.2 ~ +5.0 %），1M 不受影响。
- 第 8–11 项（集群，第 4 节）：短作业 e9_p1 全过；正式批次 e9_A/B/C/D 2026-10-28 17:37–21:40 CST 全部 COMPLETED（333 次运行全 pass，drift 0，无丢卡），E8 的全部交付在 `data/e9/` 下重做，逐点 η 表见 4.5：每卡 ≥ 4M 的点 η +0.02 ~ +0.06（64M K8 0.841 → 0.878，128M 0.868 → 0.928，64M K4 0.901 → 0.951），≤ 2M 每卡与 weak 不变；机制 = 参考工作点回到 ±1.6 %（4.4.4），C 段不加速（4.4.3）。

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

（各一次，1300 步取 300–1299，窗口含第 1000 步的周期 defrag，fps 用 nodefrag 值。）1M 在门槛内，**8M 不在：prefix 快 5.5 %**。按你的规则"小于 0.5 % 就不重跑 E7"，8M 的 2-D 点超过门槛——你 10-27 15:00 决定：**E7 不重跑，2-D 数据保持现状**（"2d数据没什么问题"）。机制与 3-D 相同：2-D 8M 的 defrag 派发 2,574 个工作组，第一波覆盖大半网格。

## 3. 第 7 项：Section 3 基线重测（工作站 600 W，v7_single，1 在飞，逐核时间戳；`tab:kernelcost` 的原协议）

**协议**：`experiment/v7_single/_run_section3_campaign.py run`（= 2026-10-10 做表 2 时的同一脚本：`_run_kernel_breakdown.py --solver v7_single --interleave 0`，预热 1000 步后第 1001–3000 步全部计时，每步一次提交 + fence，生产录制里的 tick 写 BOTTOM_OF_PIPE 时间戳，defrag 帧剔除；2-D 8M `lid_driven_cavity_2d_8m`（8,128,201 粒子）与 3-D 9.1M `cavity3d_8m_b4`（9,129,329）各 3 次独立运行，其余尺寸各 1 次；CPU 端候选数 / 邻居数统计同样重做）。GPU 1 功率上限恢复为 600 W（你点的 UAC），v7-single 提交 ffd1a2f（= rc4.1 的 solver，外加 kernel-breakdown 驱动的 scan 调用——它自己提交 defrag 命令，第一次跑在 voxel 列表一致性检查上失败，修后重跑）。结果 `docs/cluster_v6/data/e9/section3/`（v7-single 分支镜像 `docs/seam_audit/v7_single/section3_rc4_*`）。

#### Section 3 Table 2 (tab:kernelcost): old (atomic defrag order, 2026-10-10, commit f06af58) vs new (prefix order, v7-single-rc4 + ffd1a2f, 2026-10-27), 600 W, 3 runs each, frames 1001-3000, 1 in flight, defrag frames excluded

| case | kernel | old us/step (std) | new us/step (std) | change | > 0.5 %? | old ns/particle | new ns/particle | old % | new % |
|---|---|---|---|---|---|---|---|---|---|
| 2-D 8M (8,128,201) | predict | 497 (0) | 497 (0) | **+0.02 %** | no | 0.06 | 0.06 | 4.7 | 5.0 |
| 2-D 8M (8,128,201) | update_voxel | 331 (0) | 331 (0) | **+0.01 %** | no | 0.04 | 0.04 | 3.2 | 3.3 |
| 2-D 8M (8,128,201) | correction_density | 4530 (9) | 4267 (12) | **-5.81 %** | yes | 0.56 | 0.52 | 43.1 | 42.8 |
| 2-D 8M (8,128,201) | density_copy | 82 (0) | 82 (0) | **+0.06 %** | no | 0.01 | 0.01 | 0.8 | 0.8 |
| 2-D 8M (8,128,201) | force | 5062 (6) | 4792 (11) | **-5.34 %** | yes | 0.62 | 0.59 | 48.2 | 48.1 |
| 2-D 8M (8,128,201) | step total | 10502 (15) | 9969 (23) | **-5.08 %** | yes | 1.29 | 1.23 | 100.0 | 100.0 |
| 2-D 8M (8,128,201) | GPU throughput (M particle-steps/s) | 773.9 | 815.4 | **+5.35 %** | | ps/candidate (step) 5.76 | 5.47 | | |
| 3-D 8M (9,129,329) | predict | 500 (0) | 504 (0) | **+0.98 %** | yes | 0.05 | 0.06 | 0.8 | 0.9 |
| 3-D 8M (9,129,329) | update_voxel | 316 (0) | 313 (0) | **-0.84 %** | yes | 0.03 | 0.03 | 0.5 | 0.5 |
| 3-D 8M (9,129,329) | correction_density | 30354 (20) | 28934 (20) | **-4.68 %** | yes | 3.32 | 3.17 | 49.7 | 49.5 |
| 3-D 8M (9,129,329) | density_copy | 78 (0) | 93 (0) | **+19.00 %** | yes | 0.01 | 0.01 | 0.1 | 0.2 |
| 3-D 8M (9,129,329) | force | 29813 (23) | 28569 (25) | **-4.17 %** | yes | 3.27 | 3.13 | 48.8 | 48.9 |
| 3-D 8M (9,129,329) | step total | 61060 (43) | 58414 (44) | **-4.33 %** | yes | 6.69 | 6.40 | 100.0 | 100.0 |
| 3-D 8M (9,129,329) | GPU throughput (M particle-steps/s) | 149.5 | 156.3 | **+4.53 %** | | ps/candidate (step) 4.06 | 3.88 | | |

#### per-size GPU step of the single runs (frames 1001-3000, defrag frames excluded): fig:baseline's affected points

| size | particles | old GPU step (us) | new (us) | change | affected (> 0.5 %)? | new SM MHz / W / PCIe |
|---|---|---|---|---|---|---|
| 1m | 1,046,529 | 1314 | 1318 | **+0.27 %** | no | 2846.0 / 596.4749999999999 / Gen 4 x4 |
| 2m | 2,064,969 | 2599 | 2493 | **-4.08 %** | yes | 2803.5 / 599.4733333333334 / Gen 4 x4 |
| 4m | 4,182,025 | 5664 | 5364 | **-5.29 %** | yes | 2799.818181818182 / 599.8981818181818 / Gen 4 x4 |
| 4m_gen | 4,182,025 | 5662 | 5360 | **-5.34 %** | yes | 2789.8333333333335 / 600.27 / Gen 4 x4 |
| 8m | 8,128,201 | 10502 | 9969 | **-5.08 %** | yes | 2794.9 / 599.515 / Gen 4 x4 |
| 16m | 16,184,529 | 20962 | 20379 | **-2.78 %** | yes | 2779.7073170731705 / 599.1578048780489 / Gen 4 x4 |
| 32m | 32,251,041 | 40655 | 40062 | **-1.46 %** | yes | 2769.246913580247 / 598.9523456790123 / Gen 4 x4 |
| 3d_8m | 9,129,329 | 61060 | 58414 | **-4.33 %** | yes | 2808.051282051282 / 599.9686324786325 / Gen 4 x4 |

**判读**：
- 两个 sweep 核各快 4–6 %，predict / update_voxel / density_copy 的绝对值不变（3-D 的 density_copy +15 µs、predict +4 µs 是 1 µs 量级的散布放大成的百分比，按 0.5 % 规则列为"更新"但不改变任何结论）。2-D 8M 的整步 −5.1 %，与本机第 6 项的 −5.5 %（575 W，链路代码）一致；3-D 9.1M −4.3 %。
- 每候选对的皮秒数随之下降（2-D 5.76 → 5.47，3-D 4.06 → 3.88）：候选数、邻居数、占用体素完全不变（同一物理），少掉的是 atomicAdd 排列带来的访存代价。
- 1M 不动（+0.3 %）：42k 个 voxel 只有 331 个 defrag 工作组，第一波乱序的代价在 1M 上本来就小（本机第 6 项 −0.8 %）。
- 4M 凹陷仍在：4M gen 的每候选皮秒数对 2M / 8M 均值 +5.2 %（旧 +6.1 %），prefix 序不是它的原因。

**tab:kernelcost 的新行**（表 2 全表见 `section3_rc4_table2.md`；LaTeX 行如下，与旧表相差 > 0.5 % 的项都在其中）：

```latex
% columns: kernel & dispatch & 8M $\mu$s/step & ns/particle & \% & 3-D 8M $\mu$s/step & ns/particle & \% \\
% 8M: 8,128,201 particles (8,003,241 fluid), h/dx = 4.99, 224 candidates and 71.0 neighbours per particle; 3-D 8M: 9,129,329 particles (8,120,601 fluid), h/dx = 4, 1649 candidates and 259.4 neighbours per particle
% 8M: 3 runs, step spread 0.23 %; 3-D 8M: 3 runs, step spread 0.08 %
predict & per particle & 497 & 0.06 & 5.0 & 504 & 0.06 & 0.9 \\
update\_voxel & per voxel & 331 & 0.04 & 3.3 & 313 & 0.03 & 0.5 \\
correction + density (fused) & per particle & 4267 & 0.52 & 42.8 & 28934 & 3.17 & 49.5 \\
density (scratch$\to$primary copy) & buffer copy & 82 & 0.01 & 0.8 & 93 & 0.01 & 0.2 \\
force & per particle & 4792 & 0.59 & 48.1 & 28569 & 3.13 & 48.9 \\
\midrule
\textbf{step total} &  & 9969 & 1.23 & 100.0 & 58414 & 6.40 & 100.0 \\
% ps per candidate: 8M correction_density 2.34, force 2.63; 3-D 8M correction_density 1.92, force 1.90
```

### 3.1 fig:baseline 受影响的点

各尺寸 GPU 步长超过 0.5 % 的是 2M、4M、8M、16M、32M 与 3-D 8M；1M 不受影响。fig:baseline 是 2-D 曲线（v7_single 的 S 系列），所以按测量 A 的协议重测了 2M、4M gen、8M、16M、32M（`_run_single_baseline_campaign.py --sizes 2m,4m_gen,8m,16m,32m --configs v7_single:1,v7_single:2,v7_chain:2 --trials 3 --warmup 1000`，窗口 6000 / 4000 / 2000 / 2000 / 2000 步，600 W，45 次运行全部 drift 0），并入 r3 为 `throughput_rc4_merged`（1M、旧 4M、3-D 8M 与 V / Vx 列保留旧值——V / Vx 是 atomic 序的旧运行，它们与新 S 的比值已无意义，图不用它们）。图 `manuscripts/fig/single_gpu_throughput.{pdf,png}` 已重画（v7-single 工作树与 vulkan-demo 主库各一份），旧图改名 `single_gpu_throughput_e8_atomic.*`。

**吞吐（fps，均值 ± 标准差，3 次）**

| size | particles | S d1 | S d2 | V d1 | V d2 | Vx d1 | Vx d2 | C d2 |
|---|---|---|---|---|---|---|---|---|
| 1m | 1,046,529 | 705.24 ± 0.20 (n=3) | 734.60 ± 0.08 (n=3) | 698.86 ± 1.10 (n=3) | 733.32 ± 0.31 (n=3) | 699.89 ± 1.71 (n=3) | 732.97 ± 0.35 (n=3) | 723.94 ± 0.78 (n=3) |
| 2m | 2,064,969 | 371.08 ± 2.58 (n=3) | 385.72 ± 0.52 (n=3) | 361.66 ± 0.16 (n=3) | 371.69 ± 0.20 (n=3) | 361.11 ± 0.32 (n=3) | 371.45 ± 0.32 (n=3) | 382.05 ± 0.29 (n=3) |
| 4m | 4,182,025 | 172.56 ± 0.02 (n=3) | 174.97 ± 0.18 (n=3) | 172.37 ± 0.09 (n=3) | 175.36 ± 0.03 (n=3) | 172.40 ± 0.09 (n=3) | 175.25 ± 0.08 (n=3) | 174.86 ± 0.13 (n=3) |
| 4m_gen | 4,182,025 | 179.28 ± 1.01 (n=3) | 183.81 ± 0.18 (n=3) | n/a | n/a | n/a | n/a | 183.27 ± 0.12 (n=3) |
| 8m | 8,128,201 | 97.71 ± 0.52 (n=3) | 99.17 ± 0.08 (n=3) | 93.53 ± 0.06 (n=3) | 94.57 ± 0.07 (n=3) | 93.59 ± 0.05 (n=3) | 94.54 ± 0.05 (n=3) | 99.07 ± 0.05 (n=3) |
| 16m | 16,184,529 | 48.18 ± 0.23 (n=3) | 48.65 ± 0.03 (n=3) | 47.29 ± 0.01 (n=3) | 47.58 ± 0.02 (n=3) | 47.28 ± 0.01 (n=3) | 47.57 ± 0.01 (n=3) | 48.67 ± 0.05 (n=3) |
| 32m | 32,251,041 | 24.65 ± 0.01 (n=3) | 24.79 ± 0.01 (n=3) | 24.40 ± 0.00 (n=3) | 24.50 ± 0.01 (n=3) | 24.41 ± 0.00 (n=3) | 24.49 ± 0.00 (n=3) | 24.78 ± 0.00 (n=3) |
| 3d_8m | 9,129,329 | 16.30 ± 0.02 (n=3) | 16.34 ± 0.01 (n=3) | 16.28 ± 0.01 (n=3) | 16.33 ± 0.04 (n=3) | 16.28 ± 0.02 (n=3) | 16.34 ± 0.02 (n=3) | 16.32 ± 0.03 (n=3) |

S d2 的变化：2M 371.9 → 385.7（+3.7 %），4M gen 175.1 → 183.8（+5.0 %），8M 94.4 → 99.2（+5.0 %），16M 47.5 → 48.7（+2.5 %），32M 24.5 → 24.8（+1.2 %）；C/S（链路 K = 1 对单卡）在重测点上仍是 0.99–1.00。3-D 8M 的吞吐行没有重测（图里没有它；表 2 的 3-D 值已重测）。

![single_gpu_throughput](../docs/cluster_v6/data/e9/section3/single_gpu_throughput.png)

**协变量**：这次 nvidia-smi 报 GPU 1 的 PCIe 为 Gen 4 x4（10-10 为 Gen 5 x8）——你热插拔换 7900 XTX 再换回来后链路重新协商的结果。单卡核时间不依赖它（1M 点与旧值一致），本节的数字不受影响；本机的传输类测量在重启前不要做。

### 3.2 fig:baseline 补测与重画（用户 2026-10-29 晚；v7-single-rc4.1 = 39668e2 的 solver，600 W，GPU 1）

**补测**（`_run_single_baseline_campaign.py`，测量 A 协议：3 次交错试验、预热 1000 步、窗口 1M 10000 步 / 4M 4000 步、每步一次提交或两在飞；`logs/v7_single/throughput_rc4c`，9 次运行全部 drift 0）：
- 1M（`lid_driven_cavity_2d_gen`，1,046,529 粒子）两在飞：**733.54 ± 0.26 fps = 767.7 M 粒子步/s**（旧 atomic 序 734.60 ± 0.08 / 768.8；−0.14 %，与 3.1 的"1M 不受影响"一致）。
- 4M gen 414² 变体（`lid_driven_cavity_2d_4m_gen_n414`，4,272,489 粒子，网格 414 × 414，第 8 项 stride 研究的算例）：一在飞 **187.19 ± 0.07 fps = 799.8 M**，两在飞 **190.74 ± 0.11 fps = 814.9 M**；对 410² 的 4M gen（749.8 / 768.7 M）高 +6.7 / +6.0 %，即 4M 凹陷在 prefix 序下仍是 410² 网格特有的（stride 研究的旧测：414² 773.8 对 410² 731.9 M，+5.7 %）。
- 并表：新脚本 `_merge_throughput_results.py`（base r3 + rc4b + rc4c，按 (size, solver, in_flight) 整键替换）重建 `logs/v7_single/throughput_rc4_merged`（162 次运行，54 键），`_summarize_v7_single_campaign.py` 重出 summary，拷到 `docs/cluster_v6/data/e9/section3/throughput_rc4_summary.{md,json,csv}` 与 v7-single 的 `docs/seam_audit/v7_single/`（+ `throughput_rc4c_results.jsonl`、`throughput_rc4_merged_MERGE.txt`）。表里仍是旧 atomic 运行的：1M S d1、手工 4M、3-D 8M、V / Vx 列。

**重画**（`_plot_single_baseline_paper.py --v2`，新函数 `figure_throughput_v2`）：单面板 3.5 × 2.33 in（3:2），横轴粒子数对数、刻度 1M–32M，纵轴 700–820 M 线性；两在飞实心圆实线、一在飞空心圆虚线，颜色 COLOR_V0 `#2a78d6`（fig:reference 同色）；每点 3 次试验 min–max 误差棒；414 变体两个空心方块画在它自己的粒子数 4.27M 处（对数轴上与 4M 刻度重合），不连线；图例右下三项。输出 `manuscripts/fig/single_gpu_throughput.{pdf,png,csv}`（主库与 v7-single 工作树各一份，副本在 `docs/cluster_v6/data/e9/section3/`），旧两面板版另存 `single_gpu_throughput_two_panel_v1.*`（三处）。

曲线值（M 粒子步/s，rc4.1）：两在飞 767.7 / 796.5 / 768.7 / 806.1 / 787.4 / 799.6（1M → 32M），一在飞 738.1 / 766.3 / 749.8 / 794.2 / 779.7 / 795.0；正文的"732–790 M"改为 **768–806 M**（两在飞）。

**2026-10-30 修订**：按用户要求去掉图上 414 变体的两个空心方块和图例第三项，其余不动（两条线、min–max 误差棒、颜色、3.5 × 2.33 in、纵轴 700–820）；CSV 里 414 的两行保留，series 改为 `grid414`（`figure_throughput_v2` 的 `draw_variant` 默认关，`--draw-variant` 可画回 10-29 的版本）。`manuscripts/fig/single_gpu_throughput.{pdf,png,csv}` 已覆盖，v7-single 工作树与 `data/e9/section3/` 的副本同步（三处 md5 相同）。

## 4. 集群（第 8–11 项）

### 4.1 部署

- 包：`vd_e9_cb3c9f5c….tar.gz`（342 文件，`git archive v7-rc4`，内容 = E8 的清单 + `docs/cluster_v6/e9_jobs`）与 `vd_v7single_….tar.gz`（v7-single 的 `experiment/{v7,v7_single}` + materials）。
- `~/run/vulkan-demo-e9`（11:53–13:31 CST，5921 s：JuiceFS 今天建文件慢，280 个文件拷了 1002 s）：MANIFEST / SCRIPTS 校验、experiment/v7 树 37f99f2b、defrag 默认 prefix、五个 SPIR-V 与 E8 部署逐字节相同、py_compile、`_test_chain_cuts` ALL PASS、11 个算例生成器核对、五个作业脚本的语法与空跑（e9_A 46 次链路运行 / e9_D 30 次 / e9_p1 3 次完整空跑通过；e9_B、e9_C 的空跑被部署脚本 15 分钟上限截断——登录节点上仅启动检查就 14 分钟——不是失败，链结束后用 90 分钟上限重跑中），`sbatch --test-only` 接受。
- `~/run/vulkan-demo-v7single-rc4`：第一次部署的是 969823a（v7-single-rc4，有第 1 节说的死锁），在 e9_p1 结束、队列为空后用 **786e2d4（v7-single-rc4.1）`UPDATE=1` 重新部署**（树 4a8fc71f，SPIR-V 与 e9 逐字节相同）。正式批次在它 OK 之后才提交。

### 4.2 第 8 项：两卡短作业 e9_p1 = SLURM 1741843（13:33:58–13:40:58 CST，wqd10nbj02g1，GPU 16:00.0 / 27:00.0，两卡 575 W，驱动 580.82.07，7 min）

镜像 `C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56/2026-10-27/01_e9_p1_1741843/`（README / jobs.csv 已加行），数据 `docs/cluster_v6/data/e9/p1/`。PROVENANCE：harness cb3c9f5、experiment/v7 树 37f99f2b，作业末尾 PROVENANCE_END unchanged。bring-up（2-D 1M K1 675 fps、K2 898 fps）通过。

**运行（RESULT 行）**

| label | status | K | steps | steady fps | wall s | build s | alive start→end | drift / overflow / far / stamps / clamp | seam / NaN | own columns | own particles | device-local + scratch MiB | VRAM peak MiB per card | SM MHz (median / min) | W median | °C max | ns / particle-step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8m_K1 | pass (rc 0) | 1 | 300 | 22.5 | 24.2 | 9.10 | 8000000→8000000 | 0 / 0 / 0 / 0+0 / 0 | - / None | [52] | [8000000] | 2,641 | 0:2,698 | 2827/180 / 2820/172 | 575/17 | 51/24 | 5.56 |
| 8m_K2_trace | pass (rc 0) | 2 | 300 | 42.8 | 26.4 | 11.90 | 8000000→8000000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 26, 'left_overshoot_dx': -0.19, 'right_overshoot_dx': -0.46, 'duplicates': 0, 'ok': True} / False | [26, 26] | [4000000, 4000000] | 1,512/1,512 | 0:1,569/1:1,553 | 2835/2865 / 2835/2850 | 569/530 | 52/46 | 5.84 |
| 64m_K2 | pass (rc 0) | 2 | 300 | 5.5 | 146.4 | 54.90 | 64000000→64000000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 51, 'left_overshoot_dx': -0.02, 'right_overshoot_dx': -0.49, 'duplicates': 0, 'ok': True} / False | [51, 51] | [32000000, 32000000] | 11,454/11,454 | 0:11,511/1:11,495 | 2775/2827 / 2745/2797 | 575/575 | 63/64 | 5.68 |

- 每粒子成本 8M K1：**5.56 ns**（E8 阶段 1 同样两卡协议 6.28 ns，−11.5 %；本机 575 W 5.78 ns）；8M K2 trace 42.8 fps（E8 38.99，+9.8 %）；64M K2 5.5 fps（E8 5.29，+4 %）。三次运行的 defrag 审计（8M K1、8M K2 两 slab、启动与结束各一次）全部 EXACT voxel order，`defrag_order=prefix` 出现在每个 sim 的 init 行。
- seam：K2 的 seam 0（col 26 / col 51）dup=0 OK；drift / overflow / far_migration / stamps / clamp 全 0。

**显存：实测峰值对模型**

| run | card | slab own columns | model device-local + scratch MiB | measured | model VRAM | measured peak | diff |
|---|---|---|---|---|---|---|---|
| 8m_K1 | 0 | 52 | 2,616 | 2,641 | 2,666 | 2,698 | +1.2 % |
| 8m_K2_trace | 0 | 26 | 1,512 | 1,512 | 1,562 | 1,569 | +0.4 % |
| 8m_K2_trace | 1 | 26 | 1,512 | 1,512 | 1,562 | 1,553 | -0.6 % |
| 64m_K2 | 0 | 51 | 11,459 | 11,454 | 11,509 | 11,511 | +0.0 % |
| 64m_K2 | 1 | 51 | 11,459 | 11,454 | 11,509 | 11,495 | -0.1 % |

**场比较 K = 2 对 K = 1（第 300 步，dump_state 按全局粒子 id 对齐，格式同阶段 1）**：

K = 2 (cuts [26], devices [0, 1]) vs K = 1 at step 300: 7,077,888 fluid particles compared (ids missing in the test 0, in the reference 0); max |dv|/U = 4.517e-05, rms 2.457e-06; max |drho|/rho0 = 1.831e-07, rms 2.046e-08; max |dx| 1.003e-05 m

| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |
|---|---|---|---|---|---|---|
| 0 | 294,912 | 1.896e-05 | 4.353e-06 | 1.221e-07 | 2.723e-08 | 3.576e-06 |
| 1 | 294,912 | 2.064e-05 | 4.283e-06 | 1.221e-07 | 2.787e-08 | 2.980e-06 |
| 2 | 294,912 | 2.021e-05 | 3.547e-06 | 1.831e-07 | 2.508e-08 | 2.801e-06 |
| 3 | 294,912 | 1.804e-05 | 2.779e-06 | 1.221e-07 | 2.230e-08 | 2.742e-06 |
| 4 | 294,912 | 1.736e-05 | 2.293e-06 | 1.221e-07 | 2.075e-08 | 3.368e-06 |
| 5 | 294,912 | 1.496e-05 | 2.122e-06 | 1.221e-07 | 2.034e-08 | 3.219e-06 |
| 6 | 294,912 | 1.366e-05 | 2.010e-06 | 1.221e-07 | 2.002e-08 | 2.593e-06 |
| 7 | 294,912 | 2.003e-05 | 1.875e-06 | 1.221e-07 | 1.944e-08 | 3.399e-06 |
| 8 | 294,912 | 1.635e-05 | 1.902e-06 | 1.221e-07 | 1.943e-08 | 3.101e-06 |
| 9 | 294,912 | 1.799e-05 | 1.849e-06 | 1.221e-07 | 1.930e-08 | 3.368e-06 |
| 10 | 294,912 | 1.503e-05 | 1.812e-06 | 1.221e-07 | 1.943e-08 | 2.833e-06 |
| 11 | 294,912 | 1.431e-05 | 1.768e-06 | 1.831e-07 | 1.922e-08 | 3.726e-06 |
| 12 | 294,912 | 2.452e-05 | 1.815e-06 | 1.221e-07 | 1.931e-08 | 3.938e-06 |
| 13 | 294,912 | 2.720e-05 | 1.829e-06 | 1.221e-07 | 1.913e-08 | 4.387e-06 |
| 14 | 294,912 | 3.031e-05 | 1.914e-06 | 1.221e-07 | 1.891e-08 | 4.584e-06 |
| 15 | 294,912 | 3.327e-05 | 2.010e-06 | 1.221e-07 | 1.874e-08 | 5.467e-06 |
| interior | 2,359,296 | 4.517e-05 | 2.296e-06 | 1.221e-07 | 1.884e-08 | 1.003e-05 |

| slab | device | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 |
|---|---|---|---|---|---|---|
| 0 | 0 | 3,538,944 | 2.064e-05 | 2.151e-06 | 1.831e-07 | 1.845e-08 |
| 1 | 1 | 3,538,944 | 4.517e-05 | 2.729e-06 | 1.831e-07 | 2.230e-08 |

（E8 阶段 1 两卡：max |dv|/U 5.7e-5、|dρ|/ρ0 1.8e-7。）**短作业全部检查通过，提交正式批次。**

### 4.3 正式批次：提交（2026-10-27 14:15 CST；你的话："部署好了就直接提交四个作业，报我作业号，然后持续监控"）

| 作业 | SLURM | 内容 | 时限 | 估算 |
|---|---|---|---|---|
| e9_B | **1741917** | B 线：4M / 8M（5 次）、16M / 32M / 64M（3 次）K = 8 + 同步 K = 1 参考，4M / 8M 的 dump K1a / K1b / K8 + 比较，五个立方体的 K = 1 trace，弱 8 × 8M K8，v7-single-rc4.1 对照（Sec. 5.1 cadence 协议，3 对） | 8 h | 3.1 node-h |
| e9_A | **1741918** | A 线：64M K2 / K4、弱 2 / 4（各 3 次，trace 紧跟） | 7 h | 2.6 node-h |
| e9_C | **1741921** | C 线：128M K8，四对同步 K = 2 参考，K8 + 一对 K2 trace | 5 h | 1.5 node-h |
| e9_D（e9_BD） | **1741924** | D 线：76M K8（13 列/卡），K8 + K1 trace | 4 h | 1.3 node-h |

提交参数：`--export=ALL,E30_EXPECT_COMMIT=cb3c9f5cab75,E7_EXPECTED_TREE=37f99f2b…,E30_REPO=$HOME/run/vulkan-demo-e9`；整节点 `--exclusive --mem=0`，575 W，`--exclude=wqd10nbm13g5,wqd10nah09g1,wqd10nbj04g3,wqd10nbj04g5`；E7_RAW / E7_TRACE_DETAIL=full / E7_PARTITION_RECORD / E7_CONTINUE_ON_FAILURE 同 E8。提交时四个都 PENDING（Priority，无空闲整节点；backfill 预估 11-01 是悲观值）。监控：队列状态、运行中输出、丢卡信号（出现则取消 + 排除，不重提）。各作业结束后：打包 → 拉回 → 镜像 `logs/n56/2026-10-27/<seq>_e9_<job>_<id>` + README / jobs.csv → `e8_p2_summary.py --out-dir docs/cluster_v6/data/e9/phase2` → E8 对 E9 的 η 对照（`e9_compare_eta.py`）→ 本报告续写。

### 4.4 结果

四个作业排了 27 h 29 min（分区 36 个节点全部 8/8 被占，别人的作业对我们不可见），2026-10-28 17:37 / 17:38 CST e9_B、e9_A 同时拿到整节点 wqd10nbj04g4 / g2；e9_C、e9_D 因账号并发 GPU 上限（AssocGrpGRES，16 张）等 A / B 结束后自动接上。镜像 `C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56/2026-10-28/<seq>_e9_<job>_<id>/`（README / jobs.csv 已加行），后处理 = E8 的 `e8_p2_summary.py`（10 个工具）输出到 `docs/cluster_v6/data/e9/phase2/`，E8-vs-E9 的 η 对照 `eta_e8_vs_e9.md/json`（`e9_compare_eta.py`）。下面按作业结束顺序补。

#### 4.4.1 A 线：e9_A = SLURM 1741918（wqd10nbj04g2，17:38:38–19:58:47 CST，2 h 20 min；E8 的 e8_A 2 h 18 min）

- PROVENANCE harness cb3c9f5 / experiment/v7 树 37f99f2b，PROVENANCE_END unchanged；8 卡 575 W；bring-up 2-D 1M K1 677 fps / K2 882 fps 通过；52 次运行 + bring-up 全部 pass，drift 0，trace 运行的 fields=OK；协议与 E8 逐项相同（64M K2 / K4 各 3 次 + 同时 K = 1 参考 + trace；weak 2 × 8M K2、4 × 8M K4）。
- 镜像 `logs/n56/2026-10-28/01_e9_e9A_1741918/`（291 文件，npz 不拉）。

| 点 | E8 η（min–max） | E9 η（min–max） | E9 − E8 | η_min E8 / E9 | fps_K E8 / E9 | 参考 fps E8 / E9（整域等效） |
|---|---|---|---|---|---|---|
| 64M K = 2 | 0.968 (0.966–0.970) | **0.987** (0.986–0.987) | **+0.018** | 0.971 / 0.989 | 4.95 / 5.17 (+4.5 %) | 2.56 / 2.62 (+2.6 %) |
| 64M K = 4 | 0.901 (0.898–0.904) | **0.951** (0.948–0.954) | **+0.051** | 0.912 / 0.961 | 9.19 / 9.94 (+8.2 %) | 2.55 / 2.61 (+2.4 %) |
| weak 2 × 8M K = 2 | 0.984 (0.981–0.986) | **0.986** (0.983–0.988) | **+0.002** | 0.986 / 0.987 | 18.46 / 21.09 (+14.2 %) | 9.38 / 10.69 (+13.9 %) |
| weak 4 × 8M K = 4 | 0.981 (0.978–0.986) | **0.992** (0.991–0.992) | **+0.011** | 0.993 / 1.002 | 18.39 / 21.15 (+15.0 %) | 4.69 / 5.33 (+13.7 %) |

**判读**
- 64M K = 4 的 η 从 0.901 升到 **0.951**（+0.05），K = 2 从 0.968 到 **0.987**（+0.02）；weak 点基本不动（+0.002 / +0.011，本来就在 0.98–0.99）。
- 原因就是 7.18 的机制：atomicAdd 序只打乱 defrag 第一波的 ~16M 粒子，所以 K = 1 的 64M 参考只付 ~2 %（2.55 → 2.62 fps，+2.7 %），而 K = 4 的 16M slab 整块被打乱、付的更多（9.19 → 9.94 fps，+8.2 %），K = 2 的 32M slab 居中（4.95 → 5.17，+4.4 %）；prefix 序把这个随 slab 大小变化的代价抹平，η 就上来了。weak 点的 8M 每卡参考和 8M slab 一起快了 ~14 %（每卡 K = 1 18.8 → 21.4 fps），比值不变。
- 分割：K2 终切 [51]（E8 [50]；标定两轮在 50 / 51 之间摆），K4 [26, 51, 75]（与 E8 相同）；weak K2 [50]，K4 [50, 98, 145]。
- 协变量：busy 中位 SM 时钟 2685–2715 MHz（参考运行 2662–2707），温度 ≤ 71 °C，卡功率峰 582–588 W（575 W 上限的 nvidia-smi 瞬时值），主机 cgroup 峰 39.0 GiB；e8_A 为 2670–2722 MHz / ≤ 75 °C / 35.4 GiB。
- tab:phases 块（T_A / T_B / T_C）、loss sources、hops、bytes check、migration stats 都已生成在 `data/e9/phase2/`；"ideal chain" 行要等 B 线的 K = 1 trace 到齐后重跑整批再填。

#### 4.4.2 B 线：e9_B = SLURM 1741917（wqd10nbj04g4，17:37:49–20:28:58 CST，2 h 51 min；e8_B 2 h 44 min + e8_S2 的单卡对比另算）

- 协议 = e8_B 逐项：K = 8 强扩展 4M / 8M（5 次）、16M / 33M / 64M（3 次），每次同时 8 张 K = 1 参考；4M / 8M 的 K1a / K1b / K8 三份 dump + 按离 seam 列距的场比较（节点上做完，`compare/`）；五个立方体的 K = 1 trace；weak 8 × 8M K8；v7_single 对比（5.1 节 cadence 协议 2000 / 1000，3 对，用 `~/run/vulkan-demo-v7single-rc4` = rc4.1）。PROVENANCE harness cb3c9f5 / 树 37f99f2b，PROVENANCE_END unchanged；8 卡 575 W；bring-up 通过；**212 次运行全部 pass，drift 0**。
- 镜像 `logs/n56/2026-10-28/02_e9_e9B_1741917/`（1168 文件；6 个 npz dump 2.4 GB 不拉，compare 的 json / md 在）。

| 点（K = 8） | E8 η（min–max） | E9 η（min–max） | E9 − E8 | η_min E8 / E9 | fps_K E8 / E9 | K = 1 参考 fps E8 / E9 |
|---|---|---|---|---|---|---|
| 4M (160³)，每卡 0.51M，5 次 | 0.464 (0.460–0.466) | **0.463** (0.461–0.467) | **-0.000** | 0.469 / 0.469 | 136.81 / 156.77 (+14.6 %) | 36.89 / 42.28 (+14.6 %) |
| 8M (200³)，每卡 1.0M，5 次 | 0.663 (0.660–0.667) | **0.661** (0.658–0.663) | **-0.003** | 0.671 / 0.668 | 99.86 / 113.00 (+13.2 %) | 18.81 / 21.37 (+13.6 %) |
| 16M (252³)，每卡 2.0M | 0.816 (0.814–0.817) | **0.814** (0.811–0.815) | **-0.002** | 0.827 / 0.820 | 62.73 / 69.28 (+10.4 %) | 9.61 / 10.64 (+10.7 %) |
| 33M (320³)，每卡 4.1M | 0.847 (0.845–0.850) | **0.867** (0.865–0.870) | **+0.020** | 0.856 / 0.877 | 33.34 / 36.14 (+8.4 %) | 4.92 / 5.21 (+5.8 %) |
| 64M (400³)，每卡 8.0M | 0.841 (0.840–0.842) | **0.878** (0.877–0.878) | **+0.037** | 0.850 / 0.888 | 17.16 / 18.45 (+7.5 %) | 2.55 / 2.63 (+3.0 %) |
| weak 8 × 8M（参考 = 每卡 8M K = 1） | 0.985 (0.985–0.986) | **0.982** (0.978–0.986) | **-0.003** | 0.998 / 0.993 | 18.56 / 21.01 (+13.2 %) | 2.36 / 2.67 (+13.5 %) |

**判读**
- K = 8 的 η 在 4M / 8M / 16M 三点与 E8 相同（差 ≤ 0.003），在 33M / 64M 分别 **+0.020 / +0.037**（0.847 → 0.867，0.841 → 0.878）。机制仍是 7.18 的：atomicAdd 序只打乱 defrag 第一波覆盖的 ~16M 粒子。≤ 16M 的算例里 K = 1 参考和 K = 8 的 slab 都被整体打乱，两边同样快了 10–15 %，比值不动；33M / 64M 的 K = 1 参考只有前 16M 乱（prefix 后只快 +5.8 / +3.0 %），而 4M / 8M 的 slab 整块乱（+8.4 / +7.5 %），η 的差就是这两个百分比的差。
- 每粒子成本的"尺寸效应"消失：K = 1 参考 4M 5.77 ns、8M 5.85 ns、64M 5.94 ns（E8：6.64 ns @ 8M 对 6.13 ns @ 64M，比 1.09；7.18 的本机预测）。
- K = 8 在小算例的低 η（4M 0.46、8M 0.66）与 E8 一样，是每卡 0.5–1M 粒子时的链路 / 延迟地板（E8 报告 B 线诊断），与 defrag 序无关。
- weak 8 × 8M：0.982（E8 0.985），η_min 0.993；8 张卡的 K = 1 参考 21.25–21.53 fps（E8 ≈ 18.9）。
- 分割终切与 E8 相同（4M [7, 11, 16, 21, 26, 30, 35]，8M [8, 14, 20, 26, 32, 38, 44]，16M [10, 17, 25, 32, 40, 48, 55]，33M [12, 22, 31, 41, 51, 60, 70]，64M [14, 26, 39, 51, 63, 76, 88]，weak [50, 97, 146, 194, 242, 290, 338]）。

**单卡参考（第 9 项的 "single reference with v7-single-rc4"）**：链路 K = 1 / v7_single = **0.9974**（三对 0.9956–0.9985；E8 的 e8_S2 0.9984），v7_single 每对带 bootstrap defrag，drift 0。

| job | case | protocol | pair | steps (warmup + measured) | v7_single fps (bootstrap defrag) | chain K = 1 fps used (with / without defrag) | ratio chain / single | valid | single status / drift / overflow | chain status |
|---|---|---|---|---|---|---|---|---|---|---|
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | 1 | 1000 + 1000 | 20.711 (yes) | 20.679 (20.679 / 20.716) | 0.9985 | yes | ok / 0 / {} | pass |
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | 2 | 1000 + 1000 | 20.756 (yes) | 20.718 (20.718 / 20.758) | 0.9982 | yes | ok / 0 / {} | pass |
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | 3 | 1000 + 1000 | 20.736 (yes) | 20.645 (20.645 / 20.680) | 0.9956 | yes | ok / 0 / {} | pass |
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | **mean / mean** | | 20.734 | 20.680 | **0.9974** (pairs 0.9956–0.9985) | 3 valid pairs | | |

**正确性（K = 8 对 K = 1，第 1500 步，流体粒子，K = 1 噪声地板 = 两次独立 K = 1 之差）**：两个算例的 rms 比 test / noise 都在 1 附近（4M dv 0.99、dρ 0.92；8M 1.01 / 0.99），按 seam 距离分层也都 0.96–1.04，即 K = 8 与 K = 1 的差异与 K = 1 自身的重复性差异同量级：

### compare_3d_n152_K8 (02_e9_e9B_1741917): K = 8, cuts [7, 11, 16, 21, 26, 30, 35], devices [0, 1, 2, 3, 4, 5, 6, 7], step 1500 (references at 1500 / 1500); 3,511,808 fluid particles compared, ids missing 0 / 0; dumps valid: test True, reference True, reference2 True
- test |B − A1|: max |dv|/U **3.457e-01**, rms 2.838e-03; max |drho|/rho0 **8.805e-04**, rms 1.993e-06; max |dx| 8.059e-03 m
- noise |A1 − A2|: max |dv|/U 4.329e-01, rms 2.859e-03; max |drho|/rho0 7.884e-04, rms 2.169e-06; rms ratios test / noise: dv 0.99, drho 0.92
### compare_3d_n192_K8 (02_e9_e9B_1741917): K = 8, cuts [8, 14, 20, 26, 32, 38, 44], devices [0, 1, 2, 3, 4, 5, 6, 7], step 1500 (references at 1500 / 1500); 7,077,888 fluid particles compared, ids missing 0 / 0; dumps valid: test True, reference True, reference2 True
- test |B − A1|: max |dv|/U **4.778e-01**, rms 3.153e-03; max |drho|/rho0 **5.552e-04**, rms 1.830e-06; max |dx| 7.346e-03 m
- noise |A1 − A2|: max |dv|/U 4.594e-01, rms 3.134e-03; max |drho|/rho0 6.158e-04, rms 1.855e-06; rms ratios test / noise: dv 1.01, drho 0.99

**协变量**：busy 中位 SM 时钟 2707–2752 MHz（参考运行 2680–2809），温度 ≤ 71 °C，卡功率峰 584–588 W，主机 cgroup 峰 67.5 GiB（dump 阶段）。K = 1 trace 的逐核表 `kernel_table.md`（8M K8：G1 冗余份额 9.9 %，64M K8 6.1 %）、tab:phases 块、loss sources、hops、bytes check、migration stats 均已生成到 `data/e9/phase2/`。

#### 4.4.3 B 线补充（你的第一项）：trace 按相位、按核的 E8 → E9 变化

工具 `docs/cluster_v6/scripts/e9_phase_kernel_compare.py`（输入 = 两个 campaign 各自 `e8_phases_table.py` 的 `phases_table.json`、`e8_kernel_table.py --all` 的逐核表、`e8_loss_3d.py` 的 `ledger.json`），输出 `data/e9/phase2/e8_vs_e9_phases_kernels.{md,json}`。口径：相位 = 每卡稳态步中位、卡间均值（tab:phases 的口径）；核 = 每 sim 稳态步 p50（`--step-trace-detail full` 的逐核时间戳）、sim 间均值，每粒子 = 除以该 sim 的 own 粒子数；比 = E9 / E8 同一点。两个 campaign 的切口逐点相同（只有 64M K2 是 [50] 对 [51]，所以表里给的是每粒子比）。A 线的四个点一并列出。

| 点 | T_B E8 → E9 (ms) | 比 | T_C E8 → E9 (ms) | 比 | 周期比 | corr+dens interior 比 | force_deep_interior 比 | corr+dens boundary 比 | force (boundary) 比 | 小核合计比 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4M 立方体 K = 1 | 209.617 → 182.453 | **0.870** | 0.196 → 0.214 | **1.092** | 0.872 | 0.872 | 0.862 | 1.000 | 1.000 | 1.010 |
| 8M K = 1 | 406.978 → 358.220 | **0.880** | 0.491 → 0.541 | **1.103** | 0.881 | 0.881 | 0.876 | 1.000 | 1.000 | 0.999 |
| 16M K = 1 | 797.677 → 713.646 | **0.895** | 1.228 → 1.238 | **1.007** | 0.895 | 0.886 | 0.875 | 1.000 | 1.000 | 0.997 |
| 33M K = 1 | 1564.087 → 1468.380 | **0.939** | 2.810 → 2.810 | **1.000** | 0.939 | 0.937 | 0.940 | 1.000 | 1.000 | 0.999 |
| 64M K = 1 | 3043.691 → 2931.272 | **0.963** | 5.774 → 5.769 | **0.999** | 0.963 | 0.962 | 0.965 | 1.000 | 1.000 | 0.999 |
| 4M K8（每卡 0.51M） | 0.990 → 0.787 | **0.796** | 3.607 → 3.591 | **0.995** | 0.853 | 0.811 | 0.800 | 1.009 | 0.997 | 0.986 |
| 8M K8（1.0M） | 2.217 → 1.786 | **0.806** | 5.440 → 5.405 | **0.994** | 0.880 | 0.820 | 0.855 | 0.992 | 1.010 | 1.000 |
| 16M K8（2.0M） | 5.841 → 5.112 | **0.875** | 8.173 → 8.192 | **1.002** | 0.903 | 0.889 | 0.884 | 1.007 | 1.002 | 1.000 |
| 33M K8（4.1M） | 14.663 → 12.818 | **0.874** | 13.001 → 12.847 | **0.988** | 0.922 | 0.880 | 0.883 | 0.987 | 0.994 | 0.994 |
| 64M K8（8.0M） | 33.283 → 29.205 | **0.877** | 20.161 → 19.936 | **0.989** | 0.932 | 0.887 | 0.897 | 0.993 | 0.988 | 0.997 |
| weak 8 × 8M K8 | 46.027 → 39.743 | **0.863** | 5.508 → 5.461 | **0.991** | 0.877 | 0.867 | 0.866 | 0.997 | 0.988 | 0.983 |
| 64M K2（32M） | 181.655 → 173.024 | **0.952** | 12.405 → 12.348 | **0.995** | 0.970 | 0.956 | 0.961 | 0.990 | 1.004 | 0.999 |
| 64M K4（16M） | 84.538 → 76.509 | **0.905** | 17.733 → 17.641 | **0.995** | 0.924 | 0.911 | 0.917 | 0.987 | 0.993 | 0.999 |
| weak 2 × 8M K2 | 48.359 → 41.969 | **0.868** | 3.403 → 3.391 | **0.997** | 0.870 | 0.874 | 0.871 | 0.995 | 0.997 | 0.976 |
| weak 4 × 8M K4 | 46.869 → 40.455 | **0.863** | 4.768 → 4.808 | **1.008** | 0.870 | 0.865 | 0.869 | 1.004 | 1.004 | 0.980 |

（K = 1 立方体没有带核，boundary 列为 0 → 比 1.000 是占位。小核 = predict、voxel、ghost_*、expand、install_*、append_departed、density 拷贝。）

**判读**
- **C 段没有加速**：T_C 的比在每个点都是 0.988–1.008（K8 五点 0.995 / 0.994 / 1.002 / 0.988 / 0.989，weak 0.991，K2 / K4 0.995）；两个带核（correction_density_boundary、force）的每粒子成本比 0.987–1.010。小核 0.98–1.01，也不动。
- 加速全部在 B 段的两个深内部核：K8 的 corr+dens interior 0.81–0.89、force_deep_interior 0.80–0.90（T_B 比 0.80–0.88）；K4 0.91 / 0.92，K2 0.96 / 0.96；K = 1 立方体 0.87 / 0.88 / 0.89 / 0.94 / 0.96（4M → 64M，随尺寸递减——atomic 序只乱第一波 ~16M 粒子的机制在逐核数据里再次出现）。同一每卡尺寸下 K8 slab 的内部核比 K = 1 立方体多赚 2–7 个百分点（64M K8 的 8M slab 0.887 / 0.897 对 8M 立方体 0.881 / 0.876 基本一致；4M K8 的 0.5M slab 0.81 对 4M 立方体 0.87），即小 slab 在 atomic 序下整块被乱、吃亏更多。
- 所以 prefix 序改变的是 B 段与 C 段的比例：C 段占周期从 E8 的 35.1 %（64M K8）/ 54.5 %（8M K8）升到 37.3 % / 61.5 %；带核的成本不随粒子排列变——与 E31 的结论（C 段 = 带核的延迟地板）一致。这正是账本里 seam 工作上升的来源（4.4.4）。

#### 4.4.4 B 线补充（你的第二项）：E9 的三因子账本（A + B 的 trace；理想按每卡尺寸立方体，参考列照旧）

`e8_loss_3d.py` 先在 E9 的 A + B 两个镜像上重跑（账本与 tab:phases 块限定在已测的六个点；C、D 结束后用四个镜像整表重跑，8 点的账本见 4.5；2-D 检查与 memcpy 项照旧产出），输出 `docs/cluster_v6/data/e9/loss_3d/`（`ledger.{md,json,csv}`、`phases_3d.*`、`two_d_checks.*`、`memcpy.*`、`loss_sources_figure.json`、`figures/loss_sources.*`，四镜像重跑后定稿）。口径同 7.15：P、busy、暴露、τ 为稳态步均值；理想 = τ_pc / P，τ_pc = 每卡负载按"每卡尺寸立方体"K = 1 trace 的每粒子成本折算（64M → 8M 立方体 × 1.000、33M → 4M × 1.000；16M / 8M / 4M 用 4M 立方体 × 0.488 / 0.244 / 0.125 替身）；seam 工作 = busy − τ_pc；参考工作点 = τ(N)/K − τ_pc（整立方体 K = 1 trace ÷ 8）；暴露 = 每卡每步最大的 (t_tr − T_B)⁺；不平衡 = P − busy − 暴露。百分比 = 占周期。

| 点 | P (ms) | 理想 | seam 工作 | 参考工作点 | 传输暴露 | 不平衡 | η_trace | η 计时 | G1 冗余 |
|---|---|---|---|---|---|---|---|---|---|
| 4M K8（每卡 0.51M） | 7.431 → 6.383 | 45.5 → **46.0** | 18.4 → **24.9** | +0.0 → **+0.0** | 21.6 → **16.9** | 14.5 → **12.2** | 0.455 → 0.460 | 0.464 → 0.463 | 8.1 → 9.6 % |
| 8M K8（1.0M） | 10.110 → 8.909 | 65.4 → **64.4** | 11.1 → **17.8** | -0.1 → **+0.6** | 11.3 → **8.6** | 12.2 → **9.2** | 0.652 → 0.650 | 0.663 → 0.661 | 8.8 → 9.9 % |
| 16M K8（2.0M） | 16.034 → 14.440 | 82.4 → **79.5** | 7.2 → **15.3** | -1.4 → **+0.6** | 3.2 → **0.4** | 7.2 → **4.8** | 0.810 → 0.801 | 0.816 → 0.814 | 8.3 → 9.3 % |
| 33M K8（4.1M） | 30.090 → 27.644 | 89.9 → **85.0** | 4.1 → **10.5** | -5.8 → **+1.0** | 0.1 → **0.1** | 5.8 → **4.4** | 0.842 → 0.860 | 0.847 → 0.867 | 7.2 → 7.7 % |
| 64M K8（8.0M） | 58.396 → 54.069 | 90.3 → **85.7** | 3.4 → **7.8** | -6.5 → **+1.6** | 0.0 → **0.0** | 6.2 → **6.5** | 0.838 → 0.873 | 0.841 → 0.878 | 5.7 → 6.1 % |
| weak 8 × 8M K8 | 54.035 → 47.681 | 94.2 → **93.8** | 4.2 → **3.7** | +3.4 → **+3.4** | 0.0 → **0.0** | 1.6 → **2.6** | 0.976 → 0.972 | 0.985 → 0.982 | 1.6 → 1.9 % |

| 点 | busy E8 → E9 | τ_pc（每卡尺寸立方体） | τ(N)/K | seam = busy − τ_pc | 暴露 | 不平衡 = P − busy − 暴露 |
|---|---|---|---|---|---|---|
| 4M K8（每卡 0.51M） | 4.751 → 4.525 | 3.383 → 2.937 | 3.383 → 2.937 | 1.368 → 1.588 | 1.604 → 1.080 | 1.075 → 0.778 |
| 8M K8（1.0M） | 7.731 → 7.323 | 6.608 → 5.737 | 6.594 → 5.792 | 1.123 → 1.586 | 1.147 → 0.765 | 1.232 → 0.822 |
| 16M K8（2.0M） | 14.374 → 13.683 | 13.218 → 11.476 | 12.989 → 11.567 | 1.157 → 2.207 | 0.511 → 0.063 | 1.148 → 0.694 |
| 33M K8（4.1M） | 28.303 → 26.405 | 27.065 → 23.498 | 25.334 → 23.776 | 1.238 → 2.906 | 0.040 → 0.023 | 1.747 → 1.217 |
| 64M K8（8.0M） | 54.746 → 50.566 | 52.756 → 46.333 | 48.946 → 47.191 | 1.990 → 4.233 | 0.001 → 0.000 | 3.649 → 3.503 |
| weak 8 × 8M K8 | 53.156 → 46.460 | 50.909 → 44.712 | 52.756 → 46.333 | 2.247 → 1.748 | 0.000 → 0.000 | 0.878 → 1.221 |

**判读**
- **参考工作点已到 ±1 % 之内**（E8 为 −5.8 / −6.5 %）：8M +0.6、16M +0.6、33M +1.0、64M **+1.6 %**（4M 的参考与替身同源，恒为 0）。符号翻了：E8 里整立方体的 K = 1 比每卡尺寸立方体每粒子便宜 7–9 %（7.12–7.14 的尺寸效应），E9 里整立方体反而贵 1–2 %（64M K = 1 5.94 ns 对 8M 5.85 ns），即剩下的是一个很小的反向残差（整步成本随 N 轻微上升，可能是 cache 容量），7.14 的表面项消失。64M 的 +1.6 % 略超 ±1 %，8M–33M 在内。
- **seam 工作升到 7.8 %（64M）、10.5 %（33M）、15.3 %（16M）、17.8 %（8M）、24.9 %（4M）**（E8：3.4 / 4.1 / 7.2 / 11.1 / 18.4 %）；weak 3.7 %（4.2）。ms 口径更直白：64M 的 busy 降了 4.18 ms，而 τ_pc 降了 6.42 ms，差额 2.2 ms 全是 C 段不动造成的（4.4.3）。seam 工作现在是 64M 的最大损失项（7.8 % 对不平衡 6.5 %）。
- 理想份额相应下降（64M 90.3 → 85.7 %），η_trace = 理想 + 参考工作点 = 0.873（计时 η 0.878）；E8 0.838 / 0.841。所以 η 的 +0.037 = 参考工作点 +8.1 个百分点 − 理想份额 −4.6 个百分点，而不是链路本身的损失变小。
- 传输暴露在 4M / 8M / 16M 降了（21.6 → 16.9、11.3 → 8.6、3.2 → 0.4 %）：这是节点差异，不是 defrag 序——E9 的 B 线跑在 wqd10nbj04g4，E8 的在 nbj11g2；64M 包（41.1 MB）的主机 memcpy 中位 E9 9.65 GB/s 对 E8 6.37 GB/s（K2 14.6 对 15.7、K4 10.5 对 10.8 相同），最差链路 t_tr 64M K8 9.7 ms 对 17.0 ms，4M 的主机部分 1.25 对 2.28 ms。≥ 33M 两个 campaign 暴露都是 0，η 不受影响；≤ 16M 的 η 相同是因为暴露减少被 seam 工作增加抵消（4M：−4.7 与 +6.5 个百分点，不平衡 −2.3）。
- 不平衡：64M 6.2 → 6.5 %（整列切口相同，s2 / s5 的 busy = 周期，其余卡等 4–5 ms），其余点略降；G1 冗余份额 5.7 → 6.1 %（64M）等，因为带核的时间份额升了。
- 四个作业齐后的整表（含 128M、76M）见 4.5。

#### 4.4.5 C 线：e9_C = SLURM 1741921（wqd10nbj04g2 = A 的节点，19:58:47–21:19:05 CST，1 h 20 min；e8_C 1 h 15 min）

- 协议 = e8_C：128M 立方体（cavity3d_n496，504³）K = 8，3 次，每次同时 4 对 K = 2 配对参考（128M 单卡放不下），trace；另加 K = 2 配对 trace。PROVENANCE 同前，PROVENANCE_END unchanged；17 次运行 + bring-up 全部 pass，drift 0。镜像 `logs/n56/2026-10-28/03_e9_e9C_1741921/`。
- **η 0.928（0.926–0.929；E8 0.868，+0.060）**，η_min 0.934；fps_K8 8.93 → 9.80（+9.8 %），配对参考 2.62–2.65 fps（整域等效 1.29 → 1.32，+2.6 %）。机制同 B 线：配对参考每卡 64M、只有前 16M 被 atomic 序打乱（+2.6 %），而 K8 的 16M slab 整块乱（+9.7 %）。E8 → E9 的差 +0.060 是 12 个点里最大的。
- 切口 [18, 33, 48, 64, 80, 95, 110]（E8 [18, 33, 48, 64, 79, 95, 110]，差一列）；相位：T_B 71.5 → 63.3 ms（0.886），T_C 31.9 → 31.7（0.995），周期 110.3 → 101.1（0.917）；内部核 0.899 / 0.901，带核 0.998 / 0.994。
- 账本（128M）：理想 92.5 → 90.5 %，seam 2.2 → 5.8 %，参考工作点 −6.7 → +1.6 %，暴露 0，不平衡 5.3 → 3.7 %；η_trace 0.858 → 0.921。
- 协变量：busy 中位时钟 2632–2692 MHz（参考 2622–2692），温度 ≤ 75 °C，主机 cgroup 峰 52.7 GiB。

#### 4.4.6 D 线：e9_D（SLURM 名 e9_BD）= 1741924（wqd10nbj04g4 = B 的节点，20:28:58–21:39:50 CST，1 h 11 min；e8_D3 1 h 11 min）

- 协议 = e8_D3：76M 立方体（cavity3d_n416，424³）K = 8，3 次 + 同时 K = 1 参考，trace；另加 K = 1 trace。29 次运行 + bring-up 全部 pass，drift 0。镜像 `logs/n56/2026-10-28/04_e9_e9D_1741924/`。
- **η 0.929（0.927–0.930；E8 0.883，+0.046）**，η_min 0.940；fps_K8 15.22 → 16.45（+8.1 %），K = 1 参考 2.20–2.24（2.16 → 2.21，+2.7 %）。
- 切口 [15, 28, 41, 54, 67, 80, 93] = E8 的，标定一轮收敛；相位：T_B 40.4 → 35.8 ms（0.887），T_C 22.6 → 22.7（1.001），周期 65.0 → 60.4（0.930）；内部核 0.900 / 0.905，带核 1.001 / 1.000；76M 立方体 K = 1：B 段 0.974（64M 的 0.963 之后继续收窄，第一波 16M 占 76M 的 21 %）。
- 账本（76M）：理想 95.0 → 90.8 %，seam 3.0 → 8.5 %，参考工作点 −8.0 → +1.3 %，不平衡 2.0 → **0.6 %**；η_trace 0.870 → 0.921。76M 对 64M 仍多 0.05 η，仍全在不平衡（0.6 对 6.5 %），E8 7.11 的结论不变。
- 协变量：busy 中位时钟 2677–2745 MHz（参考 2677–2740），温度 ≤ 71 °C，主机 cgroup 峰 71.3 GiB。

### 4.5 整批汇总（第 10 项交付）

四个作业 2026-10-28 17:37–21:40 CST 全部 COMPLETED，333 次运行 + 4 次 bring-up 全部 pass、drift 0，没有丢卡。四个镜像整体重跑后的交付（与 E8 同一套工具、同一目录结构）：`docs/cluster_v6/data/e9/phase2/`（tables / trials / index / figures / phases_table / loss_sources / hops / bytes_check / kernel_table / correctness / single_ratio / migration_stats / summary_run，10 个工具全 ok）、`eta_e8_vs_e9.{md,json}`（第 10 项的逐点 η 表）、`e8_vs_e9_phases_kernels.{md,json}`（4.4.3）、`docs/cluster_v6/data/e9/loss_3d/`（账本 8 点、tab:phases 3-D 块、2-D 检查、memcpy、`figures/loss_sources.{pdf,png,csv}` = fig:loss 的 E9 版本；**没有覆盖论文目录里的 E8 图**，要换由你定）。镜像 `logs/n56/2026-10-28/01–04`，README 2026-10-28 节 + jobs.csv 四行。

**E8 → E9 逐点 η（η 不含 defrag 步；参考 = 同时的 K = 1，128M 为 K = 2 配对；均值与 trial 范围）**

| 点 | 作业 | E8 η（min–max） | E9 η（min–max） | E9 − E8 | η_min E8 / E9 | fps_K E8 → E9 | 参考 fps E8 → E9 | T_B 比 | T_C 比 |
|---|---|---|---|---|---|---|---|---|---|
| 4M K8 | B | 0.464 (0.460–0.466) | **0.463** (0.461–0.467) | **-0.000** | 0.469 / 0.469 | 136.81 → 156.77 (+14.6 %) | 36.89 → 42.28 (+14.6 %) | 0.796 | 0.995 |
| 8M K8 | B | 0.663 (0.660–0.667) | **0.661** (0.658–0.663) | **-0.003** | 0.671 / 0.668 | 99.86 → 113.00 (+13.2 %) | 18.81 → 21.37 (+13.6 %) | 0.806 | 0.994 |
| 16M K8 | B | 0.816 (0.814–0.817) | **0.814** (0.811–0.815) | **-0.002** | 0.827 / 0.820 | 62.73 → 69.28 (+10.4 %) | 9.61 → 10.64 (+10.7 %) | 0.875 | 1.002 |
| 33M K8 | B | 0.847 (0.845–0.850) | **0.867** (0.865–0.870) | **+0.020** | 0.856 / 0.877 | 33.34 → 36.14 (+8.4 %) | 4.92 → 5.21 (+5.8 %) | 0.874 | 0.988 |
| 64M K8 | B | 0.841 (0.840–0.842) | **0.878** (0.877–0.878) | **+0.037** | 0.850 / 0.888 | 17.16 → 18.45 (+7.5 %) | 2.55 → 2.63 (+3.0 %) | 0.877 | 0.989 |
| 76M K8 | D | 0.883 (0.881–0.884) | **0.929** (0.927–0.930) | **+0.046** | 0.892 / 0.940 | 15.22 → 16.45 (+8.1 %) | 2.16 → 2.21 (+2.7 %) | 0.887 | 1.001 |
| 128M K8（配对参考） | C | 0.868 (0.864–0.871) | **0.928** (0.926–0.929) | **+0.060** | 0.874 / 0.934 | 8.93 → 9.80 (+9.8 %) | 1.29 → 1.32 (+2.6 %) | 0.886 | 0.995 |
| weak 8 × 8M K8 | B | 0.985 (0.985–0.986) | **0.982** (0.978–0.986) | **-0.003** | 0.998 / 0.993 | 18.56 → 21.01 (+13.2 %) | 2.36 → 2.67 (+13.5 %) | 0.863 | 0.991 |
| 64M K2 | A | 0.968 (0.966–0.970) | **0.987** (0.986–0.987) | **+0.018** | 0.971 / 0.989 | 4.95 → 5.17 (+4.5 %) | 2.56 → 2.62 (+2.6 %) | 0.952 | 0.995 |
| 64M K4 | A | 0.901 (0.898–0.904) | **0.951** (0.948–0.954) | **+0.051** | 0.912 / 0.961 | 9.19 → 9.94 (+8.2 %) | 2.55 → 2.61 (+2.4 %) | 0.905 | 0.995 |
| weak 2 × 8M K2 | A | 0.984 (0.981–0.986) | **0.986** (0.983–0.988) | **+0.002** | 0.986 / 0.987 | 18.46 → 21.09 (+14.2 %) | 9.38 → 10.69 (+13.9 %) | 0.868 | 0.997 |
| weak 4 × 8M K4 | A | 0.981 (0.978–0.986) | **0.992** (0.991–0.992) | **+0.011** | 0.993 / 1.002 | 18.39 → 21.15 (+15.0 %) | 4.69 → 5.33 (+13.7 %) | 0.863 | 1.008 |

**E9 账本（8 个 K8 点；E8 → E9，占周期 %）**

| 点 | 每卡 | P (ms) | 理想 | seam 工作 | 参考工作点 | 传输暴露 | 不平衡 | η_trace → 计时 η | G1 冗余 |
|---|---|---|---|---|---|---|---|---|---|
| 4M K8 | 0.51M | 7.431 → 6.383 | 45.5 → **46.0** | 18.4 → **24.9** | +0.0 → **+0.0** | 21.6 → **16.9** | 14.5 → **12.2** | 0.460 → 0.463 | 8.1 → 9.6 % |
| 8M K8 | 1.00M | 10.110 → 8.909 | 65.4 → **64.4** | 11.1 → **17.8** | -0.1 → **+0.6** | 11.3 → **8.6** | 12.2 → **9.2** | 0.650 → 0.661 | 8.8 → 9.9 % |
| 16M K8 | 2.00M | 16.034 → 14.440 | 82.4 → **79.5** | 7.2 → **15.3** | -1.4 → **+0.6** | 3.2 → **0.4** | 7.2 → **4.8** | 0.801 → 0.814 | 8.3 → 9.3 % |
| 33M K8 | 4.10M | 30.090 → 27.644 | 89.9 → **85.0** | 4.1 → **10.5** | -5.8 → **+1.0** | 0.1 → **0.1** | 5.8 → **4.4** | 0.860 → 0.867 | 7.2 → 7.7 % |
| 64M K8 | 8.00M | 58.396 → 54.069 | 90.3 → **85.7** | 3.4 → **7.8** | -6.5 → **+1.6** | 0.0 → **0.0** | 6.2 → **6.5** | 0.873 → 0.878 | 5.7 → 6.1 % |
| 76M K8 | 9.53M | 66.134 → 60.774 | 95.0 → **90.8** | 3.0 → **8.5** | -8.0 → **+1.3** | 0.0 → **0.0** | 2.0 → **0.6** | 0.921 → 0.929 | 5.6 → 6.1 % |
| 128M K8（配对参考） | 16.00M | 112.302 → 102.247 | 92.5 → **90.5** | 2.2 → **5.8** | -6.7 → **+1.6** | 0.0 → **0.0** | 5.3 → **3.7** | 0.921 → 0.928 | 4.7 → 5.1 % |
| weak 8 × 8M K8 | 7.72M | 54.035 → 47.681 | 94.2 → **93.8** | 4.2 → **3.7** | +3.4 → **+3.4** | 0.0 → **0.0** | 1.6 → **2.6** | 0.972 → 0.982 | 1.6 → 1.9 % |

**读法**
1. **η 的变化只发生在每卡 ≥ 4M 的点**：K8 4M / 8M / 16M 与 weak 不变（|Δ| ≤ 0.003），33M +0.020、64M +0.037、76M +0.046、128M +0.060，64M K4 +0.051、K2 +0.018。原因不是链路变好，而是参考工作点：atomic 序只打乱 defrag 第一波的 ~16M 粒子，所以每卡 ≤ 2M 的点两边同样全乱、比值不动；每卡 ≥ 4M 的点里参考（整立方体或 64M 的配对半卡）只乱前 16M、链路的 slab 整块乱，E8 的 η 被这个不对称压低了 0.02–0.06。E9 把它抹平：参考工作点从 −5.8 ~ −8.0 % 回到 +0.6 ~ +1.6 %。
2. **加速全在深内部的两个 sweep 核，C 段（带核）不动**（4.4.3）：K8 的 T_B 比 0.80–0.89，T_C 0.99–1.00；所以 seam 工作的份额翻倍（64M 3.4 → 7.8 %，128M 2.2 → 5.8 %），成为大算例的最大损失项，不平衡其次（64M 6.5 %、128M 3.7 %、76M 0.6 %）。
3. 每粒子成本：K = 1 立方体 5.77 / 5.85 / 5.94 / 5.99 ns（4M / 8M / 64M / 76M；E8 6.64 → 6.13 ns 的尺寸效应没了）；链路 K = 1 / v7_single 0.9974；K8 对 K1 的场差在 K1 噪声地板上；VRAM = 模型。
4. 两个 campaign 其他协变量相同：切口逐点相同（64M K2 差一列、128M 差一列），时钟比链路 / 参考 1.01–1.03，温度 ≤ 75 °C，575 W 上限。≤ 16M 的暴露项差异是节点（nbj04g4 对 nbj11g2 的主机 memcpy）造成的，不影响 η。

### 4.6 论文切到 E9（你 2026-10-29 的四项）

交付在一个文件里：`docs/cluster_v6/data/e9/paper_e9.md`（生成脚本 `docs/cluster_v6/scripts/e9_paper_rows.py`，同名 .json 带全部数值）。内容：(1) `C:/Users/cchen/PycharmProjects/vulkan-demo/manuscripts/fig/` 里的 `strong_scaling_3d.*` 与 `loss_sources.*` 已换成 E9（2-D 部分仍是 E7），E8 版保留为 `*_e8_atomic.*`；(2) tab:strong3d、tab:weak3d、tab:phases 的 64M / 8M 3-D 行块（理想链 = 整算例 K = 1 trace ÷ 8 一行 + 链路 K = 8 一行，步长中位数与均值两列）、tab:ledger3d（η、seam、transport、imbalance、r）的 LaTeX 行；(3) 按整算例 ÷ K 的理想链重算账本：η' + 三项 = 100 % 恒等，计时 η + 三项 − 1 在 +0.003 ~ +0.013（8 点最大 0.0126 < 0.02）；(4) 64M K8 逐核每处理粒子成本（E9）：interior corr+dens 2.78 ns、force 2.80 ns，boundary 2.91 / 2.98 ns，boundary / interior = 1.047 / 1.065（合计 1.056）；E8 为 3.14 / 3.12 对 2.94 / 3.02。

#### 4.7 论文补三项（用户 2026-10-29；`docs/cluster_v6/scripts/e9_paper_addenda.py` → `data/e9/paper_e9_addenda.{md,json}`，`paper_e9.md` 末尾加了指针）

1. **64M K8 三跳 E8 → E9**（14 链路 × 1000 步合并 p50；E8 = e8_B / wqd10nbj11g2，E9 = e9_B / wqd10nbj04g4）：readback 2.71 → 1.66 ms，host 含 memcpy 6.72 → 4.30（memcpy 6.46 → 4.26，host_rest 0.25 → 0.02），upload 1.54 → 1.07；最慢链路口径 t_tr 17.04 = 1.63 + 11.36 + 1.51 → 9.75 = 2.49 + 5.66 + 1.21。41 MB 包的 memcpy（E9）：K2 2,815 us / 14.61 GB/s，K4 3,929 us / 10.47 GB/s，K8 4,260 us / 9.65 GB/s（E8 2,626 us / 15.66 GB/s、3,821 us / 10.76 GB/s、6,459 us / 6.37 GB/s）。节点对照：ab2600f..cb3c9f5 只有 31bd493 改了 `experiment/v7/utils/`（defrag 路径），传输层未动；host_rest 在 nbj11g2 150–260 us、nbj04 组 14–28 us，E8 代码的 e8_D3（nbj04g2）也是 28 us；K = 8 并发 memcpy nbj11g2 6.4 GB/s（64M）/ 3.7（128M）对 nbj04 组 7.3–10.0 → 三跳差别是节点，不进 η（64M 两边暴露都 0）。
2. **fig:loss(a) 3-D 2M / 卡（16M K8）的 host 部分**：取法 = 每步按 t_tr 选最慢链路、取它的 host、对步取中位（`e7_loss_figure.worst_link_medians`）。E9 **1.54 ms**（3→4 在 78 % 的步里最慢，其 host 中位 1.50；逐链路 host 中位 1.07–1.58，合并 1.16）；E8 3.94 ms（逐链路 1.17–3.72，合并 2.44；最慢链路在 3→4 / 5→6 / 2→1 / 2→3 间轮换，所以图值高于任一链路自己的中位）。降幅是节点差异；该点 E9 已过 t_tr = T_B 交点（r 0.81，E8 1.51）。
3. **8M / 16M K8 的 G1 复本与 boundary 核（E9，内部卡 s1–s6 均值）**：8M 每卡 6 列 × 160k = 0.96M 自有，G1 2 × 160k = 320k（33 % 自有）；corr+dens boundary 处理 0.64M 自有 + 0.32M G1 = 0.96M，2,903 us（3.02 ns / 处理粒子，K1 立方体 2.83）；force boundary 处理全部 0.96M 自有（3 列 × 2 侧 = 整个 slab），3,021 us（3.15 ns，K1 2.73）；force deep interior 空派发 7 us，corr+dens interior 只剩 2 列 1,087 us（3.40 ns）。G1 冗余 10.9 % 周期，内部卡 seam 19.3 % = G1 冗余 10.9 + 带核每粒子超额 6.6 + 窄 interior 核超额 2.0 + 小核 / 空隙 1.1 + 残差 -1.2。16M 每卡 7–8 列 × 254k，G1 2 × 254k = 508k（27 %）；corr+dens boundary 1.524M 处理 4,519 us（2.97 ns，K1 2.79），force boundary 1.524M（80 % 自有）4,675 us（3.07 ns，K1 2.70），deep interior 1–2 列 3.2–3.8 ns；G1 冗余 10.4 %，seam 17.6 % = 10.4 + 5.8 + 2.8 -0.6 -0.8。boundary 核时长与 E8 相同（8M 2,921 / 3,003 → 2,903 / 3,021 us）。正文的 seam 构成：一半多是 G1 复本的 corr+dens 冗余（≈ 10–11 % 周期），其次是带核每处理粒子比 K1 立方体贵 6–15 %（force 带更贵），再是 1M–2M 每卡时 interior 核只剩 1–2 列的窄派发。

#### 4.8 fig:loss 重画成三栏（用户 2026-10-29；`e8_loss_3d.py` 的 `figure()` 换成三栏版，旧两栏版保留为 `figure_v1`，新入口 `--figure-from-json` 只重画不重跑）

- 布局：整页宽 5.333 in，三个 1.38 × 1.62 in 面板。(a) 2-D、(b) 3-D：T_B 实线实心、t_tr 实线空心、host 部分虚线无符号，(a) 用 fig:scaling3d 的 2-D 灰（`#8c8c8c`，圆），(b) 用 3-D 蓝（`#08519c`，方）；横轴每卡粒子数（对数，各自范围），纵轴 ms（对数，共用范围 0.1–140，刻度只在 (a) 标）；t_tr = T_B 的交点画细竖虚线；图例在 (a) 左上只标三个量，维度由面板标题 "2-D" / "3-D" 给出。(c) η 对 r：2-D 灰实心圆、3-D 蓝实心方，r 对数轴，r = 1 竖虚线，η = 1 ± 0.02 带，图例 2-D / 3-D。
- 输出：`docs/cluster_v6/data/e9/loss_3d/figures/loss_sources.{pdf,png,csv}` 和 `C:/Users/cchen/PycharmProjects/vulkan-demo/manuscripts/fig/loss_sources.{pdf,png,csv}`（覆盖），两处旧版另存 `loss_sources_v1.*`；CSV 列不变（数据同 10-28 的 E9 版）。
- **交点**：2-D 1.42M / 卡（cavity2d_n2840 与 cavity2d_n4000 之间，1.60 ms；E7 数据不变）；3-D **1.31M / 卡**（cavity3d_n192 与 cavity3d_n244 之间，2.67 ms）。用户给的 1.97M / 卡是 E8 的值（E8 json：1.97M，5.62 ms）：E8 的 16M K8 t_tr 5.67 ms 贴着 T_B 5.72，E9（nbj04g4）同点 t_tr 2.92 ms 对 T_B 5.09，8M K8 t_tr 2.52 对 T_B 1.77，交点落到 1.0M 与 2.0M 之间，按 E9 数据画在 1.31M。正文若引交点数字要改成 1.31M（4.7 的第 2 项：这是节点差异带来的 t_tr 下降）。

#### 4.9 轻量版 K = 2 对 K = 1 等价性检查（用户 2026-10-31；`docs/cluster_v6/scripts/e9_equivalence_light.py`，数据 `docs/cluster_v6/data/e9/equivalence_light/`）

- 设置：Re = 1000 方腔 1001²，所有运行从 5.3 的 K = 1 检查点 c0011907000（t = 89.3025，sha256 db77ef29…）起跑 500 步，第 1、10、100、500 步取 56 点瞬时速度（MLS，wall 坐标系，5.3 运行自己的 points.npz）。t ≈ 80 的 Re 1000 检查点（c0010584000，t = 79.38）已被保留两个的规则删掉，全盘只有 Re 3200 运行的同名文件。
- 求解器 v7-rc4（`experiment/v7` 树 37f99f2b，未改），算例从 rc3 工作树读（OBJ 被 git 忽略，case.yaml 与材料库字节相同）。工作站现在是两张 5090：a1、a2 = K = 1 在无显示的卡上，b = K = 2 等权，第 103 列切开，x = +0.0009 m，离中线 0.90 Δx = 0.18 h。runner 只能停在 defrag 边界，所以驱动复刻 `run_pipelined` 的 depth 2 循环，在记录步排空读数；每个记录步查不变量，全部为 0，drift 0。
- 结果：a1 与 a2 不逐位相同（`initialize_voxelization` 的 atomicAdd 让体素内顺序逐次不同，defrag 保留它）。第 1 步三次运行排序后的 (x, y, u, v) 逐位相同，差值 2–3e-16 是插值的 float64 舍入，没有检验力。第 10 / 100 / 500 步 56 点最大差：a1 对 a2 1.0e-7 / 2.9e-6 / 8.6e-6，b 对 a1 1.0e-7 / 2.8e-6 / 8.1e-6。独立重复一遍（`repeat/`）量级相同。
- 结论：b 的差和噪声底同量级（比值 0.65–1.53）。缝上 28 点的 b 对 a1 与同样点的 a1 对 a2 之比 0.60–1.40，没有超出噪声。这个检查分辨不了 1.5 倍以内的缝效应。表见 `equivalence_light.md`。

#### 4.10 启动步多积分一步密度（2026-11-01；修复提交 216b73e，分支 `bootstrap-density-fix`，未推送；脚本 `docs/cluster_v6/scripts/e9_bootstrap_density.py`，数据 `docs/cluster_v6/data/e9/bootstrap_density/`）

- 问题：`_record_bootstrap_compute_cmd` 在 correction + density 之后做了 scratch → primary 拷贝，force 用的是 ρ^0 + Δt ρ̇^0，第 1 步再从这个值积分一次，与论文 2.5 节的 a(r^0, v^0, ρ^0) 不符。这个顺序从 V0 就有，`utils/sph`、`experiment/v5`、`v6`、`v7_single` 相同，这次只改 `experiment/v7`。
- 修复：方案 A，拷贝换成一个 compute 屏障。主循环、核、SPIR-V、force、predict、halo 包都没改。`_test_seam_layout` 新增断言（在未修复的代码上有 200 条失败，修复后为 0），历史基准模拟器施加同一处替换（必须恰好命中一次），失败集合与 a501a56 逐条相同：858 条，其中 840 条是 E9 后的 defrag 录制流，18 条是 partition 比较。
- 结果：见 `bootstrap_density.md` 的结论。旧代码 15 个组合结果相同（26 个流体粒子，全在上角 2h 内，最大 |ρ−ρ0|/ρ0 = 7.913e-4，压力 7.9 ρ0U²），修复后 ρ = ρ0、P = 0。动能相对差第 1000 步降到 1e-5 量级以下，56 点速度从 t = 0.03 起在噪声底内。毒化 16 个组合活粒子 NaN 为 0。
- 重启：`restart_init` 不走 `bootstrap_compute`，任务里担心的二次积分和二次半步回退不存在；4 个运行重启后流体状态与检查点逐位相同。Re 1000 / 3200 的 K = 1、K = 2 都是一段从第 0 步跑完（经过旧启动步），K = 8 续算（`start_from` t = 89.3025 / 79.5）走 `restart_all`。
- 结论：修的理由是一致性和公开代码要与论文一致，不是精度；论文数据不重跑。注意 `experiment/v7` 树从 37f99f2b 变为 27887d76，`cavity_runner` 的 physics 哈希随之变化，rc4 下启动的运行不能用修复后的代码 `--resume`。

## 5. 数据与复现

- 代码：提交号与标签见本节末尾的更新。
- 本机：`docs/cluster_v6/scripts/e9_local/run_k.sh`（一次 K 槽链路运行）、`local_batch.sh`（第 4–6 项的序列）、`local_summary.py` → `docs/cluster_v6/data/e9/local/local_validation.{md,json}`、`compare/compare_8m.{md,json}`；`make_atomic_case.py`（A/B 的 atomic 变体 yaml）；树的改动由 `patch_tree_prefix.py` / `patch_tree_audit.py` 施加（两个工作树同一份），`patch_plan_e9.py`（修订 18–20），`make_deploy_scripts.py`（E9 部署脚本）。
