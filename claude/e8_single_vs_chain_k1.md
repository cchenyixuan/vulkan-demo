# v7_single 比链路 K = 1 慢 17 % 的排查（e8_S 1722130，8M 立方体；只看代码与日志，没有提交作业，没有本机 3-D 运行）

2026-10-22。对象：e8_S 的两个半边——`~/run/vulkan-demo-v7single`（分支 v7-single 0e50ce4，experiment/v7_single 建在 v7-rc1 的 experiment/v7 上，树 82dd6a74）和 `~/run/vulkan-demo-e8s`（e8-3d-p2s = 3632cd3，experiment/v7 树 10781f23 = v7-rc3 的 f33a46ba + 分区器下限 3）。结果：v7_single 16.48 / 16.45 / 16.39 fps，链路 K = 1 19.07 / 19.13 / 19.31 fps，链路 / single = 1.166。

## 0. 结论

1. **核序列、SPIR-V、工作组、派发维度两边一样**（第 1 节）：每步都是 predict → update_voxel → correction_density → force + density_scratch_copy 五个派发，8 个步进 / bootstrap / defrag 模块的 sha256 逐一相同，工作组 128，派发 71,875 / 1,099 / 35,938 个工作组。差别只有三处：(a) force 的密度来源——链路读 scratch（`FORCE_DENSITY_SOURCE` 56 = 1）并在 density copy 之前跑，v7_single 先 copy 再让 force 读 primary（56 = 0）；(b) 模式常数 `CORRECTION_MODE` 47 与 `FORCE_MODE` 49（链路 interior = 1，single all = 0）——没有 ghost 列时 `in_boundary_band` 恒为假（80 / 81 = 0），两种模式折叠成同一段代码；(c) `GHOST_LAYERS` 83（2 对 1）与 `REPLICA_REGION_SIZE` 86（216,320 对 0）——五个步进核都不引用这两个常数（只有 ghost / band 辅助函数用），对生成的代码没有影响。`correction_density.comp` 确认是**一次邻居扫描同时算 L 和 ρ**（见 1.3）。
2. **rc1 → rc3 的 host 改动不碰 K = 1 路径**（第 2 节）：11 个 host 文件的改动是 E40 adami 链（只在"adami 且有邻片"时生效）和 E7 B2 的 dest guard relay（只有 worker 才有，K = 1 没有 worker）；派发几何、特化常数、deep-wall skip、缓冲区标志、核的顺序在 K = 1 + simple 墙的片上都没变。CPU 重建出的链路配置与集群日志的 `[e30-config]` / `[SimV7]` 行逐字一致。
3. **defrag 不对称**（第 3 节）：v7_single 的 bench 在 `--no-defrag`（defrag_cadence = 0）下**一次 defrag 都不做，连启动那一次也没有**（`sim.bootstrap()` = 体素化 + 一次 correction / density / force / 半踢；循环里 `with_defrag = bool(cadence) and …` 恒假）；链路的 `ChainOrchestratorV7.bootstrap_all()` **无条件**在录好 step 命令后做一次 bootstrap defrag（`--no-defrag` 只把周期 defrag 关掉）。e8_S 日志证实：single 半边没有任何 defrag 行，链路半边在 `all 1 sims bootstrapped` 之前有 `[e30] defrag_time sim0: wall_ms=74.216`（三对 74 / 144 / 26 ms），窗口内 `defrag_events in_window=0`。所以 **e8_S 的 v7_single 在生成器的初始粒子顺序上跑了全部 1500 步，链路在体素排序后的顺序上跑**。

**这就是 17 % 最可能的来源。** 3-D 对齐生成器按 x 最慢、y、z 最快写粒子，体素编码是 y 最快、z、x 最慢（`own_voxel_id_of`），于是同一体素的 64 个流体粒子在文件序里分成 16 段、每段 4 个：位置数组每个体素占 16 条 128 B 缓存行，排序后 8 条（第 3.3 节，CPU 统计）。邻居扫描里 27 个体素 × 64 个粒子的 gather 在这种半有序布局上命中率低。与 §5.1 对照：那套协议每 1000 步 defrag 一次、窗口从第 1001 步开始，两边在窗口里都是排好序的，3-D 也测过——`cavity3d_1m` 三次 99.64 %（99.49–99.88）、`cavity3d_8m_b4` 一对 99.98 %（`docs/seam_audit/v7_single/reference_ratio.md`）；e8_S 是 v7_single 第一次在**从未 defrag** 的状态下计时，也是第一次出现 17 % 的差距。两套协议之间唯一变的量就是 single 有没有 defrag 过。

**还没有直接验证**（需要跑 GPU）：最小的验证是 v7_single 加一次启动 defrag 再测——`SingleRunner.bootstrap_all()` 已有这条路径（"as the chain does"），只是 `_run_single_bench.py` 没走它；给 bench 加一个 `--bootstrap-defrag` 开关、在集群上 8M 3-D 各跑 2 次（有 / 无），约 10 min 整节点。本机闲置的 XTX 可以先做 2-D 1M 的同类旁证（v7_single 有 / 无启动 defrag，几分钟）。两件都等你的话。对 η 的含义不变：e8 的 η 用链路 K = 1 或配对做参考，不用 v7_single；若证实，改的是 7.5 节里"链路 / single 比值"的解释（e8_S 的 single 偏慢是协议造成的，§5.1 的结论在它自己的协议下仍成立）。

## 1. 每步派发的核序列并列（问题 1）

### 1.1 并列表

两边都按 8M 立方体（`cases/aligned/cavity3d_n192`，8,000,000 粒子，52³ = 140,608 体素，own pool 9,200,000，`workgroup: 128`）在 CPU 上重建：链路侧用 e8 树的 `partition_v7.compute_chain_partition(case, [1.0], 1.2)` 得到 K = 1 的片，再用 v7_single 自带的 CPU 假模拟器（`experiment/v7_single/_v7_cpu_fake.py`）取 `_build_compute_pipelines` 的特化常数并录下相位 A / B / C、bootstrap、defrag 的命令流；single 侧同样取 `pipeline_specs()` 与 `record_step` 等。SPIR-V 的 sha256 来自两棵树各自的 `experiment/v7/shaders/spv/MANIFEST.txt`。

| 链路位置 | 链路 K = 1 管线 | single 位置 | v7_single 管线 | SPIR-V 文件（sha256 前 12 位，字节；两边相同） | 工作组 | 派发 (x, y, z) = 线程 | 不同的特化常数（链路 对 single） |
|---|---|---|---|---|---|---|---|
| A-1 | predict | 1 | predict | predict.comp.spv 099b129f66ea 6392 | 128 | (71875, 1, 1) = 9,200,000 | 83 GHOST_LAYERS 2 对 1；86 REPLICA_REGION_SIZE 216320 对 0 |
| A-2 | update_voxel | 2 | update_voxel | update_voxel.comp.spv 024e04287997 4916 | 128 | (1099, 1, 1) = 140,672 | 同上 |
| B-1 | correction_density_interior | 3 | correction_density_all | correction_density.comp.spv 7871141dcdec 57868 | 128 | (71875, 1, 1) = 9,200,000 | 47 CORRECTION_MODE 1 对 0；83、86 同上 |
| B-2 | force_deep_interior_scratch | 5 | force_all | force.comp.spv 3a635216025f 46776 | 128 | (71875, 1, 1) = 9,200,000 | 49 FORCE_MODE 1 对 0；56 FORCE_DENSITY_SOURCE 1（scratch）对 0（primary）；83、86 同上 |
| C-1 | density_scratch_copy | 4 | density_scratch_copy | density_scratch_copy.comp.spv 50e9e5f6fb22 1888 | 256（固定） | (35938, 1, 1) = 9,200,128 | 无（区域都是 (1, 9,200,000)） |

其余相同的常数（每条管线全部列在附录 A 的 4.3 节）：h 0.0208333、c0 100、δ 0.1、幂 7、CFL 0.15、dt 3.125e-05、原点 −0.541667 ×3、网格 52³、ξ 0.001、行列式阈值 1e-4、Frobenius 上限 10、重力 0、VOXEL_ORDER 0、DIMENSION 3、NEIGHBOR_Z_RANGE 1、核系数 544539.09 / 26137876.25、EPS_H² 1.08507e-06、PST 0.1 / 0.0005、KCG / 密度扩散 / PST 开、前缀和 defrag 关、每体素上限 128、工作组 128、max_incoming 32、own pool 9,200,000、ghost pool 0 / 0、ghost 体素 0 / 0、NEIGHBOR_X_RANGE 82 = 2（correction_density）/ 3（force）、BAND_VOXEL_DISPATCH 57 = 0、BAND_SLOT_LANES 58 = 0、FAKE_BAND 59 = 0、GHOST_SELF_LAYER 85 = 0、departed 84 = 0、LEAN 1、EXT 0、COMPACT 1、DELTA 0、ρ_ref 1000、CLAMP 1、PACKED 1、POISON 0、WALL_BOUNDARY 100 = 0（simple）。

**链路的 28 条管线对 single 的 7 条**：链路多建的 21 条（ghost_send_lanes ×2、install_migrations ×2、append_departed、expand_ghost_lists、correction / density / force 各 3 种模式、force_deep_interior_scratch、correction_density_all / _boundary / _boundary_band、三条 _boundary_band 变体、density_scratch_copy 之外的……）在 K = 1 的录制里**一条都没派发**：相位 C 的 band 核线程数为 0（`boundary kernels over band voxels: 0/0/0 threads`），录制时跳过了 `vkCmdDispatch`；没有邻片就没有 ghost_send / install / expand / append。附录 A 第 3 节有两边的完整管线名单。

### 1.2 命令流的差别（附录 A 的 4.2 节）

- 链路把一步分成三个命令缓冲（相位 A：predict、update_voxel；B：correction_density_interior、force_deep_interior_scratch；C：density_scratch_copy），每帧三次 `vkQueueSubmit`（`V7_FAST_SUBMIT=1` 的 cffi 批）加 per-direction 时间线信号量，深度 2；single 一个命令缓冲、一次提交、两个 fence 轮换。每步 50–60 ms 的量级下这是微秒级的 host 开销。
- 屏障都是 COMPUTE → COMPUTE 的全局内存屏障（`0x600000000` = SHADER_STORAGE_READ | WRITE），数量：链路 7 条、single 6 条（single 在 copy 之后多录一条、链路在相位边界各一条）。
- **force 与 density copy 的先后**：链路 force 读 scratch（刚被 correction_density 写过）、copy 在它之后；single 先 copy 再 force 读 primary。读的是同样的 (ρ, P) 值，缓冲区不同（都是 9,200,000 × 8 B）。
- 缓冲区：链路 24 个设备缓冲 1412.9 MB，其中 14 个 `SHARING_MODE_CONCURRENT`（9 个 SoA + inside_particle_count / index + ghost_voxel_first_particle_id + ghost_packed_words + global_status）；single 24 个 1385.95 MB 全 EXCLUSIVE（逐个缓冲区的字节数见附录 A 第 7 节）。2-D 上 CONCURRENT 的代价测过 ≤ 1.2 %（`V5_CONCURRENT_BUFFERS=0` 对照）。

### 1.3 correction_density.comp 是一次扫描

`experiment/v7/shaders/correction_density.comp`（两树同一文件，sha256 7871141d…）：第 132–199 行一个 27 体素（`delta_x` → `delta_z` → `delta_y` 最内）邻居循环，对每个邻居同时累加 `correction_matrix`（L 的分母）、`density_gradient`、`kernel_sum` 和连续性方程的 S_i（`density_rate_diagonal` / `density_rate_off_diagonal`）；循环后的收尾（第 218–300 行）先做 correction.comp 的正则化 / 求逆 / Frobenius 封顶 / fallback 计数并写 `correction_inverse` 与 `density_gradient_kernel_sum`，再用写出的 L_i 收缩 S_i 得到 ρ_{n+1}、Tait 得 P_{n+1} 写 `density_pressure_scratch`。文件头的说明："correction.comp and density.comp in ONE neighbour traversal … rho_{n+1} and P_{n+1} differ from density.comp by summation rounding only (L_i applied once to the sum instead of to every pair)"。没有第二次扫描。两边都用它（链路 `[SimV7] V7_FUSED_CORRECTION_DENSITY=1: fused (one neighbour traversal for correction + density, band 2)`，single `V7_FUSED_CORRECTION_DENSITY -> fused`）；deep-wall skip 两边都是 off（`auto: 3-D, 0 deep-wall candidates … < 1 %`），所以用的是不带 `-DDEEP_WALL_SKIP` 的模块，也没有 marker 派发。

### 1.4 这些差别够不够解释 17 %

- 47 / 49：`in_boundary_band(coord)` = `(leading_x > 0 && …) || (trailing_x > 0 && …)`，leading_x / trailing_x 由特化常数 80 / 81 除以 NY·NZ 得到，两边都是 0，编译器把 band 测试折叠掉，interior 与 all 生成同样的代码。
- 83 / 86：`grep` 五个步进核的源码，`GHOST_LAYERS` / `REPLICA_REGION_SIZE` 只出现在 helpers.glsl 的 `packed_layer_base`、`migrant_region_offset` 和 band_compact 的辅助函数里，步进核没有调用它们（correction_density / force 调 `band_thread_voxel`，它用的是 85 = 0，且在 `BAND_VOXEL_DISPATCH == 1` 的死分支里）。声明而不引用的特化常数不改变生成的代码。
- 56 + 先后次序：是唯一真实的代码差别，但两边读的字节数相同、访问模式相同；如果有影响，方向更可能是链路占便宜（scratch 刚写过）而不是 single 慢 17 %。无法用代码断言大小，留作次要候选。
- 14 个 CONCURRENT 缓冲区：2-D 上 ≤ 1.2 %，方向也是让链路慢而不是 single 慢。

## 2. v7-rc1 → v7-rc3 影响 K = 1 路径的 host 改动（问题 2）

`git diff --stat v7-rc1 v7-rc3 -- experiment/v7`（e8 工作树；完整 patch 在 scratchpad `e8/diff_rc1_rc3_host.patch`，710 行）：

| 文件 | 改动 | 对 K = 1（simple 墙，无邻片）的影响 |
|---|---|---|
| `utils/simulator_v7.py`（+202/−）| E40：`_wall_pass_split` 属性、`wall_band_width()`、`_wall_mode_entries()`、`wall_extrapolate_interior / _boundary_band` 两条管线、`_record_wall_band_pass()`、bootstrap 与相位 B / C 里的 band 墙 pass 调用、replica 层数由 2 泛化为 `ghost_layers`（传输段、packed 计数、密度 copy 的 ghost 区域、ghost_send 线程数）、`wall_extrapolate_band` 模块按需加载；E7 B2：`configured_v7_switches()` 多报 `V7_DEST_GUARD` / `V7_DEST_GUARD_PRECHECK` | 全部以 `_wall_pass_split`（adami **且**有邻片）或 ghost 区域（K = 1 为 0）为条件：`_record_wall_band_pass` 记录为空，`_record_wall_extrapolate(scope="all")` 与 rc1 一样，且 simple 墙根本不录墙 pass；`_configured_band_widths(wall_pass=False)` 仍是 2,2,3；特化常数的取值函数没变（85 的 `_ghost_self_layer` 对 K = 1 的派发仍为 0）；CONCURRENT 名单没变；核顺序没变 |
| `utils/partition_v7.py`（+54）| `configured_band_widths(wall_pass)` 多一个 2,2,4 默认；`chain_ghost_layers()`：adami 且 K > 1 时 3 层；`_ghost_pool_layout` 按层数算 pool；去掉 rc1 的 adami K > 1 拒绝 | K = 1：`chain_ghost_layers` 返回配置值 2（与 rc1 相同），pool 布局公式在 ghost_layers = 2 时与 rc1 相同；K = 1 片的容量（own 9,200,000、ghost 0 / 0、departed 0、replica 216,320）与 rc1 一致（CPU 重建） |
| `utils/orchestrator_v7.py`（+25）| E7 B2：`connect_dest_guard_relays(self.workers)`、`dest_guard_record()` | K = 1 没有 worker（`N=1, 0 workers started`），relay 为空；`bootstrap_all` 的 5 个阶段（含 bootstrap defrag）没变 |
| `utils/sync_scheme_v7.py`（+15）| 文档 + `dest_guard_op` 说明 | 信号量、取值、等待 / 信号都没变（注释原话） |
| `utils/transport_v7.py`（+388）| E7 B2 dest guard relay、E40 三层 replica | worker 代码，K = 1 不启动 |
| `_run_v7_chain_bench.py`（−7/+3）| 去掉 `refuse_adami_chain`；结果 JSON 多 `dest_guard` 记录 | 无 |
| `utils/bench_v7.py`、`analysis/step_trace_model.py` | 墙 band pass 的计时标签、hop 说明 | 只影响时间戳的归类 |
| `_test_seam_layout.py`、`_test_transport_relay.py`、`_run_v7_soak.py` | 测试 | 无 |

Shader 侧（同一 diff）：`ghost_send.comp`、`ghost_send_lanes.comp`、`expand_ghost_lists.comp`、`install_migrations.comp` 变了（E40 三层 ghost），新增 `wall_extrapolate_band.comp`；`helpers.glsl` 的 21 行改动全在 `packed_layer_base` / `migrant_region_offset` 的 ghost pool 偏移公式和注释里；`wall_boundary.glsl` 8 行。MANIFEST 对照：**8 个步进 / bootstrap / defrag 模块（predict、update_voxel、correction_density、force、density_scratch_copy、defrag、initialize_voxelization、bootstrap_half_kick）在 rc1 树（v7single 部署）与 e8 树的 sha256 逐一相同**（附录 A 第 8 节），`wall_extrapolate.comp.spv` 也相同。e8-3d-p2s 相对 rc3 只改 `partition_v7.py`（最小 own 列数下限 3）和 `_test_chain_cuts.py`，只影响 K > 1 的切口。

另外两处非 diff 的核对：(a) CPU 重建的链路配置行与集群日志逐字相同——`compute pipelines: 28 (V7_CASCADE_FORCE=1 …) (V7_BAND_VOXEL_DISPATCH=1, lanes=64, bands 2/2/3: boundary kernels over band voxels: 0/0/0 threads vs 9,200,000) (V7_DENSITY_COPY_COMPUTE=1: density copy pass over 1 region(s), 9,200,000 slots) (V7_FUSED_CORRECTION_DENSITY: …)`；(b) v7_single 自己的 CPU 验收（`_test_single_cpu.py` 第 5 项、`_compare_recordings.py`）比较的是 v7 **未分区**的加载器案例（GHOST_LAYERS 1、replica 0），所以它说的"特化常数相同"没有覆盖 e8_S 实际跑的分区后 K = 1 片（83 = 2、86 = 216,320）；这两个常数步进核不用（1.4 节），结论不受影响，但验收的覆盖面可以记一笔。

## 3. v7_single 在 defrag_cadence = 0 时有没有启动 defrag（问题 3）

### 3.1 代码

- `experiment/v7_single/_run_single_bench.py`（e8_S 的 single 半边走的入口，`e7_lib.sh e8_single_ratio` → `e8_single_bench.py --repo … -- --case … --device 0 --warmup 500 --measure 1000 --in-flight 2 --no-defrag`）：`cadence = 0 if arguments.no_defrag else int(case.numerics.defrag_cadence)`；`sim.bootstrap(); sim.prepare_step(in_flight=…)`；循环 `sim.submit_step(with_defrag=bool(cadence) and frame % cadence == 0)`——cadence = 0 时**恒为 False**。整个进程没有别的 `submit_defrag_and_wait` 调用。
- `SphSimulatorSingle.bootstrap()` = `bootstrap_init()`（上传 + initialize_voxelization）+ `bootstrap_compute()`（correction_density_all → density copy → force_all → bootstrap_half_kick）；`record_bootstrap_*` 里没有 defrag。
- `runner_single.SingleRunner.bootstrap_all()`（cavity runner 走的那条路）才有 "Bootstrap, record the step command buffers and (as the chain does) defrag once before the first step" → `self.sim.submit_defrag_and_wait()`。bench 没有用 `SingleRunner`。
- 链路：`_run_v7_chain_bench.py` 第 240–243 行 `--no-defrag` → `defrag_cadence = args.max_steps + 1`（周期 defrag 关），第 417 行 `orch.bootstrap_all()`；`ChainOrchestratorV7.bootstrap_all()`（`orchestrator_v7.py` 第 523–560 行）在 `prepare_step_cmd_buffers()` 之后**无条件** `for sim in self.sims: sim.submit_defrag_and_wait()`，注释："Bootstrap defrag (2026-09-16): the generator's initial particle order is not voxel-sorted, and without this the first defrag_cadence frames run the interior kernels 5-9x slower (2-D 4M: phase B 23.7 ms at f1000 vs 3.5 ms at f2000)"（那是旧的非对齐生成器的数字）。

### 3.2 e8_S 的日志

`logs/n56/2026-10-22/05_e8_e8S_1722130/single_3d_n192_p{1,2,3}_{single,chain}.log`：

| | v7_single 半边 | 链路 K = 1 半边 |
|---|---|---|
| 启动 | `[SimSingle] uploaded initial state (5 payload buffers)` → `[SimSingle] bootstrap done: alive=8000000 …` → `[SimSingle] step + defrag command buffers recorded; 2 fence(s) in flight` | `[SimV7] uploaded initial state` → `bootstrap done` → `step cmd buffers recorded (phase A/B/C + 0 readback + 0 upload …)` → `V7_FAST_SUBMIT=1 …` → **`[e30] defrag_time sim0: wall_ms=74.216`**（p2 143.938，p3 25.690）→ `[ChainOrchV7] all 1 sims bootstrapped + step cmds ready` |
| 头行 | `[bench_single] … in_flight=2 warmup=500 measure=1000 defrag_cadence=0` | `[e30-config] case …: defrag_cadence=1000`（案例值；bench 用 `--no-defrag` 覆盖为 1501） |
| 窗口内 | 没有任何 defrag 行 | `[e30] defrag_events call=0 n=0 in_window=0 wall_ms=0.000 … steady_fps=19.0692 steady_fps_nodefrag=19.0692` |
| 窗口后 | — | `[e30] defrag_time sim0: wall_ms=25.827`（`_run_v7_chain_bench.py` 第 459 行，最终状态核对前的一次，在 STEADY 之后，不进窗口） |

`defrag_time` 行来自 `run_chain_v6.py` 对 `submit_defrag_and_wait` 的计时包装（第 688–723 行）：链路进程每次 defrag 都会打印一行；single 进程里没有这一行，和代码一致。

### 3.3 生成器顺序离体素顺序有多远（CPU 统计，`initial_order.py`，e8 树的加载器）

| 案例 | 粒子 | 占用体素 | 文件序里体素 id 相等的连续段 | 每体素粒子 | 位置数组每体素占的 128 B 行：文件序 / 排序后 | 相邻流体粒子换 x / y / z 体素坐标的比例 |
|---|---|---|---|---|---|---|
| 3-D 8M（cavity3d_n192）流体 | 7,077,888 | 110,592 | 1,769,472（= 16 段 / 体素） | 64 | 16.00 / 8.00（×2.00） | 0.00 % / 0.13 % / 25.00 %（z 最快） |
| 同上，全部粒子 | 8,000,000 | 125,000 | 2,000,000 | 64 | 16.00 / 8.00（×2.00） | |
| 2-D 1M（cavity2d_n1000）流体 | 1,000,000 | 40,000 | 200,000（= 5 段 / 体素） | 25 | 7.50 / 4.00（×1.88） | 0.02 % / 20.00 % / 0 |

3-D 对齐生成器（`_demo_cavity_case_aligned.py` 第 300–307 行：`meshgrid(ix, iy, iz, indexing="ij")` 后 `ravel()`）按 x 最慢、y、z 最快写；体素 id = y + z·NY + x·NY·NZ + 1（y 最快）。所以文件序里同一体素的粒子每 4 个（一条 z 向）就换体素，一个体素的 64 个粒子散在 16 段里；bootstrap defrag（defrag.comp 按体素 id 顺序、体素内按槽位顺序重排）之后它们连续。2-D 对齐生成器的段数也不是 1（5 段 / 体素，每个 x 行一段），2-D 也不是排好序的；§5.1 没有暴露它只是因为那套协议每 1000 步 defrag 一次（`_run_single_baseline_campaign.py`："windows … are multiples of the 1000-step defrag cadence, so every window contains exactly one defrag per 1000 steps"）、预热 1000 步，计时窗口从第 1001 步开始，两边都已排序；3-D 在同一协议下也是 ≈ 1（cavity3d_1m 99.64 %，cavity3d_8m_b4 99.98 %）。

### 3.4 直接验证（未做）

- 集群：给 `_run_single_bench.py` 加 `--bootstrap-defrag`（bootstrap 后 `sim.submit_defrag_and_wait()` 一次，即 `SingleRunner.bootstrap_all` 的做法），e7_lib 的 `e8_single_ratio` 加一个参数；8M 3-D 在同一张卡上 v7_single 有 / 无各 2 次 + 链路 K = 1 一次，约 10 min 整节点作业。预期：有启动 defrag 的 v7_single 回到链路的 19.1 fps 附近（§5.1 的 ≈ 1）。
- 本机 XTX（2-D 1M，几分钟）：同一开关的旁证——若 2-D 文件序（5 段 / 体素）也有可测的差距，就说明 3-D 的 16 段 / 体素差 17 % 不奇怪。
- 以上两件都等你的话；本文档没有跑任何 GPU。

## 4. 工具与文件

- CPU 重建脚本（scratchpad，`C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/`）：`dump_chain_k1.py`（在 e8 树里跑，产 `dump_chain_k1.json`）、`dump_single.py`（在 v7-single 树里跑，产 `dump_single.json`）、`compare_dumps.py`（产附录 A）、`initial_order.py`（3.3 节）、`diff_rc1_rc3_host.patch`。
- 8M 立方体的粒子文件按 `cases/aligned/cavity3d_n192/generate.txt` 在本地重建（`--objs-only`，CPU，git 忽略）；2-D 1M 同理。
- 附录 A 是 `compare_dumps.py` 的原始输出（英文）。

---

# 附录 A：CPU 对拍的原始输出


chain tree: `C:\Users\cchen\PycharmProjects\vulkan-demo-e8` (experiment/v7 of e8-3d-p2s = v7-rc3 + partition floor; the e8_S chain half); single tree: `C:\Users\cchen\PycharmProjects\vulkan-demo-v7-single` (experiment/v7_single on v7-rc1's experiment/v7; the e8_S v7_single half).

## 1. Case and capacities as each side sees them

| item | chain K = 1 (partition_v7, weights [1]) | v7_single (loader case) |
|---|---|---|
| particles | 8000000 | 8000000 |
| grid | [52, 52, 52] | [52, 52, 52] |
| origin | [-0.5416666666666666, -0.5416666666666666, -0.5416666666666666] | [-0.5416666666666666, -0.5416666666666666, -0.5416666666666666] |
| voxel_count | 140608 | 140608 |
| own_pool_size | 9200000 | 9200000 |
| total_pool_capacity | 9200001 | 9200001 |
| workgroup_size | 128 | 128 |
| max_particles_per_voxel | 128 | 128 |
| max_incoming_per_voxel | 32 | 32 |
| leading_ghost_pool_size | 0 | 0 |
| trailing_ghost_pool_size | 0 | 0 |
| leading_ghost_voxel_count | 0 | 0 |
| trailing_ghost_voxel_count | 0 | 0 |
| ghost_layers | 2 | 1 |
| departed_pool_size | 0 | 0 |
| replica_region_size | 216320 | 0 |
| wall_boundary | simple | simple |
| defrag_cadence | 1000 | 1000 |
| h | 0.020833333333333332 | 0.020833333333333332 |
| has peer (leading / trailing) | False / False | - |

## 2. Resolved switches

- chain: deep_wall [False, 'auto: 3-D, 0 deep-wall candidates at the initial state = 0.00 % of 8,000,000 particles < 1 %']; fused [True, 'one neighbour traversal for correction + density, band 2']; band_widths [2, 2, 3]; band_overlap [False, 'V7_BAND_OVERLAP=0']; switches {'V7_DENSITY_COPY_COMPUTE': 1, 'V7_GHOST_SEND_LANES': 32, 'V7_DEEP_WALL_SKIP': 'auto', 'V7_DEEP_WALL_CHECK': 0, 'V7_FUSED_CORRECTION_DENSITY': 1, 'V7_BAND_OVERLAP': 0, 'V7_DEST_GUARD': 'relay', 'V7_DEST_GUARD_PRECHECK': 'counter'}; module {'CASCADE_FORCE': True, 'BAND_VOXEL_DISPATCH': True, 'BAND_SLOT_LANES': 64, 'BAND_COMPACT': False, 'DENSITY_COPY_COMPUTE': True, 'GHOST_SEND_LANES': 32, 'PHASE_A_NO_WAIT': True, 'DEEP_WALL_SKIP': 'auto', 'FUSED': 1, 'DIAG_GHOST_SELF': ['correction', 'density']}
- single: deep_wall [False, 'auto: 3-D, 0 deep-wall candidates at the initial state = 0.00 % of 8,000,000 particles < 1 %']; fused [True, '']; band_widths [2, 2, 3]; switches {'V7_DENSITY_COPY_COMPUTE': 1, 'V7_DEEP_WALL_SKIP': 'auto', 'V7_FUSED_CORRECTION_DENSITY': 1}

## 3. Pipelines built

- chain: 29: append_departed, bootstrap_half_kick, correction_all, correction_boundary, correction_boundary_band, correction_density_all, correction_density_boundary, correction_density_boundary_band, correction_density_interior, correction_interior, defrag, density_all, density_boundary, density_boundary_band, density_deep_interior, density_scratch_copy, expand_ghost_lists, force_all, force_boundary, force_boundary_band, force_deep_interior, force_deep_interior_scratch, ghost_send_lanes_leading, ghost_send_lanes_trailing, initialize_voxelization, install_migrations_leading, install_migrations_trailing, predict, update_voxel
- single: 8: bootstrap_half_kick, correction_density_all, defrag, density_scratch_copy, force_all, initialize_voxelization, predict, update_voxel

## 4. The time step, dispatch by dispatch

### 4.1 Dispatch sequence, paired by kernel (position = order inside the step on each side)

| chain pos | chain K = 1 pipeline | single pos | v7_single pipeline | SPIR-V file (sha256[:12], bytes; identical on both sides) | local size | dispatch (x,y,z) = threads | spec ids that differ (chain vs single) |
|---|---|---|---|---|---|---|---|
| 1 | predict | 1 | predict | predict.comp.spv 099b129f66ea 6392 | 128 | (71875, 1, 1) = 9,200,000 | 83 GHOST_LAYERS: 2 vs 1; 86 REPLICA_REGION_SIZE: 216320 vs 0 |
| 2 | update_voxel | 2 | update_voxel | update_voxel.comp.spv 024e04287997 4916 | 128 | (1099, 1, 1) = 140,672 | 83 GHOST_LAYERS: 2 vs 1; 86 REPLICA_REGION_SIZE: 216320 vs 0 |
| 3 | correction_density_interior | 3 | correction_density_all | correction_density.comp.spv 7871141dcdec 57868 | 128 | (71875, 1, 1) = 9,200,000 | 47 CORRECTION_MODE: 1 vs 0; 83 GHOST_LAYERS: 2 vs 1; 86 REPLICA_REGION_SIZE: 216320 vs 0 |
| 4 | force_deep_interior_scratch | 5 | force_all | force.comp.spv 3a635216025f 46776 | 128 | (71875, 1, 1) = 9,200,000 | 49 FORCE_MODE: 1 vs 0; 56 FORCE_DENSITY_SOURCE: 1 vs 0; 83 GHOST_LAYERS: 2 vs 1; 86 REPLICA_REGION_SIZE: 216320 vs 0 |
| 5 | density_scratch_copy | 4 | density_scratch_copy | density_scratch_copy.comp.spv 50e9e5f6fb22 1888 | 256 | (35938, 1, 1) = 9,200,128 | none |

### 4.2 Full command streams of the step (barriers, copies, fills included)

| # | chain K = 1 (phase A + B + C command buffers) | v7_single (one step command buffer) |
|---|---|---|
| 1 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 2 | dispatch **predict** (71875, 1, 1) (predict.comp.spv) | dispatch **predict** (71875, 1, 1) (predict.comp.spv) |
| 3 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 4 | dispatch **update_voxel** (1099, 1, 1) (update_voxel.comp.spv) | dispatch **update_voxel** (1099, 1, 1) (update_voxel.comp.spv) |
| 5 | *end of phase A command buffer* | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 6 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | dispatch **correction_density_all** (71875, 1, 1) (correction_density.comp.spv) |
| 7 | dispatch **correction_density_interior** (71875, 1, 1) (correction_density.comp.spv) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 8 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | dispatch **density_scratch_copy** (35938, 1, 1) (density_scratch_copy.comp.spv) |
| 9 | dispatch **force_deep_interior_scratch** (71875, 1, 1) (force.comp.spv) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 10 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 11 | *end of phase B command buffer* | dispatch **force_all** (71875, 1, 1) (force.comp.spv) |
| 12 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |  |
| 13 | dispatch **density_scratch_copy** (35938, 1, 1) (density_scratch_copy.comp.spv) |  |
| 14 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |  |
| 15 | *end of phase C command buffer* |  |

### 4.3 Every specialization constant of the step pipelines (chain value, marker where v7_single differs)

#### dispatch 1: chain `predict` vs single `predict` (predict.comp.spv = predict.comp.spv: same sha256)

```
  0 SMOOTHING_LENGTH                     0.020833333333333332
  1 SPEED_OF_SOUND                       100.0
  2 DELTA_COEFFICIENT                    0.1
  4 POWER_PARAMETER                      7.0
  5 CFL_NUMBER                           0.15
  6 TIMESTEP                             3.125e-05
  7 OWN_ORIGIN_X                         -0.5416666666666666
  8 OWN_ORIGIN_Y                         -0.5416666666666666
  9 OWN_ORIGIN_Z                         -0.5416666666666666
 10 STRICT_BIT_EXACT                     0
 11 GRID_DIMENSION_X                     52
 12 GRID_DIMENSION_Y                     52
 13 GRID_DIMENSION_Z                     52
 14 REGULARIZATION_XI                    0.001
 15 REGULARIZATION_DETERMINANT_THRESHOLD 0.0001
 16 REGULARIZATION_MAX_FROBENIUS_NORM    10.0
 17 GRAVITY_X                            0.0
 18 GRAVITY_Y                            0.0
 19 GRAVITY_Z                            0.0
 20 VOXEL_ORDER                          0
 30 DIMENSION                            3
 31 NEIGHBOR_Z_RANGE                     1
 32 KERNEL_COEFFICIENT                   544539.0884923345
 33 KERNEL_GRADIENT_COEFFICIENT          26137876.247632056
 40 EPS_H_SQUARED                        1.0850694444444442e-06
 41 PST_MAIN_SHIFT_COEFFICIENT           0.1
 42 PST_ANTI_SHIFT_COEFFICIENT           0.0005
 43 USE_KCG_CORRECTION                   1
 44 USE_DENSITY_DIFFUSION                1
 45 USE_PST                              1
 46 USE_PREFIX_SUM_DEFRAG                0
 50 MAX_PARTICLES_PER_VOXEL              128
 51 WORKGROUP_SIZE                       128
 52 MAX_INCOMING_PER_VOXEL               32
 53 OWN_POOL_SIZE                        9200000
 54 LEADING_GHOST_POOL_SIZE              0
 55 TRAILING_GHOST_POOL_SIZE             0
 80 LEADING_GHOST_VOXEL_COUNT            0
 81 TRAILING_GHOST_VOXEL_COUNT           0
 83 GHOST_LAYERS                         2  <-- other side 1
 84 DEPARTED_POOL_SIZE                   0
 86 REPLICA_REGION_SIZE                  216320  <-- other side 0
 87 LEAN_TRANSPORT                       1
 88 TRANSPORT_EXTENSION                  0
 89 COMPACT_GHOST_LISTS                  1
 95 DELTA_DENSITY                        0
 96 REFERENCE_DENSITY                    1000.0
 97 INIT_SEAM_CLAMP                      1
 98 PACKED_REPLICAS                      1
 99 DIAGNOSTIC_POISON_INNER_REPLICA      0
100 WALL_BOUNDARY                        0
```

#### dispatch 2: chain `update_voxel` vs single `update_voxel` (update_voxel.comp.spv = update_voxel.comp.spv: same sha256)

```
  0 SMOOTHING_LENGTH                     0.020833333333333332
  1 SPEED_OF_SOUND                       100.0
  2 DELTA_COEFFICIENT                    0.1
  4 POWER_PARAMETER                      7.0
  5 CFL_NUMBER                           0.15
  6 TIMESTEP                             3.125e-05
  7 OWN_ORIGIN_X                         -0.5416666666666666
  8 OWN_ORIGIN_Y                         -0.5416666666666666
  9 OWN_ORIGIN_Z                         -0.5416666666666666
 10 STRICT_BIT_EXACT                     0
 11 GRID_DIMENSION_X                     52
 12 GRID_DIMENSION_Y                     52
 13 GRID_DIMENSION_Z                     52
 14 REGULARIZATION_XI                    0.001
 15 REGULARIZATION_DETERMINANT_THRESHOLD 0.0001
 16 REGULARIZATION_MAX_FROBENIUS_NORM    10.0
 17 GRAVITY_X                            0.0
 18 GRAVITY_Y                            0.0
 19 GRAVITY_Z                            0.0
 20 VOXEL_ORDER                          0
 30 DIMENSION                            3
 31 NEIGHBOR_Z_RANGE                     1
 32 KERNEL_COEFFICIENT                   544539.0884923345
 33 KERNEL_GRADIENT_COEFFICIENT          26137876.247632056
 40 EPS_H_SQUARED                        1.0850694444444442e-06
 41 PST_MAIN_SHIFT_COEFFICIENT           0.1
 42 PST_ANTI_SHIFT_COEFFICIENT           0.0005
 43 USE_KCG_CORRECTION                   1
 44 USE_DENSITY_DIFFUSION                1
 45 USE_PST                              1
 46 USE_PREFIX_SUM_DEFRAG                0
 50 MAX_PARTICLES_PER_VOXEL              128
 51 WORKGROUP_SIZE                       128
 52 MAX_INCOMING_PER_VOXEL               32
 53 OWN_POOL_SIZE                        9200000
 54 LEADING_GHOST_POOL_SIZE              0
 55 TRAILING_GHOST_POOL_SIZE             0
 80 LEADING_GHOST_VOXEL_COUNT            0
 81 TRAILING_GHOST_VOXEL_COUNT           0
 83 GHOST_LAYERS                         2  <-- other side 1
 84 DEPARTED_POOL_SIZE                   0
 86 REPLICA_REGION_SIZE                  216320  <-- other side 0
 87 LEAN_TRANSPORT                       1
 88 TRANSPORT_EXTENSION                  0
 89 COMPACT_GHOST_LISTS                  1
 95 DELTA_DENSITY                        0
 96 REFERENCE_DENSITY                    1000.0
 97 INIT_SEAM_CLAMP                      1
 98 PACKED_REPLICAS                      1
 99 DIAGNOSTIC_POISON_INNER_REPLICA      0
100 WALL_BOUNDARY                        0
```

#### dispatch 3: chain `correction_density_interior` vs single `correction_density_all` (correction_density.comp.spv = correction_density.comp.spv: same sha256)

```
  0 SMOOTHING_LENGTH                     0.020833333333333332
  1 SPEED_OF_SOUND                       100.0
  2 DELTA_COEFFICIENT                    0.1
  4 POWER_PARAMETER                      7.0
  5 CFL_NUMBER                           0.15
  6 TIMESTEP                             3.125e-05
  7 OWN_ORIGIN_X                         -0.5416666666666666
  8 OWN_ORIGIN_Y                         -0.5416666666666666
  9 OWN_ORIGIN_Z                         -0.5416666666666666
 10 STRICT_BIT_EXACT                     0
 11 GRID_DIMENSION_X                     52
 12 GRID_DIMENSION_Y                     52
 13 GRID_DIMENSION_Z                     52
 14 REGULARIZATION_XI                    0.001
 15 REGULARIZATION_DETERMINANT_THRESHOLD 0.0001
 16 REGULARIZATION_MAX_FROBENIUS_NORM    10.0
 17 GRAVITY_X                            0.0
 18 GRAVITY_Y                            0.0
 19 GRAVITY_Z                            0.0
 20 VOXEL_ORDER                          0
 30 DIMENSION                            3
 31 NEIGHBOR_Z_RANGE                     1
 32 KERNEL_COEFFICIENT                   544539.0884923345
 33 KERNEL_GRADIENT_COEFFICIENT          26137876.247632056
 40 EPS_H_SQUARED                        1.0850694444444442e-06
 41 PST_MAIN_SHIFT_COEFFICIENT           0.1
 42 PST_ANTI_SHIFT_COEFFICIENT           0.0005
 43 USE_KCG_CORRECTION                   1
 44 USE_DENSITY_DIFFUSION                1
 45 USE_PST                              1
 46 USE_PREFIX_SUM_DEFRAG                0
 47 CORRECTION_MODE                      1  <-- other side 0
 50 MAX_PARTICLES_PER_VOXEL              128
 51 WORKGROUP_SIZE                       128
 52 MAX_INCOMING_PER_VOXEL               32
 53 OWN_POOL_SIZE                        9200000
 54 LEADING_GHOST_POOL_SIZE              0
 55 TRAILING_GHOST_POOL_SIZE             0
 57 BAND_VOXEL_DISPATCH                  0
 58 BAND_SLOT_LANES                      0
 59 FAKE_BAND_COLUMN                     0
 80 LEADING_GHOST_VOXEL_COUNT            0
 81 TRAILING_GHOST_VOXEL_COUNT           0
 82 NEIGHBOR_X_RANGE                     2
 83 GHOST_LAYERS                         2  <-- other side 1
 84 DEPARTED_POOL_SIZE                   0
 85 GHOST_SELF_LAYER                     0
 86 REPLICA_REGION_SIZE                  216320  <-- other side 0
 87 LEAN_TRANSPORT                       1
 88 TRANSPORT_EXTENSION                  0
 89 COMPACT_GHOST_LISTS                  1
 95 DELTA_DENSITY                        0
 96 REFERENCE_DENSITY                    1000.0
 97 INIT_SEAM_CLAMP                      1
 98 PACKED_REPLICAS                      1
 99 DIAGNOSTIC_POISON_INNER_REPLICA      0
100 WALL_BOUNDARY                        0
```

#### dispatch 4: chain `force_deep_interior_scratch` vs single `force_all` (force.comp.spv = force.comp.spv: same sha256)

```
  0 SMOOTHING_LENGTH                     0.020833333333333332
  1 SPEED_OF_SOUND                       100.0
  2 DELTA_COEFFICIENT                    0.1
  4 POWER_PARAMETER                      7.0
  5 CFL_NUMBER                           0.15
  6 TIMESTEP                             3.125e-05
  7 OWN_ORIGIN_X                         -0.5416666666666666
  8 OWN_ORIGIN_Y                         -0.5416666666666666
  9 OWN_ORIGIN_Z                         -0.5416666666666666
 10 STRICT_BIT_EXACT                     0
 11 GRID_DIMENSION_X                     52
 12 GRID_DIMENSION_Y                     52
 13 GRID_DIMENSION_Z                     52
 14 REGULARIZATION_XI                    0.001
 15 REGULARIZATION_DETERMINANT_THRESHOLD 0.0001
 16 REGULARIZATION_MAX_FROBENIUS_NORM    10.0
 17 GRAVITY_X                            0.0
 18 GRAVITY_Y                            0.0
 19 GRAVITY_Z                            0.0
 20 VOXEL_ORDER                          0
 30 DIMENSION                            3
 31 NEIGHBOR_Z_RANGE                     1
 32 KERNEL_COEFFICIENT                   544539.0884923345
 33 KERNEL_GRADIENT_COEFFICIENT          26137876.247632056
 40 EPS_H_SQUARED                        1.0850694444444442e-06
 41 PST_MAIN_SHIFT_COEFFICIENT           0.1
 42 PST_ANTI_SHIFT_COEFFICIENT           0.0005
 43 USE_KCG_CORRECTION                   1
 44 USE_DENSITY_DIFFUSION                1
 45 USE_PST                              1
 46 USE_PREFIX_SUM_DEFRAG                0
 49 FORCE_MODE                           1  <-- other side 0
 50 MAX_PARTICLES_PER_VOXEL              128
 51 WORKGROUP_SIZE                       128
 52 MAX_INCOMING_PER_VOXEL               32
 53 OWN_POOL_SIZE                        9200000
 54 LEADING_GHOST_POOL_SIZE              0
 55 TRAILING_GHOST_POOL_SIZE             0
 56 FORCE_DENSITY_SOURCE                 1  <-- other side 0
 57 BAND_VOXEL_DISPATCH                  0
 58 BAND_SLOT_LANES                      0
 59 FAKE_BAND_COLUMN                     0
 80 LEADING_GHOST_VOXEL_COUNT            0
 81 TRAILING_GHOST_VOXEL_COUNT           0
 82 NEIGHBOR_X_RANGE                     3
 83 GHOST_LAYERS                         2  <-- other side 1
 84 DEPARTED_POOL_SIZE                   0
 86 REPLICA_REGION_SIZE                  216320  <-- other side 0
 87 LEAN_TRANSPORT                       1
 88 TRANSPORT_EXTENSION                  0
 89 COMPACT_GHOST_LISTS                  1
 95 DELTA_DENSITY                        0
 96 REFERENCE_DENSITY                    1000.0
 97 INIT_SEAM_CLAMP                      1
 98 PACKED_REPLICAS                      1
 99 DIAGNOSTIC_POISON_INNER_REPLICA      0
100 WALL_BOUNDARY                        0
```

#### dispatch 5: chain `density_scratch_copy` vs single `density_scratch_copy` (density_scratch_copy.comp.spv = density_scratch_copy.comp.spv: same sha256)

```
101 COPY_REGION_0_FIRST_SLOT             1
102 COPY_REGION_0_SLOT_COUNT             9200000
103 COPY_REGION_1_FIRST_SLOT             0
104 COPY_REGION_1_SLOT_COUNT             0
105 COPY_REGION_2_FIRST_SLOT             0
106 COPY_REGION_2_SLOT_COUNT             0
107 COPY_REGION_3_FIRST_SLOT             0
108 COPY_REGION_3_SLOT_COUNT             0
```

## 5. Bootstrap command streams

### bootstrap_init

| # | chain K = 1 | v7_single |
|---|---|---|
| 1 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 2 | dispatch **initialize_voxelization** (71875, 1, 1) (initialize_voxelization.comp.spv) | dispatch **initialize_voxelization** (71875, 1, 1) (initialize_voxelization.comp.spv) |

### bootstrap_compute

| # | chain K = 1 | v7_single |
|---|---|---|
| 1 | dispatch **correction_density_all** (71875, 1, 1) (correction_density.comp.spv) | dispatch **correction_density_all** (71875, 1, 1) (correction_density.comp.spv) |
| 2 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 3 | dispatch **density_scratch_copy** (35938, 1, 1) (density_scratch_copy.comp.spv) | dispatch **density_scratch_copy** (35938, 1, 1) (density_scratch_copy.comp.spv) |
| 4 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 5 | dispatch **force_all** (71875, 1, 1) (force.comp.spv) | dispatch **force_all** (71875, 1, 1) (force.comp.spv) |
| 6 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 7 | dispatch **bootstrap_half_kick** (71875, 1, 1) (bootstrap_half_kick.comp.spv) | dispatch **bootstrap_half_kick** (71875, 1, 1) (bootstrap_half_kick.comp.spv) |

## 6. Defrag command streams

| # | chain K = 1 | v7_single |
|---|---|---|
| 1 | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) | barrier COMPUTE(0x600000000) -> COMPUTE(0x600000000) |
| 2 | fill buffer:defrag_scratch_counter off 0 size 4 = 0 | fill buffer:defrag_scratch_counter off 0 size 4 = 0 |
| 3 | fill scratch:position_voxel_id off 0 size 147200016 = 0 | fill scratch:position_voxel_id off 0 size 147200016 = 0 |
| 4 | fill scratch:density_pressure off 0 size 73600008 = 0 | fill scratch:density_pressure off 0 size 73600008 = 0 |
| 5 | fill scratch:velocity_mass off 0 size 147200016 = 0 | fill scratch:velocity_mass off 0 size 147200016 = 0 |
| 6 | fill scratch:acceleration off 0 size 147200016 = 0 | fill scratch:acceleration off 0 size 147200016 = 0 |
| 7 | fill scratch:shift off 0 size 147200016 = 0 | fill scratch:shift off 0 size 147200016 = 0 |
| 8 | fill scratch:material off 0 size 36800004 = 0 | fill scratch:material off 0 size 36800004 = 0 |
| 9 | fill scratch:correction_inverse off 0 size 294400032 = 0 | fill scratch:correction_inverse off 0 size 294400032 = 0 |
| 10 | fill scratch:density_gradient_kernel_sum off 0 size 147200016 = 0 | fill scratch:density_gradient_kernel_sum off 0 size 147200016 = 0 |
| 11 | fill scratch:extension_fields off 0 size 147200016 = 0 | fill scratch:extension_fields off 0 size 147200016 = 0 |
| 12 | barrier TRANSFER(0x1000) -> COMPUTE(0x600000000) | barrier TRANSFER(0x1000) -> COMPUTE(0x600000000) |
| 13 | dispatch **defrag** (1099, 1, 1) (defrag.comp.spv) | dispatch **defrag** (1099, 1, 1) (defrag.comp.spv) |
| 14 | barrier COMPUTE(0x400000000) -> TRANSFER(0x800) | barrier COMPUTE(0x400000000) -> TRANSFER(0x800) |
| 15 | copy scratch:position_voxel_id -> buffer:position_voxel_id [[0, 0, 147200016]] | copy scratch:position_voxel_id -> buffer:position_voxel_id [[0, 0, 147200016]] |
| 16 | copy scratch:density_pressure -> buffer:density_pressure [[0, 0, 73600008]] | copy scratch:density_pressure -> buffer:density_pressure [[0, 0, 73600008]] |
| 17 | copy scratch:velocity_mass -> buffer:velocity_mass [[0, 0, 147200016]] | copy scratch:velocity_mass -> buffer:velocity_mass [[0, 0, 147200016]] |
| 18 | copy scratch:acceleration -> buffer:acceleration [[0, 0, 147200016]] | copy scratch:acceleration -> buffer:acceleration [[0, 0, 147200016]] |
| 19 | copy scratch:shift -> buffer:shift [[0, 0, 147200016]] | copy scratch:shift -> buffer:shift [[0, 0, 147200016]] |
| 20 | copy scratch:material -> buffer:material [[0, 0, 36800004]] | copy scratch:material -> buffer:material [[0, 0, 36800004]] |
| 21 | copy scratch:correction_inverse -> buffer:correction_inverse [[0, 0, 294400032]] | copy scratch:correction_inverse -> buffer:correction_inverse [[0, 0, 294400032]] |
| 22 | copy scratch:density_gradient_kernel_sum -> buffer:density_gradient_kernel_sum [[0, 0, 147200016]] | copy scratch:density_gradient_kernel_sum -> buffer:density_gradient_kernel_sum [[0, 0, 147200016]] |
| 23 | copy scratch:extension_fields -> buffer:extension_fields [[0, 0, 147200016]] | copy scratch:extension_fields -> buffer:extension_fields [[0, 0, 147200016]] |
| 24 | copy buffer:defrag_scratch_counter -> buffer:global_status [[0, 0, 4]] | copy buffer:defrag_scratch_counter -> buffer:global_status [[0, 0, 4]] |
| 25 | fill buffer:global_status off 48 size 4 = 0 | barrier TRANSFER(0x1000) -> COMPUTE(0x600000000) |
| 26 | barrier TRANSFER(0x1000) -> COMPUTE(0x600000000) |  |

## 7. Buffers (name, set, binding, bytes, sharing)

| buffer | chain set.binding | chain bytes | chain sharing | single set.binding | single bytes | single sharing |
|---|---|---|---|---|---|---|
| acceleration | 0.4 | 147,200,016 | CONCURRENT | 0.4 | 147,200,016 | EXCLUSIVE |
| band_compact_list | 1.6 | 4 | EXCLUSIVE | 1.6 | 4 | EXCLUSIVE |
| band_compact_meta | 3.9 | 128 | EXCLUSIVE | 3.9 | 16 | EXCLUSIVE |
| correction_inverse | 0.7 | 294,400,032 | CONCURRENT | 0.7 | 294,400,032 | EXCLUSIVE |
| defrag_scratch_counter | 3.8 | 4 | EXCLUSIVE | 3.8 | 4 | EXCLUSIVE |
| density_gradient_kernel_sum | 0.8 | 147,200,016 | CONCURRENT | 0.8 | 147,200,016 | EXCLUSIVE |
| density_pressure | 0.1 | 73,600,008 | CONCURRENT | 0.1 | 73,600,008 | EXCLUSIVE |
| density_pressure_scratch | 0.2 | 73,600,008 | EXCLUSIVE | 0.2 | 73,600,008 | EXCLUSIVE |
| extension_fields | 0.9 | 147,200,016 | CONCURRENT | 0.9 | 147,200,016 | EXCLUSIVE |
| ghost_packed_words | 1.7 | 27,688,960 | CONCURRENT | 1.7 | 4 | EXCLUSIVE |
| ghost_voxel_first_particle_id | 1.5 | 562,436 | CONCURRENT | 1.5 | 4 | EXCLUSIVE |
| global_status | 3.0 | 160 | CONCURRENT | 3.0 | 160 | EXCLUSIVE |
| incoming_particle_count | 1.1 | 562,436 | EXCLUSIVE | 1.1 | 562,436 | EXCLUSIVE |
| incoming_particle_index | 1.3 | 17,997,952 | EXCLUSIVE | 1.3 | 17,997,952 | EXCLUSIVE |
| inside_particle_count | 1.0 | 562,436 | CONCURRENT | 1.0 | 562,436 | EXCLUSIVE |
| inside_particle_index | 1.2 | 71,991,808 | CONCURRENT | 1.2 | 71,991,808 | EXCLUSIVE |
| material | 0.6 | 36,800,004 | CONCURRENT | 0.6 | 36,800,004 | EXCLUSIVE |
| material_parameters | 3.7 | 144 | EXCLUSIVE | 3.7 | 144 | EXCLUSIVE |
| pool_health | 3.1 | 16 | EXCLUSIVE | 3.1 | 16 | EXCLUSIVE |
| position_voxel_id | 0.0 | 147,200,016 | CONCURRENT | 0.0 | 147,200,016 | EXCLUSIVE |
| shift | 0.5 | 147,200,016 | CONCURRENT | 0.5 | 147,200,016 | EXCLUSIVE |
| velocity_mass | 0.3 | 147,200,016 | CONCURRENT | 0.3 | 147,200,016 | EXCLUSIVE |
| voxel_base_offset | 1.4 | 562,436 | EXCLUSIVE | 1.4 | 562,436 | EXCLUSIVE |
| wall_dummy_velocity | 0.10 | 16 | EXCLUSIVE | 0.10 | 16 | EXCLUSIVE |

## 8. Manifest check of the step / bootstrap / defrag modules

| module | chain tree sha256 | single tree sha256 | same |
|---|---|---|---|
| append_departed.comp.spv | 7823e926e76ae19a | 7823e926e76ae19a | yes |
| band_compact.comp.spv | e22c63cb3b498b7a | e22c63cb3b498b7a | yes |
| bootstrap_half_kick.comp.spv | ec24be0f2d0b48f8 | ec24be0f2d0b48f8 | yes |
| correction.comp.spv | ee9ff135900b69f8 | ee9ff135900b69f8 | yes |
| correction_deep_wall_skip.comp.spv | e3de58bdc5e00f89 | e3de58bdc5e00f89 | yes |
| correction_density.comp.spv | 7871141dcdec3681 | 7871141dcdec3681 | yes |
| correction_density_deep_wall_skip.comp.spv | f5faeac0e14610d9 | f5faeac0e14610d9 | yes |
| deep_wall_marker.comp.spv | 4c9b4096d57aab44 | 4c9b4096d57aab44 | yes |
| defrag.comp.spv | 52d7d9b3a09cf4ce | 52d7d9b3a09cf4ce | yes |
| density.comp.spv | 949bc2d1a5d85ca7 | 949bc2d1a5d85ca7 | yes |
| density_deep_wall_skip.comp.spv | 8c99cc51d9106584 | 8c99cc51d9106584 | yes |
| density_scratch_copy.comp.spv | 50e9e5f6fb22a599 | 50e9e5f6fb22a599 | yes |
| expand_ghost_lists.comp.spv | 4087617c2a4a14b5 | 728d4e4742a0f156 | NO |
| force.comp.spv | 3a635216025f245f | 3a635216025f245f | yes |
| force_segment.comp.spv | faa772d587f1235a | faa772d587f1235a | yes |
| ghost_send.comp.spv | 0a73e65d75e2a6c6 | de0feb09c6618201 | NO |
| ghost_send_lanes.comp.spv | 60e3eb2eba095880 | ec7da52efc61aed9 | NO |
| initialize_voxelization.comp.spv | 62f42d33a26a7087 | 62f42d33a26a7087 | yes |
| install_migrations.comp.spv | 88b442d4059c5d37 | d7f17d7808d77f9e | NO |
| predict.comp.spv | 099b129f66ea93c4 | 099b129f66ea93c4 | yes |
| update_voxel.comp.spv | 024e04287997e22a | 024e04287997e22a | yes |
| wall_extrapolate.comp.spv | 23c0d0a2410d2131 | 23c0d0a2410d2131 | yes |
| wall_extrapolate_band.comp.spv | d9563e603d1abf06 | - | NO |
