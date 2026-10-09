# v7_single:与 v7 同 kernel 的纯单卡构建(2026-10-09 起)

接 [`v7_perf.md`](v7_perf.md)(v7-rc1,附注标签 `v7-rc1` = d0c8dcb)。`experiment/v7_single` 是一个只有单卡 host 代码的求解器:它不复制、不改任何 shader,直接加载 `experiment/v7/shaders/spv` 的 SPIR-V(启动时按 `MANIFEST.txt` 逐文件校验 sha256,不一致就退出),host 侧只做纯单卡该做的事。用途:(1) 论文 §3 的"单卡求解器";(2) 测出 v7 单卡模式(`SphSimulatorV7._record_step_single_cmd`)与 v7 链式 K = 1 相对它的吞吐比(§5.1);(3) 用它跑 1000² 方腔的精度验证(§6)。η 的参照仍是 v7 链式 K = 1,不改;集群脚本不改。本文里的旧 V0 指 `utils/sph`,与 v7_single 无关。

分支 `v7-single`(从 `v7-rc1` 开出,worktree `vulkan-demo-v7-single`),`experiment/v7` 一个文件都没有改。

## 结论(TL;DR)

待填(验收与测量完成后)。

## 清单:v7 单卡模式相对 v7_single 多出的 host 侧内容

下面"V"= v7-rc1 的单卡模式:`load_case_v7` 给出的退化 slab(没有 peer、ghost 列数 0、ghost pool 0、departed pool 0)直接交给 `SphSimulatorV7`,`bootstrap()` + `prepare_step_single_cmd_buffer()` + `submit_step_single_and_wait()`,即 `experiment/v7/_run_v7_single_bench.py` 和 `_run_single_baseline_bench.py` 走的路径。"S"= `experiment/v7_single`。"C"= v7 链式 K = 1(`compute_chain_partition` 一个 slab + `ChainOrchestratorV7`),只在与 S 有区别的条目里顺带列出,它不是本清单的对象。

数字以 2-D 1M(`cases/lid_driven_cavity_2d_gen`,1 046 529 个粒子,pool_size 1 203 584,网格 206 × 206 × 1 = 42 436 体素,workgroup 128)为例,用 `SphSimulatorV7` 的 `_build_buffer_specs` / `_global_entries` 在 CPU 上算出(`experiment/v7_single/_compare_recordings.py --inventory` 可以重算任何算例)。

### 1. Vulkan 上下文

| 项 | V(`VulkanContextV7.create`) | S(`experiment/v7_single/utils/vulkan_context_single.py`) | 影响结果? |
|---|---|---|---|
| 实例 | apiVersion 1.3;验证层按参数(`create()` 的默认是开,所有 runner 都传 `enable_validation=False` 或 `--validation`) | 同样的实例;验证层默认关,`--validation` 开并计数消息 | 否 |
| 物理设备顺序 | discrete-first 稳定排序 | 同样的排序(同一张卡用同一个 index;测量按 UUID 固定) | 否 |
| 队列 | 3 条:compute 族 1 条 + transfer-only 族 2 条(`V7_SPLIT_TRANSFER_QUEUES` 默认 1;5090 的 transfer 族有 2 条队列),`_find_transfer_queue_family` 选族 | 1 条 compute 队列 | 否(单卡路径从不向 transfer 队列提交) |
| 命令池 | 2 个(compute 族、transfer 族) | 1 个 | 否 |
| 设备特性 | `timelineSemaphore`、`synchronization2`、`vertexPipelineStoresAndAtomics` | `synchronization2`(屏障用 `vkCmdPipelineBarrier2`;没有 timeline、没有渲染) | 否 |
| 设备扩展 | 可选的 calibrated timestamps(step trace) | 无 | 否 |

### 2. 同步对象与提交

| 项 | V | S | 影响结果? |
|---|---|---|---|
| timeline 信号量 | `make_sync_scheme("aggregated")` 建 1 个 timeline 信号量(单卡路径从不等待或发信号) | 无 | 否 |
| 驱动提交锁 | 每张物理卡一个 `threading.Lock`(`V7_SUBMIT_LOCK_SCOPE`),单卡路径不用 | 无 | 否 |
| fast submit | `_fast_submit_prepare` 只在链式路径(`prepare_step_cmd_buffers`)里建 cffi 批 | 无 | 否 |
| 每步的 fence | `ctx.submit_and_wait`:每步 `vkCreateFence` + `vkQueueSubmit` + `vkWaitForFences` + `vkDestroyFence`(1 帧在飞);N 帧在飞由 `_run_single_baseline_bench.py` 的 `FenceRing` 在求解器外面做 | 求解器自带 N 个持久 fence 的环(N = 1 就是每步一个 fence;`vkResetFences` 复用,不再每步创建销毁);N 帧在飞时第 k 步复用第 k − N 步的 fence,其他与 `FenceRing` 相同 | 否(host 侧;depth-1 的提交气泡可能略小,见测量) |
| 线程 | 单卡路径没有 worker 线程(C 有 0 个 worker,但 orchestrator 的 3 批提交 + timeline 等待还在) | 无 | 否 |

### 3. 缓冲:共享模式、大小、内存

V 的 `_allocate_buffers` 把 `_CONCURRENT_BUFFER_NAMES` 里存在的 14 个缓冲建成 `SHARING_MODE_CONCURRENT`(compute + transfer 两个队列族;`V7_CONCURRENT_BUFFERS=0` 是它的诊断开关),其余 `EXCLUSIVE`。S 的每个缓冲都是 `EXCLUSIVE`。缓冲的 set / binding 编号两边相同(SPIR-V 相同,描述符布局必须相同),只有下面标出的 stub 大小不同。

| set.binding | 名字 | V 大小(1M) | V 共享模式 | S | 说明 |
|---|---|---|---|---|---|
| 0.0 | position_voxel_id | 19 257 360 | CONCURRENT | 同大小,EXCLUSIVE | 16 B × (pool_size + 1) |
| 0.1 | density_pressure | 9 628 680 | CONCURRENT | 同,EXCLUSIVE | |
| 0.2 | density_pressure_scratch | 9 628 680 | EXCLUSIVE | 同 | |
| 0.3 | velocity_mass | 19 257 360 | CONCURRENT | 同,EXCLUSIVE | |
| 0.4 | acceleration | 19 257 360 | CONCURRENT | 同,EXCLUSIVE | |
| 0.5 | shift | 19 257 360 | CONCURRENT | 同,EXCLUSIVE | |
| 0.6 | material | 4 814 340 | CONCURRENT | 同,EXCLUSIVE | |
| 0.7 | correction_inverse | 38 514 720 | CONCURRENT | 同,EXCLUSIVE | |
| 0.8 | density_gradient_kernel_sum | 19 257 360 | CONCURRENT | 同,EXCLUSIVE | |
| 0.9 | extension_fields | 19 257 360 | CONCURRENT | 同,EXCLUSIVE | 没有 kernel 读它;canonical_dump 用它存全局 id |
| 0.10 | wall_dummy_velocity | 16(simple)/ 16 B × pool(adami) | EXCLUSIVE | 同 | E37 |
| 1.0 | inside_particle_count | 169 748 | CONCURRENT | 同,EXCLUSIVE | 4 B × (体素数 + 1) |
| 1.1 | incoming_particle_count | 169 748 | EXCLUSIVE | 同 | |
| 1.2 | inside_particle_index | 16 295 808 | CONCURRENT | 同,EXCLUSIVE | |
| 1.3 | incoming_particle_index | 2 715 968 | EXCLUSIVE | 同 | |
| 1.4 | voxel_base_offset | 169 748 | EXCLUSIVE | 同 | 只有 `USE_PREFIX_SUM_DEFRAG`(关)读 |
| 1.5 | ghost_voxel_first_particle_id | 169 748 | CONCURRENT | **4 B stub**,EXCLUSIVE | 只有 ghost_send / expand_ghost_lists 读写,单卡从不派发它们 |
| 1.6 | band_compact_list | 4 | EXCLUSIVE | 4 | compact band 派发(关)专用 |
| 1.7 | ghost_packed_words | 4 | CONCURRENT | 4,EXCLUSIVE | replica_region_size = 0 时 V 也是 4 B |
| 1.8 | deep_wall_voxel_flag | 8 B × (体素数 + 1),只在 B4 开启的 slab 上 | EXCLUSIVE | 同规则 | |
| 3.0 | global_status | 160 | CONCURRENT | 160,EXCLUSIVE | 40 uint,读回 drift 与所有 overflow 计数 |
| 3.1 | pool_health | 16 | EXCLUSIVE | 16 | 只有 install_migrations 写;S 保留(描述符布局),不读 |
| 3.7 | material_parameters | 48 B × 材料数 | EXCLUSIVE | 同 | |
| 3.8 | defrag_scratch_counter | 4 | EXCLUSIVE | 同 | |
| 3.9 | band_compact_meta | 128 | EXCLUSIVE | **16 B stub** | 只有 compact band 派发用(indirect) |
| 3.10 | deep_wall_skip_record | 16(不记录)/ 8 B × pool(记录),只在 B4 开启的 slab 上 | EXCLUSIVE | 同规则 | canonical_dump 的记录钩子两边都有 |
| — | 9 个 defrag scratch 孪生(set 0 的 0/1/3/4/5/6/7/8/9) | 160.70 MB | EXCLUSIVE | 同 | defrag 的目标 SoA |
| — | host staging(sender / receiver) | 0 个(没有 peer) | — | 0 个 | |

1M 的 device-local 合计:V 188.66 MB + 160.70 MB 孪生;S 比 V 少 169 744 + 112 B(两个 stub),其余逐字节相同。所以"内存占用"这一项两边实际上相同:V 单卡模式的 pool 里没有 ghost / migrant / departed 区(它们的大小都是 0),也没有为迁移多留的余量——own pool 就是 case.yaml 的 `capacities.pool_size`(1M:1 203 584,比粒子数多 15 %,这是 case 文件的余量,两边一样用)。S 保持同一个 pool_size,因为 `OWN_POOL_SIZE`(spec 53)和逐粒子派发的工作组数(ceil(pool / 128) = 9 403)都由它决定,改了就与 V 的派发数不一致(验收 1)。C 不同:`compute_chain_partition(..., pool_safety=1.2)` 把 own pool 设为 min(ceil(1.2 N) 对齐到 workgroup, pool_size) = 1 203 584(1M 时恰好等于 pool_size,因为 1.2 × 1 046 529 = 1 255 835 > pool_size),其他尺寸可能不同。

缓冲是否可以更小(对 V 和 S 都有利,不实施):pool 按 ceil(N / 128) × 128 = 1 046 656 分配,SoA 和孪生各省 13 %,派发少 13 % 的空线程;voxel_base_offset、pool_health、band_compact_meta 在单卡下可以是 stub。只记进报告。

### 4. pool 布局与网格

| 项 | V | S |
|---|---|---|
| pid 布局 | slot 0 哨兵;[1, pool_size] own;ghost 区 0 个 slot(leading = trailing = 0);departed 0;replica 0 | 相同(S 只有 own) |
| own_first_pid / own_last_pid | 1 / pool_size | 1 / pool_size |
| 网格 | 加载器从 frame.obj 的包围盒算出的网格;扩展网格 = 自有网格(ghost 列数 0),voxel id 1 … 42 436 全是 own | 同一个加载器(`experiment.v7.utils.case_loader_v7.load_case_v7`),同一个网格 |
| 迁移 / 远迁移 / departed | 没有(`_transport_segments` 为空,相应 kernel 不派发) | 没有 |

S 直接复用 v7 的 case 加载器和 `CaseV7` 数据类(`experiment/v7/utils/case_loader_v7.py`、`case_v7.py`,纯 CPU,不读任何 V7_* 开关),这样物理常数、体积校准、网格推导与 V 逐位相同;S 只检查它拿到的是退化 slab。`partition_v7`、`simulator_v7`、`orchestrator_v7`、`vulkan_context_v7`、`bench_v7`、`sync_scheme_v7` 都不导入。

### 5. Shader 模块与流水线

两边的 SPIR-V 来自同一个目录 `experiment/v7/shaders/spv`。V 按 `V7_SPV_DIR` 可以改目录,S 不认这个变量,启动时把 `MANIFEST.txt` 列出的 22 个文件逐个算 sha256 与清单比对,任何一个不一致或缺失就退出。

| | V 加载的模块 | V 建的流水线 | 单卡步骤实际绑定的 | S 加载 / 建 |
|---|---|---|---|---|
| 基本 | bootstrap_half_kick, initialize_voxelization, predict, update_voxel, ghost_send_lanes, install_migrations, correction, density, force, defrag, append_departed, expand_ghost_lists, band_compact(13) | initialize_voxelization, bootstrap_half_kick, predict, update_voxel, append_departed, expand_ghost_lists, ghost_send_lanes_{leading,trailing}, install_migrations_{leading,trailing}, correction_{all,interior,boundary}, density_{all,deep_interior,boundary}, force_{all,deep_interior,boundary}, force_deep_interior_scratch, correction_boundary_band, density_boundary_band, force_boundary_band, defrag(24) | initialize_voxelization, bootstrap_half_kick, predict, update_voxel, force_all, defrag(融合时 correction_all / density_all 也不绑定) | 只加载并建这 6 个 + 下面按开关需要的 |
| B1 融合(默认) | correction_density(+ 深壁变体) | correction_density_{all,interior,boundary,boundary_band}(4) | correction_density_all | correction_density_all(1) |
| B1 关 | — | — | correction_all, density_all | correction, density 模块;correction_all, density_all(2) |
| B9 copy(默认) | density_scratch_copy | density_scratch_copy(1) | density_scratch_copy | 同(1) |
| B4 开(auto:3-D 且候选 ≥ 1 %) | deep_wall_marker, correction_deep_wall_skip, density_deep_wall_skip(, correction_density_deep_wall_skip) | deep_wall_marker_{presence,deep};correction_{all,interior}、density_{all,deep_interior}(或 correction_density_{all,interior})改用跳过变体 | marker ×2;correction_density_all(跳过变体) | marker ×2;correction_density_all(跳过变体)或 correction_all + density_all(跳过变体) |
| adami | wall_extrapolate | wall_extrapolate, wall_extrapolate_scratch(2) | wall_extrapolate | wall_extrapolate(1) |

V 多建的流水线只在启动时多花编译时间,不进入每步的命令流。描述符布局两边相同:4 个 set(set 2 空),每个 set 的 binding 如第 3 节;defrag 另有 set 4(9 个孪生)和 5-set 的流水线布局,S 一样。

### 6. Spec constant(逐个,两边取值)

两边的取值全部相同;S 不读 `V7_LEAN_TRANSPORT`、`V7_BAND_WIDTHS` 这类变量,而是把 V 在默认环境下给出的值写死,理由在最后一列。只列 V 的单卡步骤会绑定的流水线上出现的 id;V 另外建的流水线(ghost_send、install、band、interior / boundary、force_segment …)用到的 90–94、109、110、115 和 47–49 的 1 / 2 两边都不存在于 S。

| id | 名字 | V(1M,默认开关) | S | 来源 / 不同时的理由 |
|---|---|---|---|---|
| 0 | SMOOTHING_LENGTH | 0.005 | 同 | case |
| 1 | SPEED_OF_SOUND | 100 | 同 | case |
| 2 | DELTA_COEFFICIENT | 0.1 | 同 | case |
| 4 | POWER_PARAMETER | 7 | 同 | case |
| 5 | CFL_NUMBER | 0.15 | 同 | case |
| 6 | TIMESTEP | 7.5e-6 | 同 | 加载器 |
| 7 / 8 / 9 | OWN_ORIGIN_X/Y/Z | −0.5141 / −0.5141 / −0.0025 | 同 | 加载器(退化 slab 的网格原点) |
| 10 | STRICT_BIT_EXACT | 0 | 0 | V 写死 0 |
| 11 / 12 / 13 | GRID_DIMENSION_X/Y/Z | 206 / 206 / 1 | 同 | 加载器(扩展网格 = 自有网格) |
| 14 / 15 / 16 | REGULARIZATION_XI / DET / FROBENIUS | 0.001 / 1e-4 / 10 | 同 | case |
| 17 / 18 / 19 | GRAVITY | 0 / 0 / 0 | 同 | case |
| 20 | VOXEL_ORDER | 0 | 0 | 加载器 |
| 21 | MICROPOLAR_THETA | 不设(GLSL 默认 2.0) | 不设 | 两边都不设 |
| 30 / 31 | DIMENSION / NEIGHBOR_Z_RANGE | 2 / 0 | 同 | 加载器 |
| 32 / 33 | KERNEL_COEFFICIENT / GRADIENT | 114591.559… / 22918311.8… | 同 | 加载器 |
| 40 | EPS_H_SQUARED | 6.25e-8 | 同 | case(0.0025 h²) |
| 41 / 42 | PST_MAIN / ANTI | 0.1 / 0.0005 | 同 | case |
| 43 / 44 / 45 / 46 | USE_KCG / DIFFUSION / PST / PREFIX_SUM_DEFRAG | 1 / 1 / 1 / 0 | 同 | case |
| 47 | CORRECTION_MODE | 0(correction_all / correction_density_all) | 0 | 单卡只用 ALL |
| 48 | DENSITY_MODE | 0(density_all) | 0 | 同上 |
| 49 | FORCE_MODE | 0(force_all) | 0 | 同上 |
| 50 / 51 / 52 | MAX_PARTICLES_PER_VOXEL / WORKGROUP_SIZE / MAX_INCOMING | 96 / 128 / 16 | 同 | case |
| 53 | OWN_POOL_SIZE | 1 203 584 | 同 | case 的 pool_size(第 3 节) |
| 54 / 55 | LEADING / TRAILING_GHOST_POOL_SIZE | 0 / 0 | 0 / 0 | 退化 slab |
| 56 | FORCE_DENSITY_SOURCE | 0(force_all;adami 的 wall_extrapolate 也是 0) | 0 | 单卡在拷贝之后读 primary |
| 57 | BAND_VOXEL_DISPATCH | 0(ALL 流水线) | 0 | |
| 58 | BAND_SLOT_LANES | 0(band_dispatch = 0 时) | 0 | |
| 59 | FAKE_BAND_COLUMN | 0(`V7_FAKE_BAND_TEST` 默认 0) | 0 | S 不认该变量 |
| 80 / 81 | LEADING / TRAILING_GHOST_VOXEL_COUNT | 0 / 0 | 0 / 0 | 退化 slab |
| 82 | NEIGHBOR_X_RANGE | correction / correction_density 2,density 2,force 3(`V7_BAND_WIDTHS` 默认 2,2,3);predict 等不设(GLSL 默认 0) | 写死 2 / 2 / 3,同样的流水线集合 | 单卡下 `in_boundary_band` 恒为假(80 / 81 = 0),取值不进入任何分支;写成与 V 相同只是为了 SPIR-V 特化后的代码形状一致。S 不认 `V7_BAND_WIDTHS` |
| 83 | GHOST_LAYERS | 1(加载器 `GhostGridParams` 的默认值;**不是** `configured_ghost_layers()` 的 2,单卡 case 不经过 partition) | 1 | 只在 ghost_send / band kernel 的 ghost 分支里用;C 是 2 |
| 84 | DEPARTED_POOL_SIZE | 0 | 0 | |
| 85 | GHOST_SELF_LAYER | 0 | 0 | |
| 86 | REPLICA_REGION_SIZE | 0 | 0 | |
| 87 | LEAN_TRANSPORT | 1(`V7_LEAN_TRANSPORT` 默认) | 1 写死 | 只有 ghost_send / install 读 |
| 88 | TRANSPORT_EXTENSION | 0 | 0 写死 | 同上 |
| 89 | COMPACT_GHOST_LISTS | 1 | 1 写死 | 只有 ghost_send / install / expand 读 |
| 95 | DELTA_DENSITY | 0(`V7_DELTA_DENSITY` 默认) | 0 写死 | S 不支持 δρ 存储(设了变量就报错);读 helpers.glsl 的密度存取 |
| 96 | REFERENCE_DENSITY | 1000(第一种流体材料的静止密度) | 同 | 同一规则 |
| 97 | INIT_SEAM_CLAMP | 1 | 1 写死 | initialize_voxelization 里只在 ghost 列厚度 > 0 时起作用 |
| 98 | PACKED_REPLICAS | 1(`V7_PACKED_REPLICAS` 未设时 = ghost_layers 开关 2 ∧ compact 1 ∧ density 在 `V7_DIAG_GHOST_SELF` 里 = 真) | 1 写死 | 只有 ghost_send / expand 读;V 单卡模式下也是 1 |
| 99 | DIAG_POISON_G1 | 0 | 0 | |
| 100 | WALL_BOUNDARY | case.yaml numerics.wall_boundary(simple 0 / adami 1) | 同 | case |
| 101–108 | COPY_REGION_r_FIRST / COUNT | (1, 1 203 584), (0, 0), (0, 0), (0, 0) | 同 | own 范围一个区域;派发 ceil(pool / 256) = 4 702 个工作组 |
| 111 | DEEP_WALL_SKIP_BAND_WIDTH | 2(density 的带宽) | 2 | 只在 B4 变体上 |
| 112 | DEEP_WALL_CHECK | 0(`V7_DEEP_WALL_CHECK` 默认) | 0 | S 没有这个调试开关(设了报错) |
| 113 | DEEP_WALL_RECORD | 0;canonical_dump 的钩子置 1 | 同样的钩子 | 只在 B4 变体上 |
| 114 | DEEP_WALL_MARKER_PASS | 0 / 1 | 同 | marker 的两趟 |

### 7. 命令流(逐条)

**时间步**(一次录制,`SIMULTANEOUS_USE`,V 的 `_record_step_single_cmd` 在 `step_single_use_split = False`、`V7_FAKE_BAND_TEST = 0` 下):

1. [bench 复位 + tick `step_start`,只在接了 BenchTimer 时] compute → compute 屏障
2. predict,ceil(pool / 128) 个工作组;[tick];屏障
3. update_voxel,ceil(体素数 / 128);[tick];屏障
4. B4 开启时:marker presence 一趟 + 屏障,marker deep 一趟 + 屏障;[tick]
5. 融合:correction_density_all,ceil(pool / 128);[tick]。不融合:correction_all;[tick];屏障;density_all
6. density 拷贝:B9 = 屏障 + density_scratch_copy(ceil(pool / 256))+ 屏障;B9 关 = compute → transfer 屏障 + `vkCmdCopyBuffer` own 范围 + transfer → compute 屏障
7. [tick `density_end`];屏障(与第 6 步末尾的屏障相邻,V 的录制就是两道)
8. adami:wall_extrapolate,ceil(pool / 128);[tick];屏障
9. force_all,ceil(pool / 128);[tick]

S 的录制逐条相同,包括第 7 步那道多余的相邻屏障(去掉它对 V 和 S 都有利,只记进报告,不实施)。

**bootstrap**:V `bootstrap_init` = 上传初始状态(position / velocity_mass / density_pressure / material / material_parameters 有载荷,其余缓冲与 9 个孪生清零;每个缓冲一次 staging 上传 + fence)+ 一条一次性命令(屏障;initialize_voxelization);它里面按方向循环的 ghost 轮在没有 peer 时一条命令都不录。V `bootstrap_compute` = [adami:屏障 + wall_extrapolate + 屏障] [B4 且无 peer:屏障 + marker 两趟] 融合:correction_density_all / 不融合:correction_all + 屏障 + density_all;density 拷贝;[adami:wall_extrapolate + 屏障];force_all;屏障;bootstrap_half_kick;然后读回 status 打印 alive / overflow。S 两条命令逐条相同。

**defrag**(`SIMULTANEOUS_USE`):V = [bench 复位] 屏障;fill defrag_scratch_counter = 0;fill 9 个孪生 = 0;transfer → compute 屏障;defrag(ceil(体素数 / 128),5-set 布局);compute → transfer 屏障;9 次 `vkCmdCopyBuffer` 孪生 → 主缓冲;copy counter → global_status.alive;**fill global_status.migration_install_count = 0**;transfer → compute 屏障;[tick]。S 去掉加粗那一条 fill(单卡从不写这个字段,它永远是 0),其余逐条相同。这是验收 1 允许的唯一命令差别,比对工具把它列为已知差别。

**restart**(从检查点续跑,方腔验证用):上传 9 个 restart 字段 + material_parameters,其余清零;屏障;initialize_voxelization;屏障;[adami:wall_extrapolate];提交;读回 alive == 行数、overflow_inside = 0、所有 voxel id 在自有范围内。两边相同(S 的自有范围 = 全部体素)。

**周期**:defrag 每 `numerics.defrag_cadence`(1000)步,两边相同;V 的单卡 bench 和 S 都不做 bootstrap 后的 defrag(C 的 `bootstrap_all` 做一次)。

### 8. 读回与状态块

两边都读同一个 160 B 的 `GlobalStatusBuffer`(40 个 uint,`_GLOBAL_STATUS_FIELD_NAMES`):alive_particle_count 给 drift,所有 `overflow_*` 字段(overflow_inside / incoming / ghost / install_tail / install_inside / departed / initialization_outside / deep_wall_skip)给不变量。V 另有 `readback_pool_health`(install 的水位,单卡恒 0);S 不读 pool_health。多缓冲批读回(`readback_buffers_batch`)两边同一套实现方式(一次性 staging + 一条命令 + fence)。

### 9. 开关与环境变量

V 在导入 `simulator_v7` / `partition_v7` 时读 20 多个 `V7_*` 变量(每个都有默认值):`V7_SUBMIT_LOCK_SCOPE`、`V7_CASCADE_FORCE`、`V7_BAND_VOXEL_DISPATCH`、`V7_BAND_SLOT_LANES`、`V7_BAND_COMPACT_DISPATCH`、`V7_FAKE_BAND_TEST`、`V7_DIAG_GHOST_SELF`、`V7_FAST_SUBMIT`、`V7_PHASE_A_NO_WAIT`、`V7_DENSITY_COPY_COMPUTE`、`V7_GHOST_SEND_LANES`、`V7_DEEP_WALL_SKIP`、`V7_DEEP_WALL_CHECK`、`V7_FUSED_CORRECTION_DENSITY`、`V7_BAND_OVERLAP`、`V7_CONCURRENT_BUFFERS`、`V7_SPV_DIR`、`V7_RECEIVER_CACHED`、`V7_BENCH_PARITY`、`V7_KEEP_DEPARTED`、`V7_GHOST_LAYERS`、`V7_LEAN_TRANSPORT`、`V7_TRANSPORT_EXTENSION`、`V7_INIT_SEAM_CLAMP`、`V7_COMPACT_GHOST_LISTS`、`V7_PACKED_REPLICAS`、`V7_DIAG_POISON_G1`、`V7_DELTA_DENSITY`、`V7_BAND_WIDTHS`、`V7_GHOST_POOL_FACTOR`、`V7_MIGRANT_POOL_FACTOR`、`V7_DEPARTED_FACE_FRACTION`、`V7_DEPARTED_CAPACITY`、`V7_SPLIT_TRANSFER_QUEUES`、`V7_PER_SIM_PIPELINE`、`V7_LOOP_TRACE`、`V7_WORKER_*`。其中在单卡步骤里真正改变录制或结果的只有 `V7_DENSITY_COPY_COMPUTE`、`V7_FUSED_CORRECTION_DENSITY`、`V7_DEEP_WALL_SKIP`(以及调试用的 `V7_DEEP_WALL_CHECK`、`V7_FAKE_BAND_TEST`、`V7_DELTA_DENSITY`、`V7_BAND_WIDTHS`、`V7_SPV_DIR`、`V7_CONCURRENT_BUFFERS`)。

S 只读这三个:`V7_DENSITY_COPY_COMPUTE`(0 / 1,默认 1)、`V7_FUSED_CORRECTION_DENSITY`(0 / 1,默认 1)、`V7_DEEP_WALL_SKIP`(0 / 1 / auto,默认 auto;auto 的规则与 V 相同:3-D 且初始状态深壁候选 ≥ 1 % 的 slab 开,候选数用同一个公式算),解析器与 v7 的逐字相同(去首尾空白,其余值报错)。进程环境里出现任何其他 `V7_*` 名字,S 在导入时报错退出(包括 `V7_DEEP_WALL_CHECK`、`V7_DELTA_DENSITY`、`V7_SPV_DIR`)。运行表头打印开关表和每个开关在本 slab 上的判定。

### 10. 诊断与计时钩子

| 项 | V | S |
|---|---|---|
| GPU 时间戳 | `bench_v7.BenchTimer` 挂在 `sim.bench` 上,单卡录制在每个 kernel 后 tick;parity 区域、defrag 区域 | 没有 BenchTimer;录制函数接受一个可选的 tick 回调(`record_step(cmd, tick=...)`),`_run_kernel_breakdown.py` 用它在同一条生产录制里插时间戳(屏障结构因此与生产步相同) |
| step trace / phase trace | 链式路径的工具 | 无 |
| `step_single_use_split` / `V7_FAKE_BAND_TEST` | 单卡的验证 / 诊断路径(拆成 interior + boundary 流水线) | 无 |
| 深壁记录钩子 | `simulator_v7._DEEP_WALL_RECORD_DECISIONS` | `simulator_single._DEEP_WALL_RECORD_DECISIONS`,同样的 spec 113 与记录缓冲 |

### 11. 小结:会改变结果的差别

没有。两边的 kernel、spec constant、派发数、屏障顺序、初始数据、bootstrap 与 defrag 的 kernel 序列都相同;S 相对 V 去掉的全是从不执行(transfer 队列、timeline、锁、多余的流水线)或从不改变数据(CONCURRENT 共享、ghost 缓冲的 stub、migration_install_count 的清零)的东西。验收 2 的逐位比较检验这一结论。

## 验收

待做。计划:

1. CPU 侧(`experiment/v7_single/_compare_recordings.py`):把两边的录制函数在 CPU 上跑一遍(vkCmd* 换成记录器,流水线换成 (SPIR-V 文件, spec constant) 记录),逐条比较时间步、bootstrap 两条命令、defrag、restart 的命令流,以及每条绑定的 SPIR-V 与 spec constant;算例 5 个,默认开关与 `V7_FUSED_CORRECTION_DENSITY=0` 各一次;允许的差别只有第 7 节列出的那条 fill。
2. GPU 逐位(`experiment/seam_audit/canonical_dump.py --solver v7_single` 与 `--solver v7 --v7-mode single`):n250 200 / 999 步,2-D 1M 200 步,cavity3d_8m_b4 200 步,cavity3d_1m 200 步,n250 adami 200 步;S 对 V 必须逐位;S 对 C(`--solver v7`,链式)只报告。
3. 不变量:所有运行 drift 0、overflow 0。
4. 验证层 2000 步:2-D 1M 与 cavity3d_8m_b4,0 条消息。
5. 独立子 agent 评审 host 代码与本清单。

## 测量

待做(A 吞吐与开销、B 逐 kernel、C 图)。

## 精度验证:1000² 方腔 Re = 1000(adami)

待做。

## 已知限制

待填。

## 复现

待填。

## 提交(分支 `v7-single`,从 `v7-rc1` = d0c8dcb 开出)

| 提交 | 内容 |
|---|---|
| (本提交) | 本文档的清单 |
