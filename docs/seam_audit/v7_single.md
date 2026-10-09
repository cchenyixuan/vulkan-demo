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
| 3.1 | pool_health | 16 | EXCLUSIVE | 16 | 只有 install_migrations 写;S 保留(描述符布局);S 的 `readback_pool_health` 只给 canonical_dump 的元数据(恒 0),生产路径不读 |
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

两边都读同一个 160 B 的 `GlobalStatusBuffer`(40 个 uint,`_GLOBAL_STATUS_FIELD_NAMES`):alive_particle_count 给 drift,所有 `overflow_*` 字段(overflow_inside / incoming / ghost / install_tail / install_inside / departed / initialization_outside / deep_wall_skip)给不变量。V 另有 `readback_pool_health`(install 的水位,单卡恒 0);S 保留同名方法只为 canonical_dump 的元数据(恒 0),生产 runner 不调用它。多缓冲批读回(`readback_buffers_batch`)两边同一套实现方式(一次性 staging + 一条命令 + fence)。

### 9. 开关与环境变量

V 在导入 `simulator_v7` / `partition_v7` 时读 20 多个 `V7_*` 变量(每个都有默认值):`V7_SUBMIT_LOCK_SCOPE`、`V7_CASCADE_FORCE`、`V7_BAND_VOXEL_DISPATCH`、`V7_BAND_SLOT_LANES`、`V7_BAND_COMPACT_DISPATCH`、`V7_FAKE_BAND_TEST`、`V7_DIAG_GHOST_SELF`、`V7_FAST_SUBMIT`、`V7_PHASE_A_NO_WAIT`、`V7_DENSITY_COPY_COMPUTE`、`V7_GHOST_SEND_LANES`、`V7_DEEP_WALL_SKIP`、`V7_DEEP_WALL_CHECK`、`V7_FUSED_CORRECTION_DENSITY`、`V7_BAND_OVERLAP`、`V7_CONCURRENT_BUFFERS`、`V7_SPV_DIR`、`V7_RECEIVER_CACHED`、`V7_BENCH_PARITY`、`V7_KEEP_DEPARTED`、`V7_GHOST_LAYERS`、`V7_LEAN_TRANSPORT`、`V7_TRANSPORT_EXTENSION`、`V7_INIT_SEAM_CLAMP`、`V7_COMPACT_GHOST_LISTS`、`V7_PACKED_REPLICAS`、`V7_DIAG_POISON_G1`、`V7_DELTA_DENSITY`、`V7_BAND_WIDTHS`、`V7_GHOST_POOL_FACTOR`、`V7_MIGRANT_POOL_FACTOR`、`V7_DEPARTED_FACE_FRACTION`、`V7_DEPARTED_CAPACITY`、`V7_SPLIT_TRANSFER_QUEUES`、`V7_PER_SIM_PIPELINE`、`V7_LOOP_TRACE`、`V7_WORKER_*`。其中在单卡步骤里真正改变录制或结果的只有 `V7_DENSITY_COPY_COMPUTE`、`V7_FUSED_CORRECTION_DENSITY`、`V7_DEEP_WALL_SKIP`(以及调试用的 `V7_DEEP_WALL_CHECK`、`V7_FAKE_BAND_TEST`、`V7_DELTA_DENSITY`、`V7_BAND_WIDTHS`、`V7_SPV_DIR`、`V7_CONCURRENT_BUFFERS`)。

S 只读这三个:`V7_DENSITY_COPY_COMPUTE`(0 / 1,默认 1)、`V7_FUSED_CORRECTION_DENSITY`(0 / 1,默认 1)、`V7_DEEP_WALL_SKIP`(0 / 1 / auto,默认 auto;auto 的规则与 V 相同:3-D 且初始状态深壁候选 ≥ 1 % 的 slab 开,候选数用同一个公式算),解析器与 v7 的逐字相同(去首尾空白,其余值报错)。进程环境里出现任何其他 `V7_*` 名字,S 在导入时报错退出(包括 `V7_DEEP_WALL_CHECK`、`V7_DELTA_DENSITY`、`V7_SPV_DIR`)。运行表头打印开关表和每个开关在本 slab 上的判定。

### 10. 诊断与计时钩子

| 项 | V | S |
|---|---|---|
| GPU 时间戳 | `bench_v7.BenchTimer` 挂在 `sim.bench` 上,单卡录制在每个 kernel 后 tick(`step_start` 在开头那道屏障之前,`wall_extrapolate_end` 在壁面 kernel 的屏障之前);parity 区域、defrag 区域 | 没有 BenchTimer;录制函数接受一个可选的 tick 回调(`record_step(cmd, tick=...)`),`_run_kernel_breakdown.py` 用它在同一条生产录制里插时间戳(屏障结构因此与生产步相同)。tick 位置与 V 相同,只有 `step_start` 放在开头那道屏障之后(E5 的计时器也是这样放;同步循环里这道屏障前面没有在飞的工作,差别在 1 µs 以内) |
| step trace / phase trace | 链式路径的工具 | 无 |
| `step_single_use_split` / `V7_FAKE_BAND_TEST` | 单卡的验证 / 诊断路径(拆成 interior + boundary 流水线) | 无 |
| 深壁记录钩子 | `simulator_v7._DEEP_WALL_RECORD_DECISIONS` | `simulator_single._DEEP_WALL_RECORD_DECISIONS`,同样的 spec 113 与记录缓冲 |

### 11. 小结:会改变结果的差别

没有。两边的 kernel、spec constant、派发数、屏障顺序、初始数据、bootstrap 与 defrag 的 kernel 序列都相同;S 相对 V 去掉的全是从不执行(transfer 队列、timeline、锁、多余的流水线)或从不改变数据(CONCURRENT 共享、ghost 缓冲的 stub、migration_install_count 的清零)的东西。验收 2 的逐位比较检验这一结论。

## 验收

全部通过(2026-10-09,`experiment/v7_single/_run_acceptance.py all`,日志 `logs/v7_single/acceptance/`,汇总镜像在 `docs/seam_audit/v7_single/acceptance_summary.{json,md}`、`acceptance_cpu_compare.json`、`acceptance_runs.jsonl`)。本机两张 RTX 5090、驱动 576.88,GPU 运行都在无头的那张卡(UUID `ae137c66…`,discrete-first 序号 1)上,验证层关(第 4 项除外)。

### 1. CPU 侧:命令流逐条比对

`experiment/v7_single/_compare_recordings.py`:两边的模拟器在 CPU 上用 `object.__new__` 建起(`_v7_cpu_fake.py`),把模块里的 vkCmd* / vkBegin / vkEnd 换成记录器,`_create_pipeline` 换成返回 (SPIR-V 文件名, spec constant) 的函数,然后各自跑五条录制:时间步、bootstrap 的两条命令、defrag、restart 的体素化。比较的元素:绑定的 SPIR-V 文件及其全部 spec constant、描述符集绑定(布局、first、count、集合)、派发的工作组数、每道屏障的 stage / access 掩码、fill(缓冲、偏移、长度、值)、copy(源、目标、区域)、`vkBeginCommandBuffer` 的标志。另外比较两边的缓冲表(name / set / binding 顺序相同,大小除两个 stub 外相同,9 个 defrag 孪生相同)、初始上传的每个载荷、material_parameters 的 48 B 行、以及一个合成 3 行状态的 restart 载荷,逐字节;v7 一侧再在 canonical_dump 的发布环境(`RELEASE_ENVIRONMENT` 的 `V7_*`)下录一遍,命令流不变。

| 算例 | 默认(融合 + compute 拷贝) | `V7_FUSED_CORRECTION_DENSITY=0` | `V7_DENSITY_COPY_COMPUTE=0` |
|---|---|---|---|
| n250(62k) | 5 条录制逐条相同 | 同 | 同 |
| n250 adami | 同 | 同 | 同 |
| 2-D 1M | 同 | 同 | 同 |
| cavity3d_8m_b4(10.5M,B4 关) | 同 | 同 | 同 |
| cavity3d_1m(1.69M,B4 开:280 231 个候选 = 16.63 %) | 同 | 同 | 同 |

每一格里唯一的差别是 defrag 中 V 多出的 `fill global_status[48] = 0`(migration_install_count,第 7 节),工具把它列为允许项;缓冲表、载荷、发布环境三项在 n250 / n250 adami / cavity3d_1m 上也都相同(`logs/v7_single/acceptance/cpu_compare_small_v2.json`;五算例的完整重跑见测量后的 `cpu_compare.json`)。时间步的派发数:2-D 融合 5 次(predict、update_voxel、correction_density_all、density_scratch_copy、force_all)、6 道屏障;不融合 6 次、7 道;adami 各加 1;3-D B4 开各加 2(marker 两趟)。

### 2–3. GPU 逐位(canonical lists)与不变量

`canonical_dump.py`:`--solver v7_single`(S)、`--solver v7 --v7-mode single`(V,`SphSimulatorV7` 的单命令步)、`--solver v7`(C,链式 K = 1),都用 `--canonical-lists`(初始体素化后按 pid 排序列表、没有 bootstrap defrag),同一张卡,dump 后 `--compare`。每个算例跑默认开关与 `V7_FUSED_CORRECTION_DENSITY=0` 各一次,共 36 个 dump。

| 算例 | 步数 | S 对 V(必须) | S 对 C(只报告) | drift / overflow(S / V / C) |
|---|---|---|---|---|
| n250 | 200 | 逐位(默认、不融合;`--no-mask` 也逐位) | 逐位 | 0 / 0 |
| n250 | 999 | 逐位 | 逐位 | 0 / 0 |
| 2-D 1M | 200 | 逐位 | 逐位 | 0 / 0 |
| cavity3d_8m_b4 | 200 | 逐位 | 逐位 | 0 / 0 |
| cavity3d_1m(B4 开) | 200 | 逐位(含不屏蔽深壁行) | 逐位 | 0 / 0 |
| n250 adami | 200 | 逐位 | 逐位 | 0 / 0 |

10 个场(position_voxel_id、velocity_mass、density_pressure、density_pressure_scratch、acceleration、shift、material、correction_inverse、density_gradient_kernel_sum、extension_fields;adami 另加 wall_dummy_velocity)全部逐位相同,所以 S 对 C 的 ULP 统计没有可报的差别:在这 6 个算例上 v7_single、v7 单卡模式与 v7 链式 K = 1 是同一组比特。每个 dump 的 alive = 粒子数,所有 `overflow_*` 计数、far_migration、stamp 计数都是 0。

### 4. 验证层

v7_single 在 Khronos 验证层(默认设置,实例层 + debug messenger 计数)下各跑 2000 步、2 步在飞:2-D 1M(710.9 fps)与 cavity3d_8m_b4(14.7 fps),层已加载,消息 0 条(ERROR / WARNING / INFO / VERBOSE 都是 0,stderr 也没有一行),drift 0,overflow 0。

### 5. 独立评审

一个只读的子 agent 评审了 host 代码、工具 adapter 与本清单(读 `experiment/v7_single`、改过的 adapter、对照 `simulator_v7.py` / `vulkan_context_v7.py` / `case_loader_v7.py` 与 shader 源码)。结论:**没有 blocker**,找不到让 S 与 V 产生不同比特的路径。它核实的内容:`global_entries()` 与 `_global_entries()` 的每个 id、格式、取值(含 83 = 加载器默认 1、98 在未设 `V7_PACKED_REPLICAS` 时解析为 1);发布环境下这些单卡流水线的常数不变(`V7_BAND_SLOT_LANES=64` 只进 band_dispatch = 1 的流水线);两个 stub 缓冲只被 ghost_send / ghost_send_lanes / expand_ghost_lists 和 `BAND_VOXEL_DISPATCH == 2` 的分支触及,S 绑定的任何流水线都到不了;fence 环的等待关系;两边的队列族都是第一个有 COMPUTE 位的族。它提出的 10 项应改与备注都已处理(提交 24a1886):

- 物理哈希:`cavity_runner.code_hashes` 与 canonical_dump 的代码摘要原来没有覆盖 v7 的 case 加载器(dt、网格、原点、核系数都由它算出),现在把 `experiment/v7/utils/case_loader_v7.py`、`case_v7.py` 纳入。
- CPU 比对工具:补上缓冲表断言、载荷逐字节比较、begin 标志、描述符集身份、发布环境下的 v7 录制(本节第 1 项的表就是补后的结果)。
- 链式 adapter 的稳态窗口:`run_pipelined` 在提交第 warmup 帧时就打开窗口,而 warmup = 1000 正好是 defrag 周期,窗口会比 fence 环的多含一次 defrag;改为从第 warmup + 1 帧提交后打开(窗口 2000 − 1 帧),与环的窗口含同样的 defrag。
- 壁面 kernel 的 tick 放回屏障之前(与 V 的 BenchTimer 一致);`vkQueueSubmit` 失败时重建该槽的 fence;standalone runner 的 `--warmup 0`;验收 4 由脚本强制判定(层未加载或有消息即 FAIL);canonical_dump 的单卡模式在每个 defrag 边界检查 overflow;候选对数计算断言 `voxel_order = 0`。
- 备注(不改):defrag 开头的 compute → compute 屏障后面接 transfer 阶段的 fill,与 v7 相同(v7 自己注明第二作用域形式上不含 TRANSFER;前一次写在 1000 帧之前的 defrag 里,实际无害);S 有两种起点(`SingleRunner.bootstrap_all` 像链一样在 bootstrap 后 defrag 一次,bench 与 canonical_dump 的单卡模式像 V 一样不 defrag),两种起点的 dump 之间不逐位可比,但各自与对应的 V / C 起点一致。

## 测量

通用条件:本机两张 RTX 5090、驱动 576.88;所有运行都在无头的那张卡上(nvidia-smi 序号 1,PCI 03:00.0,Vulkan deviceUUID `ae137c668a40f5acaab90a83f7cda175`;每个 runner 用 UUID 解析 discrete-first 序号,再用 `--expect-gpu "RTX 5090"` 复核);验证层关(`VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation`);1 Hz 的 nvidia-smi 遥测记 SM 时钟、功耗、温度到各活动目录的 `telemetry.csv`;每次运行前检查本机没有其他求解器进程(nvidia-smi 的利用率在这台 WDDM 机器上不可用:求解器满载时它读 0 %,杀掉进程后反而停在 100 %,所以空闲检查按进程命令行判定)。

### A. 吞吐与开销(§3 图 1、§5.1)

`_run_single_baseline_campaign.py --out logs/v7_single/throughput_r2 --sizes 1m,2m,4m,8m,16m,32m,3d_8m --configs v7_single:1,v7_single:2,v7:1,v7:2,v7x:1,v7x:2,v7_chain:2 --trials 3 --warmup 1000 --idle-check`,汇总 `_summarize_v7_single_campaign.py`。

- 构建:S = `v7_single`;V = `v7`(`SphSimulatorV7` 单卡模式,`_run_single_baseline_bench.py` 的 `V7Adapter`);Vx = `v7x`(V 加 `V7_CONCURRENT_BUFFERS=0`);C = `v7_chain`(`compute_chain_partition` 一个 slab、pool_safety 1.2,`ChainOrchestratorV7.bootstrap_all`(末尾一次 defrag)+ `run_pipelined(depth=2)`,即 `_run_v7_chain_bench.py` K = 1 的生产循环,三批提交)。
- 算例:`lid_driven_cavity_2d_gen`(1M)、`_2m`、`_4m`、`_8m`、`_16m`、`_32m`、`cavity3d_8m_b4`。
- 在飞深度:S、V、Vx 各测 1 与 2;C 只有生产的 depth 2。S 的 N 在飞用它自己的 fence 环(`SphSimulatorSingle.submit_step`,defrag 在周期帧并入同一次提交);V、Vx 的 depth 1 是 `submit_step_single_and_wait`(每步建销一个 fence),depth 2 是 bench 脚本外挂的 `FenceRing`(E5 给 V5 用的同一段代码);C 的 fps 是 orchestrator 的稳态窗口。
- 窗口:预热 1000 步,计时 10000(1M)/ 6000(2M)/ 4000(4M)/ 2000(8M、16M、32M、3-D 8M)步,都是 defrag 周期的整数倍。fence 环的两端都在队列排空时取时间,所以任何在飞深度下都精确;C 的窗口从第 1001 帧提交后打开(第 1000 帧触发的排空 + defrag 已经过去,窗口里的 defrag 次数与环的相同,长度 = 计时步数 − 1 帧)、到最后一次排空关闭,末端含 depth − 1 帧在飞的尾巴(≤ 0.05 %)。
- 重复:3 次试验;每次试验里同一尺寸的 7 个构建紧挨着跑,顺序每次轮换一位;比值按同一试验配对再取均值 ± 标准差。每次运行结束读回 alive(drift 必须 0)和所有 overflow 计数。
- 先验(v5 时代):单卡模式是 V0 的 100.1–102.0 %,链式 K = 1 对单卡模式 99.91 ± 0.59 %,CONCURRENT ≤ 1.2 %。任何比值偏离 1 超过 2 % 先定位原因。

**结果**(2026-10-09,147 次运行全部有效:drift 0、overflow 0;`logs/v7_single/throughput_r2/`,镜像 `docs/seam_audit/v7_single/throughput_summary.{md,json,csv}`、`throughput_results.jsonl`;3 次试验的均值 ± 标准差):

| 算例 | 粒子 | S d1 | S d2 | V d1 | V d2 | Vx d1 | Vx d2 | C d2 |
|---|---|---|---|---|---|---|---|---|
| 1M | 1 046 529 | 705.2 ± 0.2 | 734.6 ± 0.1 | 698.9 ± 1.1 | 733.3 ± 0.3 | 699.9 ± 1.7 | 733.0 ± 0.4 | 723.9 ± 0.8 |
| 2M | 2 064 969 | 363.6 ± 0.2 | 371.9 ± 0.5 | 361.7 ± 0.2 | 371.7 ± 0.2 | 361.1 ± 0.3 | 371.5 ± 0.3 | 368.9 ± 0.4 |
| 4M | 4 182 025 | 172.56 ± 0.02 | 174.97 ± 0.18 | 172.37 ± 0.09 | 175.36 ± 0.03 | 172.40 ± 0.09 | 175.25 ± 0.08 | 174.86 ± 0.13 |
| 8M | 8 128 201 | 93.59 ± 0.08 | 94.44 ± 0.05 | 93.53 ± 0.06 | 94.57 ± 0.07 | 93.59 ± 0.05 | 94.54 ± 0.05 | 94.46 ± 0.12 |
| 16M | 16 184 529 | 47.23 ± 0.01 | 47.47 ± 0.02 | 47.29 ± 0.01 | 47.58 ± 0.02 | 47.28 ± 0.01 | 47.57 ± 0.01 | 47.43 ± 0.01 |
| 32M | 32 251 041 | 24.42 ± 0.00 | 24.50 ± 0.00 | 24.40 ± 0.00 | 24.50 ± 0.01 | 24.41 ± 0.00 | 24.49 ± 0.00 | 24.46 ± 0.01 |
| 3-D 8M | 9 129 329 | 16.30 ± 0.02 | 16.34 ± 0.01 | 16.28 ± 0.01 | 16.33 ± 0.04 | 16.28 ± 0.02 | 16.34 ± 0.02 | 16.32 ± 0.03 |

百万粒子·步 / 秒(S):d1 738 / 751 / 722 / 761 / 764 / 788(1M → 32M),d2 769 / 768 / 732 / 768 / 768 / 790;3-D 8M 148.8 / 149.1。逐次配对的比值(均值 ± 标准差,n = 3):

| 算例 | V/S d1 | V/S d2 | Vx/S d1 | Vx/S d2 | C/S | Vx/V d1 | Vx/V d2 |
|---|---|---|---|---|---|---|---|
| 1M | 99.10 ± 0.14 % | 99.82 ± 0.03 % | 99.24 ± 0.21 % | 99.78 ± 0.05 % | 98.55 ± 0.11 % | 100.15 ± 0.14 % | 99.95 ± 0.07 % |
| 2M | 99.48 ± 0.08 % | 99.96 ± 0.11 % | 99.33 ± 0.06 % | 99.89 ± 0.22 % | 99.20 ± 0.16 % | 99.85 ± 0.07 % | 99.93 ± 0.14 % |
| 4M | 99.89 ± 0.04 % | 100.23 ± 0.12 % | 99.91 ± 0.05 % | 100.16 ± 0.08 % | 99.94 ± 0.12 % | 100.02 ± 0.02 % | 99.93 ± 0.05 % |
| 8M | 99.94 ± 0.06 % | 100.15 ± 0.06 % | 100.00 ± 0.14 % | 100.11 ± 0.08 % | 100.02 ± 0.14 % | 100.06 ± 0.12 % | 99.96 ± 0.12 % |
| 16M | 100.13 ± 0.01 % | 100.23 ± 0.06 % | 100.12 ± 0.01 % | 100.20 ± 0.06 % | 99.91 ± 0.05 % | 99.99 ± 0.03 % | 99.97 ± 0.05 % |
| 32M | 99.91 ± 0.01 % | 99.98 ± 0.03 % | 99.95 ± 0.02 % | 99.97 ± 0.00 % | 99.85 ± 0.02 % | 100.04 ± 0.01 % | 99.99 ± 0.03 % |
| 3-D 8M | 99.86 ± 0.16 % | 99.96 ± 0.25 % | 99.86 ± 0.15 % | 100.01 ± 0.17 % | 99.89 ± 0.16 % | 100.01 ± 0.11 % | 100.05 ± 0.27 % |

depth 1 相对 depth 2 的损失(fps_d1 / fps_d2 − 1):S −4.00 / −2.23 / −1.38 / −0.90 / −0.52 / −0.34 %(1M → 32M),3-D 8M −0.22 %;V −4.70 / −2.70 / −1.71 / −1.10 / −0.61 / −0.40 %;Vx −4.51 / −2.78 / −1.63 / −1.01 / −0.59 / −0.35 %。协变量:每个窗口都在 600 W 功耗墙上,SM 2765–2830 MHz,≤ 70 °C,同一尺寸内各构建的时钟差 ≤ 7 MHz(`throughput_summary.md` 的协变量表)。

读法:

- 没有一个比值离 1 超过 2 %(先验成立),所以没有需要定位的差距。
- V 对 S:depth 1 时 S 快 0.1–0.9 %(1M–2M 最明显),差别来自 V 每步创建并销毁一个 fence(`submit_and_wait`)而 S 复用 fence 环;depth 2 时两者在 1M / 2M / 32M / 3-D 相同(99.8–100.0 %),在 4M / 8M / 16M V 快 0.15–0.23 %(2–4 σ)。后者不是 CONCURRENT(Vx 与 V 在所有点上 99.85–100.15 %,所以共享模式的代价在 0.2 % 以内,与 E5 的 ≤ 1.2 % 一致),kernel 与命令流逐位相同,剩下的候选是缓冲在显存里的摆放(两边分配的缓冲数与顺序不同)或设备创建时的差别(V 多开 2 条 transfer 队列和 timelineSemaphore 特性);量级在 0.2 % 以内,本文记录而不追查(见"结论"里给用户的说明)。
- C 对 S:1M 98.55 %、2M 99.20 %,4M 起 99.85–100.02 %。链式 K = 1 每步三批提交 + timeline 等待的固定开销约 20 µs,在 1M(1.36 ms / 步)占 1.5 %,从 4M 起淹没在步长里;与 E5.3 在 8M 上的 99.91 ± 0.59 % 一致。
- depth 1 的损失就是同步循环暴露的 CPU 提交气泡:S 在 1M 约 57 µs / 步(4 %),32M 约 140 µs / 步(0.3 %);V 的气泡多出 fence 建销的 10–20 µs。
- 3-D 8M(`cavity3d_8m_b4`,9 129 329 个粒子,每步约 61 ms)上所有构建都在 ± 0.25 % 内,depth 的影响 0.2–0.4 %。

### B. 逐 kernel 成本(§3 表 2)

`experiment/v7_single/_run_kernel_breakdown_campaign.py run`(每个单元一个 `_run_kernel_breakdown.py --solver v7_single` 子进程),`summarize` 调 `_summarize_kernel_breakdown.py`。

- 算例 `lid_driven_cavity_2d_8m`(8 128 201 个粒子)与 `cavity3d_8m_b4`(10 503 459);融合(默认)与不融合(`V7_FUSED_CORRECTION_DENSITY=0`)各 3 次,交错(第 1 次的四个单元、第 2 次的四个 …)。
- 计时命令:S 的生产录制函数 `record_step(cmd, tick=...)` 本身在每个阶段末尾调用回调,回调写一个 BOTTOM_OF_PIPE 时间戳,所以计时步的屏障与生产步逐条相同(只多了 `vkCmdWriteTimestamp`);每步一次提交 + fence(1 在飞),预热 1000 + 计时 2000 步,defrag 帧剔除。阶段:融合 = predict、update_voxel、correction_density、density_copy、force;不融合 = predict、update_voxel、correction、density_kernel、density_copy、force(3-D B4 开或 adami 时各加 deep_wall_marker / wall_extrapolate,这两个算例都没有)。`step_start` 在开头那道屏障之后(E5 的计时器也是这样放),`density_copy` 是拷贝 pass 加它前后两道屏障。
- 计时步与生产步的差:预热之后计时帧与生产帧交错(奇数帧提交带时间戳的命令,偶数帧提交生产命令 `step_cmd`,各 2000 帧,窗口 = 第 1001–5000 帧;defrag 帧不计入任何一侧),报告计时步的 GPU 帧长、计时帧的每步墙钟(提交 + 每步新建的 fence)与生产帧的每步墙钟(持久 fence、无时间戳)。先按"先计时 2000 步再生产 2000 步"做过一轮(`logs/v7_single/kernel_breakdown/`),3-D 上生产段读数比计时段慢 7 %;`experiment/v7_single/_probe_loop_variants.py` 查明三种 1 在飞的 host 循环本身等价(排序后 59–64 ms / 步,三者在同一段里相差 < 1 ms),慢的是流动本身:3-D 8M 每步的 GPU 代价随流动发展在 defrag 之间持续上升(第 1100 帧 59.1 ms → 第 2900 帧 63.9 ms,见探针日志),先后两段比的是两个流动状态。所以改为交错,两侧看同一个状态。
- 每候选 ps:仓库里没有找到 E5 对"候选"的定义,这里取邻居循环实际访问的候选数——运行结束时读回体素列表,候选对数 = Σ_v count(v) · Σ_{v' ∈ 3^d 邻域} count(v'),每个 kernel 的 ps/候选 = µs/步 × 10⁶ / 候选对数(2-D 8M 约 218 个候选 / 粒子,与 `docs/sph_v4_summary.md` 里"9 体素 × 25 ≈ 225 候选"的说法一致)。若论文用的是别的定义,改汇总脚本即可,原始数据都在 `results.jsonl`。
- 输出:每个 kernel 的 µs/步、ns/粒子、占比、ps/候选;表 2 的 LaTeX 行(`summary_fused.md`、`summary_unfused.md`)。

**结果**(2026-10-09,`logs/v7_single/kernel_breakdown_v2/`,镜像 `docs/seam_audit/v7_single/kernel_breakdown_{fused,unfused}.md`、`kernel_breakdown_results.jsonl`;3 次运行的均值,p50 在镜像文件里;12 次运行 drift 0、overflow 0;步长总和的 3 次标准差 0.04–0.14 %;SM 2715–2895 MHz,600 W 功耗墙,≤ 70 °C):

| kernel | 派发 | 2-D 8M µs/步 | ns/粒子 | % | ps/候选 | 3-D 8M µs/步 | ns/粒子 | % | ps/候选 |
|---|---|---|---|---|---|---|---|---|---|
| predict | 每粒子 | 497 | 0.06 | 4.7 | – | 498 | 0.05 | 0.8 | – |
| update_voxel | 每体素 | 331 | 0.04 | 3.1 | – | 318 | 0.03 | 0.5 | – |
| correction + density(融合) | 每粒子 | 4565 | 0.56 | 43.2 | 2.50 | 31 410 | 3.44 | 49.6 | 2.09 |
| density 拷贝(scratch → primary) | compute pass | 82 | 0.01 | 0.8 | – | 78 | 0.01 | 0.1 | – |
| force | 每粒子 | 5097 | 0.63 | 48.2 | 2.79 | 31 014 | 3.40 | 49.0 | 2.06 |
| **步长** | | **10 572** | **1.30** | 100 | | **63 319** | **6.94** | 100 | |

不融合(`V7_FUSED_CORRECTION_DENSITY=0`):

| kernel | 2-D 8M µs/步 | ns/粒子 | % | ps/候选 | 3-D 8M µs/步 | ns/粒子 | % | ps/候选 |
|---|---|---|---|---|---|---|---|---|
| predict | 497 | 0.06 | 3.6 | – | 499 | 0.05 | 0.6 | – |
| update_voxel | 332 | 0.04 | 2.4 | – | 318 | 0.03 | 0.4 | – |
| correction | 3859 | 0.47 | 27.9 | 2.12 | 24 159 | 2.65 | 29.5 | 1.60 |
| density(kernel) | 3956 | 0.49 | 28.6 | 2.17 | 25 824 | 2.83 | 31.5 | 1.72 |
| density 拷贝 | 58 | 0.01 | 0.4 | – | 71 | 0.01 | 0.1 | – |
| force | 5115 | 0.63 | 37.0 | 2.80 | 31 004 | 3.40 | 37.9 | 2.06 |
| **步长** | **13 815** | **1.70** | 100 | | **81 875** | **8.97** | 100 | |

- 候选数:2-D 8M 每粒子 224.4(9 个体素 × 24.9 个粒子),3-D 8M 每粒子 1649(27 × 61);候选对数 1.82 × 10⁹ / 1.51 × 10¹⁰ 每步。
- 融合省下的步长:2-D (13 815 − 10 572) / 13 815 = **23.5 %**,3-D (81 875 − 63 319) / 81 875 = **22.7 %**;融合核比 correction + density 两核之和省 2-D 41.6 %、3-D 37.2 %。
- 计时步与生产步:2-D 的 GPU 帧 10 572 µs,计时帧墙钟 10 728(+156 µs:每步新建 fence 的提交往返),生产帧墙钟 10 695(+123 µs = +1.17 %,持久 fence);3-D 63 319 / 63 568(+249)/ 63 502(+183 µs = +0.29 %)。即生产步(1 在飞)比 GPU 帧多出 120–190 µs 的 host 往返,与 A 的 depth-1 损失(8M −0.90 %、3-D −0.22 %)同量级。
- 窗口说明:交错后的窗口是第 1001–5000 帧,而 A 的窗口是第 1001–3000 帧;3-D 8M 的每步代价随流动发展上升(见上),所以这里的 3-D 步长(63.3 ms)比 A 的 d1 墙钟(61.35 ms,窗口 1001–3000)高 3 %,不是测量差异。第一轮(窗口 1001–3000)的 kernel 数字在 `logs/v7_single/kernel_breakdown/`,2-D 相同到 0.5 %,3-D 低 3.5 %。
- 与 E5 的 V0 同窗口(1001–3000)数字相比(2-D 8M:correction 3898、density 3946、拷贝 275、force 5150、步长 14 100 µs):不融合的 v7 kernel 与 V0 的 correction / density / force 各在 1 % 内,拷贝从 275 µs(`vkCmdCopyBuffer`)降到 58 µs(B9 的 compute pass),步长 13 815 对 14 100(−2.0 %);融合后 10 572(−25 %)。

### C. 图

`_plot_single_baseline_paper.py --dir logs/v7_single/throughput_r2 --solver v7_single --figure-a-name single_gpu_throughput --no-figure-b`:用 A 里 S 的数据(1M–32M 二维曲线,1 与 2 在飞)重画 E5 的图 A,输出 `manuscripts/fig/single_gpu_throughput.pdf` 与 `.png`;旧的 `single_gpu_v0_throughput` 不动。

结果:已生成(上图吞吐 722–790 M 粒子·步 / 秒,下图 fps 对数轴;4M 那个点的吞吐比两边低 4–5 %,E5 的 V0 曲线在 4M 也有同样的凹陷,是算例本身的现象,不是本构建的)。

![single_gpu_throughput](../../manuscripts/fig/single_gpu_throughput.png)

## 精度验证:1000² 方腔 Re = 1000(adami)

只验证精度,不计时,结果不进任何性能表。生产设置:默认开关(融合、compute 拷贝、B4 auto = 2-D 关)、不开计时器、不用 canonical lists;算例 `cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami`(`numerics.wall_boundary: adami`,ξ = 0.001、ε² = 0.0025 h²、h = 5 Δx、11 层壁;1 046 529 个粒子;只支持一张卡),Δt = 0.15 h / c0 = 7.5 × 10⁻⁶。

工具与方法沿用 `docs/validation/cavity_re1000.md`:`experiment/validation/cavity_campaign.py run --solver v7_single --runs n1000_k1_float32_xi0p001_eps0p0025_adami`(runner `cavity_runner.py --solver v7_single`,`SingleRunner` 代替链式 orchestrator:depth 2 的 fence 环,排空的 defrag 边界上取样、查不变量、存检查点,每 0.2 个时间单位一个样本,每 10 个时间单位一个检查点,中断后 `--resume` 从最新检查点续跑,段记在 `segments.jsonl`)。稳态判据:相邻两个约 10 个时间单位的窗口,56 个参考点上窗口平均速度的相对 L2 变化 < 10⁻³ 且流体动能相对变化 < 10⁻³;t_s 为第一个满足的窗口终点;停在 t_stop = max(100, t_s + 20),上限 200。时间平均取最后 20 个时间单位,标准误差按积分自相关时间;插值线性 MLS(Wendland C4,支撑半径 h);参考 Marchi et al. 2021 的 T_c(`docs/validation/data/marchi2021_re1000.csv`),Ghia 1982 只画;相对 L2 = √(Σ e_i² / 28) / rms(T_c);ψ_min 及位置按 513 × 513 网格积分(`cavity_analysis.stream_function_minimum`)。报告脚本 `experiment/validation/cavity_v7_single_report.py`(图、CSV、LaTeX 行、英文图注);与 E36 的 adami_rho0 1000² 运行(`v7-wall-bc` 分支,`../vulkan-demo/logs/e36/runs/n1000_k1_v7_adami_rho0`)的交叉检查用 `experiment.validation.e39_compare`。

验收:u 与 v 的相对 L2 都 < 1 %;所有不变量 0;从干净的提交运行,记录提交、physics 哈希、GPU 与驱动。稳态之后误差不随时间下降,所以不为了凑 1 % 延长运行或改参数。

结果:待填。

## 已知限制

- S 相对 V 省掉的都是不执行或不改数据的东西(清单第 11 节),所以 S 与 V 的吞吐差只能来自 host 侧(每步的 fence 建销、CONCURRENT 共享模式、未用的队列与对象);两者的 kernel 时间相同。
- defrag 开头的 compute → compute 屏障后面接 transfer 阶段的 fill,与 v7 相同(形式上第二作用域不含 TRANSFER;这些缓冲的前一次写在 1000 帧之前的 defrag 里)。去掉时间步里拷贝之后那道相邻的多余屏障、把 pool 缩到 ceil(N / 128) × 128、把 voxel_base_offset / pool_health / band_compact_meta 做成 stub,对 V 和 S 都有利,都没有实施(规格:不调参,只记录)。
- S 的两种起点:`SingleRunner.bootstrap_all`(方腔 runner)像链一样在 bootstrap 后 defrag 一次;bench 与 canonical_dump 的单卡模式像 V 一样不 defrag。两种起点各自与对应的参照一致,互相之间不逐位可比。
- `V7_DELTA_DENSITY`、`V7_DEEP_WALL_CHECK`、`V7_SPV_DIR`、`V7_BAND_WIDTHS` 等在 S 里没有对应物,设了就退出;S 没有渲染、没有 step trace、没有 `step_single_use_split` 这类验证路径。
- 每候选 ps 的"候选"定义是本文自己定的(测量 B),E5 的定义没有在仓库里找到。
- adami K = 1 在 999 步上本来就不能自复现(`v7_perf.md`),所以逐位门只做 200 步;1000² 的精度运行是生产路径(原子追加的列表顺序),同一设置重跑是同一流动的另一个实现。

## 复现

在 worktree 根目录执行;`python` = `vulkan-demo/.venv/Scripts/python.exe`;每次运行前确认无头卡空闲。

```bash
# CPU 测试与命令流比对(不碰 GPU)
python experiment/v7_single/_test_single_cpu.py
python experiment/v7_single/_compare_recordings.py --inventory --out logs/v7_single/acceptance/cpu_compare.json \
    --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml \
    --case cases/lid_driven_cavity_2d_gen/case.yaml --case cases/cavity3d_8m_b4/case.yaml --case cases/cavity3d_1m/case.yaml

# 验收全套(GPU 逐位 36 个 dump + 验证层 2 次 + CPU 比对),约 40 min
python experiment/v7_single/_run_acceptance.py all --out logs/v7_single/acceptance --device 1
# 单个逐位比较
python experiment/seam_audit/canonical_dump.py --solver v7_single --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml --device-map 1 --steps 200 --canonical-lists --out S.npz
python experiment/seam_audit/canonical_dump.py --solver v7 --v7-mode single --case ... --device-map 1 --steps 200 --canonical-lists --out V.npz
python -m experiment.seam_audit.canonical_dump --compare S.npz V.npz [--no-mask]

# 吞吐 runner 与测量 A / B / C
python experiment/v7_single/_run_single_bench.py --case cases/lid_driven_cavity_2d_gen/case.yaml --gpu-uuid ae137c668a40f5acaab90a83f7cda175 --warmup 1000 --measure 10000 --in-flight 2
python _run_single_baseline_campaign.py --out logs/v7_single/throughput_r2 --sizes 1m,2m,4m,8m,16m,32m,3d_8m \
    --configs v7_single:1,v7_single:2,v7:1,v7:2,v7x:1,v7x:2,v7_chain:2 --trials 3 --warmup 1000 --idle-check
python _summarize_v7_single_campaign.py --dir logs/v7_single/throughput_r2
python experiment/v7_single/_run_kernel_breakdown_campaign.py run --out logs/v7_single/kernel_breakdown
python experiment/v7_single/_run_kernel_breakdown_campaign.py summarize --out logs/v7_single/kernel_breakdown
python _plot_single_baseline_paper.py --dir logs/v7_single/throughput_r2 --solver v7_single --figure-a-name single_gpu_throughput --no-figure-b

# 1000^2 adami 精度运行(约 6-11 h)与报告
python -m experiment.validation.cavity_campaign calibrate --solver v7_single --runs n1000_k1_float32_xi0p001_eps0p0025_adami
python -m experiment.validation.cavity_campaign run --solver v7_single --runs n1000_k1_float32_xi0p001_eps0p0025_adami
python -m experiment.validation.cavity_v7_single_report --run-root logs/validation/cavity_re1000_v7_single \
    --run n1000_k1_float32_xi0p001_eps0p0025_adami --id-suffix _v7single --prefix cavity_re1000
python -m experiment.validation.e39_compare --v7 logs/validation/cavity_re1000_v7_single/n1000_k1_float32_xi0p001_eps0p0025_adami \
    --v6 ../vulkan-demo/logs/e36/runs/n1000_k1_v7_adami_rho0 --out logs/v7_single/cavity/vs_e36_adami_rho0
```

## 提交(分支 `v7-single`,从 `v7-rc1` = d0c8dcb 开出)

| 提交 | 内容 |
|---|---|
| (本提交) | 本文档的清单 |
