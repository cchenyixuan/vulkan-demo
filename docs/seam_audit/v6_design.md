# v6 设计与链路字节清单(2026-10-03)

本文是 `experiment/v6` 的设计说明与每条链路每帧传输字节的清单,配合 [`v6.md`](v6.md)(seam 审计与性能)和 [`v6_single_step.md`](v6_single_step.md)(单步测试)。所有字节数在本机 2 × RTX 5090、K = 2、生产开关下实测(§2.1),只做分析和估算,不改代码。

记号:(KEEP, LAYERS) = (`V6_KEEP_DEPARTED`, `V6_GHOST_LAYERS`);v5 = (0,1)。face = NY·NZ(一个 x 列的 voxel 数),C = `MAX_PARTICLES_PER_VOXEL`,C_inc = `MAX_INCOMING_PER_VOXEL`,f = `V6_GHOST_POOL_FACTOR`。column 0 = 紧邻 seam 的 own 列;G1 = 紧贴 own 的 ghost 列(对方的 column 0),G2 = 再外一列(对方的 column 1)。

## 1. 设计概要

### 1.1 一步内的 kernel 顺序

一帧三段 compute 命令缓冲(A / B / C;`V6_FAST_SUBMIT=1` 时在一次提交里)加每个方向一次 readback、一次 upload(transfer 队列)和一次主机拷贝(worker 线程);B 与传输并行,C 等 upload 完成。

| 步 | 队列 | 内容 | v6 改动 |
|---|---|---|---|
| A1 predict | compute | own 流体粒子 kick + drift(读 aⁿ、shiftⁿ、v^{n−½}、xⁿ);换 voxel 的粒子追加进新 voxel 的 incoming 表(own 或 ghost voxel) | — |
| A2 update_voxel | compute | own voxel:压缩 inside 表 + 追加 incoming | — |
| A2b | compute | `departed_count` 清零(fill,前后 compute→clear / clear→compute 屏障) | KEEP=1 |
| A3 ghost_send_\<dir\> | compute | 每个有邻居的方向:计数器清零,每个 (y,z) 一个线程。LAYERS=1:column 0 的 replica(10 个字段)+ ghost voxel incoming 表里的 migrant(10 个字段)共用一段槽;LAYERS=2:column 0 / column 1 的 replica 各进自己的区段、只写 4 个字段(位置、速度质量、ρP、material),migrant 进 migrant 区段(10 个字段)。migrant 先 `store_departed`(KEEP=1)再 kill 自己的槽 | KEEP=1:store_departed;LAYERS=2:两层 replica、分区、far_migration 计数 |
| T1 readback | transfer | 每个方向整段 staging(按容量)device → host | 段表见 §2 |
| T2 worker | 主机线程 | 帧戳检查;count-aware 时只拷每个逐粒子段的 live 前缀(count × stride),voxel 表、计数字、帧戳整段 | 每个区段用自己的计数 |
| T3 upload | transfer | host → device,整段 | — |
| B1–B3 | compute | correction_interior → density_deep_interior(写 scratch)→ force_deep_interior_scratch(V6_CASCADE_FORCE=1);都不碰 band,与 T1–T3 并行 | — |
| C1 install_migrations_\<dir\> | compute | 只扫 migrant 区段(LAYERS=1 是整个混合池):`.w` 落在 own 范围的槽从 own 尾部分配 pid,登记进 own column 0 的 inside 表 | LAYERS=2:跳过两个 replica 区段 |
| C1b append_departed | compute | 每个 departed 槽一个线程,按 `.w`(本卡 ghost vid)CAS 追加进 ghost voxel 的 inside 表(不超过 C) | KEEP=1,新 kernel |
| C2 correction band | compute | own 每侧 2 列;LAYERS=2 时再加 G1 列当作 self(spec 85 `GHOST_SELF_LAYER`=1) | LAYERS=2:G1 as self |
| C3 density band | compute | own 每侧 3 列(+ G1 as self),写 scratch | 同上 |
| C4 scratch→primary | compute 队列上的 copy | own 区间;LAYERS=2 时再加每个方向的 G1 replica 区段与 departed 池 | LAYERS=2:G1 与 departed 拿到 ρⁿ⁺¹, Pⁿ⁺¹ |
| C5 force band | compute | own 每侧 4 列;读邻居的 ρ, P(LAYERS=1:ghost 为 ρⁿ, Pⁿ = 缺陷 1;LAYERS=2:G1 为 ρⁿ⁺¹, Pⁿ⁺¹) | — |

C5 之后,下一帧的 A1 才读 a、shift;C1 安装的 migrant 在 own column 0,本帧 C2 / C3 / C5 都会重算它的 L、∇ρ / kernel_sum、ρ、P、a、shift(§3 用到这一点)。

### 1.2 ghost 布局(G1 / G2)

一个两侧都有邻居的 slab 的 pid 空间与 voxel 列:

```
pid:   0 | leading ghost pool | own pool | trailing ghost pool | departed pool (KEEP=1)
                 ↑ 传输范围(每方向一段)↑                       ↑ 本地,不传输

LAYERS=1 每方向 ghost pool:  [ replica + migrant 混合,P 槽 ]                  P = ⌈face·(C+C_inc)·f⌉
LAYERS=2 每方向 ghost pool:  [ G1 replica R | G2 replica R | migrant M ]      R = ⌈face·(C+C_inc)·f⌉,M = ⌈face·C_inc·f⌉

voxel x 列(扩展网格):
LAYERS=1:  [ G  | own 0 … own N−1 | G  ]
LAYERS=2:  [ G2 G1 | own 0 … own N−1 | G1 G2 ]
```

- 发送方在 A3 把要发出的东西写进**自己的** ghost pid / ghost voxel 区(已按 Option B 平移成接收方坐标:pid 偏移 `GHOST_PID_OFFSET_TO_RECEIVER`,vid 偏移 `GHOST_VOXEL_ID_OFFSET_TO_RECEIVER`,纯 x 平移),传输把这段原样搬到接收方的 ghost 区。LAYERS=2 时同一个 vid 偏移对两列都成立;偏移目标改成接收方**最内** ghost 列(1 层时与 v5 相同)。
- G1 replica 区段只放对方 column 0 的粒子,G2 只放 column 1;每层一个原子计数器(`replica_inner/outer_send_*`),计数随传输到接收方的 `*_recv_*`(诊断,同时是 count-aware 拷贝的上界)。
- migrant 只来自 G1 列的 incoming 表(CFL 下一步最多越过一列);从 G2 列取到的 migrant 计入 `far_migration_count`,必须为 0。
- R 与 v5 的整池 P 相同(同样乘 f),所以 LAYERS=2 的池是 v5 的 2 + C_inc/(C+C_inc) 倍槽数,但 replica 每槽只传 44 B(v5 140 B)。

### 1.3 departed 池

- 位置:trailing ghost pool 之后的 D 个 pid(spec 84 `DEPARTED_POOL_SIZE`),在传输范围之外。
- A3:ghost_send 在 kill 一个 migrant 之前 `store_departed`:原子分配一个槽(`departed_count`,同时更新 `peak_departed_count`),复制全部 10 个字段,`.w` 改成该粒子进入的 ghost voxel(本卡坐标)。池满则 `overflow_departed_count` 加 1。
- C1b:`append_departed` 把每个 departed 粒子追加进它所在 ghost voxel 的 inside 表。此时 ghost 表刚被上传覆盖成对方的 replica,而对方打包 replica 时还没 install 这个粒子;追加之后本卡的 G1 列正好等于对方 install 之后的 column 0。修的是缺陷 2。
- LAYERS=2 时 departed 粒子在 G1 列表里,C2 / C3 把它当 self 重算,C4 把它的 ρⁿ⁺¹, Pⁿ⁺¹ 拷进 primary,C5 读到的是新值。
- 容量:默认 max(64, ⌈0.25 · face · 邻居侧数⌉);`V6_DEPARTED_CAPACITY` / `V6_DEPARTED_FACE_FRACTION` 覆盖。实测每帧峰值:2-D 2–5(容量 64–202),3-D 8M 197(容量 784)。

### 1.4 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `V6_KEEP_DEPARTED` | 0 | 1:departed 池 + `append_departed`(§1.3) |
| `V6_GHOST_LAYERS` | 1 | 2:两列 ghost、replica 4 字段、G1 as self、C4 覆盖 G1 与 departed;要求 KEEP=1 且 band-voxel 派发 |
| `V6_DEPARTED_CAPACITY` | 未设 | departed 池每 slab 槽数(覆盖下面的面积比例) |
| `V6_DEPARTED_FACE_FRACTION` | 0.25 | 容量 = max(64, ⌈fraction · face · 邻居侧数⌉) |
| `V6_DIAG_GHOST_SELF` | `correction,density` | 仅诊断:LAYERS=2 时哪些 band kernel 把 G1 当 self;去掉 `density` 就回到缺陷 1 |
| 继承自 v5(改名 `V6_`) | | `V6_GHOST_POOL_FACTOR`(默认 1;生产 2-D 0.25、3-D 1.0)、`V6_WORKER_COUNT_AWARE`(默认 0;生产 1)、`V6_SPLIT_TRANSFER_QUEUES`(默认 0;生产 1)、`V6_CASCADE_FORCE`(1)、`V6_BAND_VOXEL_DISPATCH`(1)、`V6_BAND_SLOT_LANES`(0)等,语义不变 |

spec 常量:83 `GHOST_LAYERS`、84 `DEPARTED_POOL_SIZE`、85 `GHOST_SELF_LAYER`(按 pipeline,只有 correction / density 的 band 变体为 1)、86 `REPLICA_REGION_SIZE`。

### 1.5 新增不变量

`GlobalStatus` 由 24 个 uint 扩到 40 个(160 B):

| 字段 | 含义 | 要求 |
|---|---|---|
| [21] `departed_count` | 本帧 departed 分配数(每帧清零) | ≤ D |
| [22] `overflow_departed_count` | 丢掉的 departed(累计):departed 池满、ghost voxel 表满,或 departed 槽的 `.w` 不在 ghost voxel 范围(理论上不可能) | **= 0**(否则运行无效) |
| [23] `far_migration_count` | 从 G2 列取到的 migrant(一步越两列,累计) | **= 0**(否则运行无效) |
| [24–27] `replica_{inner,outer}_send_{leading,trailing}_count` | LAYERS=2 两层 replica 的发送计数 | 超出 R 计入 `overflow_ghost_count` |
| [28–31] `replica_{inner,outer}_recv_*` | 接收方收到的同一计数 | 诊断 |
| PoolHealth[2] `peak_departed_count` | departed 每帧最大需求(从不清零) | 用于定容量 |

原有的不变量不变:alive 守恒(drift = 0)、GPU 帧戳与 worker 主机帧戳 = 0、所有 `overflow_*` = 0。`_run_v6_chain_bench.py`、`perf_campaign.py` 与 `single_step.py` 把 overflow_departed 与 far_migration 都当作必须为 0 的项;`dump_state.py` 只把 overflow_departed 计入有效性,far_migration 记录并告警(`WATCHED_STATUS_KEYS`),不改变 valid。

### 1.6 相对 v5 改动的文件

v6 由 v5 的 HEAD 复制(改名 + `V6_` 前缀;复制后、改动前 SPIR-V 与 v5 逐字节相同)。下表为相对"改名后的 v5"的改动行数(+ / −):

| 文件 | + / − | 内容 |
|---|---|---|
| `shaders/common.glsl` | +67 / −5 | spec 83–86;GlobalStatus 扩到 40 字段;PoolHealth.peak_departed_count |
| `shaders/helpers.glsl` | +28 / −8 | `departed_first_pid()`、`migrant_region_offset()`;band 函数加 `GHOST_SELF_LAYER` 列 |
| `shaders/ghost_send.comp` | +153 / −0 | `store_departed`;`send_two_layers` |
| `shaders/install_migrations.comp` | +5 / −3 | gpid 加 `migrant_region_offset()` |
| `shaders/append_departed.comp` | 新文件 57 行 | §1.3 |
| `utils/case_v6.py` | +28 / −7 | `departed_pool_size`、`replica_region_size`、`ghost_layers` |
| `utils/partition_v6.py` | +143 / −11 | 开关解析、departed 容量、两层 pool 布局、最内 ghost 列的 vid 偏移、seam 摘要;`restart_slab_rows`(单步测试) |
| `utils/simulator_v6.py` | +425 / −57 | GlobalStatus 解析;两层传输段;spec 83–86;`append_departed` pipeline 与派发;departed / replica 计数清零;C4 覆盖 G1 与 departed;bootstrap 的 G1 as self;`restart_init`(单步测试) |
| `utils/transport_v6.py` | +28 / −14 | count-aware 拷贝按段的 stride 与计数字;字节累计 |
| `utils/orchestrator_v6.py` | +25 / −0 | defrag 报告加 departed / far_migration;`restart_all`(单步测试) |
| `utils/bench_v6.py` | +4 / −0 | `append_departed_us` |
| `_run_v6_chain_bench.py` | +27 / −2 | seam 计数与每 link 字节;overflow / 帧戳 / far_migration 失败即报错 |
| `_test_seam_layout.py` | 新文件 | CPU:LAYERS=1 与 v5 的分区和传输段逐字段相同;LAYERS=2 的列 / pid 代数 |

其中 `restart_init` / `restart_all` / `restart_slab_rows` 是本次单步测试加的(在提交 `4d6f4ad` 之后,尚未提交),默认路径不经过它们;同时把 `_build_initial_data` 里的 material 参数打包抽成 `_material_parameters_payload`、给 `_upload_initial_state` 加了可选参数——这两处默认 bootstrap 也会走到,是不改行为的重构(之后 15 次正常 bootstrap 的运行全部有效,链路字节与性能实验逐字节相同)。

## 2. 每条链路每帧的字节分解

### 2.1 方法

`experiment/seam_audit/link_inventory.py`:对每个算例与配置,用生产开关(`V6_WORKER_COUNT_AWARE=1`、`V6_GHOST_POOL_FACTOR` = 0.25(2-D)/ 1.0(3-D)、`V6_SPLIT_TRANSFER_QUEUES=1`、cascade force、band-voxel 派发)建 K = 2 链,预热后每隔若干帧排空一次流水线,直接读两个方向的 sender staging:

- 段表就是模拟器实际录进 readback / upload 命令的 `_transport_segments`(每段:缓冲、staging 偏移、大小、stride、对应的计数字);
- **DMA 字节** = 段的大小(readback 与 upload 各一次,按容量整段);
- **host copy 字节** = count-aware worker 实际拷的量:逐粒子段 min(大小, count × stride),count 是同一 staging 里该区段的分配计数(v5 / (1,1) 是 ghost_send 计数,(1,2) 是 G1 / G2 / migrant 各自的计数);voxel 表、计数字、帧戳整段;
- 每 link 每帧 = 两个方向(s0 trailing = s0→s1,s1 leading = s1→s0)的平均;
- worker 自己的累计字节(`total_copy_bytes / copy_frame_count`)作交叉校验,并与 `v6.md` 性能一节在同样开关下实测的每 link 字节对照。

算例与采样:2-D 1M(预热 1000 帧后 2000 帧,每 20 帧采样)、2-D 4M(1000 + 1000,每 20 帧)、2-D 16M(500 + 600,每 20 帧)、3-D 8M(300 + 300,每 10 帧)。各区段的“采样最大”是这 30–100 个采样帧里的最大值,不是整段运行的峰值;整段峰值另由 PoolHealth 给出(3-D 8M 在测量窗口里有一次一整排 z 的突发:departed 197、install 201,采样恰好没有落在那一帧)。(1,1) 与 v5 的段表完全相同(departed 池不传输),表里只把 (1,1) 的 host 列单独列出以示实测相同。

记号:"replica+migrant" = v5 / (1,1) 的混合池(column 0 的 replica 与 migrant 共用一段,9 个字段);"slot counts / slot index" = ghost voxel 的 `inside_particle_count` / `inside_particle_index`,按列列出:v5 / (1,1) 只有一列 G,(1,2) 有 G1、G2 两列(同一个传输段的前后两半,大小相同,host 也整段拷);"计数字" = 分配计数(v5 1 个,(1,2) 3 个);帧戳 4 B 在 staging 最后。单位 KB(1024 B)。

### 2.2 按段、按字段

#### 2-D 1M

face = 206 voxel,C = 96,C_inc = 16,f = 0.25;v5 池 5,768 槽/方向;(1,2) 池 12,360 槽(R = 5,768);采样 100 帧(预热 1000 帧后每 20 帧一次)。

| 段 | 字段 | stride | v5 / (1,1) DMA | v5 host | (1,1) host | (1,2) DMA | (1,2) host |
|---|---|---|---|---|---|---|---|
| replica+migrant | position_voxel_id | 16 | 90.1 | 79.9 | 79.9 | – | – |
| replica+migrant | velocity_mass | 16 | 90.1 | 79.9 | 79.9 | – | – |
| replica+migrant | density_pressure | 8 | 45.1 | 40.0 | 40.0 | – | – |
| replica+migrant | acceleration | 16 | 90.1 | 79.9 | 79.9 | – | – |
| replica+migrant | shift | 16 | 90.1 | 79.9 | 79.9 | – | – |
| replica+migrant | material | 4 | 22.5 | 20.0 | 20.0 | – | – |
| replica+migrant | correction_inverse | 32 | 180.2 | 159.8 | 159.8 | – | – |
| replica+migrant | density_gradient_kernel_sum | 16 | 90.1 | 79.9 | 79.9 | – | – |
| replica+migrant | extension_fields | 16 | 90.1 | 79.9 | 79.9 | – | – |
| G1 replica | position_voxel_id | 16 | – | – | – | 90.1 | 79.9 |
| G1 replica | velocity_mass | 16 | – | – | – | 90.1 | 79.9 |
| G1 replica | density_pressure | 8 | – | – | – | 45.1 | 40.0 |
| G1 replica | material | 4 | – | – | – | 22.5 | 20.0 |
| G2 replica | position_voxel_id | 16 | – | – | – | 90.1 | 79.9 |
| G2 replica | velocity_mass | 16 | – | – | – | 90.1 | 79.9 |
| G2 replica | density_pressure | 8 | – | – | – | 45.1 | 40.0 |
| G2 replica | material | 4 | – | – | – | 22.5 | 20.0 |
| migrant | position_voxel_id | 16 | – | – | – | 12.9 | 0 B |
| migrant | velocity_mass | 16 | – | – | – | 12.9 | 0 B |
| migrant | density_pressure | 8 | – | – | – | 6.4 | 0 B |
| migrant | acceleration | 16 | – | – | – | 12.9 | 0 B |
| migrant | shift | 16 | – | – | – | 12.9 | 0 B |
| migrant | material | 4 | – | – | – | 3.2 | 0 B |
| migrant | correction_inverse | 32 | – | – | – | 25.8 | 0 B |
| migrant | density_gradient_kernel_sum | 16 | – | – | – | 12.9 | 0 B |
| migrant | extension_fields | 16 | – | – | – | 12.9 | 0 B |
| slot counts · G 列 | inside_particle_count |  | 0.8 | 0.8 | 0.8 | – | – |
| slot counts · G1 列 | inside_particle_count |  | – | – | – | 0.8 | 0.8 |
| slot counts · G2 列 | inside_particle_count |  | – | – | – | 0.8 | 0.8 |
| slot index · G 列 | inside_particle_index |  | 77.2 | 77.2 | 77.2 | – | – |
| slot index · G1 列 | inside_particle_index |  | – | – | – | 77.2 | 77.2 |
| slot index · G2 列 | inside_particle_index |  | – | – | – | 77.2 | 77.2 |
| count word | 计数字 |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| frame stamp | global_status |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| **合计** | | | 866.7 | 777.4 | 777.4 | 764.5 | 595.7 |

- v5 = (0,1):replica+migrant 5,115(采样最大 5,116,容量 5,768);整段运行峰值(PoolHealth)departed 0/0(单帧);install 计数峰值(两次 defrag 之间累计)0/27;worker 计数器 777.4/777.4 KB
- (1,1):replica+migrant 5,115(采样最大 5,116,容量 5,768);departed/帧 0.02(采样最大 1)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 1/0(单帧);install 计数峰值(两次 defrag 之间累计)0/27;worker 计数器 777.4/777.4 KB
- (1,2):G1 replica 5,115(采样最大 5,115,容量 5,768);G2 replica 5,115(采样最大 5,115,容量 5,768);migrant 0(采样最大 1,容量 824);departed/帧 0.02(采样最大 1)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 1/0(单帧);install 计数峰值(两次 defrag 之间累计)0/27;worker 计数器 595.7/595.7 KB

#### 2-D 4M

face = 410 voxel,C = 96,C_inc = 16,f = 0.25;v5 池 11,480 槽/方向;(1,2) 池 24,600 槽(R = 11,480);采样 50 帧(预热 1000 帧后每 20 帧一次)。

| 段 | 字段 | stride | v5 / (1,1) DMA | v5 host | (1,1) host | (1,2) DMA | (1,2) host |
|---|---|---|---|---|---|---|---|
| replica+migrant | position_voxel_id | 16 | 179.4 | 159.8 | 159.8 | – | – |
| replica+migrant | velocity_mass | 16 | 179.4 | 159.8 | 159.8 | – | – |
| replica+migrant | density_pressure | 8 | 89.7 | 79.9 | 79.9 | – | – |
| replica+migrant | acceleration | 16 | 179.4 | 159.8 | 159.8 | – | – |
| replica+migrant | shift | 16 | 179.4 | 159.8 | 159.8 | – | – |
| replica+migrant | material | 4 | 44.8 | 39.9 | 39.9 | – | – |
| replica+migrant | correction_inverse | 32 | 358.8 | 319.5 | 319.5 | – | – |
| replica+migrant | density_gradient_kernel_sum | 16 | 179.4 | 159.8 | 159.8 | – | – |
| replica+migrant | extension_fields | 16 | 179.4 | 159.8 | 159.8 | – | – |
| G1 replica | position_voxel_id | 16 | – | – | – | 179.4 | 159.8 |
| G1 replica | velocity_mass | 16 | – | – | – | 179.4 | 159.8 |
| G1 replica | density_pressure | 8 | – | – | – | 89.7 | 79.9 |
| G1 replica | material | 4 | – | – | – | 44.8 | 39.9 |
| G2 replica | position_voxel_id | 16 | – | – | – | 179.4 | 159.8 |
| G2 replica | velocity_mass | 16 | – | – | – | 179.4 | 159.8 |
| G2 replica | density_pressure | 8 | – | – | – | 89.7 | 79.9 |
| G2 replica | material | 4 | – | – | – | 44.8 | 39.9 |
| migrant | position_voxel_id | 16 | – | – | – | 25.6 | 0.0 |
| migrant | velocity_mass | 16 | – | – | – | 25.6 | 0.0 |
| migrant | density_pressure | 8 | – | – | – | 12.8 | 0.0 |
| migrant | acceleration | 16 | – | – | – | 25.6 | 0.0 |
| migrant | shift | 16 | – | – | – | 25.6 | 0.0 |
| migrant | material | 4 | – | – | – | 6.4 | 0.0 |
| migrant | correction_inverse | 32 | – | – | – | 51.2 | 0.0 |
| migrant | density_gradient_kernel_sum | 16 | – | – | – | 25.6 | 0.0 |
| migrant | extension_fields | 16 | – | – | – | 25.6 | 0.0 |
| slot counts · G 列 | inside_particle_count |  | 1.6 | 1.6 | 1.6 | – | – |
| slot counts · G1 列 | inside_particle_count |  | – | – | – | 1.6 | 1.6 |
| slot counts · G2 列 | inside_particle_count |  | – | – | – | 1.6 | 1.6 |
| slot index · G 列 | inside_particle_index |  | 153.8 | 153.8 | 153.8 | – | – |
| slot index · G1 列 | inside_particle_index |  | – | – | – | 153.8 | 153.8 |
| slot index · G2 列 | inside_particle_index |  | – | – | – | 153.8 | 153.8 |
| count word | 计数字 |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| frame stamp | global_status |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| **合计** | | | 1,724.9 | 1,553.3 | 1,553.3 | 1,521.5 | 1,189.4 |

- v5 = (0,1):replica+migrant 10,225(采样最大 10,227,容量 11,480);整段运行峰值(PoolHealth)departed 0/0(单帧);install 计数峰值(两次 defrag 之间累计)0/32;worker 计数器 1,553.3/1,553.3 KB
- (1,1):replica+migrant 10,225(采样最大 10,225,容量 11,480);departed/帧 0.00(采样最大 0)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 1/0(单帧);install 计数峰值(两次 defrag 之间累计)0/32;worker 计数器 1,553.3/1,553.3 KB
- (1,2):G1 replica 10,225(采样最大 10,225,容量 11,480);G2 replica 10,225(采样最大 10,225,容量 11,480);migrant 0(采样最大 0,容量 1,640);departed/帧 0.00(采样最大 0)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 1/0(单帧);install 计数峰值(两次 defrag 之间累计)0/32;worker 计数器 1,189.4/1,189.4 KB

#### 2-D 16M

face = 806 voxel,C = 96,C_inc = 16,f = 0.25;v5 池 22,568 槽/方向;(1,2) 池 48,360 槽(R = 22,568);采样 30 帧(预热 500 帧后每 20 帧一次)。

| 段 | 字段 | stride | v5 / (1,1) DMA | v5 host | (1,1) host | (1,2) DMA | (1,2) host |
|---|---|---|---|---|---|---|---|
| replica+migrant | position_voxel_id | 16 | 352.6 | 314.3 | 314.3 | – | – |
| replica+migrant | velocity_mass | 16 | 352.6 | 314.3 | 314.3 | – | – |
| replica+migrant | density_pressure | 8 | 176.3 | 157.1 | 157.1 | – | – |
| replica+migrant | acceleration | 16 | 352.6 | 314.3 | 314.3 | – | – |
| replica+migrant | shift | 16 | 352.6 | 314.3 | 314.3 | – | – |
| replica+migrant | material | 4 | 88.2 | 78.6 | 78.6 | – | – |
| replica+migrant | correction_inverse | 32 | 705.2 | 628.6 | 628.6 | – | – |
| replica+migrant | density_gradient_kernel_sum | 16 | 352.6 | 314.3 | 314.3 | – | – |
| replica+migrant | extension_fields | 16 | 352.6 | 314.3 | 314.3 | – | – |
| G1 replica | position_voxel_id | 16 | – | – | – | 352.6 | 314.3 |
| G1 replica | velocity_mass | 16 | – | – | – | 352.6 | 314.3 |
| G1 replica | density_pressure | 8 | – | – | – | 176.3 | 157.1 |
| G1 replica | material | 4 | – | – | – | 88.2 | 78.6 |
| G2 replica | position_voxel_id | 16 | – | – | – | 352.6 | 314.3 |
| G2 replica | velocity_mass | 16 | – | – | – | 352.6 | 314.3 |
| G2 replica | density_pressure | 8 | – | – | – | 176.3 | 157.1 |
| G2 replica | material | 4 | – | – | – | 88.2 | 78.6 |
| migrant | position_voxel_id | 16 | – | – | – | 50.4 | 0 B |
| migrant | velocity_mass | 16 | – | – | – | 50.4 | 0 B |
| migrant | density_pressure | 8 | – | – | – | 25.2 | 0 B |
| migrant | acceleration | 16 | – | – | – | 50.4 | 0 B |
| migrant | shift | 16 | – | – | – | 50.4 | 0 B |
| migrant | material | 4 | – | – | – | 12.6 | 0 B |
| migrant | correction_inverse | 32 | – | – | – | 100.8 | 1 B |
| migrant | density_gradient_kernel_sum | 16 | – | – | – | 50.4 | 0 B |
| migrant | extension_fields | 16 | – | – | – | 50.4 | 0 B |
| slot counts · G 列 | inside_particle_count |  | 3.1 | 3.1 | 3.1 | – | – |
| slot counts · G1 列 | inside_particle_count |  | – | – | – | 3.1 | 3.1 |
| slot counts · G2 列 | inside_particle_count |  | – | – | – | 3.1 | 3.1 |
| slot index · G 列 | inside_particle_index |  | 302.2 | 302.2 | 302.2 | – | – |
| slot index · G1 列 | inside_particle_index |  | – | – | – | 302.2 | 302.2 |
| slot index · G2 列 | inside_particle_index |  | – | – | – | 302.2 | 302.2 |
| count word | 计数字 |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| frame stamp | global_status |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| **合计** | | | 3,390.9 | 3,055.5 | 3,055.5 | 2,991.0 | 2,339.4 |

- v5 = (0,1):replica+migrant 20,115(采样最大 20,115,容量 22,568);整段运行峰值(PoolHealth)departed 0/0(单帧);install 计数峰值(两次 defrag 之间累计)0/18;worker 计数器 3,055.5/3,055.5 KB
- (1,1):replica+migrant 20,115(采样最大 20,116,容量 22,568);departed/帧 0.03(采样最大 1)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 1/0(单帧);install 计数峰值(两次 defrag 之间累计)0/18;worker 计数器 3,055.5/3,055.5 KB
- (1,2):G1 replica 20,115(采样最大 20,115,容量 22,568);G2 replica 20,115(采样最大 20,115,容量 22,568);migrant 0(采样最大 1,容量 3,224);departed/帧 0.03(采样最大 1)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 1/0(单帧);install 计数峰值(两次 defrag 之间累计)0/18;worker 计数器 2,339.4/2,339.4 KB

#### 3-D 8M

face = 3,136 voxel,C = 128,C_inc = 32,f = 1.0;v5 池 501,760 槽/方向;(1,2) 池 1,103,872 槽(R = 501,760);采样 30 帧(预热 300 帧后每 10 帧一次)。

| 段 | 字段 | stride | v5 / (1,1) DMA | v5 host | (1,1) host | (1,2) DMA | (1,2) host |
|---|---|---|---|---|---|---|---|
| replica+migrant | position_voxel_id | 16 | 7,840.0 | 2,997.6 | 2,997.6 | – | – |
| replica+migrant | velocity_mass | 16 | 7,840.0 | 2,997.6 | 2,997.6 | – | – |
| replica+migrant | density_pressure | 8 | 3,920.0 | 1,498.8 | 1,498.8 | – | – |
| replica+migrant | acceleration | 16 | 7,840.0 | 2,997.6 | 2,997.6 | – | – |
| replica+migrant | shift | 16 | 7,840.0 | 2,997.6 | 2,997.6 | – | – |
| replica+migrant | material | 4 | 1,960.0 | 749.4 | 749.4 | – | – |
| replica+migrant | correction_inverse | 32 | 15,680.0 | 5,995.1 | 5,995.1 | – | – |
| replica+migrant | density_gradient_kernel_sum | 16 | 7,840.0 | 2,997.6 | 2,997.6 | – | – |
| replica+migrant | extension_fields | 16 | 7,840.0 | 2,997.6 | 2,997.6 | – | – |
| G1 replica | position_voxel_id | 16 | – | – | – | 7,840.0 | 2,997.6 |
| G1 replica | velocity_mass | 16 | – | – | – | 7,840.0 | 2,997.6 |
| G1 replica | density_pressure | 8 | – | – | – | 3,920.0 | 1,498.8 |
| G1 replica | material | 4 | – | – | – | 1,960.0 | 749.4 |
| G2 replica | position_voxel_id | 16 | – | – | – | 7,840.0 | 2,997.6 |
| G2 replica | velocity_mass | 16 | – | – | – | 7,840.0 | 2,997.6 |
| G2 replica | density_pressure | 8 | – | – | – | 3,920.0 | 1,498.8 |
| G2 replica | material | 4 | – | – | – | 1,960.0 | 749.4 |
| migrant | position_voxel_id | 16 | – | – | – | 1,568.0 | 0.0 |
| migrant | velocity_mass | 16 | – | – | – | 1,568.0 | 0.0 |
| migrant | density_pressure | 8 | – | – | – | 784.0 | 0.0 |
| migrant | acceleration | 16 | – | – | – | 1,568.0 | 0.0 |
| migrant | shift | 16 | – | – | – | 1,568.0 | 0.0 |
| migrant | material | 4 | – | – | – | 392.0 | 0.0 |
| migrant | correction_inverse | 32 | – | – | – | 3,136.0 | 0.0 |
| migrant | density_gradient_kernel_sum | 16 | – | – | – | 1,568.0 | 0.0 |
| migrant | extension_fields | 16 | – | – | – | 1,568.0 | 0.0 |
| slot counts · G 列 | inside_particle_count |  | 12.2 | 12.2 | 12.2 | – | – |
| slot counts · G1 列 | inside_particle_count |  | – | – | – | 12.2 | 12.2 |
| slot counts · G2 列 | inside_particle_count |  | – | – | – | 12.2 | 12.2 |
| slot index · G 列 | inside_particle_index |  | 1,568.0 | 1,568.0 | 1,568.0 | – | – |
| slot index · G1 列 | inside_particle_index |  | – | – | – | 1,568.0 | 1,568.0 |
| slot index · G2 列 | inside_particle_index |  | – | – | – | 1,568.0 | 1,568.0 |
| count word | 计数字 |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| count word | 计数字 |  | – | – | – | 4 B | 4 B |
| frame stamp | global_status |  | 4 B | 4 B | 4 B | 4 B | 4 B |
| **合计** | | | 70,180.3 | 27,808.9 | 27,808.9 | 60,000.5 | 19,647.1 |

- v5 = (0,1):replica+migrant 191,844(采样最大 191,844,容量 501,760);整段运行峰值(PoolHealth)departed 0/0(单帧);install 计数峰值(两次 defrag 之间累计)0/201;worker 计数器 27,809.0/27,808.8 KB
- (1,1):replica+migrant 191,844(采样最大 191,844,容量 501,760);departed/帧 0.00(采样最大 0)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 197/0(单帧);install 计数峰值(两次 defrag 之间累计)0/201;worker 计数器 27,809.0/27,808.8 KB
- (1,2):G1 replica 191,844(采样最大 191,844,容量 501,760);G2 replica 191,844(采样最大 191,844,容量 501,760);migrant 0(采样最大 0,容量 100,352);departed/帧 0.00(采样最大 0)/0.00(采样最大 0);整段运行峰值(PoolHealth)departed 197/0(单帧);install 计数峰值(两次 defrag 之间累计)0/201;worker 计数器 19,647.2/19,647.1 KB

### 2.3 校验

段表求和与另外两种独立计量对照(KB / link / 帧):worker 自己累计的拷贝字节(同一次运行),以及 `v6.md` 性能一节在同样开关下、更长的稳态窗口里测得的每 link 字节(与 v6.md 相同:2-D 取 `logs/seam_audit/perf_quiet`,3-D 8M 取 `perf_final`)。DMA 是确定量,应逐字节相同;host 随流场(live 数)略有不同。

| 算例 | 配置 | 段表 DMA | perf DMA | 段表 host | worker 计数器 | perf host |
|---|---|---|---|---|---|---|
| 2-D 1M | v5 = (0,1) | 866.7 | 866.7 | 777.4 | 777.4 | 777.2 |
| 2-D 1M | (1,1) | 866.7 | 866.7 | 777.4 | 777.4 | 776.9 |
| 2-D 1M | (1,2) | 764.5 | 764.5 | 595.7 | 595.7 | 595.5 |
| 2-D 4M | v5 = (0,1) | 1,724.9 | 1,724.9 | 1,553.3 | 1,553.3 | 1,553.3 |
| 2-D 4M | (1,1) | 1,724.9 | 1,724.9 | 1,553.3 | 1,553.3 | 1,553.3 |
| 2-D 4M | (1,2) | 1,521.5 | 1,521.5 | 1,189.4 | 1,189.4 | 1,189.4 |
| 2-D 16M | v5 = (0,1) | 3,390.9 | 3,390.9 | 3,055.5 | 3,055.5 | 3,055.5 |
| 2-D 16M | (1,1) | 3,390.9 | 3,390.9 | 3,055.5 | 3,055.5 | 3,055.5 |
| 2-D 16M | (1,2) | 2,991.0 | 2,991.0 | 2,339.4 | 2,339.4 | 2,339.4 |
| 3-D 8M | v5 = (0,1) | 70,180.3 | 70,180.3 | 27,808.9 | 27,808.9 | 27,808.9 |
| 3-D 8M | (1,1) | 70,180.3 | 70,180.3 | 27,808.9 | 27,808.9 | 27,808.9 |
| 3-D 8M | (1,2) | 60,000.5 | 60,000.5 | 19,647.1 | 19,647.1 | 19,647.1 |

## 3. 必要性表

接收方在本帧里谁读传过来的每一项、读的是哪一列,以及去掉它的后果。"column 0" 指接收方紧邻 seam 的 own 列;C1 install、C1b append_departed、C2 correction、C3 density、C4 copy、C5 force 的编号同 §1.1;"下一步" = 下一帧 A1 predict。依据是 v6 着色器源码(`correction.comp`、`density.comp`、`force.comp`、`install_migrations.comp`、`ghost_send.comp`)与 §1.1 的顺序。

### 3.1 逐粒子字段

| 字段(字节) | replica 作邻居:LAYERS=1 的 G、LAYERS=2 的 G1 | LAYERS=2 的 G1 作 self | LAYERS=2 的 G2(只作 G1 的邻居) | migrant(C1 装进 own column 0) | 去掉的后果 |
|---|---|---|---|---|---|
| `position_voxel_id.xyz`(12) | C2:column 0 的 M、∇ρ、ΣVW;C3:x_ij 与 ∇W;C5:x_ij | C2 / C3 的 self 位置 | C2 / C3(G1 self 的邻居) | 全程(self 与邻居);下一步 predict | 邻居集合错 |
| `position_voxel_id.w`(4,接收方坐标的 vid) | 只有 LAYERS=1:C1 install 扫整个混合池,`.w` 不在 own 范围的槽当 replica 跳过 | C2 / C3 用 self 的 vid 定 voxel 坐标与 band 判断(band 派发线程其实已知 voxel) | **没人读** | C1 分类 + 登记;install 整个 vec4 照抄,装入后它就是该 own 粒子的 voxel id:C2 / C3 / C5 的 self 坐标与 band 判断、下一步 predict 与 update_voxel | LAYERS=1 replica:陈旧槽可能被误装;G1:需改为从派发得到 voxel;G2:无后果;migrant:必须 |
| `velocity_mass.xyz`(12,v^{n+½}) | C3:漂移项 v_j − v_i;C5:粘性项 | C3 self 速度 | C3(G1 self 的漂移项) | C3 / C5;下一步 predict 的起点 | 连续方程与粘性错 |
| `velocity_mass.w`(4,m) | C2 / C3:V_j = m_j/ρ_j(C5 用 self 的 m,V0 均匀质量) | — | C2 / C3 的 V_j | 全程;也是"槽已占用"的哨兵 | replica:本例所有材料 m 逐位相同(§4b),可由 self 质量代替;migrant:必须 |
| `density_pressure.x`(4,ρⁿ) | C2(V_j、ρ_j − ρ_i)、C3(V_j、ψ_ij)、**C5(V_j、PST disorder,LAYERS=1 读到的是 ρⁿ = 缺陷 1)** | C2 / C3 的 ρ_iⁿ;C4 后换成 ρⁿ⁺¹ 供 C5 | C2 / C3 | C2 的 ρ_i、C3 的积分起点 | 必须 |
| `density_pressure.y`(4,Pⁿ) | **只有 C5**(LAYERS=1,陈旧值 = 缺陷 1) | 无人读(C2 / C3 不读 P,C4 在 C5 前覆盖成 Pⁿ⁺¹) | **没人读** | 无人读(C3 / C4 在 C5 前覆盖) | LAYERS=1:C5 缺 P;其余:无后果 |
| `acceleration`(16) | 没人读(force 只写 self) | — | 没人读 | 无人读:C5 的 band(own 4 列)包含 column 0,在下一步 predict 之前就把 aⁿ⁺¹ 写好 | 无后果 |
| `shift`(16) | 没人读 | — | 没人读 | 同上,C5 重写 δr | 无后果 |
| `material`(4) | C3:self 是壁面时读邻居 material 跳过壁–壁对(上下壁面跨 seam);C2 / C5 不读 | C3 self 的 kind / EOS | C3(G1 self 为壁面时) | 全程 | 壁面密度 / 压力错 |
| `correction_inverse`(32) | 源码里 C3 读邻居 L,但只喂给 ψ_ij 被注释掉的第二项;编译后的 `density.comp.spv` 里这次加载已被删掉(绑定 7 只剩 self 的 2 个 vec4 × 3 个内联入口 = 6 次访问) | — | 同左 | C2 在 column 0 重算 | 无后果 |
| `density_gradient_kernel_sum`(16) | 邻居 ∇ρ 同上只进死代码(`density.comp.spv` 根本不再绑定 8);邻居 ΣVW 无人读(C5 只读 self 的) | — | 同左 | C2 重算 | 无后果 |
| `extension_fields`(16) | 没人读 | — | 没人读 | 没有物理;seam 审计的全局 id 靠它跟着 migrant 走 | 生产无后果;审计需要 migrant 带着它 |

**必需的每粒子字节:** replica = 位置 16(含 .w)+ 速度质量 16 + ρP 8 + material 4 = **44 B**(LAYERS=2 的 replica 已经只传这 4 个字段;v5 的 replica 传 140 B,其中 96 B 是死的);若打包,G2 只需 xyz、v、m、ρ、material = 36 B,G1 再去掉 P(C4 前无人读)与 .w(可由派发得到)也是 36 B。migrant = 同样 44 B(其中 P 那 4 B 是死的),审计时再加 16 B 的 id,共 60 B;v5 / v6 现在传 140 B。

### 3.2 slot 数组、计数字、帧戳

| 段 | 谁读 | 去掉的后果 |
|---|---|---|
| `inside_particle_count`(每个 ghost voxel 4 B,LAYERS=2 两列) | C2 / C3 / C5 的邻居循环(上界);C1b append_departed(CAS 追加);LAYERS=2 的 G1 band 派发 | ghost 不可见 |
| `inside_particle_index`(每个 ghost voxel C 个槽 × 4 B) | 邻居循环只读每个 voxel 的前 count 个槽 | 同上;但 C − count 个槽从来不读(§4a) |
| ghost_send 计数 → 接收方 `ghost_recv_<dir>_count` | C1 install 的槽上界(LAYERS=1 整个混合池,LAYERS=2 migrant 区段) | migrant 不能安装 |
| replica inner / outer 计数(LAYERS=2) | 无 kernel 读(诊断);count-aware worker 用它当拷贝上界 | worker 只能整段拷;可由 slot count 求和代替 |
| 帧戳 → 接收方 `ghost_stamp_<dir>` | C1 install 与本卡 frame_stamp 比较(`stamp_error_count`);worker 主机端检查 staging 最后 4 B | 失去传输陈旧 / 撕裂检测 |

## 4. 冗余与优化候选

只做分析和估算,不改代码。节省量 = 每 link 每帧(两个方向的平均),"DMA / host" 分别对应按容量整段的 readback(= upload)与 count-aware worker 的主机拷贝;由 §2 的实测段表计算(公式见 `experiment/seam_audit/link_inventory_tables.py` 的 `option_estimates`)。

**(a) slot 数组按容量 C 发送。** 每个 ghost voxel 的 `inside_particle_index` 占 C 个槽,邻居循环只读前 count 个;worker 对 voxel 表整段拷贝,DMA 也整段。三种做法:

- **(a1) 发 (base, count) 代替 index。** ghost_send 给每个 (y,z) voxel 用一次 atomicAdd 分配一段连续 replica 槽,所以该 voxel 的列表恰好是 `region_first + base + k`(k < count)。只发每个 voxel 的 base(4 B),接收方用一个很小的展开 kernel 在 upload 之后、C1b append_departed 之前把列表写回 [voxel][C] 布局。还原出的列表与现在**逐位相同**,不改变求和顺序 → **精确**。注意不能省掉展开、让邻居循环对 ghost voxel 直接用 base + k:KEEP=1 时 append_departed 会把 departed 粒子追加在同一列表的 count 之后,base + k 对这些槽指向的是本 voxel 的 migrant 槽或下一个 voxel 的 replica,缺陷 2 会回来。
- **(a2) 只发 live 项。** 发送方把每个 voxel 的前 count 个槽紧排(需要 voxel 级前缀和或沿用 base),接收方展开回 [voxel][C] 布局。字节比 (a1) 多出 4 B × replica 数(DMA 仍按容量预留),列表同样逐位相同 → 精确,但比 (a1) 多花字节和一次 gather。
- **(a3) 不发 index,接收方按 replica 的 `.w`(vid)原子重建。** 不多花一个字节(`.w` 本来就在 `position_voxel_id` 里),但多一个接收方 kernel(计数清零 + 原子追加),而且 voxel 内的顺序变成原子顺序:邻居**集合**相同,求和顺序不同 → 结果只差浮点求和顺序噪声(与 v5 现在 K = 1 两次运行之间的差同级),不是逐位相同。还把 `.w` 从"可删"变成"必须"(与 (b) 冲突)。
- **保持现状:** 最简单,代价见下表 (a) 列:2-D 下 index 数组占每 link 主机拷贝的一成到两成多((1,2) 两列)。

**(b) replica 的 `velocity_mass.w`(质量)与 `position.w`(vid)。** 本组算例所有材料 ρ₀·V 逐位相同(三种材料都是 ρ₀ = 1000、同一个标定体积),force 早已对邻居用 self 的质量(V0 均匀),correction / density 读 m_j 但 m_j = m_i,用 self 质量代替结果逐位不变。vid:LAYERS=2 的 G2 没人读,G1 作 self 时可由 band 派发线程已知的 voxel 得到。**但两者都在 vec4 里:**传输段是按字段整段搬 SoA,去掉 `.w` 不省字节,必须改成打包格式(ghost_send 写紧凑记录,接收方展开或让 kernel 直接读)。打包后 LAYERS=2 的 replica 各省 4 B;精确(对均匀质量的算例)。**v5 / (1,1) 的混合池省不下来:**同一个槽也装 migrant,migrant 的质量是它的状态、也是“槽已占用”的哨兵,install 又靠 `.w` 区分 replica / migrant;除非 install 改为从 material 表写回质量、并把 replica 与 migrant 分区,否则仍是 44 B。多相 / 变质量算例会失去质量信息,需要改为按 material 查表。

**(c) G2 是否需要全部 4 个字段。** 逐项见 §3.1:G2 只作为 G1(当 self)的 correction / density 邻居被读,需要 xyz、v、m、ρ、material;**`.w` 与 P 无人读**(G1 的 P 也在 C4 前无人读)。打包后 G2 每粒子 44 → 36 B(再去质量 32 B)。同样只有打包才省得下来。

**(d) migrant 的全字段。** 逐项见 §3.1:migrant 在 C1 装进 own column 0,本帧 C2 重算 L、∇ρ、ΣVW,C3 / C4 重算 ρ、P,C5(band 含 column 0)重算 a、δr,都在下一步 predict 读它们之前。所以**必须带的只有 位置(含 `.w`)、v + m、ρ、material**;P、a、δr、L、∇ρ / ΣVW 是死的,`extension_fields` 只有审计需要。这与 `ghost_send.comp` 头注释("Acceleration and shift are needed for the migration's next-step predict on receiver")不一致:按当前 phase C 的顺序,C5 在下一步 predict 之前已经重写了它们。做法是**不改格式**,只把 migrant 区段(LAYERS=1 是整个混合池)的传输段从 9 个字段减到 4 个(审计时 5 个);每槽 140 → 44 B(60 B)。精确。对 v5 / (1,1) 的混合池,这一项就是把 replica 也降到 44 B,是整张表里最大的一项。

**(e) DMA 按容量整段。** readback / upload 的命令缓冲是预录的,拷贝长度在录制时就定了,只能按容量;worker 才按 live 前缀拷。浪费 = 容量 − live。可做的是把 `V6_GHOST_POOL_FACTOR` 定在“峰值 live / f = 1 的容量”再留 25 % 余量(超出会计入 `overflow_ghost_count` 并使运行无效,所以必须按峰值而不是均值定;这里的峰值是采样最大值,3-D 加上 PoolHealth 记到的 201 个 migrant 突发也只到 0.383)。按这条规则,2-D 应为 0.28——**高于生产用的 0.25**:现在的 2-D 池只有约 12 % 余量,没有可省的,反而偏紧;3-D 8M 可从 1.0 降到 0.48。下表的 (e) 列按 (d) 之后的 44 B / 槽估算,不把 2-D 调高算进去。

### 4.1 逐项估算(KB / link / 帧,"DMA / host";− 为节省,+ 为增加)

| 算例 | 配置 | 当前 DMA / host | (a1) index → (base,count) | (a2) 只发 live 项 | (a3) 不发 index、按 .w 重建 | (b) 质量 + vid(要打包) | (c) G2 的 .w + P(要打包,.w 与 (b) 重叠) | (d) migrant / 混合区只传 4 个字段 | (e) f → 安全下限 |
|---|---|---|---|---|---|---|---|---|---|
| 2-D 1M | v5 / (1,1) | 866.7 / 777.4 | −76.4 / −76.4 | −53.9 / −56.5 | −77.2 / −77.2 | −0.0 / −0.0 | −0.0 / −0.0 | −540.8 / −479.5 | 规则给 f = 0.28 > 当前 0.25(采样峰值占用 0.89):不省 |
| 2-D 1M | (1,2) | 764.5 / 595.7 | −152.9 / −152.9 | −107.8 / −112.9 | −154.5 / −154.5 | −90.1 / −79.9 | −45.1 / −40.0 | −77.2 / −1 B | 规则给 f = 0.28 > 当前 0.25(采样峰值占用 0.89):不省 |
| 2-D 4M | v5 / (1,1) | 1,724.9 / 1,553.3 | −152.1 / −152.1 | −107.3 / −112.2 | −153.8 / −153.8 | −0.0 / −0.0 | −0.0 / −0.0 | −1,076.2 / −958.6 | 规则给 f = 0.28 > 当前 0.25(采样峰值占用 0.89):不省 |
| 2-D 4M | (1,2) | 1,521.5 / 1,189.4 | −304.3 / −304.3 | −214.6 / −224.4 | −307.5 / −307.5 | −179.4 / −159.8 | −89.7 / −79.9 | −153.8 / −0.0 | 规则给 f = 0.28 > 当前 0.25(采样峰值占用 0.89):不省 |
| 2-D 16M | v5 / (1,1) | 3,390.9 / 3,055.5 | −299.1 / −299.1 | −210.9 / −220.5 | −302.2 / −302.2 | −0.0 / −0.0 | −0.0 / −0.0 | −2,115.8 / −1,885.8 | 规则给 f = 0.28 > 当前 0.25(采样峰值占用 0.89):不省 |
| 2-D 16M | (1,2) | 2,991.0 / 2,339.4 | −598.2 / −598.2 | −421.9 / −441.1 | −604.5 / −604.5 | −352.6 / −314.3 | −176.3 / −157.1 | −302.2 / −2 B | 规则给 f = 0.28 > 当前 0.25(采样峰值占用 0.89):不省 |
| 3-D 8M | v5 / (1,1) | 70,180.3 / 27,808.9 | −1,555.8 / −1,555.8 | +404.2 / −806.4 | −1,568.0 / −1,568.0 | −0.0 / −0.0 | −0.0 / −0.0 | −47,040.0 / −17,985.4 | f 1 → 0.48(采样峰值占用 0.38):−11,211.2 / 0 |
| 3-D 8M | (1,2) | 60,000.5 / 19,647.1 | −3,111.5 / −3,111.5 | +808.5 / −1,612.7 | −3,136.0 / −3,136.0 | −7,840.0 / −2,997.6 | −3,920.0 / −1,498.8 | −9,408.0 / −0.0 | f 1 → 0.48(采样峰值占用 0.38):−24,664.6 / 0 |

| 算例 | 配置 | 当前 DMA / host (KB) | 精简后 DMA / host (KB):(a1)+(d)+(e),不改格式 | 再打包成必需字段(LAYERS=2 replica 32 B,migrant 40 B;混合池仍 44 B) |
|---|---|---|---|---|
| 2-D 1M | v5 / (1,1) | 866.7 / 777.4 | 249.5 / 221.4(29 % / 28 %) | 249.5 / 221.4 |
| 2-D 1M | (1,2) | 764.5 / 595.7 | 534.3 / 442.8(70 % / 74 %) | 395.9 / 322.9 |
| 2-D 4M | v5 / (1,1) | 1,724.9 / 1,553.3 | 496.5 / 442.6(29 % / 28 %) | 496.5 / 442.6 |
| 2-D 4M | (1,2) | 1,521.5 / 1,189.4 | 1,063.5 / 885.1(70 % / 74 %) | 788.0 / 645.5 |
| 2-D 16M | v5 / (1,1) | 3,390.9 / 3,055.5 | 976.0 / 870.6(29 % / 28 %) | 976.0 / 870.6 |
| 2-D 16M | (1,2) | 2,991.0 / 2,339.4 | 2,090.6 / 1,741.2(70 % / 74 %) | 1,549.0 / 1,269.8 |
| 3-D 8M | v5 / (1,1) | 70,180.3 / 27,808.9 | 10,373.3 / 8,267.8(15 % / 30 %) | 10,373.3 / 8,267.8 |
| 3-D 8M | (1,2) | 60,000.5 / 19,647.1 | 22,816.4 / 16,535.6(38 % / 84 %) | 16,983.4 / 12,039.3 |

各项占当前字节的比例(DMA / host):

| 项 | v5 / (1,1),2-D | v5 / (1,1),3-D 8M | (1,2),2-D | (1,2),3-D 8M | 精确? | 代价 |
|---|---|---|---|---|---|---|
| (a1) index → (base, count) | 9 % / 10 % | 2 % / 6 % | 20 % / 26 % | 5 % / 16 % | 逐位相同(须用展开 kernel,见上) | ghost_send 写 base(已算出);接收方在 upload 后、append_departed 前展开 |
| (a2) 只发 live 项 | 6 % / 7 % | −1 %(DMA 反增)/ 3 % | 14 % / 19 % | −1 % / 8 % | 逐位相同 | 发送方紧排 + 接收方展开;全面劣于 (a1) |
| (a3) 不发 index,按 `.w` 重建 | 9 % / 10 % | 2 % / 6 % | 20 % / 26 % | 5 % / 16 % | 邻居集合相同,求和顺序变 | 接收方多一个原子重建 kernel;`.w` 变成必须 |
| (b) 质量 + vid(打包) | 0(混合池里 migrant 的质量与 `.w` 都必须带) | 0 | 12 % / 13 % | 13 % / 15 %((e) 之后 DMA 约 6 %) | 均匀质量算例逐位相同 | 打包格式 |
| (c) G2 的 `.w` + P(打包) | — | — | 6 % / 7 % | 7 % / 8 %((e) 之后 DMA 约 3 %) | 逐位相同 | 打包格式 |
| (d) 混合池 / migrant 只传 4 个字段 | **62 % / 62 %** | **67 % / 65 %** | 10 % / 0 % | 16 % / 0 % | 逐位相同 | 只改传输段表(9 → 4 个字段;审计再加 id) |
| (e) pool factor → 峰值占用 × 1.25 | 0(规则给 0.28 > 当前 0.25,现有余量仅 ~12 %) | 16 % / 0(f 1 → 0.48) | 0(同左) | 41 % / 0 | 不溢出即逐位相同(溢出会使运行无效) | 改一个环境变量 |

### 4.2 结论

- **最大的一项是 (d),而且最便宜。** v5 / (1,1) 的混合池现在每槽传 140 B,接收方只读其中 44 B(§3):replica 的 a、δr、L、∇ρ / ΣVW、id 没人读,migrant 的这些量和 P 在下一步 predict 之前都会被 C2–C5 重算。把混合池的传输段从 9 个字段减到 4 个(审计时 5 个),2-D 每 link 少 62 % 的 DMA 和主机拷贝,3-D 少 65–67 %,结果逐位不变,不需要打包格式。LAYERS=2 的 replica 已经这样做了,只剩 migrant 区段(2-D 10 %、3-D 16 % 的 DMA)。
- **slot 数组选 (a1)。** 发每个 ghost voxel 的 base 代替 C 个槽的 index:replica 本来就按 voxel 连续分配,用一个展开 kernel(upload 之后、append_departed 之前)还原出的列表与现在逐位相同。2-D 省 9–10 %((1,2) 两列,20–26 %),3-D 2–6 %((1,2) 5–16 %)。(a3) 省同样多的字节,但改变求和顺序、要多一个 kernel,并把 `.w` 变成必须;(a2) 在每个算例上都不如 (a1),3-D 的 DMA 还会变多(紧排的列表要按 replica 容量预留)。保持现状的代价就是这 2–26 %(2-D 9–26 %)。
- **(b)、(c) 只有打包才省得下来,而且只对 (1,2)。** 质量在这组算例里逐位均匀,vid 与 G2 / G1 的 P 无人读,但它们都在 vec4 / vec2 里,按字段整段搬的传输去不掉;打包后 (1,2) 再省 12–15 %(质量 + vid)与 6–8 %(G2 的 `.w` + P)——这是 2-D 与主机拷贝的比例;3-D 的 DMA 在做了 (e) 之后只剩约 6 % 与 3 %。v5 / (1,1) 的混合池里 migrant 的质量与 `.w` 都必须带,打包不省。优先级低于 (d)、(a1)、(e)。
- **(e) 只对 3-D 有空间,2-D 反而偏紧。** 2-D 生产设置 f = 0.25 时 replica 区段采样峰值占用已是 0.89(余量约 12 %,按 25 % 的规则应为 0.28);3-D 8M 在 f = 1.0 时峰值只有 0.38,降到 0.48(留 25 %)可再省 16 %((1,1))/ 41 %((1,2))的 DMA。按峰值定、不是均值;超出会计入 `overflow_ghost_count` 并让运行无效,所以改之前要用目标算例的最坏帧验证。目前 2-D 与 3-D 8M 的传输链都藏在 phase B 后面(`v6.md` 的 b→c 间隙只有几 µs),所以这些节省换来的是隐藏余量(更小的问题、更多的 GPU)与主机内存带宽,不是当前的 fps。
- **合起来((a1) + (d) + (e),都不改格式):** (1,1) 降到当前的 29 % / 28 %(2-D DMA / host)与 15 % / 30 %(3-D);这比当前的 (1,2) 还少约 2.7 倍(2-D host 221 对 596 KB / link / 帧,1M)。(1,2) 降到 70 % / 74 %(2-D)与 38 % / 84 %(3-D),再打包到 52 % / 54 %(2-D)、28 % / 61 %(3-D)。(1,1) 打包不再省。`v6.md` 推荐的 (1,1) 是精简后字节最少的配置。
- 以上都是由实测段表推出的估算,没有实现或测速;(d)、(a1)、(e) 的精确性依据是 §3 的读者清单,实现时应当用 `_test_seam_layout.py` 的段表检查与单步测试(`v6_single_step.md`,看 column 0 逐位相同)验证。
