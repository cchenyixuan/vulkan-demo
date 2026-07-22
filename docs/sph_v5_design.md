# SPH V5 Design: N-GPU Generalization via Virtual-GPU Emulation

V5 scope 有两章：**(已完成)** 2×RTX 5090 same-vendor scaling study —— 结果记录在
`docs/sph_v4_summary.md` §3c 与 `memory/project-v5-2x5090-plan.md`，本文档不重复；
**(本文档)** 把 dual-GPU 架构泛化到 **N-GPU 1D slab chain**，并用现有 2 块物理 GPU
以**虚拟 GPU（oversubscription）**方式提前开发和验证全部接口，等远程 10×3090 +
NVLink + Linux 机器到手时做到 plug-and-play。

> **Status**: M1–M5 全部完成（2026-07-22）。开发就地在 `experiment/v5/` 进行，不再 fork。
> **Hardware（开发期）**: 2× RTX 5090（device[0] 带 Windows 桌面，device[2] = AMD iGPU 忽略）。
> **Hardware（部署目标）**: 远程 10× RTX 3090，NVLink 成对桥接（pairwise，无 NVSwitch），Linux。
> **前置结论**（勿重查）: 消费级 GeForce 无 P2P（`_probe_p2p_interop.py`，2026-06-19）；
> 2×5090 只能 host-staging。NVLink P2P 后端留到远程机（Linux = OPAQUE_FD）。
>
> **本文档 §2 的全部 file:line 引用于 2026-07-15 经代码核查确认。**

---

## 1. 目标与策略

### 1.1 要什么

论文需要 GPU-count scaling（1→2→4→…→10）。手头只有 2 块卡，但 N-GPU 的**全部功能代码**
（N 路 partition、双侧 ghost、链式同步协议、多 slab 迁移）不需要 N 块物理卡来开发——
用 **K 个虚拟 GPU（VGPU）映射到 2 块物理卡**即可：每个 VGPU 是一个常驻的
`SphSimulatorV5` 实例（自己的 buffers/timeline/staging），`vgpu_index → device_index`
只是一张映射表。同一物理卡上的多个 sim 在队列上自然分时串行（"做完自己的再做别人的"），
**零状态换入换出，orchestrator 代码与真集群 100% 同构**。到 10×3090 时只改映射表 + Linux 路径。

### 1.2 模拟能验证什么 / 不能验证什么（诚实边界）

| 可完整验证（功能层） | 不可验证（性能层） |
|---|---|
| N 路 partition 正确性（切分、双侧 ghost 池、每链路 pid offset） | 绝对 fps / η（同卡 VGPU 分时会污染时序） |
| 链式同步协议无死锁、timeline 单调性 | 真 3090+NVLink 的绝对时间（硬件不同，本就不可迁移） |
| 多 slab 迁移守恒（drift=0）、与单卡数值等价 | 多卡并发 PCIe/DMA 争用的精确形态 |
| host worker memcpy 争用（10×3090 也是单机箱——2(N−1) 线程打一块 host RAM 在模拟中**真实存在**） | |

### 1.3 帧时间的正确算法：逐段测量 + 离线调度重放（不是 max(compute+comm)）

`max(compute+comm)` 是全局 barrier 模型，与 Path A+ 的设计矛盾——V5 的全部意义就是把
transfer chain 藏在 phase B 后面；把已隐藏的通信重复计入会系统性高估帧时间（大 N 时
通信完全隐藏，高估最严重），得出"扩展性差"的假结论。正确做法：

1. **逐段测量**：每个 VGPU 每步记录 phase A / readback DMA / worker memcpy /
   upload DMA / phase B / phase C 的时长（隔离、depth-1 采集）。
2. **离线重放（replay）**：帧内依赖图是确定的（就是 timeline 的等待边，见 §3.1），
   用一个小的离散事件模拟重建"这 K 个 VGPU 若真并行"的帧时间。
3. `max(compute+comm)` 仅作为上界 sanity check 一并输出。

重放模型的两个额外价值：**what-if**（把链路段时长换成 NVLink 量级 ~100µs、把 memcpy
置 0，即可预测 10×3090 的 P2P 收益，不写一行 Vulkan 代码）；**论文材料**（到真集群后
predicted vs measured 就是性能模型验证实验）。

**校准门槛**：重放模型必须先在真实 2-GPU dual 上校准——用实测段时长重放出的帧时间
与实测 dual fps 偏差 ≤5%，才允许引用虚拟 N 卡的重放数字。

---

## 2. 现状审计（2026-07-15 核查，file:line 均已验证）

### 2.1 已经 N-ready 的部分（不需要动）

- **数据模型天然按方向设计**：`DirectionalTransportSpec`（`case_v5.py:148-166`，spec
  const ids 90-94）严格 per-direction；`TransportConfig.leading/trailing` 两个
  Optional 字段可同时非空，docstring 已写明链式语义（"End-of-chain GPUs leave the
  non-peer direction as None"，`case_v5.py:154-156`）。
- **pid 空间布局支持双侧 ghost 池并存**：`[1..L] leading ghosts / [L+1..L+own] own /
  尾部 trailing`（`simulator_v5.py:605-621`）；Capacities 有独立的
  leading/trailing_ghost_pool_size（spec ids 54/55）与 ghost voxel counts（80/81）。
- **simulator 全部 per-direction 循环**：staging 分配（`simulator_v5.py:702-726`，
  sender HOST_CACHED + receiver HOST_COHERENT，persistent-mapped）；ghost_send +
  install_migrations pipelines **两个方向无条件构建**（`1036-1047`）；phase A/C 的
  per-direction dispatch（`1737-1745`、`1826-1838`）；transfer cmd 按
  `for direction in self._transport_segments` 录制（`1922-1926`）。
- **shader 无任何"我是哪块 GPU"的绑定**：身份全部来自 spec constants + case 派生；
  `GlobalStatusBuffer` 的 4 个 transport 计数字段（send/recv × leading/trailing，
  `common.glsl:502-518`）已覆盖 1D 链所需的全部方向。
- **worker 线程 pathway-generic**：`GhostMigrationWorker` 接受任意
  (source_sim, dest_sim, direction) 对（`transport_v5.py:54-66`），类内无 2 卡假设。
- **协议拓扑可无死锁泛化**：跨 sim 依赖只有"worker 中介的最近邻边"
  （src.readback → host memcpy → dst.worker_done → dst.upload → dst.phase C），
  GPU 之间从不直接等对方信号量；每帧依赖图是 DAG，帧间由自身 frame_done 链自限 1 帧偏差。

### 2.2 必须先修的协议缺陷（M1）

**5N timeline 的 worker_done（5N+3）对链条内部节点冲突。** 内部 slab 有两个入向
worker，都要 host-signal 同一个 5N+3（`transport_v5.py:193-205`）：
(a) Vulkan 禁止时间线非递增 signal——第二个 signal 非法（`transport_v5.py:198-204`
的 assert 只防 readback 竞争，不防 double-signal）；
(b) 两个方向的 upload cmd 都等这同一个 5N+3（`simulator_v5.py:2209/2216`），
第一个 worker 的 signal 会提前放行另一方向的 upload，而那个方向的 memcpy 可能还在写
receiver_staging → **静默数据损坏**。

次要问题：readback_done（5N+2）只由最后一个方向的 readback cmd signal
（`simulator_v5.py:2177-2196`），内部节点的两条出向链路被耦合（正确但过度同步，
重放模型若把链路当独立会预测错）。

### 2.3 硬编码 2 卡的三层薄壳（M2/M3）

- **partition_v5.py**：`compute_k_split` 对 `len(weights)!=2` 直接 raise（`77-78`）；
  `_build_slab_case` 只有 slot 0（trailing-only）/ slot 1（leading-only）两种角色
  （`191-192`）——interior slab 不可构造；`_compute_pid_offset` 硬编码 2-slab pid
  几何、offset 只依赖 slot0_own_pool（`294-367`）；返回签名 `(slab0, slab1, k_split)`
  （`370-377`）。
- **orchestrator_v5.py**：构造函数 `(sim_a, sim_b)`（`36-42`）；端点角色校验拒绝
  interior slab（`55-58`）；恰好 2 个 worker 硬连线（`60-68`）；bootstrap 手写双向
  memcpy 桥（`122-123`）；watchdog/记录键名 a/b（`198-215`、`230-231`）。
  **但所有提交循环体已是 `for sim in self.sims` / `for w in self.workers`**
  （`142-168`、`283-304`）——泛化是机械性的。顺手修：`:170` 的注释还写着 "3N+3"
  （Path A+ 之前的旧文本，实际是 5N+5）。
- **bench 层**：`_run_v5_dual_bench.py` weights 恰好 2 个值（`214-216`）、CSV 列名
  gpu_a/gpu_b（`118-123`）、`_WORKER_KEYS` 硬编码两条 pathway（`110-115`）、打印标签
  "AMD/a"/"NV/b"（`347`，在 2×5090 上本来就是错的）。另：CSV 里 V4 时代的
  ghost_send_*/install_* 拆分列在 V5 从未填充（`84-102`）——重放工具不得依赖。

### 2.4 其他已核实的事实与约束

- **VulkanContextV5.create 每次新建 VkInstance + VkDevice**（`vulkan_context_v5.py:283,
  375`），每 family 只取 queueCount=1（`326-336`），无注入接口。同一物理卡跑两个 sim
  = 两套 instance/device，合法但偏重；K≤8 可接受，不值得为模拟重构。
- **最小 slab 宽度**：interior slab 两侧都有 band（correction 2 / density 3 / force 4
  列每侧，`simulator_v5.py:963-967`；`NEIGHBOR_X_RANGE` 是单标量、两侧对称，
  `common.glsl:231`），own 宽度 >8 列 force 深内区才非空（>6 列 density）；
  install_migrations 的无 barrier 双方向并发证明还要求 own ≥2 列
  （`install_migrations.comp:80-87`）。1M cavity 有 200 列，K=10 → 每 slab 20 列无忧；
  几十个 VGPU 配小 case 会让 phase B 空转（正确但测不出隐藏效果）。
- **pool_safety 必须启用**：`pool_safety=None` 时每个 sim 按全局 pool 分配
  （`partition_v5.py:416-418`），K 个 sim 直接爆显存。
- **notify 背压**：worker 的 `queue.Queue(maxsize=1)` 的 `notify()` 在主线程阻塞
  （`transport_v5.py:79-80,137-143`），且 orchestrator 在 `_submit_frame` 最先发
  notify（`orchestrator_v5.py:283-294`）——2(N−1) 条链路时一条慢链路会卡住所有 sim
  的提交。M3 改为非阻塞或后置。
- **计时缺口**（M5 要补）：transfer 队列 cmd 无任何 timestamp（`simulator_v5.py:
  1865-1898`）；`BenchTimer` 只在 compute family 校验 timestampValidBits
  （`bench_v5.py:78-87`）；depth>1 流水线下 GPU timestamp 不可用
  （`orchestrator_v5.py:324-330`）；三个时钟域（两卡 GPU 时钟 + host
  perf_counter_ns）无校准（未用 VK_KHR_calibrated_timestamps）；worker 已有
  host 时钟的 {wait_ns, copy_ns, signal_ns}（`transport_v5.py:210-214`）但
  `worker.timestamps` 与 `orchestrator._records` 无界增长，长跑前要加修剪。
- **CLAUDE.md 声称的 "pluggable transport backend" 在代码里不存在**——换传输要改
  simulator 的 cmd 录制 + worker，没有接口 seam。本轮泛化顺手建出来（§3.6）。

---

## 3. 设计

### 3.1 M1 — 协议修复：per-direction transport timeline（方案 B，推荐）

每个 sim 持有 **1 条 main timeline + 每个 peer 方向 1 条 transport timeline**：

```
main timeline（3 值/帧）:      3N+1 phase_a_done   3N+2 upload_done   3N+3 frame_done
transport[dir] timeline（2 值/帧）:  2N+1 readback_done(dir)   2N+2 worker_done(dir)
```

等待关系（每帧 n，方向 dir ∈ {leading, trailing}，仅存在的 peer 方向参与）：

```
phase_a          waits main.frame_done(n-1)          signals main.phase_a_done(n)
readback[dir]    waits main.phase_a_done(n)          signals transport[dir].readback_done(n)
worker(src→dst,dir): waits src.transport[src_dir].readback_done(n)
                     AND  dst.transport[dst_dir].readback_done(n)   ← backwards-signal 防护，同今
                     memcpy 后 host-signals dst.transport[dst_dir].worker_done(n)
upload[dir]      waits transport[dir].worker_done(n) signals（最后一个方向）main.upload_done(n)
phase_c          waits main.upload_done(n)（无 peer 时 main.phase_a_done(n)） signals main.frame_done(n)
phase_b          无信号量，compute 队列排在 A 后（不变）
```

**为什么不是把单条 timeline 扩成 7N（方案 A）**：单条 timeline 上两个入向 worker 的
host signal（+4 与 +5）必须严格保序，第二个线程要先等第一个的值——跨 worker 耦合 +
易错。**为什么不是 host 侧聚合（方案 C）**：两个 memcpy 都完成才统一 signal，最简单
但快方向的 upload 被慢方向拖住（~百µs 级过度同步），后期还得重做。方案 B 中两条
transport timeline 完全独立，无跨 worker 排序约束；backwards-signal 风险仅剩
"worker 的 host signal(2n+2) 必须晚于该方向自己的 readback GPU-signal(2n+1)"——
沿用今天的 wait-dest-readback 规则即可，且下一帧的 readback(2(n+1)+1) 被
frame_done 链挡住，不可能越过。

**端点 sim（单 peer）在方案 B 下与今天行为完全一致** → 回归标准：真实 2-GPU dual
在 1M 与 8M 上 fps 与当前实现差异 ≤2%（噪声内），50k 步 drift=0。

> **M1 落地记录（2026-07-15）**：实现为可插拔 `FrameSyncScheme`
> （`experiment/v5/utils/sync_scheme_v5.py`：`AggregatedTimelineScheme` 保留原
> 5N 语义 + `PerDirectionTimelineScheme`），command buffer 零改动——两种方案的
> 全部差异只在 vkQueueSubmit2 的信号量参数。runner 开关 `--sync-scheme`，
> **默认仍为 aggregated**（对照基准保留到 M3）。ABAB 交替回归（warmup 5000）：
> 1M 583.1 vs 581.2 fps（−0.33%，组内离散内）；8M 120.0/120.0 vs 119.9/120.0
> （−0.02%）；50k 步长跑 571.6 vs 570.2 fps（−0.24%）；全部 drift=0。depth-1 逐 kernel 时间戳全部 ±1.2% 内重合；
> b_to_c_gap 两 GPU 均 −2.4%（新方案略优，量级属噪声）。位级等价不可用——
> 基线本身跨 run 不确定（atomicAdd 槽位次序 + 浮点求和次序 + 混沌放大），
> 改用聚合物理量判据：alive 精确一致，动能/动量/平均密度/质心的跨 scheme
> 差异与 run 间基线同量级（相对 1e-6~1e-10）。

### 3.2 M2 — partition N 路泛化

新 API（保留 `compute_dual_gpu_partition` 作为 N=2 薄包装）：

```python
compute_chain_partition(case, weights: list[float], *, pool_safety=1.05)
    -> ChainPartition  # .slabs: list[CaseV5]（含双侧 TransportConfig）, .cuts: list[int]
```

- **切分**：对 x 列粒子数前缀和按权重累积占比 searchsorted 出 N−1 个单调切点；
  强制每 slab own 宽度 ≥12 列（<8 列 force 深内区为空，硬拒；12–20 列告警——
  phase B 可藏空间不足）。
- **interior slab 构造**：`_build_slab_case` 从"binary slot 身份"改为按
  `(has_left_neighbor, has_right_neighbor)` 独立设置两侧 ghost thickness / 池 /
  `DirectionalTransportSpec`；扩展网格几何从相邻 slab 的实际布局推导，不再从
  k_split 标量硬推。
- **per-link pid/vid offset 重推导**（最繁琐的数学）：每条有向链路的
  `GHOST_PID_OFFSET_TO_RECEIVER = receiver_ghost_range_start − sender_boundary_range_start`，
  依赖**两端** slab 各自的 (L, own_pool, T) 布局；per-slab pool_safety 缩池后
  offset 不再有全局共享量。为 N=2 时必须与现实现**逐字段一致**（单元测试比对整个
  CaseV5）。
- **slab 元数据显式化**：CaseV5（或 sidecar struct）增加 `slot_index`、
  `own_global_x_range`、`neighbor slot ids`——weights sweep、DLB、调试都需要。

> **M2 落地记录（2026-07-16）**：`compute_chain_partition` + `ChainPartition`/
> `SlabGeometry`/`PidLayout`/`LinkSpec` + `isolate_slab`（η_weak 口子）落在
> `partition_v5.py` M2 段；旧实现原样保留为 `legacy_dual_gpu_partition`（金标
> 准参照，勿改）；`compute_dual_gpu_partition` 降为 N=2 包装
> （`minimum_own_columns=1` 精确复刻旧钳制）。**offset 代数比设计时预想的简单**：
> trailing 发送 = −(L_sender+O_sender)，leading 发送 = +(L_receiver+O_receiver)
> ——dual 的"只依赖 slot 0 池"是 L_0=0 时的特例。验收
> `_test_partition_chain.py`：**2670 项检查全过**（金标准 1M×20 组合 + 8M×4；
> 不对称权重链不变量；**独立 Capacities oracle**（接收范围平铺池空间，
> 不经产码 PidLayout，避免同义反复）；ghost_voxel_x_local/boundary spec 断言；
> 世界坐标接缝列相等；别名与全局不可变检查；GPU interior slab 构造冒烟——
> 双侧 ghost 池 + 4 staging + per-direction 双 transport 时间线在 5090 上
> 实例化成功）。对抗审查：offset 数学 CLEAN（不等池 N=3 手算 + 独立重构验证
> 双射）、8 个消费方全兼容。**M3 备忘**：(a) N 路 orchestrator 必须保留 worker
> 的 dest-readback 守卫（backwards-signal 防护）；(b) `ghost_voxel_x_local =
> extended−1` 隐含 GHOST_THICKNESS=1，加厚 ghost 时需同步改。

### 3.3 M3 — orchestrator N 化 + bench 泛化

`ChainOrchestratorV5(sims: list, *, defrag_cadence)`（或直接泛化现类）：

- links = [(i, i+1) for i in range(N−1)]，每条链路 2 个有向 worker → 2(N−1) 个；
- bootstrap 桥改为 per-link 通用循环（替代 `orchestrator_v5.py:122-123` 的手写双拷）；
- notify 改非阻塞（或移到 submit 之后），消除单慢链路卡全局；
- watchdog / per-frame records 以链路名（`"w{i}->{i+1}"`）为键；
- defrag 维持 per-sim 串行 `submit_defrag_and_wait`（泡泡随 N 线性涨，模拟期可接受，
  记为已知项）；
- bench：CSV 列由 sims/links 动态生成，删除从未填充的 V4 遗留列；打印标签用
  device 名而非 "AMD/a"。

> **M3 落地记录（2026-07-17）**：`ChainOrchestratorV5`（orchestrator_v5.py，
> Dual 类原样保留为对照）+ `_run_v5_chain_bench.py`（--weights K 值 +
> --device-map 虚拟 GPU 映射 + 内置每接缝数值完整性检查）。notify 后置 +
> worker queue_depth=4 消除了单慢链路的队头阻塞。**验收结果**：K=2 ABAB
> （同 scheme 同 pool_safety）chain 602.6 vs dual 585.0 fps——链版反而
> **+3.0%**（归因于 notify 重排移除了每帧 put 阻塞；两次 chain run 离散仅
> 0.3 fps）；**K=3 首航**（第一个 interior slab + 首次同卡双 sim 分时）2000
> 步与 50k 步均 drift=0、双接缝 overshoot ≤0.01dx、零重复；**K=4**（两个
> interior、三接缝）顺手通过。M1 的 per-direction 协议在其目标场景（interior
> 双入向 worker）首次实战即正确。

### 3.4 M4 — VGPU 映射层 + 正确性战役

- 配置：`--vgpus K --device-map 0,0,1,1,...`（长度 K）；每 VGPU 一个
  `VulkanContextV5.create(device_index=map[i])` + 一个 sim。同卡相邻 slab 的链路
  走与跨卡完全相同的代码路径（staging + worker + timeline）——正确性验证等效。
- 战役内容（2 块物理卡）：
  - K=2（回归，必须与 Dual 逐帧等价）→ K=3（首个 interior slab）→ K=4/6/8；
  - 1M 与 8M case，各 50k 步，**drift=0 硬门槛**；
  - **数值等价**：同 IC 跑 T 步，K-slab 结果 vs 单卡结果场差对比（注意跨 slab 的
    归约顺序差异，容差按 STRICT_BIT_EXACT 语义分档：迁移路径 bit-exact，
    浮点场允许 1e-6 级相对差）。

> **M4 落地记录（2026-07-21）**。**核心验收全过**：正确性战役 K=4/6/8 ×
> {1M,8M} × 50k 全部 drift=0 + 接缝净（fps：1M 330.8/226.5/162.3；8M
> 92.8/79.1/58.5）；数值等价 ALL PASS（判据两次校准：float32 底噪 ×
> 特征 SI 尺度；K=2 标定配置——|K1−K2| ≈ |K1−K1重跑| 本身即论文结论
> "验证过的分解对聚合量的扰动不超过调度随机性"）；K=3/K=4 30 分钟可视化
> 目检零异常；收发配对探针验证传输零丢失。
>
> **M4-ultra（用户追加的 48h+ 极限 soak）以负结果+完整取证结案**：K=8
> （每卡 4 VkDevice）超出 Windows 驱动可靠包线——6 次楔死（MTTF 均值
> 4.1h，min 1.9h，两卡均匀），尸检 6/6 证明 host 栈无过失（信号量计数器
> 直读：等待已满足的已提交批次不执行、设备可查询未丢失）；nvlddmkm 153
> (TDR) 事件全部出现在 teardown 后 18–30 分钟 = 事后恢复而非起因；提交/
> 信号全局锁（r3）无效证明非提交竞态。守恒丢失（−44/−85/−52，全部溢出
> 计数器为零）为同族亚致死形态——传输在信号量正常签署下静默丢数据；
> K=2 的 478 万帧零丢失把两者都限定在超额订阅形态。**生产路径不受影响**
> （K≤2/卡无瑕疵；10×3090 = Linux 每卡 1 sim，无 WDDM）。硬化产物：
> stall 尸检（120s 有界等待 + 全信号量转储）、soak 监督器（分段自动重启
> + 僵尸处理）、ghost 溢出 + 收发配对计数器全量入日志。K=4 小时级包线
> 未测（30 min 探测全绿），留作可选项。数据：`logs/soak_8m_k8_60h_*`
> （r1–r3 + p1–p3 六轮）、`logs/soak_8m_k4_30min_20260721/`。

### 3.5 M5 — 计时补全 + 重放模型

- **instrumentation**：transfer 队列独立 query pool（先按 family 校验
  timestampValidBits）给 readback/upload DMA 计时；优先探测
  `VK_KHR_calibrated_timestamps`（两卡 + host 对齐），不可用则用依赖图推断相对对齐
  （每段时长本身各时钟域内可靠）；worker 已有 host 时钟三段戳；给
  `worker.timestamps` / `_records` 加环形缓冲。
- **采集协议**：段时长在 depth-1 + 隔离条件下采集（同卡分时会互相污染；段测量期
  一次只跑一个 VGPU 的 step，多次取中位数）；生产 depth-2 的 CPU 泡泡单独测一次
  作为重放的常量输入。
- **replay 引擎**（纯 Python，~200 行）：输入 = 每 VGPU 段时长 + §3.1 依赖图 +
  设备映射（虚拟模式下解除同卡串行），输出 = 虚拟帧时间、每链路 slack、
  瓶颈路径；旁路输出 `max(compute+comm)` 上界。**先过 §1.3 的 ±5% 校准门槛。**
- **what-if 预设**：NVLink 链路（readback/upload → ~10µs，memcpy → 0）、
  x16 vs x8、去 desktop 税。

### 3.6 顺手：transport backend seam（为 NVLink P2P 铺路）

抽出最小接口（CpuStagingBackend = 现实现，仅做代码搬移不改行为）：

```python
class TransportBackend:  # per directed link
    def allocate(self, source_sim, dest_sim, direction): ...
    def record_readback_cmd(self, ...): ...   # 今：device→sender_staging DMA
    def record_upload_cmd(self, ...): ...     # 今：receiver_staging→device DMA
    def bridge(self, frame_n): ...            # 今：worker 线程 host memcpy
    def bootstrap_bridge(self): ...
```

远程机上 NVLinkP2PBackend（OPAQUE_FD 导入 + `vkCmdCopyBuffer` 直拷，bridge 为空）
按链路选择——3090 的 NVLink 是 pairwise，同一次运行中 NVLink-within-pair 与
PCIe-across 会混用，**backend 必须是 per-link 而非全局的**（这就是接口按链路
实例化的原因）。

---

## 4. 里程碑与验收

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M1** ✅ 2026-07-15 | per-direction transport timeline（§3.1）+ 修 `:170` 旧注释 | 真实 dual 1M/8M fps 差 ≤2%，50k 步 drift=0 |
| **M2** ✅ 2026-07-16 | partition N 路（§3.2） | N=2 输出与现实现逐字段一致；N=4 合成断言（切点单调、池/offset 自洽、最小宽度拒绝） |
| **M3** ✅ 2026-07-17 | orchestrator/bench N 化（§3.3） | K=2 与 Dual 回归等价；K=3 在 2 卡上首跑通过（interior slab 落地） |
| **M4** ✅ 2026-07-21 | VGPU 战役（§3.4） | K=4/6/8 × {1M, 8M} × 50k 步全部 drift=0；数值等价过容差 |
| **M5** ✅ 2026-07-22 | 计时（M5a）+ 重放（M5b）+ 仪器审查与弱扩展（M5c） | M5a: transfer 队列 GPU 时间戳全链路落地；M5b: K=2 校准重放，7 个 chain 点零拟合误差 ≤9.1%；M5c: 见下方落地记录 |

**总退出判据（远程机就绪）**：换一台机器只需要改 `--device-map`、`.venv` 路径、
glslc 路径；OPAQUE_FD probe（`_probe_p2p_interop.py` 的 Linux 版）就绪待跑。

### M5c 落地记录（2026-07-22）— 仪器审查 + 等负载弱扩展

**仪器审查**（用户质疑探针/计时脚本后发起，4 项裁定）与修复（commit `aaf42cf`）：

1. transfer-only 队列上的 `vkCmdResetQueryPool` 为**规范违规**（vk.xml queues 列表无
   TRANSFER；timestamp 写本身合法）。重置已挪入 phase A 的 compute cmd，信号量链
   （phase_a_done / frame_done）在任意流水深度下保证顺序。验收：validation 开启的
   dual 1M 冒烟零报错，DMA 中位数与 M5a 基线一致。
2. 跨 query-pool 的时间差（`*_sched_gap_us` / `*_to_c_gap_us`）**无规范背书**（未启用
   calibrated timestamps）——降级为"驱动一致的观测值"；同池差值不受影响。
3. worker 主机时间戳的 GIL 污染只影响尾部统计；均值偏低 ~5-8%，K=2 memcpy 中位数
   存活。宿主机只有**单条 32GB DIMM（单通道 DDR5，~24-27GB/s 拷贝上限）**——
   host 拷贝天花板部分是机器配置产物。
4. 全局提交/信号锁改为**按物理 GPU 分锁**（`V5_SUBMIT_LOCK_SCOPE={device,global,none}`）。
   实测：weak K=8 三种作用域无差（55.1–55.5 fps）；1M K=8 固定 N 形态 global 比
   device/none 慢 ~2.5%（159.2 vs 163.2/163.4）——旧全局锁对旧 sweep 有小幅污染，
   但**不是**线性增长的主因。

**等负载弱扩展战役**（用户设计：固定每 slab 工作量，否则 K 增大时 Phase B 隐藏窗口
收缩与协调成本增长混淆）。case 族 `cavity_weak_k{1,2,3,4,6,8}`：等高 1 m、宽度 ∝ K、
同 dx 的拉伸方腔，每 slab 恒定 ~2.0M 流体粒子（散布 <0.04%；生成器 `--half-x`，
commit `2abde25`）。配对设计：chain1_k{2,3,4}（单卡）↔ dual_k{4,6,8}（ABAB 全跨卡），
每对的单卡 slab 数相同、链长翻倍。战役 `_run_weak_scaling_campaign.py`（逐点归档
stdout + git HEAD + 锁作用域 + validation 环境），分析 `_plot_weak_scaling.py`，
图 `docs/weak_scaling_2m_per_slab.png`，数据 `logs/weak_scaling_20260722/`。

**结果（12 点全 drift=0）**：T_solo dev0/dev1 = 3898.7/3881.3 µs（桌面显示惩罚≈0）。
核心结论：

- **dual_k2（每卡 1 slab = 真实集群形态）：overhead 仅 +203 µs，η_weak 95.0%**——
  外推 N-GPU 链式集群的锚点；链式拓扑每卡 ≤2 条链路，该开销不随 N 累积。
- **每新增一条 host-staged 跨卡链路的边际成本 ~100–170 µs 且随规模递减**（配对
  增量 +339/+291/+396 µs 对应 +2/+3/+4 条链路）——传输链路亚可加，无带宽墙迹象
  （2M/slab 的隐藏窗口下）。
- **每卡多挤一个共驻 sim 的成本 ~500–1000 µs**——旧固定 N sweep 的"744µs×K 线性
  增长"主要是共驻 sim 的时间片/提交开销（VGPU 模拟产物，真实集群不存在），
  不是 memcpy 串行化。M5b 重放模型的 `memcpy_channels=1` 机制解读据此**撤回**；
  模型拟合的数值预测能力不受影响（机制简并，audit 第 4 项）。
- 本战役同时补上论文的 **η_weak 弱扩展实验**（roadmap 缺口 #1）。

**运行环境教训**：桌面活动（Blender/浏览器等）可使 1M K=8 共驻形态出现 fps 压低
~14% 甚至 drift≠0（+17/+2，一次性、不可复现、溢出计数全零）；基准窗口内保持桌面
静默，脏点必须复跑判别。

---

## 5. 风险与非目标

- **非目标：2D/3D 域分解。** 全部机制（x-slowest voxel 编码、按列的
  in_boundary_band、GlobalStatusBuffer 的 4 方向计数）是 1D x-slab 专用的；
  论文范围内维持 1D 链。
- **非目标：用模拟数字充当论文性能结果。** 模拟只出功能验证 + 重放预测；
  绝对数字等真集群。
- **GIL/cffi 风险（M3 早期验证）**：2(N−1) 个 worker 的 `vkWaitSemaphores` 在
  python-vulkan/cffi 下是否释放 GIL 未经证实；若不释放，K 大时 worker 互卡。
  验证方式：K=4 时测 worker wait 段的互相干扰；若有问题，改用超时轮询或独立
  进程池。（numpy memcpy 已确认释放 GIL。）
- **VkInstance-per-sim 的句柄开销**：K≤8 无虞；若未来 K>16 再考虑 context 共享
  改造，现在不做。
- **同卡时序污染**：所有段时长采集走隔离协议（§3.5）；device[0] 带桌面（~6%
  mem-BW 税），隔离采集优先放 device[1]。
- **显存**：K 个 sim 的 staging + 池冗余随 K 增长；1M/8M case 上 K=8 远离 32GB
  上限，64M 级别的大 case 不做高 K 模拟。

---

## 6. 参考

- `docs/sph_v4_summary.md` — V4/V5 架构与两条 scaling 曲线的权威入口（§3b 跨厂商、§3c 2×5090）
- `docs/sph_v2_design.md` — Path A+ 所继承的 V2 基线设计
- `docs/sph_v3_design.md` — V3 规划（其中 V3.2 P2P 前提已死于消费卡；NVLink 复活于远程机）
- `memory/project-roadmap-paper-and-ngpu.md` — 论文实验缺口（η_weak ⭐ / 3D / Ghia）与 10×3090 计划
- `memory/project-v5-2x5090-plan.md` — V5 第一章（2×5090 study）的完整记录
