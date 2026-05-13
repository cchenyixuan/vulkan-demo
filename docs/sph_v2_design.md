# SPH V2 Design: Async Multi-GPU with Sync Hiding

V2 scope = **V1 的双 GPU 数据流骨架不变（cross-vendor + CPU-staged transport + 1-thick ghost），加入：(1) timeline semaphore 替代 fence；(2) 两条独立 CPU sync pathway 在 worker thread 上并行；(3) `correction.comp` 拆 interior / boundary 两个 pipeline，用 interior pass 填 CPU sync 窗口；(4) wait-time 反馈驱动的自适应 partition**。V1 是「能跑的最慢正确版」（实测 ~60 fps for 1M），V2 把所有可隐藏的 orchestration 成本压下去，理论下限接近 `max(per-GPU compute)`。

**Authoritative sources for values**: `shaders/sph/common.glsl` for spec constants + descriptor bindings, `shaders/sph/README.md` for per-kernel invariants, `docs/sph_v1_design.md` for set 0/1/2/3 layout + V1 baseline data flow (unchanged in V2). When this document and those files disagree, trust them.

> **Status**: v2 spec, ready for engineering implementation
> **Scope**: 跨 vendor 双 GPU SPH 模拟（NVIDIA RTX 5090 + AMD RX 7900 XTX），CPU staged ghost particle 交换，timeline semaphore 同步，多线程 CPU pathway
> **Goal**: 用 GPU interior 计算填充 CPU sync 窗口，最小化 GPU stall；wait-time 反馈做自适应 partition

---

## 1. 背景

跨 vendor 双 GPU SPH 模拟。Ghost particle 通过 CPU staging 交换（跨 vendor 无 GPU P2P）。x 轴单维度 domain partitioning。V2 在 V1 baseline 之上引入异步 overlap 和反馈控制，不改变物理数据流和正确性语义。

## 2. Pipeline 一览

每帧 shader 链（顺序依赖）：

1. `predict.comp` — 基于 v, a 预测下一步粒子位置
2. `update_voxel.comp` — 更新背景 voxel 内部粒子序号
3. `ghost_send.comp` — 复制 boundary voxel 及粒子到 ghost buffer
4. **CPU sync** — 读两 GPU 的 ghost，swap，写回两 GPU 的 migration buffer
5. `install_migration.comp` — 注册跨 GPU 迁入的粒子
6. `correction.comp` — KGC 修正矩阵
7. `density.comp` — 密度和压强
8. `force.comp` — 加速度

## 3. 同步原语选型

使用 **timeline semaphore**（Vulkan 1.2+，配合 sync2 / `vkQueueSubmit2`）：

- 替代 fence（本 pipeline 无 swapchain interop，可完全替代）
- 一个 uint64 计数器折叠多个 logical fence
- 双向同步：GPU→CPU (`vkWaitSemaphores`), CPU→GPU (`vkSignalSemaphore`), GPU→GPU
- `vkGetSemaphoreCounterValue` 提供无阻塞 introspection

**Pipeline barrier 仍然必要**，用于：

- 同一 cmd buffer 内 dispatch 之间的 RAW 依赖
- Submit 内的 cache flush / 内存可见性

Pipeline barrier 和 timeline semaphore 作用层不同，互补不竞争。

## 4. Timeline Value 编号

每个 GPU 一个独立 timeline semaphore。每帧消耗 3 个 value：

| Value | Signaled by | 含义 |
|-------|-------------|------|
| 3N+1 | GPU (Submit 1 末尾) | ghost_buffer 已就绪，CPU 可读 |
| 3N+2 | CPU | migration_buffer 已写入，GPU 可继续 |
| 3N+3 | GPU (Submit 3 末尾) | 整帧完成 |

CPU 协调两个 timeline 的 wait / signal。

## 5. 每 GPU 每帧的 3 个 Submit

### 5.1 Submit 1 — Phase A

- `pWaitSemaphoreValues = [3(N-1)+3]`（上一帧结束；frame 0 等初始值 0）
- `pSignalSemaphoreValues = [3N+1]`
- `pWaitDstStageMask = [COMPUTE_SHADER]`

命令序列：

1. `vkCmdDispatch(predict)` — 全粒子范围
2. `vkCmdPipelineBarrier2`（global memory barrier，规格见 §7）
3. `vkCmdDispatch(update_voxel)` — 全粒子范围；**同时通过原子计数器写 `interior_indices[]` 和 `boundary_indices[]` 做 stream compaction**
4. `vkCmdPipelineBarrier2`
5. `vkCmdDispatch(ghost_send)` — 仅 boundary voxel 范围

### 5.2 Submit 2 — Interior Chain（sync hiding 核心）

- 无 wait（同 queue 上 Submit 1 之后自动顺序执行）
- 无 signal

命令序列：

1. `vkCmdPipelineBarrier2`（cross-submit memory visibility — 入口）
2. `vkCmdDispatch(correction_interior_pipeline, per_own_particle_dispatch_count, 1, 1)` — direct dispatch over own pid range；shader inline 早退 boundary 粒子（见 §7）
3. `vkCmdPipelineBarrier2`（**出口；存疑待排查**）— Submit 2 写了 `correction_inverse` 和 `density_gradient_kernel_sum` 的 INTERIOR pid 部分。Submit 3 的 density 要读全 own pid range 的 `correction_inverse`。Submit 2 不 signal、Submit 3 wait 的是 3N+2（worker 的 host signal，只承载 host 写可见性），所以 Submit 2 → Submit 3 的 GPU 写可见性没有 semaphore 桥接，理论上必须 explicit barrier。**TODO**：bring-up 时跑一次去掉这条 barrier 的 differential check，确认是否真的必需（同 queue 上 vkQueueSubmit 之间的隐式 memory dependency 在 spec 里有微妙措辞，需要实测验证）。

实施时 host 端 assert `per_own_particle_dispatch_count * workgroup_size >= own_last_pid() - own_first_pid() + 1`，避免 dispatch shape 与 shader 内的 `own_first_pid() ... own_last_pid()` 范围错位。

### 5.3 Submit 3 — Boundary Chain + 剩余 chain

- `pWaitSemaphoreValues = [3N+2]`
- `pSignalSemaphoreValues = [3N+3]`
- `pWaitDstStageMask = [COMPUTE_SHADER]`

命令序列：

1. `vkCmdDispatch(install_migration)` — 注册迁入粒子到 voxel buffer
2. memory barrier
3. `vkCmdDispatch(correction_boundary_pipeline, per_own_particle_dispatch_count, 1, 1)` — direct dispatch over own pid range；shader inline 早退 interior 粒子（处理 boundary + 新迁入粒子）
4. memory barrier
5. `vkCmdDispatch(density)` — 全粒子
6. memory barrier
7. `vkCmdDispatch(force)` — 全粒子

### 5.4 Frame pipelining (深度 > 1)

V2 v1 实现走「每帧末等 `3N+3` 再提交下一帧的 Submit 1」的同步节奏，与 V1 一致，便于
instrumentation 阅读和 bring-up 调试。但 timeline semaphore 的 wait/signal 编号天然
支持 CPU 端把下一帧的 Submit 1 提前排入 queue —— 例如把 frame N+1 的 Submit 1 在
frame N 的 Submit 3 还在跑时就 enqueue，让 GPU 在完成 N 的 Submit 3 后立刻接 N+1 的
Submit 1（不需要回到 CPU 一圈）。这对纯 GPU-bound 场景能再压一些 frame time。

实现接口预留：`SphSimulatorV2.submit_phase_a/b/c(frame_n)` 已经把 `frame_n` 作为入参，
orchestrator 控制是否提前 enqueue。开启深度 > 1 时唯一要变的是 orchestrator 的 `step()`
循环（不再在每帧末调 `wait_frame_done`，改为只在最后一帧或 readback 时调）和 cmd
buffer 的并发使用上界（`SIMULTANEOUS_USE_BIT` 已设，但要确认 cmd buffer pool 容量够
支撑同时 in-flight 的多帧）。这是 V2 后期 perf 优化项，不在 v1 范围。

## 6. CPU Sync 逻辑

**两条完全独立的 pathway，按"哪个 GPU 先 signal 就先处理哪个"**——不是阻塞式的"等两边都好再 swap"。每条 pathway 用一个 GPU 的 ghost 生产对方 GPU 的 migration，触及的 buffer 完全不重叠，可并行：

```
# Pathway A→B：A 的 ghost 字节拷贝到 B 的 migration
pathway_A_to_B:
    vkWaitSemaphores(timelineA, 3N+1, INFINITE)        # 等 A 的 ghost ready
    memcpy(B.migration_buffer, A.ghost_buffer)         # 纯 byte memcpy
    vkSignalSemaphore(timelineB, 3N+2)                 # B 的 boundary chain 解锁

# Pathway B→A：对称
pathway_B_to_A:
    vkWaitSemaphores(timelineB, 3N+1, INFINITE)
    memcpy(A.migration_buffer, B.ghost_buffer)
    vkSignalSemaphore(timelineA, 3N+2)
```

**重要：CPU 不做 voxel_id / pid 转换**。sender 的 `ghost_send.comp` 通过 spec const `GHOST_VOXEL_ID_OFFSET_TO_RECEIVER` 和 `GHOST_PID_OFFSET_TO_RECEIVER` 在写 packet 时已经把 `.w` 和 set 1 inside_particle_index 编码成 receiver 坐标系；receiver 的 `install_migrations.comp` 第 144 行注释里也明确「no translation needed」。worker 拿到的是已经 ready-to-install 的字节，整段 transport 与 partition 几何完全解耦——这是 design doc §1 「transport pluggable backend」 的几何基础，未来切到 P2P / 共享内存 backend 时这条 transport 路径也不需要重写。

**关键性质**：A 的 ghost 数据**只**喂 B 的 migration，B 的 ghost **只**喂 A 的 migration——两条 pathway 没有任何共享中间状态，CPU 端不需要 mutex。快 GPU（先 signal 3N+1）的 ghost 立刻被处理，在慢 GPU 还在跑 Phase A 时，对方 GPU 的 migration 数据已经准备好了。

**命名澄清**：上文 `A.ghost_buffer` / `B.migration_buffer` 指的是 **host-visible 中间 staging buffer**，不是 device-local 的 set 0/set 1 ghost 存储。device-local ghost 存储是 hot kernel（correction/density/force）的 neighbor 读取目标，频率极高，必须留在 VRAM；让它做 HOST_VISIBLE 会把 hot kernel 拉去走 PCIe，~30% neighbor 命中 ghost 时 PCIe 流量约 1.5 GB/s（1M 粒子 350 fps 量级），实际 latency + 与其它流量竞争下会卡死 hot kernel。V2 沿用 V1 的 3-跳拓扑（device-local ghost → sender host staging → memcpy → receiver host staging → device-local ghost），只把同步原语换成 timeline；不改变 device-local 与 staging 的分离。

**实现选项**（v1.0 落地选 2-thread；见 §14.1）：

- **两个 persistent worker thread**（v1.0 选定）：每帧主线程通过 condition variable 通知两个 worker 开始，每个 worker 跑自己的 pathway。线程开销在程序启动时一次性付掉，两条 pathway 真正并行
- **Single thread + `vkWaitSemaphores` 配合 `WAIT_ANY`**：单线程轮询哪个 timeline 先到 3N+1，按到达顺序处理。简单，但两条 pathway 串行，read+memcpy+write 不能并行

**内存类型**：`ghost_buffer` 和 `migration_buffer` 均 `HOST_VISIBLE | HOST_COHERENT`，persistent mapped，避免显式 `vkFlushMappedMemoryRanges` 和 transfer-queue staging 复杂度。

注意 sender 端的 `ghost_buffer`（CPU 需要 memcpy 读出）建议同时申请 `HOST_CACHED` flag。仅 `HOST_COHERENT` 在部分实现下会落到 write-combined 内存类型，CPU 读侧带宽显著低于 cached 路径。这是基于 RTX 4060 Ti + RX 7900 XTX 的初步观测，跨硬件的具体行为还需要更多数据；如果设备没有暴露 `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` 的组合，回退到纯 `HOST_COHERENT` 是可接受的，host memcpy 慢一点而已。receiver 端 `migration_buffer`（CPU 写入）对 cached 与否不敏感。

**Release / acquire 语义**：`vkSignalSemaphore` 提供 release semantics 给前面的 CPU 写；GPU 的 wait 提供 acquire semantics。不需要额外 barrier。

## 7. Interior / Boundary 粒子分类

**通过 inline check 实现，不维护任何集合 / 计数器 / compaction 数组**。`update_voxel.comp` 不动，逻辑等同 V1.0。

`correction.comp` 在 main() 开头读完 self voxel_id 之后做一次整数比较，根据当前是 Submit 2（interior pass）还是 Submit 3（boundary pass）决定是否早退：

```glsl
// correction.comp — actual values are in experiment/v2/shaders/common.glsl:
//   const uint CORRECTION_MODE_ALL      = 0u;  // V1-equivalent (default): no skip
//   const uint CORRECTION_MODE_INTERIOR = 1u;  // Submit 2: skip boundary-band particles
//   const uint CORRECTION_MODE_BOUNDARY = 2u;  // Submit 3: skip interior particles
layout(constant_id = 47) const uint CORRECTION_MODE = 0u;  // default = CORRECTION_MODE_ALL

void main() {
    uint self_particle_id = gl_GlobalInvocationID.x + own_first_pid();
    if (self_particle_id > own_last_pid()) return;

    vec4 self_position_voxel_id = position_voxel_id[self_particle_id];
    uint self_voxel_id = uint(round(self_position_voxel_id.w));
    if (self_voxel_id == VOXEL_ID_DEAD) return;

    ivec3 self_voxel_coord = own_coord_of(self_voxel_id);
    bool self_is_boundary = in_boundary_band(self_voxel_coord);

    if (CORRECTION_MODE == CORRECTION_MODE_INTERIOR &&  self_is_boundary) return;
    if (CORRECTION_MODE == CORRECTION_MODE_BOUNDARY && !self_is_boundary) return;
    // CORRECTION_MODE_ALL falls through and processes every own particle —
    // used by single-pipeline / overlap=false validation runs (§12).

    // ... rest of correction (M_i 计算 + 求逆) ...
}
```

**枚举约定**：default 取 `CORRECTION_MODE_ALL = 0` 而不是直接 `0 = interior` 是为了 §12 的 overlap=false 验证 run 能直接复用同一份 SPV 而不需要 bespoke spec data —— V2 simulator 只在 split 模式下显式把两条 pipeline 的 `CORRECTION_MODE` 分别设为 `INTERIOR` (1) 和 `BOUNDARY` (2)。

`in_boundary_band(coord)` 就是整数比较：粒子所在的 x 列距 partition 边界 < `NEIGHBOR_X_RANGE`。**典型 SPH 配置（h = voxel_size，27-voxel 邻居遍历）下取 `NEIGHBOR_X_RANGE = 2`**，覆盖两类粒子：

1. **自身 support 半径触及 ghost zone**：partition 紧邻的那一列（column 0），neighbor search 跨进 ghost column
2. **自身 support 半径覆盖到 migrant 落点**：column 1，neighbor search reach 到 column 0，而 migrant 在 install_migration 之后才注册到 column 0 的 voxel grid

如果只取 1，column 1 粒子在 Submit 2 的 correction 会漏掉来自 column 0 的 migrant 邻居贡献（Submit 2 时 install_migration 还没执行，column 0 的 voxel grid 里没有 migrant），M_i 计算偏差约 1 / N_neighbors（30 邻居约 3-5%），破坏 KGC first-order consistency。取 2 让 column 1 也走 boundary pass，等 Submit 3 install_migration 之后再算，得到正确 M_i。

partition 列号、boundary 列号、`NEIGHBOR_X_RANGE` 全部由 spec const 提供，编译期常量。

**两个 pipeline 共用同一份 shader 源码**，通过 spec const `CORRECTION_MODE` 区分行为：

- `correction_interior_pipeline`：`CORRECTION_MODE = CORRECTION_MODE_INTERIOR (1)`，处理远离边界的多数粒子
- `correction_boundary_pipeline`：`CORRECTION_MODE = CORRECTION_MODE_BOUNDARY (2)`，处理边界列粒子 + 新迁入的 migrant
- 第三档 `CORRECTION_MODE_ALL (0)` 用于 §12 的 overlap=false 单 pipeline 验证 run

**不需要**：stream compaction、atomic counter、interior_indices / boundary_indices 数组、indirect dispatch。

**Dispatch shape**：两个 pass 都用 `vkCmdDispatch` 跑 own 全 pid range（即 `_per_own_particle_dispatch_count`）。早退的 lane 成本接近零 —— defrag 之后粒子按 voxel_id 排序，boundary 粒子的 particle_id 是**连续的**，对应到 GPU 上落在**连续的 workgroup**，整个 workgroup 要么全做要么全早退，warp divergence 接近 0。

**Defrag 周期内的 coherence 衰减**：上面「整 workgroup 全跳过」的零成本严格只在 defrag 刚做完的第 0 步成立。粒子在接下来 `defrag_cadence`（默认 1000）步会漂移，boundary 粒子的 pid 连续性会被逐步打散，混合 workgroup 增多。Masked lane 本身计算成本为零，但 dispatch launch 的 workgroup 数会从 `boundary_count / wg_size` 涨到最坏接近 `N / wg_size`，dispatch overhead 膨胀几倍。对当前 1–10M 粒子规模这是 10s of μs 量级，初步判断在噪声里。

监控建议：profile boundary pass 在 defrag 第 0 步 vs 第 `cadence-1` 步的耗时差。如果衰减显著，可选缩短 `defrag_cadence` 到 100-200，或记为 known limitation 留到后续优化。

**几何依据**：interior 粒子（距 partition 边界 ≥ kernel_radius 列）的支持半径不会触及 ghost zone，也不会触及 d ≈ 0 的 migrant 粒子。所以它的 M_i = Σ_j (x_j - x_i) ⊗ ∇W_ij · V_j 计算只用 owned、非 migrant 粒子，可以在 CPU sync 完成之前执行。

**Shader 改动范围**：只有 `correction.comp` 需要加这个 inline check（+ 一个 spec const）。`predict.comp / update_voxel.comp / ghost_send.comp / install_migration.comp / density.comp / force.comp` 全部不变。

## 8. Barrier 规格

所有 dispatch 之间用 **global `VkMemoryBarrier2`**（不带 buffer/image list）：

```cpp
VkMemoryBarrier2 b{};
b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
```

若 dispatch 读写同一 buffer，两侧加 `READ_BIT | WRITE_BIT`。

理由：

- 多个 storage buffer（position, velocity, acceleration, density, pressure, KGC matrices, voxel indices）同时被读写
- Global barrier 在驱动层面的优化和细粒度 buffer barrier 基本一致
- 本 pipeline 纯 storage buffer，无 image layout 转换需求

## 9. Stall 行为

当快 GPU 完成 Submit 2 而 CPU sync 还没结束：**queue 在 Submit 3 入口的 `wait 3N+2` 处自动 stall**。这是 hardware-level wait（不消耗 SM / CU，不 busy spin）。

**Timeline semaphore wait 本身就是 stall 机制**，不需要额外同步原语。

Stall 的根本来源是数据依赖：

```
GPU2.Phase_A → CPU 读 B.ghost → CPU 写 A.migration → GPU1 可读 A.migration
```

减少 stall 的途径（见 §11 Future Extensions）：

- 减少 Phase A 不平衡（partition 调整）
- 扩 overlap 工作量（density / force 也分 interior/boundary）
- 加快 CPU sync（DMA 优化）

## 10. 诊断 Instrumentation

CPU 端每帧记录的时间戳（`std::chrono::steady_clock` 或等价 high-resolution clock）：

- `t_A_phaseA_done` — `vkWaitSemaphores(timelineA, 3N+1)` 返回时刻（A 的 ghost ready）
- `t_B_phaseA_done` — 同上 timelineB
- `t_A_unblocked` — `vkSignalSemaphore(timelineA, 3N+2)` 之后的时刻（pathway B→A 完成）
- `t_B_unblocked` — `vkSignalSemaphore(timelineB, 3N+2)` 之后的时刻（pathway A→B 完成）
- `t_A_frame_done` — `vkWaitSemaphores(timelineA, 3N+3)` 返回时刻
- `t_B_frame_done` — 同上 timelineB

派生指标：

- **Phase A 不平衡** = `|t_B_phaseA_done - t_A_phaseA_done|`
- **GPU A 的 CPU sync 占用** = `t_A_unblocked - t_A_phaseA_done`
- **GPU B 的 CPU sync 占用** = `t_B_unblocked - t_B_phaseA_done`
- **GPU X 每帧 idle 时间** = `t_X_unblocked - t_X_phaseA_done - interior_chain_time_X`
- **每 GPU 帧时间** = `t_X_frame_done - t_frame_start`

`vkGetSemaphoreCounterValue` 可以在帧中无阻塞查询当前 timeline 值，做更细粒度诊断或自适应决策（见 §11.3）。

## 11. Future Extensions（不在 v1 范围）

### 11.1 扩展 overlap 窗口

把 `density(level-2 interior)` 和 `force(level-3 interior)` 加进 overlap。需要多层粒子分类（level-k = 支持半径 k 倍范围内仍在 owned 区域的粒子）。**仅在 profile 显示 CPU sync 在 v1 之后仍然 dominant 时实施**。

### 11.2 External timeline semaphore

仅适用于同 vendor + P2P memory access（NVLink、AMD InfinityFabric）。跨 vendor (NV + AMD) 不适用。

### 11.3 自适应负载均衡

**核心思想（跨厂商策略）**：把每个 GPU 当成黑盒函数 `time = f(particle_count)`。f 的形状取决于硬件——计算 throughput、内存带宽、cache 大小、PCIe link state、atomic latency、DVFS 行为、shader compiler codegen 等等——**都不需要建模**。所有这些差异最终只 manifest 成一个 observable：**这个 GPU 让 partner 等了多久**。

跨厂商优化的全部内容就是这一条**反馈控制回路**：

```
观测：每帧每 GPU 的 wait_time（被 timeline 解锁前 idle 了多少 ms）
执行器：partition.x（哪个 voxel 列是边界）
反馈律：wait_A > wait_B → A 更快 → 把更多列分给 A（增大 own_voxel_count_A）
```

只要差异是稳定的（同样工作量同样耗时），这个反馈会收敛到平衡点。**不需要在 shader / pipeline / driver 层做任何 vendor-specific 优化**——任何 vendor-specific 慢的地方，最终都会被 partition 缩小工作量来吸收。

基于 Phase A 不平衡指标，每 1000 step 在 defrag **之前**调整 `partition.x`。defrag 按新边界自然完成大规模 particle migration；如果放在 defrag 之后调整，需要额外触发一次"一次性大量 migration"步骤。

**度量**：每帧累加每 GPU 的 phase-A-to-CPU-signal 等待时间：

```
wait_X = t_X_unblocked - t_X_phaseA_done
```

（变量名与 §10 instrumentation 一致。）直接对应 partition 不平衡导致的 GPU idle。

**决策算法**（CPU 端，不需要 shader）：

```cpp
struct LoadBalancer {
    double accum_wait_A   = 0;
    double accum_wait_B   = 0;
    double accum_phaseA_A = 0;
    double accum_phaseA_B = 0;
    int    frame_count    = 0;
};

// 每帧末尾累加
lb.accum_wait_A   += t_A_unblocked    - t_A_phaseA_done;
lb.accum_wait_B   += t_B_unblocked    - t_B_phaseA_done;
lb.accum_phaseA_A += t_A_phaseA_done  - t_frame_start;
lb.accum_phaseA_B += t_B_phaseA_done  - t_frame_start;
lb.frame_count++;

// 每 1000 frame、defrag 之前
double mean_diff   = (lb.accum_wait_A - lb.accum_wait_B) / lb.frame_count;
double phaseA_mean = (lb.accum_phaseA_A + lb.accum_phaseA_B) / (2.0 * lb.frame_count);
double ratio       = (mean_diff / 2.0) / phaseA_mean;       // 占 Phase A 时间的比例

int n_cols = std::round(ratio * total_x_voxel_cols * DAMPING);
n_cols     = std::clamp(n_cols, -MAX_STEP, +MAX_STEP);

if (std::abs(n_cols) >= 1 &&
    new_partition_valid(partition.x + n_cols * voxel_size_x))
{
    partition.x += n_cols * voxel_size_x;
    push_partition_to_gpus(partition.x);   // push constant 或 uniform buffer
}

lb.reset();
// 然后 defrag 按新 partition.x 跑
```

**符号方向**：
- `wait_A > wait_B`（`mean_diff > 0`）：A 等更久 = A 更快 = boundary 向 B 偏移 → `partition.x += n_cols * voxel_size_x`（A 多 n_cols 列）
- 反向同理

**常数建议**：

| 常数 | 推荐值 | 说明 |
|---|---|---|
| `DAMPING` | 0.5 | 每次只走一半距离（PI-style），防 oscillation |
| `MAX_STEP` | 3 | 单次最多 ±3 列，防过度调整 |
| `MIN_COLS_PER_GPU` | 4 | `new_partition_valid` 中检查 |

**跨 GPU 大粒子转移**：partition.x 变化的那一列粒子需要从一个 GPU 转给另一个。**重用现有 migration 通道**——在调整后的那一帧，让 defrag 把"列粒子"打包成大型 migration buffer 经 CPU staging 转移。`migration_buffer` 容量按"最大可能 column 列宽 × voxel 高度 × max 粒子密度" 预分配。

**冷启动**：前 3 个 defrag 周期用 `DAMPING = 0.8`（更激进），之后切回 0.5。

**监控**：每个 defrag 周期记录 `(mean_diff, n_cols, new_partition.x)`，profile 是否在 0 附近震荡。如果震荡明显：

- 加大 `DAMPING`
- 或把指标改成 `wait_X / owned_particle_count_X`（用粒子密度校正负载，处理非均匀分布）

**已知限制**：粒子全集中到一侧的退化场景（如自由表面流剧烈聚集）下，单维度 x 轴 partition 切不出平衡——这套自适应不能解决，监控并报警，不要静默 thrash。

## 12. 验证策略

启用 overlap 优化前，先达成以下两个 run 的物理结果一致（数值在合理 tolerance 内）：

- 单 GPU full-pipeline 参考 run
- 双 GPU `overlap=false` run（即 2-submit 设计，correction 不拆分）

然后启用 overlap 做 differential check。Interior / boundary 分类错误会以 partition 接缝处物理量微妙漂移的形式显现——重点检查接缝附近的密度和速度分布。

## 13. 跨 Vendor 性能注意（RTX 5090 + RX 7900 XTX）

- 所有 shader 层面的 vendor 差异由 §11.3 的 load balancer 自动吸收。**不在 shader / pipeline / dispatch 层做任何 vendor-specific 优化**——观测 wait time、调 partition、收工
- Submit count 从 2 涨到 3，验证两 vendor 上 `vkQueueSubmit2` CPU 开销可接受
- 用 §10 的 instrumentation 监控两 vendor 的 timeline semaphore signal / wait 延迟，确认无显著差异；如有，依然交给 §11.3 吸收

## 14. V2 v1.0 实现切片（Implementation Plan）

V2 spec 整体很大；v1.0 切出最小端到端可跑的版本。本节记录落地决议（2026-05-12 确定），便于后续 session 直接读 doc 接力，不依赖会话上下文。

### 14.1 类与文件结构

3 个类，2 个文件，全部位于 `experiment/v2/utils/`：

```
experiment/v2/utils/
├── simulator_v2.py
│     ├── class SphSimulatorV2         (per-GPU；与 V1 类平行但代码独立)
│     └── class GhostMigrationWorker    (per-pathway worker thread 封装)
└── orchestrator_v2.py
      └── class DualGpuOrchestratorV2  (拥有 2 个 sim + 2 个 worker)
```

**`SphSimulatorV2`** 拥有 per-GPU 的所有 Vulkan 资源：buffers、pipelines（含 V2 新增的 2 个 correction pipeline，spec const `CORRECTION_MODE` 不同）、descriptor sets、3 个 pre-recorded cmd buffer (`phase_a_cmd`/`phase_b_cmd`/`phase_c_cmd`)、独立的 defrag cmd buffer、1 个 timeline `VkSemaphore`。

**`DualGpuOrchestratorV2`** 跨 GPU 协调：frame 计数、worker thread 生命周期、§10 instrumentation 收集、未来的 LoadBalancer 挂载点。每帧 `step()` 6 个 submit + worker notify + `wait_frame_done`。

**`GhostMigrationWorker`** 封装一条 pathway（A→B 或 B→A）：持久 thread，主循环里 `vkWaitSemaphores(source.timeline, 3N+1)` → memcpy + remap + memcpy → `vkSignalSemaphore(dest.timeline, 3N+2)`。两条 pathway 触碰的 buffer 不重叠 → 无 lock。

### 14.2 时间线编号 API

`SphSimulatorV2` 封装 3N+k 编号（doc §4），对外暴露：

```python
# 编号源（caller 用来取数，例如 worker 计算 wait/signal 值）
value_phase_a_done(n)      # 3n+1
value_cpu_sync_done(n)     # 3n+2
value_frame_done(n)        # 3n+3
current_timeline_value()   # vkGetSemaphoreCounterValue 非阻塞 peek（instrumentation 用）

# 高层 frame API（orchestrator 主用）
submit_phase_a(n) / submit_phase_b(n) / submit_phase_c(n)
wait_frame_done(n)

# 底层 escape hatch（probe / 临时 submit / 未来扩展用）
submit_with_timeline(cmd, *, wait_value, signal_value)
wait_timeline(value)
```

**编号集中**在 sim 内部一处定义，worker / orchestrator / 未来 LoadBalancer 都从 `value_*` helper 取数，避免 `3*n+k` 算术散落在 3 处导致 off-by-one。`submit_phase_*` 是 `submit_with_timeline` 的薄壳，保留底层接口以支持未来 §5.4 frame pipelining 等扩展。

### 14.3 v1.0 范围内 / 范围外

**v1.0 范围内**：
- 3-submit per frame 骨架完整可跑
- 2 个 persistent worker thread (per-pathway，§6)
- timeline semaphore 替代 V1 fence
- **同步式 frame loop**：每帧末 `wait_frame_done(N)` 后再 submit N+1 的 phase_a
- §10 instrumentation 全部 6 个时间戳收集 + 派生指标打印
- §12 验证 path：`overlap=false` 退化模式 = `CORRECTION_MODE_ALL` + 单 correction pipeline，跑 V1-等价的 2-submit 风格做 differential check

**v1.0 范围外**（留给 v1.x / v2）：
- §5.4 Frame pipelining (深度 > 1)。接口预留 (`submit_phase_*(frame_n)` 已经把 `frame_n` 作为入参)，开启时只改 orchestrator 的 `step()` 循环
- §11.3 LoadBalancer / adaptive partition。v1.0 用静态 partition；instrumentation 先到位，反馈律单独迭代（避免 oscillation 调参与 bring-up 调试纠缠）
- §11.1 扩展 overlap 窗口 (density / force 也分 interior/boundary)
- §11.2 External timeline semaphore（跨 vendor 不适用）

### 14.4 编码约束

- **V2 完全自包含**：`experiment/v2/utils/` 不 import `experiment/v1/utils/` 或 `utils/sph/`。VulkanContext / buffer alloc / descriptor / pipeline build / shader load 全部 V2 内部 fresh write。V1 frozen 作为 baseline；V2 演进不能反向污染 V1。
- **不与 V1 共享基类**。两者算法相同但同步原语完全不同（V1 fence-wait 顺序、V2 timeline + 异步 overlap），抽象成本 > 收益。
- **Defrag 走独立 fence-wait pass**，不进 timeline 编号。`SphSimulatorV2.submit_defrag_and_wait()` 行为同 V1，每 `defrag_cadence` 帧由 orchestrator 调一次。这样 timeline 永远是 3N+k 干净编号，不被 defrag 打断。
- **`vkWaitSemaphores` timeout 一律 `UINT64_MAX`**（无限等）。死锁监控 = v1.x 的 watchdog 任务，不在 v1.0。
- **device feature `timelineSemaphore = VK_TRUE`** 必须在 V2 自己的 VulkanContext 构造时启用（VK 1.2 core feature，但默认 false）。

### 14.5 内存与 Transport 拓扑

- **3-hop transport（V2 v1.0 沿用 V1）**：`VRAM_sender → sender_host_staging → memcpy → receiver_host_staging → VRAM_receiver`。中间两块 staging 是分别属于 `device_sender` / `device_receiver` 的 `VkDeviceMemory`，物理上都在系统 RAM 但 Vulkan 不允许跨 device 边界访问 → CPU memcpy 是合法路径。2-hop（共享 host buffer via `VK_EXT_external_memory_host`）是 v2.x 优化候选。
- **Staging memory type**：
  - sender 侧 `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`（CPU 读优化；fallback 到 `HOST_VISIBLE | HOST_COHERENT` 单独，warn）
  - receiver 侧 `HOST_VISIBLE | HOST_COHERENT`（write-combined，CPU 顺序写优化）
- **持久映射**：sim 构造时一次性 `vkMapMemory`，destroy 时 `vkUnmapMemory`。buffer 大小固定 → 每帧 map/unmap 是纯 API 开销，无收益。
- **device-local ghost storage 不动**：set 0 / set 1 的 ghost-pid / ghost-vid 范围依然 `DEVICE_LOCAL`，hot kernel neighbor 读不走 PCIe。staging 只是 transport 的中转区，hot kernel 不读它。
- **cmd buffer 录制位置**：readback (`vkCmdCopyBuffer device→staging` + `compute→host` barrier) 折进 Phase A 末尾；upload (`vkCmdCopyBuffer staging→device` + `transfer→compute` barrier) 折进 Phase C 开头。timeline 编号保持 3N+k 不变；cmd buffer 数仍为 3 个 / sim / frame。Transport backend 切换时重录 Phase A/C cmd buffer；v1.0 只有 CpuStaging backend，此简化可接受。
