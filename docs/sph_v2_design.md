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

1. `vkCmdPipelineBarrier2`（cross-submit memory visibility）
2. `vkCmdDispatch(correction_interior_pipeline, per_own_particle_dispatch_count, 1, 1)` — direct dispatch over own pid range；shader inline 早退 boundary 粒子（见 §7）

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

## 6. CPU Sync 逻辑

**两条完全独立的 pathway，按"哪个 GPU 先 signal 就先处理哪个"**——不是阻塞式的"等两边都好再 swap"。每条 pathway 用一个 GPU 的 ghost 生产对方 GPU 的 migration，触及的 buffer 完全不重叠，可并行：

```
# Pathway A→B：A 的 ghost 转成 B 的 migration
pathway_A_to_B:
    vkWaitSemaphores(timelineA, 3N+1, INFINITE)        # 等 A 的 ghost ready
    memcpy(staging_AtoB, A.ghost_buffer)               # 从 A 读
    remap_indices_for_B(staging_AtoB)                  # A 的 local id → B 的 local id
    memcpy(B.migration_buffer, staging_AtoB)           # 写到 B
    vkSignalSemaphore(timelineB, 3N+2)                 # B 的 boundary chain 解锁

# Pathway B→A：B 的 ghost 转成 A 的 migration
pathway_B_to_A:
    vkWaitSemaphores(timelineB, 3N+1, INFINITE)
    memcpy(staging_BtoA, B.ghost_buffer)
    remap_indices_for_A(staging_BtoA)
    memcpy(A.migration_buffer, staging_BtoA)
    vkSignalSemaphore(timelineA, 3N+2)
```

**关键性质**：A 的 ghost 数据**只**喂 B 的 migration，B 的 ghost **只**喂 A 的 migration——两条 pathway 没有任何共享中间状态，CPU 端不需要 mutex。快 GPU（先 signal 3N+1）的 ghost 立刻被处理，在慢 GPU 还在跑 Phase A 时，对方 GPU 的 migration 数据已经准备好了。

**实现选项**：

- **两个 persistent worker thread**（推荐）：每帧主线程通过 condition variable 通知两个 worker 开始，每个 worker 跑自己的 pathway。线程开销在程序启动时一次性付掉，两条 pathway 真正并行
- **Single thread + `vkWaitSemaphores` 配合 `WAIT_ANY`**：单线程轮询哪个 timeline 先到 3N+1，按到达顺序处理。简单，但两条 pathway 串行，read+remap+write 不能并行

**内存类型**：`ghost_buffer` 和 `migration_buffer` 均 `HOST_VISIBLE | HOST_COHERENT`，persistent mapped，避免显式 `vkFlushMappedMemoryRanges` 和 transfer-queue staging 复杂度。

注意 sender 端的 `ghost_buffer`（CPU 需要 memcpy 读出）建议同时申请 `HOST_CACHED` flag。仅 `HOST_COHERENT` 在部分实现下会落到 write-combined 内存类型，CPU 读侧带宽显著低于 cached 路径。这是基于 RTX 4060 Ti + RX 7900 XTX 的初步观测，跨硬件的具体行为还需要更多数据；如果设备没有暴露 `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` 的组合，回退到纯 `HOST_COHERENT` 是可接受的，host memcpy 慢一点而已。receiver 端 `migration_buffer`（CPU 写入）对 cached 与否不敏感。

**Release / acquire 语义**：`vkSignalSemaphore` 提供 release semantics 给前面的 CPU 写；GPU 的 wait 提供 acquire semantics。不需要额外 barrier。

## 7. Interior / Boundary 粒子分类

**通过 inline check 实现，不维护任何集合 / 计数器 / compaction 数组**。`update_voxel.comp` 不动，逻辑等同 V1.0。

`correction.comp` 在 main() 开头读完 self voxel_id 之后做一次整数比较，根据当前是 Submit 2（interior pass）还是 Submit 3（boundary pass）决定是否早退：

```glsl
// correction.comp
layout(constant_id = N) const uint CORRECTION_MODE = 0u;  // 0 = interior, 1 = boundary

void main() {
    uint self_particle_id = gl_GlobalInvocationID.x + own_first_pid();
    if (self_particle_id > own_last_pid()) return;

    vec4 self_position_voxel_id = position_voxel_id[self_particle_id];
    uint self_voxel_id = uint(round(self_position_voxel_id.w));
    if (self_voxel_id == VOXEL_ID_DEAD) return;

    ivec3 self_voxel_coord = own_coord_of(self_voxel_id);
    bool self_is_boundary = in_boundary_band(self_voxel_coord);

    if (CORRECTION_MODE == 0u && self_is_boundary) return;       // interior pass
    if (CORRECTION_MODE == 1u && !self_is_boundary) return;      // boundary pass

    // ... rest of correction (M_i 计算 + 求逆) ...
}
```

`in_boundary_band(coord)` 就是整数比较：粒子所在的 x 列距 partition 边界 < `NEIGHBOR_X_RANGE`。**典型 SPH 配置（h = voxel_size，27-voxel 邻居遍历）下取 `NEIGHBOR_X_RANGE = 2`**，覆盖两类粒子：

1. **自身 support 半径触及 ghost zone**：partition 紧邻的那一列（column 0），neighbor search 跨进 ghost column
2. **自身 support 半径覆盖到 migrant 落点**：column 1，neighbor search reach 到 column 0，而 migrant 在 install_migration 之后才注册到 column 0 的 voxel grid

如果只取 1，column 1 粒子在 Submit 2 的 correction 会漏掉来自 column 0 的 migrant 邻居贡献（Submit 2 时 install_migration 还没执行，column 0 的 voxel grid 里没有 migrant），M_i 计算偏差约 1 / N_neighbors（30 邻居约 3-5%），破坏 KGC first-order consistency。取 2 让 column 1 也走 boundary pass，等 Submit 3 install_migration 之后再算，得到正确 M_i。

partition 列号、boundary 列号、`NEIGHBOR_X_RANGE` 全部由 spec const 提供，编译期常量。

**两个 pipeline 共用同一份 shader 源码**，通过 spec const `CORRECTION_MODE` 区分行为：

- `correction_interior_pipeline`：`CORRECTION_MODE = 0`，处理远离边界的多数粒子
- `correction_boundary_pipeline`：`CORRECTION_MODE = 1`，处理边界列粒子 + 新迁入的 migrant

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
