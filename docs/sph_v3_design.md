# SPH V3 Design: Beyond V2 Path A+

V3 scope = **V2 Path A+ 之后的剩余优化空间 + same-vendor (2×RTX 5090) 落地**。V2 Path A+ 把 cross-vendor 推到 78% 理论效率（273 fps cavity 1M），证明 transfer-queue 并行 + cascading interior/boundary split 的设计正确。V3 把剩下 22% 的 gap 关上，并把同 vendor 配对的潜力释放出来。

**Authoritative sources for current state**:
- `docs/sph_v2_design.md` — V2 baseline 设计（3N timeline + correction split）
- 实际实现：Path A+ 在 `commit 1214643` 落地，超过 V2 设计原本范围。本文档把那些"超 V2 的"部分正式化为 V3。
- 性能基线：`memory/project_v2_baseline_cavity_1m.md`

> **Status**: V3 设计草案。V3.0 - V3.2 待实现，V3.3 待评估。
> **Hardware target**: 2× RTX 5090 same-vendor (主); cross-vendor (NV+AMD) 持续支持（portability claim）。
> **Goal**: 把 cross-vendor 推到 ≥85% 理论效率；同 vendor 上达到 ≥90% 效率 + 350 fps @ 2M particles (paper target)。

---

## 1. V2 Path A+ 实际成绩与残余 gap

### 1.1 实测（cavity 1M cross-vendor AMD 7900XTX + NV 4060Ti）

| Configuration | fps | wall_time | scaling efficiency |
|---|---|---|---|
| Single AMD | 234 | 4274 µs | — |
| Single NV | 115 | 8696 µs | — |
| V2.0 baseline @ 2.6:1.0 | 228 | 4385 µs | 65% |
| **V2 Path A+ @ 3.2:1.0** | **273.9** | **3662 µs** | **78%** |
| 完美并行理论上限 | 349 | 2866 µs | 100% |

### 1.2 剩余 gap (~800 µs/frame = 75 fps) 拆解

```
完美并行 wall                            2866 µs
+ CPU orchestrator overhead              + 190 µs  (V3.0 目标)
+ 负载不均（AMD/NV wall 差距 410 µs）    + 205 µs  (V3.1 目标)
+ Path A+ self-overhead（barriers/ticks）+ 50 µs   (微调，~5 fps)
+ CONCURRENT-sharing NV-side regression  + ~150 µs (诊断中)
─────────────────────────────────────────────────
实测 wall                                3662 µs
```

**V3 不是要把 cross-vendor 推到 100%**——异构硬件的能力差距是物理约束，已经"逼近天花板"了。V3 真正的杠杆在**同 vendor + P2P**：2×5090 没有 PCIe DMA 慢路径，transfer chain 从 1304 µs 缩到 ~100 µs，Phase B 留下大量 slack，可以做更多 GPU 端工作或者直接以更高 fps 运行。

---

## 2. V3 阶段规划

### V3.0 — CPU Orchestrator Overhead Elimination

**目标**：把 frame_total - max(GPU_busy) 从 ~190 µs 压到 < 50 µs。

**问题诊断（需先做）**：
- 当前 `orchestrator.step()` 顺序：notify workers (2) → submit_phase_a (×2 sim) → submit_transfer_readback (×2) → submit_phase_b (×2) → submit_transfer_upload (×2) → submit_phase_c (×2) → wait_frame_done (×2 + watchdog)
- 6 个 submit 阶段 × 2 sim = **12 个 vkQueueSubmit2 调用**，每个走 python-vulkan FFI → cffi → C
- 单次 cffi 调用估 15-30 µs；12 次 ≈ 180-360 µs（跟实测 190 µs 吻合）
- 加上 watchdog 中的 dict creation、record building、Python interpreter overhead

**可选解决方案**：

| 方案 | 工作量 | 预期收益 | 复杂度 |
|---|---|---|---|
| (a) 把 12 次 submit batch 成 ≤4 次（per-queue-per-frame） | ~50 行 | -100 µs | 低 |
| (b) Orchestrator 跑独立线程，CPU 主线程零阻塞 | ~150 行 | -180 µs | 中 |
| (c) 用 C extension 包裹 hot submit loop，绕过 ffi | ~80 行 + build setup | -150 µs | 高 |
| (d) Frame pipelining (depth=2)：CPU 录下一帧时 GPU 跑当前帧 | ~300 行 | -190 µs（完全隐藏） | 极高（timeline 编号 + 资源拷贝） |

**推荐 V3.0a = (a) + (b)**：submit batching 是低风险 quick win，独立线程 orchestrator 把 CPU work 完全藏在 GPU work 后面。预期合计 +15 fps（273 → 288）。

### V3.1 — Dynamic Load Balancing (DLB)

**目标**：用 wait-time 反馈自动调整 K_split，无需手工 sweep。

**触发条件**：当 AMD 的 b_to_c_gap 和 NV 的 b_to_c_gap 长期不对称时（窗口 N 帧滑动平均），慢的那一侧需要更少粒子，快的那一侧需要更多。

**算法草案**（基于 V1 设计的 wait-time-driven K_split shift，见 `memory/project_dynamic_load_balancing.md`）：

```python
class LoadBalancer:
    def observe(self, gpu_a_wait_us, gpu_b_wait_us):
        # gpu_x_wait = max(0, frame_total - phase_a - phase_b - phase_c)
        # 即 GPU 等待时间（不是 b_to_c_gap，那是 transfer leak）
        imbalance = gpu_a_wait_us - gpu_b_wait_us  # >0 means AMD idles more
        self._window.append(imbalance)
        if len(self._window) >= 100:
            avg = mean(self._window[-100:])
            if abs(avg) > 100:  # µs threshold
                # 给 AMD 加粒子（K_split → 更高），减少 AMD idle
                delta_cols = int(avg / 50)  # 1 col per 50µs imbalance
                self.pending_k_split_delta = clamp(delta_cols, -3, +3)
                self._window.clear()
```

**K_split 切换时机**：piggyback on defrag cadence（每 1000 帧）——defrag 期间所有粒子重新 sort，正好可以重新 partition。

**实现需求**：
- partition_v2 加 `repartition(new_k_split)` 方法，可以原地修改 case + 重建 transport segments
- simulator_v2 加 `rebind_descriptors_after_repartition()` 处理 ghost pool size 变化
- 集成到 orchestrator defrag 路径

**预期收益**：现有 sweep 显示 3.2:1.0 vs 3.5:1.0 差 ~2 fps，DLB 不会显著超过 hand-tuned，但**消除手工 sweep 工作量**，跨硬件/场景自适应。

### V3.2 — Same-Vendor P2P Backend

**目标**：在 same-vendor 配对（NV+NV 或 AMD+AMD）上，用 GPU-to-GPU 直接 DMA 替代 host staging，砍掉 worker memcpy。

**已知**：
- NV+AMD shared memory `OPAQUE_WIN32` handle 互导**失败**（probe_interop.py 测过，commit 之前的 finding）
- **NV+NV 5090 配对没测过**——同 vendor 同代 + 同驱动，预期可用
- Vulkan extension: `VK_KHR_external_memory_win32` + `VK_KHR_external_semaphore_win32`

**架构方案**：在 `utils/transport_v2.py` 引入 `TransportBackend` 抽象：

```python
class TransportBackend(Protocol):
    def setup(self, source: Sim, dest: Sim, direction: str): ...
    def do_transport(self, frame_n: int) -> None:
        """Source readback + cross-GPU transfer + dest upload."""
        ...

class CpuStagingBackend:   # V2 当前实现（worker thread + host staging）
class P2PBackend:          # V3.2 新增（device→device，跳过 host）
class SharedMemoryBackend: # V3.2 备选（一份 device-local 内存被两 GPU map）
```

**P2PBackend 工作流**：
- 取代 transfer_queue 的 readback + worker memcpy + upload 三步
- 一步 device→device `vkCmdCopyBuffer` 或 SHM 直读
- 仍然在 transfer queue 上跑，跟 Phase B 并行

**预期收益（基于带宽估算）**：
- NV+NV 5090: PCIe 5.0 x16 = 64 GB/s（每方向），3.2 MB / 64 GB/s = 50 µs / direction
- transfer chain: 50 + 50 + 0 (no worker memcpy) = 100 µs（vs cross-vendor 1304 µs）
- Phase B 现状 ~1800 µs ≫ 100 µs → transfer 完全可忽略
- 实际可以**进一步缩 Phase B**（不需要 cascading split），fps 跳到接近单卡 wall

**对论文 paper target 的意义**：350 fps @ 2M particles on 2×5090 需要单卡有效带宽 ≈ 100 fps @ 1M = 10ms wall。5090 物理上能达到（HBM-like 1.79 TB/s vs 7900XTX 0.96 TB/s 的 ~2× 带宽 → 单卡 fps 接近 234 × 2 = 468 fps for 1M）。2 卡 + 78% efficiency = 730 fps @ 1M = ~365 fps @ 2M。**target 安全余量足够**。

### V3.3 — Conditional Cascading Force（仅在需要时）

V2 P3 时已经建了 `force_deep_interior` + `force_boundary` 两个 pipeline（spec const FORCE_MODE），但目前 force 仍走 `force_all` 单 dispatch in Phase C。

**触发条件**：当 transfer chain > Phase B 时（即 Phase B 的 correction + density 不够长，留出可观的 transfer leak），加 force_deep_interior 进 Phase B 扩展。

**当前 cross-vendor 不触发**：Phase B 1774 µs > transfer chain 1304 µs，留有 470 µs slack。

**可能触发场景**：
- 同 vendor 后 Phase B 变成主要"等待源"
- 高维度问题（neighbor list 更大）让 transfer chain 占更多比例
- 添加更多 cross-GPU 通信（多 GPU chain 而非 dual）

**实现负担（需要 P3.4 前置）**：
- `force.comp` 加 `FORCE_DENSITY_SOURCE` spec const（读 scratch vs primary 二选一）
- `force_deep_interior` pipeline 用 FORCE_DENSITY_SOURCE=SCRATCH 重建
- simulator 加 `cascade_force: bool` flag，Phase B/C 条件 dispatch
- 总计 ~80 行（详细 plan 见 `docs/sph_v3_cascading_force.md` 暂未创建）

**决策**：V3.3 是 V3.2 之后的可选项，按需启用。

---

## 3. V3 实施次序与里程碑

| Phase | Scope | 预期 fps（cross-vendor cavity 1M） | 工作量 |
|---|---|---|---|
| V2 Path A+ (done) | transfer Q + 5N timeline + density split | **273 fps** | 完成 |
| V3.0a | submit batching | ~283 fps | 1-2 天 |
| V3.0b | orchestrator thread | ~290 fps | 3-4 天 |
| V3.1 | DLB | ~290 fps（不依赖手 tune） | 3-5 天 |
| V3.2 | P2P backend (NV+NV) | **N/A on AMD+NV pair**；NV+NV 实测 ~500 fps | 7-10 天 |
| V3.3 | cascading force | 仅在 V3.2 后场景需要 | 2-3 天 |

**Cross-vendor 路径最终 fps 目标：~290 fps**（V3.0+V3.1 完成后）= 83% efficiency = paper portability section 的最终数字。

**Same-vendor 路径最终 fps 目标：~500-700 fps @ 1M** = 90%+ efficiency = paper primary result。

---

## 4. 关键设计原则（V3 不动 V2 已定的）

- **不引入新 timeline 值**：5N 编号 stable，新增 backend 走相同 timeline
- **不破坏 single-GPU 路径**：`_run_v2_single_bench.py` + `step_single_cmd` 维持原状
- **bench 接口 stable**：`utils/bench_v2.py` + `BenchTimer` 协议不改
- **CONCURRENT 共享留着**：即使 V3.2 P2P 不再用 host staging，仍可能跨 queue 访问 → CONCURRENT 安全网

## 5. 已知风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 5090 P2P 需要的 external_memory 扩展不在 Vulkan baseline | 中 | V3.2 受阻 | 先用 `probe_interop.py` 在 5090 上测，确认 handle 互导是否 work |
| DLB feedback loop 振荡（K_split 反复来回切换） | 低 | fps 波动 | 加 dampening + 最小切换间隔（如 ≥5000 帧） |
| Orchestrator 线程化引入新的 race | 中 | crash/corruption | 线程间用 lock-free queue (similar to current worker thread pattern) |
| CONCURRENT 在同 vendor 上仍有未量化开销 | 低 | <5% perf cost | 等 5090 实测；如果显著则切回 EXCLUSIVE + ownership transfer |

## 6. Out of Scope (后续版本)

- **>2 GPU chain**：1D partition + multi-step ghost relay。需要新的 partition 算法 + transport graph
- **3D physics**：当前 case 是 2D；3D 时 neighbor count ↑3× 触发不同 bottleneck
- **GPU async compute** 多 stream（NV 独有）：超出 Vulkan baseline 抽象
- **混合精度（fp16 for density）**：算法层面，跟 V3 编排无关
