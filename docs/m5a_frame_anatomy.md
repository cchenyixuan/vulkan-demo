# M5a — 一帧的完整解剖（transfer 队列时间戳落地后首份直测报告）

**测量条件**：真实 dual（2×RTX 5090，每卡 1 sim）、per-direction scheme、
pool_safety 1.2、depth-1（全量 GPU 时间戳）、warmup 5000 后取中位数
（1M：10,000 帧；8M：7,000 帧）。数据 `logs/m5a_anatomy/anatomy_{1m,8m}_d1.csv`；
图 `docs/m5a_frame_anatomy.png`。测量于 2026-07-21（注意：本机自 07-18 起经历
6 次 TDR 驱动重置未重启，见 §5 异常项）。

**新增测量能力**（本报告与以往的差别）：readback/upload DMA 首次在 transfer
队列上直测（此前只有 `_probe_transfer_chain` 的孤立估计）；compute 与 transfer
两个 query pool 共享同一设备时钟域，跨队列偏移（DMA 调度延迟、upload 落地到
phase C 的间隙）为精确值而非推断。

## 1. 逐段耗时表（µs，中位数；两卡对称，取 gpu_a 侧）

| 段 | 1M | 8M | 备注 |
|---|---|---|---|
| predict | 11.5 | 230.9 | |
| update_voxel | 10.2 | 120.8 | |
| ghost_send dispatch | 55.3 | 106.2 | |
| **phase A 合计** | **78.6** | **457.5** | |
| correction_interior | 277.0 | 1947.9 | |
| density_deep_interior | 286.7 | 2054.4 | |
| **phase B 合计** | **564.5** | **4005.1** | 隐藏预算 |
| **b_to_c_gap（暴露的传输）** | **332.5** | **4.6** | 1M 暴露 / 8M 全隐藏 |
| install_migrations | 3.6 | 5.4 | |
| correction_boundary | 85.8 | 161.8 | |
| density_boundary(+copy) | 118.8 | 290.8 | |
| force_all | 374.8 | 2810.9 | 最大单核 |
| **phase C 合计** | **580.9** | **3273.0** | |
| **帧合计（depth-1）** | **1740.3** | **8073.8** | |

## 2. 传输链逐跳（本报告的核心新数据）

| 跳 | 1M | 8M | 解读 |
|---|---|---|---|
| DMA 调度延迟（ghost 数据就绪→readback 开始） | 44.8 | 44.8 | **与规模无关的固定协议开销**（信号量传播+队列调度） |
| readback DMA（device→host） | 154.1 | 489.7 | ~21 / ~19 GB/s |
| host 一致性屏障 | 0.8 | 1.0 | 可忽略 |
| worker memcpy（host↔host） | 425.5 | 1158.9 | **1M 链条中的最大项（50%）** |
| upload DMA（host→device） | 185.1 | 545.0 | 比 readback 慢 ~20%（方向不对称） |
| upload 落地 → phase C 启动 | 35.8 | 1554.7 | 1M=信号量延迟；8M=**1.55ms 富余 slack** |

**对账**：1M 串行链 ≈ 45+155+426+185+36 ≈ 850µs，扣除 phase B 的 564µs
隐藏预算后暴露 ≈ 285µs，与直测 b_to_c_gap 332.5µs 一阶吻合（残差来自
worker 等待与 DMA 的部分重叠计入、三时钟域尚未绝对对齐——若需闭合到 µs 级，
M5b 启用 `VK_KHR_calibrated_timestamps`，两卡均已确认支持）。
8M 链 ≈ 3.4ms 完全沉入 4.0ms 的 phase B，upload 提前 1.55ms 落地——
与 soak 时代 b_to_c_gap≈5µs 的观察闭环。

## 3. 规模标度律（1M → 8M，粒子 ×7.77，边界 ×√7.77≈2.79）

| 量 | 实测比 | 期望 | 判定 |
|---|---|---|---|
| phase B（∝N） | 7.10 | ~7.8 | ✓（5090 大 N 吞吐更优，次线性） |
| force（∝N） | 7.50 | ~7.8 | ✓ |
| readback DMA（∝√N） | 3.18 | ~2.8 | ✓ |
| worker memcpy（∝√N） | 2.72 | ~2.8 | ✓ |
| 调度延迟（常数） | 1.00 | 1.0 | ✓ 固定 45µs |

传输 ∝√N、计算 ∝N 的老结论首次全部由直测段构成。

## 4. fps 构成与 depth-2 对账

| | 1M | 8M |
|---|---|---|
| depth-1 实测 fps | 542.1 | 118.8 |
| depth-2 实测 fps | 516.4 ⚠ | 125.4 |
| depth-2 收益 | **−4.7% ⚠** | +5.6% |

8M 的 +5.6% 是 depth-2 消掉每帧 CPU 提交气泡的正常收益。**1M 的负收益是
异常**（M3.3 时代同配置 dual depth-2 为 585 fps）——见 §5。

## 5. 异常项与建议

- **疑似驱动状态劣化**：本机自 07-18 以来经历 6 次 TDR 重置未重启。1M
  dual-orchestrator depth-2 从 M3.3 时代的 585 掉到 516（−12%），而 depth-1
  与 chain 路径数字正常。**建议：M5b 校准基线采集前重启机器**，并把
  1M depth-2 复测一次作为重启前后的对照。
- worker memcpy 是 1M 传输链的半壁江山（425µs/850µs）——shared-host 跳过
  memcpy 的旧设想（曾被 V3.3 否决于整体方案）在"仅小 N"场景仍是最大杠杆；
  10×3090 的 NVLink P2P 会把整段链条（850µs）压到 ~100µs 量级。
- 调度延迟固定 45µs/跳：N-GPU 链上每帧每方向都要付,K=10 时协议固定开销
  ~90µs/帧（双向）——M5b 重放模型的必要输入。

## 6. M5b 待决输入（供讨论）

本报告给出了重放模型需要的全部分段：K=2 两个端点角色的完整段表。
尚缺：interior 角色的段表（轮换映射法采集）、发展态 regime 锚点、
校准门槛运行。是否推进由用户决定。
