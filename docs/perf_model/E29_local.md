# E29:§5 的逐步测量仪器与本地验证(2 × RTX 5090,K = 2,2026-10-05)

**状态:第一段(仪器)完成,检查 a–c 通过,检查 d(开销)在 2-D 1M 没有达到 ≤ 1 %(三个版本的仪器、每个 3 对交替:−1.27 ± 0.71 %、−1.14 ± 0.58 %、−1.45 ± 0.33 %;2-D 16M 分辨不出)。按任务要求停在第二段之前,等你决定。** 第二段的扫描与模型工具已经写好并在检查运行上跑通(`step_trace_campaign.py --mode scan`、`step_trace_model.py`),没有跑扫描。

分支 `v4-multigpu-orchestration`,代码提交 `8780b75`(基于 E6b 之后的 HEAD);生产默认(发布组合),depth 2,3000 步、warmup 1000(steady = 后 2000 步)。每次计时运行前用 nvidia-smi 确认没有别的 python 计算进程(campaign 日志里每次都记了两卡的利用率、功率、SM 时钟;GPU 0 带桌面,利用率读数一直在 11–14 %,所以只把别的 python 计算进程当作"忙")。这一轮所有运行都没有遇到别的计算进程。

## 记录字段(E7 沿用)

开关:`experiment/v6/_run_v6_chain_bench.py --step-trace DIR`(默认关;关时录制与提交的 GPU 命令和以前完全相同),`--step-trace-detail phases|full`(默认 phases:各 phase 的开始与最后一个 tick、两个 ghost_send 的结束、全部传输 tick;full 另加 phase 内每个 kernel 的 tick),`--step-trace-calibrate-ms`(默认 500)。实现:`experiment/v6/utils/phase_trace_v6.py` 的 `StepTracer` / `StepTraceTimer`;共享文件只改了 `simulator_v6.py`(按帧奇偶各录一份 phase A/B 与传输命令,只在 trace 打开时)和 `transport_v6.py`(worker 的主机时钟可替换、多一个帧戳检查时间点、每步拷贝字节)。每次运行一个目录,四个文件;时间一律是**主机时钟上的 ns**(整数)。

**时间基准。** GPU 时间戳经 `VK_KHR_calibrated_timestamps` 映射到主机时钟:一个辅助线程每 500 ms 采一次(每次连采 3 对、取驱动 maxDeviation 最小的一对;驱动调用不占主循环),开头、warmup 边界(流水线排空时)和结尾各采 5 / 3 / 5 次(每次 8 对)。每台设备的映射 = 稳健最小二乘直线(3σ 迭代剔除)+ 直线残差的 9 点滑动中位数(按设备时间线性插值);同一设备的 compute 与 transfer 两个计时池共用这一映射。需要后一项是因为两种钟的频率比在几分钟里会漂几个 ppm:3-D 8M(140 s)单用直线时残差是一条 ±16 µs 的抛物线(rms 9.7 / 7.5 µs),加上后 rms 2.1 / 1.3 µs。驱动报的 maxDeviation 最小约 12 µs:每一对标定样本的两个读数之间就有这么宽的窗口,所以 GPU 与主机之间的**绝对**对齐只能保证到这个量级;拟合残差(下面检查里的容差)衡量的是映射本身的一致性。主机时钟 = worker 时间点用的钟:Windows 是 QPC(`time.perf_counter_ns`,本机 QPC 10 MHz,与 perf_counter_ns 一致到 100 ns 一格);Linux 是 CLOCK_MONOTONIC(`perf_counter_ns`),驱动只给 CLOCK_MONOTONIC_RAW 时 tracer 与 worker 都改用 `time.clock_gettime_ns(CLOCK_MONOTONIC_RAW)`——**Linux 两条路径都没测**,E30 的 S1 会报集群上的时间域。

**时间戳为什么每步都在。** 每个按帧的标签在计时池里有两个槽(偶数帧一块、奇数帧一块),第 n 步的 phase A 第一件事是复位本奇偶的块(compute 池直接复位;transfer 池由 phase A 在 compute 队列上复位——transfer 队列不能复位查询池)。第 n 步的块要到第 n + 2 步的 phase A 才会被复位,而 depth-2 主循环在读完第 n 步之后才提交第 n + 2 步;第 n − 2 步的 readback / upload 也一定在第 n 步的 phase A 之前完成(phase A(n) 等 frame_done(n − 1),后者经 upload_done → worker → 本卡 readback_done(n − 1) 排在本卡 readback(n − 1) 之后)。所以只能用 depth ≤ 2(bench 会拒绝 depth 3)。

### steps_device.csv(每步每 sim 一行)

| 字段 | 含义 |
|---|---|
| `step` / `sim` / `device` / `parity` | 步号(0 起)、slab 序号(0 = 最左)、物理设备、步号奇偶(用哪块时间戳槽) |
| `host_read` | tracer 读取这一步的主机时间(所有 sim 的 frame_done(step) 之后) |
| `a_start` | phase A 第一个 tick(复位之后) |
| `a_predict_end` / `a_voxel_end` | predict / update_voxel 结束 |
| `a_ghost_leading_end` / `a_ghost_trailing_end` | 该方向的 ghost_send 结束(没有该侧邻居则空) |
| `a_end` | phase A 最后一个 tick |
| `b_start` | phase B 第一个 tick |
| `b_correction_interior_end` / `b_density_deep_interior_end` / `b_force_deep_interior_end` | phase B 三个 sweep 各自结束 |
| `b_end` | phase B 最后一个 tick |
| `c_start` | phase C 第一个 tick(等过 upload_done 之后) |
| `c_expand_end` / `c_install_{leading,trailing}_end` / `c_append_departed_end` / `c_band_compact_end` / `c_correction_boundary_end` / `c_density_boundary_end` / `c_density_end` / `c_force_end` | phase C 各段结束(配置里没有的段为空;detail = phases 时只有 `c_force_end`,A / B 里同样只留 `a_voxel_end`、两个 ghost_send、`b_density_deep_interior_end`、`b_force_deep_interior_end`) |
| `c_end` | phase C 最后一个 tick |
| `complete` | a/b/c 的 start 与 end 都在 |

tick 都是 BOTTOM_OF_PIPE 的 `vkCmdWriteTimestamp`:标签 = 命令缓冲里它之前的工作全部结束的时刻。

### steps_link.csv(每步每条有向链路一行)

| 字段 | 含义 |
|---|---|
| `step` / `link` / `sender` / `receiver` | 步号、worker 名(`s0_to_s1`)、发送 / 接收 slab |
| `sender_direction` / `receiver_direction` | 发送方送出的一侧 / 接收方收的一侧(s0→s1 是 trailing / leading) |
| `send_end` | 发送方这一侧的 ghost_send 结束(`a_ghost_<侧>_end`) |
| `readback_start` / `readback_copy_end` / `readback_end` | 发送方 transfer 队列:DMA 前、`vkCmdCopyBuffer` 后、主机一致性屏障后 |
| `worker_dequeue` | worker 从队列取到这一步(主循环在提交时通知) |
| `worker_source_wait` | worker 看到发送方 readback_done(n) |
| `worker_dest_guard` | worker 看到接收方 readback_done(n)(时间线安全守卫) |
| `worker_upload_guard` | 接收方第 n − 1 步的 upload 读完 receiver staging |
| `worker_stamp` | 帧戳检查结束 |
| `worker_copy` | 计数字 + count-aware memcpy 结束 |
| `worker_signal` | consumed(给发送方)与 worker_done(给接收方)两次主机 signal 结束 |
| `upload_start` / `upload_end` | 接收方 upload 队列:DMA 前 / 后 |
| `receiver_b_start` / `receiver_b_end` / `receiver_c_start` | 接收方这一步的 phase B 起止与 phase C 开始(取自 steps_device) |
| `host_copy_bytes` | worker 这一步实际拷贝的字节(live 前缀 + 整块的 voxel 表、计数字、帧戳) |
| `dma_bytes` | staging 大小 = readback 字节 = upload 字节(每条链路是常数) |
| `complete` | 上面所有时间点与字节都在 |

### run_meta.json 与 calibration.csv

`run_meta.json`:算例、K、weights、device map、depth、sync scheme、pool safety、步数、warmup、defrag 周期、结果(fps / steady fps);分区(每 slab 的全局 own 列范围、初始 own 粒子数、两侧邻居);commit 与 `experiment/v6` 的未提交文件;时钟(时间域、主机时钟函数、QPC 频率、采样间隔、每台设备的拟合:斜率、截距、样本数 / 保留数、残差 rms / 最大、驱动 maxDeviation 最小 / 中位);**slabs**(第一个 ≥ warmup 的 defrag 边界、流水线排空时从 voxel 计数读一次:每 slab 的 own 列数 w、band 宽度 (c, d, f)、两侧邻居、own 粒子数、n_B = force sweep 覆盖的粒子数(own 列里离 seam 不到 f 列的不算)、correction / density interior sweep 的粒子数、每列粒子数);链路(名字、两端、方向、DMA 字节);两个计时池的标签;记录行数与完整行数;设了的 `V6_*`。`calibration.csv`:每个保留的标定对(sim、步号、设备 ns、主机 ns、驱动 maxDeviation)。

### 派生量(`experiment/v6/analysis/step_trace_model.py`)

- t_tr = `upload_end − send_end`,无重叠地拆成 11 段:readback 启动(`send_end → readback_start`)、readback DMA、屏障、readback 结束 → worker 看到(`readback_end → worker_source_wait`)、等接收方 readback_done、等接收方上一步 upload、帧戳检查、memcpy、signal、signal → upload 开始、upload DMA。按任务的五段合并:readback = 前三段;readback 结束 → worker 看到;host 段 = 中间四段(两段等待 + 帧戳 + memcpy);拷贝结束 → upload 开始 = signal + signal → upload 开始;upload = upload DMA。
- host 步(worker,补完 E24):等发送方 readback(`dequeue → source_wait`,与 GPU 段重叠,不在 t_tr 的关键路径上)、等接收方 readback_done、等接收方上一步 upload、帧戳检查、memcpy、signal。
- T_B = 接收方 `b_end − b_start`;r = t_tr / T_B,逐步取较差的链路。
- 相位差 Δ' = 接收方 `b_start` − 发送方 `send_end`。
- 暴露时间 = max(0, `upload_end` − 接收方 `b_end`);C 前总等待 = 接收方 `c_start − b_end`。
- 主循环门控 = 每 sim 的 `a_start(n) − c_end(n − 1)`。

## 第一段的四项检查

检查运行:2-D 1M = 开销 campaign(最终版仪器)里的 `2d_1m__on__t1`;2-D 16M、3-D 8M = 最终版仪器的单独运行(`checks_final/`)。都是 K = 2、GPU 0,1、drift 0、溢出 0、帧戳错误 0。

### a. 因果顺序

每步每条链路检查 `send_end ≤ readback_start ≤ readback_end ≤ worker_source_wait ≤ worker_copy ≤ upload_start ≤ upload_end ≤ 接收方 c_start`(steady 的 2000 步 × 2 条链路 = 4000 条链)。同一块 GPU 上的两点(compute 与 transfer 池共用设备时钟和同一映射)容差为 0;GPU 与主机之间的两点容差为该设备映射的最大残差。

| 算例 | 违反 | 任何一对出现负间隔 | readback_end → worker 看到,最小 µs | 拷贝结束 → upload 开始,最小 µs | 其余各对最小 µs | 映射残差 rms / 最大 ns(s0;s1) | 驱动 maxDeviation 最小 ns |
|---|---|---|---|---|---|---|---|
| 2-D 1M | 0 | 0 | 9.7 | 23.3 | 19.5–35.1 | 114 / 275;68 / 116 | 12,416;12,288 |
| 2-D 16M | 0 | 0 | 9.4 | 33.6 | 26.9–126.8 | 1,762 / 5,419;1,774 / 6,083 | 12,384;12,224 |
| 3-D 8M | 0 | 0 | 17.5 | 37.5 | 26.4–1,263 | 2,149 / 18,901;1,256 / 6,851 | 12,544;12,320 |

**违反比例 0**(目标 0)。跨时钟域的两对最小间隔(9.4–17.5 µs、23–38 µs)都大于映射残差;它们与驱动的配对窗口(≈ 12 µs)同量级,这一项的"0"只说明映射在这个精度内自洽。

### b. 覆盖率

三个检查运行、开销 campaign 的 6 次开仪器运行与开发运行:**每一步都有完整记录(3000 / 3000,100 %)**,目标 ≥ 95 %。

### c. 与 E24 的一致性(DMA 时长中位数,µs)

E24 = depth-1 解剖帧(构建 21ce20d、band 2/2/3),这里 = 生产 depth 2 的每一步。

| 算例 | 链路 | readback 这里 | readback E24 | 差 | upload 这里 | upload E24 | 差 |
|---|---|---|---|---|---|---|---|
| 2-D 1M | s0 → s1 | 19.5 | 19.2 | +1.6 % | 31.2 | 30.5 | +2.3 % |
| 2-D 1M | s1 → s0 | 18.9 | 19.0 | −0.5 % | 30.5 | 28.0 | +8.9 % |
| 2-D 16M | s0 → s1 | 67.1 | 70.1 | −4.3 % | 97.3 | 102.1 | −4.7 % |
| 2-D 16M | s1 → s0 | 66.3 | 70.9 | −6.5 % | 101.8 | 99.7 | +2.1 % |
| 3-D 8M | s0 → s1 | 572.4 | 908.5 | **−37.0 %** | 1,039.8 | 1,002.5 | +3.7 % |
| 3-D 8M | s1 → s0 | 570.6 | 911.4 | **−37.4 %** | 1,034.4 | 992.8 | +4.2 % |

2-D 两边差 −6.5 … +8.9 %。3-D 8M 的 readback 在生产模式下快 37 %(每步 15,815 KiB:572 µs = 28.3 GB/s,E24 的 908 µs = 17.8 GB/s)。原因在相位:depth-1 的解剖帧里两卡锁步,两条 readback 同时往主机写;depth 2 下两卡错开了 T_B 的很大一部分(见下),两条 readback 的时间重叠只占 0.9 %(2-D 1M)、0.0 %(2-D 16M)、0.1 %(3-D 8M)。2-D 16M 的 readback 每步 1,677 KiB,两种情况下都是 24.5–25.6 GB/s。

### d. 开销

仪器开与关交替 3 对(试验 1 先关、试验 2 先开、试验 3 先关),每次一个进程,K = 2,3000 步、warmup 1000。仪器改了两版,每版都重测:

| 版本 | 改动 | 2-D 1M:关 / 开 fps,开 ÷ 关 − 1 | 平均 ± 标准差 | 2-D 16M:开 ÷ 关 − 1 | 平均 ± 标准差 |
|---|---|---|---|---|---|
| v1 | python-vulkan 读查询池;主循环里每 100 步标定一次;全部 tick | 772.1 / 767.0,−0.66 %;773.3 / 757.4,−2.06 %;772.7 / 764.2,−1.10 % | −1.27 ± 0.71 % | −0.45 %、−0.15 %、0.00 % | −0.20 ± 0.23 % |
| v2 | 原始 cffi 读查询池;标定挪到辅助线程(每 20 ms);全部 tick | 754.7 / 749.9,−0.64 %;762.5 / 754.8,−1.01 %;758.5 / 745.0,−1.78 % | −1.14 ± 0.58 % | −0.46 %、−1.36 %、+0.15 % | −0.56 ± 0.76 % |
| v3(最终) | 标定每 500 ms;默认只写各 phase 的起止 tick | 754.4 / 742.4,−1.59 %;749.2 / 736.6,−1.68 %;755.5 / 747.4,−1.07 % | **−1.45 ± 0.33 %** | +1.52 %、+0.30 %、−1.20 % | +0.21 ± 1.37 % |

**结论:2-D 1M 的开销可以分辨,三个版本都超过 1 %;2-D 16M 分辨不出来**(三版的平均 −0.20 / −0.56 / +0.21 %,试验间散布 0.2–1.4 %)。另外,同一份代码"关"的 fps 在三个时间段是 772–773、755–763、749–756:本机的基线在几十分钟里漂了约 3 %,所以只有同一 campaign 里相邻的开 / 关配对可以比。

**开销在哪(2-D 1M,v2 仪器,各 3 次轮换)。** 只挂计时器、按奇偶录命令但不读不标定("GPU 侧"):

| 配置 | 相对关的 fps 差(3 次) | 平均 |
|---|---|---|
| 只复位查询池、不写 tick | −1.30 %、−0.58 %、+0.35 % | −0.51 % |
| 只写各 phase 起止的 tick | −0.96 %、−0.61 %、+0.01 % | −0.52 % |
| 写全部 tick | −0.79 %、−1.58 %、+0.42 % | −0.65 % |
| 完整仪器(只写 phase 起止) | −1.94 %、−1.62 %、−0.40 % | −1.32 % |

GPU 侧(每步两次查询池复位 + 约 13 个时间戳)约 0.5 %,tick 多少差别在噪声内;其余约 0.8 % 在主机侧。主机侧各项(同一运行里计时):每步读取 hook 平均 9.1 µs(p50 7.5 µs,四次原始读取各约 1.3–2.3 µs);warmup 边界的 voxel 快照一次 3.3 ms(在 steady 窗口内,1M 的窗口 2.6 s,即 0.13 %);v2 的标定线程每 20 ms 一次连采,中位 154 µs、累计 218 ms——v3 改成 500 ms 之后 1M 的开销没有变小(−1.45 ± 0.33 %),所以它不是主要来源。worker 各段的中位数开与关相同(拷贝 52–56 µs、signal 15–17 µs、上一步 upload 等待 6–8 µs),主机侧的成本没有表现为 worker 变慢;更细的定位没有做。

## 检查运行里已经看到的(第二段要用)

| 算例 | T_B p50 µs(s0 / s1) | s1 − s0 的 send_end 差 p50 | t_tr p50 / p95 µs,s0 → s1 | t_tr p50 / p95 µs,s1 → s0 | r = t_tr / T_B(逐步较差的链路)p50 / p95 |
|---|---|---|---|---|---|
| 2-D 1M | 878 / 898 | −364 µs(≈ 0.41 T_B) | 268 / 759 | 672 / 1,384 | 0.80 / 1.56 |
| 2-D 16M | 12,810 / 13,036 | −12.1 ms(≈ 0.94 T_B) | 521 / 1,017 | 12,630 / 14,939 | 0.97 / 1.14 |
| 3-D 8M | 37,969 / 39,134 | +33.8 ms(≈ 0.88 T_B) | 37,593 / 41,238 | 3,706 / 5,619 | 0.96 / 1.03 |

- **生产 depth 2 下两卡不同相**:错开 0.4–0.94 个 T_B。
- **落后的接收方那条链路,t_tr 几乎等于相位差**:worker 要等接收方自己的 readback_done(n)(时间线守卫)才能拷贝,所以按任务的定义(从发送方 ghost_send 结束算)t_tr 里含着 Δ'。r 因此在 0.8–0.97(p50),但这本身不等于暴露:暴露要看 upload 结束是否晚于接收方 B 结束(第二段的分析脚本已经按任务定义算 Δ'、暴露时间与 C 前等待)。旧的 r 假设两卡同时进入 phase B,正好把这一项混进去了。
- readback 互不重叠是同一件事的另一面(上面检查 c)。

## 需要你决定

1. **接受 2-D 1M 约 1.1–1.5 % 的仪器开销**(16M 起分辨不出),照原计划跑第二段(K = 1 与 K = 2 都开仪器)。
2. **再压开销**:每台设备合并成一个查询池(每步 2 次读取而不是 4 次)、warmup 边界的标定连采移出计时窗口、voxel 快照改成异步拷贝到结束后再读。GPU 侧约 0.5 % 的底(复位 + 时间戳)去不掉,能否压到 1 % 以下不确定。
3. **η 用不开仪器的运行,分解用开仪器的运行**:每个算例的 K = 2 多跑一次不开仪器的,η 只用不开的 fps;机时约多一半(扫描约 20 → 30 分钟)。

我倾向 3(η 不受仪器影响,分解与 r 用开仪器的运行),可以加上 2 里便宜的两项。

## 文件、复现与数据

| 文件 | 内容 |
|---|---|
| `experiment/v6/utils/phase_trace_v6.py` | `StepTraceTimer`(按帧奇偶分块的计时器)、`StepTracer`(逐步读取、时钟标定与映射、n_B 快照、写出) |
| `experiment/v6/_run_v6_chain_bench.py` | `--step-trace DIR`、`--step-trace-detail`、`--step-trace-calibrate-ms` |
| `experiment/v6/utils/simulator_v6.py` | `step_trace_parity`:打开时 phase A / B 与传输命令按帧奇偶各录一份,提交时按步号奇偶选(快速与普通两条提交路径);关闭时命令与提交不变 |
| `experiment/v6/utils/transport_v6.py` | worker 时间点的主机时钟可替换(`set_host_clock`,Linux RAW 用)、帧戳检查之后多一个时间点 `stamp_ns`、每步 `copy_bytes`;关闭时 worker 每步多一次取时与一个字典项,GPU 命令不变 |
| `experiment/v6/analysis/step_trace_model.py` | 检查(因果、覆盖率、DMA)、逐步派生量、常数拟合、r_pred 与闭式、表与 4 张图;`--run DIR` 打印单次运行的汇总 |
| `experiment/v6/analysis/step_trace_campaign.py` | `--mode overhead`(开销)/ `--mode scan`(第二段:9 个算例,K = 1 两卡同时 + K = 2) |

```bash
# 一次带仪器的运行与它的检查
.venv/Scripts/python.exe experiment/v6/_run_v6_chain_bench.py --case cases/lid_driven_cavity_2d_gen/case.yaml     --weights 1,1 --device-map 0,1 --max-steps 3000 --warmup 1000 --step-trace logs/e29_step_trace/<dir>
.venv/Scripts/python.exe -m experiment.v6.analysis.step_trace_model --run logs/e29_step_trace/<dir>
# 开销(2-D 1M 与 2-D 16M,开 / 关交替 3 对)
.venv/Scripts/python.exe -m experiment.v6.analysis.step_trace_campaign --mode overhead --out logs/e29_step_trace/overhead
# 第二段(没有跑):扫描与模型
.venv/Scripts/python.exe -m experiment.v6.analysis.step_trace_campaign --mode scan --out logs/e29_step_trace/scan
.venv/Scripts/python.exe -m experiment.v6.analysis.step_trace_model --scan logs/e29_step_trace/scan     --overhead logs/e29_step_trace/overhead --out docs/perf_model/e29
```

第二段要用的 62k 与 250k 算例已用 1M 的同一生成器与参数生成(`utils/geometry/_demo_cavity_case.py --half 124 / 250`,h/Δx = 5、C = 96 / 16、ξ = 0.001、ε² = 0.0025 h²;`cases/cavity_2d_62k`(73,441 个粒子)、`cases/cavity_2d_250k`(273,529 个),未入库)。

数据在 `logs/e29_step_trace/`(不入库):`overhead/`(v3,最终)、`overhead_v2_calibration_20ms/`、`overhead_v1_inline_calibration/`、`overhead_split2/`(开销分解)、`checks_final/`(2-D 16M、3-D 8M)、`dev/`(开发运行与主机侧计时)、`overhead_summary.json`、`phase1_checks.json`。`overhead_aborted_gpucheck/` 是第一次开销 campaign:GPU 空闲检查把带桌面的 GPU 0 的利用率当成"忙",卡住后停掉,只有 1 次有效运行,未使用。
