# E30：v6 集群冒烟（N56 小规模冒烟 + N32-H A100 移植冒烟）

- **N56（8 × RTX 5090 / 节点，驱动 580.82.07）**：部署 `347be6f`（E6b 之后的 HEAD，发布组合是代码默认）。作业 1665986（K = 2）、1666242（K = 3）、1665987（K = 4），2026-10-05/06。
- **N32-H（4 × A100-PCIE-40GB / 节点，Kunpeng-920 aarch64，驱动 535.104.12）**：部署 `c411b11`（E32 HEAD：phase A no-wait 默认、nearest-column 切点、`--weights auto / --weights-file`）。step trace 那一次用的是 `a6bce2b` = `c411b11` + 本次修复（见第 6 节）。作业 1550014（失败）、1550133 / 1550137 / 1550139（诊断）、1550151（绕过验证）、1550154（计划）、1550194（step trace + P2P），2026-10-06；合计约 1.6 GPU·h。
- 本报告与脚本的提交见文末。

这不是测量 campaign：N56 部分不算 η，不做参照，不做多次试验，数据不进论文（E30 的约定）。A100 那一节按用户补充的计划给出单次的 η / η_min，只作示意。

## 0. 结论

1. **N56 上没有发现会导致返工的硬问题，但 E30 的主体没有跑。** 已经跑的是小规模冒烟：K = 2/3/4 的 2-D 16M 与 3-D 8M（1500 步，warmup 500，含一次 defrag），以及 bring-up（1M 的 K = 1 / K = 2）和一次 1M K = 2 的 step trace。全部跑完；drift 0；所有 overflow 计数 0；far_migration 0；帧戳错误 0；init clamp 0；无 NaN；seam 检查通过。K = 3/4 里第一次出现了有两条 seam 的内部 slab（s1；s1、s2），也都通过。
   E30 要求的 8 卡部分（S2 的 K = 8，S3 的 128M、3-D 64M K = 8 与 K = 1、cube 401³，S4 的 64M K = 1 / K = 8 + anatomy）**没有跑**：`hp_5090` 不对本账号开放（`AllowAccounts=hp5090`），`gpu_5090` 拿不到整节点。所以"14 条链路、14 个 host 线程同时工作"和"最大算例装不装得下"这两个问题仍然没有答案。
2. **A100（aarch64）上，v6 原样运行会在第一次 `vkCmdBindDescriptorSets` 时 SIGBUS。** 原因在平台，不在 v6：NVIDIA 驱动（libnvidia-eglcore）用 glibc 的 `memcpy` 往一段 `/dev/nvidia0`（BAR1）映射里拷 192 字节。这段映射在这台机器上是 Device 内存。glibc 2.28 的 aarch64 `memcpy` 只对齐源地址，于是在 8 字节对齐（但不是 16 字节对齐）的目标地址上发出 16 字节的存储，触发 `si_code = BUS_ADRALN`。用一个 LD_PRELOAD 垫片绕开：它只改写落在 `/dev/nvidiaN` 映射里的 `memcpy` / `memmove` / `memset`。这样的调用每个 sim 只有 4 次、每次 192 字节，都在命令缓冲录制阶段；其余拷贝仍走 glibc。v6 代码不改。
3. **step trace（E29 仪器）在驱动 535 上建不了设备**：这版驱动只有 `VK_EXT_calibrated_timestamps`，没有 KHR 版。修复 `a6bce2b`：KHR 不在时改用 EXT。单独提交，先在本机做了 K = 2 冒烟（第 6 节）。
4. **E29 的 Linux 时钟方案能用。** 两个驱动都提供 `CLOCK_MONOTONIC` 和 `CLOCK_MONOTONIC_RAW`。
   - N56（KHR，`CLOCK_MONOTONIC`，1M K = 2）：映射残差 rms 85–92 ns，覆盖率 600 / 600，800 条链的因果违反 0。
   - A100（EXT，`CLOCK_MONOTONIC_RAW`，2-D 16M K = 4）：残差 rms 16–115 ns，覆盖率 2000 / 2000，9000 条链的因果违反 0；t_chain 中位数 0.5–1.1 ms，只占 phase B 的 3–6 %。
5. **A100 的数字**（等权重，单次，只作示意）：
   - 2-D 16M：K = 1 每卡 12.72–12.89 fps；K = 2 24.75 fps（η 96.4 %，η_min 96.7 %）；K = 4 47.41 fps（η 92.7 %，η_min 93.2 %）。
   - 3-D 8M：K = 1 4.26–4.31 fps；K = 2 8.31 fps（η 96.9 %）；K = 4 15.81 fps（η 92.4 %）。
   - 2-D 16M K = 4 的权重校准：等权重时端部与中间 slab 的忙时比为 1.0023；ω = 0.9932 / 0.9979 / 1.0036 / 1.0052，切点 203 / 403 / 603 → 201 / 401 / 602；fps 比等权重低 0.28 %，在噪声内。
   - P2P：没有 Vulkan device group；OPAQUE_FD 导入被接受，但读到的全是 0，没有共享。

## 1. 版本与 SPIR-V

| 平台 | 部署的提交 | 部署方式 |
|---|---|---|
| N56 | `347be6fd38a091390b73d8aa6ea57be47e59cfe6` | `git archive` 子集解到 `~/run/vulkan-demo-v6`（旧 checkout 不动）；`COMMIT` + `MANIFEST.sha256`（90 个文件）+ `SCRIPTS.sha256` 在每个作业开头校验；粒子文件（`.obj`）用符号链接复用已有算例（`setup_cases_v6.sh`） |
| N32-H | `c411b1171118a6a800b01c25adf565e914675c71`；step trace 作业用 `a6bce2b734b746e060f157046553b039079e91de` | 同样方式解到 `~/e30/vulkan-demo-v6`；`.obj` 在 N32-H 上用同一个生成器重新生成（1M 算例的 `.obj` sha256 与 N56 相同） |

着色器在 `347be6f..a6bce2b` 之间没有任何改动（`git diff --stat 347be6f a6bce2b -- experiment/v6/shaders` 为空）。

**SPIR-V。** 两个集群上都没有 `glslc` / `glslangValidator`（N32-H 的 `module avail` 里也没有），所以 E30 要求的"在集群上重编"做不到。改为：在 `a6bce2b` 上用 `experiment/v6/compile_shaders_v6.py`（shaderc v2026.2，SDK 1.4.350.0，`--target-env=vulkan1.2 -O`）把全部着色器重编到一个干净目录，再与仓库里的 `.spv` 比较。结果 13 / 13 逐字节相同。另外两个 render 着色器（`particle.vert/.frag`）不入库，无头求解器也不用。集群上运行的就是仓库里的这 13 个文件；N56 bring-up 报告逐个打印的 sha256 也与下表一致。

| .spv | sha256 |
|---|---|
| `bootstrap_half_kick.comp.spv` | `ec24be0f2d0b48f85b35d4177190b1ca5d6da008114f43798c1db18d98a0aabe` |
| `initialize_voxelization.comp.spv` | `62f42d33a26a70871521e30aa448dd3f620342e912120cc2e3d3bc4a6e3bdafa` |
| `predict.comp.spv` | `099b129f66ea93c40d2a3751a1aa809d74c4f5b6c0f0759a9f7fb9f5c725442b` |
| `update_voxel.comp.spv` | `024e04287997e22a6e37212b18f161b19a093571918d15337a2a69f4f2ba25ee` |
| `ghost_send.comp.spv` | `de0feb09c6618201e572ebc39b5982f8577d6e87b7131090faca73f4fae40cd5` |
| `install_migrations.comp.spv` | `d7f17d7808d77f9e885d7fe6c6a5ccfc49495dccb887e68369790f3a94360e64` |
| `correction.comp.spv` | `ee9ff135900b69f881c6fc9ad0dfc5c87ed14de015497e8060dceae90ebdabc7` |
| `density.comp.spv` | `ba4d4657c15d304ae5389cd908224a128317f41ab85f9af79c3bc57142a469fc` |
| `force.comp.spv` | `4614f26d6c3da85de415ce488fc989551abff8b74fe710cb50ba2219edb0bf75` |
| `defrag.comp.spv` | `52d7d9b3a09cf4ce33e8f2981c6e699b0ff213beabf1935e46e1ff8ad7969c42` |
| `append_departed.comp.spv` | `7823e926e76ae19ac340ee12864e1e8e4ff6e0a08fe68b3298edd54a0b08d870` |
| `expand_ghost_lists.comp.spv` | `728d4e4742a0f156bac518fae4243b177d130bdf5696858357a298412b02c251` |
| `band_compact.comp.spv` | `e22c63cb3b498b7ac784a62cd0ad10af0a33f85db222dba9165448e2d9a8f25a` |

## 2. 主机侧设置（两个集群相同）

- 作业开头清掉继承来的 `V5_*` / `V6_*`，只导出 `V6_WORKER_AFFINITY`，按 probe38 的方式从 sysfs 生成。每卡一项，取该卡所在 NUMA 节点的 cpulist：N56 的 K = 2 作业是 `0-31,64-95;32-63,96-127`，K = 4 是 `0-31,64-95;0-31,64-95;32-63,96-127;32-63,96-127`；N32-H 是 `0-31;0-31;64-95;64-95`（GPU 0、1 在 node 0，GPU 2、3 在 node 2）。日志里的 `env | grep ^V6_` 只有这一行；程序自己报告的有效配置（`[e30-config]`）与 E6b 之后的发布组合一致：ghost_layers 2、keep_departed、lean、compact lists、packed replicas、init seam clamp、band 2,2,3、lanes 64、cascade force、fast submit；E32 之后另有 phase A no-wait。
- GIL switch interval 0.2 ms，在 `run_chain_v6.py` 里设置，每次运行都打印 `switchinterval_s=0.0002`。
- 每台设备两个 transfer 队列（readback / upload 分开）。每次运行都打印 `transfer queues: 2`，K 卡运行的 `tq2 = K`。worker 绑核 2(K−1) 个，跳过 0 个。
- 解析：`parse_run_v6.py` 从每次运行的日志里取 S2 的全部记录项；`nvidia-smi` 每秒采一次显存、利用率、功率、温度和 SM 时钟。
- N32-H 另加 `LD_PRELOAD=devmem_safe_copy.so`（第 5.2 节）。这不是 `V6_*` 开关，也不改 v6 的任何行为，只绕开驱动的对齐问题。

## 3. 能力表

N56（每张卡都相同的部分合并成一行；作业 1665986 / 1666242 / 1665987 一共 9 张卡，`caps_v6.py` 逐卡记录在 `docs/cluster_v6/data/n56_caps_*.json`）：

| 项 | N56 RTX 5090 | N32-H A100-PCIE-40GB |
|---|---|---|
| 驱动 / Vulkan API | 580.82.07 / 1.4.312 | 535.104.12 / 1.3.242 |
| 显存（device-local heap） | 31.84 GiB | 40 GiB（见 5.1） |
| 队列族（flags × queueCount，timestampValidBits） | 0: graphics+compute+transfer+sparse × 16, 64；**1: transfer+sparse × 2, 64**；2: compute+transfer+sparse × 8, 64；3: transfer+video_decode × 2, 32；4: transfer+video_encode × 2, 32；5: transfer+optical_flow × 1, 64 | 见 5.1 |
| v6 选的 transfer 族 | family 1（transfer-only，2 个队列 → 两个 transfer 队列分开可行） | family 1 × 2（同） |
| timestampPeriod | 1 ns | 1 ns |
| calibrated timestamps | KHR 与 EXT 都有；域：DEVICE、CLOCK_MONOTONIC、CLOCK_MONOTONIC_RAW | **只有 EXT**；域：DEVICE、CLOCK_MONOTONIC、CLOCK_MONOTONIC_RAW |
| 实测校准（每卡 5 次，括在主机读数之间） | 5 / 5 落在括号内；驱动 maxDeviation 1.76–2.0 µs（Windows 5090 上 ≥ 12 µs） | `caps_v6.py --live-sample` 只做 KHR，A100 上跳过；见 5.5 |
| NUMA 与 cpulist | GPU 0x27/0x38/0x5a → node 0（`0-31,64-95`）；0xa8/0xb8/0xd8 → node 1（`32-63,96-127`） | GPU 0,1 → node 0（`0-31`）；GPU 2,3 → node 2（`64-95`） |
| 主机时钟源 | tsc | arch_sys_counter |

`nvidia-smi` 的序号与 v6 的设备序号在每个作业里都一致（`v6_index == nvidia_smi_index` 断言通过）。表里 PCIe 链路速度的读数有 32 GT/s 和 2.5 GT/s 两种，后者是卡空闲时降速的状态，不是协商能力。

## 4. N56 运行表

每次运行一行，字段同 S2（显存峰值是运行窗口内 `nvidia-smi` 采样的最大值，GiB）：

| job | run | case | K | steps / warmup | steady fps | drift | overflow | far_migration | stamp errors GPU+host | alive start → end | init clamp | NaN | wall s | VRAM peak GiB (card:value) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1665986 | bringup_k1 | lid_driven_cavity_2d_n1000 | 1 | 300 / 100 | 555.1 | 0 | 0 | 0 | 0+0 | 1,046,529 → 1,046,529 | 0 | no | 13.9 | – | pass |
| 1665986 | bringup_k2 | lid_driven_cavity_2d_n1000 | 2 | 300 / 100 | 723.4 | 0 | 0 | 0 | 0+0 | 1,046,529 → 1,046,529 | 0 | no | 23.1 | – | pass |
| 1665986 | k2_2d16m | lid_driven_cavity_2d_16m | 2 | 1500 / 500 | 72.8 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 70.1 | 0:2.8 / 1:2.8 | pass |
| 1665986 | k2_3d8m | cavity3d_weak4_k2_8m_b4 | 2 | 1500 / 500 | 25.3 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 106.0 | 0:1.7 / 1:1.7 | pass |
| 1665986 | k2_1m_steptrace | lid_driven_cavity_2d_n1000 | 2 | 600 / 200 | 801.2 | 0 | 0 | 0 | 0+0 | 1,046,529 → 1,046,529 | 0 | no | 4.5 | 0:0.2 / 1:0.2 | pass |
| 1666242 | k3_2d16m | lid_driven_cavity_2d_16m | 3 | 1500 / 500 | 105.6 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 75.6 | 0:1.9 / 1:1.9 / 2:1.9 | pass |
| 1666242 | k3_3d8m | cavity3d_weak4_k2_8m_b4 | 3 | 1500 / 500 | 36.5 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 89.4 | 0:1.2 / 1:1.2 / 2:1.2 | pass |
| 1666242 | k3_2d16m_anatomy | lid_driven_cavity_2d_16m | 3 | 1500 / 500 | 104.8 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 48.3 | 0:1.9 / 1:1.9 / 2:1.9 | pass |
| 1665987 | k4_2d16m | lid_driven_cavity_2d_16m | 4 | 1500 / 500 | 134.5 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 87.9 | 0:1.4 / 1:1.4 / 2:1.4 / 3:1.4 | pass |
| 1665987 | k4_3d8m | cavity3d_weak4_k2_8m_b4 | 4 | 1500 / 500 | 47.9 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 89.4 | 0:0.9 / 1:1.0 / 2:1.0 / 3:0.9 | pass |
| 1665987 | k4_2d16m_anatomy | lid_driven_cavity_2d_16m | 4 | 1500 / 500 | 124.7 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 47.7 | 0:1.4 / 1:1.4 / 2:1.4 / 3:1.4 | pass |

另外：每次运行的 seam 完整性检查都通过（无跨侧 overshoot、无重复粒子）；ρ 在 [998.1, 1001.7] 之内，vmax = 1.0000。bring-up 是 1M 300 步（warmup 100、defrag 周期 100），它的显存不在 telemetry 窗口里。

**传输（只记录，不处理）。** 2-D 16M 每条链路每帧的 host 拷贝为 1.24 MiB（count-aware），DMA 1.64 MiB；host 拷贝 p50 为 102–199 µs，DMA 为 45–72 µs。K = 3 / 4 的 anatomy 只取一帧（f1000，正好是 defrag 帧），phase B 为 5.9–8.1 ms。K = 3 三个 sim 的 b→c 间隙都约 4 µs，传输完全藏住。K = 4 的 s1、s3 在这一帧的 b→c 间隙为 3.75 / 4.04 ms，是一次暴露的等待。不过 K = 4 稳态 134.5 fps（7.4 ms/帧）与 A + B + C ≈ 7.0 ms 只差 0.4 ms，所以这一帧不代表稳态。K = 3 有一次 `signal` 最大值 49 ms，属于单次离群。这两件事都只记下来，不处理（E30："性能上的意外只记录"）。

## 5. N32-H A100（移植冒烟）

### 5.1 平台与环境

- 节点：4 × A100-PCIE-40GB（BAR1 64 GiB，`EnableResizableBar=0`），PCIe Gen4 x16；CPU 是 Kunpeng-920（2 × 64 核，4 个 NUMA 节点，页大小 64 KiB）；Kylin V10，内核 4.19.90 aarch64，glibc 2.28；驱动 535.104.12，Vulkan 1.3.242；时钟源 `arch_sys_counter`。GPU 0、1 在 NUMA node 0（cpus 0-31），GPU 2、3 在 node 2（cpus 64-95）。
- 软件环境与 N56 一致：Python 3.13.15、python-vulkan 1.3.275.1、Vulkan loader 1.4.357（为 aarch64 重新编译），numpy 等包版本相同（`~/e30/env`）。算例用同一个生成器在 N32-H 上重新生成，1M 算例的 `.obj` sha256 与 N56 相同。
- 能力（四张卡相同）：队列族 0: graphics+compute+transfer+sparse × 16（时间戳 64 位）；1: transfer+sparse × 2（64）；2: compute+transfer+sparse × 8（64）；3: transfer+video_decode × 5（32）。v6 选 family 1，可以开两个 transfer 队列（每次运行 `tq2 = K`）。timestampPeriod 1 ns。device-local 显存 40.0 GiB。calibrated timestamps：**没有 KHR 版**；EXT 版提供 DEVICE、CLOCK_MONOTONIC、CLOCK_MONOTONIC_RAW。`nvidia-smi` 序号与 v6 设备序号一致。

### 5.2 SIGBUS：诊断与绕过

| 作业 | 内容 | 结果 |
|---|---|---|
| 1550014（4 卡，`c411b11` 原样） | A100 计划 | s0 环境、s1 能力通过；5 次 chain 运行都在 `[SimV6] uploaded initial state` 之后以 rc 135（SIGBUS）退出；按规则停在第一个硬失败的 K = 2 运行 |
| 1550133（1 卡，`python -X faulthandler`） | 1M K = 1 | Python 栈：`vkCmdBindDescriptorSets` ← `_bind_pipeline_and_sets` ← `_record_bootstrap_init_cmd` ← `bootstrap_init`，即第一次录制带描述符绑定的命令缓冲 |
| 1550137 / 1550139（1 卡，gdb） | 同上 | frame #0 是 libc 的 `memcpy`，#1–#2 在 `libnvidia-eglcore.so.535.104.12`。`si_signo = 7`（SIGBUS），`si_code = 1`（**BUS_ADRALN**，对齐错误），`si_addr = 0x40005e830008`，落在一段 2 MiB 的 `/dev/nvidia0` 映射里（GPU 内存经 BAR1 映射到用户态）。故障指令 `str q0, [x3, #16]`（16 字节 SIMD 存储） |
| 1550151（2 卡） | 1M，加垫片 / 只换 glibc 的 memcpy 变体 | 都通过（见运行表） |

机理：目标 `x0 = 0x40005e830000` 是 16 字节对齐的，源 `x1` 在进程堆里，相对 16 字节错开 8。glibc 2.28 在这台机器上选的 `memcpy` 变体先把**源**对齐到 16，目标就跟着错开 8，于是 16 字节存储落在 8 字节对齐的地址上。普通内存允许这样的非对齐访问，Device 类型的内存不允许，所以报 BUS_ADRALN。也就是说，这个驱动和内核的组合把 BAR 映射成了 Device 类型，驱动自己却又用 glibc 的 `memcpy` 往里写。问题出在驱动和平台之间，v6 只是第一个走到这条路径的调用者。

绕过：`docs/cluster_v6/scripts/devmem_safe_copy.c`，一个 LD_PRELOAD 垫片，在登录节点上用 gcc 7.3 编译。它记录每次对 NVIDIA GPU 设备节点的 `mmap`（主设备号 195、次设备号 < 254，即 `/dev/nvidia0..N`，不含 `nvidiactl`）。`memcpy`、`memmove`、`memset` 只要碰到这些区间，就改用天然对齐的 1、4、8 字节访问；其余调用原样交给 glibc。

- 正确性：穷举单元测试 0 失败。覆盖源偏移和目标偏移各 0–15、长度 0–300，分别测正向拷贝、反向拷贝、填充和重叠移动，共 308,224 例。编译产物里没有 q 寄存器存储，也没有 DC ZVA。
- 开销：每个 sim 恰好有 4 次 `memcpy` 走对齐路径，每次 192 字节，都发生在命令缓冲录制阶段（K = 4 时一共 16 次、3 KiB）。步进循环里一次都没有，`memset` 和 `memmove` 也从未落到设备映射上。所以它不影响性能，这一节的 fps 就是平台本身的 fps。
- 替代办法：只设 `GLIBC_TUNABLES=glibc.cpu.name=generic`，让 glibc 换用对齐**目标**的通用 `memcpy`，1M K = 1 也能通过（60 步，185.4 fps）。但它改的是整个进程的 `memcpy`，而且只验证了这一次，所以全计划用的是垫片。

### 5.3 扩展效率（等权重；单次运行，只作示意）

K = 1 在四张卡上同时跑（参照与 K 卡运行处在同一功率和温度状态）；K = 2 用 GPU 0、1（同在 NUMA node 0）；K = 4 用 0–3（跨两个 NUMA 节点）。η = fps_K /（K × 参与卡 K = 1 的均值），η_min = fps_K /（K × 参与卡 K = 1 的最小值）。fps 取自日志 STEADY 行的"步数 / 秒数"。每卡的 SM 时钟、功率和温度是运行窗口内利用率 ≥ 90 % 的 `nvidia-smi` 样本均值（`a100_summary.py`）。

**2-D 16M（2000 步，warmup 500）**

| 运行 | 卡 | steady fps | η | η_min | SM 时钟 / 功率 / 温度 |
|---|---|---|---|---|---|
| K = 1 | 0 | 12.792 | – | – | 1406 MHz, 245 W, 41 °C |
| K = 1 | 1 | 12.892 | – | – | 1410 MHz, 244 W, 45 °C |
| K = 1 | 2 | 12.758 | – | – | 1395 MHz, 247 W, 47 °C |
| K = 1 | 3 | 12.722 | – | – | 1392 MHz, 247 W, 49 °C |
| K = 2 | 0,1 | 24.752 | 96.4 % | 96.7 % | 1410 / 1410 MHz, 244 / 240 W |
| K = 4 | 0–3 | 47.408 | 92.7 % | 93.2 % | 1393–1404 MHz, 228–246 W |

**3-D 8M（`cavity3d_weak4_k2_8m_b4`，1000 步，warmup 500）**

| 运行 | 卡 | steady fps | η | η_min | SM 时钟 / 功率 / 温度 |
|---|---|---|---|---|---|
| K = 1 | 0 | 4.270 | – | – | 1403 MHz, 242 W, 43 °C |
| K = 1 | 1 | 4.309 | – | – | 1410 MHz, 241 W, 47 °C |
| K = 1 | 2 | 4.273 | – | – | 1398 MHz, 244 W, 49 °C |
| K = 1 | 3 | 4.255 | – | – | 1394 MHz, 242 W, 50 °C |
| K = 2 | 0,1 | 8.313 | 96.9 % | 97.3 % | 1409 / 1410 MHz, 239 / 235 W |
| K = 4 | 0–3 | 15.808 | 92.4 % | 92.9 % | 1410 MHz（四卡），226–241 W |

- 四张卡 K = 1 的差异是 1.3 %（两种算例相同）。卡跑在 A100 PCIe 的上限附近：SM 1392–1410 MHz（boost 上限 1410），功率 226–247 W（上限 250 W）。
- 量级：1M K = 1 时 A100 是 183.6 fps，N56 的 5090 是 555.1 fps（都是 300 步 bring-up），单卡约为 5090 的 1/3。
- 传输：2-D 16M 每条链路每帧 host 拷贝 1.24 MiB、DMA 1.64 MiB；3-D 8M 是 6.84 MiB 和 9.11 MiB。K = 4 时 3-D 的 host 拷贝 p50 为 1.0–1.8 ms。Kunpeng 上单个 worker 的拷贝约 3–8 GB/s，大约是 N56 的一半（N56 上 1.24 MiB 只要 0.10–0.20 ms，6.5–12.7 GB/s）。phase B 约 19 ms（2-D 16M K = 4），足以把它藏住（见 5.5）。

### 5.4 2-D 16M K = 4：先校准权重，再用这份权重跑

`--weights auto --calibrate-only --weights-file`（E32：两轮 pilot，每轮 warmup 200 + 测量 300 帧，用时 72.4 s），然后 `--weights-file` 跑 2000 步。

| 轮 | weights | 切点 | own 列 | 每段流体粒子 | 每卡 busy（µs） | ω | 下一轮切点 |
|---|---|---|---|---|---|---|---|
| 1 | 1, 1, 1, 1 | 203, 403, 603 | 203, 200, 200, 203 | 4,005,001 / 4,001,000 / 4,001,000 / 4,001,000 | 21047.7 / 20911.5 / 20780.4 / 20738.9 | 0.9923, 0.9977, 1.0040, 1.0060 | 201, 401, 602 |
| 2 | 0.9923, 0.9977, 1.0040, 1.0060 | 201, 401, 602 | 201, 200, 201, 204 | 3,964,991 / 4,001,000 / 4,021,005 / 4,021,005 | 20798.6 / 20889.7 / 20874.0 / 20840.4 | 0.9932, 0.9979, 1.0036, 1.0052 | 201, 401, 602（不变） |

- **ω**（最终）= 0.99323, 0.99789, 1.00363, 1.00525；**切点** 201, 401, 602（等权重是 203, 403, 603）。
- **端部与中间 slab 的忙时比**（第 1 轮，等权重）：端部 21047.7 / 20738.9 µs，中间 20911.5 / 20780.4 µs，端部均值 ÷ 中间均值 = **1.0023**。分相看：端部 slab 的 phase B 更长（19.1–19.3 ms 对 18.7–18.8 ms，B 里的深内部粒子更多），中间 slab 有两条 seam，A 和 C 更长（A 0.92 对 0.82 ms，C 1.18 对 0.81–0.92 ms），两者几乎抵消。第 1 轮各卡 busy 最大比最小大 1.5 %，第 2 轮降到 0.44 %。
- **与等权重的 fps 差**：等权重 47.408 fps，校准权重 47.274 fps，**−0.28 %**（各一次）。这在单次运行的噪声里（四张卡 K = 1 的差异是 1.3 %）。2-D 16M 的墙很薄，四张卡又是同一型号，几乎没有可平衡的，所以校准只挪了 1–2 列，与上面 1.0023 的忙时比一致。
- 两次 pilot 的不变量都是 drift 0、overflow 0、帧戳错误 0、far_migration 0；pilot 的 fps 为 47.3 / 47.7。

### 5.5 step trace：A100 上的 Linux 时钟路径（`a6bce2b`，作业 1550194）

2-D 16M K = 4，等权重，2000 步，warmup 500，depth 2，E29 默认细度。开着 trace 是 47.53 fps（1550154 不开 trace 是 47.41，单次）。

- **时间域**：`CLOCK_MONOTONIC_RAW`。四个设备都用 `VK_EXT_calibrated_timestamps`（`run_meta.json` 的 `clock.extensions`）。主机时钟是 `clock_gettime_ns(CLOCK_MONOTONIC_RAW)`，传输 worker 的时间点也用它（`transport_v6.set_host_clock`）。
- **映射残差**（fit v2：直线加 9 点滑动中位数漂移项）：四个设备 rms 60 / 115 / 18 / 16 ns，最大 230 / 457 / 107 / 83 ns；每设备 126 个校准对，保留 78–94 个。只拟合直线时 rms 为 5.5–8.1 µs：45 s 里设备时钟与主机时钟的频率比在漂，要靠漂移项吸收。驱动报告的 maxDeviation 最小 8.0–8.1 µs、中位 8.2–8.5 µs（N56 驱动 580 是 1.76–1.97 µs，Windows 5090 ≥ 12 µs）。
- **覆盖率**：2000 / 2000 步；设备行 8000 / 8000、链路行 12000 / 12000 全部完整。
- **因果违反**：9000 条链、8 对相邻时间点，违反数 **0**（每一对的违反比例都是 0.0）。

t_chain 的三跳分解（p50，µs；三跳各自取 p50，所以加起来不严格等于 t_chain 的 p50）：

| 链路 | t_chain p50 / p95 | readback | host（等待 + 拷贝 + 通知） | upload | host 拷贝 p50 | phase B p50（接收方） | r_chain p50 / p95 | 暴露帧比例 |
|---|---|---|---|---|---|---|---|---|
| s0→s1 | 540 / 716 | 77 | 353 | 98 | 195 | 18,693 | 0.03 / 0.04 | 0 |
| s1→s0 | 504 / 6357 | 192 | 195 | 98 | 172 | 19,165 | 0.03 / 0.33 | 0 |
| s1→s2 | 821 / 6713 | 151 | 437 | 195 | 407 | 18,807 | 0.04 / 0.36 | 0 |
| s2→s1 | 706 / 857 | 232 | 364 | 101 | 243 | 18,693 | 0.04 / 0.05 | 0.54（平均 68 µs） |
| s2→s3 | 1131 / 1607 | 239 | 560 | 192 | 385 | 19,225 | 0.06 / 0.08 | 0.05 |
| s3→s2 | 921 / 1283 | 125 | 438 | 230 | 373 | 18,807 | 0.05 / 0.07 | 0 |

传输链只占 phase B 的 3–6 %（p95 最坏 36 %），已经藏住。55 % 的帧有暴露，但每帧平均只有 70 µs（暴露的帧里平均 128 µs），约占 21 ms 帧时的 0.3 %。暴露几乎全在 s2→s1：内部 slab s1 的 phase B 比端部短 0.5 ms，先到 phase C，要等相邻 slab 的数据。这是 slab 之间的相位差，不是传输慢。

N56 的对照（1M K = 2，`347be6f`，作业 1665986）：KHR，`CLOCK_MONOTONIC`（347be6f 在 Linux 上还是 MONOTONIC 优先，c7445c6 起改成 RAW 优先）；残差 rms 92 / 85 ns、最大 172 / 153 ns；600 / 600 步；800 条链违反 0；t_chain p50 218.5 / 129.2 µs = readback 21.5 / 23.8 + host 153.2 / 63.2 + upload 43.1 / 41.9 µs。

### 5.6 P2P（OPAQUE_FD，可选项）

- `vkEnumeratePhysicalDeviceGroups`：4 组，每组 1 张卡，没有 Vulkan device group（这台节点上没有 NVLink 或对等分组）。
- OPAQUE_FD：GPU 0 导出成功；GPU 1 导入时驱动**没有拒绝**（`vkAllocateMemory` 带 `VkImportMemoryFdInfoKHR` 返回成功），GPU 1 上的拷贝也正常完成；但读回的 1,048,576 字节里**没有一个**是 GPU 0 写进去的图样，全是 0。规范只允许在 deviceUUID 相同的设备之间导入 OPAQUE_FD，驱动没有报错，但导入得到的并不是导出方的内存。结论：这几张 A100 之间没有可用的 Vulkan 显存共享，传输仍然只能经主机中转（与 N56、Windows 5090 相同）。
- 1550154 那次探测在读回那一行失败了：python-vulkan 的 `vkMapMemory` 已经返回 buffer，脚本又套了一层 `ffi.buffer`。修好后在 1550194 重跑，结果如上。探测进程退出时段错误（rc 139）：它没有销毁设备，没有深究。导出方没有做对照读回。

### 5.7 运行表（N32-H）

| job | run | case | K | steps / warmup | steady fps | drift | overflow | far_migration | stamp errors GPU+host | alive start → end | init clamp | NaN | wall s | VRAM peak GiB (card:value) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1550151 | shim_k1_1m | lid_driven_cavity_2d_n1000 | 1 | 300 / 100 | 183.6 | 0 | 0 | 0 | 0+0 | 1,046,529 → 1,046,529 | 0 | no | 8.4 | – | pass |
| 1550151 | tunable_k1_1m（无垫片） | lid_driven_cavity_2d_n1000 | 1 | 60 / 10 | 185.4 | 0 | 0 | 0 | 0+0 | 1,046,529 → 1,046,529 | 0 | no | 2.5 | – | pass |
| 1550151 | shim_k2_1m | lid_driven_cavity_2d_n1000 | 2 | 300 / 100 | 290.9 | 0 | 0 | 0 | 0+0 | 1,046,529 → 1,046,529 | 0 | no | 5.9 | – | pass |
| 1550154 | k1_2d16m_g0 | lid_driven_cavity_2d_16m | 1 | 2000 / 500 | 12.8 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 201.4 | 0:5.4 | pass |
| 1550154 | k1_2d16m_g1 | lid_driven_cavity_2d_16m | 1 | 2000 / 500 | 12.9 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 199.5 | 1:5.3 | pass |
| 1550154 | k1_2d16m_g2 | lid_driven_cavity_2d_16m | 1 | 2000 / 500 | 12.8 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 202.7 | 2:5.3 | pass |
| 1550154 | k1_2d16m_g3 | lid_driven_cavity_2d_16m | 1 | 2000 / 500 | 12.7 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 202.7 | 3:5.3 | pass |
| 1550154 | k2_2d16m | lid_driven_cavity_2d_16m | 2 | 2000 / 500 | 24.8 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 104.3 | 0:2.8 / 1:2.8 | pass |
| 1550154 | k4_2d16m | lid_driven_cavity_2d_16m | 4 | 2000 / 500 | 47.4 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 73.1 | 0:1.4 / 1:1.4 / 2:1.4 / 3:1.4 | pass |
| 1550154 | k1_3d8m_g0 | cavity3d_weak4_k2_8m_b4 | 1 | 1000 / 500 | 4.3 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 259.4 | 0:3.0 | pass |
| 1550154 | k1_3d8m_g1 | cavity3d_weak4_k2_8m_b4 | 1 | 1000 / 500 | 4.3 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 258.4 | 1:3.0 | pass |
| 1550154 | k1_3d8m_g2 | cavity3d_weak4_k2_8m_b4 | 1 | 1000 / 500 | 4.3 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 259.8 | 2:3.0 | pass |
| 1550154 | k1_3d8m_g3 | cavity3d_weak4_k2_8m_b4 | 1 | 1000 / 500 | 4.3 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 261.3 | 3:3.0 | pass |
| 1550154 | k2_3d8m | cavity3d_weak4_k2_8m_b4 | 2 | 1000 / 500 | 8.3 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 138.0 | 0:1.7 / 1:1.7 | pass |
| 1550154 | k4_3d8m | cavity3d_weak4_k2_8m_b4 | 4 | 1000 / 500 | 15.8 | 0 | 0 | 0 | 0+0 | 9,119,703 → 9,119,703 | 0 | no | 89.9 | 0:0.9 / 1:1.0 / 2:1.0 / 3:0.9 | pass |
| 1550154 | k4_2d16m_weighted | lid_driven_cavity_2d_16m | 4 | 2000 / 500 | 47.3 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 71.0 | 0:1.4 / 1:1.4 / 2:1.4 / 3:1.4 | pass |
| 1550154 | k4_2d16m_steptrace（`c411b11`） | lid_driven_cavity_2d_16m | 4 | – | – | – | – | – | – | – | – | – | 5.6 | – | 失败：`VkErrorExtensionNotPresent`（只请求 KHR），已由 `a6bce2b` 修复 |
| 1550194 | k4_2d16m_steptrace（`a6bce2b`） | lid_driven_cavity_2d_16m | 4 | 2000 / 500 | 47.5 | 0 | 0 | 0 | 0+0 | 16,184,529 → 16,184,529 | 0 | no | 94.3 | 0:1.4 / 1:1.4 / 2:1.4 / 3:1.4 | pass |

每次运行的 seam 检查都通过，ρ ∈ [998.1, 1001.7]，vmax = 1.0000。worker 绑核 2(K−1) 个（1550154 的 k4_3d8m 里有两条绑核日志被并发打印到同一行，解析器原来只数到 5 个，`parse_run_v6.py` 已修，见第 6 节）。

## 6. 发现的问题与修复

| # | 问题 | 性质 | 处理 |
|---|---|---|---|
| 1 | A100 / aarch64：驱动内部的 `memcpy` 写 Device 类型的 BAR 映射，SIGBUS（`BUS_ADRALN`） | 平台（驱动 535 + Kunpeng-920 + glibc 2.28），v6 无关 | 环境级垫片 `docs/cluster_v6/scripts/devmem_safe_copy.c`（LD_PRELOAD），v6 不改；也可以用 `GLIBC_TUNABLES=glibc.cpu.name=generic`（1M K = 1 单次通过，未在全计划上验证） |
| 2 | step trace 只认 `VK_KHR_calibrated_timestamps`，驱动 535 只有 EXT → 建设备失败 | 仪器的可移植性缺口 | 修复 `a6bce2b`（单独提交）：`VulkanContextV6.create` 的 `extra_device_extensions` 可以给备选元组（启用设备提供的第一个）；step trace 请求 (KHR, EXT)，按设备选函数；EXT 路径显式传规范的 sType 1000184000（python-vulkan 1.3.275.1 没有 EXT 结构体，KHR 结构体默认填 1000543000）；`run_meta.json` 的 `clock.extensions` 记录选择。KHR 设备上的行为不变 |
| 3 | `caps_v6.py --live-sample` 只采 KHR | 脚本局限 | 不修；A100 的时钟质量由 step trace 的校准样本给出 |
| 4 | `parse_run_v6.py` 的绑核计数：两个 worker 线程把绑核日志打印到同一行时，cpu 列表的正则吞掉了后一条的开头，K = 4 少数一个 | harness 脚本 | cpu 列表改为只匹配数字、逗号和连字符；N56 的日志重新解析，结果不变（2 / 4 / 6） |
| 5 | `p2p_probe_linux.py` 读回时对 `vkMapMemory` 的返回值又套了一层 `ffi.buffer`（python-vulkan 已经返回 buffer），探测在最后一步报 TypeError | harness 脚本 | 改为直接切片，并核对整 1 MiB；1550194 重跑 |
| 6 | A100 上进程在 Vulkan 设备没有销毁时退出会段错误（rc 139）：step trace 在 `c411b11` 上抛异常退出时、P2P 探测正常结束时都出现 | 驱动的退出路径；v6 正常运行时会销毁上下文，没有遇到 | 只记录 |

修复 2 的本机 K = 2 冒烟（2 × 5090，2-D 1M n1000，300 步，warmup 100，depth 2）：普通运行 803.4 fps、step trace（KHR）792.8 fps、step trace 强制 EXT（测试包装把备选限定为 EXT，不新增开关）794.8 fps。三次 drift 0、粒子数守恒、600 / 600 行完整；映射残差 rms 24–45 ns，`clock.extensions` 分别记为 KHR × 2 和 EXT × 2；`step_trace_model --run` 读 EXT 的 trace 正常。

## 7. 对后续工作的影响

**E29 的时钟方案能用。** 两代驱动在 Linux 上都提供 `CLOCK_MONOTONIC` 和 `CLOCK_MONOTONIC_RAW`。N56（580，KHR，`CLOCK_MONOTONIC`）和 A100（535，EXT，`CLOCK_MONOTONIC_RAW`）的结果都是：映射残差 ≤ 0.5 µs，覆盖率 100 %，因果违反 0。条件和差别：

- 只有 EXT 的驱动需要 `a6bce2b`。
- 驱动报告的 maxDeviation，580 上约 2 µs，535 上约 8 µs，都比 Windows 小。
- A100 上只拟合一条直线的残差有 5–8 µs，几十秒以上的运行必须依赖拟合里的漂移项，不能只用直线。

还没测的：K = 8（14 条链路）时 trace 的开销和覆盖率，因为 N56 的 8 卡部分没有跑。

**E7 的算例矩阵：这次没有答案**，因为 S3 没跑。能给出的只有估计和建议：

- K = 1 参照装不装得下（3-D 64M `cavity3d_weak8_k8_64m_b4`，70,632,177 个粒子，v5 时占 23 GB）。实测每粒子显存峰值：A100 上 3-D 8M K = 1 为 3078 MiB / 9,119,703 ≈ 354 B，2-D 16M K = 1 为 356 B；`nvidia-smi` 读数含上下文开销，按粒子数线性外推会偏大。两个算例的 `pool_size` 都是粒子数的 1.15 倍（10,487,680 和 81,227,008），按 354 B 外推，64M 的 K = 1 约需 23.3 GiB，5090 有 31.84 GiB，应能装下，余量约 8.5 GiB。**这是外推，不是测量**。最便宜的确认是本机一张 5090（32 GB）上跑 300 步，不用集群。
- cube 401³ 的 K = 8：409³ = 68.4M 个粒子，约 104 个 voxel 列，每段约 13 列，贴着 `MINIMUM_OWN_COLUMNS_HARD = 12`。分区器接不接受、每段实际几列，不需要 GPU 也能离线回答：在登录节点生成或读入算例后只调用切点规则（`partition_v6.chain_cuts_from_counts`）。这次没做。
- 3-D 的 departed 池在起步阶段就已经用到容量的 13 %（K = 2/3/4 的 3-D 8M：每段峰值 126–155 / 1184 或 2367），2-D 只有约 0.3 %（≤ 2 / 645–1290）。

**E15 还要测什么。** 这次从静止起步，最多跑 2000 步，得不出余量结论（E30 已预见）。能看到的只是起步状态：所有 overflow 计数为 0，far_migration 为 0，3-D departed 池峰值约为容量的 13 %，2-D 约 0.3 %。E15 要测的是：

1. 发展流（流动发展完成之后：8M soak 里迁移量是在小时尺度上爬升的）里 ghost、migrant、departed 各池和 install tail 的高水位。
2. 在 K = 8 上测，内部 slab 有两条 seam，边界占比最大。
3. 以 3-D 为主：池因子 0.5 / 0.02 / 0.64，起步就到 13 %。
4. 打开 `V6_POOL_PEAKS=1`。这个开关已经有了，E30 不允许导出，E15 可以用。

**A100 移植的结论。**

- v6 的算法和代码在 aarch64 / A100 / 驱动 535 上不用改就能正确运行：drift 0，全部计数 0，seam 检查通过。前提是绕开驱动的对齐问题，这是平台问题，用 LD_PRELOAD 垫片或 glibc tunable 解决。
- 单卡约为 5090 的 1/3。
- K = 4 的 η 约 92–93 %（单次）。传输链在 2-D 16M 上占 phase B 的 3–6 %。Kunpeng 上 host 拷贝比 N56 慢约一半，但仍然藏得住。
- 卡之间没有可用的 Vulkan 显存共享。

## 8. 文件

- 脚本：`docs/cluster_v6/scripts/`（`e30_lib.sh`、`smoke_small.sbatch`、`a100_smoke.sbatch`、`a100_steptrace.sbatch`、`run_chain_v6.py`、`parse_run_v6.py`、`caps_v6.py`、`report_tables_v6.py`、`p2p_probe_linux.py`、`devmem_safe_copy.c`、`deploy_v6.sh`、`setup_cases_v6.sh`），`remote/bringup_check_v6.py`。
- 数据（入库的小文件）：`docs/cluster_v6/data/`。
- 原始日志（不入库，按惯例镜像）：`logs/n56/2026-10-05/01_e30_smoke_1665986`、`02_e30_smoke_1665987`、`logs/n56/2026-10-06/01_e30_smoke_1666242`；`logs/n32h/2026-10-06/`。
