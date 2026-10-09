# E7 第一步：v7 冒烟（N56 hp_5090，2026-10-09）

> 数据不进论文。作业 1679191：节点 wqd10nbj04g2，8 × RTX 5090，`-p hp_5090 -A hp5090 -N 1 --gpus=8`，15:05–15:58 CST，用时 53 分 29 秒。17 次链运行全部有效，5 个单步检验全部 PASS。
> 版本：标签 `v7-rc1`（d0c8dcb，`E30_SOLVER=v7`）。脚本：分支 `e7-cluster` 的 f88d2e0（`docs/cluster_v6/scripts/e7_smoke.sbatch`）。

## 0. 结论

1. **v7-rc1 在集群驱动上运行正确。** 驱动 580.82.07，Linux。
   - bring-up 通过：8 张卡都能建上下文，每张卡两个传输队列；v7 的 22 个 SPIR-V 齐全；1M 的 K = 1 与 K = 2 通过。
   - 2-D 16M 与 3-D 立方体各跑 K = 2 / 4 / 8，每次 1500 步，中间过一次 defrag（第 1000 步）。全部 17 次运行的 drift、溢出、far_migration、帧戳都是 0，seam 检查也全部 OK。
2. **B1 的逐位结论在集群驱动上成立。**
   - `fused_single_step.py` 跑了 5 个配置：2-D 1M 的 K = 2 / 8 / 1，3-D n104（9 层壁）的 K = 2 / 1，全部 PASS。
   - L、kernel sum、∇ρ 在每个比较行上都是 0 ULP。
   - ρ 每步有 1–38 个自有流体粒子差 1 ULP（本机 0–37）。P 最大差 1.36 Pa。没有差 2 ULP 以上的行。
3. **计算节点上没有 Khronos 验证层。**
   - loader 只提供 NV_optimus、NV_present、MESA_device_select、INTEL_nullhw、MESA_overlay 五个层，系统目录和离线环境里都找不到 `VK_LAYER_KHRONOS_validation`。
   - 因此 K = 8 的验证运行做不了，按“无”记录。
4. **255.6M（`cavity2d_n5640_k8`）K = 8 带 `--obj-cache`：**
   - 建链 372 s，其中构造 241 s、bootstrap 131 s。这是该配置在本作业里第一次运行，包含 pipeline 编译（见第 4 节）。
   - 稳态 24.9 fps，粒子数守恒（255,594,004）。
   - 进程主机内存峰值 21.5 GiB；建链结束时为 16.7 GiB，峰值出现在运行后的检查阶段。
   - 8 个 K = 1 参照（`cavity2d_n5640`，32.1M）同时建链：每个进程峰值 4.7 GiB，合计 37.9 GiB，节点内存比窗口开始时高 36.4 GiB。
5. **E15 池峰值**（K = 8，3000 步，从静止起步，不计时）：

   | | 2-D 16M | 3-D 立方体 |
   |---|---|---|
   | replica 区峰值 / 容量 | 20,111 / 26,179（77 %） | 720,091 / 933,120（77 %） |
   | migrant 区峰值 / 容量 | 2 / 645（0.3 %） | 414 / 7,465（5.5 %） |
   | departed 池峰值 / 容量 | 2 / 1,290 | 414 / 14,930（2.8 %） |
   | install tail 最深 | 65 | 7,931 |
   | own pool 余量最小值 | 40 万 | 186 万 |

   replica 区的需求就是一列 voxel 的静止粒子数，与容量规则的预期一致。迁移相关的区是起步状态的值；发展流的高水位还要更长的运行（E15 的原定义）。
6. **建链慢的原因：** 见第 4 节。它在计时窗口之外，不影响 fps 和扩展效率，按你的意见先不改。
7. **全量 E7 的作业清单与预算：** 见第 6 节。51 个点；计时运行不做 seam 检查（已定），约 24.0 node-h（192 GPU-h），两个节点并行约 13.5 h。

## 1. 版本、部署与主机设置

- **求解器：** 标签 `v7-rc1`，附注标签对象 aea1b01，指向提交 d0c8dcb，即 v6-rc2 + B9 / B6 / B4 / B1，B3 默认关。由 E39 会话推送。
- **部署：** `~/run/vulkan-demo-v7rc1`，共 225 个文件。
  - 来源是标签的子集归档：experiment/v7、seam_audit 工具、materials、cases/aligned 和对齐生成器。归档 sha256 3eb23e50…，用 `deploy_v6.sh` 的新选项 `E30_DEPLOY_ALL=1` 解出。
  - 作业开头检查 MANIFEST.sha256（225 个文件）和 SCRIPTS.sha256（21 个文件），两者都 OK。
  - SPIR-V 共 22 个，与 `experiment/v7/shaders/spv/MANIFEST.txt` 一致。
- **主机设置：** 继承的 V5_ / V6_ / V7_ 变量全部清掉，只导出 `V7_WORKER_AFFINITY`。
  - affinity 按 nvidia-smi 顺序：GPU 0–3 对应 NUMA0 的 0-31,64-95，4–7 对应 NUMA1 的 32-63,96-127。`caps_v6.py` 确认 Vulkan 的 discrete-first 序号与 nvidia-smi 序号相同。
  - GIL 切换间隔 0.2 ms，两个传输队列，其余开关都是代码默认值。
  - 作业开头的 `run_chain_v6.py --config-only` 打印确认了 v7-rc1 的默认值：`V7_DENSITY_COPY_COMPUTE=1 V7_GHOST_SEND_LANES=32 V7_DEEP_WALL_SKIP=auto V7_FUSED_CORRECTION_DENSITY=1 V7_BAND_OVERLAP=0`。
  - 另外只有 E15 的两次运行在命令行上设了 `V7_POOL_PEAKS=1`。
- **节点：** wqd10nbj04g2。2 × Xeon 6530，每个 NUMA 节点挂 4 张卡；功率上限 575 W；时钟域 DEVICE、CLOCK_MONOTONIC、CLOCK_MONOTONIC_RAW（KHR 与 EXT）。
  - 负载下 8 张卡都在 575 W 上限。SM 时钟中位数 2722–2775 MHz（p10–p90 为 2692–2820 MHz），温度不超过 71 °C，没有降频。

## 2. 准备：算例与 `--obj-cache`（登录节点，不计费）

- **生成：** 40 个对齐算例都在登录节点上生成，用 `gen_aligned_cases.sh`，即逐个跑 generate.txt 里的 `--objs-only` 那一行。
  - 每个粒子文件的行数都与 generate.txt 第一行的计数核对过。生成器是原地写文件、不回读的，被截断的文件否则发现不了。
  - case.yaml 前后的 sha256 不变。
  - 冒烟用的 6 个算例在 13:00–13:22 生成，其余 34 个在 13:26–13:56。
  - Linux 上生成的 obj 与本机生成器的输出逐字节相同（n1000 与 n104_b9 的 sha256 都已核对）。
- **缓存：** `--obj-cache` 文件由 `obj_cache_build.py` 预先建好，共 17 GB，放在 `~/run/e7_objcache`。
  - 它用的是 harness 自己的缓存键和写法，只是解析改成分块，结果与原解析器逐位相同（`--verify` 已核对）。255.6M 的建缓存过程只占约 2 GiB 内存，原解析器约需 37 GB。
  - 255.6M 一个算例的生成加建缓存用了 1280 s。
- **作业内：** 每个作业开头把需要的缓存从 `~/run` 拷到节点本地 /dev/shm。冒烟拷了 4.56 GB，用时 38.3 s（119 MB/s）。之后的运行都读本地副本，每次运行 4 次命中。

## 3. 冒烟结果

### 3.1 bring-up

env、case、K1、K2 四个阶段都通过：
- 8 个 NVIDIA discrete 设备，`VulkanContextV7[0..7]` 都 OK（compute qf 0，transfer qf 1，两个队列分开）。
- loader 1.4.357。
- 1M 对齐算例 K = 1：651 fps；K = 2：558 fps（300 步，每 100 步 defrag，只作功能检查）。

### 3.2 `fused_single_step`（B1，集群驱动）

做法：先从初始状态 K = 1 发展 2000 步，保存状态；再在同一进程里从这个状态各跑一步分开的 kernel 与融合 kernel，逐粒子比较。

| 配置 | K | 结论 | 比较行数 | 跳过的深壁行 | L / kernel sum / ∇ρ 逐位相同 | ρ 有差的自有流体粒子 | 其中 1 ULP | ghost-self 行 | 最大 ULP | 最大 \|Δρ\| | 最大 \|ΔP\|（Pa） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2-D 1M（n1000） | 2 | PASS | 1,054,704 | 0 | 是 | 2 | 2 | 0 | 1 | 6.1e-5 | 1.19 |
| 2-D 1M | 8 | PASS | 1,116,024 | 0 | 是 | 1 | 1 | 0 | 1 | 6.1e-5 | 0 |
| 2-D 1M | 1 | PASS | 1,044,484 | 0 | 是 | 1 | 1 | 0 | 1 | 6.1e-5 | 0.60 |
| 3-D n104_b9 | 2 | PASS | 1,561,453 | 373,480 | 是 | 38 | 38 | 1 | 1 | 6.1e-5 | 1.36 |
| 3-D n104_b9 | 1 | PASS | 1,404,928 | 410,920 | 是 | 36 | 36 | 0 | 1 | 6.1e-5 | 1.36 |

- “0 ULP 依赖驱动对相同表达式做相同的 FMA 合并”这条附注（`v7_perf.md`），在 580.82.07 上的实测结果是成立。
- ρ 的差都是 1 ULP。3-D 每步 36–38 个，本机同一算例为 35–37 个；“0–32”不是硬上限（E39 评审时有一次为 38）。

### 3.3 K = 2 / 4 / 8：2-D 16M（`cavity2d_n4000`）与 3-D 立方体（`cavity3d_n416`），各 1500 步

| 运行 | K | 稳态 fps | 建链（s） | 计时循环（s） | 运行后检查（s） | 每卡显存峰值（MiB） | 不变量 |
|---|---|---|---|---|---|---|---|
| 2-D 16M | 2 | 92.2 | 40.3 | 16.2 | 7.9 | 2,885 | 全 0，seam OK |
| 2-D 16M | 4 | 156.8 | 76.7 | 9.5 | 8.5 | 1,489 | 全 0 |
| 2-D 16M | 8 | 110.2 | 156.5 | 13.5 | 9.4 | 790 | 全 0 |
| 立方体 76.2M | 2 | 4.2 | 89.4 | 349.9 | 43.2 | 13,658 | 全 0 |
| 立方体 | 4 | 7.9 | 117.3 | 186.4 | 71.1 | 7,570 | 全 0 |
| 立方体 | 8 | 15.2 | 186.6 | 97.0 | 118.0 | 4,393 | 全 0 |

- 立方体在 K = 8 时切成 15 / 13 × 6 / 15 列，每段至少 13 列，高于硬下限 12，只打印了 WARN。
- 2-D 16M 的 K = 8 是每卡 2M，属于传输暴露区。同一配置在 S5 里再跑一次（3000 步）是 146.1 fps，与这里的 110.2 fps 差 33 %；轻负载下这种双峰以前在 N56 上见过（每卡不超过 4M 时）。所以全量里每个点要 3 次以上试验。
- 以上是单次运行，不能当效率数。

### 3.4 验证层

作业最后用节点上 loader 实际提供的层判断（去掉 env.sh 的禁用变量后调 `vkEnumerateInstanceLayerProperties`）：

```
VK_LAYER_NV_optimus 1.4.312, VK_LAYER_NV_present 1.4.312, VK_LAYER_MESA_device_select 1.3.211,
VK_LAYER_INTEL_nullhw 1.1.73, VK_LAYER_MESA_overlay 1.3.211
```

- 没有 `VK_LAYER_KHRONOS_validation`。目录清单（`s0_layers.log`）显示系统里也没有它的 .so。
- 这一项写明“离线环境没有验证层，未运行”。本机的 v7-rc1 验证层运行（E39：K = 2 2-D 1M 与 K = 4 3-D 8M，0 条消息）是现有的唯一证据。

### 3.5 255.6M 与 8 个 K = 1 参照的主机内存

| 窗口 | 进程 RSS 合计峰值 | 最大单进程 VmHWM | 各进程 VmHWM 之和 | 节点内存：峰值 / 高于窗口开始 | 作业 cgroup（含页缓存与 /dev/shm 缓存）：峰值 / 高于开始 |
|---|---|---|---|---|---|
| 255.6M K = 8（1 个进程） | 21.4 GiB | 21.5 GiB | 21.5 GiB | 129.9 / 21.8 GiB | 29.7 / 20.0 GiB |
| 8 × K = 1 32.1M（8 个进程） | 36.3 GiB | 4.7 GiB | 37.9 GiB | 144.5 / 36.4 GiB | 41.8 / 32.1 GiB |

- 255.6M 进程在建链结束时为 16.7 GiB，退出前为 21.5 GiB；后者来自运行后 seam 检查的全量回读和 np.unique。
- 每张卡显存约 11.2 GB。8 个 K = 1 的 32.1M 每张卡 10.7 GB。
- 节点有 1 TB 内存，按卡分配 126 GB，余量很大。缓存命中时不会出现纯 Python 解析的约 37 GB 峰值。
- 时间分解：
  - 255.6M K = 8：构造 241 s（首次配置，含编译）、bootstrap 131 s、300 步 12 s、运行后检查 107 s。
  - 8 个 K = 1：各自构造约 26 s、bootstrap 约 24 s，8 个进程并行，互不拖慢。

### 3.6 E15 池峰值（K = 8，3000 步，`V7_POOL_PEAKS=1`，harness 打印）

- 每条链路、每个区的容量与峰值见 `data/e7/summary_1679191.md`。汇总在第 0 节第 5 条。
- 3-D 的 migrant 峰值 414 出现在起步阶段，均值只有 3.9。departed 池峰值 414 / 14,930。install tail 最深 7,931，相对于 186 万的余量微不足道。
- 2-D 起步阶段几乎没有迁移。
- 结论：按发布的池因子，这两个 K = 8 配置在起步 3000 步内离容量都远。发展流（小时尺度）的高水位仍未测，这是 E15 本身的任务。

## 4. 建链为什么慢

依据：同一作业的数据（telemetry 每秒采样、主机内存采样、各运行的阶段时间），以及代码路径。三路独立分析后再汇总。

**结论：** 慢在每种新配置第一次运行时要建 pipeline，驱动的 shader 磁盘缓存不命中。建链在计时窗口之外，不影响 fps 和扩展效率；按你的意见，求解器与协议都不改。

- **机制（代码）：**
  - 每个 simulator 建 29 条 compute pipeline（`simulator_v7.py:834-855`），每条单独调用 `vkCreateComputePipelines(..., VK_NULL_HANDLE, 1, ...)`，没有应用层 pipeline cache（`:2502-2503`）。
  - 每个 slab 的几何写在 specialization constants 里（`:1778-1845`；`partition_v7.py:1041-1047`），包括 origin_x、网格宽度、own / ghost / departed 池大小和传输偏移。所以每个（算例、K、切点、pool_safety、slab）都是一个新的缓存键，同一条链里的 slab 之间也不能复用。
  - 各 sim 串行构造（`_run_v7_chain_bench.py:349-357`）。
- **判定性证据（实测，同一作业、同一配置、同一组卡）：**
  - 2-D 16M K = 8：开始 bootstrap 的时刻从 145.4 s 降到 9.7 s。冷构造每个 sim 13.9–20.7 s，均值 17.3 s，期间 GPU 利用率全是 0 %；热构造时 8 个 sim 在约 2 s 内全部建完。
  - 立方体 K = 8：143.0 s → 39.1 s，其中两次都有约 30 s 在读入算例。
  - 8 个 K = 1 进程各自并行付这笔代价，互不串行。
- **慢在哪一步（推断，未测）：**
  - 本机（Windows、SSD）同一份代码冷构造只多约 0.8 s / sim（约 28 ms / pipeline），集群上约 0.5 s / pipeline。
  - 立方体内部几何相同的 slab，冷构造从 7.5 s 到 22.4 s 不等。
  - 所以更可能是驱动把新条目同步写进 JuiceFS 上的 `~/.cache/nvidia/GLCache`，而不是 CPU 编译。一次冷构造的 A/B（缓存放 /dev/shm，或 `__GL_SHADER_DISK_CACHE=0`）可以定论。
- **不影响效率：**
  - 循环里不编译：pipeline 在构造时建好，命令缓冲在 bootstrap 里录好。
  - 冷热两次的 bootstrap 时长相同（11.1 / 10.4 s，43.6 / 44.2 s）。
  - 本机 7 对冷热构造的同一配置，稳态 fps 相差不超过 0.3 %。
  - 集群上两对的方向相反：立方体冷 15.2、热 14.5 fps。2-D 16M K = 8 的 110 对 146 fps 是传输等待造成的：冷那次 GPU 4–7 的链路每帧中位多等约一帧，GPU 每帧忙时相同（2.85 / 2.88 ms），差距从前 500 帧就已存在。这是每卡 ≤ 4M 时的老问题，靠试验次数与 η_min 处理。
- **在全量里的代价：**
  - 按协议，冷的只有校准 pilot 和每个 K = 1 参照算例的第一组；计划里约 2.3 node-h，约占 10 %。
  - 两项未建模的冷构造，合计至多约 1 h：第 2 轮仍移动切点时，校准臂的第一次运行是冷的；step trace 多开的设备扩展可能改变缓存键。
  - 风险：全量要新写约 2 GiB 缓存，超过驱动默认的 1 GiB 上限，可能触发清理。两个节点同时写同一个 JuiceFS 缓存文件也没测过。最坏全部不命中，多 7.7 node-h。只多花时间，不改结果。
  - 冷构造会让 K = 8 的单次运行多约 3 min，per-run timeout 要留余量。
- **以后若要省时间（只改环境，不改求解器）：** 每个作业把驱动缓存放到节点本地（`__GL_SHADER_DISK_CACHE_PATH`，加大 `__GL_SHADER_DISK_CACHE_SIZE`）。可能省约 2 node-h，并消除上限和共享写的风险，但要先做一次冷构造 A/B。
- **零成本的事后检查（全量里照做）：**
  - 热运行的构造时间是否约等于“读入 + 1 s × K”；
  - GLCache 的大小；
  - 小算例第 1 次试验时，K = 1 参照组的计时窗口是否重叠。参照进程并行启动，冷构造的结束时间相差约 6 s；全量脚本会加启动屏障。
- **与缓存无关、但同样在计时窗口外：**
  - 读入算例加分区随 K 增长（每个 slab 都要对全体粒子过滤一遍，`partition_v7.py:1169-1170`；255.6M 为 95 s）。
  - bootstrap 里有逐粒子的 Python 循环（`simulator_v7.py:3271-3283`），约 0.5 s / 百万粒子，255.6M 为 131 s。
  - 都只影响 node-hour，已计入预算。

## 5. 问题与修改（都在 harness，求解器未动）

- **E15 的池峰值原来不输出。** `V7_POOL_PEAKS=1` 只在 transport worker 里记录，没有任何地方打印。`run_chain_v6.py` 加了钩子：按链路、按区打印容量、峰值、出现的帧、p99.9、均值，并逐 slab 打印 pool-health 原始值（13f620d、c925d5e）。
- **bench 不打印建链时间和内存。** `run_chain_v6.py` 在 v7 下按阶段打印时间与 VmRSS / VmHWM（13f620d）。
- **部署。** `deploy_v6.sh` 加了 `E30_DEPLOY_ALL=1`，用来部署子集归档；生成的 *.obj 不进 manifest（d299f49）。
- **算例与缓存。** 新增 `gen_aligned_cases.sh` 与 `obj_cache_build.py`（0cc185d）。
- **提交前的对抗式审查，共 3 个角度，每条中等以上的问题再由独立的验证者尝试反驳（c925d5e）：**
  - 内存采样器停止时会崩溃，并丢掉每个进程的峰值。这是真问题，已修好，并在登录节点实测。
  - 验证层运行原来排在 S4 / S5 之前。在本节点上没有影响（层不存在），仍移到最后，并改成直接问 loader。
  - 另有几处改进：
    - 缓存在作业内先拷到节点本地；
    - fused 单步结果为 INVALID 或没有 JSON 时停止作业；
    - 8 个 K = 1 中有硬失败时停止作业；
    - nvidia-smi 调用和配置打印加超时；
    - 日志同步只拷变化的文件。
  - “E15 缺 install tail”这一条被驳回：`peak_migration` 本来就是尾部深度的精确峰值。
- **第一次提交（1678956）在排队时取消。** SLURM 在提交时就固定了作业脚本，审查后的修复只能重新提交（1679191）。hp_5090 上没有别的排队作业，优先级也是平的，所以重提不吃亏。
- **登录节点的一次失误。** 第一次启动生成任务时，心跳循环每分钟把所有 domain.obj 读一遍。1 分钟后发现，已停掉，重新在 tmux 里启动；日志在 `00_setup_and_casegen/first_launch_aborted.out`。

## 6. 全量 E7：作业清单与 node-hour 预算

**规则（你给的 E7 规格，加上 E29 / E31 / E32 / E30 里对 E7 的建议）：**
- 算例全部来自 `cases/aligned/`，simple 壁，不用 *_adami。
- 每个点（算例、K ≥ 2）都在它所在的节点上、紧挨着试验之前校准一次：`--weights auto --calibrate-only --weights-file`，显式给 `--device-map`。
- 然后做 3 次试验。每次试验包括三项，顺序在三次之间轮换（R E C / E C R / C R E）：
  - K = 1 参照：在参与的每张卡上同时跑，窗口与 K 运行相同；
  - 等权重臂：`--weights 1,…,1 --device-map`；
  - 校准臂：`--weights-file` 加 `--device-map`。
- 每个点另加一次 step trace 运行（等权重），记录 E29 的字段：t_tr、T_B、各 phase 时间。
- 窗口：2-D 3000 步（warmup 1000），3-D 1500 步（warmup 500），与 probe34–37 相同。
- 每个日志都用 `parse_run_v6.py` 判定：drift、溢出、far_migration、帧戳全为 0 才有效。
- 节点 A 跑 K ≤ 4（用 GPU 0–3，整机），节点 B 跑 K = 8（整机），两个节点并行。

**点（51 个）：**

| 族 | 内容 | 节点 A（K ≤ 4） | 节点 B（K = 8） |
|---|---|---|---|
| F1 2-D 强扩展 | n2000 / n2840 / n4000 / n5640 / n8000 / **n9000**（有真正的 K = 1 参照）/ **n11320**（128M：参照为 4 对 K = 2 同时跑，η × K/2，单独标注） | — | 7 点 |
| F2 2-D 固定 N 的 K 扫描 | n5640、n8000 的 K = 2 / 4（K = 8 即 F1） | 4 点 | — |
| F3 2-D weak | 每卡 2M / 4M / 8M / 16M / 32M：k2 / k4 / k8 成员，参照为 k1 算例 | 10 点 | 5 点（含 255.6M） |
| F4 3-D weak | 每卡 4M / 8M：n160、n200 的 k2 / k4 / k8，参照为 k1 | 4 点 | 2 点 |
| F5 3-D 强扩展（stretched 32M / 64M） | n160_k8、n200_k8 的 K = 2 / 4 / 8，参照为 K = 1 的同一算例 | 4 点 | 与 F4 的 K = 8 共用运行，另加强扩展参照 |
| F6 立方体 | n416 K = 8，K = 1 参照（24.5 GiB，装得下） | — | 1 点 |
| F7 E29 每卡点（2-D） | 每卡 32k / 130k / 0.5M / 2M / 8M：K = 2 的 n240…n4000，K = 4 的 n360…n5640，K = 8 的 n520…n8000 | 9 点（n5640 K = 4 与 F2 共用） | 2 点（其余与 F1 共用） |
| F8 E29 每卡点（3-D） | n104_b9、n200_b9、窄 slab n200_x96，K = 2 | 3 点 | — |

**时间模型（`docs/cluster_v6/scripts/e7_plan.py`）：** 每次运行 = 构造 + bootstrap + 计时循环 + 运行后检查，各项都用冒烟实测值：
- **构造：**
  - 首次运行某个配置要编译 pipeline，每个 sim 约 17 s；重复运行命中缓存，每个 sim 约 1.2 s。另加载入与分区，约 0.35 s / 百万粒子。
  - 按协议，只有校准的 pilot 和每个算例的第一组 K = 1 参照是“冷”的。原因：pilot 1 就是等权重，等权重臂与 trace 随后命中缓存；最后一轮 pilot 是校准后的切点，校准臂随后命中缓存。
- **bootstrap：** 2 s + 0.5 s / 百万粒子。
- **计时循环：** K = 1 的单步成本在 2-D 为 1.258 ns / 粒子步（32M 参照 24.8 fps），3-D 为 6.0 ns（立方体 K = 2 为 4.2 fps）。再按 K 取效率（0.95 / 0.90 / 0.87），轻负载另设下限。
- **运行后检查：**
  - 带 seam 检查：2-D 为 0.42 s / 百万粒子 + 8 s；3-D 为 (0.26 + 0.16 K) s / 百万粒子，与立方体实测的 43 / 71 / 118 s 吻合。
  - `--no-seam-check` 时约为 0.1 s / 百万粒子 + 1 s / slab（估计）。
- 模型对冒烟里重复运行的预测与实测相差在 ±10 % 以内（立方体 K = 8：预测 37 + 40 + 99 + 117 s，实测 39 + 44 + 97 + 117 s）。

**已定（2026-10-09）：** 计时运行用 `--no-seam-check`，seam 检查只在每点的 trace 运行上做。drift、溢出、far_migration、帧戳在 seam 检查之前单独算，判定不受影响。

**预算（3 次试验、每点 1 次 trace，每条节点线 2 个作业的开销各 10 分钟）：**

| 方案 | 节点 A | 节点 B | 合计 | GPU-h（整机 8 卡计费） | 两节点并行的墙钟 |
|---|---|---|---|---|---|
| **计时运行不做 seam 检查（已定）** | 13.5 h | 10.4 h | **24.0 node-h** | 192 | ≈ 13.5 h |
| 对照：计时运行也做 seam 检查 | 14.9 h | 11.8 h | 26.7 node-h | 213 | ≈ 15 h |

**按族（已定方案，h）：**
- 节点 A：F2 2.2、F3 3.4、F4 1.4、F5 4.3、F7 1.3、F8 0.6。
- 节点 B：F1 3.6、F3 2.9、F4 + F5 1.9、F6 1.2、F7 0.4。
- 逐点的表见附录 A。

**作业组织（建议）：**
- 每条节点线分 3–4 个作业，每个 2–4.5 h。作业名按节点线取 `e7_A` / `e7_B`，用 `--dependency=singleton`，同一条线一次只跑一个。hp5090 账户的 16 卡上限正好容纳两条线。
- 节点 A 的作业：
  - A1 = F2 + F7（2-D，约 3.5 h）；
  - A2 = F3（约 3.4 h）；
  - A3 = F4 + F8（约 2 h）；
  - A4 = F5（约 4.3 h）。
- 节点 B 的作业：
  - B1 = F1 + F7（约 4.1 h）；
  - B2 = F3（约 2.9 h）；
  - B3 = F4 / F5 + F6（约 3.2 h）。
- 每个作业开头：prelude、配置打印、env 级 bring-up、缓存拷到本地。
- 每次运行都加 timeout（按冷构造留余量，K = 8 每次多约 3 min）；遇到硬失败就停在那一步。
- 每个作业 `-A hp5090 -N 1 --gpus=8`，整机。
- 排队：今天整机等了约 50 分钟（重新提交到开始）；之前的估计一度到 20:29。7 个作业的排队时间不可预估。

**还可以减少的地方（都不影响效率的定义）：**
1. **节点 A 上 K = 2 与 K = 4 共用 4 卡的 K = 1 参照组**（同一算例、同一次试验）：省约 1.5 node-h。代价是 K = 2 的参照运行时，GPU 2–3 也在跑参照，与“只在参与的卡上同时跑”有一点偏离。
2. **驱动 shader 缓存放到节点本地**：可能省约 2 node-h，要先做一次 A/B（见第 4 节）。

**不确定性：**
- 模型只用了一台节点上的冒烟数据。
- 轻负载点（每卡不超过 4M）的 fps 在运行之间可差 30 %，这会影响它们的循环时间，但这些点本来就短。
- 驱动缓存超过上限后可能多出冷构造，最坏多 7.7 node-h（第 4 节）。
- 预算没有计入失败重跑。

**下一步：** 你看完 artifact，评估缺什么，再给完整任务。之后我写全量作业脚本（每个作业一份点清单，K = 1 参照组加启动屏障），在本机用小算例预跑，做一轮审查，再提交。

## 7. 文件

- **脚本：** `docs/cluster_v6/scripts/` 下的 `e7_smoke.sbatch`、`e7_plan.py`、`e7_summary.py`、`host_memory_sampler.py`、`obj_cache_build.py`、`gen_aligned_cases.sh`，以及改动过的 `run_chain_v6.py`、`parse_run_v6.py`、`e30_lib.sh`、`deploy_v6.sh`、`report_tables_v6.py`。
- **数据（入库的小文件）：** `docs/cluster_v6/data/e7/`，包括 `results_1679191.jsonl`、`memory_windows_1679191.jsonl`、`caps_1679191.json`、`summary_1679191.md` 和 `fss_1679191/*.json|md`。
- **原始日志（不入库，按惯例镜像）：** `logs/n56/2026-10-09/01_e7_smoke_1679191/`（含 .out、sbatch、遥测、各运行日志）和 `logs/n56/2026-10-09/00_setup_and_casegen/`；README 与 jobs.csv 已更新。
- **集群上：** `~/run/logs/e7_smoke_1679191/`、`~/run/logs/e7_setup/`、`~/run/e7_objcache/`。

## 附录 A：逐点预算（`e7_plan.py --markdown`，已定方案：计时运行不做 seam 检查）

每点 = 校准（2 轮 pilot）+ 3 ×（参照组 + 等权重臂 + 校准臂）+ 1 次 trace（做 seam 检查）。“运行”与“参照组”是命中缓存的值（分钟）；第一组参照另加冷编译。

| node | family | case | K | particles | reference per trial | run (min) | reference set (min) | calibration (min) | point total (h) |
|---|---|---|---|---|---|---|---|---|---|
| A | F2 | cavity2d_n5640 | 2 | 32.1M | K=1 of the case | 1.8 | 2.8 | 3.3 | 0.42 |
| A | F2 | cavity2d_n8000 | 2 | 64.4M | K=1 of the case | 3.4 | 5.3 | 4.7 | 0.75 |
| A | F2 | cavity2d_n8000 | 4 | 64.4M | K=1 of the case | 2.5 | 5.3 | 5.5 | 0.66 |
| A | F2+F7 | cavity2d_n5640 | 4 | 32.1M | K=1 of the case | 1.4 | 2.8 | 4.3 | 0.38 |
| A | F3 | cavity2d_n1440_k2 | 2 | 4.2M | K=1 cavity2d_n1440 | 0.5 | 0.4 | 2.1 | 0.12 |
| A | F3 | cavity2d_n1440_k4 | 4 | 8.5M | K=1 cavity2d_n1440 | 0.7 | 0.4 | 3.5 | 0.17 |
| A | F3 | cavity2d_n2000_k2 | 2 | 8.1M | K=1 cavity2d_n2000 | 0.7 | 0.6 | 2.3 | 0.15 |
| A | F3 | cavity2d_n2000_k4 | 4 | 16.2M | K=1 cavity2d_n2000 | 0.9 | 0.6 | 3.7 | 0.20 |
| A | F3 | cavity2d_n2840_k2 | 2 | 16.3M | K=1 cavity2d_n2840 | 1.1 | 0.9 | 2.6 | 0.22 |
| A | F3 | cavity2d_n2840_k4 | 4 | 32.6M | K=1 cavity2d_n2840 | 1.4 | 0.9 | 4.4 | 0.29 |
| A | F3 | cavity2d_n4000_k2 | 2 | 32.3M | K=1 cavity2d_n4000 | 1.9 | 1.5 | 3.3 | 0.36 |
| A | F3 | cavity2d_n4000_k4 | 4 | 64.4M | K=1 cavity2d_n4000 | 2.5 | 1.5 | 5.5 | 0.47 |
| A | F3 | cavity2d_n5640_k2 | 2 | 64.0M | K=1 cavity2d_n5640 | 3.4 | 2.8 | 4.7 | 0.62 |
| A | F3 | cavity2d_n5640_k4 | 4 | 127.9M | K=1 cavity2d_n5640 | 4.6 | 2.8 | 7.9 | 0.82 |
| A | F4 | cavity3d_n160_k2 | 2 | 9.3M | K=1 cavity3d_n160 | 1.2 | 1.0 | 2.7 | 0.24 |
| A | F4 | cavity3d_n160_k4 | 4 | 18.3M | K=1 cavity3d_n160 | 1.4 | 1.0 | 4.2 | 0.29 |
| A | F4 | cavity3d_n200_k2 | 2 | 17.7M | K=1 cavity3d_n200 | 1.9 | 1.7 | 3.4 | 0.38 |
| A | F4 | cavity3d_n200_k4 | 4 | 35.0M | K=1 cavity3d_n200 | 2.4 | 1.7 | 5.2 | 0.46 |
| A | F5 | cavity3d_n160_k8 | 2 | 36.4M | K=1 of the case | 3.7 | 6.3 | 5.0 | 0.84 |
| A | F5 | cavity3d_n160_k8 | 4 | 36.4M | K=1 of the case | 2.4 | 6.3 | 5.3 | 0.70 |
| A | F5 | cavity3d_n200_k8 | 2 | 69.6M | K=1 of the case | 6.9 | 11.8 | 7.8 | 1.53 |
| A | F5 | cavity3d_n200_k8 | 4 | 69.6M | K=1 of the case | 4.3 | 11.8 | 7.3 | 1.24 |
| A | F7 | cavity2d_n1000 | 2 | 1.0M | K=1 of the case | 0.4 | 0.3 | 2.0 | 0.10 |
| A | F7 | cavity2d_n1440 | 4 | 2.1M | K=1 of the case | 0.6 | 0.4 | 3.3 | 0.15 |
| A | F7 | cavity2d_n2000 | 2 | 4.1M | K=1 of the case | 0.5 | 0.6 | 2.1 | 0.12 |
| A | F7 | cavity2d_n240 | 2 | 0.1M | K=1 of the case | 0.4 | 0.3 | 2.0 | 0.10 |
| A | F7 | cavity2d_n2840 | 4 | 8.2M | K=1 of the case | 0.7 | 0.9 | 3.5 | 0.19 |
| A | F7 | cavity2d_n360 | 4 | 0.1M | K=1 of the case | 0.6 | 0.3 | 3.2 | 0.14 |
| A | F7 | cavity2d_n4000 | 2 | 16.2M | K=1 of the case | 1.1 | 1.5 | 2.6 | 0.25 |
| A | F7 | cavity2d_n520 | 2 | 0.3M | K=1 of the case | 0.4 | 0.3 | 2.0 | 0.10 |
| A | F7 | cavity2d_n720 | 4 | 0.6M | K=1 of the case | 0.6 | 0.3 | 3.2 | 0.14 |
| A | F8 | cavity3d_n104_b9 | 2 | 1.8M | K=1 of the case | 0.4 | 0.5 | 2.1 | 0.12 |
| A | F8 | cavity3d_n200_b9 | 2 | 10.4M | K=1 of the case | 1.3 | 2.0 | 2.8 | 0.30 |
| A | F8 | cavity3d_n200_x96 | 2 | 4.5M | K=1 of the case | 0.7 | 1.0 | 2.3 | 0.17 |
| B | F1 | cavity2d_n11320 | 8 | 128.6M | 4 x K=2 pairs | 3.7 | 6.6 | 10.0 | 0.95 |
| B | F1 | cavity2d_n2840 | 8 | 8.2M | K=1 of the case | 1.0 | 0.9 | 5.9 | 0.27 |
| B | F1 | cavity2d_n5640 | 8 | 32.1M | K=1 of the case | 1.4 | 2.8 | 6.7 | 0.42 |
| B | F1 | cavity2d_n9000 | 8 | 81.4M | K=1 of the case | 2.5 | 6.6 | 8.4 | 0.78 |
| B | F1+F7 | cavity2d_n2000 | 8 | 4.1M | K=1 of the case | 1.0 | 0.6 | 5.8 | 0.24 |
| B | F1+F7 | cavity2d_n4000 | 8 | 16.2M | K=1 of the case | 1.1 | 1.5 | 6.2 | 0.32 |
| B | F1+F7 | cavity2d_n8000 | 8 | 64.4M | K=1 of the case | 2.1 | 5.3 | 7.8 | 0.65 |
| B | F3 | cavity2d_n1440_k8 | 8 | 16.9M | K=1 cavity2d_n1440 | 1.2 | 0.4 | 6.2 | 0.26 |
| B | F3 | cavity2d_n2000_k8 | 8 | 32.4M | K=1 cavity2d_n2000 | 1.4 | 0.6 | 6.7 | 0.31 |
| B | F3 | cavity2d_n2840_k8 | 8 | 65.1M | K=1 cavity2d_n2840 | 2.1 | 0.9 | 7.8 | 0.43 |
| B | F3 | cavity2d_n4000_k8 | 8 | 128.8M | K=1 cavity2d_n4000 | 3.7 | 1.5 | 10.0 | 0.69 |
| B | F3 | cavity2d_n5640_k8 | 8 | 255.6M | K=1 cavity2d_n5640 | 6.9 | 2.8 | 14.4 | 1.20 |
| B | F4+F5 | cavity3d_n160_k8 | 8 | 36.4M | K=1 cavity3d_n160 + K=1 of the case | 1.9 | 7.3 | 7.2 | 0.72 |
| B | F4+F5 | cavity3d_n200_k8 | 8 | 69.6M | K=1 cavity3d_n200 + K=1 of the case | 3.1 | 13.5 | 8.7 | 1.22 |
| B | F6 | cavity3d_n416 | 8 | 76.2M | K=1 of the case | 3.3 | 12.9 | 9.0 | 1.22 |
| B | F7 | cavity2d_n1000 | 8 | 1.0M | K=1 of the case | 0.9 | 0.3 | 5.7 | 0.22 |
| B | F7 | cavity2d_n520 | 8 | 0.3M | K=1 of the case | 0.9 | 0.3 | 5.7 | 0.22 |
