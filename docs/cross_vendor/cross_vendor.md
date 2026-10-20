# S6.6 跨厂商实验报告（本机 RTX 5090 + RX 7900 XTX；v7-rc3 64eecf0；E0 冒烟与平台记录，2026-10-20）

> **范围**：你 10-20 的任务——S6.6 跨厂商实验，本机 RTX 5090 + RX 7900 XTX，E0 冒烟与平台记录、E1 单卡参考（520²–5640²）、E2 标定后的强扩展（K = 2）、E3 权重扫描（4000²，ω_5090/ω_XTX 1.4–2.6 步长 0.1）。**本版只做了 E0**，E1–E3 等你看过 E0 再继续。
> **版本**：rc3 = 标签 `v7-rc3` = 64eecf015942b25cfe460fc9ba75f30ba22f9009（E40：adami 壁面上链；没有更晚的标签）；experiment/v7 的 git 树 f33a46ba；工作树 `vulkan-demo-xv`（分支 `cross-vendor`，从标签建）与标签逐字节一致（`git diff v7-rc3 -- experiment/v7 shaders utils cases` 为空）。测量用的 harness 是 e7-cluster 473c178 的 `docs/cluster_v6/scripts/`（run_chain_v6.py 的 defrag 事件、起步屏障、构建阶段；parse_run_v6.py 的判定；trace 分析工具），原样拷进这个分支；本机的驱动脚本在 `docs/cross_vendor/scripts/`。
> **口径**（与集群一致）：每次运行 1500 步、两步在途（`--depth 2`，`--sync-scheme per-direction`，`--pool-safety 1.2`，切换间隔 0.2 ms），T 取第 500–1499 步的平均步长，不计 defrag 事件（`steady_fps_nodefrag`）；每个点 3 次试验；K = 1 参考在两张卡上同时跑、共用一个起步屏障（`--barrier`）；每次运行必须通过内建检查（粒子数不变、drift / overflow / far migration / stamp 错误 / init clamp 全 0、无 NaN），不通过就报告，不重跑丢弃。标定按 sec:slab 的规则 = `weight_calibration.py` 的规则：先等权重 pilot（暖机 200 + 计时 300 步），切口变了再用新权重跑一次 pilot，最多两轮，采用最后一轮的权重（`--calibrate-rounds 2`）。计时运行 `--no-seam-check`，trace 运行保留 seam 检查。
> **设备**：VulkanContextV7 的"独立显卡优先"顺序在本机是 [0] RTX 5090、[1] RX 7900 XTX、[2] Ryzen 核显（不用）；`--device-map 0,1`：slab 0（左）在 5090，slab 1（右）在 7900 XTX；权重从左到右 = (ω_5090, ω_XTX)。
> 只报数据，不下机制结论。

## 0. 概要

1. **E0 通过。** 1000²（`cavity2d_n1000`，1,044,484 粒子）K = 2 跨厂商链路，标定权重 ω = (1.2957, 0.7043)（ω_5090/ω_XTX = 1.840），切口在第 133 列（5090 / XTX 的流体粒子 650,000 / 350,000），trace 运行 1500/500 步：**863.5 fps（不计 defrag 869.8）**，粒子数 1,044,484 → 1,044,484，drift 0，overflow 全 0，far migration 0，stamp 错误 0+0，init clamp 0，无 NaN，场检查 OK，seam 检查 OK（第 133 列，越界 −0.03 / −0.36 dx，重复 0），dest guard fallback / released 0。step trace 两张卡都用 VK_KHR_calibrated_timestamps（QPC 时域），时钟拟合残差 rms 41 ns、最大 97 ns，1500 步的设备行和链路行全部完整。
2. **分相（trace，第 500–1499 步各自中位数，µs）**：5090（slab 0）A 36.7、B 822.4、C 158.4、B→C 等待 112.6、周期 1144.1；7900 XTX（slab 1）A 46.1、B 831.0、C 143.9、A→B 等待 84.9、周期 1141.7。两条链路：5090→XTX t_own 130.7（readback 20.4、主机步 71.8、upload 38.4），r_own 0.157；XTX→5090 t_own 148.4（32.2 / 82.1 / 34.2），r_own 0.180。
3. **平台**：Windows 11 Pro 10.0.26200；Ryzen 9 9950X（16C/32T）；1 × 32 GiB DDR5（G.Skill F5-6400，WMI 报 4800 MT/s）；ASUS ProArt X870E-Creator WiFi。RTX 5090：驱动 576.88（Windows 32.0.15.7688），Vulkan 1.4.303，**PCIe Gen5 x8**（上限 x16）；RX 7900 XTX：Adrenalin 26.6.1（Windows 驱动 32.0.31019.2002，Vulkan driverVersion 2.0.395），Vulkan 1.4.349，**PCIe Gen4 x16**（上限 Gen4 x16）；Vulkan loader 1.4.350；Python 3.13.9，python-vulkan 1.3.275.1，numpy 2.4.6。两张卡的链路代数与宽度在运行中读到的与空闲时相同。
4. **运行中的时钟与温度**（15000 步的 1000² K = 2 链路，0.25 s 采样，第 2 节表 1.4）：5090 SM 2670–2895 MHz（中位 2880）、显存 14001 MHz、核心 46–66 °C、功耗 59–518 W（中位 515）；7900 XTX GFXCLK 207–2997 MHz（中位 2986）、显存时钟 946–1538 MHz、边缘 47–60 °C、热点 54–83 °C；两个最小值都是循环起步的第一个样本。7900 XTX 的功耗没有可信的读数（第 2.3 节）。
5. **三次尝试才得到 E0 的数据，前两次没有上 GPU 或没有计时运行**：第 1 次在加载算例时退出（`cases/aligned/cavity2d_n1000` 的 .obj 粒子文件不在 git 里，本机没生成；从 e7 工作树拷来同一 case.yaml 的生成文件）；第 2 次的标定 pilot 本身完成了（权重 1.2974 / 0.7026），但我把 `--calibrate-only` 进程交给 parse_run_v6 判定，它没有计时运行的结尾行被判 hard_failure，驱动脚本停下（改成集群 `e7_calibrate` 的做法：只看退出码和权重文件）；第 3 次 = 本报告的 E0。两次的记录留在 `data/e0/attempt1_missing_obj/`、`attempt2_cal_parsed_as_run/`。时钟记录也跑了两次：第 1 次每 0.25 s 起一个 nvidia-smi 进程读数，5090 的时钟 / 功耗在循环的前 15 s 停在 600 MHz / 30 W（利用率、温度却在变），改成一个常驻的 `nvidia-smi -lms` 流（集群 harness 的做法）后读数正常；第 1 次的文件带 `_r1_stale_nvsmi` 后缀保留。

## 1. E0：1000² K = 2 跨厂商链路

### 1.1 运行与检查

| 运行 | 内容 | 状态 | fps（不计 defrag） | 粒子数 | drift / overflow / far / stamp / clamp / NaN | seam | 其它 |
|---|---|---|---|---|---|---|---|
| `cal_cavity2d_n1000_K2` | 标定 pilot × 2（暖机 200 + 计时 300 步） | calibrated（rc 0，权重文件已写） | pilot 671.1 / 813.5 | — | 两轮不变量全 0 | — | 4.7 s |
| `cavity2d_n1000_K2_trace` | K = 2，权重文件，1500/500，step trace，seam 检查 | **pass** | **863.5（869.8）** | 1,044,484 → 1,044,484 | 0 / 0 / 0 / 0+0 / 0 / 无 | 第 133 列 OK（越界 −0.03 / −0.36 dx，重复 0） | 构建 3.1 s，defrag 事件 1 次 8.4 ms，VRAM 峰值 1441 / 2018 MiB，dest guard 0/0，trace 1500 步 3000+3000 行完整 |
| `clocks_cavity2d_n1000_K2` | 同上、15000/500、无 trace、无 seam 检查（时钟记录，第 1 次采样有问题，数据保留） | pass | 847.4（851.8） | 不变 | 全 0 | — | defrag 事件 15 次 88.7 ms |
| `clocks_cavity2d_n1000_K2_r2` | 同上（时钟记录，流式采样） | pass | 846.1（850.7） | 不变 | 全 0 | — | 循环 17.7 s |

分区（bench 打印）：206 列（200 列流体 + 两侧各 3 列壁面），切口第 133 列 → 两 slab 133 / 73 列，粒子 675,542 / 368,942（其中流体 650,000 / 350,000），每方向 ghost 池 13,547。

### 1.2 标定（sec:slab 规则）

| 轮 | 权重（ω_5090, ω_XTX） | 切口 | 流体 5090 / XTX | busy p50 5090 / XTX（µs；A/B/C） | 得到的 ω | 新切口 | pilot fps |
|---|---|---|---|---|---|---|---|
| 1 | 1, 1 | 103 | 500,000 / 500,000 | 830.9（31.1 / 648.4 / 151.5）/ 1369.4（61.6 / 1162.3 / 145.4） | 1.2447 / 0.7553 | 127（变了） | 671.1 |
| 2 | 1.2447 / 0.7553 | 127 | 620,000 / 380,000 | 973.5（35.3 / 783.5 / 154.6）/ 1097.7（47.5 / 905.0 / 144.9） | **1.2957 / 0.7043** | 133（又变了；规则到此为止，采用第 2 轮的 ω） | 813.5 |

最终权重 1.2957151 / 0.7042849，ω_5090/ω_XTX = **1.840**；计时运行按它切在第 133 列。等权重时两卡的 busy 之比 XTX/5090 = 1.648（round 1）。权重文件 `data/e0/weights/cal_cavity2d_n1000_K2.json`（sha256 1506d66c…）。

### 1.3 trace：分相与链路（第 500–1499 步，各自中位数，µs）

| 卡 | slab | A | A→B | B（校正+密度 / 力） | B→C | C | C→下一 A | 周期（均值） |
|---|---|---|---|---|---|---|---|---|
| RTX 5090（dev 0） | 0，流体 650k | 36.7 | 4.2 | 822.4（402.4 / 419.8） | 112.6 | 158.4 | 9.6 | 1144.1（1156.9） |
| RX 7900 XTX（dev 1） | 1，流体 350k | 46.1 | 84.9 | 831.0（429.2 / 401.6） | 18.3 | 143.9 | 15.8 | 1141.7（1157.0） |

| 链路 | t_chain | t_own | readback | 主机步 | upload | r_own |
|---|---|---|---|---|---|---|
| 5090 → XTX（s0_to_s1） | 199.6 | 130.7 | 20.4 | 71.8 | 38.4 | 0.157 |
| XTX → 5090（s1_to_s0） | 325.4 | 148.4 | 32.2 | 82.1 | 34.2 | 0.180 |

口径与 E7 报告 4.4 节相同（`step_trace_model.link_metrics` + `e7_full_summary.transfer_queue_idle`：t_own = t_chain 减两个传输队列的空闲等待；readback / upload 同样减去各自的空闲；r_own = t_own / 接收卡的 T_B，按步算再取中位数）。1000 个稳态步每条链路都完整。

### 1.4 运行中的时钟与温度（`clocks_cavity2d_n1000_K2_r2`，循环窗口 17.7 s，0.25 s 采样）

| 卡 | 样本 | SM / GFX 时钟 MHz（最小 / 中位 / 最大） | 显存时钟 MHz | 温度 °C | 热点 °C | 功耗 W | 利用率 % | PCIe（代 × 宽） | 显存占用 MiB |
|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | 69 | 2670 / 2880 / 2895 | 14001 | 46 / 65 / 66 | — | 58.5 / 514.7 / 518.5 | 3 / 91 / 92 | Gen5 × 8（全程） | 1405 |
| RX 7900 XTX | 71 | 207 / 2986 / 2997 | 946 / 1538 / 1538 | 47 / 57 / 60 | 54 / 81 / 83 | 无读数 | 15 / 95 / 96 | Gen4 × 16（全程） | 2492 |

最小值（5090 2670 MHz、3 %，XTX 207 MHz、15 %）都来自循环起步的第一个样本。1500 步的正式 E0 运行只有 1.7 s，其中的单个样本：5090 2895 MHz / 326 W / 64 °C，XTX 3013 MHz / 48 °C（热点 72）。5090 的 nvidia-smi 节流原因全程 0x0。

## 2. 平台记录

### 2.1 主机

| 项 | 值 |
|---|---|
| 操作系统 | Microsoft Windows 11 Pro 10.0.26200（build 26200） |
| CPU | AMD Ryzen 9 9950X，16 核 / 32 线程，WMI 最大 4300 MHz |
| 内存 | 1 条 32 GiB DDR5，G.Skill F5-6400J3239G32G，WMI 报 4800 MT/s（配置 4800）；系统可见 31.1 GiB |
| 主板 | ASUSTeK ProArt X870E-CREATOR WIFI |
| 主机名 | DESKTOP-LDQTHT7 |

### 2.2 两张卡

| 项 | RTX 5090（Vulkan 设备 [0]） | RX 7900 XTX（Vulkan 设备 [1]） |
|---|---|---|
| PCI | 01:00.0（10de:2b85） | 05:00.0（1002:744c） |
| PCIe 链路（空闲 / 运行中） | **Gen5 × 8**（上限 Gen5 × 16）/ Gen5 × 8 | **Gen4 × 16**（上限 Gen4 × 16）/ Gen4 × 16 |
| Windows 驱动（WMI） | 32.0.15.7688，2025-06-24 | 32.0.31019.2002，2026-05-29 |
| 厂商驱动版本 | nvidia-smi 576.88；Vulkan driverInfo "576.88"，driverVersion 576.88.0.0 | Vulkan driverInfo "26.6.1 (LLPC)"（Adrenalin 26.6.1），driverVersion 2.0.395 |
| Vulkan API / 合规版本 | 1.4.303 / 1.4.1.0 | 1.4.349 / 1.4.3.3 |
| 设备内存 | 32607 MiB（device-local 堆 31.35 GiB） | 24576 MiB（device-local 堆 23.98 GiB） |
| 功耗上限 / 最高时钟 | 600 W / SM 3090 MHz、显存 14001 MHz（nvidia-smi） | 未读到（ADL，见 2.3） |
| timestampPeriod | 1 ns | 10 ns |
| 校准时间戳 | VK_KHR_calibrated_timestamps（trace 用 QPC 时域） | 同左 |
| 队列族 | 16×GCT，2×T，8×CT，2×T，2×T，1×T | 8×GCT，8×CT，1×T，1×，1× |
| 显示 | 桌面在 5090 上 | — |

核显 AMD Radeon(TM) Graphics（1002:13c0，Vulkan [2]，API 1.4.315）在机器里但不参与。

### 2.3 软件与读数方法

| 项 | 值 |
|---|---|
| Vulkan loader | 1.4.350（Vulkan SDK 1.4.350.0 装在 `C:/VulkanSDK/1.4.350.0`；运行时 `VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation`，bench 本身不开校验层；parse 记到的校验消息 0） |
| Python | 3.13.9（`vulkan-demo/.venv`） |
| python-vulkan | 1.3.275.1（包名 `vulkan`）；cffi 2.0.0；numpy 2.4.6 |
| 求解器 | 标签 v7-rc3 = 64eecf0；experiment/v7 树 f33a46ba（git）；开关（bench 打印）：V7_FUSED_CORRECTION_DENSITY=1、V7_DEST_GUARD=relay、V7_DEST_GUARD_PRECHECK=counter、V7_DENSITY_COPY_COMPUTE=1、V7_GHOST_SEND_LANES=32、band 2,2,3、cascade_force=1、phase_a_no_wait=1、split_transfer_queues=1、worker_count_aware=1；V7_WORKER_AFFINITY 未设 |
| harness | e7-cluster 473c178 的 `docs/cluster_v6/scripts/`（run_chain_v6.py、parse_run_v6.py、e7_full_summary.py、e7_late_wake.py 等） |
| 5090 遥测 | 一个常驻 `nvidia-smi --query-gpu=... -lms 250` 流（`xv_telemetry.py`）；每样本起新进程的读法在本机会给出滞后约 15 s 的时钟 / 功耗（概要第 5 条） |
| 7900 XTX 遥测 | ADL（`atiadlxx.dll`，`ADL2_New_QueryPMLogData_Get`）：GFXCLK（id 1）、MEMCLK（2）、边缘温度（8）、热点（27）、GFX 活动（19）；显存占用 `ADL2_Adapter_VRAMUsage_Get`。**功耗没有可信读数**：负载下候选 id 40 恒为 3、id 73 恒为 403，都不是功耗；全部受支持的 id 与值在 `data/e0/adl_all_sensors_under_load.txt` |
| PCIe 链路 | 两张卡都从 PnP 设备属性（DEVPKEY_PciDevice_CurrentLinkSpeed/Width）读，运行中再读一次；5090 另有 nvidia-smi 每样本一个值 |

## 3. E1–E3（等你确认后做）

- **算例**：`cases/aligned/cavity2d_n{520,1000,1440,2000,2840,4000,5640}`（case.yaml 都在 git 里；.obj 粒子文件只有 n1000、n1440 在本机有现成的，其余四个要用各自 `generate.txt` 里的生成行生成，5640² 的 domain.obj 约 1.3 GB）。
- **E1**：7 个尺寸 × 3 次试验，每次两张卡同时各跑一个 K = 1（`xv_lib.Experiment.references`，一个屏障）。5640²（31.8M 粒子）按集群 64.35M K = 1 峰值 21.5 GiB 折算约 10.6 GiB，7900 XTX 的 23.98 GiB 应该放得下；跑前用同样的方式确认，放不下就报告跳过。
- **E2**：7 个尺寸，每个先标定（1.2 节的规则，记两轮权重），再 3 次计时运行 + 1 次 trace 运行；η = τ̄/(2T)，τ̄ = 同一 trial 两张卡 E1 步长的调和平均；输出项按你的清单。
- **E3**：4000²（如 E2 的 r_own ≥ 1 则改用 r_own < 0.5 的最小尺寸并告诉你），ω_5090/ω_XTX = 1.4, 1.5, …, 2.6（13 个点）× 3 次试验，不标定。
- 估计：E1 约 21 组参考、E2 约 35 次运行、E3 39 次运行；大算例以 5640² 为主（本机 2-D 32M 的 K = 1 在 5090 上约 18 fps、在 XTX 上更慢，1500 步各约 1.5–3 min），总计约 2–3 h 墙钟，不占集群。

## 4. 数据与复现

- 报告 `docs/cross_vendor/cross_vendor.md`；数据 `docs/cross_vendor/data/e0/`：`platform.json`（平台记录的原始读数）、`summary.json`（E0 的运行、标定、trace、时钟摘要）、`results.jsonl` + `index.tsv`（parse_run_v6 的行）、`logs/<label>.{log,cmd}`、`weights/cal_cavity2d_n1000_K2.json`、`traces/cavity2d_n1000_K2_trace/`、`telemetry.csv`（E0 运行，1 Hz）、`telemetry_clocks_r2.csv` + `clocks_summary_r2.json`（时钟记录）、`adl_all_sensors_under_load.txt`、失败尝试 `attempt1_missing_obj/`、`attempt2_cal_parsed_as_run/`、`*_r1_stale_nvsmi.*`。
- 脚本 `docs/cross_vendor/scripts/`：`xv_lib.py`（协议：一次运行 = run_chain_v6 --solver v7 + parse_run_v6；calibrate / chain / references）、`xv_e0.py`（E0）、`xv_e0_clocks.py`（时钟记录）、`xv_telemetry.py`（两卡遥测）、`xv_probe_devices.py`（Vulkan 设备探测）。
- 复现（工作树根，`vulkan-demo/.venv` 的 python）：`python docs/cross_vendor/scripts/xv_e0.py`；`python docs/cross_vendor/scripts/xv_e0_clocks.py --suffix _r2`。
