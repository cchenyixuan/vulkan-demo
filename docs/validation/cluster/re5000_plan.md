# Re = 5000 方腔：粗 → 2×5090 细 → 5090 + 7900 XTX 混跑

用户 2026-11-02 的安排。目的是给 5090 + 7900 XTX 这对卡做一个精度验证算例。做法沿用 Re 1000 / 3200：求解器 `experiment/v7` = tag v7-rc3 的树（f33a46ba，与 `vulkan-demo-xv` 相同），release 开关，adami 壁，ξ 0.001，ε² 0.0025 h²，U = 1，ν = 2e-4。

## 阶段

| 阶段 | 算例 | 设备 | 停止规则 | 检查点 | 估时 |
|---|---|---|---|---|---|
| 粗 | `lid_driven_cavity_2d_n1000_re5000_adami`，1,046,529 粒子，dt 7.5e-6，从 Re 3200 1000² 末态启动 | ZeroTier 的 5090（K = 1） | 窗口变化 < 1e-3 后再平均 20 t，最早 t = 60，上限 t = 300 | 全留，每 10 t；快照从 t = 40 起每 1 t | 已完成（11-03 02:09）：t = 49.61 达标，t = 69.61 停，约 590 步/秒，共约 4.4 h |
| 细 | `lid_driven_cavity_2d_n2500_re5000_adami`，6,365,529 粒子，dt 3e-6，从粗网格 t = 68.83 快照 warm 启动 | ZeroTier 的 5090（K = 1，用户 2026-11-03 定；11-03 02:11 起） | t = 80 起可换卡；不换就继续走完整判据（窗口变化 < 1e-3 后再平均 20 t，上限 t = 300） | 全留，每 2 t | 实测 95 步/秒 = 1.03 t/h：t = 80 约 78 h（11-06 上午）；若到 t ≈ 145 约 141 h（11-08 夜） |
| 混跑 | 同一细算例，`--start-from` 检查点 | 5090 + 7900 XTX | 与 2×5090 重叠 10 t | 无 | 约 6.6 h |

前两段由 `re5000_pipeline.py` 自动完成，第三段要手动换卡。估时来自外推（2026-11-02 的推算），不是实测。

## 为什么这样缩短（用户 2026-11-02：越快越好）

都来自已完成的 Re 3200 运行（1000² 冷启动 → 2000² warm）。数字由 `docs/validation/cluster/re3200_window_errors.py` 和 `re3200_coarse_vs_fine.py` 算出，在工作树根目录运行。粗网格的距离是对细网格尾部外推极限（Aitken）的距离。

- Re 3200 粗网格离细网格末态最近是在 t ≈ 135（窗口变化约 3.3e-3），距离 0.77 %；跑到稳态变成 1.17 %。粗网格的稳态越过了细网格的稳态（动能 54.2 对 53.0）。这是冷启动路径上的结果。用户 2026-11-02 决定粗网格跑到 1e-3 判据：这次从 Re 3200 末态启动，路径不同，3.3e-3 的早停点没有验证过。快照保留，细网格的起点事后还能选。
- Re 3200 与 Re 5000 的 Marchi 解相差 5.9 / 6.5 %，相当于粗网格冷启动已走完前段，所以从 Re 3200 末态启动。实测（窗口结束时刻，轮廓 / KE 变化）：t = 19.84 1.31e-2 / 1.07e-2，t = 29.77 2.79e-3 / 3.14e-3；Re 3200 冷启动到这个水平要 t ≈ 125–139（`n1000_k1_..._re3200/run.log`），这比较的是窗口变化，不是到稳态的距离。判据要求轮廓和 KE 都 < 1e-3，Re 3200 上限制项是 KE（轮廓 t = 178.6 达标，KE t = 208.4 才达标）。t ≈ 150 的上界是假定 t = 30 的变化已全是慢模态（τ ≈ 78）；t = 20–30 这一窗变化降了 4.7 倍，说明还有快模态，实际可能更早。结果：t = 49.61 达标（轮廓 5.91e-4，KE 6.68e-4），t = 69.61 停，早于 t ≈ 80–150 的估计；窗口 49.6–69.5 对 Marchi 的相对 L2 为 u 1.76 %、v 1.86 %。
- 细网格对 Marchi 的误差很快走平，后面是慢模态在收尾（τ = 48.7）：

| Re 3200 细网格窗口 | u / v 相对 L2 | 对应 Re 5000 时刻（× 1.6） | 到此的细网格耗时 |
|---|---|---|---|
| t 20–30 | 0.46 / 0.53 % | 48 | 25 h |
| t 30–40 | 0.40 / 0.46 % | 64 | 34 h |
| t 40–50 | 0.36 / 0.43 % | 80 | 42 h |
| t 50–60 | 0.35 / 0.40 % | 96 | 51 h |
| t 69.6–89.6（完整判据） | 0.35 / 0.38 % | 145 | 77 h |

没有直接放宽 `--steady-tol`：Re 5000 的 τ ≈ 78，同样的窗口变化对应 1.6 倍的剩余距离，放宽判据比 Re 3200 上看起来危险。所以用固定时刻 t = 80 提示换卡，同时让运行继续，晚换只会更准。× 1.6 是 τ 之比的外推，以运行中实际的窗口数据为准。

t = 80 不满足完整稳态判据。混跑与 2×5090 的逐点比较不受影响，因为两边是同一时间窗；对 Marchi 的误差要标明是 t = 80 附近的值。

检查点每 5 t 一个，就是为了能选混跑的起点。

## 第三张 5090（ZeroTier 机器，2026-11-02 起）

**2026-11-02 22:1x 起本机流水线已手动停止，只在这台机器上跑（用户决定）。** 要交给本机 K = 2 时，用下面的命令带 `--take-over` 重新启动本机流水线。不重启就不会自动交接，细网格留在这边单卡跑。用户 2026-11-03 确认细网格留在这台机器上继续跑。单卡模式不写 `ready_for_swap`，t = 80 没有提示，混跑的起点要从这台机器取检查点。

本机两张卡被占着时，粗网格和细网格先在 `Administrator@10.147.20.1`（9800X3D，单张 5090，Gen5 x16）上用 K = 1 跑，粗网格实测 592 步/秒。

- 位置 `D:/vulkan-demo-re5000`（打包自本工作树，物理 / 采样 / runner 哈希与本机相同；独立 venv，没动系统 Python）。进度在 `logs/validation/cavity_re5000_v7/` 下，内容与本机同名。
- 启动：`re5000_pipeline.py --single-card a85cce699d0fb7294c33d0a5981a4666`，用 WMI 建进程（与 SSH 会话无关），细网格每 2 t 一个检查点。那台机器重启后不会自动继续，要重新启动（已完成的阶段会跳过，未完成的 `--resume`）。
- 交接：本机流水线带 `--take-over` 运行。本机卡空出来后，它等远端做完 warm 文件，然后停掉远端，取回最新的细网格检查点，用 K = 2 `--start-from` 接着跑；远端还没有检查点时取 warm 文件。远端失败或连不上 30 分钟，退回本机自己跑粗网格。
- 密码只放在本机进程的环境变量 `REMOTE_PW` 里，不落盘。本机重启后要带着它用 `--take-over` 重新启动；不带的话流水线会在本机从头跑粗网格（已经交接过的除外，`pipeline_state/take_over.json` 记着起点）。
- 交接后细网格的 `samples/` 分在两台机器上，分析 t 早于交接点的部分要从远端取。

## 流水线

```
cd C:\Users\cchen\PycharmProjects\vulkan-demo-v7-rc3
..\vulkan-demo\.venv\Scripts\python.exe docs\validation\cluster\re5000_pipeline.py --check
```

- 启动（分离进程，关掉会话也继续）：
  `powershell -NoProfile -Command "$env:REMOTE_PW='<密码>'; Start-Process -FilePath 'C:\Users\cchen\PycharmProjects\vulkan-demo\.venv\Scripts\python.exe' -ArgumentList 'docs\validation\cluster\re5000_pipeline.py','--take-over' -WorkingDirectory 'C:\Users\cchen\PycharmProjects\vulkan-demo-v7-rc3' -WindowStyle Hidden"`
- 进度：`logs/validation/cavity_re5000_v7/pipeline.log`；运行内 `<run>/run.log` 每个窗口一行，`<run>/heartbeat.json` 有当前步和 fps。
- 等卡：显示卡利用率 ≤ 30 %，无显示卡 ≤ 5 % 且显存 ≤ 500 MiB，连续 20 分钟才开跑。
- 重启电脑后用同一条命令接着跑：跑完的阶段跳过，未完成的从最新检查点 `--resume`。只能有一个实例（`pipeline.pid`）。
- 崩溃自动 `--resume`，最多 6 次。不变量违反直接停，不重试；两次 3 分钟内失败也停。停止原因写在 `pipeline_state/STOPPED`。
- 跑的过程中不要改这个工作树的求解器和算例：`--resume` 会比较 physics 与 sampling 哈希，不同就拒绝。

## 换卡前先用两张 5090 推进度：`--boost`（用户 2026-11-03）

ZeroTier 机器上的单卡继续跑。本机两张 5090 空闲满 20 分钟后，取远端当时最新的检查点，用 K = 2 跑 `n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_from_k1`，每 1 t 一个检查点（约半小时一个），换卡关机时最多丢 1 t。第二天换上 XTX 后，`--mixed` 从这个运行最新的检查点接着跑（没有就取远端的）。这样逐点比较分两段：t 从起点到换卡这段是 2×5090 对 K = 1，换卡之后是混跑对 K = 1，混跑的起始状态带着前一段 2×5090 的差。

启动：同下面的命令，把 `'--mixed'` 换成 `'--boost'`。加 `'--start-at','2026-11-03T21:48'` 时，先睡到那个时刻，再只查 3 分钟空闲（用户说卡几点空出来时用）。2026-11-03 21:08 按用户要求就是这样挂上的（pid 13284）。

## 2026-11-04 20:17：放宽判稳阈值的选项 `--mixed-steady-tol`

集群（19:14 CST）hp_5090 只空 2 卡、没有 8 卡位，`--test-only` 估 11-18 才能开跑，不可用。于是加了 `--mixed-steady-tol TOL`：取 2×5090 `windows.jsonl` 里第一个两项都低于 TOL 的窗口，混跑从它的检查点出发，这个判定作为混跑的 `steady.json`（`pipeline_state/mixed_steady.json`），停在 t_s + `--mixed-average-span`。TOL = 2.5e-3 时 t 60.30 的窗口（1.21e-3 / 2.10e-3）已满足，可以立刻换卡：

`powershell -NoProfile -Command "Start-Process -FilePath 'C:\Users\cchen\PycharmProjects\vulkan-demo\.venv\Scripts\python.exe' -ArgumentList 'docs\validation\cluster\re5000_pipeline.py','--mixed','--mixed-average-span','5','--mixed-steady-tol','2.5e-3' -WorkingDirectory 'C:\Users\cchen\PycharmProjects\vulkan-demo-v7-rc3' -WindowStyle Hidden"`

ZeroTier 的 K = 1 照常跑到完整规则（1e-3 + 20 t），Re 5000 2500² 的收敛值由它给。

## 2026-11-04 20:05 定案：2×5090 判稳，换卡后混跑只平均 5 t

用户：收敛以后换异构跑几步就够了（小算例早已验证过等价），这次只是补个形式。于是：

- 2×5090 跑到判稳（t 100.5 或 110.55），判稳后再多跑约 1 h（约 2 t），这一段给混跑当同步号对照；然后关机换卡。
- 启动命令加 `'--mixed-average-span','5'`：混跑从判稳那一步的检查点出发，继承 `steady.json`，停在 t_s + 5（约 3.5 h），流水线的 Marchi 报告也按 5 t 的窗口算。
- 结果：判稳那一刻后约 5.5–6 h。下面"今晚"那一节的手工起点已撤掉（改名 `pipeline_state/mixed_start_tonight_unused.json`），但第 1–3 项的分析内容不变，只是重叠段换成判稳点之后、平均窗口换成 [t_s, t_s + 5]。

## 2026-11-04 晚（已撤回）：用户要求今晚出异构结果

收敛后 20 t 的平均今晚出不来（2×5090 上也要 10.5 h）。当时改成：

- 起点手工定在 2×5090 的 `c0019765000.npz`（t = 59.295），写进 `pipeline_state/mixed_start.json`（19:06，pipeline.log 有记录）。这样混跑和 2×5090 在 t = 59.295 到关机时刻（约 t 61.5–62.5）有一段同起点、同步号的重叠。
- 换卡后照常 `--mixed`：pilot 定权重，然后混跑一直跑到完整的停止规则（窗口比较由混跑接着做，带复制进来的样本），收敛结果约 11-06 下午到夜里。
- 今晚的结果（混跑走过关机时刻后就能算，CPU 即可）：
  1. 重叠段每个共同样本：混跑 − 2×5090 的差，相对 Marchi 范数的 u / v（同观察器 `l2_newest.csv` 的口径），28 个 Marchi 点和 1001 个稠密点上的最大 |Δu|、|Δv|。对照 2×5090 − K = 1 的同口径差 0.014–0.07 %。
  2. 混跑样本对 Marchi 2021 Re 5000：最新单样本和混跑段平均的 L2 u / v；稠密剖面上的 u_min、v_min、v_max 及位置对 Table 27（`docs/validation/data/marchi2021_re5000.csv`），2×5090 同步号的值并列。主涡 ψ_min 用快照（可选）。
  3. 粒子数守恒、overflow / far_migration 计数、切口列和权重（`meta.json`、`pipeline_state/mixed_weights.json`）。
  - 只用混跑起点之后（step > 19765000）的样本，复制进来的是 2×5090 的。

## 混跑：`--mixed`（2026-11-03 起，本机空出来就换卡）

ZeroTier 机器上的单卡细网格不停，作 K = 1 参照。本机从它最新的检查点接着跑，判据和细网格相同。两边的全局步号一致，所以每个样本都能逐点比较，不只是一个 10 t 的重叠窗口。混跑跑在前面，参照晚一些才覆盖到同样的时刻。

1. 关机，用 7900 XTX 换下一张 5090。开机后确认设备序：[0] 5090，[1] 7900 XTX，[2] 核显。
2. 启动（分离进程）：
   `powershell -NoProfile -Command "$env:REMOTE_PW='<密码>'; Start-Process -FilePath 'C:\Users\cchen\PycharmProjects\vulkan-demo\.venv\Scripts\python.exe' -ArgumentList 'docs\validation\cluster\re5000_pipeline.py','--mixed','--mixed-average-span','5' -WorkingDirectory 'C:\Users\cchen\PycharmProjects\vulkan-demo-v7-rc3' -WindowStyle Hidden"`
   设备 UUID 确认后，可以再加 `'--mixed-uuids','<5090>,<XTX>'`。
3. 起点（用户 2026-11-04：不上集群，判稳时再换卡）：
   - 2×5090 已经判稳时，取它判稳那一步的检查点（`steady.json` 的 `decided_at_step`，窗口终点正好是检查点），不取最新的。跑步器会复制 `steady.json`，停止步就是 t_s + 20，平均窗口里的样本全部来自混跑，换卡晚几个小时也不影响。
   - 还没判稳时，取 2×5090 最新的检查点，窗口比较在混跑里接着做；没有 2×5090 运行时取远端最新的。
   - 起点记在 `pipeline_state/mixed_start.json`，重启后不变。
4. 权重：论文 sec:slab 的 pilot（`experiment/v7/_run_v7_chain_bench.py --weights auto --calibrate-only`，每轮 200 + 300 帧，最多两轮），记录在 `pipeline_state/mixed_weights.json`，输出在 `pipeline_state/mixed_pilot.out`。pilot 的设备名不是 [5090, 7900 XTX] 就停下。原来的三组功比短跑已删掉。然后跑 `n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000`，每 2 t 一个检查点，崩溃自动 `--resume`。
5. 重启电脑后用同一条命令接着跑。起点和权重都已记下，不再需要密码。
   - 从检查点新起的运行，起点之前没有样本，跑步器会跳过它的头两次窗口比较。2×5090 这条从 t = 20.10 起步，t = 30.15 就没有比较，第一次比较要到 t = 40.20。
   - 所以混跑从 2×5090 的检查点起步时，流水线会先把 2×5090 运行在起点之前的样本复制进混跑目录（`seed_samples`，带 `--resume` 启动），混跑的第一次窗口比较就是完整的。
   - 分析混跑本身时，只取起点（`meta.json` 的 `start_from`）之后的样本。从判稳点起步时不再有窗口比较，复制进来的样本只是历史。
6. 速度未实测，估计 130–140 步/秒（1.4–1.5 t/h）。磁盘：检查点每个约 0.9 GB，加上 t = 20 起的快照，跑到 t ≈ 150 约需 60 GB，本机剩余 177 GB。

下面的旧步骤（t = 80 后从细网格的检查点只跑 10 t）只在不换卡、继续用两张 5090 时才用。

## 混跑（旧步骤：细网格结束后，手动）

`pipeline.log` 出现 `the 7900 XTX may be swapped in now`（同时生成 `pipeline_state/ready_for_swap`）后随时可以换。细网格不用等它结束，关机时它留下的检查点就够用。

1. 关机，换上 7900 XTX。Vulkan 设备序：[0] 5090，[1] 7900 XTX，[2] 核显。这之后 `--require-uuid` 的两张 5090 不再匹配，不要重启流水线的细网格阶段。
2. 选起点：从 ZeroTier 机器取最新的细网格检查点（每 2 t 一个，约 0.89 GB，按 38 MB/s 约半分钟），记它的时刻为 T0，重叠窗口是 [T0, T0 + 10]，`T = T0 + 10`。远端单卡运行不停，等它也跑过 T0 + 10，两边就有了共同窗口。建议 T0 ≈ 70–80（2026-11-05 夜到 11-06 上午）：Re 3200 细网格的单样本误差在 t = 40 / 50 离终值 +0.02 / +0.04 和 +0.00 / +0.02（u / v），按 × 1.6 对应 t ≈ 64 / 80；Re 5000 到 t = 15.7 的下降比 × 1.6 慢一些，所以取偏晚的一端。只做逐点比较的话，更早也可以，但对 Marchi 的误差要注明是哪个时刻的值。
3. 标定权重：`vulkan-demo-xv` 的 E2 / E3 里校准比值 1.98–2.05，先按 1.3333,0.6667，切口约在第 336 列（5090 约 423 万，XTX 约 213 万）。用 `--max-steps` 短跑看两边的相位时间再定。
4. 运行：

```
python -m experiment.validation.cavity_runner --case cases/lid_driven_cavity_2d_n2500_re5000_adami/case.yaml
    --run-dir logs/validation/cavity_re5000_v7/n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000
    --start-from <checkpoint> --slabs 2 --device-map 0,1 --weights 1.3333,0.6667 --reynolds 5000 --expect release --solver v7
    --t-end 40 --t-max <T> --average-span 20
```

环境变量同流水线的 `runner_environment()`。`--t-max <T>` 让混跑停在重叠窗口的末端。如果细网格已经走完完整判据，`--start-from` 会复制它的 `steady.json`，这时改用细网格的 `--t-max 300`，混跑就停在同一步。

## 验证

- 对照一：同一时间窗内，混跑与参照运行的 56 点平均速度差（参照运行是 ZeroTier 机器上的单卡 K = 1，没有接缝；混跑接缝在第 336 列附近），用 max |Δ|/σ 和接缝带 / 对照带，方法同 `adami_k8_report.py`（`--reynolds 5000 --k2 none`）。
- 对照二：与 Marchi 2021 Re 5000（`docs/validation/data/marchi2021_re5000.csv`，Tables 27–29，由 PDF 解析）的相对 L2 和极值。
- 2×5090 与混跑的接缝位置不同（第 253 列 / 第 336 列），都不在同一批点上，这对检验有利。

## 已知偏离

- 启动步修复（分支 `bootstrap-density-fix`）不在这棵树上，以便与 Re 1000 / 3200 和 `xv` 使用同一份求解器。测得的影响到 t = 0.03 已进入噪声底。
- `cavity_runner.py` 比 Re 3200 时多了 `--weights`，默认行为不变。
- 粗网格不是冷启动，这一点在报 Re 5000 1000² 的结果时要写明。它走的是同一稳态判据。细网格的稳态与初值无关。
- 粗网格运行的 `meta.json` 记着启动时的 `steady_tol` 0.0033、`average_span` 2。t = 9.92 起 `--resume` 用的是 1e-3 和 20（`--resume` 不改写 meta），以 `pipeline.log` 的命令行为准。
