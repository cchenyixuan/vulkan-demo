# Re = 5000 方腔：粗 → 2×5090 细 → 5090 + 7900 XTX 混跑

用户 2026-11-02 的安排。目的是给 5090 + 7900 XTX 这对卡做一个精度验证算例。做法沿用 Re 1000 / 3200：求解器 `experiment/v7` = tag v7-rc3 的树（f33a46ba，与 `vulkan-demo-xv` 相同），release 开关，adami 壁，ξ 0.001，ε² 0.0025 h²，U = 1，ν = 2e-4。

## 阶段

| 阶段 | 算例 | 设备 | 停止规则 | 检查点 | 估时 |
|---|---|---|---|---|---|
| 粗 | `lid_driven_cavity_2d_n1000_re5000_adami`，1,046,529 粒子，dt 7.5e-6 | 2×5090 | 窗口变化 < 1e-3 后再平均 20 t；上限 t = 600 | 全留，每 10 t | t ≈ 330，约 15 h |
| 细 | `lid_driven_cavity_2d_n2500_re5000_adami`，6,365,529 粒子，dt 3e-6，从粗网格末态 warm 启动 | 2×5090 | 同上；最早 t = 40，上限 t = 300 | 全留，每 5 t | t ≈ 145，约 77 h（61–106 h） |
| 混跑 | 同一细算例，`--start-from` 检查点 | 5090 + 7900 XTX | 与细网格同一停止步 | 无 | 10 t 约 6.6 h |

前两段由 `re5000_pipeline.py` 自动完成，第三段要手动换卡。估时来自外推（2026-11-02 的推算），不是实测。

`--start-from` 会复制源运行的 `steady.json`，所以混跑停在细网格的同一步，能和 2×5090 在同一时间窗上逐点比较。检查点每 5 t 一个，就是为了能选混跑的起点。

## 流水线

```
cd C:\Users\cchen\PycharmProjects\vulkan-demo-v7-rc3
..\vulkan-demo\.venv\Scripts\python.exe docs\validation\cluster\re5000_pipeline.py --check
```

- 启动（分离进程，关掉会话也继续）：
  `powershell -NoProfile -Command "Start-Process -FilePath 'C:\Users\cchen\PycharmProjects\vulkan-demo\.venv\Scripts\python.exe' -ArgumentList 'docs\validation\cluster\re5000_pipeline.py' -WorkingDirectory 'C:\Users\cchen\PycharmProjects\vulkan-demo-v7-rc3' -WindowStyle Hidden"`
- 进度：`logs/validation/cavity_re5000_v7/pipeline.log`；运行内 `<run>/run.log` 每个窗口一行，`<run>/heartbeat.json` 有当前步和 fps。
- 等卡：显示卡利用率 ≤ 30 %，无显示卡 ≤ 5 % 且显存 ≤ 500 MiB，连续 20 分钟才开跑。
- 重启电脑后用同一条命令接着跑：跑完的阶段跳过，未完成的从最新检查点 `--resume`。只能有一个实例（`pipeline.pid`）。
- 崩溃自动 `--resume`，最多 6 次。不变量违反直接停，不重试；两次 3 分钟内失败也停。停止原因写在 `pipeline_state/STOPPED`。
- 跑的过程中不要改这个工作树的求解器和算例：`--resume` 会比较 physics 与 sampling 哈希，不同就拒绝。

## 混跑（细网格结束后，手动）

1. 关机，换上 7900 XTX。Vulkan 设备序：[0] 5090，[1] 7900 XTX，[2] 核显。这之后 `--require-uuid` 的两张 5090 不再匹配，不要重启流水线的细网格阶段。
2. 选起点：`<fine run>/checkpoints/` 里比停止时刻早 10 t 左右的检查点（按需 5 / 10 / 15 / 20 t）。
3. 标定权重：`vulkan-demo-xv` 的 E2 / E3 里校准比值 1.98–2.05，先按 1.3333,0.6667，切口约在第 336 列（5090 约 423 万，XTX 约 213 万）。用 `--max-steps` 短跑看两边的相位时间再定。
4. 运行：

```
python -m experiment.validation.cavity_runner --case cases/lid_driven_cavity_2d_n2500_re5000_adami/case.yaml
    --run-dir logs/validation/cavity_re5000_v7/n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000
    --start-from <checkpoint> --slabs 2 --device-map 0,1 --weights 1.3333,0.6667 --reynolds 5000 --expect release --solver v7
    --t-end 40 --t-max 300 --average-span 20
```

环境变量同流水线的 `runner_environment()`。`--t-end / --t-max / --average-span` 必须与细网格相同，否则停止步不同。

## 验证

- 对照一：同一时间窗内，混跑与 2×5090 的 56 点平均速度差，用 max |Δ|/σ 和接缝带 / 对照带，方法同 `adami_k8_report.py`（`--reynolds 5000 --k2 none`）。
- 对照二：与 Marchi 2021 Re 5000（`docs/validation/data/marchi2021_re5000.csv`，Tables 27–29，由 PDF 解析）的相对 L2 和极值。
- 2×5090 与混跑的接缝位置不同（第 253 列 / 第 336 列），都不在同一批点上，这对检验有利。

## 已知偏离

- 启动步修复（分支 `bootstrap-density-fix`）不在这棵树上，以便与 Re 1000 / 3200 和 `xv` 使用同一份求解器。测得的影响到 t = 0.03 已进入噪声底。
- `cavity_runner.py` 比 Re 3200 时多了 `--weights`，默认行为不变。
