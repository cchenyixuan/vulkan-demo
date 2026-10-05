# two-hop 跨 GPU 传输实验(共享 host 内存,去掉 worker memcpy)

本目录是实验目录 `experiment/two_hop_experiment/`(2026-09-28,当时未入库)的原样拷贝,只改了包路径(见"放进本分支时的改动")。实验报告见 [`REPORT.md`](REPORT.md),即原来的 README,内容未改;它里面的命令仍写旧路径,复现请用本文的命令。实验只用 v5 求解器,没有移植到 v6。

## 基于哪个提交

- 实验在提交 `21e74e0` 上运行(REPORT.md 开头:当时 `git status experiment/v5` 为空,`experiment/v5/` 与所有 shader 零改动)。
- `21e74e0` 的 `experiment/v5/` = freeze `eda4b8f`(2026-09-17,"v5: freeze — V5_CASCADE_FORCE and V5_BAND_VOXEL_DISPATCH default ON")加上 2026-09-23 的 `4686348` 与 `f9c660e`(邻居循环把 y 放在最内层;改了 correction / density / force 的源码与 SPIR-V 和 `simulator_v5.py`,共 7 个文件)。本分支已提交的 `experiment/v5/` 与 `21e74e0` 完全相同。
- 算例:2-D 方腔 1M–16M 用库里的 `cases/lid_driven_cavity_2d*`;10k–500k 由 `utils/geometry/_demo_cavity_case.py` 生成(REPORT.md §12.8)。注意 `f2b6d67`(2026-10-05)把库里 case.yaml 的 KCG ξ 从 0.1 改成 0.001,v5 会读这个值;要按原条件复现,用 `21e74e0` 的算例(见下面的 worktree 做法)。

## 硬件与软件

2 × RTX 5090,PCIe x8/x8(riser 分叉),NVIDIA 驱动 576.88,Windows 11,Vulkan SDK 1.4.350,Python 3.13(主工作区的 `.venv`)。共享 host 内存用 `VirtualAlloc` 加 `VK_EXT_external_memory_host` 导入,只支持 Windows。

## 复现

在 `21e74e0` 上开一个干净的 worktree(求解器与算例都是实验时的版本,也避开主工作区里 `experiment/v5/` 的未提交改动),再从本分支取出这个目录:

```bash
git worktree add --detach ../vulkan-demo-two-hop 21e74e0
git -C ../vulkan-demo-two-hop checkout v4-multigpu-orchestration -- experiment/two_hop
cd ../vulkan-demo-two-hop
PY=../vulkan-demo/.venv/Scripts/python.exe
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation

# Step 0:共享 host 内存探针(两张卡导入同一块 host 内存、双向逐字节、DMA 速率)
V5_SPLIT_TRANSFER_QUEUES=1 $PY experiment/two_hop/probe_shared_host.py

# 主矩阵:1M–16M,two-hop 对 three-hop,3 次交错 trial
$PY experiment/two_hop/_run_two_hop_campaign.py --sizes 1m,2m,4m,8m,16m --trials 3

# 单次运行
$PY experiment/two_hop/_run_two_hop_bench.py --transport two_hop --case cases/lid_driven_cavity_2d/case.yaml

# 汇总(campaign 默认写到 logs/two_hop_experiment/<组>_<时间>/)
$PY experiment/two_hop/_summarize_two_hop.py <campaign 目录>

# 数值等价性(包络法)
$PY experiment/two_hop/_run_two_hop_equivalence.py --steps 2000

# 第二轮(10k–500k,REPORT.md §12):先生成算例,再跑主配置;cascade off 再加 --environment V5_CASCADE_FORCE=0
cases=logs/two_hop_experiment/cases
for spec in 10k:50 20k:71 50k:112 100k:160 200k:223 500k:354; do
  $PY utils/geometry/_demo_cavity_case.py --half ${spec##*:} --out $cases/cavity_2d_${spec%%:*} --no-preview
done
$PY experiment/two_hop/_run_two_hop_campaign.py --sizes "" \
  --case 10k=$cases/cavity_2d_10k/case.yaml --steps 10k=3000:30000 \
  --case 100k=$cases/cavity_2d_100k/case.yaml --steps 100k=3000:21000 \
  --trials 5 --loop-trace --switch-interval-ms 0.2 --group low_load_main
$PY experiment/two_hop/_summarize_low_load.py --main <主配置目录> --cascade-off <目录>[,<目录>] --extra depth3=<目录>
```

每组的完整参数(算例、步数、开关、trial 数)在 `results/<组>/campaign.json`。

## 放进本分支时的改动

- 包路径与脚本路径:`experiment.two_hop_experiment` → `experiment.two_hop`,`experiment/two_hop_experiment/` → `experiment/two_hop/`,共 11 行。其余代码逐字节不变。
- 原 `README.md` 改名为 `REPORT.md`,内容不变。
- 输出目录的默认值 `logs/two_hop_experiment/` 没有改(已有的运行数据在那里,`logs/` 不入库)。
- `results/` 原样拷贝(各组的 `campaign.json`、`results.jsonl`、汇总表);没有拷贝 `__pycache__/`。

## 文件

| 文件 | 内容 |
|---|---|
| `shared_host_v5.py` | 共享 host 内存的 staging 与 two-hop 传输(v5 orchestrator 的替换部件) |
| `probe_shared_host.py` | Step 0 探针 |
| `_run_two_hop_bench.py` | 单次运行(three-hop 或 two-hop,可选 loop trace) |
| `_run_two_hop_campaign.py` | 交错 trial 的 campaign |
| `_run_two_hop_equivalence.py` | 两种传输的数值等价性(包络法:K = 1 参照与重跑、three-hop、two-hop) |
| `_summarize_two_hop.py`、`_summarize_low_load.py` | 主矩阵与低负载组(10k–1M)的汇总表 |
| `results/` | 各组的结果与汇总 |
| `REPORT.md` | 实验报告(原 README) |
