# 3-D matrix on N56 — job script and budget (2026-09-17, awaiting confirmation; nothing submitted)

Script: `scripts/probe37_3d_matrix.sbatch` (one 8-GPU job, `#SBATCH --time=03:15:00`). Code: freeze commit
**eda4b8f** (tag `freeze-2026-09-17`; the script aborts if `git rev-parse` differs). Cases: border-4 families
from `scripts/gen_3d_families.sh` (default `BORDER=4`; the script aborts if the `.cavity3d_*_ready` markers are
missing).

Protocol: chain bench **1500 steps, warmup 500, steady = last 1000**; reference and K-run share the window;
**one 8-GPU simultaneous K=1 reference per N per job**, K-runs **×3**; `--anatomy` on every run;
`V5_GHOST_POOL_FACTOR=1.0`; `--pool-safety 1.2`; switches otherwise identical to the 2-D curve jobs
(cascade, band dispatch, fast submit, host stack, NUMA-pinned workers). η_strong(N, K) = fps_K / (K · mean of
the reference fps over GPUs 0..K−1); η_weak(K) = fps_K(K·n) / mean of the K=1 reference over GPUs 0..K−1.

Order: strong 64M stretched → weak 8M/GPU → strong 32M stretched → weak 4M/GPU → cube 64M K=8 control.
The K=8 runs of the strong blocks double as the weak families' K=8 points (same case, same K), so the weak
blocks run K=2 and K=4 only.

## Rates used for the estimate

Local 2× RTX 5090, border 4, after the bootstrap defrag: 3-D 8M K=1 12.4 fps (9.13M particles incl. walls →
8.8 ns per particle per frame), 64M stretched K=1 1.7 fps (70.6M), K=2 at 4M/GPU 24.0 fps. N56 5090s were
≈ 1.1× the local card on 2-D 4M (137–141 vs 122 fps) → **8.0 ns per particle per frame per GPU** here; K-runs
at 8M/GPU and below are taken at the same per-GPU rate (η ≈ 90–95% would add ≤ 10%). Case load (obj parse +
upload + bootstrap): 39 s for the 64M case locally → 2.0 min budgeted at 64M-class sizes on JuiceFS, 1.0 min at
32M, 0.5 min at ≤ 16M, plus 0.5 min process start / readback per run.

## Budget

| block | case (particles incl. walls) | reference (8 × K=1 at once) | K-runs (×3 each) | block |
|---|---|---|---|---|
| 1 strong 64M stretched | cavity3d_weak8_k8_64m (70.6M) | 1500 / 1.77 fps = 14.1 min + 2.5 = **16.6** | K=2 35.3M/GPU 3.5 fps 7.1 + 2.5 = 9.6 ×3 = 28.8; K=4 7.1 fps 3.5 + 2.5 = 6.0 ×3 = 18.0; K=8 14 fps 1.8 + 2.5 = 4.3 ×3 = 12.9 | **76** |
| 2 weak 8M/GPU | k1 9.1M, k2 18.2M, k4 36.4M (k8 = block 1) | 1500 / 13.7 = 1.8 + 1.0 = **2.8** | K=2 1.8 + 1.0 = 2.8 ×3 = 8.4; K=4 1.8 + 1.5 = 3.3 ×3 = 9.9 | **21** |
| 3 strong 32M stretched | cavity3d_weak4_k8_32m (≈ 35.2M) | 1500 / 3.55 = 7.0 + 1.5 = **8.5** | K=2 7.1 fps 3.5 + 1.5 = 5.0 ×3 = 15.0; K=4 14.2 fps 1.8 + 1.5 = 3.3 ×3 = 9.9; K=8 28 fps 0.9 + 1.5 = 2.4 ×3 = 7.2 | **41** |
| 4 weak 4M/GPU | k1 4.6M, k2 9.1M, k4 18.2M (k8 = block 3) | 1500 / 27 = 0.9 + 0.5 = **1.4** | K=2 0.9 + 0.5 = 1.4 ×3 = 4.2; K=4 0.9 + 0.8 = 1.7 ×3 = 5.1 | **11** |
| 5 cube 64M K=8 control | cavity3d_cube_64m (≈ 68.4M) | none (fps + anatomy vs block 1's K=8) | K=8 8.55M/GPU, 64% of each slab in the force band at 1.4× → ≈ 11 fps: 2.3 + 2.5 = 4.8 ×3 | **14** |
| **total** | | | | **≈ 163 min ≈ 2.7 h** |

Against the ≤ 2.5 h target the estimate is ≈ 10 min over. Knobs, in order of least information lost:
1. 64M K=2 trials ×3 → ×2 (−9.6 min → 2.55 h): K=2 at 35M/GPU was the tightest point of the 2-D sweep (±0.5%).
2. cube control ×3 → ×2 (−4.8 min).
3. `--time` stays 3:15 either way so a slow load or a failed run does not lose the job; every RESULT is copied to
   `~/run/logs/probe37_<job>/` as it completes.
Not touched: steps (1500), the one-reference-per-N rule, K=4/K=8 trials, anatomy.

If the measured per-GPU rate on N56 is the local one (no 1.1× advantage), every time above grows by 10% →
≈ 3.0 h; the 3:15 limit still covers it.

## Case generation (login node, before the job; not started)

`~/run/tools/gen_3d_families.sh` (upload the border-4 version first). Local generation times: 8M-class 2 min,
64M-class 10 min; login node ≈ 1.5× slower (2-D families took 57 min there for 4 + 8 + 32M/GPU sets):

| family | cases | est. login-node time | obj size |
|---|---|---|---|
| weak4 (4M/GPU) | k1 4.6M, k2 9.1M, k4 18.2M, k8 35.2M | 1 + 2 + 4 + 8 ≈ 15 min | ≈ 2.2 GB |
| weak8 (8M/GPU) | k1 9.1M, k2 18.2M, k4 36.4M, k8 70.6M | 2 + 4 + 8 + 15 ≈ 29 min | ≈ 4.2 GB |
| cube 64M | 401³ ≈ 68.4M | ≈ 15 min | ≈ 2.0 GB |

≈ 1 h sequential (≈ 35 min with two families in parallel), ≈ 8.5 GB under `~/run/vulkan-demo/cases/`
(large files are fine on JuiceFS; it is the small-file creates that are slow). Markers
`cases/.cavity3d_weak4_ready`, `.cavity3d_weak8_ready`, `.cavity3d_cube64_ready` gate the job.

## Outputs the job leaves behind

`~/run/logs/probe37_<job>.out` (RESULT lines: label, fps, drift, stamps, particles, wall), per-run logs with
anatomy at f500 / f1000 / f1500, `telemetry.csv`, `nvidia-smi topo -m`, the commit hash and the full `V5_*`
environment. Analysis: adapt `plot_ksweep36.py` / `plot_weak35.py` to the `strong64m_*` / `weak8_*` /
`strong32m_*` / `weak4_*` / `cube64m_*` labels (to be written after the data is in).
