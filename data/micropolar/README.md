# data/micropolar

Results behind `docs/micropolar_design.md` (sections 8 to 10), copied from the untracked `logs/micropolar/` on
2026-11-11. Produced by `experiment/validation/micropolar_tests.py` on one RTX 5090 (device 0) and one RX 7900 XTX
(device 1). Top-level `data/` did not exist on this branch; the earlier result sets live under
`docs/cluster_v6/data/` (e.g. `e9`).

| path | content |
|---|---|
| `stage1_2/` | T3 operators (json + printed lines), T4 relaxation, the K = 2 per-particle seam check, advection across the cut, restart; `seam_layout_base.log` = the 858-failure reference of `_test_seam_layout.py` on `micropolar-base` |
| `t1/` | T1 / T2 comparisons on sample points: `k1_compare.json` (10 base repeats), `k2_compare.json` / `.log` (40 base repeats) |
| `t5/` | the four cavity runs to t = 40.2 (centreline samples every 5,000 steps + run.json) and their pairwise comparisons |
| `checks/` | checks A (label), B (K = 3 seam), C (factors), E (3-D operators): json + printed lines |
| `checks/d/` | check D: `k1_source` (K = 1 from rest to t = 40.2), `k1_t40.npz` (its final step-boundary state, eleven restart fields, 74,529 particles), the fifteen continuations, `paired.json` / `paired.log` |

Not copied (large, still in `logs/micropolar/` of the worktree `vulkan-demo-v9-micropolar`): the particle dumps of
T1 / T2 (`logs/micropolar/t1/*.npz`, 323 MB) and the full run logs.

Added 2026-11-11 (night):

| path | content |
|---|---|
| `viscous/operators_host.md` / `.json` | the host comparison of the viscous Laplacian discretizations (`micropolar_tests.py operators-host`) |
| `viscous/cross_check.json`, `seam_laplacian_k{2,3}.json`, `t1_k{1,2}_compare.json` | shader against host for both viscous forms; seam checks with `laplacian`; T1 re-run |
| `viscous/cavity_n{250,500}.json`, `viscous/cavity_runs/` | the Re = 1000 cavity with both forms (summaries; per run samples.jsonl, meta.json, steady.json) |
| `checks/a_walls_differ.txt` | check A: simple against adami near the walls |
| `checks/d/k1_{6..10}`, `k2s_{6..10}`, `k1lap_1`, `paired10.*` | check D extended to ten runs per group; the micropolar continuation with `laplacian` |

Not copied: the cavity runs' profile samples, snapshots and checkpoints (`logs/micropolar/cavity_forms/`, 6 runs).
