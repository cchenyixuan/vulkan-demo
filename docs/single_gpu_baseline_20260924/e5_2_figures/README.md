# E5.2 — paper figures from the single-GPU baseline campaign (no re-run)

Data: `logs/single_baseline_20260923_v5fix/results.jsonl` (post-fix rerun of 2026-09-24, 120 runs,
drift 0; mirrored in `docs/single_gpu_baseline_20260923/v5fix/`). The 1M-orig point is excluded;
the 1M point is `cases/lid_driven_cavity_2d_gen`.

| figure | file | content |
|---|---|---|
| A (Sec. 3.5, V0 only) | `manuscripts/fig/single_gpu_v0_throughput.pdf` (+ `.png`) | top: throughput, million particle-steps/s; bottom: fps (log). Two lines: 1 and 2 frames in flight. Log x, ticks 1M…32M (6M/10M/14M as unlabeled minor ticks). |
| B (end of Sec. 4) | `manuscripts/fig/v5_single_vs_v0.pdf` (+ `.png`) | V5 single-GPU mode / V0 fps ratio at the same in-flight depth (two lines); error bars = std of the trial-wise ratio over 3 interleaved trials. |

Typography: Times New Roman (STIX fallback), 9 pt, TrueType embedded (`pdf.fonttype 42`),
single-column width 3.5 in; PNG at 300 dpi.

Command (code: this commit; script `_plot_single_baseline_paper.py`, data aggregation shared with
`_summarize_single_baseline.py`):

```
.venv/Scripts/python.exe _plot_single_baseline_paper.py \
    --dir logs/single_baseline_20260923_v5fix --out-dir manuscripts/fig
```

Values plotted (mean over 3 trials; V0 sync / V0 2-in-flight throughput in M particle-steps/s,
V5/V0 ratio at 1 / 2 in flight in %):

| size | V0 d1 | V0 d2 | V5/V0 d1 | V5/V0 d2 |
|---|---|---|---|---|
| 1M | 546.2 | 571.3 | 100.2 ± 0.3 | 100.5 ± 0.1 |
| 2M | 551.7 | 566.6 | 100.9 ± 0.2 | 101.0 ± 0.1 |
| 4M | 542.7 | 551.2 | 100.1 ± 0.1 | 100.2 ± 0.1 |
| 6M | 569.6 | 576.4 | 101.8 ± 0.5 | 102.0 ± 0.4 |
| 8M | 567.5 | 572.8 | 101.2 ± 0.3 | 101.3 ± 0.4 |
| 10M | 582.3 | 588.3 | 101.8 ± 0.4 | 101.8 ± 0.0 |
| 14M | 558.2 | 561.9 | 100.8 ± 0.1 | 100.9 ± 0.1 |
| 16M | 572.7 | 575.2 | 101.1 ± 0.2 | 101.2 ± 0.2 |
| 32M | 586.8 | 589.2 | 101.3 ± 0.1 | 101.2 ± 0.1 |
