"""Per-piece durations of the m = 8 split runs (user 2026-10-27 addendum) + the piece -> x-column mapping.

Pieces divide the live own range [0, N) evenly in particle order; the own buffer is voxel-sorted with x slowest, so piece k
covers x from the -x end (k = 0) to the +x end (k = 7). The lattice (e8_vram.lattice) gives the particle count of every
x-column (4 cells thick): the two end-wall columns (all wall) and the fluid columns (fluid 4 n^2 + lid 16 n + side walls);
the cumulative counts map a piece's index range to a column range; inside a partially covered column the composition is
attributed proportionally (the voxel order inside a column interleaves walls, lid and fluid). Durations: median over the
steady steps 100-299 of tick_i - tick_(i-1) from the runs' step traces (b_split_<k>_p<i>_end; the last piece ends at the
kernel's own tick). Reads split/runs/<cube>_<k>8t_r<r>/; writes piece_times.md / .json."""
import csv
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, "C:/Users/cchen/PycharmProjects/vulkan-demo-e8/docs/cluster_v6/scripts")
import e8_vram  # noqa: E402

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/split")
RUNS = ROOT / "runs"
CUBES = {"8m": ("cube 8M (200^3)", 192, 8_000_000), "64m": ("cube 64M (400^3)", 392, 64_000_000)}
KERNELS = {"cd": ("correction_density_interior", "b_start", "b_correction_density_interior_end"),
           "fo": ("force_deep_interior_scratch", "b_correction_density_interior_end", "b_force_deep_interior_end")}
M = 8
WARMUP = 100


def columns_of(n):
    lat = e8_vram.lattice(n, n)
    cols = []                                   # (label, count, fluid, lid, wall) in x order, empty entries dropped
    fluid_index = 0
    for count, fluid in zip(lat["columns"].tolist(), lat["fluid"].tolist()):
        if count == 0:
            continue
        if fluid == 0:
            cols.append(("end wall" if not cols else "end wall", count, 0, 0, count))
        else:
            fluid_index += 1
            lid = 16 * n
            cols.append((f"fluid column {fluid_index}", count, fluid, lid, count - fluid - lid))
    # name the two end walls distinctly
    cols[0] = ("-x end wall",) + cols[0][1:]
    cols[-1] = ("+x end wall",) + cols[-1][1:]
    assert sum(c[1] for c in cols) == lat["total"]
    return cols, lat


def piece_columns(n, live_n):
    cols, lat = columns_of(n)
    starts = []
    acc = 0
    for c in cols:
        starts.append(acc)
        acc += c[1]
    out = []
    for k in range(M):
        lo, hi = live_n * k // M, live_n * (k + 1) // M
        covered = []
        fluid = lid = wall = 0.0
        for index, c in enumerate(cols):
            c_lo, c_hi = starts[index], starts[index] + c[1]
            o_lo, o_hi = max(lo, c_lo), min(hi, c_hi)
            if o_hi <= o_lo:
                continue
            share = (o_hi - o_lo) / c[1]
            covered.append((index, c[0], share))
            fluid += c[2] * share; lid += c[3] * share; wall += c[4] * share
        first, last = covered[0], covered[-1]
        # x-column index: 0 = -x end wall, 1..F fluid columns, F + 1 = +x end wall; x cells = 4 per column
        out.append({"piece": k, "range": [lo, hi], "particles": hi - lo, "first_column": first[0], "first_share": round(first[2], 3),
                    "last_column": last[0], "last_share": round(last[2], 3), "columns_text": f"{first[1]} ({first[2] * 100:.0f} %) .. {last[1]} ({last[2] * 100:.0f} %)",
                    "n_columns": len(covered), "fluid": round(fluid), "lid": round(lid), "wall": round(wall),
                    "wall_share": wall / (hi - lo), "lid_share": lid / (hi - lo)})
    return out, cols


def trace_pieces(path, short, start_label, end_label):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labels = [start_label] + [f"b_split_{short}_p{i}_end" for i in range(M - 1)] + [end_label]
    missing = [l for l in labels if l not in rows[0]]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    per_piece = [[] for _ in range(M)]
    for r in rows:
        if int(r["step"]) < WARMUP:
            continue
        try:
            ticks = [int(r[l]) for l in labels]
        except ValueError:
            continue
        for i in range(M):
            per_piece[i].append((ticks[i + 1] - ticks[i]) / 1e6)
    return [statistics.median(v) for v in per_piece], [len(v) for v in per_piece]


result, lines = {}, []
for cube, (label, n, live_n) in CUBES.items():
    mapping, cols = piece_columns(n, live_n)
    result[cube] = {"mapping": mapping, "columns": [{"label": c[0], "count": c[1], "fluid": c[2], "lid": c[3], "wall": c[4]} for c in cols], "pieces": {}}
    lines += [f"### {label}: pieces (m = 8, by particle order = x from -x to +x; live N = {live_n:,}; {len(cols)} x-columns of 4 cells: "
              f"-x end wall, {len(cols) - 2} fluid columns, +x end wall; fluid column = {cols[1][1]:,} particles = fluid {cols[1][2]:,} + lid {cols[1][3]:,} + side walls {cols[1][4]:,}; "
              f"end wall column = {cols[0][1]:,})", ""]
    header = "| piece | particles | columns covered (x order) | fluid | lid | wall | wall share |"
    kernel_cols = ""
    pieces_ms = {}
    for short, (kname, start_label, end_label) in KERNELS.items():
        runs = sorted(p for p in RUNS.glob(f"{cube}_{short}8t_r*") if (p / "trace" / "steps_device.csv").exists())
        if not runs:
            continue
        meds = []
        for run in runs:
            med, counts = trace_pieces(run / "trace" / "steps_device.csv", short, start_label, end_label)
            meds.append(med)
        pieces_ms[short] = [statistics.mean(m[i] for m in meds) for i in range(M)]
        result[cube]["pieces"][short] = {"runs": [r.name for r in runs], "per_run_ms": meds, "mean_ms": pieces_ms[short]}
        kernel_cols += f" {kname} (ms, {len(runs)} runs) | per 1M particles (ms) |"
        header += f" {kname} (ms) | per 1M (ms) |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for p in mapping:
        row = (f"| {p['piece']} | {p['particles']:,} | {p['columns_text']} | {p['fluid']:,} | {p['lid']:,} | {p['wall']:,} | {p['wall_share'] * 100:.1f} % |")
        for short in KERNELS:
            if short in pieces_ms:
                ms = pieces_ms[short][p["piece"]]
                row += f" {ms:.3f} | {ms / p['particles'] * 1e6:.3f} |"
        lines.append(row)
    for short, (kname, _, _) in KERNELS.items():
        if short in pieces_ms:
            ms = pieces_ms[short]
            total = sum(ms)
            lines.append(f"- {kname}: pieces sum {total:.3f} ms; min {min(ms):.3f} (piece {ms.index(min(ms))}), max {max(ms):.3f} (piece {ms.index(max(ms))}), "
                         f"max / median {max(ms) / statistics.median(ms):.3f}; per-run medians agree within "
                         f"{max(abs(a - b) for a, b in zip(*result[cube]['pieces'][short]['per_run_ms'])) if len(result[cube]['pieces'][short]['per_run_ms']) > 1 else 0:.3f} ms")
    lines.append("")
text = "\n".join(lines)
(ROOT / "piece_times.md").write_text(text, encoding="utf-8")
json.dump(result, open(ROOT / "piece_times.json", "w", encoding="utf-8"), indent=1)
print(text)
