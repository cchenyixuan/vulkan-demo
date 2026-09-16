"""Inter-GPU phase-offset trajectory from a chain-bench --phase-trace directory.

usage: plot_phase_trace.py <trace_dir> <out_png> [--warmup N] [--title TEXT]

Reads phase_trace.csv (per frame, per sim: GPU timestamps of phase A/B/C start and
end, device-clock ns) and calibration.csv ((device_ns, QPC, perf_counter_ns) pairs per
sim) written by experiment/v5/utils/phase_trace_v5.py. One linear map per sim puts
every GPU on the host perf_counter clock; the figure shows, frame by frame:
  1. phase A start of sim i minus sim 0 (ms), against the frame period T
  2. the same as a phase fraction (Δ mod T) / T
  3. per-sim waiting: b->c gap (phase C waited for the neighbour's upload) and
     c->a gap (queue ran dry) in ms
  4. per-sim frame period (rolling median)
Frames whose A/B ticks were not caught use c_end(n-1) as phase A start (flagged).
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("trace_dir")
parser.add_argument("out_png")
parser.add_argument("--warmup", type=int, default=1000, help="first frame used in the statistics")
parser.add_argument("--title", default="")
args = parser.parse_args()
trace_dir = pathlib.Path(args.trace_dir)

# ---------------------------------------------------------------- calibration
samples = {}
with open(trace_dir / "calibration.csv") as handle:
    for row in csv.DictReader(handle):
        samples.setdefault(int(row["sim"]), []).append(
            (float(row["device_ns"]), int(row["qpc_ticks"]), int(row["perf_ns"]), float(row["max_deviation_ns"])))
maps = {}
for sim, rows in sorted(samples.items()):
    device = np.array([r[0] for r in rows]); perf = np.array([r[2] for r in rows], dtype=np.float64)
    d0, p0 = device[0], perf[0]
    slope, offset = np.polyfit(device - d0, perf - p0, 1) if len(rows) > 1 else (1.0, 0.0)
    residual = (perf - p0) - (slope * (device - d0) + offset)
    maps[sim] = (slope, offset, d0, p0)
    print(f"sim{sim}: {len(rows)} calibration samples, slope {slope:.9f} (ppm {1e6 * (slope - 1):+.1f}), "
          f"fit residual rms {np.sqrt(np.mean(residual ** 2)) / 1e3:.1f} us, max deviation reported {max(r[3] for r in rows) / 1e3:.1f} us")

def to_host_ms(sim, device_ns):
    slope, offset, d0, p0 = maps[sim]
    return (slope * (device_ns - d0) + offset + p0) / 1e6

# ---------------------------------------------------------------------- trace
fields = ("a_start", "a_end", "b_start", "b_end", "c_start", "c_end", "prev_c_end")
data = {}   # sim -> frame -> {field: host_ms}
with open(trace_dir / "phase_trace.csv") as handle:
    for row in csv.DictReader(handle):
        sim, frame = int(row["sim"]), int(row["frame"])
        entry = {}
        for f in fields:
            # a zero tick (query reported available but unwritten — seen once in
            # ~24k reads) is treated as missing
            if row[f] != "" and float(row[f]) > 1e6:
                entry[f] = to_host_ms(sim, float(row[f]))
        data.setdefault(sim, {})[frame] = entry
sims = sorted(data)
frames_all = sorted(set.intersection(*(set(data[s]) for s in sims)))
frames = np.array([f for f in frames_all if f >= 1])
print(f"sims {sims}, frames {frames[0]}..{frames[-1]} ({len(frames)})")

def a_start_series(sim):
    """phase A start per frame (host ms) + flag whether it is the real a_start tick
    or the fallback c_end(n-1)."""
    out = np.full(len(frames), np.nan); real = np.zeros(len(frames), dtype=bool)
    for k, n in enumerate(frames):
        e = data[sim][n]
        if "a_start" in e:
            out[k] = e["a_start"]; real[k] = True
        elif "prev_c_end" in e:
            out[k] = e["prev_c_end"]
        elif (n - 1) in data[sim] and "c_end" in data[sim][n - 1]:
            out[k] = data[sim][n - 1]["c_end"]
    return out, real

def series(sim, field):
    return np.array([data[sim][n].get(field, np.nan) for n in frames])

A = {s: a_start_series(s) for s in sims}
c_start = {s: series(s, "c_start") for s in sims}
c_end = {s: series(s, "c_end") for s in sims}
b_end = {s: series(s, "b_end") for s in sims}
a_end = {s: series(s, "a_end") for s in sims}
period = {s: np.append(np.diff(A[s][0]), np.nan) for s in sims}
steady = frames >= args.warmup
T = np.nanmedian(np.concatenate([period[s][steady] for s in sims]))
print(f"median frame period T = {T:.3f} ms (steady frames >= {args.warmup})")
for s in sims:
    caught = A[s][1]
    print(f"sim{s}: a_start caught on {100 * caught.mean():.1f}% of frames; period median {np.nanmedian(period[s][steady]):.3f} ms, "
          f"p10-p90 {np.nanpercentile(period[s][steady], 10):.3f}-{np.nanpercentile(period[s][steady], 90):.3f}")

delta = {s: A[s][0] - A[sims[0]][0] for s in sims[1:]}
# Per-frame waiting inside the frame. The A/B slots are reset by the next frame's
# phase A, so b_end is rarely captured; the robust per-frame proxy for the b->c
# wait is the excess of the A-start -> C-start span over its steady-state floor
# (A + B durations are constant to ~1%; the floor = 2nd percentile of the span).
span = {s: c_start[s] - A[s][0] for s in sims}
floor = {s: np.nanpercentile(span[s][steady], 2) for s in sims}
gap_bc = {s: span[s] - floor[s] for s in sims}
gap_ca = {s: np.array([data[s][n]["a_start"] - data[s][n]["prev_c_end"] if ("a_start" in data[s][n] and "prev_c_end" in data[s][n]) else np.nan for n in frames]) for s in sims}
for s in sims[1:]:
    d = delta[s][steady]; d = d[~np.isnan(d)]
    phase = np.mod(d, T) / T
    slope = np.polyfit(frames[steady][~np.isnan(delta[s][steady])], d, 1)[0] if len(d) > 2 else float("nan")
    print(f"sim{s} - sim0 phase A start: mean {d.mean():+.3f} ms, std {d.std():.3f}, min {d.min():+.3f}, max {d.max():+.3f}; "
          f"phase fraction mean {phase.mean():.3f} (std {phase.std():.3f}); drift {1000 * slope:+.3f} ms per 1000 frames")
for s in sims:
    g = gap_bc[s][steady]; g = g[~np.isnan(g)]; h = gap_ca[s][steady]; h = h[~np.isnan(h)]
    print(f"sim{s}: A-start->C-start span floor {floor[s]:.3f} ms; excess (≈b->c wait) mean {g.mean():.3f} ms (p90 {np.percentile(g, 90):.3f}, "
          f">0.2 ms on {100 * (g > 0.2).mean():.1f}% of frames); c->a gap mean {h.mean():.3f} ms (p90 {np.percentile(h, 90):.3f}) on {len(h)} frames")

# --------------------------------------------------------------------- figure
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
colors = ["#2a78d6", "#eb6834", "#2f9e5c", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#d4a017"]
ax = axes[0]
for s in sims[1:]:
    ax.plot(frames, delta[s], ".", ms=2, color=colors[s % len(colors)], label=f"sim{s} − sim0")
for y, ls in ((T / 2, "--"), (-T / 2, "--"), (T, ":"), (-T, ":")):
    ax.axhline(y, color="#888", linestyle=ls, linewidth=0.8)
ax.axhline(0, color="#444", linewidth=0.8)
ax.set_ylabel("phase A start offset (ms)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
ax.set_title((args.title + " — " if args.title else "") + f"inter-GPU phase offset per frame (T = {T:.2f} ms; dashed ±T/2, dotted ±T)", fontsize=10)
ax = axes[1]
for s in sims[1:]:
    ax.plot(frames, np.mod(delta[s], T) / T, ".", ms=2, color=colors[s % len(colors)], label=f"sim{s}")
ax.set_ylim(0, 1); ax.set_ylabel("(Δ mod T) / T"); ax.grid(alpha=0.3)
ax = axes[2]
for s in sims:
    ax.plot(frames, gap_bc[s], ".", ms=2, color=colors[s % len(colors)], label=f"sim{s} b→c wait (A→C span excess)")
    ax.plot(frames, gap_ca[s], "x", ms=2, color=colors[s % len(colors)], alpha=0.6, label=f"sim{s} c→a gap")
ax.set_ylabel("waiting (ms)"); ax.set_yscale("symlog", linthresh=0.1); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=len(sims), loc="upper right")
ax = axes[3]
win = 21
for s in sims:
    p = period[s]; roll = np.array([np.nanmedian(p[max(0, k - win // 2): k + win // 2 + 1]) for k in range(len(p))])
    ax.plot(frames, roll, "-", lw=1, color=colors[s % len(colors)], label=f"sim{s} period (rolling median {win})")
ax.axhline(T, color="#888", linestyle="--", linewidth=0.8)
ax.set_ylabel("period (ms)"); ax.set_xlabel("frame"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(args.out_png, dpi=130); print("saved", args.out_png)
