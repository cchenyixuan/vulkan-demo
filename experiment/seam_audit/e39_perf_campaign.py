"""
e39_perf_campaign.py - E39 performance campaign: the incremental v7 build chain (v6-rc2 -> +B9 -> +B6 -> +B4 -> +B1
-> +B3) on 13 chain-bench configurations, every K = 1 run on both GPUs at once (the simultaneous reference of the
local efficiency), every run's invariants, switch provenance and GPU clocks recorded.

Builds (BUILDS): v6 = experiment/v6 (v6-rc2) with its code defaults; the v7 builds up to b1 set all five E39 switches
explicitly, each adding one switch to the previous build (0 = that build command for command): b9
V7_DENSITY_COPY_COMPUTE=1 (lanes 0, deep-wall 0, fused 0, overlap 0), b6 + V7_GHOST_SEND_LANES=32, b4 +
V7_DEEP_WALL_SKIP=auto, b1 + V7_FUSED_CORRECTION_DENSITY=1; b3 = b1 + V7_BAND_OVERLAP=auto (B3's rule: one verdict
per chain; BAND_OVERLAP_VALUE_IN_B3, 1 or auto). In the E39 campaign b3 set nothing and ran the code defaults of
22701d4 (auto); since v7-rc1 the code default is 0, i.e. the b1 build. A
build whose added switch does not occur in experiment/v7/utils/*.py is left out of the plan, loudly (it would run the
previous build under a new name); a v7 run whose "v7 switches" header prints a set switch with another value is
invalid.

Configurations (CONFIGURATIONS; chain bench --weights 1,..,1 --device-map M --max-steps N --warmup W, everything else
the runner's defaults: depth 2, pool safety 1.2, per-direction sync, switch interval 0.2 ms, seam check, no validation):
  2d_62k_k1, _k2                       cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025          N 10000, W 2000
  2d_1m_k1, _k2                        cases/lid_driven_cavity_2d_gen                             N  6000, W 1000
  2d_16m_k1, _k2                       cases/lid_driven_cavity_2d_16m                             N  3000, W 1000
  3d_8m_walls4_k1, _k2, _k4_shared     cases/cavity3d_weak4_k2_8m_b4 (4-layer walls)              N  1300, W  300
  3d_1m_walls9_k1, _k2                 cases/cavity3d_1m (9-layer walls)                          N  3000, W 1000
  2d_adami_250_k1                      cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami    N 10000, W 2000
  2d_adami_1000_k1                     cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami   N  6000, W 1000
K = 2 runs map 0,1; K = 4 shared maps 0,1,0,1 (two slabs per GPU, timesliced). K = 1 runs are always two processes
started together, one on device 0 and one on device 1 (same build, case and steps), both recorded: K = 1 fps per
device and their mean, and the simultaneous K = 1 reference of the efficiency. Steps: the steady window (the runner's
STEADY line, the last N - W steps) is 8000 / 5000 steps at 62k / 1M (2.7 - 9 s of loop, as the B4 / B6 perf checks;
their run-to-run spread was <= 0.3 %), 2000 steps at 16M and 3-D 1M (17 - 56 s), 1000 steps at 3-D 8M (39 - 74 s);
the warmup covers the boost-clock ramp. Process wall at the priors: 4 - 14 s (2-D 62k / 1M, adami), 29 - 42 s (3-D
1M), 64 - 117 s (16M, 3-D 8M).

Skip rule (applied by the plan; the summary shows a skipped build identical to the previous one, with the reason):
  b6 at K = 1 (no ghost_send without peers); b4 where V7_DEEP_WALL_SKIP=auto resolves off on every slab (2-D by the
  dimension; 3-D with < 1 % deep-wall candidates: simulator_v7's own rule, SphSimulatorV7._resolve_deep_wall_skip with
  deep_wall_candidate_count, applied to every slab of the configuration's partition by the deep-wall-probe subprocess
  (CPU, below-normal priority: case load + partition, ~10 s for the 3-D cases), cached in <out>/deep_wall_auto.json,
  or logs/e39/perf_campaign_cache/ for a plan without --out, under fingerprints of the case files and the rule's
  code); b3 at K = 1 and in 3-D (B3 overlaps phase C's band chain of 2-D slabs with peers only, off there under every
  value) and, while b3 runs auto, where V7_BAND_OVERLAP=auto's chain verdict (simulator_v7.band_overlap_chain_verdict
  on the configuration's partition, the same probe subprocess, cached in band_overlap_auto.json next to the deep-wall
  cache) is off on every slab. A build is skipped only when its predecessor in the chain is part of the campaign (the
  result is inherited from it). The summary checks the per-slab resolutions the later v7 builds print (b3 is the last
  build: its skip rests on the probe; where it runs, a slab that prints off is noted). Measured on this checkout (plan --preflight): 3d_8m_walls4 has 0
  deep-wall candidates on every slab at K = 1, 2 and 4 (b4 skipped there), 3d_1m_walls9 16.63 % at K = 1 and 12.90 /
  17.18 % at K = 2 (b4 runs).
Order: trial-major; configurations in CONFIGURATIONS order; inside a configuration the builds in chain order rotated
by one per trial (trial t starts with the t-th build, so no build always runs first after a configuration switch);
v6 runs in every trial of every configuration.

Every run (run mode): before it, nvidia-smi must show both GPUs idle (no python compute process on any GPU, no compute
process at all on a GPU without a display, utilization <= 30 % on the display GPU (the desktop's baseline; E29 saw 13
- 25 %) and <= 5 % on the others; 5 s settle first, then 15 - 60 s retries, each logged; never starts on a busy GPU);
nvidia-smi --loop-ms=1000 samples SM / memory clock, power, temperature, utilization and memory of both GPUs during
the run (OUT/telemetry/*.csv), summarized (median / min / max) over each process's steady window: the runner's stdout
is read live (PYTHONUNBUFFERED=1) and the wall time of its TOTAL line minus the STEADY seconds gives the window. The
environment is the caller's minus every V5_* / V6_* / V7_* variable, plus VK_LOADER_LAYERS_DISABLE=
VK_LAYER_KHRONOS_validation, PYTHONIOENCODING=utf-8, PYTHONUNBUFFERED=1 and the build's switches (the release
defaults are the code defaults). Each process has a timeout (4 x its estimate, at least 300 s); a timed-out process
is killed with its tree (taskkill /T: the venv's python.exe is a launcher) and recorded, the campaign continues;
processes still running when the driver exits are killed. One JSON line per run unit (a K = 1 unit = both processes)
in OUT/ledger.jsonl: command, environment difference, build, configuration, trial, Vulkan device map -> nvidia-smi
GPU (deviceUUID probe at start, identity if it fails), start / end, exit code, parsed metrics, invariant verdict,
switch check, idle pre-check, telemetry, code digests. --resume skips units whose last record is valid with the
current code digest; without --resume an existing ledger is refused. --dry-run (and the plan mode) print the run
list in execution order with per-run steps and the GPU wall-time estimate per configuration and in total (PRIORS:
steady fps and per-process overhead measured in earlier logs; ledger medians replace them once a configuration ran).

summarize: OUT/summary.md + summary.json: fps per configuration and build (K = 1 per device and their mean), mean
+- std over trials; the incremental chain (each build against the previous one, paired by trial: ratio per trial,
mean +- std; and against v6); local efficiency of the K = 2 and K = 4 shared configurations, eta = fps_K / (G
mean_i fps_1(GPU i)) with G = the distinct GPUs of the device map and the K = 1 runs of the same build, case and
trial, also eta_min (min_i), +- = std over the trial-wise eta, the per-GPU reference spread and the SM clocks; every
run attempt's invariants (drift, every overflow_* counter, far migrations, GPU / host stamp errors, seam and field
checks, validation failure, exit code, switch provenance; any violation is listed first and summarize exits 1); the
SM clock medians per GPU and run; the skip-rule evidence.

trace: step traces (chain bench --step-trace DIR [--step-trace-detail phases|full]) of 2d_1m_k2 and 3d_8m_walls4_k2,
builds v6 and the final v7 build of each configuration (the last one the plan runs there), same idle checks,
telemetry and ledger (mode "trace"); OUT/trace_summary.md + json: per sim the medians of phase A, ghost_send
(a_voxel_end -> a_ghost_*_end, the B6 definition), A -> B, phase B, B -> C, phase C, C -> next A and the period
(steady complete steps, the step before a defrag boundary excluded: the B4 trace_summary definition), per link the
readback start delay, t_tr (send_end -> upload_end), upload_end -> receiver C start and the exposed steps, plus E29's
t_chain (experiment.v7.analysis.step_trace_model.link_metrics; its load_run reads every trace), v6 vs final.

selftest (CPU only, no GPU): the parser on real chain-bench logs of logs/e39 (skipped where absent) and on an
embedded v6-rc2 log against hand-read values, and on every logs/e39/*/perf* log against the runner's own verdict and
STEADY line; nvidia-smi parsing, the idle rule and the telemetry windows on canned output; the device mapping; the
plan rules (skip, rotation, v6 everywhere, unit status from the ledger) on a synthetic deep-wall resolution; a
synthetic ledger through summarize (pairing, ratios, inheritance, eta, loud violations); a fake campaign end to end
(fake GPU backend, fake runners that replay real logs, one hung run killed by its timeout, resume); the process-tree
kill through the venv launcher; the trace metrics against the B4 / B6 trace summaries; and the dry-run plan of the
default campaign.

Usage (from the checkout root; GPU only for run and trace):
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_perf_campaign plan [--configs A,B] [--builds v6,b9] \\
      [--trials 3] [--out DIR] [--preflight]
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_perf_campaign run --out logs/e39/perf_campaign \\
      [--configs ...] [--builds ...] [--trials 3] [--resume] [--dry-run]
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_perf_campaign summarize --out logs/e39/perf_campaign
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_perf_campaign trace --out logs/e39/perf_trace \\
      [--step-trace-detail phases|full] [--trials 1] [--resume] [--dry-run]
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_perf_campaign selftest [--out logs/e39/perf_campaign_selftest]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import yaml

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

LOG_PREFIX = "[e39_perf]"
LEDGER_SCHEMA = "e39_perf_campaign/1"

# ----------------------------------------------------------------------------- builds

# B3 (band overlap, 22701d4): its switch, simulator_v7's default (auto in 22701d4..8f8cbd9, 0 = off since v7-rc1; the
# selftest compares it with simulator_v7.py) and the value build b3 sets: 1 or auto (0 would be the b1 build under
# another name, refused here). b3 stays out of the plan while the name does not occur in experiment/v7.
BAND_OVERLAP_SWITCH = "V7_BAND_OVERLAP"
BAND_OVERLAP_CODE_DEFAULT = "0"
BAND_OVERLAP_VALUE_IN_B3 = "auto"
if BAND_OVERLAP_VALUE_IN_B3 not in ("1", "auto"):
    raise SystemExit(f"{LOG_PREFIX} BAND_OVERLAP_VALUE_IN_B3={BAND_OVERLAP_VALUE_IN_B3!r}: b3 must set 1 or auto "
                     "(0 is the b1 build under another name)")
B1_SWITCHES = {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "32", "V7_DEEP_WALL_SKIP": "auto",
               "V7_FUSED_CORRECTION_DENSITY": "1", BAND_OVERLAP_SWITCH: "0"}

# The incremental chain, in order. solver: the experiment/<solver> runner; adds: the switch this build turns on (the
# one the skip rule and the availability check look at); switches: the build's complete switch environment. Every v7
# build pins all five switches (a later default change cannot leak into a build); b3 = b1 + the overlap value (in
# the E39 campaign b3 set nothing and ran the code defaults of 22701d4, the same five values).
BUILDS = {
    "v6": {"solver": "v6", "adds": None, "label": "v6-rc2 (experiment/v6, code defaults)",
           "switches": {}},
    "b9": {"solver": "v7", "adds": "V7_DENSITY_COPY_COMPUTE", "label": "+B9 density copy as a compute pass",
           "switches": {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "0", "V7_DEEP_WALL_SKIP": "0",
                        "V7_FUSED_CORRECTION_DENSITY": "0", BAND_OVERLAP_SWITCH: "0"}},
    "b6": {"solver": "v7", "adds": "V7_GHOST_SEND_LANES", "label": "+B6 ghost_send lane groups (32 lanes)",
           "switches": {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "32", "V7_DEEP_WALL_SKIP": "0",
                        "V7_FUSED_CORRECTION_DENSITY": "0", BAND_OVERLAP_SWITCH: "0"}},
    "b4": {"solver": "v7", "adds": "V7_DEEP_WALL_SKIP", "label": "+B4 deep-wall skip (auto)",
           "switches": {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "32", "V7_DEEP_WALL_SKIP": "auto",
                        "V7_FUSED_CORRECTION_DENSITY": "0", BAND_OVERLAP_SWITCH: "0"}},
    "b1": {"solver": "v7", "adds": "V7_FUSED_CORRECTION_DENSITY", "label": "+B1 fused correction + density",
           "switches": dict(B1_SWITCHES)},
    "b3": {"solver": "v7", "adds": BAND_OVERLAP_SWITCH,
           "label": f"+B3 band overlap (b1 + V7_BAND_OVERLAP={BAND_OVERLAP_VALUE_IN_B3})",
           "switches": {**B1_SWITCHES, BAND_OVERLAP_SWITCH: BAND_OVERLAP_VALUE_IN_B3}},
}
CHAIN = tuple(BUILDS)
REFERENCE_BUILD = "v6"
RUNNERS = {"v6": "experiment/v6/_run_v6_chain_bench.py", "v7": "experiment/v7/_run_v7_chain_bench.py"}
SWITCH_PREFIXES = ("V5_", "V6_", "V7_")

# ----------------------------------------------------------------------------- cases and configurations

# steps = (max steps N, warmup W): the steady window is the last N - W steps (module docstring has the choice)
CASES = {
    "2d_62k": {"path": "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml", "steps": (10000, 2000),
               "label": "2-D 62k (n250)"},
    "2d_1m": {"path": "cases/lid_driven_cavity_2d_gen/case.yaml", "steps": (6000, 1000), "label": "2-D 1M"},
    "2d_16m": {"path": "cases/lid_driven_cavity_2d_16m/case.yaml", "steps": (3000, 1000), "label": "2-D 16M"},
    "3d_8m_walls4": {"path": "cases/cavity3d_weak4_k2_8m_b4/case.yaml", "steps": (1300, 300),
                     "label": "3-D 8M, 4-layer walls"},
    "3d_1m_walls9": {"path": "cases/cavity3d_1m/case.yaml", "steps": (3000, 1000), "label": "3-D 1M, 9-layer walls"},
    "2d_adami_250": {"path": "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml",
                     "steps": (10000, 2000), "label": "2-D adami 250^2"},
    "2d_adami_1000": {"path": "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami/case.yaml",
                      "steps": (6000, 1000), "label": "2-D adami 1000^2"},
}
# device_map None = K = 1: two processes at once, on device 0 and on device 1 (K1_DEVICE_MAPS)
CONFIGURATIONS = {
    "2d_62k_k1": {"case": "2d_62k", "device_map": None},
    "2d_62k_k2": {"case": "2d_62k", "device_map": "0,1"},
    "2d_1m_k1": {"case": "2d_1m", "device_map": None},
    "2d_1m_k2": {"case": "2d_1m", "device_map": "0,1"},
    "2d_16m_k1": {"case": "2d_16m", "device_map": None},
    "2d_16m_k2": {"case": "2d_16m", "device_map": "0,1"},
    "3d_8m_walls4_k1": {"case": "3d_8m_walls4", "device_map": None},
    "3d_8m_walls4_k2": {"case": "3d_8m_walls4", "device_map": "0,1"},
    "3d_8m_walls4_k4_shared": {"case": "3d_8m_walls4", "device_map": "0,1,0,1"},
    "3d_1m_walls9_k1": {"case": "3d_1m_walls9", "device_map": None},
    "3d_1m_walls9_k2": {"case": "3d_1m_walls9", "device_map": "0,1"},
    "2d_adami_250_k1": {"case": "2d_adami_250", "device_map": None},
    "2d_adami_1000_k1": {"case": "2d_adami_1000", "device_map": None},
}
K1_DEVICE_MAPS = ("0", "1")
TRACE_CONFIGURATIONS = ("2d_1m_k2", "3d_8m_walls4_k2")
DEFAULT_TRIALS = 3
DEFAULT_TRACE_TRIALS = 1
POOL_SAFETY = 1.2                       # the runner's default, for the deep-wall probe's partition

# ----------------------------------------------------------------------------- GPU time estimate

# Priors of the estimate: steady fps and process overhead in seconds (process wall minus the loop's TOTAL seconds: case
# load, Vulkan init, bootstrap, end checks) per configuration, two RTX 5090. fps of the closest measured build:
#   2d_62k: v7 B4 build (deep-wall 0) K = 1 3001 / 3030 (device 1), v6-rc2 2994 (E37); K = 2 B6 1855 - 1861, B9 1762
#   2d_1m: v7 K = 1 553 / 554 (device 1), v6 550 (device 1) / 529 - 555 (device 0, E32); K = 2 B6 850, B9 820, v6 ~800
#   2d_16m: v6 K = 1 34.6 (device 0) / 35.7 (device 1), K = 2 65.1 - 66.8 (E29 scan)
#   3d_8m_walls4: K = 2 B6 25.6 - 25.9 (B4 perf); K = 1 NOT measured: assumed 13.5 (cavity3d_8m's 10.5M particles
#     run 12.3 fps, scaled to 9.1M); K = 4 shared NOT measured without validation layers: assumed 22 (19.7 with them)
#   3d_1m_walls9: K = 1 v6 73.7 / 77.3 (E29), B4 build 77.6 (off) / 81.8 (on); K = 2 v6 109 (E29), B4 121.5 / 126
#   adami: v6-rc2 K = 1 (device 1) 2504 (250^2), 500.4 (1000^2), E37 timing
# overhead: logs/e39/b4/perf/run_perf{1,2}.out and logs/e39/b6/perf/run_perf.out (bash SECONDS, 1 s resolution: 62k
# 0.5 - 1 s, 1M 1 - 2 s, 3-D 1M 2 - 4 s, 9 - 10M 3-D 13 s), E29 scan ledger (finish-time differences minus the 5 s
# settle: 16M K = 1 ~32 s, K = 2 ~22 s). ESTIMATE_SPEEDUP: projected gains of the builds not in the priors (the audit:
# B1 +22 - 30 % over the configurations, B1's own perf check 2-D 1M K = 1 555 -> 729 fps = +31 %; B3 +1 - 11 % on 2-D
# K >= 2), applied to the prior fps; ledger medians replace both once measured.
PRIORS = {
    "2d_62k_k1": (3000.0, 1.0), "2d_62k_k2": (1800.0, 1.0),
    "2d_1m_k1": (550.0, 2.0), "2d_1m_k2": (830.0, 2.0),
    "2d_16m_k1": (35.5, 32.0), "2d_16m_k2": (66.0, 22.0),
    "3d_8m_walls4_k1": (13.5, 13.0), "3d_8m_walls4_k2": (25.7, 13.0), "3d_8m_walls4_k4_shared": (22.0, 15.0),
    "3d_1m_walls9_k1": (78.0, 3.0), "3d_1m_walls9_k2": (118.0, 4.0),
    "2d_adami_250_k1": (2500.0, 1.0), "2d_adami_1000_k1": (500.0, 2.0),
}
ESTIMATE_SPEEDUP = {"b1": 1.25, "b3": 1.30}
DEFAULT_PROCESS_ESTIMATE_SECONDS = 120.0
IDLE_SETTLE_SECONDS = 5.0
IDLE_CHECK_SECONDS = 1.0                # the two nvidia-smi queries
IDLE_RETRY_SECONDS = (15.0, 30.0, 60.0)  # waits between checks while busy (the last one repeats)
MINIMUM_RUN_TIMEOUT_SECONDS = 300.0
TIMEOUT_FACTOR = 4.0
DISPLAY_UTILIZATION_LIMIT_PERCENT = 30.0
HEADLESS_UTILIZATION_LIMIT_PERCENT = 5.0
TELEMETRY_INTERVAL_MS = 1000

# ----------------------------------------------------------------------------- small helpers


def log_line(log_path: Optional[pathlib.Path], message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(stamped + "\n")


def json_ready(value):
    """JSON-safe copy: non-finite floats -> None, numpy scalars -> Python, tuples -> lists."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_ledger(ledger_path: pathlib.Path, record: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_ready(record)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_ledger(ledger_path: pathlib.Path) -> list:
    """Every record in file order; a line that does not parse (a write cut short) is reported and skipped."""
    records = []
    if not ledger_path.exists():
        return records
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"{LOG_PREFIX} {ledger_path}:{line_number}: unreadable line skipped", flush=True)
    return records


def number_or_none(text: str):
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean_and_deviation(values) -> tuple:
    """(mean, sample standard deviation (nan for one value), count) of the finite values."""
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan"), 0
    deviation = statistics.stdev(finite) if len(finite) > 1 else float("nan")
    return statistics.fmean(finite), deviation, len(finite)


def format_number(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:,.{digits}f}"


def format_mean(mean, deviation, digits: int = 1) -> str:
    if mean is None or not math.isfinite(mean):
        return "-"
    if deviation is None or not math.isfinite(deviation):
        return format_number(mean, digits)
    return f"{format_number(mean, digits)} ± {format_number(deviation, digits)}"


def format_percent(mean, deviation, digits: int = 2) -> str:
    """A ratio as a signed percentage change: 1.0123 +- 0.0004 -> '+1.23 ± 0.04 %'."""
    if mean is None or not math.isfinite(mean):
        return "-"
    text = f"{100.0 * (mean - 1.0):+.{digits}f}"
    if deviation is not None and math.isfinite(deviation):
        text += f" ± {100.0 * deviation:.{digits}f}"
    return text + " %"


def host_memory_available_gib():
    """Available physical memory in GiB (Windows: GlobalMemoryStatusEx; Linux: /proc/meminfo), None elsewhere."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                            ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                            ("available_extended_virtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.available_physical / 2 ** 30
        elif pathlib.Path("/proc/meminfo").exists():
            for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        return None
    return None


# ----------------------------------------------------------------------------- environment, commands, provenance


def clean_environment() -> dict:
    """The caller's environment without any V5_* / V6_* / V7_* variable (the solvers' code defaults)."""
    return {key: value for key, value in os.environ.items() if not key.upper().startswith(SWITCH_PREFIXES)}


def run_environment(build_name: str) -> dict:
    environment = clean_environment()
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8",
                        "PYTHONUNBUFFERED": "1"})
    environment.update(BUILDS[build_name]["switches"])
    return environment


def environment_difference(environment: dict) -> dict:
    """What a run's environment changes against the caller's: removed variable names, set or changed values."""
    return {"removed": sorted(key for key in os.environ if key not in environment),
            "set": {key: value for key, value in sorted(environment.items()) if os.environ.get(key) != value}}


def slab_count(configuration_name: str) -> int:
    device_map = CONFIGURATIONS[configuration_name]["device_map"]
    return 1 if device_map is None else len(device_map.split(","))


def process_device_maps(configuration_name: str) -> list:
    """The --device-map of each process of a run unit: K = 1 = one process per GPU, started together."""
    device_map = CONFIGURATIONS[configuration_name]["device_map"]
    return list(K1_DEVICE_MAPS) if device_map is None else [device_map]


def configuration_steps(configuration_name: str) -> tuple:
    return CASES[CONFIGURATIONS[configuration_name]["case"]]["steps"]


def runner_command(build_name: str, configuration_name: str, device_map: str, trace_directory=None,
                   trace_detail: str = "phases") -> list:
    configuration = CONFIGURATIONS[configuration_name]
    max_steps, warmup = configuration_steps(configuration_name)
    command = [sys.executable, RUNNERS[BUILDS[build_name]["solver"]], "--case",
               CASES[configuration["case"]]["path"], "--weights", ",".join(["1"] * len(device_map.split(","))),
               "--device-map", device_map, "--max-steps", str(max_steps), "--warmup", str(warmup)]
    if trace_directory is not None:
        command += ["--step-trace", str(trace_directory), "--step-trace-detail", trace_detail]
    return command


def solver_code_digest(solver: str) -> str:
    """sha256 (16 hex) over experiment/<solver>/utils/*.py, shaders/spv/*.spv and the chain bench runner, relative
    path and bytes in sorted order: the code a run of that solver executes."""
    directory = _REPOSITORY_ROOT / "experiment" / solver
    paths = (list((directory / "utils").glob("*.py")) + list((directory / "shaders" / "spv").glob("*.spv"))
             + [_REPOSITORY_ROOT / RUNNERS[solver]])
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(_REPOSITORY_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()[:16]


def git_state() -> dict:
    def git(*arguments) -> str:
        return subprocess.run(["git", *arguments], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=_REPOSITORY_ROOT).stdout.strip()
    try:
        return {"head": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": {solver: bool(git("status", "--porcelain", f"experiment/{solver}")) for solver in RUNNERS}}
    except OSError:
        return {"head": None, "branch": None, "dirty": {}}


_V7_SOURCE_TEXT: dict = {}


def build_available(build_name: str) -> bool:
    """Does the build's added switch occur in experiment/v7/utils/*.py (v6: always)? Without it the switch is
    ignored and the run is the previous build under a new name."""
    switch = BUILDS[build_name]["adds"]
    if switch is None:
        return True
    if "text" not in _V7_SOURCE_TEXT:
        _V7_SOURCE_TEXT["text"] = "".join(path.read_text(encoding="utf-8", errors="replace") for path in
                                          sorted((_REPOSITORY_ROOT / "experiment" / "v7" / "utils").glob("*.py")))
    return switch in _V7_SOURCE_TEXT["text"]


def previous_build(build_name: str) -> Optional[str]:
    position = CHAIN.index(build_name)
    return CHAIN[position - 1] if position > 0 else None


# ----------------------------------------------------------------------------- cases: files and the deep-wall rule


def case_yaml(case_key: str) -> dict:
    return yaml.safe_load((_REPOSITORY_ROOT / CASES[case_key]["path"]).read_text(encoding="utf-8"))


def case_dimension(case_key: str) -> int:
    return int(case_yaml(case_key)["physics"]["dimension"])


def case_inputs(case_key: str) -> list:
    """Every file the case.yaml names (material library, frame, particle sets), resolved like load_case_v7, with
    existence and size; the .obj files are gitignored, a fresh checkout lacks them."""
    path = _REPOSITORY_ROOT / CASES[case_key]["path"]
    if not path.exists():
        return [{"role": "case.yaml", "path": str(path), "exists": False, "bytes": 0}]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = [("case.yaml", path.name), ("material_library", data.get("material_library")),
             ("frame", data.get("geometry", {}).get("frame"))]
    names += [("particles", entry.get("file")) for entry in data.get("geometry", {}).get("particles", [])]
    inputs = []
    for role, name in names:
        if not name:
            inputs.append({"role": role, "path": None, "exists": False, "bytes": 0})
            continue
        resolved = (path.parent / name).resolve()
        inputs.append({"role": role, "path": str(resolved), "exists": resolved.is_file(),
                       "bytes": resolved.stat().st_size if resolved.is_file() else 0})
    return inputs


def missing_case_inputs(configuration_names) -> dict:
    """case key -> missing input paths, for the cases of these configurations."""
    missing = {}
    for case_key in dict.fromkeys(CONFIGURATIONS[name]["case"] for name in configuration_names):
        absent = [item["path"] or item["role"] for item in case_inputs(case_key) if not item["exists"]]
        if absent:
            missing[case_key] = absent
    return missing


DEEP_WALL_PROBE_PREFIX = "DEEP_WALL_PROBE "
DEEP_WALL_REASON = re.compile(r"auto: 3-D, ([\d,]+) deep-wall candidates at the initial state = ([\d.]+) % of "
                              r"([\d,]+) particles")


def deep_wall_rule_fingerprint() -> str:
    """Digest of what decides V7_DEEP_WALL_SKIP=auto per slab: the source of deep_wall_candidate_count and
    _resolve_deep_wall_skip with the threshold line (simulator_v7.py; the whole file if the slices are not found),
    partition_v7.py, case_loader_v7.py, case_v7.py."""
    utilities = _REPOSITORY_ROOT / "experiment" / "v7" / "utils"
    digest = hashlib.sha256()
    simulator_text = (utilities / "simulator_v7.py").read_text(encoding="utf-8", errors="replace")
    slices = []
    for start_marker, end_marker in (("\ndef deep_wall_candidate_count", "\ndef "),
                                     ("    def _resolve_deep_wall_skip", "\n    def ")):
        start = simulator_text.find(start_marker)
        end = simulator_text.find(end_marker, start + len(start_marker)) if start >= 0 else -1
        slices.append(simulator_text[start:end] if start >= 0 and end > start else None)
    threshold = re.search(r"^_DEEP_WALL_AUTO_MINIMUM_CANDIDATE_FRACTION = .*$", simulator_text, re.MULTILINE)
    if None in slices or threshold is None:
        digest.update(simulator_text.encode("utf-8"))
    else:
        for part in slices + [threshold.group(0)]:
            digest.update(part.encode("utf-8"))
    for name in ("partition_v7.py", "case_loader_v7.py", "case_v7.py"):
        digest.update((utilities / name).read_bytes())
    return digest.hexdigest()[:16]


def case_fingerprint(case_key: str) -> str:
    """Digest of the case.yaml bytes and every input file's size and modification time."""
    digest = hashlib.sha256((_REPOSITORY_ROOT / CASES[case_key]["path"]).read_bytes())
    for item in case_inputs(case_key):
        if item["path"] and item["exists"]:
            status = pathlib.Path(item["path"]).stat()
            digest.update(f"{item['path']}|{status.st_size}|{status.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:16]


def deep_wall_probe_main(case_path: str, weight_texts: list) -> int:
    """Subprocess side of the deep-wall probe (CPU; this process imports simulator_v7 with V7_DEEP_WALL_SKIP unset):
    load the case, partition it per weight list as the runner does, and resolve the auto rule on every slab with the
    simulator's own method on a stand-in object (the CPU tests' object.__new__ pattern)."""
    if any(key.upper().startswith(SWITCH_PREFIXES) for key in os.environ):
        print(f"{LOG_PREFIX} deep-wall-probe must run without V5_/V6_/V7_ variables", flush=True)
        return 2
    from experiment.v7.utils import simulator_v7
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    if simulator_v7._DEEP_WALL_SKIP != "auto":
        print(f"{LOG_PREFIX} simulator_v7 reads V7_DEEP_WALL_SKIP={simulator_v7._DEEP_WALL_SKIP!r}, expected auto",
              flush=True)
        return 2
    # E39 B3: V7_BAND_OVERLAP=auto's chain verdict on the same partition (absent in a checkout before B3). The probe
    # evaluates the auto rule whatever the code default is (auto in 22701d4..8f8cbd9, 0 since v7-rc1): it sets the
    # module's value, as the CPU tests do.
    chain_verdict = getattr(simulator_v7, "band_overlap_chain_verdict", None)
    if chain_verdict is not None:
        simulator_v7._BAND_OVERLAP = "auto"
    started = time.time()
    case = load_case_v7(case_path)
    results = []
    for weight_text in weight_texts:
        weights = [float(part) for part in weight_text.split(",")]
        chain = compute_chain_partition(case, weights, POOL_SAFETY)
        slabs = []
        for slab_index, slab in enumerate(chain.slabs):
            stand_in = object.__new__(simulator_v7.SphSimulatorV7)
            stand_in.case = slab
            active, reason = stand_in._resolve_deep_wall_skip()
            numbers = DEEP_WALL_REASON.search(reason)
            geometry = chain.geometry[slab_index]
            slabs.append({"slab": slab_index, "active": bool(active), "reason": reason,
                          "particles": int(slab.initial.positions.shape[0]),
                          "candidates": int(numbers.group(1).replace(",", "")) if numbers else None,
                          "candidate_percent": float(numbers.group(2)) if numbers else None,
                          "own_columns": [int(geometry.own_global_first_column),
                                          int(geometry.own_global_last_column)],
                          "band_overlap": (None if chain_verdict is None
                                           else [bool(value) if index == 0 else value
                                                 for index, value in enumerate(chain_verdict(slab))])})
        results.append({"weights": weight_text, "slabs": slabs, "cuts": [int(cut) for cut in chain.cuts]})
    print(DEEP_WALL_PROBE_PREFIX + json.dumps({"case": case_path, "dimension": int(case.physics.dimension),
                                               "particles": int(case.initial.positions.shape[0]),
                                               "results": results, "seconds": round(time.time() - started, 1)}),
          flush=True)
    return 0


def run_deep_wall_probe(case_key: str, weight_texts: list, log) -> dict:
    """One probe subprocess per case (below-normal priority on Windows: plan mode may run beside a GPU job)."""
    command = [sys.executable, "-m", "experiment.seam_audit.e39_perf_campaign", "deep-wall-probe", "--case",
               CASES[case_key]["path"]] + [item for text in weight_texts for item in ("--weights", text)]
    log(f"deep-wall probe {CASES[case_key]['path']} weights {weight_texts} (CPU: case load + partition)")
    completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env={**clean_environment(), "PYTHONIOENCODING": "utf-8"},
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
                               creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    for line in completed.stdout.splitlines():
        if line.startswith(DEEP_WALL_PROBE_PREFIX):
            return json.loads(line[len(DEEP_WALL_PROBE_PREFIX):])
    raise RuntimeError(f"deep-wall probe of {case_key} failed (exit {completed.returncode}): "
                       f"{(completed.stdout + completed.stderr)[-1500:]}")


def deep_wall_resolutions(configuration_names, cache_path: Optional[pathlib.Path], log, refresh: bool = False) -> dict:
    """configuration -> {"active": [per slab], "reasons": [...], "source": ...} of V7_DEEP_WALL_SKIP=auto. 2-D: off by
    the dimension (the rule's first test, no probe). 3-D: the probe, cached per (case, weights) under the case and
    rule fingerprints. A failed probe leaves the configuration unresolved (None): b4 then runs there."""
    resolutions, wanted = {}, {}
    for name in configuration_names:
        case_key = CONFIGURATIONS[name]["case"]
        slabs = slab_count(name)
        try:
            dimension = case_dimension(case_key)
        except (OSError, KeyError, yaml.YAMLError) as error:
            resolutions[name] = None
            log(f"deep-wall rule for {name}: case.yaml unreadable ({error})")
            continue
        if dimension != 3:
            resolutions[name] = {"active": [False] * slabs, "reasons": [f"auto: {dimension}-D slab"] * slabs,
                                 "candidate_percent": [None] * slabs, "source": "dimension (no probe)"}
        else:
            wanted.setdefault(case_key, []).append(name)
    if not wanted:
        return resolutions
    cache = {}
    if cache_path is not None and cache_path.exists() and not refresh:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    rule = deep_wall_rule_fingerprint()
    changed = False
    for case_key, names in wanted.items():
        try:
            fingerprint = f"{CASES[case_key]['path']}|{case_fingerprint(case_key)}|{rule}"
        except OSError as error:
            for name in names:
                resolutions[name] = None
            log(f"deep-wall rule for {case_key}: inputs unreadable ({error})")
            continue
        entry = cache.get(case_key) if (cache.get(case_key) or {}).get("fingerprint") == fingerprint else None
        weight_texts = sorted({",".join(["1"] * slab_count(name)) for name in names}, key=len)
        missing = [text for text in weight_texts if entry is None or text not in entry["results"]]
        if missing:
            try:
                probe = run_deep_wall_probe(case_key, missing, log)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                for name in names:
                    resolutions[name] = None
                log(f"deep-wall rule for {case_key}: probe failed - b4 will run there. {error}")
                continue
            if entry is None:
                entry = {"fingerprint": fingerprint, "particles": probe["particles"], "results": {}}
            for result in probe["results"]:
                entry["results"][result["weights"]] = result
            entry["probe_seconds"] = probe["seconds"]
            cache[case_key] = entry
            changed = True
        for name in names:
            result = entry["results"][",".join(["1"] * slab_count(name))]
            resolutions[name] = {"active": [slab["active"] for slab in result["slabs"]],
                                 "reasons": [slab["reason"] for slab in result["slabs"]],
                                 "candidate_percent": [slab["candidate_percent"] for slab in result["slabs"]],
                                 "particles": [slab["particles"] for slab in result["slabs"]],
                                 "own_columns": [slab["own_columns"] for slab in result["slabs"]],
                                 "source": "probe" if missing else "cache"}
    if changed and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return resolutions


def default_deep_wall_cache(out_directory: Optional[pathlib.Path]) -> pathlib.Path:
    return (out_directory if out_directory is not None
            else _REPOSITORY_ROOT / "logs" / "e39" / "perf_campaign_cache") / "deep_wall_auto.json"


def default_band_overlap_cache(out_directory: Optional[pathlib.Path]) -> pathlib.Path:
    return default_deep_wall_cache(out_directory).with_name("band_overlap_auto.json")


def band_overlap_rule_fingerprint() -> str:
    """Digest of what decides V7_BAND_OVERLAP=auto's chain verdict: band_overlap_chain_verdict's source and the two
    thresholds (simulator_v7.py; the whole file if the slice is not found), partition_v7.py, case_loader_v7.py,
    case_v7.py."""
    utilities = _REPOSITORY_ROOT / "experiment" / "v7" / "utils"
    digest = hashlib.sha256()
    simulator_text = (utilities / "simulator_v7.py").read_text(encoding="utf-8", errors="replace")
    start = simulator_text.find("\ndef band_overlap_chain_verdict")
    end = simulator_text.find("\ndef ", start + 1) if start >= 0 else -1
    thresholds = re.findall(r"^_BAND_OVERLAP_AUTO_M(?:INIMUM|AXIMUM)_OWN_PARTICLES = .*$", simulator_text,
                            re.MULTILINE)
    if start < 0 or end <= start or len(thresholds) != 2:
        digest.update(simulator_text.encode("utf-8"))
    else:
        for part in [simulator_text[start:end]] + thresholds:
            digest.update(part.encode("utf-8"))
    for name in ("partition_v7.py", "case_loader_v7.py", "case_v7.py"):
        digest.update((utilities / name).read_bytes())
    return digest.hexdigest()[:16]


def band_overlap_resolutions(configuration_names, cache_path: Optional[pathlib.Path], log,
                             refresh: bool = False) -> dict:
    """configuration -> {"active": [per slab], "reasons": [...], "source": ...} of V7_BAND_OVERLAP=auto (E39 B3): K = 1
    and 3-D off without a probe (_resolve_band_overlap returns before the verdict there); 2-D K >= 2 the chain verdict
    (simulator_v7.band_overlap_chain_verdict) on every slab of the configuration's partition, from the probe
    subprocess, cached per (case, weights) under the case and rule fingerprints. The per-slab legality checks after
    the verdict (fused correction + density, cascade force, band-voxel dispatch, compute copy, simple walls) pass on
    every slab with the release defaults; the summary checks what the b3 runs print. A failed probe leaves the
    configuration unresolved (None): b3 then runs there."""
    resolutions, wanted = {}, {}
    for name in configuration_names:
        case_key = CONFIGURATIONS[name]["case"]
        slabs = slab_count(name)
        try:
            dimension = case_dimension(case_key)
        except (OSError, KeyError, yaml.YAMLError) as error:
            resolutions[name] = None
            log(f"band-overlap rule for {name}: case.yaml unreadable ({error})")
            continue
        if slabs == 1:
            resolutions[name] = {"active": [False], "reasons": ["no peer (K = 1): no band chain"],
                                 "source": "slab count (no probe)"}
        elif dimension != 2:
            resolutions[name] = {"active": [False] * slabs,
                                 "reasons": [f"{dimension}-D slab: band kernels throughput-bound"] * slabs,
                                 "source": "dimension (no probe)"}
        else:
            wanted.setdefault(case_key, []).append(name)
    if not wanted:
        return resolutions
    cache = {}
    if cache_path is not None and cache_path.exists() and not refresh:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    rule = band_overlap_rule_fingerprint()
    changed = False
    for case_key, names in wanted.items():
        try:
            fingerprint = f"{CASES[case_key]['path']}|{case_fingerprint(case_key)}|{rule}"
        except OSError as error:
            for name in names:
                resolutions[name] = None
            log(f"band-overlap rule for {case_key}: inputs unreadable ({error})")
            continue
        entry = cache.get(case_key) if (cache.get(case_key) or {}).get("fingerprint") == fingerprint else None
        weight_texts = sorted({",".join(["1"] * slab_count(name)) for name in names}, key=len)
        missing = [text for text in weight_texts if entry is None or text not in entry["results"]]
        if missing:
            try:
                probe = run_deep_wall_probe(case_key, missing, log)
                if any(slab.get("band_overlap") is None for result in probe["results"] for slab in result["slabs"]):
                    raise RuntimeError("the probe printed no band-overlap verdict (simulator_v7 before B3?)")
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                for name in names:
                    resolutions[name] = None
                log(f"band-overlap rule for {case_key}: probe failed - b3 will run there. {error}")
                continue
            if entry is None:
                entry = {"fingerprint": fingerprint, "particles": probe["particles"], "results": {}}
            for result in probe["results"]:
                entry["results"][result["weights"]] = [
                    {"slab": slab["slab"], "particles": slab["particles"], "band_overlap": slab["band_overlap"]}
                    for slab in result["slabs"]]
            entry["probe_seconds"] = probe["seconds"]
            cache[case_key] = entry
            changed = True
        for name in names:
            slabs = entry["results"][",".join(["1"] * slab_count(name))]
            resolutions[name] = {"active": [bool(slab["band_overlap"][0]) for slab in slabs],
                                 "reasons": [slab["band_overlap"][1] for slab in slabs],
                                 "particles": [slab["particles"] for slab in slabs],
                                 "source": "probe" if missing else "cache"}
    if changed and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return resolutions


# ----------------------------------------------------------------------------- plan


def inert_reason(build_name: str, configuration_name: str, deep_wall: dict,
                 band_overlap: Optional[dict] = None) -> Optional[str]:
    """Why the build's added switch cannot act in this configuration (None: it can, or that is unknown).
    band_overlap: band_overlap_resolutions (None: b3 is skipped by the slab count and the dimension only)."""
    slabs = slab_count(configuration_name)
    if build_name == "b6" and slabs == 1:
        return "B6 inert at K = 1: no ghost_send without peers"
    if build_name == "b4":
        resolution = deep_wall.get(configuration_name)
        if resolution is not None and not any(resolution["active"]):
            percents = [value for value in resolution.get("candidate_percent") or [] if value is not None]
            detail = (f"{resolution['reasons'][0]}" if not percents else
                      f"3-D, deep-wall candidates {max(percents):.2f} % at most (< 1 %) on every slab")
            return f"B4 inert: V7_DEEP_WALL_SKIP=auto resolves off on every slab ({detail})"
    if build_name == "b3":
        if slabs == 1:
            return "B3 inert at K = 1: no band chain without peers (off there under every value)"
        if case_dimension(CONFIGURATIONS[configuration_name]["case"]) == 3:
            return "B3 inert in 3-D: overlap acts on 2-D slabs with peers only (off there under every value)"
        resolution = (band_overlap or {}).get(configuration_name)
        b3_value = normalized_switch(BUILDS["b3"]["switches"].get(BAND_OVERLAP_SWITCH, BAND_OVERLAP_CODE_DEFAULT))
        if resolution is not None and not any(resolution["active"]) and b3_value == "auto":
            return f"B3 inert: V7_BAND_OVERLAP=auto resolves off on every slab ({resolution['reasons'][0]})"
    return None


def unit_identifier(trial: int, configuration_name: str, build_name: str) -> str:
    return f"t{trial}/{configuration_name}/{build_name}"


def unit_stem(trial: int, configuration_name: str, build_name: str) -> str:
    return f"t{trial}__{configuration_name}__{build_name}"


def unit_status(records: list, digests: dict, build_name: str) -> str:
    """done: the last record is valid with the current code digest of the build's solver; stale: valid with an older
    digest (run again); failed: the last record is not valid (run again); pending: no record."""
    if not records:
        return "pending"
    last = records[-1]
    if not last.get("valid"):
        return "failed"
    solver = BUILDS[build_name]["solver"]
    return "done" if (last.get("code") or {}).get(solver) == digests.get(solver) else "stale"


def configuration_builds(configuration_name: str, selected: list, deep_wall: dict,
                         band_overlap: Optional[dict] = None) -> tuple:
    """(builds to run in chain order, skipped build -> {reason, previous, inherits}). A build is skipped only when its
    chain predecessor is in the campaign (run or skipped itself): its result is then inherited from the run build."""
    run_builds, skipped = [], {}
    for build_name in selected:
        predecessor = previous_build(build_name)
        reason = inert_reason(build_name, configuration_name, deep_wall, band_overlap)
        if reason is not None and predecessor in selected:
            source = predecessor
            while source in skipped:
                source = skipped[source]["previous"]
            skipped[build_name] = {"reason": reason, "previous": predecessor, "inherits": source}
        else:
            run_builds.append(build_name)
    return run_builds, skipped


def process_estimate_seconds(configuration_name: str, build_name: str, measured: Optional[dict] = None) -> tuple:
    """(seconds, source) of one process (a K = 1 pair: both, in parallel): the ledger's median when this
    (configuration, build) ran before, else overhead + max_steps / (prior fps x projected speedup)."""
    if measured and measured.get((configuration_name, build_name)):
        return statistics.median(measured[(configuration_name, build_name)]), "ledger"
    prior = PRIORS.get(configuration_name)
    if prior is None:
        return DEFAULT_PROCESS_ESTIMATE_SECONDS, "default"
    fps, overhead = prior
    max_steps, _ = configuration_steps(configuration_name)
    return overhead + max_steps / (fps * ESTIMATE_SPEEDUP.get(build_name, 1.0)), "prior"


def measured_process_seconds(records: list) -> dict:
    """(configuration, build) -> the wall seconds of every valid unit (the slower process of a K = 1 pair)."""
    measured = {}
    for record in records:
        if record.get("valid") and record.get("processes"):
            walls = [process.get("wall_s") for process in record["processes"] if process.get("wall_s")]
            if walls:
                measured.setdefault((record["configuration"], record["build"]), []).append(max(walls))
    return measured


def build_plan(configuration_names: list, build_names: list, trials: int, deep_wall: dict, records: list,
               digests: dict, mode: str = "campaign", availability: Optional[dict] = None,
               band_overlap: Optional[dict] = None) -> dict:
    """The run list in execution order (module docstring: order and skip rule) with each unit's status, estimate and
    timeout, plus the per-configuration build lists. availability: build -> its switch is in the code (default: read
    experiment/v7/utils); band_overlap: band_overlap_resolutions (None: b3's skip by slab count and dimension only)."""
    availability = availability or {name: build_available(name) for name in build_names}
    selected = [name for name in CHAIN if name in build_names and availability[name]]
    by_unit = {}
    for record in records:
        if record.get("mode", "campaign") == mode:
            by_unit.setdefault(record["unit"], []).append(record)
    measured = measured_process_seconds([record for record in records if record.get("mode", "campaign") == mode])
    configurations = {}
    for name in configuration_names:
        run_builds, skipped = configuration_builds(name, selected, deep_wall, band_overlap)
        configurations[name] = {"run": run_builds, "skipped": skipped}
    units = []
    for trial in range(1, trials + 1):
        for name in configuration_names:
            run_builds = configurations[name]["run"]
            if not run_builds:
                continue
            shift = (trial - 1) % len(run_builds)
            for build_name in run_builds[shift:] + run_builds[:shift]:
                identifier = unit_identifier(trial, name, build_name)
                seconds, source = process_estimate_seconds(name, build_name, measured)
                prior_seconds, _ = process_estimate_seconds(name, build_name)
                units.append({"unit": identifier, "trial": trial, "configuration": name, "build": build_name,
                              "device_maps": process_device_maps(name), "steps": configuration_steps(name),
                              "status": unit_status(by_unit.get(identifier, []), digests, build_name),
                              "attempts": len(by_unit.get(identifier, [])),
                              "estimate_seconds": seconds, "estimate_source": source,
                              "timeout_seconds": max(MINIMUM_RUN_TIMEOUT_SECONDS,
                                                     TIMEOUT_FACTOR * max(seconds, prior_seconds))})
    return {"mode": mode, "units": units, "configurations": configurations, "selected": selected,
            "unavailable": [name for name in build_names if not availability[name]], "trials": trials,
            "deep_wall": deep_wall, "band_overlap": band_overlap}


def print_plan(plan: dict, out_directory: Optional[pathlib.Path], digests: dict) -> None:
    """The --dry-run report: the run list, the skipped and unavailable builds, the deep-wall resolutions, the
    environments, and the GPU wall-time estimate per configuration and in total."""
    units = plan["units"]
    counts = {status: sum(1 for unit in units if unit["status"] == status)
              for status in ("pending", "failed", "stale", "done")}
    state = git_state()
    builds = ("trace builds " + "; ".join(f"{name}: {', '.join(entry.get('trace_builds', []))}"
                                          for name, entry in plan["configurations"].items())
              if plan["mode"] == "trace" else f"builds {', '.join(plan['selected'])}")
    print(f"{LOG_PREFIX} plan ({plan['mode']}): {len(units)} run units over {plan['trials']} trial(s), "
          f"{len(plan['configurations'])} configuration(s), {builds}; "
          + ", ".join(f"{count} {status}" for status, count in counts.items() if count)
          + (f"; out {out_directory}" if out_directory else ""), flush=True)
    print(f"{LOG_PREFIX} code: v6 {digests.get('v6')}, v7 {digests.get('v7')}; HEAD {state.get('head')} "
          f"({state.get('branch')})" + "".join(f"; experiment/{solver} has uncommitted changes"
                                               for solver, dirty in (state.get("dirty") or {}).items() if dirty))
    for name in plan["unavailable"]:
        print(f"{LOG_PREFIX} *** build {name} LEFT OUT: its switch {BUILDS[name]['adds']} does not occur in "
              f"experiment/v7/utils (not implemented yet?) ***", flush=True)
    print(f"  {'#':>4} {'trial':>5}  {'configuration':<24} {'build':<5} {'processes (--device-map)':<26} "
          f"{'steps N / W / steady':<22} {'est s':>7}  status")
    for position, unit in enumerate(units, 1):
        max_steps, warmup = unit["steps"]
        maps = (" | ".join(unit["device_maps"]) + " (at once)" if len(unit["device_maps"]) > 1
                else unit["device_maps"][0])
        print(f"  {position:>4} {unit['trial']:>5}  {unit['configuration']:<24} {unit['build']:<5} {maps:<26} "
              f"{f'{max_steps} / {warmup} / {max_steps - warmup}':<22} {unit['estimate_seconds']:>7.0f}  "
              f"{unit['status']}" + (f" ({unit['attempts']} attempt(s))" if unit["attempts"] else ""))
    print(f"{LOG_PREFIX} skip rule (a skipped build = its predecessor's result, inherited):")
    for name, configuration in plan["configurations"].items():
        for build_name, skip in configuration["skipped"].items():
            print(f"    {name:<24} {build_name}: = {skip['inherits']} ({skip['reason']})")
    print(f"{LOG_PREFIX} V7_DEEP_WALL_SKIP=auto per slab:")
    for name in plan["configurations"]:
        resolution = plan["deep_wall"].get(name)
        if resolution is None:
            print(f"    {name:<24} unresolved (b4 runs)")
            continue
        slabs = "; ".join(f"s{slab_index} {'ON' if active else 'off'}"
                          + (f" {percent:.2f} %" if percent is not None else "")
                          for slab_index, (active, percent) in enumerate(zip(resolution["active"],
                                                                             resolution["candidate_percent"])))
        print(f"    {name:<24} {slabs} [{resolution['source']}]")
    if plan.get("band_overlap") is not None:
        print(f"{LOG_PREFIX} V7_BAND_OVERLAP=auto per slab (b3):")
        for name in plan["configurations"]:
            resolution = plan["band_overlap"].get(name)
            if resolution is None:
                print(f"    {name:<24} unresolved (b3 runs)")
                continue
            states = ", ".join("ON" if active else "off" for active in resolution["active"])
            print(f"    {name:<24} {states} ({resolution['reasons'][0]}) [{resolution['source']}]")
    for build_name in plan["selected"]:
        environment = run_environment(build_name)
        switches = " ".join(f"{key}={value}" for key, value in sorted(environment.items())
                            if key.upper().startswith(SWITCH_PREFIXES) or key == "VK_LOADER_LAYERS_DISABLE")
        example = next((unit for unit in units if unit["build"] == build_name), None)
        print(f"{LOG_PREFIX} {build_name} ({BUILDS[build_name]['label']}): the caller's environment without "
              f"V5_/V6_/V7_ + {switches}")
        if example:
            command = runner_command(build_name, example["configuration"], example["device_maps"][0])
            print(f"      e.g. {' '.join(command)}")
    to_run = [unit for unit in units if unit["status"] != "done"]
    fixed = IDLE_SETTLE_SECONDS + IDLE_CHECK_SECONDS
    print(f"{LOG_PREFIX} GPU wall-time estimate of the {len(to_run)} unit(s) to run (process wall: overhead + N / "
          f"fps, PRIORS / ledger; + {fixed:.0f} s idle settle and check per unit):")
    print(f"    {'configuration':<24} {'units':>5} {'per trial':>10} {'all trials':>11}  sources")
    total = 0.0
    for name in plan["configurations"]:
        items = [unit for unit in to_run if unit["configuration"] == name]
        if not items:
            continue
        seconds = sum(unit["estimate_seconds"] + fixed for unit in items)
        trials_present = len({unit["trial"] for unit in items})
        total += seconds
        sources = ",".join(sorted({unit["estimate_source"] for unit in items}))
        print(f"    {name:<24} {len(items):>5} {seconds / max(1, trials_present) / 60:>8.1f} m "
              f"{seconds / 60:>9.1f} m  {sources}")
    print(f"    {'total':<24} {len(to_run):>5} {'':>10} {total / 60:>9.1f} m  = {total / 3600:.2f} h "
          f"(GPU busy {sum(unit['estimate_seconds'] for unit in to_run) / 3600:.2f} h)", flush=True)


# ----------------------------------------------------------------------------- runner log parser

# The runner prints (experiment/v6/_run_v6_chain_bench.py and experiment/v7/_run_v7_chain_bench.py, prefix chain_v6 /
# chain_v7; the v7 header adds the switch line, simulator_v7 the per-slab switch resolutions):
LINE_PATTERNS = {
    "switch_interval": re.compile(r"^\[chain_v(?P<version>\d+)\] switchinterval_s=(?P<value>\S+)$"),
    "case": re.compile(r"^\[case_loader_v\d+\] loading (?P<name>\S+)"),
    "loaded": re.compile(r"^\[case_loader_v\d+\] total loaded: (?P<count>[\d,]+) particles"),
    "partition": re.compile(r"^\[partition_v\d+\] chain N=(?P<slabs>\d+): columns (?P<columns>.*) of "
                            r"(?P<grid>\d+); particles (?P<particles>[\d, ]+)$"),
    "header": re.compile(r"^\[chain_v(?P<version>\d+)\] K=(?P<slabs>\d+) weights=\[(?P<weights>[^\]]*)\] "
                         r"device_map=\[(?P<device_map>[^\]]*)\] sync=(?P<sync>\S+) depth=(?P<depth>\d+) "
                         r"pool_safety=(?P<pool_safety>\S+)"),
    "cuts": re.compile(r"^\[chain_v\d+\] weights source=(?P<source>\S+) cuts=\[(?P<cuts>[^\]]*)\]"),
    "switches": re.compile(r"^\[chain_v\d+\] v7 switches: ?(?P<switches>.*)$"),
    "resolution": re.compile(r"^\[SimV\d+\] (?P<switch>V\d_[A-Z0-9_]+)=(?P<value>\S+): (?P<resolution>.*)$"),
    "pipelines": re.compile(r"^\[SimV\d+\] compute pipelines: (?P<count>\d+)(?P<notes>.*)$"),
    "total": re.compile(r"^\[chain_v\d+\] TOTAL: (?P<steps>\d+) steps in (?P<seconds>[\d.]+)s = (?P<fps>[\d.]+) fps"),
    "steady": re.compile(r"^\[chain_v\d+\] STEADY \(post-warmup (?P<warmup>\d+)\): (?P<steps>\d+) steps in "
                         r"(?P<seconds>[\d.]+)s = (?P<fps>[\d.]+) fps"),
    "seam_pool": re.compile(r"^\[chain_v\d+\] sim(?P<sim>\d+) seam: ghost_layers=(?P<ghost_layers>\d+) departed "
                            r"peak/frame=(?P<departed_peak>\d+) capacity=(?P<departed_capacity>\d+) "
                            r"far_migration=(?P<far_migration>\d+)(?P<counters>.*)$"),
    "sim": re.compile(r"^\[chain_v\d+\] sim(?P<sim>\d+) \(dev(?P<device>\d+)\): alive=(?P<alive>[\d,]+) "
                      r"pool_used=(?P<pool_used>[\d.]+)% peak_migration=(?P<peak_migration>\d+) "
                      r"drops=(?P<drops>\d+) stamp_err=(?P<stamp_errors>\d+)"),
    "worker": re.compile(r"^\[worker (?P<label>\S+)\] us p50/p90/max: (?P<segments>.*)$"),
    "link": re.compile(r"^\[link (?P<label>\S+)\] bytes/frame: host_copy=(?P<host_copy>[\d.]+) KiB "
                       r"dma=(?P<dma>[\d.]+) KiB over (?P<frames>\d+) frames"),
    "final": re.compile(r"^\[chain_v\d+\] final: total=(?P<total>[\d,]+) \(expected (?P<expected>[\d,]+)\) "
                        r"drift=(?P<drift>-?\d+) stamp_errors gpu=(?P<stamp_gpu>\d+) host=(?P<stamp_host>\d+) "
                        r"overflow_total=(?P<overflow_total>\d+) far_migration_total=(?P<far_migration_total>\d+)"),
    "seam_check": re.compile(r"^\[chain_v\d+\] seam (?P<seam>\d+) \(col (?P<column>\d+)\): "
                             r"L_overshoot=(?P<left>[-+\d.]+)dx R_overshoot=(?P<right>[-+\d.]+)dx "
                             r"dup=(?P<duplicates>\d+) (?P<verdict>OK|\*\*\* FAIL \*\*\*)"),
    "fields": re.compile(r"^\[chain_v\d+\] fields: rho\[(?P<density_min>[-\d.]+),(?P<density_max>[-\d.]+)\] "
                         r"vmax=(?P<speed_max>[-\d.]+) (?P<verdict>OK|\*\*\* FAIL \*\*\*)"),
    "drops": re.compile(r"^\[migration\] frame (?P<frame>\d+): interval (?P<interval>\S+) \*\*\* "
                        r"DROPS=(?P<drops>\d+) \*\*\*"),
}
VALIDATION_FAILED_TEXT = "*** VALIDATION FAILED ***"
# Wall time of these output lines (read live): the loop starts after "ready", TOTAL ends it.
MARKER_PATTERNS = (("ready", re.compile(r"^\[ChainOrchV\d+\] all \d+ sims bootstrapped")),
                   ("loop_end", re.compile(r"^\[chain_v\d+\] TOTAL: ")),
                   ("steady_line", re.compile(r"^\[chain_v\d+\] STEADY ")))


def integer(text: str) -> int:
    return int(text.replace(",", "").strip())


def precise_fps(steps: int, seconds: float, printed: float) -> float:
    """The runner prints fps with 0.1 fps and seconds with 0.01 s: the more precise of the printed fps and steps /
    seconds (step_trace_model.precise_fps's rule, half a unit of the last digit each)."""
    if seconds > 0 and printed > 0 and 0.005 / seconds < 0.05 / printed:
        return steps / seconds
    return printed


def parse_runner_log(text: str) -> dict:
    """Everything the summary and the invariant verdict need from one chain-bench log (v6 or v7)."""
    metrics = {"version": None, "case": None, "particles": None, "slabs": None, "weights": None, "device_map": None,
               "depth": None, "cuts": None, "partition": None, "v7_switches": None, "switch_resolutions": {},
               "pipeline_notes": [], "total": None, "steady": None, "fps": None, "sims": {}, "workers": {},
               "links": {}, "final": None, "seams": [], "fields": None, "validation_failed": False, "drops": [],
               "error": None, "switch_interval_s": None}
    lines = text.splitlines()
    for raw_line in lines:
        line = raw_line.rstrip()
        if VALIDATION_FAILED_TEXT in line:
            metrics["validation_failed"] = True
        for kind, pattern in LINE_PATTERNS.items():
            match = pattern.match(line)
            if not match:
                continue
            group = match.groupdict()
            if kind == "switch_interval":
                metrics["version"] = int(group["version"])
                metrics["switch_interval_s"] = number_or_none(group["value"])
            elif kind == "case":
                metrics["case"] = group["name"]
            elif kind == "loaded":
                metrics["particles"] = integer(group["count"])
            elif kind == "partition":
                metrics["partition"] = {"slabs": int(group["slabs"]), "columns": group["columns"],
                                        "grid_columns": int(group["grid"]),
                                        "particles": [integer(part) for part in group["particles"].split(", ")]}
            elif kind == "header":
                metrics["version"] = int(group["version"])
                metrics["slabs"] = int(group["slabs"])
                metrics["weights"] = [float(part) for part in group["weights"].split(",") if part.strip()]
                metrics["device_map"] = [int(part) for part in group["device_map"].split(",") if part.strip()]
                metrics["depth"] = int(group["depth"])
            elif kind == "cuts":
                metrics["cuts"] = [int(part) for part in group["cuts"].split(",") if part.strip()]
            elif kind == "switches":
                metrics["v7_switches"] = dict(item.split("=", 1) for item in group["switches"].split() if "=" in item)
            elif kind == "resolution":
                metrics["switch_resolutions"].setdefault(group["switch"], []).append(
                    {"value": group["value"], "resolution": group["resolution"]})
            elif kind == "pipelines":
                metrics["pipeline_notes"].append(group["notes"].strip())
            elif kind == "total":
                metrics["total"] = {"steps": int(group["steps"]), "seconds": float(group["seconds"]),
                                    "fps": float(group["fps"])}
            elif kind == "steady":
                steps, seconds, printed = int(group["steps"]), float(group["seconds"]), float(group["fps"])
                metrics["steady"] = {"warmup": int(group["warmup"]), "steps": steps, "seconds": seconds,
                                     "fps_printed": printed}
                metrics["fps"] = precise_fps(steps, seconds, printed)
            elif kind == "seam_pool":
                sim = metrics["sims"].setdefault(group["sim"], {})
                sim.update({"ghost_layers": int(group["ghost_layers"]), "departed_peak": int(group["departed_peak"]),
                            "departed_capacity": int(group["departed_capacity"]),
                            "far_migration": int(group["far_migration"]),
                            "overflow": {name: int(value) for name, value in
                                         re.findall(r"(overflow_[a-z_]+)=(\d+)", group["counters"])}})
            elif kind == "sim":
                sim = metrics["sims"].setdefault(group["sim"], {})
                sim.update({"device": int(group["device"]), "alive": integer(group["alive"]),
                            "pool_used_percent": float(group["pool_used"]),
                            "peak_migration": int(group["peak_migration"]), "drops": int(group["drops"]),
                            "stamp_errors": int(group["stamp_errors"])})
            elif kind == "worker":
                metrics["workers"][group["label"]] = {
                    name: [number_or_none(part) for part in values.split("/")]
                    for name, values in re.findall(r"(\w+)=([\d/.]+)", group["segments"])}
            elif kind == "link":
                metrics["links"][group["label"]] = {"host_copy_kib": float(group["host_copy"]),
                                                    "dma_kib": float(group["dma"]), "frames": int(group["frames"])}
            elif kind == "final":
                metrics["final"] = {"total": integer(group["total"]), "expected": integer(group["expected"]),
                                    "drift": int(group["drift"]), "stamp_errors_gpu": int(group["stamp_gpu"]),
                                    "stamp_errors_host": int(group["stamp_host"]),
                                    "overflow_total": int(group["overflow_total"]),
                                    "far_migration_total": int(group["far_migration_total"])}
            elif kind == "seam_check":
                metrics["seams"].append({"seam": int(group["seam"]), "column": int(group["column"]),
                                         "left_overshoot_dx": float(group["left"]),
                                         "right_overshoot_dx": float(group["right"]),
                                         "duplicates": int(group["duplicates"]), "ok": group["verdict"] == "OK"})
            elif kind == "fields":
                metrics["fields"] = {"density_min": float(group["density_min"]),
                                     "density_max": float(group["density_max"]),
                                     "speed_max": float(group["speed_max"]), "ok": group["verdict"] == "OK"}
            elif kind == "drops":
                metrics["drops"].append({"frame": int(group["frame"]), "drops": int(group["drops"])})
            break
    for marker in ("Traceback (most recent call last):", "Fatal Python error", "Windows fatal exception"):
        if marker in text:
            tail = [line.strip() for line in text[text.rfind(marker):].splitlines() if line.strip()]
            metrics["error"] = tail[-1] if tail else marker
            break
    if metrics["error"] is None:
        refusal = [line for line in lines if line.strip() and not line.startswith(("[", " "))]
        if refusal and metrics["final"] is None and metrics["steady"] is None:
            metrics["error"] = refusal[-1].strip()           # a sys.exit message of the runner
    return metrics


def deep_wall_states(metrics: dict) -> list:
    """Per slab 'on' / 'off' from '[SimV7] V7_DEEP_WALL_SKIP=<value>: on|off (...)' (absent: the switch was 0)."""
    return [item["resolution"].split(" ", 1)[0] for item in metrics["switch_resolutions"].get("V7_DEEP_WALL_SKIP", [])]


def invariant_verdict(metrics: dict, exit_code, timed_out: bool) -> dict:
    """Every invariant of one process: exit code, the runner's end checks (drift, overflow counters, far migrations,
    stamp errors, seam overshoot / duplicates, field band, VALIDATION FAILED), migration drops, exceptions, a missing
    steady line. ok = no violation."""
    violations = []
    if timed_out:
        violations.append("timed out (killed)")
    if exit_code not in (0, None) and not timed_out:
        violations.append(f"exit code {exit_code}")
    if metrics.get("error"):
        violations.append(f"error: {metrics['error']}")
    final = metrics.get("final")
    if final is None:
        violations.append("no final line (the run did not reach its end checks)")
    else:
        if final["drift"] != 0:
            violations.append(f"drift {final['drift']} (total {final['total']:,}, expected {final['expected']:,})")
        if final["overflow_total"]:
            violations.append(f"overflow_total {final['overflow_total']}")
        if final["far_migration_total"]:
            violations.append(f"far_migration_total {final['far_migration_total']}")
        if final["stamp_errors_gpu"] or final["stamp_errors_host"]:
            violations.append(f"stamp errors gpu {final['stamp_errors_gpu']} host {final['stamp_errors_host']}")
    for sim, values in sorted(metrics.get("sims", {}).items()):
        nonzero = {name: value for name, value in (values.get("overflow") or {}).items() if value}
        if nonzero:
            violations.append(f"sim{sim} " + " ".join(f"{name}={value}" for name, value in nonzero.items()))
        for key in ("far_migration", "drops", "stamp_errors"):
            if values.get(key):
                violations.append(f"sim{sim} {key}={values[key]}")
    for seam in metrics.get("seams", []):
        if not seam["ok"]:
            violations.append(f"seam {seam['seam']} FAIL (overshoot {seam['left_overshoot_dx']:+.2f} / "
                              f"{seam['right_overshoot_dx']:+.2f} dx, duplicates {seam['duplicates']})")
    fields = metrics.get("fields")
    if fields is not None and not fields["ok"]:
        violations.append(f"fields FAIL (rho {fields['density_min']}..{fields['density_max']}, "
                          f"vmax {fields['speed_max']})")
    if metrics.get("validation_failed"):
        violations.append("VALIDATION FAILED")
    if metrics.get("drops"):
        violations.append(f"migration drops at {len(metrics['drops'])} defrag(s), "
                          f"{sum(item['drops'] for item in metrics['drops'])} in all")
    if metrics.get("steady") is None:
        violations.append("no STEADY line")
    return {"ok": not violations, "violations": violations}


def normalized_switch(value) -> str:
    return str(value).strip().lower()


def provenance_check(build_name: str, metrics: dict, configuration_name: Optional[str] = None,
                     device_map: Optional[str] = None) -> dict:
    """Did this log come from the run it is filed under? The runner (chain_v6 / chain_v7) is the build's solver; with
    the configuration: the case directory, K, the device map, the steps and the warmup; a v6 run prints no v7 switch
    line; a v7 run's header shows every switch the build sets with that value where it prints it (a mismatch = not
    this build: invalid) and should print the build's added switch (added_reported False = not verifiable)."""
    build = BUILDS[build_name]
    problems = []
    if metrics.get("version") is not None and f"v{metrics['version']}" != build["solver"]:
        problems.append(f"the log is chain_v{metrics['version']}'s, build {build_name} runs "
                        f"experiment/{build['solver']}")
    if configuration_name is not None:
        case_directory = pathlib.Path(CASES[CONFIGURATIONS[configuration_name]["case"]]["path"]).parent.name
        if metrics.get("case") not in (None, case_directory):
            problems.append(f"case {metrics.get('case')}, expected {case_directory}")
        max_steps, warmup = configuration_steps(configuration_name)
        if metrics.get("total") is not None and metrics["total"]["steps"] != max_steps:
            problems.append(f"{metrics['total']['steps']} steps, expected {max_steps}")
        if metrics.get("steady") is not None and metrics["steady"]["warmup"] != warmup:
            problems.append(f"warmup {metrics['steady']['warmup']}, expected {warmup}")
    if device_map is not None and metrics.get("device_map") is not None:
        expected = [int(part) for part in device_map.split(",")]
        if metrics["device_map"] != expected or metrics.get("slabs") != len(expected):
            problems.append(f"K={metrics.get('slabs')} device_map={metrics['device_map']}, expected {expected}")
    printed = metrics.get("v7_switches")
    if build["solver"] == "v6":
        if printed:
            problems.append("a 'v7 switches' line in a v6 run")
        return {"ok": not problems, "problems": problems, "added_reported": None, "printed": printed}
    if printed is None:
        return {"ok": False, "problems": problems + ["no 'v7 switches' line"], "added_reported": False,
                "printed": None}
    for name, value in build["switches"].items():
        if name in printed and normalized_switch(printed[name]) != normalized_switch(value):
            problems.append(f"{name}={printed[name]} printed, {value} set")
    added = build["adds"]
    return {"ok": not problems, "problems": problems, "added_reported": (added in printed) if added else None,
            "printed": printed}


# ----------------------------------------------------------------------------- nvidia-smi: idle check and telemetry

GPU_QUERY_FIELDS = ("index", "uuid", "utilization.gpu", "memory.used", "power.draw", "clocks.sm", "temperature.gpu",
                    "display_active")
APPLICATION_QUERY_FIELDS = ("pid", "process_name", "gpu_uuid", "used_memory")
TELEMETRY_FIELDS = ("timestamp", "index", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu",
                    "utilization.gpu", "memory.used")
TELEMETRY_KEYS = ("sm_clock_mhz", "memory_clock_mhz", "power_watts", "temperature_celsius", "utilization_percent",
                  "memory_used_mib")


def parse_gpu_rows(text: str) -> list:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(GPU_QUERY_FIELDS) or not parts[0].isdigit():
            continue
        rows.append({"index": parts[0], "uuid": parts[1], "utilization_percent": number_or_none(parts[2]),
                     "memory_used_mib": number_or_none(parts[3]), "power_watts": number_or_none(parts[4]),
                     "sm_clock_mhz": number_or_none(parts[5]), "temperature_celsius": number_or_none(parts[6]),
                     "display_active": parts[7]})
    return rows


def parse_compute_applications(text: str) -> list:
    """pid, name, gpu uuid, memory; the name is everything between the first and the last two fields (a path could
    hold ', ')."""
    applications = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(", ")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        applications.append({"pid": int(parts[0]), "name": ", ".join(parts[1:-2]), "gpu_uuid": parts[-2],
                             "used_memory": parts[-1]})
    return applications


def idle_verdict(gpus: list, applications: list, display_limit: float = DISPLAY_UTILIZATION_LIMIT_PERCENT,
                 headless_limit: float = HEADLESS_UTILIZATION_LIMIT_PERCENT) -> dict:
    """busy when: a python compute process on any GPU (another solver run, or a leftover); any compute process on a
    GPU without a display; utilization above the limit (the display GPU keeps a desktop baseline). The display GPU =
    display_active Enabled; when none reports a display, GPU 0 (this rig's desktop GPU)."""
    display = {gpu["uuid"] for gpu in gpus if gpu["display_active"].lower() == "enabled"}
    if not display:
        display = {gpu["uuid"] for gpu in gpus if gpu["index"] == "0"}
    python_processes = [f"{item['pid']} {item['name']}" for item in applications if "python" in item["name"].lower()]
    headless_processes = [f"{item['pid']} {item['name']} on {item['gpu_uuid']}" for item in applications
                          if item["gpu_uuid"] not in display and "python" not in item["name"].lower()]
    reasons = []
    if python_processes:
        reasons.append(f"python compute process(es): {python_processes}")
    if headless_processes:
        reasons.append(f"compute process(es) on a GPU without a display: {headless_processes}")
    for gpu in gpus:
        limit = display_limit if gpu["uuid"] in display else headless_limit
        utilization = gpu["utilization_percent"]
        if utilization is not None and utilization > limit:
            reasons.append(f"GPU {gpu['index']} utilization {utilization:.0f} % > {limit:.0f} %")
    return {"busy": bool(reasons), "reasons": reasons, "gpus": gpus, "display_uuids": sorted(display),
            "python_processes": python_processes, "headless_processes": headless_processes}


def parse_telemetry_line(line: str, arrival_time: float) -> Optional[dict]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(TELEMETRY_FIELDS) or not parts[1].isdigit():
        return None
    try:
        sample_time = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
    except ValueError:
        sample_time = arrival_time
    sample = {"time": sample_time, "gpu": parts[1]}
    sample.update({key: number_or_none(text) for key, text in zip(TELEMETRY_KEYS, parts[2:])})
    return sample


def telemetry_summary(samples: list, start: float, end: float) -> dict:
    """Per GPU (nvidia-smi index): sample count and median / min / max of every telemetry key over [start, end]."""
    summary = {}
    for gpu in sorted({sample["gpu"] for sample in samples}):
        inside = [sample for sample in samples if sample["gpu"] == gpu and start <= sample["time"] <= end]
        entry = {"samples": len(inside)}
        for key in TELEMETRY_KEYS:
            values = [sample[key] for sample in inside if sample.get(key) is not None]
            entry[key] = ({"median": statistics.median(values), "min": min(values), "max": max(values)}
                          if values else None)
        summary[gpu] = entry
    return summary


class TelemetrySampler:
    """nvidia-smi --query-gpu=TELEMETRY_FIELDS --loop-ms=TELEMETRY_INTERVAL_MS in the background: every line goes to
    the CSV file and, parsed, to samples (time = nvidia-smi's own timestamp column, local time)."""

    def __init__(self, csv_path: pathlib.Path, interval_ms: int = TELEMETRY_INTERVAL_MS):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples: list = []
        self.error = None
        self.stream = open(csv_path, "w", encoding="utf-8", newline="\n")
        self.stream.write(",".join(TELEMETRY_FIELDS) + "\n")
        self.process = subprocess.Popen(["nvidia-smi", "--query-gpu=" + ",".join(TELEMETRY_FIELDS),
                                         "--format=csv,noheader,nounits", f"--loop-ms={interval_ms}"],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        text=True, encoding="utf-8", errors="replace")
        self.reader = threading.Thread(target=self._read, name="telemetry", daemon=True)
        self.reader.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.stream.write(line)
            sample = parse_telemetry_line(line, time.time())
            if sample is not None:
                self.samples.append(sample)
            elif line.strip() and self.error is None:
                self.error = line.strip()

    def stop(self) -> list:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.reader.join(timeout=10)
        self.stream.close()
        return self.samples


class NvidiaSmiBackend:
    """The GPU state of the run mode: nvidia-smi queries (idle check) and the telemetry sampler."""

    def query(self) -> dict:
        gpu_text = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=60)
        application_text = subprocess.run(["nvidia-smi", "--query-compute-apps=" + ",".join(APPLICATION_QUERY_FIELDS),
                                           "--format=csv,noheader"], capture_output=True, text=True, encoding="utf-8",
                                          errors="replace", timeout=60)
        if gpu_text.returncode != 0 or application_text.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {gpu_text.stdout[-500:]} {gpu_text.stderr[-500:]} "
                               f"{application_text.stderr[-500:]}")
        gpus = parse_gpu_rows(gpu_text.stdout)
        if not gpus:
            raise RuntimeError(f"nvidia-smi listed no GPU: {gpu_text.stdout[-500:]}")
        return idle_verdict(gpus, parse_compute_applications(application_text.stdout))

    def start_telemetry(self, csv_path: pathlib.Path):
        try:
            return TelemetrySampler(csv_path)
        except OSError as error:                 # the run still goes ahead; its record says why it has no clocks
            return UnavailableSampler(f"nvidia-smi telemetry did not start: {error}")


class UnavailableSampler:
    def __init__(self, error: str):
        self.error = error

    def stop(self) -> list:
        return []


def wait_for_idle_gpus(backend, log, settle_seconds: float = IDLE_SETTLE_SECONDS,
                       retry_seconds: tuple = IDLE_RETRY_SECONDS) -> dict:
    """Settle (nvidia-smi's utilization lags about a second behind a run that just ended), then check until idle;
    every busy check is logged. Never returns while busy: the campaign waits for another job to finish. A failing
    nvidia-smi is retried the same way (the state cannot be verified, so nothing starts)."""
    time.sleep(settle_seconds)
    waited, checks = settle_seconds, 0
    while True:
        checks += 1
        try:
            state = backend.query()
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            state = {"busy": True, "reasons": [f"nvidia-smi unusable: {error}"], "gpus": []}
        if not state["busy"]:
            state.update({"waited_s": round(waited, 1), "checks": checks})
            return state
        pause = retry_seconds[min(checks - 1, len(retry_seconds) - 1)]
        log(f"GPUs busy (check {checks}, waited {waited:.0f} s), next check in {pause:.0f} s: "
            + "; ".join(state["reasons"]))
        time.sleep(pause)
        waited += pause


# ----------------------------------------------------------------------------- device mapping

DEVICE_PROBE_PREFIX = "DEVICE_PROBE "


def device_probe_main() -> int:
    """Subprocess side: VkPhysicalDeviceIDProperties.deviceUUID of every physical device in VulkanContextV7's order
    (discrete GPUs first, stable): the runner's --device-map index. Instance only, no logical device (the pattern of
    _run_single_baseline_bench.resolve_device_index)."""
    from vulkan import (VK_MAKE_VERSION, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VkApplicationInfo,
                        VkInstanceCreateInfo, VkPhysicalDeviceIDProperties, VkPhysicalDeviceProperties2,
                        vkCreateInstance, vkDestroyInstance, vkEnumeratePhysicalDevices,
                        vkGetInstanceProcAddr, vkGetPhysicalDeviceProperties)
    application_info = VkApplicationInfo(pApplicationName="e39_perf_device_probe", applicationVersion=1,
                                         pEngineName="e39_perf_device_probe", engineVersion=1,
                                         apiVersion=VK_MAKE_VERSION(1, 1, 0))
    instance = vkCreateInstance(VkInstanceCreateInfo(
        pApplicationInfo=application_info, enabledExtensionCount=1,
        ppEnabledExtensionNames=["VK_KHR_get_physical_device_properties2"]), None)
    entries = []
    try:
        get_properties2 = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR")
        for raw_index, physical_device in enumerate(vkEnumeratePhysicalDevices(instance)):
            id_properties = VkPhysicalDeviceIDProperties()
            properties2 = VkPhysicalDeviceProperties2(pNext=id_properties)
            get_properties2(physical_device, properties2)
            properties = vkGetPhysicalDeviceProperties(physical_device)
            entries.append({"raw_index": raw_index, "uuid": bytes(id_properties.deviceUUID).hex(),
                            "name": str(properties.deviceName),
                            "discrete": properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU})
    finally:
        vkDestroyInstance(instance, None)
    ordered = sorted(entries, key=lambda entry: 0 if entry["discrete"] else 1)
    for index, entry in enumerate(ordered):
        entry["index"] = index
    print(DEVICE_PROBE_PREFIX + json.dumps(ordered), flush=True)
    return 0


def match_devices(vulkan_devices: list, gpus: list) -> dict:
    """Vulkan index (str) -> nvidia-smi index (str) by UUID (nvidia-smi 'GPU-xxxxxxxx-...' = deviceUUID hex)."""
    by_uuid = {gpu["uuid"].lower().replace("gpu-", "").replace("-", ""): gpu["index"] for gpu in gpus}
    return {str(device["index"]): by_uuid[device["uuid"].lower()] for device in vulkan_devices
            if device["uuid"].lower() in by_uuid}


def resolve_device_mapping(backend, log) -> dict:
    """{"mapping": vulkan -> nvidia-smi index, "source": ...}; identity over K1_DEVICE_MAPS when the probe fails."""
    identity = {device: device for device in K1_DEVICE_MAPS}
    try:
        completed = subprocess.run([sys.executable, "-m", "experiment.seam_audit.e39_perf_campaign", "device-probe"],
                                   cwd=_REPOSITORY_ROOT, env={**clean_environment(), "PYTHONIOENCODING": "utf-8"},
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        line = next(line for line in completed.stdout.splitlines() if line.startswith(DEVICE_PROBE_PREFIX))
        devices = json.loads(line[len(DEVICE_PROBE_PREFIX):])
        mapping = match_devices(devices, backend.query()["gpus"])
        if not all(device in mapping for device in K1_DEVICE_MAPS):
            raise RuntimeError(f"devices {K1_DEVICE_MAPS} not all matched: {mapping}")
        log(f"device mapping (Vulkan discrete-first index -> nvidia-smi index, by deviceUUID): {mapping}; "
            + "; ".join(f"[{device['index']}] {device['name']} {device['uuid']}" for device in devices))
        return {"mapping": mapping, "source": "deviceUUID probe", "devices": devices}
    except (StopIteration, RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        log(f"*** device probe failed ({error}): telemetry assumes Vulkan index = nvidia-smi index ***")
        return {"mapping": identity, "source": f"assumed identity (probe failed: {error})", "devices": []}


# ----------------------------------------------------------------------------- processes

_LIVE_PROCESSES: set = set()


def kill_process_tree(process: subprocess.Popen) -> None:
    """Kill a runner and everything it started. Windows: taskkill /T (the venv's python.exe is a launcher that runs
    the interpreter as a child); POSIX: the process group (started with start_new_session)."""
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, text=True,
                           timeout=60)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass
    for _ in range(2):              # never raises: it runs in finally blocks; a survivor is reported by the idle check
        try:
            process.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            process.kill()
    print(f"{LOG_PREFIX} *** process {process.pid} survived the kill: the next idle check waits for it ***", flush=True)


def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        known = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(known) and code.value == 259                            # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def kill_live_processes() -> None:
    for runner in list(_LIVE_PROCESSES):
        kill_process_tree(runner.process)
        _LIVE_PROCESSES.discard(runner)


class RunnerProcess:
    """One runner subprocess: stdout and stderr into its log file, read live by a thread that keeps the wall time of
    the first line and of MARKER_PATTERNS' lines (the run environment has PYTHONUNBUFFERED=1)."""

    def __init__(self, command: list, environment: dict, log_path: pathlib.Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.command = command
        self.log_path = log_path
        self.markers: dict = {}
        self.ended = None
        self.stream = open(log_path, "w", encoding="utf-8", newline="\n", buffering=1)     # line-buffered: tail -f
        options = {"start_new_session": True} if sys.platform != "win32" else {}
        self.started = time.time()
        self.process = subprocess.Popen(command, cwd=_REPOSITORY_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", **options)
        _LIVE_PROCESSES.add(self)
        self.reader = threading.Thread(target=self._read, name=f"reader {log_path.name}", daemon=True)
        self.reader.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            now = time.time()
            self.stream.write(line)
            text = line.rstrip("\r\n")
            self.markers.setdefault("first_output", now)
            for name, pattern in MARKER_PATTERNS:
                if name not in self.markers and pattern.match(text):
                    self.markers[name] = now

    def wait(self, deadline: float) -> tuple:
        """(exit code, timed out) by the absolute deadline; past it the process tree is killed."""
        timed_out = False
        try:
            code = self.process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(self.process)
            code = self.process.poll()
        self.finish()
        return code, timed_out

    def finish(self) -> None:
        if self.ended is None:
            self.reader.join(timeout=30)
            self.stream.close()
            self.ended = time.time()
            _LIVE_PROCESSES.discard(self)


# ----------------------------------------------------------------------------- run


def steady_window(markers: dict, metrics: dict) -> tuple:
    """(start, end, source): the steady window on the wall clock from the TOTAL line's arrival and the STEADY
    seconds; the whole loop (ready -> TOTAL) or the process when the markers are missing."""
    loop_end = markers.get("loop_end")
    steady = metrics.get("steady")
    if loop_end is not None and steady is not None:
        return loop_end - steady["seconds"], loop_end, "steady"
    if loop_end is not None and markers.get("ready") is not None:
        return markers["ready"], loop_end, "loop"
    return None, None, "process"


def unit_record_base(unit: dict, mode: str, environment: dict, device_mapping: dict, digests: dict,
                     plan_configuration: dict, unavailable: list) -> dict:
    configuration = CONFIGURATIONS[unit["configuration"]]
    return {"schema": LEDGER_SCHEMA, "mode": mode, "unit": unit["unit"], "trial": unit["trial"],
            "configuration": unit["configuration"], "case": configuration["case"],
            "case_path": CASES[configuration["case"]]["path"], "slabs": slab_count(unit["configuration"]),
            "device_maps": unit["device_maps"], "build": unit["build"], "solver": BUILDS[unit["build"]]["solver"],
            "switch_environment": BUILDS[unit["build"]]["switches"],
            "environment_difference": environment_difference(environment),
            "max_steps": unit["steps"][0], "warmup": unit["steps"][1], "timeout_seconds": unit["timeout_seconds"],
            "code": digests, "git": git_state(), "device_mapping": device_mapping,
            "skipped_builds": plan_configuration["skipped"], "run_builds": plan_configuration["run"],
            "unavailable_builds": unavailable, "host": {"cpu_count": os.cpu_count(),
                                                        "memory_available_gib": host_memory_available_gib()}}


def run_unit(unit: dict, out_directory: pathlib.Path, mode: str, backend, campaign_log, device_mapping: dict,
             digests: dict, plan: dict, command_factory: Callable = runner_command, settle_seconds: float =
             IDLE_SETTLE_SECONDS, trace_detail: str = "phases", retry_seconds: tuple = IDLE_RETRY_SECONDS) -> dict:
    """One run unit: idle check, telemetry, the process(es) (K = 1: both GPUs at once), the record."""
    def log(message: str) -> None:
        log_line(campaign_log, message)
    environment = run_environment(unit["build"])
    attempt = unit["attempts"] + 1
    stem = unit_stem(unit["trial"], unit["configuration"], unit["build"]) + f"__a{attempt}"
    record = unit_record_base(unit, mode, environment, device_mapping, digests,
                              plan["configurations"][unit["configuration"]], plan["unavailable"])
    record["attempt"] = attempt
    idle = wait_for_idle_gpus(backend, log, settle_seconds, retry_seconds)
    record["idle_check"] = idle
    log(f"start {unit['unit']} (attempt {attempt}, {' | '.join(unit['device_maps'])}, timeout "
        f"{unit['timeout_seconds']:.0f} s): idle after {idle['waited_s']} s, GPUs "
        + "; ".join(f"{gpu['index']}: {format_number(gpu['utilization_percent'], 0)} % "
                    f"{format_number(gpu['sm_clock_mhz'], 0)} MHz {format_number(gpu['power_watts'], 0)} W "
                    f"{format_number(gpu['temperature_celsius'], 0)} C" for gpu in idle.get("gpus", [])))
    telemetry_path = out_directory / "telemetry" / f"{stem}.csv"
    sampler = backend.start_telemetry(telemetry_path)
    runners, results = [], []
    record["start_unix"] = time.time()
    record["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    interrupted = False
    try:
        for device_map in unit["device_maps"]:
            trace_directory = None
            if mode == "trace":
                trace_directory = out_directory / "traces" / stem
                if trace_directory.exists():
                    shutil.rmtree(trace_directory)
            command = command_factory(unit["build"], unit["configuration"], device_map, trace_directory, trace_detail)
            log_path = out_directory / "runs" / f"{stem}__dev{device_map.replace(',', '')}.log"
            runners.append((device_map, command, trace_directory, RunnerProcess(command, environment, log_path)))
        deadline = record["start_unix"] + unit["timeout_seconds"]
        for device_map, command, trace_directory, runner in runners:
            code, timed_out = runner.wait(deadline)
            results.append((device_map, command, trace_directory, runner, code, timed_out))
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        for _, _, _, runner in runners:
            kill_process_tree(runner.process)
            runner.finish()
        samples = sampler.stop()
        record["end_unix"] = time.time()
        record["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["telemetry_file"] = telemetry_path.relative_to(out_directory).as_posix()
        record["telemetry_error"] = getattr(sampler, "error", None)
        record["telemetry"] = telemetry_summary(samples, record["start_unix"], record["end_unix"])
        if interrupted:
            record.update({"valid": False, "interrupted": True, "processes": []})
            append_ledger(out_directory / "ledger.jsonl", record)
    processes = []
    for device_map, command, trace_directory, runner, code, timed_out in results:
        metrics = parse_runner_log(runner.log_path.read_text(encoding="utf-8", errors="replace"))
        verdict = invariant_verdict(metrics, code, timed_out)
        provenance = provenance_check(unit["build"], metrics, unit["configuration"], device_map)
        window_start, window_end, window_source = steady_window(runner.markers, metrics)
        if window_start is None:
            window_start, window_end = runner.started, runner.ended
        gpus = sorted({device_mapping["mapping"].get(device, device) for device in device_map.split(",")})
        entry = {"device_map": device_map, "gpus": gpus, "command": command,
                 "log": runner.log_path.relative_to(out_directory).as_posix(), "exit_code": code,
                 "timed_out": timed_out, "started_unix": runner.started,
                 "wall_s": round(runner.ended - runner.started, 2),
                 "markers_s": {name: round(value - runner.started, 3) for name, value in runner.markers.items()},
                 "steady_window": {"source": window_source, "start_s": round(window_start - runner.started, 3),
                                   "end_s": round(window_end - runner.started, 3)},
                 "metrics": metrics, "invariants": verdict, "switch_check": provenance,
                 "telemetry": telemetry_summary(samples, window_start, window_end)}
        if trace_directory is not None:
            entry["trace_directory"] = trace_directory.relative_to(out_directory).as_posix()
            entry["trace_written"] = (trace_directory / "run_meta.json").exists()
        processes.append(entry)
    record["processes"] = processes
    record["valid"] = bool(processes) and all(
        entry["exit_code"] == 0 and not entry["timed_out"] and entry["invariants"]["ok"]
        and entry["switch_check"]["ok"] and entry["metrics"]["fps"] is not None
        and entry.get("trace_written", True) for entry in processes)
    append_ledger(out_directory / "ledger.jsonl", record)
    for entry in processes:
        problems = entry["invariants"]["violations"] + entry["switch_check"]["problems"]
        log(f"  {unit['unit']} dev {entry['device_map']}: exit {entry['exit_code']}"
            + (" TIMED OUT" if entry["timed_out"] else "") + f" in {entry['wall_s']:.1f} s, steady fps "
            f"{format_number(entry['metrics']['fps'], 2)}"
            + (f" *** {'; '.join(problems)} ***" if problems else "")
            + (f" (added switch {BUILDS[unit['build']]['adds']} not in the header)"
               if entry["switch_check"]["added_reported"] is False else ""))
    return record


def run_campaign(out_directory: pathlib.Path, configuration_names: list, build_names: list, trials: int,
                 resume: bool, dry_run: bool, preflight: bool = False, mode: str = "campaign",
                 trace_detail: str = "phases", backend=None, command_factory: Callable = runner_command,
                 device_mapping: Optional[dict] = None, settle_seconds: float = IDLE_SETTLE_SECONDS,
                 deep_wall: Optional[dict] = None, timeout_override: Optional[float] = None,
                 refresh_deep_wall: bool = False, retry_seconds: tuple = IDLE_RETRY_SECONDS,
                 availability: Optional[dict] = None, code_digests: Optional[dict] = None,
                 band_overlap: Optional[dict] = None) -> int:
    """plan / run / trace: build the plan, print it (dry run) or execute every unit not done yet. The keyword
    arguments after refresh_deep_wall exist for the self-test (fake GPU backend and runners, fixed digests).
    refresh_deep_wall recomputes both probe caches (deep-wall and band-overlap); the band-overlap rule is resolved
    only when b3 is among the builds."""
    ledger_path = out_directory / "ledger.jsonl" if out_directory is not None else None
    records = read_ledger(ledger_path) if ledger_path is not None else []
    if records and not resume and not dry_run:
        print(f"{LOG_PREFIX} {ledger_path} holds {len(records)} record(s): pass --resume to continue it, or choose "
              f"a new --out", flush=True)
        return 2
    digests = code_digests or {solver: solver_code_digest(solver) for solver in RUNNERS}
    if deep_wall is None:
        deep_wall = deep_wall_resolutions(configuration_names, default_deep_wall_cache(out_directory),
                                          lambda message: print(f"{LOG_PREFIX} {message}", flush=True),
                                          refresh=refresh_deep_wall)
    if band_overlap is None and "b3" in build_names:
        band_overlap = band_overlap_resolutions(configuration_names, default_band_overlap_cache(out_directory),
                                                lambda message: print(f"{LOG_PREFIX} {message}", flush=True),
                                                refresh=refresh_deep_wall)
    plan = build_plan(configuration_names, build_names, trials, deep_wall, records, digests, mode, availability,
                      band_overlap)
    if mode == "trace":
        plan = trace_plan(plan, build_names)
    if timeout_override is not None:
        for unit in plan["units"]:
            unit["timeout_seconds"] = timeout_override
    missing = missing_case_inputs(configuration_names)
    if dry_run:
        print_plan(plan, out_directory, digests)
        for case_key, paths in missing.items():
            print(f"{LOG_PREFIX} *** {case_key}: missing input(s) {paths} (the .obj files are gitignored: copy them "
                  f"from the checkout that generated the case) ***", flush=True)
        if preflight:
            return run_preflight(configuration_names)
        return 0
    if missing:
        print(f"{LOG_PREFIX} refusing to start, missing case inputs: {missing}", flush=True)
        return 2
    out_directory.mkdir(parents=True, exist_ok=True)
    campaign_log = out_directory / "campaign.log"
    (out_directory / f"plan_{mode}.json").write_text(json.dumps(json_ready(plan), indent=1), encoding="utf-8")
    backend = backend or NvidiaSmiBackend()
    if device_mapping is None:
        device_mapping = resolve_device_mapping(backend, lambda message: log_line(campaign_log, message))
    to_run = [unit for unit in plan["units"] if unit["status"] != "done"]
    log_line(campaign_log, f"{mode} start: {len(to_run)} of {len(plan['units'])} unit(s) to run, builds "
                           f"{plan['selected']}, code v6 {digests['v6']} v7 {digests['v7']}"
             + (f"; LEFT OUT (switch not in the code): {plan['unavailable']}" if plan["unavailable"] else ""))
    failures = 0
    try:
        for unit in plan["units"]:
            if unit["status"] == "done":
                continue
            if unit["status"] in ("stale", "failed"):
                log_line(campaign_log, f"{unit['unit']}: {unit['status']} before, run again")
            # the digest is taken again right before the unit: the record names the code that unit executed
            digests_now = code_digests or {solver: solver_code_digest(solver) for solver in RUNNERS}
            record = run_unit(unit, out_directory, mode, backend, campaign_log, device_mapping, digests_now, plan,
                              command_factory, settle_seconds, trace_detail, retry_seconds)
            failures += not record["valid"]
    except KeyboardInterrupt:
        log_line(campaign_log, "interrupted: running processes killed")
        return 130
    finally:
        kill_live_processes()
    log_line(campaign_log, f"{mode} finished: {failures} unit(s) without a valid result")
    if mode == "trace":
        summarize_traces(out_directory)
    else:
        summarize_campaign(out_directory)
    return 1 if failures else 0


def run_preflight(configuration_names: list) -> int:
    """--preflight: load and partition every case of the plan (2-D too) in the probe subprocess; CPU only."""
    failures = 0
    by_case = {}
    for name in configuration_names:
        by_case.setdefault(CONFIGURATIONS[name]["case"], set()).add(",".join(["1"] * slab_count(name)))
    for case_key, weight_texts in by_case.items():
        try:
            probe = run_deep_wall_probe(case_key, sorted(weight_texts, key=len),
                                        lambda message: print(f"{LOG_PREFIX} {message}", flush=True))
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            failures += 1
            print(f"{LOG_PREFIX} preflight {case_key}: FAIL {error}", flush=True)
            continue
        print(f"{LOG_PREFIX} preflight {case_key}: {probe['particles']:,} particles, {probe['dimension']}-D, "
              f"{probe['seconds']} s", flush=True)
        for result in probe["results"]:
            print(f"    weights {result['weights']}: cuts {result['cuts']}; " + "; ".join(
                f"s{slab['slab']} columns {slab['own_columns']} {slab['particles']:,} particles, deep-wall "
                f"{'ON' if slab['active'] else 'off'} ({slab['reason']})"
                + (f", band-overlap chain verdict {'ON' if slab['band_overlap'][0] else 'off'} "
                   f"({slab['band_overlap'][1]})" if slab.get("band_overlap") else "")
                for slab in result["slabs"]))
    return 1 if failures else 0


# ----------------------------------------------------------------------------- summary


def effective_records(records: list, mode: str) -> tuple:
    """(unit -> its last valid record, unit -> every record) of one mode."""
    effective, attempts = {}, {}
    for record in records:
        if record.get("mode", "campaign") != mode:
            continue
        attempts.setdefault(record["unit"], []).append(record)
        if record.get("valid"):
            effective[record["unit"]] = record
    return effective, attempts


def process_fps(record: dict) -> dict:
    """device map -> steady fps of each process of a record (K = 1: '0' and '1')."""
    return {process["device_map"]: process["metrics"]["fps"] for process in record.get("processes", [])}


class CampaignView:
    """The ledger seen through the plan's skip decisions: the value of (configuration, build, trial), inherited from
    the run build when the build was skipped there."""

    def __init__(self, records: list, mode: str = "campaign"):
        self.records = [record for record in records if record.get("mode", "campaign") == mode]
        self.effective, self.attempts = effective_records(records, mode)
        self.skipped: dict = {}
        self.measured: dict = {}
        self.unavailable: set = set()
        for record in self.records:
            self.skipped[record["configuration"]] = record.get("skipped_builds") or {}
            self.measured.setdefault(record["configuration"], set()).add(record["build"])
            self.unavailable.update(record.get("unavailable_builds") or [])
        self.configurations = [name for name in CONFIGURATIONS if name in self.measured]
        self.trials = sorted({record["trial"] for record in self.records})

    def chain(self, configuration_name: str) -> list:
        present = self.measured.get(configuration_name, set()) | set(self.skipped.get(configuration_name, {}))
        return [name for name in CHAIN if name in present]

    def inherited(self, configuration_name: str, build_name: str) -> bool:
        """Skipped there by the plan and never measured there (a measured record wins over the skip decision)."""
        return (build_name in self.skipped.get(configuration_name, {})
                and build_name not in self.measured.get(configuration_name, set()))

    def record(self, configuration_name: str, build_name: str, trial: int) -> Optional[dict]:
        """The build's own valid record of that trial, else (skipped there) its predecessor's, recursively."""
        skipped = self.skipped.get(configuration_name, {})
        while True:
            record = self.effective.get(unit_identifier(trial, configuration_name, build_name))
            if record is not None or not self.inherited(configuration_name, build_name):
                return record
            build_name = skipped[build_name]["previous"]

    def fps(self, configuration_name: str, build_name: str, trial: int) -> Optional[dict]:
        """{"value": fps (K = 1: mean of the two GPUs), "devices": {map: fps}} or None."""
        record = self.record(configuration_name, build_name, trial)
        if record is None:
            return None
        devices = process_fps(record)
        if any(value is None for value in devices.values()) or not devices:
            return None
        return {"value": statistics.fmean(devices.values()), "devices": devices, "record": record}


def ratio_statistics(view: CampaignView, configuration_name: str, numerator: str, denominator: str) -> dict:
    """Paired by trial: fps(numerator, t) / fps(denominator, t) over the trials where both exist."""
    ratios = {}
    for trial in view.trials:
        top = view.fps(configuration_name, numerator, trial)
        bottom = view.fps(configuration_name, denominator, trial)
        if top and bottom and bottom["value"]:
            ratios[trial] = top["value"] / bottom["value"]
    mean, deviation, count = mean_and_deviation(ratios.values())
    return {"per_trial": ratios, "mean": mean, "standard_deviation": deviation, "trials": count}


def reference_configuration(configuration_name: str) -> Optional[str]:
    """The K = 1 configuration of the same case (the efficiency's reference)."""
    case_key = CONFIGURATIONS[configuration_name]["case"]
    return next((name for name, configuration in CONFIGURATIONS.items()
                 if configuration["case"] == case_key and configuration["device_map"] is None), None)


def sm_clock_median(process: dict, gpu: str):
    entry = (process.get("telemetry") or {}).get(gpu) or {}
    return (entry.get("sm_clock_mhz") or {}).get("median")


def efficiency_statistics(view: CampaignView, configuration_name: str, build_name: str) -> Optional[dict]:
    """eta = fps_K / (G mean_i fps_1(GPU i)), eta_min with min_i, per trial, against the simultaneous K = 1 runs of
    the same build (inherited where skipped), case and trial; G = the distinct GPUs of the device map."""
    reference = reference_configuration(configuration_name)
    if reference is None:
        return None
    gpus = sorted(set(CONFIGURATIONS[configuration_name]["device_map"].split(",")))
    per_trial = {}
    for trial in view.trials:
        multi = view.fps(configuration_name, build_name, trial)
        single = view.fps(reference, build_name, trial)
        if not multi or not single or not all(gpu in single["devices"] for gpu in gpus):
            continue
        references = [single["devices"][gpu] for gpu in gpus]
        mean_reference = statistics.fmean(references)
        multi_process = multi["record"]["processes"][0]
        per_trial[trial] = {
            "fps_k": multi["value"], "fps_k1": {gpu: single["devices"][gpu] for gpu in gpus},
            "efficiency": multi["value"] / (len(gpus) * mean_reference),
            "efficiency_minimum": multi["value"] / (len(gpus) * min(references)),
            "reference_spread": (max(references) - min(references)) / mean_reference,
            "k1_sm_clock_mhz": {process["device_map"]: sm_clock_median(process, (process.get("gpus") or ["?"])[0])
                                for process in single["record"]["processes"]},
            "k_sm_clock_mhz": {gpu: sm_clock_median(multi_process, gpu) for gpu in multi_process.get("gpus", [])}}
    efficiency = mean_and_deviation(item["efficiency"] for item in per_trial.values())
    minimum = mean_and_deviation(item["efficiency_minimum"] for item in per_trial.values())
    spread = mean_and_deviation(item["reference_spread"] for item in per_trial.values())
    devices = {gpu: mean_and_deviation(item["fps_k1"][gpu] for item in per_trial.values()) for gpu in gpus}
    return {"reference": reference, "gpus": gpus, "per_trial": per_trial,
            "efficiency": {"mean": efficiency[0], "standard_deviation": efficiency[1], "trials": efficiency[2]},
            "efficiency_minimum": {"mean": minimum[0], "standard_deviation": minimum[1]},
            "reference_spread": {"mean": spread[0], "standard_deviation": spread[1]},
            "reference_fps": {gpu: {"mean": value[0], "standard_deviation": value[1]}
                              for gpu, value in devices.items()}}


def skip_evidence(view: CampaignView, configuration_name: str, build_name: str) -> str:
    """What the later v7 runs of the configuration printed about the skipped build's switch."""
    later = [record for record in view.effective.values() if record["configuration"] == configuration_name
             and record["solver"] == "v7" and CHAIN.index(record["build"]) > CHAIN.index(build_name)]
    if build_name == "b4":
        states = [state for record in later for process in record["processes"]
                  for state in deep_wall_states(process["metrics"])]
        if not states:
            return "no later v7 run printed the auto resolution"
        if all(state == "off" for state in states):
            return f"confirmed: auto off on all {len(states)} slab resolutions of {len(later)} later run(s)"
        return f"CONTRADICTED: auto resolved {states.count('on')} of {len(states)} slab(s) on in later runs"
    if build_name == "b6":
        notes = [note for record in later for process in record["processes"]
                 for note in process["metrics"]["pipeline_notes"]]
        if not notes:
            return "no later v7 run printed its pipelines"
        if any("V7_GHOST_SEND_LANES=" in note for note in notes):
            return "CONTRADICTED: a later run built ghost_send lanes"
        return f"confirmed: no ghost_send pipeline in {len(later)} later run(s)"
    if build_name == "b3":
        return ("rule only: b3 is the last build, no later run prints V7_BAND_OVERLAP (K = 1 / 3-D: off under every "
                "value; 2-D: the probe's chain verdict in the reason)")
    return "rule only (not visible in the logs)"


# First words of a printed per-slab resolution that mean the switch does nothing on that slab
# ("[SimV7] V7_DEEP_WALL_SKIP=auto: off (...)", "[SimV7] V7_FUSED_CORRECTION_DENSITY=1: separate kernels (...)").
INERT_RESOLUTIONS = ("off", "separate kernels")


def printed_resolutions(view: CampaignView, configuration_name: str, chain: list) -> dict:
    """build -> switch -> the distinct per-slab resolutions ('[SimV7] V7_X=value: <resolution> (reason)', the part
    before the reason) its valid runs printed in this configuration (V7_FAST_SUBMIT's banner left out)."""
    resolutions = {}
    for build_name in chain:
        seen: dict = {}
        for trial in view.trials:
            record = view.effective.get(unit_identifier(trial, configuration_name, build_name))
            for process in (record or {}).get("processes", []):
                for switch, items in process["metrics"].get("switch_resolutions", {}).items():
                    if switch.endswith("FAST_SUBMIT"):
                        continue
                    for item in items:
                        seen.setdefault(switch, set()).add(item["resolution"].split(" (", 1)[0])
        if seen:
            resolutions[build_name] = {switch: sorted(states) for switch, states in sorted(seen.items())}
    return resolutions


def summarize_campaign(out_directory: pathlib.Path) -> tuple:
    """(exit code, summary dict): writes OUT/summary.md and OUT/summary.json; violations first."""
    records = read_ledger(out_directory / "ledger.jsonl")
    view = CampaignView(records, "campaign")
    summary = {"out": str(out_directory), "records": len(view.records), "trials": view.trials,
               "valid_units": len(view.effective), "violations": [], "notes": [], "configurations": {}, "runs": [],
               "unavailable_builds": sorted(view.unavailable)}
    # every attempt's invariants and provenance
    for record in view.records:
        superseded = view.effective.get(record["unit"]) is not record
        for process in record.get("processes", []):
            problems = process["invariants"]["violations"] + process["switch_check"]["problems"]
            row = {"unit": record["unit"], "attempt": record.get("attempt"), "device_map": process["device_map"],
                   "gpus": process.get("gpus"), "exit_code": process["exit_code"], "timed_out": process["timed_out"],
                   "fps": process["metrics"]["fps"], "final": process["metrics"]["final"],
                   "overflow": {sim: values.get("overflow") for sim, values in process["metrics"]["sims"].items()},
                   "far_migration": {sim: values.get("far_migration")
                                     for sim, values in process["metrics"]["sims"].items()},
                   "seams_ok": all(seam["ok"] for seam in process["metrics"]["seams"]),
                   "fields_ok": (process["metrics"]["fields"] or {}).get("ok"),
                   "validation_failed": process["metrics"]["validation_failed"], "problems": problems,
                   "added_switch_reported": process["switch_check"]["added_reported"],
                   "superseded": superseded,
                   "sm_clock_mhz": {gpu: (entry.get("sm_clock_mhz") or {}).get("median")
                                    for gpu, entry in (process.get("telemetry") or {}).items()},
                   "power_watts": {gpu: (entry.get("power_watts") or {}).get("median")
                                   for gpu, entry in (process.get("telemetry") or {}).items()},
                   "temperature_celsius_max": {gpu: (entry.get("temperature_celsius") or {}).get("max")
                                               for gpu, entry in (process.get("telemetry") or {}).items()},
                   "telemetry_samples": {gpu: entry.get("samples")
                                         for gpu, entry in (process.get("telemetry") or {}).items()},
                   "steady_window_source": (process.get("steady_window") or {}).get("source"),
                   "wall_s": process.get("wall_s")}
            summary["runs"].append(row)
            if problems:
                summary["violations"].append(f"{record['unit']} attempt {record.get('attempt')} dev "
                                             f"{process['device_map']}: {'; '.join(problems)}")
        if record.get("interrupted"):
            summary["violations"].append(f"{record['unit']} attempt {record.get('attempt')}: interrupted")
    # the switch provenance must be one code version per build
    for build_name in CHAIN:
        solver = BUILDS[build_name]["solver"]
        digests = {(record.get("code") or {}).get(solver) for record in view.effective.values()
                   if record["build"] == build_name}
        if len(digests) > 1:
            summary["violations"].append(f"build {build_name}: valid runs of {len(digests)} code versions "
                                         f"{sorted(map(str, digests))} (mixed code)")
    # fps, chain, efficiency per configuration
    for name in view.configurations:
        chain = view.chain(name)
        entry = {"label": CASES[CONFIGURATIONS[name]["case"]]["label"], "slabs": slab_count(name),
                 "device_map": CONFIGURATIONS[name]["device_map"] or "0 | 1", "builds": {}, "chain": [],
                 "efficiency": {}}
        for build_name in chain:
            skip = view.skipped.get(name, {}).get(build_name) if view.inherited(name, build_name) else None
            values = {trial: view.fps(name, build_name, trial) for trial in view.trials}
            item = {"status": "inherited" if skip else "measured", "skip": skip,
                    "trials_present": sorted(trial for trial, value in values.items() if value)}
            mean, deviation, count = mean_and_deviation(value["value"] for value in values.values() if value)
            item["fps"] = {"mean": mean, "standard_deviation": deviation, "trials": count,
                           "per_trial": {trial: value["value"] for trial, value in values.items() if value}}
            if entry["slabs"] == 1:
                item["devices"] = {}
                for device in K1_DEVICE_MAPS:
                    device_values = [value["devices"].get(device) for value in values.values() if value]
                    device_mean, device_deviation, device_count = mean_and_deviation(device_values)
                    item["devices"][device] = {"mean": device_mean, "standard_deviation": device_deviation,
                                               "trials": device_count}
            entry["builds"][build_name] = item
        for position, build_name in enumerate(chain):
            if position == 0:
                continue
            previous = chain[position - 1]
            row = {"build": build_name, "previous": previous,
                   "inherited": entry["builds"][build_name]["status"] == "inherited",
                   "against_previous": ratio_statistics(view, name, build_name, previous)}
            if REFERENCE_BUILD in chain and build_name != REFERENCE_BUILD:
                row["against_reference"] = ratio_statistics(view, name, build_name, REFERENCE_BUILD)
            entry["chain"].append(row)
        if entry["slabs"] > 1:
            for build_name in chain:
                statistics_entry = efficiency_statistics(view, name, build_name)
                if statistics_entry is not None:
                    entry["efficiency"][build_name] = statistics_entry
        entry["skip_evidence"] = {build_name: skip_evidence(view, name, build_name)
                                  for build_name in view.skipped.get(name, {}) if view.inherited(name, build_name)}
        for build_name, text in entry["skip_evidence"].items():
            if text.startswith("CONTRADICTED"):
                summary["violations"].append(f"{name} skip of {build_name}: {text}")
        entry["missing_units"] = [unit_identifier(trial, name, build_name) for trial in view.trials
                                  for build_name in chain if not view.inherited(name, build_name)
                                  and view.fps(name, build_name, trial) is None]
        entry["resolutions"] = printed_resolutions(view, name, chain)
        for build_name, switches in entry["resolutions"].items():
            added = BUILDS[build_name]["adds"]
            states = switches.get(added, [])
            if added and any(state in INERT_RESOLUTIONS for state in states):
                summary["notes"].append(f"{name} {build_name}: {added} printed {states} - its switch did not act on "
                                        f"every slab, the increment is partly inert there")
        summary["configurations"][name] = entry
    for row in summary["runs"]:
        if row["added_switch_reported"] is False and not row["problems"]:
            summary["notes"].append(f"{row['unit']} dev {row['device_map']}: the build's added switch is not in the "
                                    f"'v7 switches' header (provenance of that switch not verifiable from the log)")
    markdown = campaign_markdown(summary)
    out_directory.mkdir(parents=True, exist_ok=True)
    (out_directory / "summary.md").write_text(markdown, encoding="utf-8")
    (out_directory / "summary.json").write_text(json.dumps(json_ready(summary), indent=1), encoding="utf-8")
    print(f"{LOG_PREFIX} summary: {summary['records']} record(s), {summary['valid_units']} valid unit(s), trials "
          f"{summary['trials']}; wrote {out_directory / 'summary.md'} and summary.json", flush=True)
    if summary["violations"]:
        print(f"{LOG_PREFIX} *** {len(summary['violations'])} INVARIANT / PROVENANCE VIOLATION(S) ***", flush=True)
        for violation in summary["violations"]:
            print(f"    {violation}", flush=True)
    return (1 if summary["violations"] else 0), summary


def campaign_markdown(summary: dict) -> str:
    lines = ["# E39 performance campaign", "",
             f"Ledger records {summary['records']}, valid units {summary['valid_units']}, trials {summary['trials']}"
             + (f"; builds left out (switch not in the code): {summary['unavailable_builds']}"
                if summary["unavailable_builds"] else "") + ".", ""]
    if summary["violations"]:
        lines += [f"## INVARIANT / PROVENANCE VIOLATIONS: {len(summary['violations'])}", ""]
        lines += [f"- **VIOLATION** {violation}" for violation in summary["violations"]] + [""]
    else:
        lines += ["No invariant or provenance violation in any run attempt.", ""]
    if summary["notes"]:
        lines += [f"Notes ({len(summary['notes'])}, not violations):", ""]
        lines += [f"- {note}" for note in summary["notes"]] + [""]
    lines += ["## Steady fps per configuration and build (mean ± std over trials)", "",
              "K = 1: two processes at once, GPU 0 and GPU 1; 'mean' = per trial the mean of the two. Inherited = "
              "skipped (its switch is inert there), the value of the build it inherits from.", "",
              "| configuration | build | status | K = 1 GPU 0 | K = 1 GPU 1 | fps (K = 1: mean) | trials |",
              "|---|---|---|---|---|---|---|"]
    for name, entry in summary["configurations"].items():
        for build_name, item in entry["builds"].items():
            devices = item.get("devices") or {}
            status = (f"= {item['skip']['inherits']} ({item['skip']['reason']})" if item["skip"] else "measured")
            lines.append(f"| {name} | {build_name} | {status} | "
                         + " | ".join(format_mean(devices[device]["mean"], devices[device]["standard_deviation"])
                                      if device in devices else "" for device in K1_DEVICE_MAPS)
                         + f" | {format_mean(item['fps']['mean'], item['fps']['standard_deviation'])} | "
                           f"{item['fps']['trials']} |")
    lines += ["", "## Incremental chain (paired by trial: per-trial ratio, mean ± std, as a change)", "",
              "| configuration | build | vs previous | per trial | vs v6 | per trial |", "|---|---|---|---|---|---|"]
    for name, entry in summary["configurations"].items():
        for row in entry["chain"]:
            previous = row["against_previous"]
            reference = row.get("against_reference") or {}
            label = f"{row['build']} vs {row['previous']}" + (" (inherited)" if row["inherited"] else "")
            lines.append(f"| {name} | {label} | "
                         f"{format_percent(previous['mean'], previous['standard_deviation'])} | "
                         + ", ".join(f"t{trial} {format_percent(value, None)}"
                                     for trial, value in previous["per_trial"].items())
                         + f" | {format_percent(reference.get('mean'), reference.get('standard_deviation'))} | "
                         + ", ".join(f"t{trial} {format_percent(value, None)}"
                                     for trial, value in (reference.get("per_trial") or {}).items()) + " |")
    lines += ["", "## Local efficiency (K = 2; K = 4 shared = 0,1,0,1: two slabs per GPU, timesliced)", "",
              "η = fps_K / (G · mean_i fps_1(GPU i)), G = distinct GPUs of the device map (2), against the "
              "simultaneous K = 1 runs of the same build, case and trial; η_min with min_i; ± = std over the "
              "trial-wise η; spread = (max - min) / mean of the two K = 1 references. Fewer than 3 trials: not "
              "quotable.", "",
              "| configuration | build | η | η_min | trials | K = 1 GPU 0 / GPU 1 fps | reference spread | "
              "SM clock MHz K = 1 (GPU 0 / 1) | SM clock MHz K run (GPU 0 / 1) |",
              "|---|---|---|---|---|---|---|---|---|"]

    def as_percent(statistics_entry: dict, digits: int) -> str:
        return f"{format_mean(100 * statistics_entry['mean'], 100 * statistics_entry['standard_deviation'], digits)} %"

    def clock_cells(item: dict, key: str) -> str:
        return "; ".join(" / ".join(format_number(value, 0) for value in per_trial[key].values())
                         for per_trial in item["per_trial"].values())
    for name, entry in summary["configurations"].items():
        label = name + (" (K = 4 SHARED: 2 slabs per GPU)" if entry["slabs"] == 4 else "")
        for build_name, item in entry["efficiency"].items():
            trials = item["efficiency"]["trials"]
            references = " / ".join(format_mean(value["mean"], value["standard_deviation"])
                                    for value in item["reference_fps"].values())
            lines.append(f"| {label} | {build_name} | {as_percent(item['efficiency'], 1)} | "
                         f"{as_percent(item['efficiency_minimum'], 1)} | {trials}"
                         + (" (not quotable)" if trials < 3 else "")
                         + f" | {references} | {as_percent(item['reference_spread'], 2)} | "
                           f"{clock_cells(item, 'k1_sm_clock_mhz')} | {clock_cells(item, 'k_sm_clock_mhz')} |")
    lines += ["", "## Every run attempt: invariants, provenance, clocks", "",
              "fps = steady fps (the more precise of the printed fps and steps / seconds). SM clock = median over the "
              "process's steady window (nvidia-smi index: MHz). A superseded attempt was run again.", "",
              "| unit | attempt | dev | exit | fps | drift | overflow (non-zero) | far migr. | stamps gpu/host | "
              "seams | fields | verdict | SM clock MHz | power W | max °C | wall s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary["runs"]:
        final = row["final"] or {}
        nonzero = [f"s{sim} {name}={value}" for sim, counters in row["overflow"].items()
                   for name, value in (counters or {}).items() if value]
        counters_read = sum(len(counters or {}) for counters in row["overflow"].values())
        far = sum(value or 0 for value in row["far_migration"].values())
        verdict = ("VIOLATION: " + "; ".join(row["problems"])) if row["problems"] else "ok"
        if row["superseded"]:
            verdict += " (superseded)"
        if row["added_switch_reported"] is False:
            verdict += " (added switch not in the header)"
        lines.append(f"| {row['unit']} | {row['attempt']} | {row['device_map']} | {row['exit_code']}"
                     + (" TIMEOUT" if row["timed_out"] else "") + f" | {format_number(row['fps'], 2)} | "
                     f"{final.get('drift', '-')} | {', '.join(nonzero) or f'0 ({counters_read} counters)'} | {far} | "
                     f"{final.get('stamp_errors_gpu', '-')}/{final.get('stamp_errors_host', '-')} | "
                     f"{'ok' if row['seams_ok'] else 'FAIL'} | "
                     f"{'-' if row['fields_ok'] is None else ('ok' if row['fields_ok'] else 'FAIL')} | {verdict} | "
                     + " / ".join(f"{gpu}: {format_number(value, 0)}" for gpu, value in row["sm_clock_mhz"].items())
                     + " | " + " / ".join(format_number(value, 0) for value in row["power_watts"].values())
                     + " | " + " / ".join(format_number(value, 0) for value in row["temperature_celsius_max"].values())
                     + f" | {format_number(row['wall_s'], 1)} |")
    lines += ["", "## Skip rule evidence", "", "| configuration | skipped build | = | reason | evidence |",
              "|---|---|---|---|---|"]
    for name, entry in summary["configurations"].items():
        for build_name, text in entry.get("skip_evidence", {}).items():
            skip = entry["builds"][build_name]["skip"]
            lines.append(f"| {name} | {build_name} | {skip['inherits']} | {skip['reason']} | {text} |")
    lines += ["", "## Per-slab switch resolutions printed by the runs ('[SimV7] V7_X=value: <resolution> (...)')", "",
              "| configuration | build | resolutions |", "|---|---|---|"]
    for name, entry in summary["configurations"].items():
        for build_name, switches in entry.get("resolutions", {}).items():
            lines.append(f"| {name} | {build_name} | "
                         + "; ".join(f"{switch}: {', '.join(states)}" for switch, states in switches.items()) + " |")
    missing = [unit for entry in summary["configurations"].values() for unit in entry["missing_units"]]
    if missing:
        lines += ["", f"## Units without a valid result: {len(missing)}", "", ", ".join(missing)]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- trace mode


def trace_plan(plan: dict, build_names: list) -> dict:
    """Trace mode: per configuration v6 and the final v7 build (the last one the campaign plan runs there), unless
    --builds names builds explicitly (then those that run there)."""
    explicit = set(build_names) != set(CHAIN)
    for name, configuration in plan["configurations"].items():
        run_builds = configuration["run"]
        if explicit:
            chosen = [build_name for build_name in run_builds if build_name in build_names]
        else:
            finals = [build_name for build_name in run_builds if BUILDS[build_name]["solver"] == "v7"]
            chosen = [build_name for build_name in run_builds if build_name == REFERENCE_BUILD] + finals[-1:]
        configuration["trace_builds"] = chosen
    plan["units"] = [unit for unit in plan["units"]
                     if unit["build"] in plan["configurations"][unit["configuration"]]["trace_builds"]]
    return plan


def steady_mask(table: dict, warmup: int, cadence: int):
    """Steady complete rows, the step before a defrag boundary excluded (logs/e39/b4/tools/trace_summary.py)."""
    steps = table["step"]
    return (table["complete"] == 1) & (steps >= warmup) & ((steps + 1) % cadence != 0)


def median_or_nan(values) -> float:
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def trace_run_metrics(trace_directory: pathlib.Path) -> dict:
    """Per sim and per link medians (us) of one step trace (module docstring: definitions); step_trace_model.load_run
    reads it (fit-version-1 traces are re-mapped there), its link_metrics gives E29's t_chain and hops."""
    import numpy as np
    from experiment.v7.analysis import step_trace_model
    run = step_trace_model.load_run(trace_directory)
    meta = run["meta"]
    warmup, cadence = int(meta["warmup"]), int(meta["defrag_cadence"])
    device = run["device"]
    nan = float("nan")

    def column(name):
        return device[name] if name in device else np.full(len(device["step"]), np.nan)
    result = {"case": meta.get("case"), "K": meta.get("K"), "steady_fps": meta["result"].get("steady_fps"),
              "switches": meta.get("v7_switches"), "sims": {}, "links": {}}
    for sim in sorted(set(device["sim"].astype(int))):
        mask = steady_mask(device, warmup, cadence) & (device["sim"] == sim)
        a_start, a_end = column("a_start"), column("a_end")
        b_start, b_end = column("b_start"), column("b_end")
        c_start, c_end = column("c_start"), column("c_end")
        voxel_end = column("a_voxel_end")
        leading, trailing = column("a_ghost_leading_end"), column("a_ghost_trailing_end")
        ghost_end = np.where(np.isfinite(trailing), trailing, leading)
        intervals = {"phase_a": a_end - a_start, "ghost_send": ghost_end - voxel_end, "a_to_b": b_start - a_end,
                     "phase_b": b_end - b_start, "b_to_c": c_start - b_end, "phase_c": c_end - c_start}
        # phase B split at the end of correction + density (B1's fused kernel, else density_deep_interior; the
        # deep-wall marker of B4 sits in the first part), as step_trace_model.cycle_components splits it
        split = next((column(label) for label in ("b_correction_density_interior_end", "b_density_deep_interior_end")
                      if np.isfinite(column(label)[mask]).any()), None)
        if split is not None:
            intervals["phase_b_correction_density"] = split - b_start
            intervals["phase_b_force"] = b_end - split
        sim_rows = device["sim"] == sim
        complete = sim_rows & (device["complete"] == 1)
        next_a = {int(step): value for step, value in zip(device["step"][complete], a_start[complete])}
        following = np.array([next_a.get(int(step) + 1, np.nan) for step in device["step"]])
        intervals["c_to_next_a"] = following - c_end
        intervals["period"] = following - a_start
        entry = {}
        for key, values in intervals.items():
            chosen = values[mask] / 1000.0
            chosen = chosen[np.isfinite(chosen)]
            entry[key] = median_or_nan(chosen)
            if key == "period":
                entry["period_mean"] = float(np.mean(chosen)) if len(chosen) else nan
                entry["steps"] = int(len(chosen))
        result["sims"][str(sim)] = entry
    link = run["link"]
    if len(link["step"]):
        chain_metrics = step_trace_model.link_metrics(run)
        for name in sorted(set(link["link"])):
            mask = steady_mask(link, warmup, cadence) & (link["link"] == name)

            def span(start: str, end: str):
                return (link[end][mask] - link[start][mask]) / 1000.0
            transfer = span("send_end", "upload_end")
            transfer = transfer[np.isfinite(transfer)]
            exposure = span("receiver_b_end", "upload_end")
            exposure = exposure[np.isfinite(exposure)]
            chain = chain_metrics.get(name, {})
            result["links"][name] = {
                "readback_start_delay": median_or_nan(span("send_end", "readback_start")),
                "t_tr": median_or_nan(transfer),
                "t_tr_p95": float(np.percentile(transfer, 95)) if len(transfer) else nan,
                "upload_end_to_receiver_c_start": median_or_nan(span("upload_end", "receiver_c_start")),
                "receiver_phase_b": median_or_nan(span("receiver_b_start", "receiver_b_end")),
                "exposed_steps_fraction": float(np.mean(exposure > 0)) if len(exposure) else nan,
                **{key + "_e29": median_or_nan(chain.get(key, [])) for key in
                   ("t_chain", "readback_dma", "memcpy", "upload_dma")}}
    return result


SIM_TRACE_ROWS = (("phase_a", "phase A"), ("ghost_send", "ghost_send (a_voxel_end -> ghost end)"),
                  ("a_to_b", "A -> B gap"), ("phase_b", "phase B"),
                  ("phase_b_correction_density", "phase B: correction + density"),
                  ("phase_b_force", "phase B: force deep interior"), ("b_to_c", "B -> C wait"),
                  ("phase_c", "phase C"), ("c_to_next_a", "C -> next A"), ("period", "period p50"),
                  ("period_mean", "period mean"))
LINK_TRACE_ROWS = (("readback_start_delay", "send_end -> readback_start"),
                   ("t_tr", "t_tr p50 (send_end -> upload_end)"), ("t_tr_p95", "t_tr p95"),
                   ("t_chain_e29", "t_chain p50 (E29)"),
                   ("readback_dma_e29", "readback DMA p50 (E29)"), ("memcpy_e29", "host memcpy p50 (E29)"),
                   ("upload_dma_e29", "upload DMA p50 (E29)"),
                   ("upload_end_to_receiver_c_start", "upload_end -> receiver C start"),
                   ("receiver_phase_b", "receiver phase B"), ("exposed_steps_fraction", "exposed steps (fraction)"))


def summarize_traces(out_directory: pathlib.Path) -> int:
    """OUT/trace_summary.md + json: v6 against the final v7 build per traced configuration (mean over trials of the
    per-trace medians)."""
    records = read_ledger(out_directory / "ledger.jsonl")
    effective, attempts = effective_records(records, "trace")
    by_configuration = {}
    for record in effective.values():
        process = record["processes"][0]
        trace_directory = out_directory / process["trace_directory"]
        try:
            metrics = trace_run_metrics(trace_directory)
        except (OSError, KeyError, ValueError) as error:
            print(f"{LOG_PREFIX} {record['unit']}: trace unreadable ({error})", flush=True)
            continue
        by_configuration.setdefault(record["configuration"], {}).setdefault(record["build"], []).append(metrics)
    summary = {"configurations": {}}
    lines = ["# E39 step traces: v6 against the final v7 build", "",
             "Medians in µs over steady complete steps, the step before a defrag boundary excluded; mean over trials. "
             "E29 rows: experiment.v7.analysis.step_trace_model.link_metrics (all steady complete steps).", ""]
    for name, builds in by_configuration.items():
        order = [build_name for build_name in CHAIN if build_name in builds]
        averaged = {}
        for build_name in order:
            runs = builds[build_name]
            sims = sorted({sim for run in runs for sim in run["sims"]})
            links = sorted({link for run in runs for link in run["links"]})
            nan = float("nan")
            averaged[build_name] = {
                "trials": len(runs), "steady_fps": statistics.fmean(run["steady_fps"] for run in runs),
                "sims": {sim: {key: statistics.fmean(run["sims"][sim].get(key, nan) for run in runs
                                                     if sim in run["sims"])
                               for key, _ in SIM_TRACE_ROWS} for sim in sims},
                "links": {link: {key: statistics.fmean(run["links"][link].get(key, nan) for run in runs
                                                       if link in run["links"])
                                 for key, _ in LINK_TRACE_ROWS} for link in links}}
        summary["configurations"][name] = averaged
        lines += [f"## {name} ({CASES[CONFIGURATIONS[name]['case']]['label']})", "",
                  "| metric | " + " | ".join(f"{build_name} ({averaged[build_name]['trials']} trace(s), "
                                              f"{format_number(averaged[build_name]['steady_fps'], 1)} fps)"
                                              for build_name in order)
                  + (" | change |" if len(order) == 2 else " |"),
                  "|---|" + "---|" * (len(order) + (1 if len(order) == 2 else 0))]
        sims = sorted({sim for build_name in order for sim in averaged[build_name]["sims"]})
        for sim in sims:
            for key, label in SIM_TRACE_ROWS:
                values = [averaged[build_name]["sims"].get(sim, {}).get(key) for build_name in order]
                lines.append(f"| s{sim} {label} | " + " | ".join(format_number(value, 1) for value in values)
                             + (f" | {trace_change(values)} |" if len(order) == 2 else " |"))
        links = sorted({link for build_name in order for link in averaged[build_name]["links"]})
        for link in links:
            for key, label in LINK_TRACE_ROWS:
                values = [averaged[build_name]["links"].get(link, {}).get(key) for build_name in order]
                digits = 3 if key == "exposed_steps_fraction" else 1
                lines.append(f"| {link} {label} | " + " | ".join(format_number(value, digits) for value in values)
                             + (f" | {trace_change(values, digits)} |" if len(order) == 2 else " |"))
        lines.append("")
    failed = [record["unit"] for unit, items in attempts.items() for record in items[-1:] if not record.get("valid")]
    if failed:
        lines += [f"Units whose last attempt is not valid: {', '.join(failed)}", ""]
    (out_directory / "trace_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_directory / "trace_summary.json").write_text(json.dumps(json_ready(summary), indent=1), encoding="utf-8")
    print(f"{LOG_PREFIX} trace summary of {sum(len(builds) for builds in by_configuration.values())} build(s) in "
          f"{len(by_configuration)} configuration(s): {out_directory / 'trace_summary.md'}", flush=True)
    return 1 if failed else 0


def trace_change(values: list, digits: int = 1) -> str:
    before, after = values
    if before is None or after is None or not (math.isfinite(before) and math.isfinite(after)):
        return "-"
    return f"{after - before:+,.{digits}f}" + (f" ({100.0 * (after / before - 1.0):+.1f} %)" if before else "")


# ----------------------------------------------------------------------------- self-test (CPU only)

# Real logs and the values read off them by hand (the runner's printed lines; fps = the more precise value).
REAL_LOG_EXPECTATIONS = (
    ("logs/e39/b4/perf/62k_k2_on_r1.log",
     {"version": 7, "slabs": 2, "device_map": [0, 1], "fps": 1817.7, "steady_steps": 8000, "steady_seconds": 4.40,
      "total": (10000, 5.47, 1828.8), "drift": 0, "ok": True, "alive": {"0": 37490, "1": 37039},
      "switches": {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "32", "V7_DEEP_WALL_SKIP": "1",
                   "V7_DEEP_WALL_CHECK": "0"},
      "deep_wall": ["on", "on"], "overflow_counters": 8, "host_copy_kib": {"s0_to_s1": 86.2, "s1_to_s0": 86.1},
      "cuts": [28], "seams_ok": True}),
    ("logs/e39/b4/perf/3d8mb4_k2_auto.log",
     {"version": 7, "slabs": 2, "fps": 400 / 15.56, "drift": 0, "ok": True, "deep_wall": ["off", "off"],
      "deep_wall_reason_contains": "0 deep-wall candidates", "particles": 9119703,
      "partition_particles": [4518018, 4601685], "cuts": [41]}),
    ("logs/e39/b4/perf/3d1m_k1_on_r1.log",          # 2000 steps / 24.45 s is more precise than the printed 81.8
     {"version": 7, "slabs": 1, "device_map": [1], "fps": 2000 / 24.45, "drift": 0, "ok": True,
      "alive": {"0": 1685159}, "cuts": [], "seams_ok": True}),
    ("logs/e39/b9_review/perf/k2_on_t1.log",
     {"version": 7, "slabs": 2, "fps": 820.1, "ok": True, "switches": {"V7_DENSITY_COPY_COMPUTE": "1"}}),
    ("logs/e39/b6_review/perf/1m_off_r1.log",       # lanes 0: a b9 run of 2d_1m_k2
     {"version": 7, "slabs": 2, "fps": 818.8, "total": (6000, 7.30, 822.1), "ok": True,
      "switches": {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "0"},
      "provenance": [("b9", "2d_1m_k2", "0,1", True), ("b6", "2d_1m_k2", "0,1", False),
                     ("b9", "2d_62k_k2", "0,1", False)]}),
    ("logs/e39/b6_review/perf/1m_on_r2.log",        # lanes 32: a b6 run of 2d_1m_k2
     {"version": 7, "slabs": 2, "fps": 851.0, "ok": True,
      "switches": {"V7_DENSITY_COPY_COMPUTE": "1", "V7_GHOST_SEND_LANES": "32"},
      "provenance": [("b6", "2d_1m_k2", "0,1", True), ("b9", "2d_1m_k2", "0,1", False),
                     ("b6", "2d_1m_k1", "0", False)]}),
    ("logs/e39/perf_campaign/runs/t1__2d_1m_k2__b1__a1__dev01.log",   # E39 campaign: b1 = overlap 0
     {"version": 7, "slabs": 2, "ok": True,
      "provenance": [("b1", "2d_1m_k2", "0,1", True), ("b3", "2d_1m_k2", "0,1", False)]}),
    ("logs/e39/perf_campaign/runs/t1__2d_1m_k2__b3__a1__dev01.log",   # E39 campaign: b3 printed overlap auto
     {"version": 7, "slabs": 2, "ok": True,
      "provenance": [("b3", "2d_1m_k2", "0,1", True), ("b1", "2d_1m_k2", "0,1", False)]}),
    ("logs/e39/b6/overflow/chain_two_layer_on.log",
     {"version": 7, "drift": -141, "ok": False, "overflow_total": 24750,
      "sim_overflow": {"0": {"overflow_ghost_count": 12715}, "1": {"overflow_ghost_count": 12035}},
      "violation_contains": ["drift -141", "overflow_total 24750", "VALIDATION FAILED"]}),
    ("logs/e39/b4_review/perf/small9_off_r1.log",
     {"version": 7, "drift": 0, "ok": False, "fields_ok": False, "speed_max": 1.0052, "fps": 428.5,
      "violation_contains": ["fields FAIL", "VALIDATION FAILED"]}),
)
# v6-rc2 chain bench output, verbatim (the Vulkan device listing cut): the main checkout's
# logs/e37/A/bench/n250_adami_r1.out (E37 timing, K = 1, device 1).
EMBEDDED_V6_LOG = """[chain_v6] switchinterval_s=0.0002
[case_loader_v6] loading lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami
  - ../lid_driven_cavity_2d_n250/domain.obj: 63,001 particles (stick_water)
  - ../lid_driven_cavity_2d_n250/wall.obj: 8,767 particles (stick_wall)
  - ../lid_driven_cavity_2d_n250/wall_top.obj: 2,761 particles (stick_lid)
[case_loader_v6] total loaded: 74,529 particles
[case_loader_v6] grid: dim=(56, 56, 1) origin=(-0.5564000105857849, -0.5564000105857849, -0.01)
[partition_v6] chain N=1: columns [0,56) of 56; particles 74,529
[chain_v6] K=1 weights=[1.0] device_map=[1] sync=per-direction depth=2 pool_safety=1.2 switchinterval_s=0.0002
[chain_v6] weights source=given cuts=[]
[VulkanContextV6] available physical devices (3, discrete-first stable order):
[VulkanContextV6] transfer queues: 2 (readback + upload split)
[SimV6] device-local buffers: 24, 15.02 MB (14 CONCURRENT for transfer queue)
[SimV6] defrag scratch buffers: 9, 11.45 MB
[SimV6] host staging buffers: 0 (single-GPU mode, no peer)
[SimV6] compute pipelines: 25 (V6_CASCADE_FORCE=1: force_deep_interior in Phase B) (V6_BAND_VOXEL_DISPATCH=1, lanes=64, bands 2/2/3: boundary kernels over band voxels: 0/0/0 threads vs 85,760)
[SimV6] init complete on NVIDIA GeForce RTX 5090 (own=85760, ghost L=0 T=0, wall_boundary=adami)
[ChainOrchV6] N=1, 0 workers started ()
[SimV6] uploaded initial state (5 payload buffers)
[SimV6] bootstrap done: alive=74529 overflow_inside=0 overflow_incoming=0
[SimV6] step cmd buffers recorded (phase A/B/C + 0 readback + 0 upload on transfer Q)
[SimV6] V6_FAST_SUBMIT=1: cached cffi submit batches (compute A+B+C, 0 readback, 0 upload), raw waits/signals
[ChainOrchV6] all 1 sims bootstrapped + step cmds ready
[chain_v6] TOTAL: 20000 steps in 8.00s = 2501.2 fps
[chain_v6] STEADY (post-warmup 5000): 15000 steps in 6.02s = 2492.8 fps
[chain_v6] sim0 seam: ghost_layers=2 departed peak/frame=0 capacity=0 far_migration=0 overflow_inside_count=0 overflow_incoming_count=0 overflow_ghost_count=0 overflow_install_tail=0 overflow_install_inside=0 overflow_departed_count=0 overflow_initialization_outside=0
[chain_v6] sim0 (dev1): alive=74,529 pool_used=0.0% peak_migration=0 drops=0 stamp_err=0
[chain_v6] final: total=74,529 (expected 74,529) drift=0 stamp_errors gpu=0 host=0 overflow_total=0 far_migration_total=0
[chain_v6] fields: rho[999.6,1000.5] vmax=1.0000 OK
"""
# Canned nvidia-smi output in this rig's format (2026-10-08; names and UUIDs made generic).
CANNED_GPU_ROWS = ("0, GPU-aaaaaaaa-0000-1111-2222-333333333333, 17, 2794, 20.31, 960, 46, Enabled\n"
                   "1, GPU-bbbbbbbb-4444-5555-6666-777777777777, 0, 0, 5.61, 28, 43, Disabled\n")
CANNED_APPLICATIONS_IDLE = ("2488, [Insufficient Permissions], GPU-aaaaaaaa-0000-1111-2222-333333333333, [N/A]\n"
                            "9700, C:\\Windows\\explorer.exe, GPU-aaaaaaaa-0000-1111-2222-333333333333, [N/A]\n")
CANNED_APPLICATION_PYTHON = ("31337, C:\\Program Files\\Python313\\python.exe, "
                             "GPU-aaaaaaaa-0000-1111-2222-333333333333, [N/A]\n")
CANNED_APPLICATION_HEADLESS = "4242, C:\\Tools\\miner.exe, GPU-bbbbbbbb-4444-5555-6666-777777777777, 512 MiB\n"
# B4 / B6 step-trace summaries (logs/e39/b4/perf/trace_summary.txt, logs/e39/b6/perf/run_perf.out), us:
# (trace directory, sim or link, metric, value, tolerance: exact definitions 0.051, B6's unmasked phase A / ghost
# medians and B4's stricter period 0.5)
TRACE_EXPECTATIONS = (
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "0", "phase_a", 56.9, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "0", "phase_b", 5410.3, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "0", "phase_c", 2707.6, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "0", "a_to_b", 4.1, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "1", "phase_b", 5727.3, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "1", "phase_c", 2546.6, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "sim", "0", "period", 8326.9, 0.5),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "link", "s0_to_s1", "readback_start_delay", 44.1, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "link", "s0_to_s1", "t_tr", 5610.2, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "link", "s0_to_s1", "upload_end_to_receiver_c_start", 4644.0, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "link", "s1_to_s0", "t_tr", 1095.3, 0.051),
    ("logs/e39/b4/perf/trace_3d1m_k2_s0", "link", "s1_to_s0", "receiver_phase_b", 5410.3, 0.051),
    ("logs/e39/b6/perf/trace_1m_on", "sim", "0", "ghost_send", 6.2, 0.5),
    ("logs/e39/b6/perf/trace_1m_on", "sim", "1", "ghost_send", 6.2, 0.5),
    ("logs/e39/b6/perf/trace_1m_on", "sim", "0", "phase_a", 28.3, 0.5),
    ("logs/e39/b6/perf/trace_1m_on", "sim", "0", "period", 1142.6, 0.051),
    ("logs/e39/b6/perf/trace_1m_on", "sim", "1", "period", 1145.0, 0.051),
    ("logs/e39/b6/perf/trace_1m_off", "sim", "0", "ghost_send", 43.0, 0.5),
    ("logs/e39/b6/perf/trace_1m_off", "sim", "1", "ghost_send", 51.2, 0.5),
    ("logs/e39/b6/perf/trace_1m_off", "sim", "0", "period", 1190.9, 0.051),
)


class SelfTest:
    def __init__(self):
        self.failures: list = []
        self.passed = 0
        self.skipped: list = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(message)

    def close(self, value, expected, tolerance: float, message: str) -> None:
        ok = value is not None and expected is not None and abs(float(value) - float(expected)) <= tolerance
        self.check(ok, f"{message}: {value!r}, expected {expected!r} +- {tolerance}")


def selftest_parser(test: SelfTest) -> None:
    for relative, expected in REAL_LOG_EXPECTATIONS:
        path = _REPOSITORY_ROOT / relative
        if not path.exists():
            test.skipped.append(f"{relative} (absent in this checkout)")
            continue
        metrics = parse_runner_log(path.read_text(encoding="utf-8", errors="replace"))
        verdict = invariant_verdict(metrics, 0 if expected["ok"] else 1, False)
        test.check(metrics["version"] == expected["version"], f"{relative}: version {metrics['version']}")
        test.check(verdict["ok"] == expected["ok"], f"{relative}: verdict {verdict}")
        if "fps" in expected:
            test.close(metrics["fps"], expected["fps"], 1e-6, f"{relative}: fps")
        for key, value in (("slabs", "slabs"), ("device_map", "device_map"), ("cuts", "cuts"),
                           ("particles", "particles")):
            if key in expected:
                test.check(metrics[value] == expected[key], f"{relative}: {key} {metrics[value]}")
        if "steady_steps" in expected:
            test.check(metrics["steady"]["steps"] == expected["steady_steps"]
                       and metrics["steady"]["seconds"] == expected["steady_seconds"], f"{relative}: steady")
        if "total" in expected:
            total = metrics["total"]
            test.check((total["steps"], total["seconds"], total["fps"]) == expected["total"], f"{relative}: {total}")
        if "drift" in expected:
            test.check(metrics["final"]["drift"] == expected["drift"], f"{relative}: drift")
        if "alive" in expected:
            test.check({sim: values["alive"] for sim, values in metrics["sims"].items()} == expected["alive"],
                       f"{relative}: alive")
        if "switches" in expected:
            test.check(metrics["v7_switches"] == expected["switches"], f"{relative}: switches {metrics['v7_switches']}")
        if "deep_wall" in expected:
            test.check(deep_wall_states(metrics) == expected["deep_wall"], f"{relative}: deep wall "
                                                                          f"{deep_wall_states(metrics)}")
        if "deep_wall_reason_contains" in expected:
            test.check(all(expected["deep_wall_reason_contains"] in item["resolution"]
                           for item in metrics["switch_resolutions"]["V7_DEEP_WALL_SKIP"]), f"{relative}: reason")
        if "overflow_counters" in expected:
            test.check(all(len(values["overflow"]) == expected["overflow_counters"]
                           and not any(values["overflow"].values()) for values in metrics["sims"].values()),
                       f"{relative}: overflow counters")
        if "host_copy_kib" in expected:
            test.check({label: values["host_copy_kib"] for label, values in metrics["links"].items()}
                       == expected["host_copy_kib"], f"{relative}: links {metrics['links']}")
        if "seams_ok" in expected:
            test.check(all(seam["ok"] for seam in metrics["seams"]) == expected["seams_ok"]
                       and metrics["fields"]["ok"], f"{relative}: seams")
        if "partition_particles" in expected:
            test.check(metrics["partition"]["particles"] == expected["partition_particles"], f"{relative}: partition")
        if "overflow_total" in expected:
            test.check(metrics["final"]["overflow_total"] == expected["overflow_total"], f"{relative}: overflow_total")
        if "sim_overflow" in expected:
            for sim, counters in expected["sim_overflow"].items():
                for name, value in counters.items():
                    test.check(metrics["sims"][sim]["overflow"][name] == value, f"{relative}: sim{sim} {name}")
        if "fields_ok" in expected:
            test.check(metrics["fields"]["ok"] == expected["fields_ok"]
                       and metrics["fields"]["speed_max"] == expected["speed_max"], f"{relative}: fields")
        for text in expected.get("violation_contains", []):
            test.check(any(text in violation for violation in verdict["violations"]),
                       f"{relative}: no violation containing {text!r} in {verdict['violations']}")
        for build_name, configuration_name, device_map, accepted in expected.get("provenance", []):
            check = provenance_check(build_name, metrics, configuration_name, device_map)
            test.check(check["ok"] == accepted and (not accepted or check["added_reported"]),
                       f"{relative}: provenance as {build_name} {configuration_name} {device_map}: {check}")
    metrics = parse_runner_log(EMBEDDED_V6_LOG)
    test.check(metrics["version"] == 6 and metrics["slabs"] == 1 and metrics["device_map"] == [1], "v6: header")
    test.close(metrics["fps"], 2492.8, 1e-9, "v6: fps (printed, more precise than 15000 / 6.02)")
    test.check(metrics["v7_switches"] is None and invariant_verdict(metrics, 0, False)["ok"], "v6: verdict")
    test.check(len(metrics["sims"]["0"]["overflow"]) == 7 and metrics["fields"]["speed_max"] == 1.0, "v6: counters")
    test.check(provenance_check("v6", metrics)["ok"] and not provenance_check("b9", metrics)["ok"], "v6: provenance")
    adami = provenance_check("v6", metrics, "2d_adami_250_k1", "1")["problems"]
    test.check(len(adami) == 2 and "20000 steps" in adami[0] and "warmup 5000" in adami[1],
               f"v6: against 2d_adami_250_k1 on device 1 only the E37 steps differ: {adami}")
    wrong = provenance_check("v6", metrics, "2d_62k_k1", "0")["problems"]
    test.check(any("case" in item for item in wrong) and any("device_map" in item for item in wrong)
               and any("steps" in item for item in wrong), f"v6: wrong case / device / steps detected: {wrong}")
    v7_metrics = parse_runner_log((_REPOSITORY_ROOT / REAL_LOG_EXPECTATIONS[0][0]).read_text(encoding="utf-8")) \
        if (_REPOSITORY_ROOT / REAL_LOG_EXPECTATIONS[0][0]).exists() else None
    if v7_metrics is not None:
        # that run forced V7_DEEP_WALL_SKIP=1: b4 (auto) must be refused; a v6 build is the wrong runner
        test.check(not provenance_check("b4", v7_metrics)["ok"] and not provenance_check("v6", v7_metrics)["ok"],
                   "provenance: a forced deep-wall run is neither b4 nor v6")
        test.check(provenance_check("b6", dict(v7_metrics, v7_switches=dict(v7_metrics["v7_switches"],
                                                                            V7_DEEP_WALL_SKIP="0")),
                                    "2d_62k_k2", "0,1")["ok"], "provenance: b6 accepted on 2d_62k_k2")
        changed = json.loads(json.dumps(v7_metrics))
        changed["v7_switches"]["V7_DEEP_WALL_SKIP"] = "0"
        check = provenance_check("b6", changed)
        test.check(check["ok"] and check["added_reported"], f"provenance: b6 accepted ({check})")
        check = provenance_check("b1", changed)
        test.check(not check["ok"], "provenance: b1 refused (deep wall 0 printed, auto set)")
    test.check(precise_fps(400, 15.56, 25.7) == 400 / 15.56 and precise_fps(8000, 4.40, 1817.7) == 1817.7,
               "precise fps rule")
    broken = parse_runner_log("[chain_v7] switchinterval_s=0.0002\nTraceback (most recent call last):\n  File x\n"
                              "RuntimeError: vkCreateDevice failed\n")
    verdict = invariant_verdict(broken, 1, False)
    test.check(not verdict["ok"]
               and any("RuntimeError: vkCreateDevice failed" in item for item in verdict["violations"])
               and any("no final line" in item for item in verdict["violations"]), f"exception verdict {verdict}")
    refused = parse_runner_log("[chain_v7] switchinterval_s=0.0002\nwall_boundary adami (case.yaml numerics) "
                               "supports one GPU (K = 1) only in this release; got K = 2\n")
    test.check(refused["error"] is not None and "K = 1" in refused["error"], f"refusal message {refused['error']}")


def selftest_bulk_logs(test: SelfTest) -> None:
    """Every chain-bench log under logs/e39/*/perf*: the parser's verdict must agree with the runner's own end checks
    (ok exactly when the log has its final line and no VALIDATION FAILED), the fps must be the STEADY line's (printed
    or steps / seconds), and a log with a v7 switch line must name the solver chain_v7."""
    paths = sorted(path for pattern in ("logs/e39/*/perf*/*.log", "logs/e39/*/perf*/*/*.log")
                   for path in _REPOSITORY_ROOT.glob(pattern))
    if not paths:
        test.skipped.append("bulk log check (no logs/e39 perf logs)")
        return
    checked = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "[chain_v" not in text:
            continue
        metrics = parse_runner_log(text)
        runner_passed = "] final:" in text and VALIDATION_FAILED_TEXT not in text
        verdict = invariant_verdict(metrics, 0 if runner_passed else 1, False)
        relative = path.relative_to(_REPOSITORY_ROOT).as_posix()
        test.check(verdict["ok"] == runner_passed, f"{relative}: verdict {verdict} but the runner "
                                                   f"{'passed' if runner_passed else 'failed'}")
        steady = LINE_PATTERNS["steady"].search("\n".join(line for line in text.splitlines() if "STEADY" in line))
        if steady:
            steps, seconds = int(steady.group("steps")), float(steady.group("seconds"))
            test.check(metrics["fps"] in (float(steady.group("fps")), steps / seconds),
                       f"{relative}: fps {metrics['fps']}")
        if metrics["v7_switches"] is not None:
            test.check(metrics["version"] == 7, f"{relative}: switch line in a chain_v{metrics['version']} log")
        checked += 1
    print(f"{LOG_PREFIX} selftest: {checked} chain-bench logs under logs/e39 cross-checked", flush=True)


def selftest_nvidia(test: SelfTest) -> None:
    gpus = parse_gpu_rows(CANNED_GPU_ROWS)
    test.check(len(gpus) == 2 and gpus[0]["display_active"] == "Enabled" and gpus[1]["sm_clock_mhz"] == 28.0,
               f"gpu rows {gpus}")
    idle = idle_verdict(gpus, parse_compute_applications(CANNED_APPLICATIONS_IDLE))
    test.check(not idle["busy"], f"desktop apps + 17 % on the display GPU = idle: {idle['reasons']}")
    busy = idle_verdict(gpus, parse_compute_applications(CANNED_APPLICATIONS_IDLE + CANNED_APPLICATION_PYTHON))
    test.check(busy["busy"] and busy["python_processes"], "a python process on the display GPU = busy")
    busy = idle_verdict(gpus, parse_compute_applications(CANNED_APPLICATION_HEADLESS))
    test.check(busy["busy"] and busy["headless_processes"], "any process on the headless GPU = busy")
    loaded = parse_gpu_rows(CANNED_GPU_ROWS.replace(", 17, ", ", 45, "))
    test.check(idle_verdict(loaded, [])["busy"], "45 % on the display GPU = busy")
    loaded = parse_gpu_rows(CANNED_GPU_ROWS.replace(", 0, 0, 5.61", ", 9, 0, 5.61"))
    test.check(idle_verdict(loaded, [])["busy"], "9 % on the headless GPU = busy")
    no_display = parse_gpu_rows(CANNED_GPU_ROWS.replace("Enabled", "Disabled"))
    test.check(not idle_verdict(no_display, [])["busy"], "no display reported: GPU 0 keeps the desktop allowance")
    applications = parse_compute_applications("17, C:\\A, B\\tool.exe, GPU-x, 10 MiB\n")
    test.check(applications[0]["name"] == "C:\\A, B\\tool.exe" and applications[0]["gpu_uuid"] == "GPU-x",
               f"application name with ', ': {applications}")
    base = datetime.datetime(2026, 10, 8, 22, 0, 0)
    lines = []
    for second in range(10):
        stamp = (base + datetime.timedelta(seconds=second)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        lines.append(f"{stamp}, 0, {2700 + second}, 14001, {400 + second}.5, {50 + second}, 99, 3000")
        lines.append(f"{stamp}, 1, {2800 + second}, 14001, 450.0, 60, 100, 2000")
    samples = [parse_telemetry_line(line, 0.0) for line in lines]
    test.check(all(sample is not None for sample in samples), "telemetry lines parse")
    start = base.timestamp() + 2.5
    summary = telemetry_summary(samples, start, start + 4.0)
    test.check(summary["0"]["samples"] == 4 and summary["0"]["sm_clock_mhz"]["median"] == 2704.5
               and summary["0"]["sm_clock_mhz"]["min"] == 2703 and summary["1"]["sm_clock_mhz"]["max"] == 2806
               and summary["0"]["temperature_celsius"]["max"] == 56, f"telemetry window {summary}")
    test.check(parse_telemetry_line("2026/10/08 22:00:00.000, 0, [N/A], 1, 2, 3, 4, 5", 7.0)["sm_clock_mhz"] is None,
               "telemetry [N/A]")
    vulkan = [{"index": 0, "uuid": "bbbbbbbb444455556666777777777777", "discrete": True},
              {"index": 1, "uuid": "aaaaaaaa000011112222333333333333", "discrete": True},
              {"index": 2, "uuid": "cccccccc000000000000000000000000", "discrete": False}]
    mapping = match_devices(vulkan, gpus)
    test.check(mapping == {"0": "1", "1": "0"}, f"device mapping by UUID (crossed on purpose): {mapping}")


def selftest_plan(test: SelfTest) -> None:
    """The plan rules on a synthetic deep-wall resolution: every 3-D configuration's auto rule on except
    3d_8m_walls4 (as the probe finds), 2-D off."""
    simulator_source = (_REPOSITORY_ROOT / "experiment" / "v7" / "utils" / "simulator_v7.py").read_text(
        encoding="utf-8", errors="replace")
    default = re.search(r'_parse_band_overlap\(os\.environ\.get\("V7_BAND_OVERLAP", "(\w+)"\)\)', simulator_source)
    test.check(default is not None and default.group(1) == BAND_OVERLAP_CODE_DEFAULT,
               f"BAND_OVERLAP_CODE_DEFAULT {BAND_OVERLAP_CODE_DEFAULT!r} is simulator_v7's default "
               f"{default.group(1) if default else None!r}")
    b3_expected = {**BUILDS["b1"]["switches"], BAND_OVERLAP_SWITCH: BAND_OVERLAP_VALUE_IN_B3}
    test.check(BAND_OVERLAP_VALUE_IN_B3 in ("1", "auto") and BUILDS["b3"]["switches"] == b3_expected,
               "b3 = b1 + V7_BAND_OVERLAP=1 or auto")
    deep_wall = {}
    for name in CONFIGURATIONS:
        slabs = slab_count(name)
        active = case_dimension(CONFIGURATIONS[name]["case"]) == 3 and not name.startswith("3d_8m_walls4")
        deep_wall[name] = {"active": [active] * slabs, "reasons": ["synthetic"] * slabs,
                           "candidate_percent": [16.6 if active else 0.0] * slabs, "source": "selftest"}
    builds = list(CHAIN)
    every_build = {name: True for name in CHAIN}
    plan = build_plan(list(CONFIGURATIONS), builds, 3, deep_wall, [], {"v6": "x", "v7": "y"}, availability=every_build)
    test.check(plan["selected"] == list(CHAIN), "all builds selected")
    configurations = plan["configurations"]
    for name, entry in configurations.items():
        slabs = slab_count(name)
        dimension = case_dimension(CONFIGURATIONS[name]["case"])
        test.check(entry["run"][0] == "v6", f"{name}: v6 runs")
        if "b6" in plan["selected"]:
            test.check(("b6" in entry["skipped"]) == (slabs == 1), f"{name}: b6 skip {entry}")
        if "b4" in plan["selected"]:
            test.check(("b4" in entry["skipped"]) == (not any(deep_wall[name]["active"])), f"{name}: b4 skip")
        if "b3" in plan["selected"]:
            test.check(("b3" in entry["skipped"]) == (slabs == 1 or dimension == 3), f"{name}: b3 skip")
        for skipped, item in entry["skipped"].items():
            test.check(item["inherits"] in entry["run"], f"{name}: {skipped} inherits a run build")
    k1 = configurations["2d_1m_k1"]["skipped"]
    if "b4" in k1 and "b6" in k1:
        test.check(k1["b4"]["inherits"] == "b9", f"2-D K = 1: b4 -> b6 -> b9 {k1}")
    for trial in (1, 2, 3):
        for name in CONFIGURATIONS:
            units = [unit for unit in plan["units"] if unit["trial"] == trial and unit["configuration"] == name]
            run_builds = configurations[name]["run"]
            test.check([unit["build"] for unit in units] == run_builds[(trial - 1) % len(run_builds):]
                       + run_builds[:(trial - 1) % len(run_builds)], f"{name} t{trial}: rotation")
    first = {}
    for unit in plan["units"]:
        first.setdefault((unit["trial"], unit["configuration"]), unit["build"])
    for name in CONFIGURATIONS:
        firsts = {first[(trial, name)] for trial in (1, 2, 3)}
        test.check(len(firsts) == min(3, len(configurations[name]["run"])), f"{name}: first builds {firsts}")
    order = [(unit["trial"], list(CONFIGURATIONS).index(unit["configuration"])) for unit in plan["units"]]
    test.check(order == sorted(order), "trial-major, configurations in table order")
    test.check(all(unit["device_maps"] == ["0", "1"] for unit in plan["units"]
                   if slab_count(unit["configuration"]) == 1)
               and all(unit["device_maps"] == ["0,1,0,1"] for unit in plan["units"]
                       if unit["configuration"] == "3d_8m_walls4_k4_shared"), "device maps")
    partial = build_plan(["2d_1m_k1"], ["v6", "b4"], 1, deep_wall, [], {"v6": "x", "v7": "y"},
                         availability=every_build)
    test.check(partial["configurations"]["2d_1m_k1"]["run"] == ["v6", "b4"],
               "b4 without its predecessor in the campaign is run, not skipped")
    missing_b3 = build_plan(["2d_1m_k2"], list(CHAIN), 1, deep_wall, [], {"v6": "x", "v7": "y"},
                            availability=dict(every_build, b3=False))
    test.check(missing_b3["unavailable"] == ["b3"] and "b3" not in missing_b3["configurations"]["2d_1m_k2"]["run"],
               "a build whose switch is not in the code is left out")
    expected_units = {"2d_62k_k1": 3, "2d_62k_k2": 5, "2d_1m_k1": 3, "2d_1m_k2": 5, "2d_16m_k1": 3, "2d_16m_k2": 5,
                      "3d_8m_walls4_k1": 3, "3d_8m_walls4_k2": 4, "3d_8m_walls4_k4_shared": 4, "3d_1m_walls9_k1": 4,
                      "3d_1m_walls9_k2": 5, "2d_adami_250_k1": 3, "2d_adami_1000_k1": 3}
    test.check({name: len(entry["run"]) for name, entry in configurations.items()} == expected_units,
               f"builds run per configuration: {[(name, entry['run']) for name, entry in configurations.items()]}")
    # B3 auto: a 2-D K >= 2 configuration whose chain verdict is off on every slab skips b3 (inherits b1); on: runs
    band_overlap = {name: {"active": [name == "2d_1m_k2"] * slab_count(name), "reasons": ["synthetic"] * slab_count(name),
                           "source": "selftest"} for name in CONFIGURATIONS}
    with_rule = build_plan(list(CONFIGURATIONS), builds, 1, deep_wall, [], {"v6": "x", "v7": "y"},
                           availability=every_build, band_overlap=band_overlap)
    for name, entry in with_rule["configurations"].items():
        test.check(("b3" in entry["skipped"]) == (name != "2d_1m_k2"), f"{name}: b3 skip under the auto rule")
    for name in ("2d_62k_k2", "2d_16m_k2"):
        skip = with_rule["configurations"][name]["skipped"].get("b3") or {}
        test.check(skip.get("inherits") == "b1" and "V7_BAND_OVERLAP=auto resolves off" in skip.get("reason", ""),
                   f"{name}: b3 = b1 by the auto verdict {skip}")
    # a ledger record makes its unit done / stale / failed
    record = {"mode": "campaign", "unit": unit_identifier(1, "2d_1m_k2", "b9"), "valid": True,
              "code": {"v6": "x", "v7": "y"}, "configuration": "2d_1m_k2", "build": "b9",
              "processes": [{"wall_s": 42.0}]}
    statuses = {}
    for variant, digests in (("done", {"v6": "x", "v7": "y"}), ("stale", {"v6": "x", "v7": "z"})):
        resumed = build_plan(["2d_1m_k2"], ["v6", "b9"], 1, deep_wall, [record], digests, availability=every_build)
        unit = next(unit for unit in resumed["units"] if unit["build"] == "b9")
        statuses[variant] = (unit["status"], unit["estimate_seconds"], unit["estimate_source"])
    failed = build_plan(["2d_1m_k2"], ["v6", "b9"], 1, deep_wall, [dict(record, valid=False)], {"v6": "x", "v7": "y"},
                        availability=every_build)
    statuses["failed"] = next(unit["status"] for unit in failed["units"] if unit["build"] == "b9")
    test.check(statuses == {"done": ("done", 42.0, "ledger"), "stale": ("stale", 42.0, "ledger"), "failed": "failed"},
               f"unit status from the ledger: {statuses}")


def synthetic_process(device_map: str, fps: float, gpus: list, clocks: dict, violations: Optional[list] = None,
                      deep_wall: Optional[list] = None, notes: Optional[list] = None, solver: str = "v7",
                      fused: Optional[list] = None) -> dict:
    resolutions = {}
    if deep_wall:
        resolutions["V7_DEEP_WALL_SKIP"] = [{"value": "auto", "resolution": state + " (synthetic)"}
                                            for state in deep_wall]
    if fused:
        resolutions["V7_FUSED_CORRECTION_DENSITY"] = [{"value": "1", "resolution": state + " (synthetic)"}
                                                      for state in fused]
    metrics = {"version": int(solver[1]), "fps": fps, "final": {"drift": 0 if not violations else -3, "total": 100,
                                                                 "expected": 100 if not violations else 103,
                                                                 "stamp_errors_gpu": 0, "stamp_errors_host": 0,
                                                                 "overflow_total": 0, "far_migration_total": 0},
               "sims": {"0": {"overflow": {"overflow_ghost_count": 0}, "far_migration": 0}}, "seams": [],
               "fields": {"ok": True}, "validation_failed": False, "pipeline_notes": notes or [],
               "switch_resolutions": resolutions, "steady": {"seconds": 5.0, "steps": 1000}}
    return {"device_map": device_map, "gpus": gpus, "exit_code": 0 if not violations else 1, "timed_out": False,
            "wall_s": 10.0, "metrics": metrics,
            "invariants": {"ok": not violations, "violations": violations or []},
            "switch_check": {"ok": True, "problems": [], "added_reported": True},
            "telemetry": {gpu: {"samples": 5, "sm_clock_mhz": {"median": clocks[gpu]},
                                "power_watts": {"median": 500.0}, "temperature_celsius": {"max": 60.0}}
                          for gpu in clocks}}


def selftest_summary(test: SelfTest, out_directory: pathlib.Path) -> None:
    """A synthetic ledger with known fps: 2d_1m_k1 / k2 and 3d_8m_walls4_k1 / k2 / k4_shared, builds v6, b9, b6, b4
    (skipped where inert), three trials, one failed attempt superseded by a valid one, one violation that stays."""
    directory = out_directory / "synthetic_summary"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    deep_wall = {name: {"active": [False] * slab_count(name), "reasons": ["synthetic"] * slab_count(name),
                        "candidate_percent": [0.0] * slab_count(name), "source": "selftest"}
                 for name in CONFIGURATIONS}
    names = ["2d_1m_k1", "2d_1m_k2", "3d_8m_walls4_k1", "3d_8m_walls4_k2", "3d_8m_walls4_k4_shared"]
    builds = ["v6", "b9", "b6", "b4", "b1"]
    plan = build_plan(names, builds, 3, deep_wall, [], {"v6": "digest6", "v7": "digest7"},
                      availability={name: True for name in CHAIN})
    # fps model: K = 1 GPU 0 = base, GPU 1 = 1.02 base; K = 2 = 1.9 base; K = 4 = 1.7 base; base grows 1 % per
    # trial; cumulative build factors v6 1.0, b9 1.015, b6 1.04 (K >= 2; inert at K = 1), b4 (inert everywhere
    # here), b1 x 1.25 on top
    factors = {"v6": 1.0, "b9": 1.015, "b6": 1.04, "b4": 1.04, "b1": 1.04 * 1.25}
    k1_factors = {"v6": 1.0, "b9": 1.015, "b6": 1.015, "b4": 1.015, "b1": 1.015 * 1.25}
    base = {"2d_1m": 500.0, "3d_8m_walls4": 13.0}
    multiplier = {1: None, 2: 1.9, 4: 1.7}
    ledger = directory / "ledger.jsonl"
    k1_notes = ["(V7_DENSITY_COPY_COMPUTE=1: density copy pass over 1 region(s), 1,203,584 slots)"]
    for unit in plan["units"]:
        configuration = CONFIGURATIONS[unit["configuration"]]
        slabs = slab_count(unit["configuration"])
        value = base[configuration["case"]] * (1 + 0.01 * (unit["trial"] - 1))
        factor = factors[unit["build"]] if slabs > 1 else k1_factors[unit["build"]]
        record = unit_record_base(unit, "campaign", run_environment(unit["build"]),
                                  {"mapping": {"0": "0", "1": "1"}, "source": "selftest"},
                                  {"v6": "digest6", "v7": "digest7"}, plan["configurations"][unit["configuration"]],
                                  plan["unavailable"])
        record.update({"attempt": 1, "start_unix": 0.0, "end_unix": 1.0})
        solver = BUILDS[unit["build"]]["solver"]
        if slabs == 1:
            processes = [synthetic_process("0", value * factor, ["0"], {"0": 2800.0, "1": 2850.0}, solver=solver,
                                           deep_wall=["off"] if solver == "v7" else None, notes=k1_notes),
                         synthetic_process("1", value * 1.02 * factor, ["1"], {"0": 2800.0, "1": 2850.0},
                                           solver=solver, deep_wall=["off"] if solver == "v7" else None,
                                           notes=k1_notes)]
        else:
            # b1 falls back on one slab of the K = 4 shared runs: the summary must note it (partly inert)
            fused = (["fused"] * slabs if unit["configuration"] != "3d_8m_walls4_k4_shared"
                     else ["fused", "separate kernels", "fused", "fused"]) if unit["build"] == "b1" else None
            processes = [synthetic_process(configuration["device_map"], value * multiplier[slabs] * factor, ["0", "1"],
                                           {"0": 2790.0, "1": 2840.0}, solver=solver,
                                           deep_wall=["off"] * slabs if solver == "v7" else None,
                                           notes=["(V7_GHOST_SEND_LANES=32: ghost_send over 2 lane groups)"],
                                           fused=fused)]
        if unit["unit"] == unit_identifier(2, "2d_1m_k2", "b6"):
            # a failed first attempt (its fps must not count), then the valid one
            failed = dict(record, attempt=1, valid=False,
                          processes=[synthetic_process("0,1", 1.0, ["0", "1"], {"0": 1.0, "1": 1.0},
                                                       violations=["drift -3"])])
            append_ledger(ledger, failed)
            record["attempt"] = 2
        record.update({"processes": processes, "valid": True})
        append_ledger(ledger, record)
    code, summary = summarize_campaign(directory)
    test.check(code == 1 and len(summary["violations"]) == 1 and "drift -3" in summary["violations"][0],
               f"one loud violation (the superseded attempt): {summary['violations']}")
    test.check("INVARIANT / PROVENANCE VIOLATIONS: 1" in (directory / "summary.md").read_text(encoding="utf-8"),
               "violation heading in summary.md")
    k2 = summary["configurations"]["2d_1m_k2"]
    expected_v6 = statistics.fmean(500.0 * 1.9 * (1 + 0.01 * trial) for trial in range(3))
    test.close(k2["builds"]["v6"]["fps"]["mean"], expected_v6, 1e-9, "2d_1m_k2 v6 mean")
    test.close(k2["builds"]["b6"]["fps"]["per_trial"][2], 500.0 * 1.9 * 1.01 * 1.04, 1e-9,
               "the superseded failed attempt does not count")
    chain = {row["build"]: row for row in k2["chain"]}
    test.close(chain["b9"]["against_previous"]["mean"], 1.015, 1e-12, "b9 vs v6 paired ratio")
    test.close(chain["b9"]["against_previous"]["standard_deviation"], 0.0, 1e-12, "paired ratio std 0")
    test.close(chain["b6"]["against_previous"]["mean"], 1.04 / 1.015, 1e-12, "b6 vs b9")
    test.close(chain["b6"]["against_reference"]["mean"], 1.04, 1e-12, "b6 vs v6")
    test.check(chain["b4"]["inherited"] and abs(chain["b4"]["against_previous"]["mean"] - 1.0) < 1e-12,
               "b4 inherited from b6 at 2-D: ratio 1")
    test.close(chain["b1"]["against_previous"]["mean"], 1.25, 1e-12, "b1 vs b4 (= b6) paired ratio")
    test.close(chain["b1"]["against_reference"]["mean"], 1.04 * 1.25, 1e-12, "b1 vs v6")
    k1 = summary["configurations"]["2d_1m_k1"]
    test.check(k1["builds"]["b6"]["status"] == "inherited" and k1["builds"]["b6"]["skip"]["inherits"] == "b9"
               and k1["builds"]["b4"]["skip"]["inherits"] == "b9", "K = 1: b6 and b4 inherit b9")
    k1_chain = {row["build"]: row for row in k1["chain"]}
    test.close(k1_chain["b1"]["against_previous"]["mean"], 1.25, 1e-12, "K = 1: b1 vs b4 (= b9)")
    test.close(summary["configurations"]["2d_1m_k2"]["efficiency"]["b1"]["efficiency"]["mean"],
               1.9 * 1.04 * 1.25 / (2 * 1.01 * 1.015 * 1.25), 1e-12, "eta b1 against b1's own K = 1 runs")
    test.close(k1["builds"]["v6"]["devices"]["1"]["mean"] / k1["builds"]["v6"]["devices"]["0"]["mean"], 1.02, 1e-12,
               "K = 1 per device")
    efficiency = k2["efficiency"]["b6"]
    expected = 1.9 * 1.04 / (2 * 1.01 * 1.015)
    test.close(efficiency["efficiency"]["mean"], expected, 1e-12, "eta b6 (K = 1 reference inherited from b9)")
    test.close(efficiency["efficiency_minimum"]["mean"], 1.9 * 1.04 / (2 * 1.015), 1e-12, "eta_min")
    test.close(efficiency["reference_spread"]["mean"], 0.02 / 1.01, 1e-12, "reference spread")
    test.check(efficiency["efficiency"]["trials"] == 3, "eta over 3 trials")
    shared = summary["configurations"]["3d_8m_walls4_k4_shared"]["efficiency"]["v6"]
    test.close(shared["efficiency"]["mean"], 1.7 / (2 * 1.01), 1e-12, "K = 4 shared: G = 2 distinct GPUs")
    evidence = summary["configurations"]["3d_8m_walls4_k2"]["skip_evidence"]
    test.check(evidence.get("b4", "").startswith("confirmed"), f"b4 skip evidence: {evidence}")
    evidence = summary["configurations"]["2d_1m_k1"]["skip_evidence"]
    test.check(evidence.get("b6", "").startswith("confirmed") and evidence.get("b4", "").startswith("confirmed"),
               f"b6 / b4 skip evidence at K = 1: {evidence}")
    markdown = (directory / "summary.md").read_text(encoding="utf-8")
    test.check("K = 4 SHARED" in markdown and "| 2d_1m_k2 | b4 | = b6" in markdown, "markdown labels")
    test.check(len(summary["notes"]) == 1 and "3d_8m_walls4_k4_shared b1" in summary["notes"][0]
               and "partly inert" in summary["notes"][0], f"fallback note: {summary['notes']}")
    test.check(summary["configurations"]["2d_1m_k2"]["resolutions"]["b1"]
               == {"V7_DEEP_WALL_SKIP": ["off"], "V7_FUSED_CORRECTION_DENSITY": ["fused"]},
               f"printed resolutions: {summary['configurations']['2d_1m_k2']['resolutions']}")
    # a build measured where the plan had skipped it (a later run with another build selection): measured wins
    forced = out_directory / "synthetic_summary_forced"
    if forced.exists():
        shutil.rmtree(forced)
    forced.mkdir(parents=True)
    shutil.copy(ledger, forced / "ledger.jsonl")
    extra = json.loads(json.dumps(next(record for record in read_ledger(ledger)
                                       if record["unit"] == unit_identifier(1, "2d_1m_k1", "b9"))))
    extra.update({"unit": unit_identifier(1, "2d_1m_k1", "b4"), "build": "b4",
                  "skipped_builds": {key: value for key, value in extra["skipped_builds"].items() if key != "b4"}})
    append_ledger(forced / "ledger.jsonl", extra)
    _, forced_summary = summarize_campaign(forced)
    item = forced_summary["configurations"]["2d_1m_k1"]["builds"]["b4"]
    missing_units = forced_summary["configurations"]["2d_1m_k1"]["missing_units"]
    test.check(item["status"] == "measured" and list(item["fps"]["per_trial"]) == [1]
               and unit_identifier(2, "2d_1m_k1", "b4") in missing_units, f"a measured build is not inherited: {item}")


def replay_text(source: pathlib.Path, build_name: str, device_map: str) -> str:
    """A real v7 chain-bench log as the run of another build / device would print it: the runner prefix of the
    build's solver, its 'v7 switches' line (what configured_v7_switches prints for the build's environment) or none
    for v6, the device map in the header and the sim lines."""
    text = source.read_text(encoding="utf-8", errors="replace")
    solver = BUILDS[build_name]["solver"]
    devices = [int(part) for part in device_map.split(",")]
    lines = []
    for line in text.splitlines():
        if line.startswith("[chain_v7] v7 switches:"):
            if solver == "v6":
                continue
            printed = {name: value for name, value in BUILDS[build_name]["switches"].items()
                       if name != BAND_OVERLAP_SWITCH}
            printed.setdefault("V7_DEEP_WALL_CHECK", "0")
            line = "[chain_v7] v7 switches: " + " ".join(f"{name}={value}" for name, value in printed.items())
        line = re.sub(r"device_map=\[[^\]]*\]", f"device_map={devices}", line)
        match = re.match(r"^(\[chain_v\d\] sim(\d+)) \(dev\d+\)", line)
        if match:
            line = f"{match.group(1)} (dev{devices[int(match.group(2))]})" + line[match.end():]
        if solver == "v6":
            for old, new in (("chain_v7", "chain_v6"), ("ChainOrchV7", "ChainOrchV6"), ("SimV7", "SimV6"),
                             ("case_loader_v7", "case_loader_v6"), ("partition_v7", "partition_v6")):
                line = line.replace(old, new)
        lines.append(line)
    return "\n".join(lines) + "\n"


def fake_runner_command(sources: dict, replay_directory: pathlib.Path, hang_units: set,
                        trace_sources: Optional[dict] = None) -> Callable:
    """A command factory replaying real logs (replay_text): the fake process prints them line by line, pausing after
    the bootstrap and TOTAL lines (the markers); a (configuration, build) in hang_units prints one line and sleeps
    (its timeout kills it). Trace mode: the solver's real step trace (trace_sources) is copied to --step-trace."""
    counter = [0]

    def factory(build_name, configuration_name, device_map, trace_directory=None, trace_detail="phases"):
        if (configuration_name, build_name) in hang_units:
            return [sys.executable, "-c", "import time; print('[chain_v7] switchinterval_s=0.0002', flush=True); "
                                          "time.sleep(600)"]
        if trace_directory is not None:
            shutil.copytree(trace_sources[BUILDS[build_name]["solver"]], trace_directory)
        counter[0] += 1
        replay = replay_directory / f"replay_{counter[0]:03d}.log"
        replay.write_text(replay_text(sources[slab_count(configuration_name)], build_name, device_map),
                          encoding="utf-8")
        script = ("import time\n"
                  f"for line in open({str(replay)!r}, encoding='utf-8').read().splitlines():\n"
                  "    print(line, flush=True)\n"
                  "    if 'bootstrapped' in line or 'TOTAL' in line: time.sleep(0.3)\n")
        return [sys.executable, "-c", script]
    return factory


class FakeGpuBackend:
    """Canned idle state (busy for the first `busy_checks` queries) and synthetic telemetry samples."""

    def __init__(self, busy_checks: int = 1):
        self.busy_checks = busy_checks
        self.queries = 0

    def query(self) -> dict:
        self.queries += 1
        applications = CANNED_APPLICATIONS_IDLE + (CANNED_APPLICATION_PYTHON if self.queries <= self.busy_checks
                                                   else "")
        return idle_verdict(parse_gpu_rows(CANNED_GPU_ROWS), parse_compute_applications(applications))

    def start_telemetry(self, csv_path: pathlib.Path):
        return FakeSampler(csv_path)


class FakeSampler:
    def __init__(self, csv_path: pathlib.Path):
        self.csv_path = csv_path
        self.started = time.time()
        self.error = None

    def stop(self) -> list:
        samples = []
        moment = self.started
        while moment <= time.time() + 0.01:
            for gpu, clock in (("0", 2790.0), ("1", 2845.0)):
                samples.append({"time": moment, "gpu": gpu, "sm_clock_mhz": clock, "memory_clock_mhz": 14001.0,
                                "power_watts": 400.0, "temperature_celsius": 55.0, "utilization_percent": 99.0,
                                "memory_used_mib": 1000.0})
            moment += 0.1
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.write_text("fake\n", encoding="utf-8")
        return samples


def selftest_fake_campaign(test: SelfTest, out_directory: pathlib.Path) -> None:
    """run_campaign end to end on CPU: fake GPU backend (busy at the first check), fake runners replaying real 2-D 1M
    chain-bench logs as v6 / b9 / b6 runs (replay_text), v6 at K = 2 hung and killed by its timeout, then --resume
    with the hang fixed (only the two failed units run again), then summarize."""
    sources = {1: _REPOSITORY_ROOT / "logs/e39/b4/perf/1m_k1_off_r1.log",
               2: _REPOSITORY_ROOT / "logs/e39/b6/perf/1m_on_r1.log"}
    if not all(path.exists() for path in sources.values()):
        test.skipped.append("fake campaign (logs/e39 1M logs absent)")
        return
    directory = out_directory / "fake_campaign"
    if directory.exists():
        shutil.rmtree(directory)
    replay_directory = directory / "replay"
    replay_directory.mkdir(parents=True)
    deep_wall = {name: {"active": [False] * slab_count(name), "reasons": ["synthetic"] * slab_count(name),
                        "candidate_percent": [0.0] * slab_count(name), "source": "selftest"} for name in CONFIGURATIONS}
    mapping = {"mapping": {"0": "0", "1": "1"}, "source": "selftest"}
    names, builds = ["2d_1m_k1", "2d_1m_k2"], ["v6", "b9", "b6"]
    options = {"device_mapping": mapping, "settle_seconds": 0.0, "deep_wall": deep_wall, "retry_seconds": (0.2,),
               "availability": {name: True for name in CHAIN}, "code_digests": {"v6": "digest6", "v7": "digest7"}}
    started = time.time()
    code = run_campaign(directory, names, builds, 2, resume=False, dry_run=False, backend=FakeGpuBackend(busy_checks=1),
                        command_factory=fake_runner_command(sources, replay_directory, {("2d_1m_k2", "v6")}),
                        timeout_override=6.0, **options)
    records = read_ledger(directory / "ledger.jsonl")
    test.check(code == 1, f"first pass: exit 1 (the hung units), got {code}")
    hung = [record for record in records if record["build"] == "v6" and record["configuration"] == "2d_1m_k2"]
    test.check(len(hung) == 2 and all(record["processes"][0]["timed_out"] and not record["valid"] for record in hung),
               "the hung units were killed and recorded")
    good = [record for record in records if record["valid"]]
    unexpected = {record["unit"]: [process["invariants"]["violations"] + process["switch_check"]["problems"]
                                   for process in record["processes"]]
                  for record in records if not record["valid"] and record not in hung}
    # b6 is inert at K = 1: 2d_1m_k1 x {v6, b9} + 2d_1m_k2 x {v6, b9, b6}, two trials = 10 units, 2 of them hung
    test.check(len(records) == 10 and len(good) == 8, f"records {len(records)}, valid {len(good)}: {unexpected}")
    k1 = next(record for record in good if record["configuration"] == "2d_1m_k1")
    test.check(len(k1["processes"]) == 2 and {process["device_map"] for process in k1["processes"]} == {"0", "1"},
               "K = 1 unit = two processes")
    test.check(k1["skipped_builds"].get("b6", {}).get("inherits") == "b9", f"skip map {k1['skipped_builds']}")
    test.check(all(process["steady_window"]["source"] == "steady"
                   for record in good for process in record["processes"]), "steady windows from the markers")
    test.check(all(process["telemetry"].get(process["gpus"][0], {}).get("samples", 0) >= 1
                   for record in good for process in record["processes"]), "telemetry inside the steady windows")
    test.check(records[0]["idle_check"]["checks"] == 2 and all(record["idle_check"]["checks"] == 1
                                                               for record in records[1:]),
               "the first unit waited out the busy check")
    test.check(all(record["environment_difference"]["set"].get("V7_GHOST_SEND_LANES") == "32"
                   for record in records if record["build"] == "b6")
               and all(record["environment_difference"]["set"].get("VK_LOADER_LAYERS_DISABLE")
                       == "VK_LAYER_KHRONOS_validation" for record in records), "environment difference")
    order = [(record["trial"], record["configuration"], record["build"]) for record in records]
    test.check(order[:5] == [(1, "2d_1m_k1", "v6"), (1, "2d_1m_k1", "b9"), (1, "2d_1m_k2", "v6"),
                             (1, "2d_1m_k2", "b9"), (1, "2d_1m_k2", "b6")]
               and order[5:7] == [(2, "2d_1m_k1", "b9"), (2, "2d_1m_k1", "v6")], f"execution order {order}")
    test.check(not _LIVE_PROCESSES, "no live process left")
    code = run_campaign(directory, names, builds, 2, resume=True, dry_run=False, backend=FakeGpuBackend(busy_checks=0),
                        command_factory=fake_runner_command(sources, replay_directory, set()), timeout_override=30.0,
                        **options)
    records = read_ledger(directory / "ledger.jsonl")
    test.check(code == 0 and len(records) == 12 and all(record["valid"] for record in records[10:]),
               f"resume ran only the 2 failed units: exit {code}, {len(records)} records")
    code, summary = summarize_campaign(directory)
    test.check(code == 1 and len(summary["violations"]) == 2
               and all("timed out" in item for item in summary["violations"]),
               f"the two timeouts stay listed: {summary['violations']}")
    entry = summary["configurations"]["2d_1m_k2"]
    test.check(entry["efficiency"].get("v6", {}).get("efficiency", {}).get("trials") == 2
               and entry["efficiency"].get("b6", {}).get("efficiency", {}).get("trials") == 2, "eta from the fake runs")
    test.check(summary["configurations"]["2d_1m_k1"]["builds"]["b6"]["status"] == "inherited", "b6 inherited at K = 1")
    test.check(time.time() - started < 180, f"fake campaign duration {time.time() - started:.0f} s")
    refused = run_campaign(directory, names, builds, 2, resume=False, dry_run=False,
                           backend=FakeGpuBackend(busy_checks=0),
                           command_factory=fake_runner_command(sources, replay_directory, set()), **options)
    test.check(refused == 2, "an existing ledger without --resume is refused")


def selftest_fake_trace(test: SelfTest, out_directory: pathlib.Path) -> None:
    """The trace mode end to end on CPU: default builds (v6 and the final v7 build: b3 with every build available),
    fake runners that print a real 2-D 1M K = 2 log and copy a real B6 step trace (v6 <- lanes 0, v7 <- lanes 32),
    then the trace summary: ghost_send must drop from ~43 / 51 us to ~6 us as in the B6 measurement."""
    sources = {2: _REPOSITORY_ROOT / "logs/e39/b6/perf/1m_on_r1.log"}
    traces = {"v6": _REPOSITORY_ROOT / "logs/e39/b6/perf/trace_1m_off",
              "v7": _REPOSITORY_ROOT / "logs/e39/b6/perf/trace_1m_on"}
    if not sources[2].exists() or not all((path / "run_meta.json").exists() for path in traces.values()):
        test.skipped.append("fake trace (logs/e39/b6/perf traces absent)")
        return
    directory = out_directory / "fake_trace"
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "replay").mkdir(parents=True)
    deep_wall = {name: {"active": [False] * slab_count(name), "reasons": ["synthetic"] * slab_count(name),
                        "candidate_percent": [0.0] * slab_count(name), "source": "selftest"} for name in CONFIGURATIONS}
    code = run_campaign(directory, ["2d_1m_k2"], list(CHAIN), 1, resume=False, dry_run=False, mode="trace",
                        backend=FakeGpuBackend(busy_checks=0),
                        command_factory=fake_runner_command(sources, directory / "replay", set(), traces),
                        device_mapping={"mapping": {"0": "0", "1": "1"}, "source": "selftest"}, settle_seconds=0.0,
                        deep_wall=deep_wall, timeout_override=30.0, availability={name: True for name in CHAIN},
                        code_digests={"v6": "digest6", "v7": "digest7"})
    records = read_ledger(directory / "ledger.jsonl")
    test.check(code == 0 and [record["build"] for record in records] == ["v6", "b3"]
               and all(record["mode"] == "trace" and record["processes"][0]["trace_written"] for record in records),
               f"trace units v6 and the final build: exit {code}, {[(r['build'], r['valid']) for r in records]}")
    summary = json.loads((directory / "trace_summary.json").read_text(encoding="utf-8"))["configurations"]["2d_1m_k2"]
    test.close(summary["v6"]["sims"]["0"]["ghost_send"], 43.0, 0.5, "trace summary: v6 (lanes 0) ghost_send s0")
    test.close(summary["v6"]["sims"]["1"]["ghost_send"], 51.2, 0.5, "trace summary: v6 (lanes 0) ghost_send s1")
    test.close(summary["b3"]["sims"]["0"]["ghost_send"], 6.2, 0.5, "trace summary: final ghost_send s0")
    test.close(summary["b3"]["sims"]["0"]["period"], 1142.6, 0.051, "trace summary: final period s0")
    test.check(all(math.isfinite(summary[build]["links"][link]["t_chain_e29"]) for build in ("v6", "b3")
                   for link in ("s0_to_s1", "s1_to_s0")), "trace summary: E29 t_chain of both links")
    markdown = (directory / "trace_summary.md").read_text(encoding="utf-8")
    test.check("| s0 ghost_send (a_voxel_end -> ghost end) |" in markdown and "change |" in markdown,
               "trace summary markdown")


def selftest_process_tree(test: SelfTest, out_directory: pathlib.Path) -> None:
    """The venv's python.exe starts the interpreter as a child: kill_process_tree must end both."""
    log_path = out_directory / "process_tree.log"
    runner = RunnerProcess([sys.executable, "-c", "import os, time; print(os.getpid(), flush=True); time.sleep(300)"],
                           clean_environment(), log_path)
    child_pid = None
    for _ in range(100):
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if text.strip():
            child_pid = int(text.split()[0])
            break
        time.sleep(0.1)
    test.check(child_pid is not None, "sleeper printed its pid")
    code, timed_out = runner.wait(time.time() + 1.0)
    test.check(timed_out, "sleeper timed out")
    if child_pid is not None:
        time.sleep(0.5)
        test.check(not process_alive(child_pid), f"interpreter {child_pid} (launcher {runner.process.pid}) killed")
        test.check(not process_alive(runner.process.pid), "launcher killed")
        if child_pid != runner.process.pid:
            print(f"{LOG_PREFIX} selftest: launcher pid {runner.process.pid}, interpreter pid {child_pid} "
                  f"(two processes, both ended)", flush=True)


def selftest_traces(test: SelfTest) -> None:
    by_directory = {}
    for relative, kind, key, metric, value, tolerance in TRACE_EXPECTATIONS:
        directory = _REPOSITORY_ROOT / relative
        if not (directory / "run_meta.json").exists():
            test.skipped.append(f"{relative} (trace absent)")
            continue
        if relative not in by_directory:
            by_directory[relative] = trace_run_metrics(directory)
        metrics = by_directory[relative]
        group = metrics["sims"] if kind == "sim" else metrics["links"]
        test.close((group.get(key) or {}).get(metric), value, tolerance, f"{relative} {kind} {key} {metric}")


def selftest(out_directory: pathlib.Path, skip_plan: bool) -> int:
    out_directory.mkdir(parents=True, exist_ok=True)
    test = SelfTest()
    started = time.time()
    for name, function in (("parser", lambda: selftest_parser(test)), ("bulk logs", lambda: selftest_bulk_logs(test)),
                           ("nvidia-smi", lambda: selftest_nvidia(test)),
                           ("plan rules", lambda: selftest_plan(test)),
                           ("summary", lambda: selftest_summary(test, out_directory)),
                           ("process tree", lambda: selftest_process_tree(test, out_directory)),
                           ("fake campaign", lambda: selftest_fake_campaign(test, out_directory)),
                           ("fake trace", lambda: selftest_fake_trace(test, out_directory)),
                           ("traces", lambda: selftest_traces(test))):
        before = (test.passed, len(test.failures))
        function()
        print(f"{LOG_PREFIX} selftest {name}: {test.passed - before[0]} passed, {len(test.failures) - before[1]} "
              f"failed", flush=True)
    if not skip_plan:
        print(f"{LOG_PREFIX} selftest: the dry-run plan of the default campaign", flush=True)
        code = run_campaign(None, list(CONFIGURATIONS), list(CHAIN), DEFAULT_TRIALS, resume=False, dry_run=True)
        test.check(code == 0, f"default dry-run plan exit {code}")
    for item in test.skipped:
        print(f"{LOG_PREFIX} selftest skipped: {item}", flush=True)
    print(f"{LOG_PREFIX} selftest {'PASS' if not test.failures else 'FAIL'}: {test.passed} checks passed, "
          f"{len(test.failures)} failed, {len(test.skipped)} skipped, {time.time() - started:.0f} s"
          + "".join(f"\n  - {failure}" for failure in test.failures), flush=True)
    return 0 if not test.failures else 1


# ----------------------------------------------------------------------------- command line


def resolved(path_text: Optional[str]) -> Optional[pathlib.Path]:
    if path_text is None:
        return None
    path = pathlib.Path(path_text)
    return path if path.is_absolute() else (_REPOSITORY_ROOT / path).resolve()


def names_from(text: Optional[str], known, default: list, what: str, parser) -> list:
    if not text:
        return list(default)
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if unknown:
        parser.error(f"unknown {what} {unknown}; expected {list(known)}")
    return [name for name in known if name in names]


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("plan", "run", "summarize", "trace", "selftest", "deep-wall-probe",
                                         "device-probe"))
    parser.add_argument("--out", default=None, help="campaign directory (relative to the checkout root)")
    parser.add_argument("--configs", default=None, help=f"comma list of {', '.join(CONFIGURATIONS)}")
    parser.add_argument("--builds", default=None, help=f"comma list of {', '.join(CHAIN)} (chain order is kept)")
    parser.add_argument("--trials", type=int, default=None,
                        help=f"trials (default {DEFAULT_TRIALS}; trace {DEFAULT_TRACE_TRIALS})")
    parser.add_argument("--resume", action="store_true", help="run / trace: continue the ledger in --out")
    parser.add_argument("--dry-run", action="store_true", help="run / trace: print the plan, no GPU")
    parser.add_argument("--preflight", action="store_true",
                        help="plan / --dry-run: also load and partition every case (CPU)")
    parser.add_argument("--refresh-deep-wall", action="store_true", help="recompute the deep-wall probe cache")
    parser.add_argument("--step-trace-detail", choices=("phases", "full"), default="phases",
                        help="trace: the chain bench's --step-trace-detail")
    parser.add_argument("--skip-plan", action="store_true", help="selftest: skip the default dry-run plan")
    parser.add_argument("--case", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--weights", action="append", default=[], help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.mode == "deep-wall-probe":
        return deep_wall_probe_main(arguments.case, arguments.weights)
    if arguments.mode == "device-probe":
        return device_probe_main()
    if arguments.mode == "selftest":
        return selftest(resolved(arguments.out or "logs/e39/perf_campaign_selftest"), arguments.skip_plan)
    out_directory = resolved(arguments.out)
    if arguments.mode == "summarize":
        if out_directory is None:
            parser.error("summarize needs --out")
        records = read_ledger(out_directory / "ledger.jsonl")
        if not records:
            print(f"{LOG_PREFIX} no ledger records in {out_directory}", flush=True)
            return 1
        code = 0
        if any(record.get("mode", "campaign") == "campaign" for record in records):
            code, _ = summarize_campaign(out_directory)
        if any(record.get("mode") == "trace" for record in records):
            code = max(code, summarize_traces(out_directory))
        return code
    trace = arguments.mode == "trace"
    configuration_names = names_from(arguments.configs, CONFIGURATIONS,
                                     list(TRACE_CONFIGURATIONS) if trace else list(CONFIGURATIONS),
                                     "configuration(s)", parser)
    build_names = names_from(arguments.builds, BUILDS, list(CHAIN), "build(s)", parser)
    trials = arguments.trials or (DEFAULT_TRACE_TRIALS if trace else DEFAULT_TRIALS)
    if trials < 1:
        parser.error("--trials must be at least 1")
    dry_run = arguments.dry_run or arguments.mode == "plan"
    if not dry_run and out_directory is None:
        parser.error(f"{arguments.mode} needs --out")
    return run_campaign(out_directory, configuration_names, build_names, trials, arguments.resume, dry_run,
                        arguments.preflight, "trace" if trace else "campaign", arguments.step_trace_detail,
                        refresh_deep_wall=arguments.refresh_deep_wall)


if __name__ == "__main__":
    sys.exit(main())
