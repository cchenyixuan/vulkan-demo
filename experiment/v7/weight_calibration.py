"""
weight_calibration.py — E32 part 2: slab weights from a short pilot run (--weights auto).

A pilot builds the chain with the starting weights (equal unless given),
bootstraps it (bootstrap + defrag, as every run), runs W warmup frames and M
measured frames at the run's depth (<= 2) and reads, for every measured frame
and sim, only the phase start / end timestamps (phase_trace_v7.PHASE_TICKS,
compute pool, no transfer timers, no clock calibration):

    T_A = a_start -> last of a_voxel_end / a_ghost_leading_end / a_ghost_trailing_end
    T_B = b_start -> last of b_density_deep_interior_end / b_correction_density_interior_end
          (E39 B1, the fused kernel) / b_force_deep_interior_end
    T_C = c_start -> c_force_end

a_start follows C(n - 1) and phase A's wait, c_start follows the upload_done
wait, so busy = T_A + T_B + T_C holds no waiting. A physical device hosting
one sim gets busy_d = the median over the measured frames (the E32 rule). A
device hosting several sims (--device-map 0,0,1) is busy whenever any of its
sims runs, and each of its sims advances one step per device step: busy_d =
the length of the union of all phase intervals of its sims over the measured
frames, per frame (one GPU timer: whether the driver runs the sims one after
the other or interleaves them, the union is the time the device works; if it
exceeds the elapsed time per frame the timestamps share no clock and the
largest sim median is used instead). Every sim k on device d gets
omega_k = fluid_k / busy_d, fluid_k from the case's initial fluid histogram
(_bin_fluid_counts, the histogram the cut rule splits). Walls count in the
time, not in the numerator: an end slab with walls gets fewer fluid particles.

The cuts of omega follow from chain_cuts_from_counts (the E31 rule). If they
differ from the pilot's, one more pilot runs with omega (at most
``rounds`` pilots in all, default 2); the last omega is the result. Each pilot
is a separate chain (new contexts, sims, timelines) that is torn down before
the next build: a chain cannot be run twice (frame numbers restart at 0, the
timeline values do not). The timed run is built afterwards, so pilot time is
never inside its timing window.

Weights file (--weights-file): the full record (rounds, busy times, fluid
counts, devices, node, commit, environment) plus the final weights at full
precision and their cuts. Reading it checks the slab count, the device map
and the fluid histogram (sha256), warns on a different node or case.yaml, and
recomputes the cuts from the stored weights (they must equal the stored cuts).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import time

import numpy as np

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
WEIGHTS_FILE_FORMAT = "e32_slab_weights_v1"
DEFAULT_PILOT_WARMUP = 200
DEFAULT_PILOT_STEPS = 300
DEFAULT_PILOT_ROUNDS = 2
MINIMUM_COMPLETE_FRACTION = 0.9
LOG_PREFIX = "[calibrate]"
# phase -> (start label, end labels: the phase ends at the last one present)
PHASE_LABELS = {
    "A": ("a_start", ("a_voxel_end", "a_ghost_leading_end", "a_ghost_trailing_end")),
    "B": ("b_start", ("b_density_deep_interior_end", "b_correction_density_interior_end", "b_force_deep_interior_end")),
    "C": ("c_start", ("c_force_end",)),
}


# ----------------------------------------------------------------------------- pure functions

def phase_intervals(ticks: dict):
    """{phase: (start tick, end tick)} of one frame of one sim (integer device ticks), or None when a
    phase lacks its start label or every end label (an incomplete frame)."""
    intervals = {}
    for phase, (start_label, end_labels) in PHASE_LABELS.items():
        ends = [ticks[label] for label in end_labels if label in ticks]
        if start_label not in ticks or not ends:
            return None
        intervals[phase] = (int(ticks[start_label]), int(max(ends)))
    return intervals


def interval_union_length(intervals) -> int:
    """Total length covered by integer intervals [start, end] (overlaps counted once)."""
    total, current_start, current_end = 0, None, None
    for start, end in sorted((int(start), int(end)) for start, end in intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def slab_fluid_counts(fluid_histogram, cuts) -> list:
    boundaries = [0] + [int(cut) for cut in cuts] + [len(fluid_histogram)]
    return [int(np.asarray(fluid_histogram)[boundaries[index]:boundaries[index + 1]].sum())
            for index in range(len(boundaries) - 1)]


def busy_statistics(frames_per_sim, device_map, ns_per_tick_per_sim) -> dict:
    """Per sim (median / p95 busy and per-phase medians, microseconds) and per physical device (the
    busy time its sims' weights use; see the module docstring). ``frames_per_sim[k]`` is the list of
    per-frame phase_intervals of sim k (None = incomplete)."""
    per_sim = []
    for frames, ns_per_tick in zip(frames_per_sim, ns_per_tick_per_sim):
        complete = [intervals for intervals in frames if intervals is not None]
        if not complete:
            per_sim.append({"frames": len(frames), "complete_frames": 0})
            continue
        phase_us = {phase: np.array([intervals[phase][1] - intervals[phase][0] for intervals in complete],
                                    dtype=np.float64) * ns_per_tick / 1000.0 for phase in PHASE_LABELS}
        busy_us = sum(phase_us.values())
        per_sim.append({"frames": len(frames), "complete_frames": len(complete),
                        "busy_p50_us": float(np.median(busy_us)), "busy_p95_us": float(np.percentile(busy_us, 95)),
                        "busy_mean_us": float(np.mean(busy_us)),
                        **{f"{phase.lower()}_p50_us": float(np.median(values)) for phase, values in phase_us.items()}})
    per_device = {}
    for device in sorted(set(device_map)):
        members = [index for index, value in enumerate(device_map) if value == device]
        if any(per_sim[index]["complete_frames"] == 0 for index in members):
            raise RuntimeError(f"device {device}: a sim has no complete pilot frame")
        if len(members) == 1:
            per_device[device] = {"sims": members, "busy_us": per_sim[members[0]]["busy_p50_us"],
                                  "method": "median of T_A + T_B + T_C"}
            continue
        # frames complete on every member; the union over all of them, per frame
        frame_count = min(len(frames_per_sim[index]) for index in members)
        usable = [frame for frame in range(frame_count)
                  if all(frames_per_sim[index][frame] is not None for index in members)]
        intervals = [interval for index in members for frame in usable
                     for interval in frames_per_sim[index][frame].values()]
        ns_per_tick = ns_per_tick_per_sim[members[0]]
        union_us = interval_union_length(intervals) * ns_per_tick / 1000.0 / max(1, len(usable))
        # elapsed time per frame of each sim (first measured start to last measured end): on one clock the
        # union cannot exceed it; when it does, the sims' timestamps do not share a time base and the
        # union counts concurrent work twice
        spans = []
        for index in members:
            complete = [(frame, intervals_of_frame) for frame, intervals_of_frame
                        in enumerate(frames_per_sim[index]) if intervals_of_frame is not None]
            (first_frame, first), (last_frame, last) = complete[0], complete[-1]
            spans.append((max(end for _, end in last.values()) - min(start for start, _ in first.values()))
                         * ns_per_tick / 1000.0 / (last_frame - first_frame + 1))
        entry = {"sims": members, "union_us": union_us, "span_per_frame_us": max(spans),
                 "sim_medians_us": [per_sim[index]["busy_p50_us"] for index in members], "frames_used": len(usable)}
        if union_us <= max(spans) * 1.02:
            entry.update({"busy_us": union_us, "method": "union of the sims' phase intervals per frame"})
        else:
            entry.update({"busy_us": max(entry["sim_medians_us"]),
                          "method": "largest sim median (union longer than the elapsed time: no common clock)"})
        per_device[device] = entry
    return {"per_sim": per_sim, "per_device": per_device}


def omega_from_busy(fluid_per_slab, busy_per_slab) -> list:
    """Fluid particles per microsecond of busy time of every slab (only ratios matter)."""
    return [float(fluid) / float(busy) for fluid, busy in zip(fluid_per_slab, busy_per_slab)]


def normalized(weights) -> list:
    """Weights scaled to mean 1 (only the ratios enter the cut rule)."""
    weights = [float(value) for value in weights]
    mean = sum(weights) / len(weights)
    return [value / mean for value in weights]


def histogram_digest(fluid_histogram) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(fluid_histogram, dtype=np.int64)).tobytes()).hexdigest()


def file_digest(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- pilot (GPU)

class PhaseBusyRecorder:
    """Phase start / end ticks of every frame in [first_frame, stop_frame) of every sim. Attach BEFORE
    bootstrap_all (installs a StepTraceTimer per sim, compute pool only, and the frame-parity command
    buffers), set ``orchestrator.on_frame_done = recorder.on_frame_done`` after it, close() before the
    sims are destroyed. Needs depth <= 2: frame n's parity block survives until phase A of frame n + 2,
    which the loop submits only after on_frame_done(n)."""

    def __init__(self, sims, first_frame: int, stop_frame: int):
        from experiment.v7.utils.phase_trace_v7 import PHASE_TICKS, StepTraceTimer
        self.sims = list(sims)
        self.first_frame, self.stop_frame = int(first_frame), int(stop_frame)
        self.timers = []
        for index, sim in enumerate(self.sims):
            timer = StepTraceTimer(sim.ctx, f"calibrate_s{index}", keep_labels=PHASE_TICKS)
            sim.bench, sim.bench_transfer = timer, None
            sim.step_trace_parity = True
            self.timers.append(timer)
        self.raw = [[] for _ in self.sims]
        self._closed = False

    def on_frame_done(self, frame_n: int, sim_index=None) -> None:
        if not self.first_frame <= frame_n < self.stop_frame:
            return
        indices = range(len(self.sims)) if sim_index is None else (sim_index,)
        for index in indices:
            self.raw[index].append(self.timers[index].read_block_raw(frame_n % 2))

    def frames(self) -> list:
        return [[phase_intervals(timer.parse_block(raw)) for raw in blocks]
                for timer, blocks in zip(self.timers, self.raw)]

    def ns_per_tick(self) -> list:
        return [float(timer.ns_per_tick) for timer in self.timers]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        from vulkan import vkDeviceWaitIdle
        for sim, timer in zip(self.sims, self.timers):
            try:
                vkDeviceWaitIdle(sim.ctx.device)
            except Exception:   # noqa: BLE001 — a lost device must not hide the original error
                pass
            timer.destroy()
            sim.bench = sim.bench_transfer = None
            sim.step_trace_parity = False


def device_identity(ctx, device_index: int) -> dict:
    from vulkan import (VkPhysicalDeviceIDProperties, VkPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties,
                        vkGetPhysicalDeviceProperties2)
    id_properties = VkPhysicalDeviceIDProperties()
    vkGetPhysicalDeviceProperties2(ctx.physical_device, VkPhysicalDeviceProperties2(pNext=id_properties))
    name = vkGetPhysicalDeviceProperties(ctx.physical_device).deviceName
    return {"device_index": int(device_index), "name": str(name), "uuid": bytes(id_properties.deviceUUID).hex()}


def run_pilot(global_case, weights, device_map, settings: dict, log) -> dict:
    """One pilot chain: build, bootstrap, W + M frames, statuses, teardown. Raises on any invariant."""
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    started = time.perf_counter()
    chain = compute_chain_partition(global_case, list(weights), settings["pool_safety"])
    expected_total = int(global_case.initial.positions.shape[0])
    warmup, steps = int(settings["warmup"]), int(settings["steps"])
    contexts, sims, recorder = [], [], None
    try:
        for index, device in enumerate(device_map):
            contexts.append(VulkanContextV7.create(device_index=device, enable_validation=settings["validation"],
                                                   application_name=f"calibrate_s{index}"))
            sims.append(SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme=settings["sync_scheme"]))
        recorder = PhaseBusyRecorder(sims, warmup, warmup + steps)
        with ChainOrchestratorV7(sims, defrag_cadence=settings["defrag_cadence"]) as orchestrator:
            orchestrator.bootstrap_all()
            orchestrator.on_frame_done = recorder.on_frame_done
            result = orchestrator.run_pipelined(warmup + steps, depth=min(int(settings["depth"]), 2), warmup=warmup)
            for sim in sims:
                sim.submit_defrag_and_wait()
            statuses = [dict(sim.readback_global_status()) for sim in sims]
            host_stamp_errors = sum(int(getattr(worker, "stamp_error_count", 0))
                                    for worker in getattr(orchestrator, "workers", []))
        devices = [device_identity(ctx, device) for ctx, device in zip(contexts, device_map)]
        frames, ns_per_tick = recorder.frames(), recorder.ns_per_tick()
    finally:
        if recorder is not None:
            recorder.close()
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()
    invariants = {"drift": sum(int(status["alive_particle_count"]) for status in statuses) - expected_total,
                  "overflow_total": sum(int(value) for status in statuses for key, value in status.items()
                                        if key.startswith("overflow_")),
                  "stamp_errors_gpu": sum(int(status.get("stamp_error_count", 0)) for status in statuses),
                  "stamp_errors_host": int(host_stamp_errors),
                  "far_migration_total": sum(int(status.get("far_migration_count", 0)) for status in statuses)}
    if any(invariants.values()):
        raise RuntimeError(f"pilot invariant violated: {invariants}")
    statistics = busy_statistics(frames, device_map, ns_per_tick)
    for index, entry in enumerate(statistics["per_sim"]):
        if entry["complete_frames"] < MINIMUM_COMPLETE_FRACTION * steps:
            raise RuntimeError(f"pilot sim {index}: only {entry['complete_frames']} of {steps} frames have every "
                               f"phase tick")
    return {"cuts": [int(cut) for cut in chain.cuts],
            "own_columns": [int(g.own_global_last_column - g.own_global_first_column + 1) for g in chain.geometry],
            "all": [int(g.own_particle_count) for g in chain.geometry],
            "statistics": statistics, "invariants": invariants, "devices": devices,
            "pilot_steady_fps": float(result.get("steady_fps", 0.0)),
            "seconds": time.perf_counter() - started}


def git_provenance() -> dict:
    def git(*arguments):
        try:
            return subprocess.run(["git", *arguments], cwd=REPOSITORY, capture_output=True, text=True,
                                  timeout=30).stdout
        except Exception:   # noqa: BLE001 — provenance is best effort
            return ""
    status = git("status", "--porcelain", "--", "experiment/v7")
    return {"commit": git("rev-parse", "HEAD").strip(),
            "v7_dirty_files": [line[3:] for line in status.splitlines() if len(line) > 3]}


def calibrate(global_case, case_path, device_map, settings: dict, initial_weights=None, log=print) -> dict:
    """Pilot rounds (module docstring). Returns the record; record['weights'] / ['cuts'] are final."""
    from experiment.v7.utils.partition_v7 import (MINIMUM_OWN_COLUMNS_HARD, _bin_fluid_counts,
                                                  chain_cuts_from_counts, configured_band_widths)
    slab_count = len(device_map)
    fluid_histogram = _bin_fluid_counts(global_case)
    weights = normalized(initial_weights or [1.0] * slab_count)
    cuts = chain_cuts_from_counts(fluid_histogram, weights, MINIMUM_OWN_COLUMNS_HARD)
    rounds = []
    started = time.perf_counter()
    for round_number in range(1, int(settings["rounds"]) + 1):
        pilot = run_pilot(global_case, weights, device_map, settings, log)
        if pilot["cuts"] != cuts:
            raise AssertionError(f"pilot cuts {pilot['cuts']} != cut rule {cuts}")
        fluid = slab_fluid_counts(fluid_histogram, cuts)
        per_device = pilot["statistics"]["per_device"]
        busy = [per_device[device]["busy_us"] for device in device_map]
        omega = normalized(omega_from_busy(fluid, busy))
        next_cuts = chain_cuts_from_counts(fluid_histogram, omega, MINIMUM_OWN_COLUMNS_HARD)
        entry = {"round": round_number, "weights": weights, "cuts": cuts, "own_columns": pilot["own_columns"],
                 "fluid": fluid, "all": pilot["all"], "busy_us": busy, "omega": omega, "next_cuts": next_cuts,
                 "changed": next_cuts != cuts, **{key: pilot[key] for key in
                                                  ("statistics", "invariants", "devices", "pilot_steady_fps",
                                                   "seconds")}}
        rounds.append(entry)
        per_sim = pilot["statistics"]["per_sim"]
        log(f"{LOG_PREFIX} round {round_number}: weights={[round(value, 4) for value in weights]} cuts={cuts} "
            f"own_columns={pilot['own_columns']} fluid={fluid} all={pilot['all']}")
        for index, item in enumerate(per_sim):
            log(f"{LOG_PREFIX} round {round_number}: sim{index} (dev{device_map[index]}) busy p50 "
                f"{item['busy_p50_us']:.1f} us p95 {item['busy_p95_us']:.1f} us, A/B/C p50 "
                f"{item['a_p50_us']:.1f}/{item['b_p50_us']:.1f}/{item['c_p50_us']:.1f} us, "
                f"{item['complete_frames']}/{item['frames']} frames")
        for device, item in per_device.items():
            log(f"{LOG_PREFIX} round {round_number}: device {device} (sims {item['sims']}) busy "
                f"{item['busy_us']:.1f} us ({item['method']})")
        log(f"{LOG_PREFIX} round {round_number}: omega={[round(value, 4) for value in omega]} -> cuts {next_cuts} "
            f"({'changed' if entry['changed'] else 'unchanged'}); pilot {pilot['pilot_steady_fps']:.1f} fps, "
            f"{pilot['seconds']:.1f} s, invariants {pilot['invariants']}")
        weights, cuts = omega, next_cuts
        if not entry["changed"]:
            break
    environment = {key: value for key, value in sorted(os.environ.items()) if key.startswith(("V7_", "VK_"))}
    record = {"format": WEIGHTS_FILE_FORMAT, "case": str(case_path),
              "case_sha256": file_digest(case_path), "particle_count": int(global_case.initial.positions.shape[0]),
              "grid_nx": int(len(fluid_histogram)), "fluid_histogram_sha256": histogram_digest(fluid_histogram),
              "slab_count": slab_count, "device_map": [int(device) for device in device_map],
              "devices": rounds[-1]["devices"], "host": platform.node(),
              "settings": {key: settings[key] for key in ("warmup", "steps", "rounds", "depth", "pool_safety",
                                                          "sync_scheme", "defrag_cadence")},
              "band_widths": [int(value) for value in configured_band_widths()],
              "environment": environment, **git_provenance(),
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": time.perf_counter() - started,
              "rounds": rounds, "weights": weights, "cuts": cuts}
    log(f"{LOG_PREFIX} final weights={weights} cuts={cuts} after {len(rounds)} pilot round(s), "
        f"{record['seconds']:.1f} s")
    return record


# ----------------------------------------------------------------------------- weights file

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def write_weights_file(path, record: dict) -> str:
    """Write the calibration record; returns its sha256."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(json_ready(record), indent=1), encoding="utf-8")
    os.replace(temporary, path)
    return file_digest(path)


def read_weights_file(path, global_case, case_path, device_map=None, log=print) -> dict:
    """The record of a weights file, checked against this run (module docstring). device_map None =
    take the file's."""
    from experiment.v7.utils.partition_v7 import (MINIMUM_OWN_COLUMNS_HARD, _bin_fluid_counts,
                                                  chain_cuts_from_counts)
    record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if record.get("format") != WEIGHTS_FILE_FORMAT:
        raise ValueError(f"{path}: format {record.get('format')!r}, expected {WEIGHTS_FILE_FORMAT!r}")
    weights = [float(value) for value in record["weights"]]
    if len(weights) != int(record["slab_count"]) or len(record["device_map"]) != len(weights):
        raise ValueError(f"{path}: {len(weights)} weights for slab count {record['slab_count']} and device map "
                         f"{record['device_map']}")
    if device_map is not None and [int(device) for device in device_map] != record["device_map"]:
        raise ValueError(f"{path}: calibrated on device map {record['device_map']}, this run uses {list(device_map)}")
    fluid_histogram = _bin_fluid_counts(global_case)
    if histogram_digest(fluid_histogram) != record["fluid_histogram_sha256"]:
        raise ValueError(f"{path}: calibrated on another particle set (fluid histogram differs)")
    cuts = chain_cuts_from_counts(fluid_histogram, weights, MINIMUM_OWN_COLUMNS_HARD)
    if cuts != record["cuts"]:
        raise ValueError(f"{path}: the stored weights give cuts {cuts}, the file says {record['cuts']}")
    if file_digest(case_path) != record["case_sha256"]:
        log(f"{LOG_PREFIX} WARNING: {case_path} differs from the calibrated case.yaml (same particles)")
    if platform.node() != record["host"]:
        log(f"{LOG_PREFIX} WARNING: weights calibrated on node {record['host']}, this is {platform.node()}")
    return record
