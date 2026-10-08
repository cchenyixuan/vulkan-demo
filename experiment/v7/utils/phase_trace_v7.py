"""
phase_trace_v7.py — per-frame, per-sim phase timestamps on ONE host clock.

Question (2026-09-16, after the N56 fixed-N K sweep): the two directions of a
K=2 link show ~25 ms different transport slack, i.e. the two GPUs may run with
a systematic phase offset. This tracer records, for every frame and every sim,
the GPU timestamps of phase A start / end, phase B start / end and phase C
start / end, and maps them onto the host QueryPerformanceCounter clock through
VK_KHR_calibrated_timestamps (device tick <-> QPC pairs sampled every
``calibrate_every`` frames). The analysis script fits one linear map per sim
and plots the inter-GPU phase difference frame by frame.

How the per-frame capture works with the pre-recorded command buffers: the
BenchTimer's parity regions keep phase-C ticks of frame n alive until phase C
of frame n+2 resets them, so a read right after frame_done(n) reliably yields
C(n). The A/B slots are reset by phase A of every frame; a read after
frame_done(n) yields either A/B(n) (if A(n+1) has not started) or A/B(n+1),
distinguished by a_start vs c_end(n): ticks are in queue order, so
a_start <= c_start(n) means frame n, a_start > c_end(n) means frame n+1.
Frames whose A/B ticks were never caught keep only their C ticks; the analysis
falls back to c_end(n-1) (+ the measured c_to_a gap) for phase A start.

Usage (chain bench): ``--phase-trace DIR`` creates compute BenchTimers if
``--anatomy`` is off, requests VK_KHR_calibrated_timestamps on every device
and writes DIR/phase_trace.csv + DIR/calibration.csv at the end.

E29 (2026-10-05): ``StepTracer`` (``--step-trace DIR``) records EVERY step of
the depth-2 production loop instead: per-frame-parity timestamp slots for
phase A/B/C and both transfer queues, the transport workers' host time points
and bytes, all on one host clock -> DIR/steps_device.csv, steps_link.csv,
run_meta.json, calibration.csv (field spec: docs/perf_model/E29_local.md;
analysis: experiment/v7/analysis/step_trace_model.py).
"""
from __future__ import annotations

import csv
import ctypes
import json
import os
import pathlib
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np
from vulkan import (
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    VK_QUERY_RESULT_64_BIT,
    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
    VK_TIME_DOMAIN_DEVICE_KHR,
    VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR,
    VkCalibratedTimestampInfoKHR,
    vkCmdResetQueryPool,
    vkCmdWriteTimestamp,
    vkGetDeviceProcAddr,
    vkGetInstanceProcAddr,
)
from vulkan._vulkan import lib as _lib
from vulkan._vulkancache import ffi

from experiment.v7.utils.bench_v7 import _MAX_TICKS, BenchTimer, split_parity_ticks
from experiment.v7.utils.clock_map_v7 import CLOCK_FIT_VERSION, clock_to_host, fit_clock

CALIBRATED_TIMESTAMPS_EXTENSION = "VK_KHR_calibrated_timestamps"
# --step-trace requests either name (the context enables the first the device offers): drivers older
# than the KHR promotion offer only the EXT one (A100 driver 535 on N32-H, E30). Same functions and
# structures; python-vulkan 1.3.275.1 has no VkCalibratedTimestampInfoEXT and fills
# VkCalibratedTimestampInfoKHR with sType 1000543000, so the EXT path passes the spec value
# 1000184000 (VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT) explicitly.
CALIBRATED_TIMESTAMPS_EXTENSION_CHOICES = ("VK_KHR_calibrated_timestamps", "VK_EXT_calibrated_timestamps")
CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE_EXT = 1000184000


def calibrated_timestamps_suffix(ctx) -> str:
    """'KHR' or 'EXT': the calibrated-timestamps extension the context enabled (KHR when neither,
    so the lookup fails with the KHR name as before)."""
    enabled = getattr(ctx, "enabled_device_extensions", ())
    if "VK_KHR_calibrated_timestamps" not in enabled and "VK_EXT_calibrated_timestamps" in enabled:
        return "EXT"
    return "KHR"

_A_END_LABELS = ("a_ghost_trailing_end", "a_ghost_leading_end", "a_voxel_end", "a_predict_end")
_B_END_LABELS = ("b_force_deep_interior_end", "b_density_deep_interior_end",
                 "b_correction_density_interior_end", "b_correction_interior_end")   # fused: E39 B1
_FIELDS = ("a_start", "a_end", "b_start", "b_end", "c_start", "c_end", "prev_c_end")


def _last_present(ticks: dict, labels) -> Optional[float]:
    for label in labels:
        if label in ticks:
            return ticks[label]
    return None


class PhaseTracer:
    """One instance per chain run. ``timers[i]`` is sim i's compute BenchTimer
    (parity regions enabled). Call ``on_frame_done(frame_n, sim_index)`` from
    the orchestrator as soon as frame ``frame_n`` of one sim (or of every sim
    when ``sim_index`` is None) is known complete; call ``write(out_dir)`` at
    the end."""

    def __init__(self, sims, timers, calibrate_every: int = 100):
        if len(sims) != len(timers):
            raise ValueError("one BenchTimer per sim required")
        self.sims = list(sims)
        self.timers = list(timers)
        self.calibrate_every = max(1, int(calibrate_every))
        self._get_calibrated = []
        self._infos = [VkCalibratedTimestampInfoKHR(timeDomain=VK_TIME_DOMAIN_DEVICE_KHR),
                       VkCalibratedTimestampInfoKHR(
                           timeDomain=VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR)]
        self._stamps = ffi.new("uint64_t[2]")
        for sim in self.sims:
            function = vkGetDeviceProcAddr(sim.ctx.device, "vkGetCalibratedTimestampsKHR")
            if function is None:
                raise RuntimeError("vkGetCalibratedTimestampsKHR unavailable — was "
                                   f"{CALIBRATED_TIMESTAMPS_EXTENSION} enabled on the device?")
            self._get_calibrated.append(function)
        # rows[(frame, sim)] -> {field: device_ns}
        self.rows: dict[tuple[int, int], dict[str, float]] = {}
        # A/B ticks read after frame_done(n) but belonging to frame n+1
        self._pending_ab: dict[int, tuple[int, dict[str, float]]] = {}
        self._last_frame = [-1] * len(self.sims)
        self._host_seen: dict[tuple[int, int], int] = {}
        # calibration samples: (sim, frame, device_ns, qpc_ticks, perf_ns, deviation_ns)
        self.calibration: list[tuple[int, int, float, int, int, float]] = []
        for sim_index in range(len(self.sims)):
            self.calibrate(sim_index, frame_n=-1, repeat=5)

    # ------------------------------------------------------------ calibration

    def calibrate(self, sim_index: int, frame_n: int, repeat: int = 1) -> None:
        """Sample (device tick, QPC) pairs; keep them all (the analysis fits a
        line), ``repeat`` > 1 takes several back-to-back samples."""
        sim = self.sims[sim_index]
        ns_per_tick = self.timers[sim_index].ns_per_tick
        for _ in range(repeat):
            deviation = self._get_calibrated[sim_index](
                sim.ctx.device, 2, self._infos, self._stamps)
            perf_ns = time.perf_counter_ns()
            device_ns = float(int(self._stamps[0])) * ns_per_tick
            qpc = int(self._stamps[1])
            self.calibration.append((sim_index, frame_n, device_ns, qpc, perf_ns,
                                     float(int(deviation)) * ns_per_tick))

    # ---------------------------------------------------------------- capture

    def on_frame_done(self, frame_n: int, sim_index: Optional[int] = None) -> None:
        indices = range(len(self.sims)) if sim_index is None else (sim_index,)
        for index in indices:
            if self._last_frame[index] >= frame_n:
                continue   # drains re-wait frames already captured
            self._last_frame[index] = frame_n
            self._capture(frame_n, index)
            if frame_n % self.calibrate_every == 0:
                self.calibrate(index, frame_n)

    def _capture(self, frame_n: int, index: int) -> None:
        host_ns = time.perf_counter_ns()
        timer = self.timers[index]
        ticks = {label: value for label, value in
                 timer.read_frame(include_defrag=False).items() if value > 0.0}
        previous_c_end = None
        if timer.parity_regions:
            ticks, previous_c_end = split_parity_ticks(ticks, frame_n % 2)
        row = self.rows.setdefault((frame_n, index), {})
        self._host_seen[(frame_n, index)] = host_ns
        c_start = ticks.get("c_start")
        c_end = ticks.get("c_force_end")
        if c_start is not None:
            row["c_start"] = c_start
        if c_end is not None:
            row["c_end"] = c_end
        if previous_c_end is not None:
            row["prev_c_end"] = previous_c_end
        # A/B ticks caught earlier (read after frame n-1) and attributed to n
        pending = self._pending_ab.pop(index, None)
        if pending is not None and pending[0] == frame_n:
            row.update(pending[1])
        ab = {}
        a_start = ticks.get("a_start")
        if a_start is not None:
            ab["a_start"] = a_start
            # a_end / b_end only when the phase's LAST tick (in recording
            # order) is present: the A/B slots are reset by the next frame's
            # phase A and refill in order, so a partial read must not pass
            # an earlier tick off as the phase end.
            a_end_label, b_end_label = self._phase_end_labels(timer)
            if a_end_label in ticks:
                ab["a_end"] = ticks[a_end_label]
            if "b_start" in ticks:
                ab["b_start"] = ticks["b_start"]
            if b_end_label in ticks:
                ab["b_end"] = ticks[b_end_label]
            if c_start is not None and a_start <= c_start:
                row.update(ab)                       # frame n's own A/B
            elif c_end is not None and a_start > c_end:
                self._pending_ab[index] = (frame_n + 1, ab)   # already frame n+1
            elif c_start is None and c_end is None:
                row.update(ab)                       # no C reference: assume n

    def _phase_end_labels(self, timer) -> tuple[str, str]:
        """(last A label, last B label) in the timer's recording order."""
        cached = getattr(timer, "_phase_end_labels", None)
        if cached is None:
            a_labels = [l for l in timer.label_to_slot if l.startswith("a_")]
            b_labels = [l for l in timer.label_to_slot if l.startswith("b_")]
            cached = (a_labels[-1] if a_labels else "a_start",
                      b_labels[-1] if b_labels else "b_start")
            timer._phase_end_labels = cached
        return cached

    # ------------------------------------------------------------------ output

    def write(self, out_dir) -> dict:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for sim_index in range(len(self.sims)):
            self.calibrate(sim_index, frame_n=max(self._last_frame), repeat=5)
        with open(out / "phase_trace.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "sim", "host_seen_ns"] + list(_FIELDS))
            for (frame_n, index) in sorted(self.rows):
                row = self.rows[(frame_n, index)]
                writer.writerow([frame_n, index, self._host_seen.get((frame_n, index), "")]
                                + [f"{row[f]:.0f}" if f in row else "" for f in _FIELDS])
        with open(out / "calibration.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sim", "frame", "device_ns", "qpc_ticks", "perf_ns", "max_deviation_ns"])
            for sample in self.calibration:
                writer.writerow([sample[0], sample[1], f"{sample[2]:.0f}", sample[3], sample[4],
                                 f"{sample[5]:.0f}"])
        frames = [f for (f, _) in self.rows]
        complete = sum(1 for row in self.rows.values() if "a_start" in row and "c_end" in row)
        summary = {"rows": len(self.rows), "frames": (min(frames), max(frames)) if frames else None,
                   "rows_with_a_start": complete, "calibration_samples": len(self.calibration),
                   "max_deviation_ns": max(s[5] for s in self.calibration) if self.calibration else None}
        print(f"[phase_trace] wrote {out / 'phase_trace.csv'}: {summary}", flush=True)
        return summary


# ============================================================================
# E29 step trace (docs/perf_model/E29_local.md): EVERY step of the production
# depth-2 loop — every sim's phase A/B/C, both transfer queues' DMAs and the
# transport workers' host time points — on ONE host clock.
#
# Slot layout: per-frame labels own one slot per frame parity (block 0 = even
# frames, block 1 = odd), defrag ticks live after both blocks. simulator_v7
# records phase A / B / C and the transfer cmds once per parity
# (step_trace_parity); phase A of parity p resets block p of the compute pool
# (its first action) and of the transfer pool (record_external_reset, compute
# queue). Frame n's block therefore survives until phase A of frame n + 2,
# which the depth-2 loop submits only after on_frame_done(n) has read it.
#
# Clock: VK_KHR_calibrated_timestamps pairs (device tick, host time) are
# sampled by a helper thread every ``calibrate_ms`` (off the main loop; the
# driver call does not hold the GIL) and in larger bursts at the start, at the
# warmup boundary and at the end; each burst keeps the pair with the
# smallest driver maxDeviation (on the 5090 / Windows driver >= ~12 us, and the
# host stamp is biased by up to that much). Device ticks stay integers until an
# integer per-device origin is subtracted (NVIDIA ticks are epoch ns ~1.8e18,
# where float64 has a 256 ns grid). fit_clock keeps only pairs within 1 us of
# the run's smallest maxDeviation (wider pairs sit late by up to their width),
# fits a least-squares line plus the running median of its residuals (9 pairs,
# interpolated in device time; the frequency ratio drifts by a few ppm over
# minutes) and drops pairs off line + drift by more than 3 sigma. The map takes
# every GPU tick (compute and transfer pools share the device domain) to the
# host clock of the worker time points: QueryPerformanceCounter =
# time.perf_counter_ns on Windows; on Linux (untested) CLOCK_MONOTONIC_RAW
# (time.clock_gettime_ns, also for the workers through
# transport_v7.set_host_clock) when offered, else CLOCK_MONOTONIC =
# perf_counter_ns.
# ============================================================================

STEP_TRACE_BLOCK = 32              # timestamp slots per frame parity
STEP_TRACE_DEFRAG_BASE = 64        # defrag ticks after the two parity blocks
TIME_DOMAIN_DEVICE = 0             # VkTimeDomainKHR
TIME_DOMAIN_CLOCK_MONOTONIC = 1
TIME_DOMAIN_CLOCK_MONOTONIC_RAW = 2
TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER = 3
TIME_DOMAIN_NAMES = {TIME_DOMAIN_CLOCK_MONOTONIC: "CLOCK_MONOTONIC",
                     TIME_DOMAIN_CLOCK_MONOTONIC_RAW: "CLOCK_MONOTONIC_RAW",
                     TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER: "QUERY_PERFORMANCE_COUNTER"}

# steps_device.csv: one row per step and sim, host ns. *_end = the phase's last
# tick present (queue order); a phase whose cmd has no tick of a label leaves
# that column empty (e.g. no install on a slab without that peer).
STEP_DEVICE_TIMES = (
    "a_start", "a_predict_end", "a_voxel_end", "a_ghost_leading_end", "a_ghost_trailing_end", "a_end",
    "b_start", "b_deep_wall_marker_end", "b_correction_interior_end", "b_density_deep_interior_end",
    "b_correction_density_interior_end", "b_wall_extrapolate_end", "b_force_deep_interior_end", "b_end",
    "c_start", "c_expand_end", "c_install_leading_end", "c_install_trailing_end", "c_append_departed_end",
    "c_band_compact_end", "c_correction_boundary_end", "c_density_boundary_end",
    "c_correction_density_boundary_end", "c_density_end",
    "c_wall_extrapolate_end", "c_force_end", "c_end")       # *_wall_extrapolate_end: E37 adami only;
# b_deep_wall_marker_end: E39 B4 V7_DEEP_WALL_SKIP only (the deep-wall marker at the start of phase B);
# b_correction_density_interior_end / c_correction_density_boundary_end: E39 B1 V7_FUSED_CORRECTION_DENSITY
# (the fused kernel's end, in place of the correction / density pairs' two ticks)
# steps_link.csv: one row per step and directed link (sender -> receiver), host ns.
STEP_LINK_TIMES = (
    "send_end",                       # sender's ghost_send for this link done (compute queue)
    "readback_start", "readback_copy_end", "readback_end",    # sender transfer queue
    "worker_dequeue", "worker_source_wait", "worker_dest_guard", "worker_upload_guard",
    "worker_stamp", "worker_copy", "worker_dest_signal", "worker_signal",   # transport worker (host)
    "upload_start", "upload_end",                             # receiver transfer queue
    "receiver_b_start", "receiver_b_end", "receiver_c_start")
# detail = "phases": only these compute ticks are written (a phase's last tick depends on the
# configuration: a_voxel_end without peers, b_density_deep_interior_end (E39 B1 fused:
# b_correction_density_interior_end) without cascade force).
# Transfer ticks are always written. "full" writes every tick the simulator records.
PHASE_TICKS = ("a_start", "a_voxel_end", "a_ghost_leading_end", "a_ghost_trailing_end",
               "b_start", "b_density_deep_interior_end", "b_correction_density_interior_end",
               "b_force_deep_interior_end",
               "c_start", "c_force_end", "defrag_start", "defrag_end")
_WORKER_KEYS = (("worker_dequeue", "dequeue_ns"), ("worker_source_wait", "source_wait_ns"),
                ("worker_dest_guard", "dest_guard_ns"), ("worker_upload_guard", "wait_ns"),
                ("worker_stamp", "stamp_ns"), ("worker_copy", "copy_ns"),
                ("worker_dest_signal", "dest_signal_ns"), ("worker_signal", "signal_ns"))


class StepTraceTimer(BenchTimer):
    """BenchTimer whose per-frame labels own one slot per frame parity (see the
    section comment). Labels keep their plain names; the parity of the cmd
    being recorded is set by the simulator (set_recording_parity) and by phase
    C's begin_phase_c_region."""

    def __init__(self, ctx, label: str, queue_family_index=None, keep_labels=None):
        super().__init__(ctx, label, queue_family_index)
        self.keep_labels = None if keep_labels is None else frozenset(keep_labels)
        self.parity_regions = True        # the simulator records the odd-frame phase C cmd
        self.recording_parity = 0
        self.block_labels: list[str] = []
        self.defrag_labels: list[str] = []
        self._results = ffi.new(f"uint64_t[{2 * STEP_TRACE_BLOCK}]")

    def enable_parity_regions(self) -> None:
        return

    def set_recording_parity(self, parity: int) -> None:
        self.recording_parity = int(parity)

    def begin_phase_c_region(self, cmd, parity: int) -> None:
        self.recording_parity = int(parity)   # block already reset by this parity's phase A

    def end_phase_c_region(self) -> None:
        return

    def _slot(self, label: str) -> int:
        if label.startswith("defrag"):
            if label not in self.defrag_labels:
                self.defrag_labels.append(label)
            slot = STEP_TRACE_DEFRAG_BASE + self.defrag_labels.index(label)
        else:
            if label not in self.block_labels:
                if len(self.block_labels) >= STEP_TRACE_BLOCK:
                    raise RuntimeError(f"StepTraceTimer({self.label}): parity block full: {self.block_labels}")
                self.block_labels.append(label)
            slot = self.recording_parity * STEP_TRACE_BLOCK + self.block_labels.index(label)
        if slot >= _MAX_TICKS:
            raise RuntimeError(f"StepTraceTimer({self.label}): slot {slot} >= {_MAX_TICKS}")
        return slot

    def tick(self, cmd, label: str) -> None:
        if self.keep_labels is not None and label not in self.keep_labels:
            return
        slot = self._slot(label)
        self.label_to_slot.setdefault(label, slot)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.pool, slot)

    def record_step_reset_and_start(self, cmd, start_label: str = "a_start") -> None:
        vkCmdResetQueryPool(cmd, self.pool, self.recording_parity * STEP_TRACE_BLOCK, STEP_TRACE_BLOCK)
        self.tick(cmd, start_label)

    def record_external_reset(self, cmd) -> None:
        vkCmdResetQueryPool(cmd, self.pool, self.recording_parity * STEP_TRACE_BLOCK, STEP_TRACE_BLOCK)

    def record_defrag_reset_and_start(self, cmd, start_label: str = "defrag_start") -> None:
        vkCmdResetQueryPool(cmd, self.pool, STEP_TRACE_DEFRAG_BASE, _MAX_TICKS - STEP_TRACE_DEFRAG_BASE)
        self.tick(cmd, start_label)

    def read_block_raw(self, parity: int) -> bytes:
        """The parity block as raw (value, availability) uint64 pairs; parsed later. Raw
        cffi call (as the fast submit path): this runs once per step on the main loop.
        VK_NOT_READY only means some slot is unavailable; its availability word says so."""
        count = len(self.block_labels)
        if count == 0:
            return b""
        _lib.vkGetQueryPoolResults(self.ctx.device, self.pool, parity * STEP_TRACE_BLOCK, count,
                                   16 * count, self._results, 16,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
        return ffi.buffer(self._results, 16 * count)[:]

    def parse_block(self, raw: bytes) -> dict:
        """{label: raw device tick (int)} of the available slots."""
        if not raw:
            return {}
        values = struct.unpack(f"<{len(raw) // 8}Q", raw)
        return {label: values[2 * index]
                for index, label in enumerate(self.block_labels[:len(values) // 2])
                if values[2 * index + 1]}


def _git_provenance() -> dict:
    root = pathlib.Path(__file__).resolve().parents[3]

    def git(*arguments):
        try:
            return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True,
                                  timeout=30).stdout
        except Exception:   # noqa: BLE001 — provenance is best effort
            return ""
    status = git("status", "--porcelain", "--", "experiment/v7")
    return {"commit": git("rev-parse", "HEAD").strip(),
            "v7_dirty_files": [line[3:] for line in status.splitlines() if len(line) > 3]}


class StepTracer:
    """Attach to the sims BEFORE prepare_step_cmd_buffers (it installs the
    timers and turns on step_trace_parity), then set ``orchestrator.on_frame_done
    = tracer.on_frame_done`` after bootstrap, call ``on_defrag(frame_n, warmup)``
    from the run's defrag hook (reads the voxel counts once, at the first
    boundary >= warmup, while the pipeline is drained), ``write(...)`` at the end
    and ``close()`` before the sims are destroyed (also on errors). Needs
    VK_KHR_calibrated_timestamps on every device, depth <= 2 and a loop that calls
    on_frame_done for every frame (not V7_PER_SIM_PIPELINE=1). VK_EXT_calibrated_timestamps
    serves where the KHR name is missing (CALIBRATED_TIMESTAMPS_EXTENSION_CHOICES)."""

    def __init__(self, sims, calibrate_ms: float = 500.0, detail: str = "phases"):
        if detail not in ("phases", "full"):
            raise ValueError(f"step trace detail {detail!r}: 'phases' or 'full'")
        self.sims = list(sims)
        self.calibrate_ms = float(calibrate_ms)
        self.detail = detail
        self.compute_timers, self.transfer_timers = [], []
        for index, sim in enumerate(self.sims):
            compute = StepTraceTimer(sim.ctx, f"s{index}",
                                     keep_labels=PHASE_TICKS if detail == "phases" else None)
            transfer = StepTraceTimer(sim.ctx, f"s{index}_transfer",
                                      queue_family_index=sim.ctx.transfer_queue_family_index)
            sim.bench, sim.bench_transfer = compute, transfer
            sim.step_trace_parity = True
            self.compute_timers.append(compute)
            self.transfer_timers.append(transfer)
        self.extension_suffixes = [calibrated_timestamps_suffix(sim.ctx) for sim in self.sims]
        self.domain = self._choose_domain()
        self.domain_name = TIME_DOMAIN_NAMES[self.domain]
        if self.domain == TIME_DOMAIN_CLOCK_MONOTONIC_RAW:
            from experiment.v7.utils import transport_v7

            def raw_clock_ns():
                return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            self.host_clock_ns, self.host_clock_name = raw_clock_ns, "clock_gettime_ns(CLOCK_MONOTONIC_RAW)"
            transport_v7.set_host_clock(raw_clock_ns)
        else:
            self.host_clock_ns, self.host_clock_name = time.perf_counter_ns, "time.perf_counter_ns"
        self._qpc_frequency = None
        if self.domain == TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER:
            frequency = ctypes.c_int64()
            ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(frequency))
            self._qpc_frequency = int(frequency.value)
        infos_khr = [VkCalibratedTimestampInfoKHR(timeDomain=VK_TIME_DOMAIN_DEVICE_KHR),
                     VkCalibratedTimestampInfoKHR(timeDomain=self.domain)]
        infos_ext = [VkCalibratedTimestampInfoKHR(sType=CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE_EXT,
                                                  timeDomain=VK_TIME_DOMAIN_DEVICE_KHR),
                     VkCalibratedTimestampInfoKHR(sType=CALIBRATED_TIMESTAMP_INFO_STRUCTURE_TYPE_EXT,
                                                  timeDomain=self.domain)]
        self._infos = [infos_ext if suffix == "EXT" else infos_khr for suffix in self.extension_suffixes]
        self._get_calibrated = []
        for sim, suffix in zip(self.sims, self.extension_suffixes):
            function = vkGetDeviceProcAddr(sim.ctx.device, f"vkGetCalibratedTimestamps{suffix}")
            if function is None:
                raise RuntimeError(f"vkGetCalibratedTimestamps{suffix} unavailable — enable "
                                   f"{' or '.join(CALIBRATED_TIMESTAMPS_EXTENSION_CHOICES)} on every device")
            self._get_calibrated.append(function)
        self.raw: dict[int, list] = {}           # frame -> [(compute raw, transfer raw) per sim]
        self.host_read: dict[int, int] = {}      # frame -> host time of its first read
        self.calibration: list[tuple] = []       # (sim, frame, device tick, host ns, max deviation ns)
        self.voxel_snapshot: Optional[dict] = None
        self.latest_frame = -1
        for index in range(len(self.sims)):
            self.calibrate(index, frame_n=-1, repeat=5, burst=8)
        # integer origin per device: ticks are subtracted from it before any float conversion
        self.origin_tick = [next(sample[2] for sample in self.calibration if sample[0] == index)
                            for index in range(len(self.sims))]
        self._closed = False
        self._stop = threading.Event()
        self._calibration_thread = threading.Thread(target=self._calibration_loop, name="step_trace_calibration",
                                                    daemon=True)
        self._calibration_thread.start()

    def _calibration_loop(self) -> None:
        while not self._stop.wait(self.calibrate_ms / 1000.0):
            for index in range(len(self.sims)):
                self.calibrate(index, frame_n=self.latest_frame)

    def _choose_domain(self) -> int:
        """The host domain every device can calibrate against (see section comment)."""
        available = None
        for sim, suffix in zip(self.sims, self.extension_suffixes):
            function = vkGetInstanceProcAddr(sim.ctx.instance,
                                             f"vkGetPhysicalDeviceCalibrateableTimeDomains{suffix}")
            domains = {int(value) for value in function(sim.ctx.physical_device)}
            available = domains if available is None else available & domains
        order = ((TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER,) if sys.platform == "win32"     # RAW: not slewed by NTP
                 else (TIME_DOMAIN_CLOCK_MONOTONIC_RAW, TIME_DOMAIN_CLOCK_MONOTONIC))
        for domain in order:
            if domain in available:
                return domain
        raise RuntimeError(f"no usable host time domain: devices offer {sorted(available)}")

    def _host_ns(self, value: int) -> int:
        if self._qpc_frequency is not None:
            return value * 1_000_000_000 // self._qpc_frequency
        return value

    def calibrate(self, sim_index: int, frame_n: int, repeat: int = 1, burst: int = 3) -> None:
        """``repeat`` samples, each the smallest-maxDeviation pair of a ``burst``. The driver
        reports maxDeviation in ns; the device value stays a raw integer tick."""
        sim = self.sims[sim_index]
        stamps = ffi.new("uint64_t[2]")          # per call: the helper thread calibrates too
        for _ in range(repeat):
            best = None
            for _ in range(burst):
                deviation = self._get_calibrated[sim_index](sim.ctx.device, 2, self._infos[sim_index], stamps)
                sample = (sim_index, frame_n, int(stamps[0]), self._host_ns(int(stamps[1])), float(int(deviation)))
                if best is None or sample[4] < best[4]:
                    best = sample
            self.calibration.append(best)

    def device_ns(self, sim_index: int, tick: int) -> float:
        """Device ns of a raw tick, relative to the device's origin tick."""
        return float(tick - self.origin_tick[sim_index]) * self.compute_timers[sim_index].ns_per_tick

    # ---------------------------------------------------------------- capture

    def on_frame_done(self, frame_n: int, sim_index: Optional[int] = None) -> None:
        """Every sim's frame ``frame_n`` (or sim ``sim_index``'s) is complete: copy both
        pools' parity block raw (parsed in write())."""
        parity = frame_n % 2
        entry = self.raw.setdefault(frame_n, [None] * len(self.sims))
        for index in (range(len(self.sims)) if sim_index is None else (sim_index,)):
            if entry[index] is None:
                entry[index] = (self.compute_timers[index].read_block_raw(parity),
                                self.transfer_timers[index].read_block_raw(parity))
        self.host_read.setdefault(frame_n, self.host_clock_ns())
        self.latest_frame = max(self.latest_frame, frame_n)

    def on_defrag(self, frame_n: int, warmup: int) -> None:
        """At the first drained defrag boundary >= warmup: own particles and n_B
        (own particles outside the force band = phase B's force sweep) per sim
        from the voxel counts (inside_particle_count, vid = 1 + x * NY * NZ + yz)."""
        if self.voxel_snapshot is not None or frame_n < warmup:
            return
        for index in range(len(self.sims)):          # drained: a cheap moment for a big burst
            self.calibrate(index, frame_n, repeat=3, burst=8)
        per_sim = []
        for sim in self.sims:
            grid = sim.case.grid
            face = grid.grid_dimension_y * grid.grid_dimension_z
            leading = sim.case.ghost_grid.leading_ghost_voxel_count // face
            trailing = sim.case.ghost_grid.trailing_ghost_voxel_count // face
            raw = sim.readback_buffers_batch(["inside_particle_count"])["inside_particle_count"]
            counts = np.frombuffer(raw, dtype=np.uint32)[1:1 + grid.grid_dimension_x * face]
            columns = counts.reshape(grid.grid_dimension_x, face).sum(axis=1).astype(np.int64)
            first, last = leading, grid.grid_dimension_x - 1 - trailing     # own columns (local x)
            correction_band, density_band, force_band = sim.band_widths

            def outside(band):
                return [x for x in range(first, last + 1)
                        if not ((leading > 0 and x < first + band) or (trailing > 0 and x > last - band))]
            per_sim.append({
                "own_columns": last - first + 1, "leading_peer": leading > 0, "trailing_peer": trailing > 0,
                "band_widths": [correction_band, density_band, force_band],
                "own_particles": int(columns[first:last + 1].sum()),
                "n_B": int(columns[outside(force_band)].sum()),
                "n_correction_interior": int(columns[outside(correction_band)].sum()),
                "n_density_deep_interior": int(columns[outside(density_band)].sum()),
                "own_column_particles": [int(value) for value in columns[first:last + 1]]})
        self.voxel_snapshot = {"frame": frame_n, "per_sim": per_sim}

    def close(self) -> None:
        """Stop the calibration thread and free the query pools (idempotent; call before the
        sims are destroyed, also when the run failed)."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._calibration_thread.is_alive():
            self._calibration_thread.join()
        from vulkan import vkDeviceWaitIdle
        for sim, compute, transfer in zip(self.sims, self.compute_timers, self.transfer_timers):
            try:
                vkDeviceWaitIdle(sim.ctx.device)
            except Exception:   # noqa: BLE001 — a lost device must not hide the original error
                pass
            compute.destroy()
            transfer.destroy()
            sim.bench = sim.bench_transfer = None

    # ----------------------------------------------------------------- output

    def _fits(self) -> list:
        fits = []
        for index in range(len(self.sims)):
            samples = [sample for sample in self.calibration if sample[0] == index]
            fit = fit_clock([self.device_ns(index, sample[2]) for sample in samples],
                            [sample[3] for sample in samples], [sample[4] for sample in samples])
            fit.update({"device_origin_tick": int(self.origin_tick[index]),
                        "ns_per_tick": float(self.compute_timers[index].ns_per_tick)})
            fits.append(fit)
        return fits

    def write(self, out_dir, orchestrator, meta: dict) -> dict:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._stop.set()
        self._calibration_thread.join()
        last_frame = max(self.raw) if self.raw else -1
        for index in range(len(self.sims)):
            self.calibrate(index, frame_n=last_frame, repeat=5, burst=8)
        fits = self._fits()

        def host(index, tick):
            return int(round(float(clock_to_host(fits[index], self.device_ns(index, tick)))))
        device_rows: dict[tuple[int, int], dict] = {}
        transfer_rows: dict[tuple[int, int], dict] = {}
        missing: dict[tuple[int, int], int] = {}
        for frame_n, entry in self.raw.items():
            for index, captured in enumerate(entry):
                if captured is None:
                    continue
                compute = self.compute_timers[index].parse_block(captured[0])
                transfer = self.transfer_timers[index].parse_block(captured[1])
                missing[(frame_n, index)] = (len(self.compute_timers[index].block_labels) - len(compute)
                                             + len(self.transfer_timers[index].block_labels) - len(transfer))
                row = {label: host(index, tick) for label, tick in compute.items()}
                for phase in ("a", "b", "c"):
                    ticks = [value for label, value in row.items() if label.startswith(phase + "_")]
                    if ticks:
                        row[phase + "_end"] = max(ticks)
                device_rows[(frame_n, index)] = row
                transfer_rows[(frame_n, index)] = {label: host(index, tick) for label, tick in transfer.items()}

        def device_complete(key) -> bool:
            # every tick this sim's cmds write was available (a missing last tick would
            # otherwise shorten the phase silently)
            return missing.get(key, 1) == 0 and all(
                phase in device_rows[key] for phase in ("a_start", "a_end", "b_start", "b_end", "c_start", "c_end"))
        with open(out / "steps_device.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "sim", "device", "parity", "host_read"] + list(STEP_DEVICE_TIMES)
                            + ["missing_ticks", "complete"])
            for (frame_n, index), row in sorted(device_rows.items()):
                writer.writerow([frame_n, index, meta.get("device_map", [None] * len(self.sims))[index],
                                 frame_n % 2, self.host_read.get(frame_n, "")]
                                + [row.get(key, "") for key in STEP_DEVICE_TIMES]
                                + [missing[(frame_n, index)], int(device_complete((frame_n, index)))])
        sim_index = {id(sim): index for index, sim in enumerate(self.sims)}
        links = []
        link_complete = 0
        link_rows = 0
        with open(out / "steps_link.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "link", "sender", "receiver", "sender_direction", "receiver_direction"]
                            + list(STEP_LINK_TIMES) + ["host_copy_bytes", "dma_bytes", "complete"])
            for worker in getattr(orchestrator, "workers", []):
                sender, receiver = sim_index[id(worker.source)], sim_index[id(worker.dest)]
                send_direction, receive_direction = worker.source_direction, worker.dest_direction
                links.append({"link": worker.label, "sender": sender, "receiver": receiver,
                              "sender_direction": send_direction, "receiver_direction": receive_direction,
                              "dma_bytes": int(worker.staging_bytes)})
                for frame_n in sorted(self.raw):
                    sender_row = device_rows.get((frame_n, sender), {})
                    receiver_row = device_rows.get((frame_n, receiver), {})
                    sender_transfer = transfer_rows.get((frame_n, sender), {})
                    receiver_transfer = transfer_rows.get((frame_n, receiver), {})
                    stamps = worker.timestamps.get(frame_n, {})
                    values = {"send_end": sender_row.get(f"a_ghost_{send_direction}_end"),
                              "readback_start": sender_transfer.get(f"t_rb_{send_direction}_start"),
                              "readback_copy_end": sender_transfer.get(f"t_rb_{send_direction}_copy_end"),
                              "readback_end": sender_transfer.get(f"t_rb_{send_direction}_end"),
                              "upload_start": receiver_transfer.get(f"t_up_{receive_direction}_start"),
                              "upload_end": receiver_transfer.get(f"t_up_{receive_direction}_end"),
                              "receiver_b_start": receiver_row.get("b_start"),
                              "receiver_b_end": receiver_row.get("b_end"),
                              "receiver_c_start": receiver_row.get("c_start")}
                    for column, key in _WORKER_KEYS:
                        values[column] = stamps.get(key)
                    copy_bytes = stamps.get("copy_bytes")
                    complete = (all(values[key] is not None for key in STEP_LINK_TIMES) and copy_bytes is not None
                                and missing.get((frame_n, sender), 1) == 0 and missing.get((frame_n, receiver), 1) == 0)
                    link_rows += 1
                    link_complete += int(complete)
                    writer.writerow([frame_n, worker.label, sender, receiver, send_direction, receive_direction]
                                    + ["" if values[key] is None else values[key] for key in STEP_LINK_TIMES]
                                    + ["" if copy_bytes is None else copy_bytes, worker.staging_bytes,
                                       int(complete)])
        device_complete_count = sum(1 for key in device_rows if device_complete(key))
        summary = {"steps": len(self.raw), "device_rows": len(device_rows),
                   "device_rows_complete": device_complete_count,
                   "link_rows": link_rows, "link_rows_complete": link_complete}
        run_meta = dict(meta)
        run_meta.update(_git_provenance())
        run_meta.update({
            "clock": {"time_domain": self.domain_name, "host_clock": self.host_clock_name,
                      "extensions": [f"VK_{suffix}_calibrated_timestamps" for suffix in self.extension_suffixes],
                      "qpc_frequency": self._qpc_frequency, "calibrate_ms": self.calibrate_ms,
                      "detail": self.detail, "fit_version": CLOCK_FIT_VERSION, "fits": fits},
            "slabs": self.voxel_snapshot,
            "links": links,
            "labels": {"compute": [timer.block_labels for timer in self.compute_timers],
                       "transfer": [timer.block_labels for timer in self.transfer_timers]},
            "records": summary,
            "switches_env": {key: value for key, value in sorted(os.environ.items()) if key.startswith("V7_")},
        })
        with open(out / "run_meta.json", "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, indent=1)
        with open(out / "calibration.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sim", "frame", "device_tick", "device_ns", "host_ns", "max_deviation_ns"])
            for sample in self.calibration:
                writer.writerow([sample[0], sample[1], sample[2], f"{self.device_ns(sample[0], sample[2]):.1f}",
                                 sample[3], f"{sample[4]:.0f}"])
        print(f"[step_trace] wrote {out}: {summary}; clock {self.domain_name}, map residual rms / max "
              + "; ".join(f"{fit['residual_rms_ns']:.0f} / {fit['residual_max_ns']:.0f}" for fit in fits)
              + " ns", flush=True)
        return summary
