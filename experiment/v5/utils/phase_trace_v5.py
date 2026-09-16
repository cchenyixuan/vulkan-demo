"""
phase_trace_v5.py — per-frame, per-sim phase timestamps on ONE host clock.

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
"""
from __future__ import annotations

import csv
import pathlib
import time
from typing import Optional

from vulkan import (
    VK_TIME_DOMAIN_DEVICE_KHR,
    VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR,
    VkCalibratedTimestampInfoKHR,
    vkGetDeviceProcAddr,
)
from vulkan._vulkancache import ffi

from experiment.v5.utils.bench_v5 import split_parity_ticks

CALIBRATED_TIMESTAMPS_EXTENSION = "VK_KHR_calibrated_timestamps"

_A_END_LABELS = ("a_ghost_trailing_end", "a_ghost_leading_end", "a_voxel_end", "a_predict_end")
_B_END_LABELS = ("b_force_deep_interior_end", "b_density_deep_interior_end",
                 "b_correction_interior_end")
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
