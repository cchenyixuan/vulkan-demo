"""
volume_constraint_v7.py - host side of case.yaml numerics.volume_constraint: the constraint on the fluid's total
volume in a closed, filled, rigid container (off by default; wall_boundary crossing only).

Why. In such a container the fluid's volume is fixed, so the volume the densities stand for,
    V = sum over the fluid particles of m_i / rho_i,
has to stay what it was at the start. Nothing in the scheme keeps it there: the walls exchange no density with
the fluid, and with a negative pressure the pressure term is written with p_j - p_i, which does not see the
level of the pressure. Any small systematic bias of the continuity equation therefore accumulates for ever.

What. V_target = V of the state the run starts from. Every update interval the host takes V_now and sets
    theta = (V_now - V_target) / V_target,        rate = theta / relaxation_time,
and until the next update the fused correction + density pass adds, for every fluid particle,
    d rho_i / dt += rho_i * rate
(shaders/wall_boundary.glsl, spec const 130; the rate is in the 16-byte buffer volume_constraint_state). With
the densities too low on the whole theta > 0 and the term raises them. It is one relative rate for every fluid
particle: it moves the level of the pressure and makes no pressure gradient. Cases with an inlet, an outlet or a
free surface have no fixed volume and must leave it off.

How the sum is taken. The simulator reads its own particles back (position, mass, density, material) when it
is drained anyway, at a defrag, every update_steps steps, and sums in float64 on the host. A chain of slabs
adds the slabs' sums: every slab reports to the one VolumeConstraintGroup of the process, and when the last
slab of an update has reported, the group fixes theta and the rate and every slab uploads it at once. (One
simulation per process: every simulator with the constraint enabled joins the same group.)

case.yaml (every key optional):
    numerics:
      volume_constraint:
        enabled: false            # default
        relaxation_time: 1.0      # tau, in simulated time
        update_interval: null     # in simulated time; default relaxation_time / 5 (the largest allowed)
        gpu_mean_volume: true     # default since report section 25 (false before): the measure below
        integral: true            # default since report section 25 (false before): the integral term below
        integral_time: null       # T_I in simulated time; default 4 x relaxation_time

integral (report section 24; on by default since report section 25, 2026-11-20). The proportional rate theta / tau
leaves a steady offset when the
scheme has a steady density sink (a falling mean density; 3-D cavity: theta stays at sink x tau). With the switch
    rate = theta / tau + I,        at every update  I <- I + theta * Delta / (tau * T_I)
(Delta = the update interval actually used, update_steps x timestep; I is updated first, so the rate of an update
takes that update's theta in both terms). The continuous model d theta/dt = -rate + sink gives
s^2 + s / tau + 1 / (tau T_I) = 0: T_I = 4 tau (the default) is critically damped, and theta -> 0 in steady
state with I -> the sink. I goes into the runners' checkpoints (_volume_constraint_integral) and a resumed run
continues with it; a checkpoint without it starts again from I = 0, with a warning. Every log record then also
has "integral".

gpu_mean_volume (report section 22; on by default since report section 25, 2026-11-20). The measure is the mean
volume per fluid particle,
    v_ref = the mean of m_i / rho_i over the fluid particles of the initial state (float64, on the host),
    theta = (the mean of m_i / rho_i now) / v_ref - 1,         rate = theta / relaxation_time,
which equals the total-volume theta in a closed domain whose particle count does not change, and does not
mistake a lost particle for a volume change. Each slab sums delta_i = (m_i / rho_i) / v_ref - 1 and its fluid
count on the GPU (shaders/volume_constraint_reduction.comp, one partial sum per workgroup, before the defrag of
an update) and reads back only those partial sums; the group adds them in float64 over the slabs. v_ref is
written to the GPU as float32 (v32, volume_constraint_state.z); the host corrects for its rounding:
    theta = (v32 / v_ref) (1 + sum delta / count) - 1.
v_ref, the rate and theta go into the runners' checkpoints (SphSimulatorV7.volume_constraint_checkpoint_entries)
and a resumed run takes them from there; a checkpoint without them recomputes v_ref from the case's initial
particles (as the total-volume target is taken) and starts with rate 0, with a warning.
The interval is rounded down to a whole number of defrag cadences (the updates happen at defrags); a cadence
longer than relaxation_time / 5 is refused.

Log. Every update appends one JSON line {"unix", "step", "time", "volume", "target_volume", "theta", "rate",
"fluid_particles", "slabs"} to the file named by the environment variable SPH_VOLUME_CONSTRAINT_LOG (the runners
of experiment/wall_crossing set it to <run directory>/volume_constraint.jsonl), and to the group's history.
"step" counts the steps of this process (a resumed run starts again at 0), "time" = step x timestep.
(The variable's name does not start with V7_ on purpose: experiment/validation/cavity_runner.py accepts only the
release set of V7_* switches in its environment, and this is a log path, not a switch.)

Nothing here imports Vulkan.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

LOG_ENVIRONMENT_VARIABLE = "SPH_VOLUME_CONSTRAINT_LOG"
INTERVAL_SHARE_OF_RELAXATION_TIME = 5.0         # update interval <= relaxation_time / this


@dataclass(frozen=True)
class VolumeConstraintSettings:
    """case.yaml numerics.volume_constraint (see the module docstring)."""
    enabled: bool = False
    relaxation_time: float = 1.0
    update_interval: Optional[float] = None
    gpu_mean_volume: bool = True              # default since report section 25 (False before)
    integral: bool = True                     # default since report section 25 (False before)
    integral_time: Optional[float] = None

    def integral_time_value(self) -> float:
        """T_I of the integral term (report section 24): the given one, or 4 x relaxation_time."""
        if self.integral_time is not None:
            return float(self.integral_time)
        return 4.0 * float(self.relaxation_time)

    def integral_increment_factor(self, timestep: float, defrag_cadence: int) -> Optional[float]:
        """Delta / (tau T_I) of the integral term's update (None without the switch)."""
        if not self.integral:
            return None
        interval = self.update_steps(timestep, defrag_cadence) * float(timestep)
        return interval / (float(self.relaxation_time) * self.integral_time_value())

    def update_interval_value(self) -> float:
        """The update interval in simulated time: the given one, or relaxation_time / 5."""
        if self.update_interval is not None:
            return float(self.update_interval)
        return float(self.relaxation_time) / INTERVAL_SHARE_OF_RELAXATION_TIME

    def update_steps(self, timestep: float, defrag_cadence: int) -> int:
        """The update interval in steps: the largest multiple of the defrag cadence not above the interval."""
        cadence = int(defrag_cadence)
        # (a relative margin: an interval that is a whole number of cadences must not lose one to rounding)
        cadences = int(self.update_interval_value() / (float(timestep) * cadence) * (1.0 + 1e-9))
        if cadences < 1:
            raise ValueError(
                f"numerics.volume_constraint: the update interval {self.update_interval_value():g} is shorter "
                f"than one defrag cadence ({cadence} steps = {cadence * float(timestep):g}); the updates happen "
                "at defrags, so raise relaxation_time / update_interval or lower numerics.defrag_cadence")
        return cadences * cadence


def parse_volume_constraint_settings(numerics_mapping: dict) -> VolumeConstraintSettings:
    """numerics.volume_constraint of case.yaml (optional mapping, see the module docstring)."""
    mapping = numerics_mapping.get("volume_constraint") or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"numerics.volume_constraint: expected a mapping, got {mapping!r}")
    unknown = sorted(set(mapping) - {"enabled", "relaxation_time", "update_interval", "gpu_mean_volume",
                                     "integral", "integral_time"})
    if unknown:
        raise ValueError(f"numerics.volume_constraint: unknown keys {unknown} (known: enabled, relaxation_time, "
                         "update_interval, gpu_mean_volume, integral, integral_time)")
    integral = mapping.get("integral", True)
    if not isinstance(integral, bool):
        raise ValueError(f"numerics.volume_constraint.integral: {integral!r}, expected true or false")
    integral_time = mapping.get("integral_time")
    if integral_time is not None:
        integral_time = float(integral_time)
        if not integral_time > 0.0:
            raise ValueError(f"numerics.volume_constraint.integral_time: {integral_time!r}, expected > 0")
    gpu_mean_volume = mapping.get("gpu_mean_volume", True)
    if not isinstance(gpu_mean_volume, bool):
        raise ValueError(f"numerics.volume_constraint.gpu_mean_volume: {gpu_mean_volume!r}, expected true or false")
    enabled = mapping.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"numerics.volume_constraint.enabled: {enabled!r}, expected true or false")
    relaxation_time = float(mapping.get("relaxation_time", 1.0))
    if not relaxation_time > 0.0:
        raise ValueError(f"numerics.volume_constraint.relaxation_time: {relaxation_time!r}, expected > 0")
    update_interval = mapping.get("update_interval")
    if update_interval is not None:
        update_interval = float(update_interval)
        largest = relaxation_time / INTERVAL_SHARE_OF_RELAXATION_TIME
        if not 0.0 < update_interval <= largest * (1.0 + 1e-9):
            raise ValueError(f"numerics.volume_constraint.update_interval: {update_interval!r}, expected > 0 and "
                             f"not above relaxation_time / {INTERVAL_SHARE_OF_RELAXATION_TIME:g} = {largest:g}")
    return VolumeConstraintSettings(enabled=enabled, relaxation_time=relaxation_time,
                                    update_interval=update_interval, gpu_mean_volume=gpu_mean_volume,
                                    integral=integral, integral_time=integral_time)


class VolumeConstraintGroup:
    """The slabs of one simulation: their target volumes add up, and so do the volumes of an update."""

    def __init__(self) -> None:
        self.members: list = []
        self.target_parts: dict = {}
        self.fluid_count_parts: dict = {}        # gpu_mean_volume: the initial fluid count per slab
        self.reference_override: Optional[float] = None   # gpu_mean_volume: v_ref of a resumed run's checkpoint
        self.pending: dict = {}             # step -> {id(member): (volume, fluid particle count)}
        self.theta = 0.0
        self.rate = 0.0
        self.integral = 0.0                 # report section 24: I of the integral term (0 without the switch)
        self.history: list = []

    def register(self, member, target_volume: float) -> None:
        if not self.members:                # a new simulation in this process starts from nothing
            self.pending.clear()
            self.history.clear()
            self.theta = self.rate = 0.0
            self.integral = 0.0
            self.fluid_count_parts.clear()
            self.reference_override = None
        self.members.append(member)
        self.target_parts[id(member)] = float(target_volume)

    def register_fluid_count(self, member, fluid_count: int) -> None:
        """gpu_mean_volume: the slab's initial fluid count (its target volume is registered with register)."""
        self.fluid_count_parts[id(member)] = int(fluid_count)

    def reference_mean_volume(self) -> float:
        """gpu_mean_volume: v_ref = the initial fluid volume / the initial fluid count over every registered slab, or
        the value a resumed run took from its checkpoint."""
        if self.reference_override is not None:
            return float(self.reference_override)
        count = sum(self.fluid_count_parts.values())
        return self.target_volume() / count if count else 0.0

    def restore(self, reference_mean_volume: float, rate: float, theta: float) -> None:
        """gpu_mean_volume, a resumed run: v_ref, the rate and theta of its checkpoint."""
        self.reference_override = float(reference_mean_volume)
        self.rate = float(rate)
        self.theta = float(theta)

    def restore_integral(self, integral: float) -> None:
        """integral (report section 24), a resumed run: I of its checkpoint."""
        self.integral = float(integral)

    def _set_rate(self, relaxation_time: float, integral_increment_factor: Optional[float], record: dict) -> None:
        """rate from self.theta: theta / tau, or with the integral term (factor = Delta / (tau T_I)) I is advanced
        first and rate = theta / tau + I; the record gets "integral" then."""
        if integral_increment_factor is None:
            self.rate = self.theta / float(relaxation_time)
            return
        self.integral += self.theta * float(integral_increment_factor)
        self.rate = self.theta / float(relaxation_time) + self.integral
        record["integral"] = self.integral

    def report_mean(self, member, step: int, deviation_sum: float, fluid_particles: int, reference_float32: float,
                    relaxation_time: float, timestep: float,
                    integral_increment_factor: Optional[float] = None) -> Optional[dict]:
        """gpu_mean_volume: one slab's GPU partial sums at an update (sum of delta_i = (m_i / rho_i) / v32 - 1 and the
        fluid count). Returns the update's record once every slab has reported for this step (theta and rate are
        then set), None before."""
        reports = self.pending.setdefault(int(step), {})
        reports[id(member)] = (float(deviation_sum), int(fluid_particles))
        if len(reports) < len(self.members):
            return None
        del self.pending[int(step)]
        for older in [key for key in self.pending if key < int(step)]:
            del self.pending[older]
        deviation = sum(value for value, _ in reports.values())
        count = sum(number for _, number in reports.values())
        reference = self.reference_mean_volume()
        ratio = (float(reference_float32) / reference) * (1.0 + deviation / count)
        self.theta = ratio - 1.0
        extra: dict = {}
        self._set_rate(relaxation_time, integral_increment_factor, extra)
        record = {"unix": time.time(), "step": int(step), "time": int(step) * float(timestep),
                  "method": "gpu_mean_volume", "mean_volume": ratio * reference, "reference_mean_volume": reference,
                  "reference_float32": float(reference_float32), "deviation_sum": deviation,
                  "theta": self.theta, "rate": self.rate, "fluid_particles": int(count), "slabs": len(reports),
                  **extra}
        self.history.append(record)
        path = os.environ.get(LOG_ENVIRONMENT_VARIABLE)
        if path:
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    def unregister(self, member) -> None:
        if member in self.members:
            self.members.remove(member)
            self.target_parts.pop(id(member), None)
            self.fluid_count_parts.pop(id(member), None)

    def target_volume(self) -> float:
        return float(sum(self.target_parts.values()))

    def report(self, member, step: int, volume: float, fluid_particles: int, relaxation_time: float,
               timestep: float, integral_increment_factor: Optional[float] = None) -> Optional[dict]:
        """One slab's fluid volume at an update. Returns the update's record once every slab has reported for
        this step (theta and rate are then set), None before."""
        reports = self.pending.setdefault(int(step), {})
        reports[id(member)] = (float(volume), int(fluid_particles))
        if len(reports) < len(self.members):
            return None
        del self.pending[int(step)]
        for older in [key for key in self.pending if key < int(step)]:
            del self.pending[older]         # an update a slab never completed
        volume_now = sum(value for value, _ in reports.values())
        target = self.target_volume()
        self.theta = (volume_now - target) / target
        extra: dict = {}
        self._set_rate(relaxation_time, integral_increment_factor, extra)
        record = {"unix": time.time(), "step": int(step), "time": int(step) * float(timestep),
                  "volume": volume_now, "target_volume": target, "theta": self.theta, "rate": self.rate,
                  "fluid_particles": int(sum(count for _, count in reports.values())), "slabs": len(reports),
                  **extra}
        self.history.append(record)
        path = os.environ.get(LOG_ENVIRONMENT_VARIABLE)
        if path:
            with open(path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
        return record


PROCESS_GROUP = VolumeConstraintGroup()
