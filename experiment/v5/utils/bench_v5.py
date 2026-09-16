"""
bench_v5.py — Per-GPU GPU-side timestamp collector for V5 benchmarking.

VkQueryPool wrapper that the simulator threads through its pre-recorded
SIMULTANEOUS_USE phase A / B / C / defrag command buffers. Each frame the
timeline semaphore guarantees only one execution of any given cmd buffer
is in flight, so re-using fixed query slots is safe as long as the slot
pool is reset at the start of the first cmd that writes into it.

Design choices:
  - Single query pool per GPU, sized for the maximum number of ticks.
  - "tick" is a single VkCmdWriteTimestamp at BOTTOM_OF_PIPE; per-kernel
    duration is the diff between consecutive ticks. Half the slots of a
    naive begin/end pair scheme and arithmetically equivalent.
  - Pool reset is recorded by the *first* phase cmd's first action
    (phase_a_reset_and_start). Defrag has its own reset over its own slot
    range (so non-defrag frames don't leave defrag slots stale-then-read).
  - vkGetQueryPoolResults uses WITH_AVAILABILITY (no WAIT) so calls after
    a non-defrag frame correctly skip the unwritten defrag slots instead
    of hanging the host thread.

Caller contract:
  1. Construct BenchTimer(ctx, label) AFTER VulkanContextV5 is built.
  2. Attach to simulator: ``sim.bench = timer`` BEFORE prepare_step_cmd_buffers
     (so phase cmd recording sees a live bench and inserts ticks).
  3. After ``orchestrator.step()`` returns (i.e. wait_frame_done resolved
     for both sims; defrag, if any, has also waited), call ``read_frame()``
     to pull the durations.
  4. ``destroy()`` before VulkanContextV5.destroy().
"""
from __future__ import annotations

from typing import Optional

from vulkan import (
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    VK_QUERY_RESULT_64_BIT,
    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
    VK_QUERY_TYPE_TIMESTAMP,
    VkQueryPoolCreateInfo,
    vkCmdResetQueryPool,
    vkCmdWriteTimestamp,
    vkCreateQueryPool,
    vkDestroyQueryPool,
    vkGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceQueueFamilyProperties,
    vkGetQueryPoolResults,
)
from vulkan._vulkancache import ffi


# Max distinct tick labels per GPU. 15 is the steady-state count for the
# 4-direction case; 64 leaves headroom for instrumentation experiments and
# is cheap (64 * 8 = 512 B of query pool VRAM).
_MAX_TICKS = 96
# Parity-region layout (BenchTimer.enable_parity_regions): A/B labels in
# [0, 24), phase-C labels of even frames in [24, 40), of odd frames in
# [40, 56), defrag from 56. Phase A resets only [0, 24); each phase-C cmd
# resets its own region, so after a pipeline drain the pool still holds the
# PREVIOUS frame's c_force_end -> cross-frame c_to_a_gap (2026-09-15).
_PARITY_AB_HI = 24
_PARITY_REGION_BASE = {0: 24, 1: 40}
_PARITY_REGION_SIZE = 16
_PARITY_STEP_HI = 56
_ODD_SUFFIX = "~odd"


class BenchTimer:
    """GPU-timestamp collector for one sim's command stream.

    Slots are allocated lazily as labels are first seen (during cmd
    recording). Once recording is done the slot map is frozen; subsequent
    read_frame() calls reuse the same slots every frame.
    """

    def __init__(self, ctx, label: str, queue_family_index: Optional[int] = None):
        """``queue_family_index`` selects which queue family's cmd buffers
        will write into this pool (validity check only — the pool itself is
        queue-agnostic). Default = the compute family. Pass
        ``ctx.transfer_queue_family_index`` for a transfer-queue timer (M5a:
        the 5090's dedicated transfer family reports timestampValidBits=64).

        NOTE (2026-07-22 audit): a transfer-only queue may WRITE timestamps
        but may NOT reset a query pool (vkCmdResetQueryPool queue list
        excludes TRANSFER). A transfer-pool timer's reset must be recorded
        from a graphics/compute-queue cmd via ``record_external_reset``.

        Cross-pool diffs (compute tick minus transfer tick of one GPU) are
        empirically consistent on the NV driver but NOT spec-guaranteed
        without VK_KHR_calibrated_timestamps — treat *_sched_gap_us /
        *_to_c_gap_us as driver-specific observations, same-pool diffs as
        exact."""
        self.ctx = ctx
        self.label = label

        # Timestamp validity + ns conversion.
        properties = vkGetPhysicalDeviceProperties(ctx.physical_device)
        self.ns_per_tick: float = float(properties.limits.timestampPeriod)
        if self.ns_per_tick <= 0.0:
            raise RuntimeError(
                f"BenchTimer({label}): physical device reports "
                f"timestampPeriod={self.ns_per_tick}; timestamps unsupported.")

        if queue_family_index is None:
            queue_family_index = ctx.compute_queue_family_index
        queue_family_properties_list = vkGetPhysicalDeviceQueueFamilyProperties(
            ctx.physical_device)
        valid_bits = queue_family_properties_list[
            queue_family_index].timestampValidBits
        if valid_bits == 0:
            raise RuntimeError(
                f"BenchTimer({label}): queue family {queue_family_index} "
                f"has timestampValidBits=0; GPU timestamps not supported "
                f"on this queue.")
        self.valid_bits = valid_bits

        # Single query pool covering both step phases and defrag.
        self.pool = vkCreateQueryPool(
            ctx.device,
            VkQueryPoolCreateInfo(
                queryType=VK_QUERY_TYPE_TIMESTAMP,
                queryCount=_MAX_TICKS,
            ),
            None,
        )

        # label → slot_index. Insertion order = recording order.
        self.label_to_slot: dict[str, int] = {}
        # Slot ranges that need separate reset commands. Populated by
        # mark_phase_start_reset() callers; orchestrator runs them per
        # frame implicitly by replaying the pre-recorded cmd buffers.
        self._step_slot_lo = 0   # phase A/B/C labels live here
        self._step_slot_hi = 0
        self._defrag_slot_lo = 0
        self._defrag_slot_hi = 0
        self._defrag_recording = False
        # Parity regions (see _PARITY_* above). Off unless enabled before
        # the first recording.
        self.parity_regions = False
        self._active_region = None
        self._region_counts = {0: 0, 1: 0}
        self._ab_count = 0

    def enable_parity_regions(self) -> None:
        """Fixed slot layout so phase-C ticks of consecutive frames survive
        each other (cross-frame c_to_a_gap). Call before any recording."""
        if self.parity_regions:
            return
        if self.label_to_slot:
            raise RuntimeError(f"BenchTimer({self.label}): enable_parity_regions "
                               f"must precede the first tick()")
        self.parity_regions = True
        self._step_slot_hi = _PARITY_STEP_HI

    def begin_phase_c_region(self, cmd, parity: int) -> None:
        """First action of phase_c_cmd[parity]: reset THIS parity's region
        only. No-op (except bookkeeping) when parity regions are off."""
        if not self.parity_regions:
            return
        self._active_region = parity
        self._region_counts[parity] = 0
        vkCmdResetQueryPool(cmd, self.pool, _PARITY_REGION_BASE[parity],
                            _PARITY_REGION_SIZE)

    def end_phase_c_region(self) -> None:
        self._active_region = None

    # ----------------------------------------------------------------- recording

    def record_step_reset_and_start(self, cmd, start_label: str = "a_start") -> None:
        """First action of phase_a_cmd: reset all step slots and write the
        Phase A start tick. Defrag slots are NOT reset here (they have
        their own reset embedded in defrag_cmd)."""
        # Step slots cover everything except the defrag range, which is
        # appended *after* all step labels were seen on first recording.
        # On the very first call the step range may equal the entire pool;
        # subsequent defrag recording shrinks the step range.
        if self.parity_regions:
            # A/B slots only — the two phase-C regions are reset by their
            # own cmds so the previous frame's C ticks stay readable.
            vkCmdResetQueryPool(cmd, self.pool, 0, _PARITY_AB_HI)
        elif self._defrag_recording:
            # defrag has already been recorded; step range was fixed at that
            # point. Reset only step slots.
            vkCmdResetQueryPool(cmd, self.pool, 0, self._step_slot_hi)
        else:
            # No defrag yet; reset the whole pool so we never have stale
            # data in unallocated slots.
            vkCmdResetQueryPool(cmd, self.pool, 0, _MAX_TICKS)
        self.tick(cmd, start_label)

    def record_external_reset(self, cmd) -> None:
        """Reset ALL slots of this pool from an EXTERNALLY-owned cmd buffer
        on a graphics/compute queue (query pools are queue-agnostic objects;
        only the reset COMMAND has queue-type restrictions). Used for the
        transfer-pool timer: the simulator records this into phase_a_cmd,
        whose phase_a_done signal orders it before every same-frame transfer
        write, and whose frame_done(N-1) wait orders it after every prior-
        frame transfer write. No tick is written here — transfer labels are
        allocated lazily by tick() inside the transfer cmds themselves."""
        vkCmdResetQueryPool(cmd, self.pool, 0, _MAX_TICKS)

    def record_defrag_reset_and_start(self, cmd, start_label: str = "defrag_start") -> None:
        """First action of defrag_cmd: reset defrag slots and write the
        defrag start tick. Splits the slot space so that step recording
        from this point on is locked to [0, defrag_lo)."""
        if not self._defrag_recording:
            self._defrag_recording = True
            # Lock the step range at whatever was seen by now.
            if self.parity_regions:
                self._step_slot_hi = _PARITY_STEP_HI
            else:
                self._step_slot_hi = len(self.label_to_slot)
            self._defrag_slot_lo = self._step_slot_hi
        vkCmdResetQueryPool(
            cmd, self.pool, self._defrag_slot_lo,
            _MAX_TICKS - self._defrag_slot_lo)
        self.tick(cmd, start_label)

    def tick(self, cmd, label: str) -> None:
        """Record a timestamp at BOTTOM_OF_PIPE = "all prior work in this
        cmd buffer has finished". Per-kernel duration is computed by the
        runner as (tick[label_N+1] - tick[label_N])."""
        if self.parity_regions and self._active_region == 1:
            label = label + _ODD_SUFFIX
        if label not in self.label_to_slot:
            if self.parity_regions and self._active_region is not None:
                region = self._active_region
                if self._region_counts[region] >= _PARITY_REGION_SIZE:
                    raise RuntimeError(f"BenchTimer({self.label}): phase-C region full")
                slot = _PARITY_REGION_BASE[region] + self._region_counts[region]
                self._region_counts[region] += 1
            elif self.parity_regions and not self._defrag_recording:
                if self._ab_count >= _PARITY_AB_HI:
                    raise RuntimeError(f"BenchTimer({self.label}): A/B region full")
                slot = self._ab_count
                self._ab_count += 1
            else:
                slot = (self._defrag_slot_hi if (self.parity_regions and self._defrag_recording)
                        else len(self.label_to_slot))
                if self.parity_regions and slot < self._defrag_slot_lo:
                    slot = self._defrag_slot_lo
            if slot >= _MAX_TICKS:
                raise RuntimeError(
                    f"BenchTimer({self.label}): out of slots ({_MAX_TICKS}); "
                    f"labels so far: {list(self.label_to_slot)}")
            self.label_to_slot[label] = slot
            if not self._defrag_recording:
                if not self.parity_regions:
                    self._step_slot_hi = slot + 1
            else:
                self._defrag_slot_hi = slot + 1
        slot = self.label_to_slot[label]
        vkCmdWriteTimestamp(
            cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.pool, slot)

    # --------------------------------------------------------------- readback

    def read_frame(self, include_defrag: bool) -> dict[str, float]:
        """Pull all available timestamps and return {label: tick_count_ns}
        as wall-clock-ns offsets from an unspecified origin (only diffs
        between same-frame ticks are meaningful).

        ``include_defrag=False`` skips defrag slots even if they happen to
        be available, which avoids leaking last-defrag-frame's data into
        the current non-defrag frame's CSV row.
        """
        if not self.label_to_slot:
            return {}
        last_slot = (self._defrag_slot_hi if include_defrag and self._defrag_recording
                     else self._step_slot_hi)
        if last_slot == 0:
            return {}
        # Query only ALLOCATED slot ranges: a reset-but-never-written slot
        # makes vkGetQueryPoolResults report VK_NOT_READY for the whole
        # range (parity layout has such holes; a region whose parity has
        # not executed yet is simply skipped).
        if self.parity_regions:
            ranges = [(0, self._ab_count),
                      (_PARITY_REGION_BASE[0], self._region_counts[0]),
                      (_PARITY_REGION_BASE[1], self._region_counts[1])]
            if include_defrag and self._defrag_recording:
                ranges.append((self._defrag_slot_lo,
                               self._defrag_slot_hi - self._defrag_slot_lo))
        else:
            ranges = [(0, last_slot)]
        values: dict[int, float] = {}
        stride = 16
        for first, count in ranges:
            if count <= 0:
                continue
            # WITH_AVAILABILITY: each result is (uint64 value, uint64 available_flag).
            # Allocate as cffi array; python-vulkan needs a cdata pointer for pData.
            data = ffi.new(f"uint64_t[{2 * count}]")
            try:
                vkGetQueryPoolResults(
                    self.ctx.device, self.pool,
                    first, count, stride * count, data, stride,
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
                )
            except Exception:
                # VK_NOT_READY (some slot of the range unavailable): the
                # available ones are still written; fall through to the
                # per-slot availability flags below.
                pass
            for offset in range(count):
                if int(data[2 * offset + 1]) != 0:
                    values[first + offset] = float(int(data[2 * offset])) * self.ns_per_tick
        result: dict[str, float] = {}
        for label, slot in self.label_to_slot.items():
            if slot >= last_slot and not (include_defrag and self._defrag_recording):
                continue
            if slot in values:
                result[label] = values[slot]
        return result

    # --------------------------------------------------------------- teardown

    def destroy(self) -> None:
        if self.pool is not None:
            vkDestroyQueryPool(self.ctx.device, self.pool, None)
            self.pool = None


# ============================================================================
# Frame analyzer — turns a {label: ns} pair into named per-kernel durations.
# Pure-Python, no Vulkan dependency; runner uses it to format CSV / stderr.
# ============================================================================


def split_parity_ticks(ticks: dict[str, float], current_parity: int):
    """Parity-region pools: return (same-frame ticks with plain labels,
    previous-frame c_force_end or None). Non-C labels are kept as they are;
    C labels of ``current_parity`` are renamed to their plain form; the
    other parity's c_force_end is the previous frame's."""
    current: dict[str, float] = {}
    previous_c_end = None
    for label, value in ticks.items():
        is_odd = label.endswith(_ODD_SUFFIX)
        plain = label[:-len(_ODD_SUFFIX)] if is_odd else label
        if not plain.startswith("c_"):
            current[label] = value
            continue
        if int(is_odd) == current_parity:
            current[plain] = value
        elif plain == "c_force_end":
            previous_c_end = value
    return current, previous_c_end


def compute_durations(ticks: dict[str, float]) -> dict[str, float]:
    """Compute per-kernel durations (microseconds) from a single GPU's
    one-frame tick dict. Missing source ticks → key absent in output.

    Two modes auto-detected by tick keys:
      - SINGLE: ``step_start`` present (single-GPU combined cmd buffer);
        emits predict_us / update_voxel_us / correction_us / density_us /
        force_us / step_total_us / defrag_us.
      - DUAL: ``a_start`` present (dual-GPU 3-submit pattern); emits the
        per-phase + per-kernel keys documented below.

    SINGLE mode:
        predict_us       = predict_end    - step_start
        update_voxel_us  = voxel_end      - predict_end
        correction_us    = correction_end - voxel_end
        density_us       = density_end    - correction_end
        force_us         = force_end      - density_end
        step_total_us    = force_end      - step_start

    DUAL mode — Phase A:
        predict_us            = a_predict_end  - a_start
        update_voxel_us       = a_voxel_end    - a_predict_end
        ghost_send_leading_us = a_ghost_leading_end  - a_voxel_end
        ghost_send_trailing_us = a_ghost_trailing_end
                                 - (a_ghost_leading_end OR a_voxel_end)
        phase_a_us            = (last A tick) - a_start

    Phase B:
        correction_interior_us = b_correction_interior_end - b_start
        phase_b_us             = same
        a_to_b_gap_us          = b_start - (last A tick)
                                 (cross-submit GPU idle; usually ~0 in steady state)

    Phase C:
        install_leading_us    = c_install_leading_end - c_start
        install_trailing_us   = c_install_trailing_end
                                - (c_install_leading_end OR c_start)
        correction_boundary_us = c_correction_boundary_end
                                 - (last install OR c_start)
        density_us            = c_density_end - c_correction_boundary_end
        force_us              = c_force_end - c_density_end
        phase_c_us            = c_force_end - c_start
        b_to_c_gap_us         = c_start - b_correction_interior_end
                                ← KEY KPI: sync-hiding efficiency

    Defrag:
        defrag_us             = defrag_end - defrag_start
    """
    def diff_us(end_label: str, start_label: str) -> Optional[float]:
        if end_label in ticks and start_label in ticks:
            return (ticks[end_label] - ticks[start_label]) / 1000.0
        return None

    out: dict[str, float] = {}

    # --- SINGLE mode (combined cmd buffer) ---
    # Detected by the presence of step_start. Emits a flat per-kernel set
    # without phase A/B/C aggregation (single mode has no phases). Defrag
    # is handled by the shared tail block below.
    if "step_start" in ticks:
        if (v := diff_us("predict_end", "step_start")) is not None:
            out["predict_us"] = v
        if (v := diff_us("voxel_end", "predict_end")) is not None:
            out["update_voxel_us"] = v
        if (v := diff_us("correction_end", "voxel_end")) is not None:
            out["correction_us"] = v
        if (v := diff_us("density_end", "correction_end")) is not None:
            out["density_us"] = v
        if (v := diff_us("force_end", "density_end")) is not None:
            out["force_us"] = v
        if (v := diff_us("force_end", "step_start")) is not None:
            out["step_total_us"] = v
        if (v := diff_us("defrag_end", "defrag_start")) is not None:
            out["defrag_us"] = v
        return out

    # --- DUAL mode — Phase A ---
    if (v := diff_us("a_predict_end", "a_start")) is not None:
        out["predict_us"] = v
    if (v := diff_us("a_voxel_end", "a_predict_end")) is not None:
        out["update_voxel_us"] = v
    # Ghost sends are conditional; track last A tick for downstream gap.
    last_a_label = "a_start"
    if "a_predict_end" in ticks:
        last_a_label = "a_predict_end"
    if "a_voxel_end" in ticks:
        last_a_label = "a_voxel_end"
    if "a_ghost_leading_end" in ticks:
        out["ghost_send_leading_us"] = diff_us("a_ghost_leading_end", last_a_label)
        # Three-way split: (setup+dispatch) / readback DMA / host-coherence barrier.
        if "a_ghost_leading_dispatch_end" in ticks:
            out["ghost_send_leading_dispatch_us"] = diff_us(
                "a_ghost_leading_dispatch_end", last_a_label)
            out["ghost_send_leading_readback_us"] = diff_us(
                "a_ghost_leading_readback_end", "a_ghost_leading_dispatch_end")
            out["ghost_send_leading_host_barrier_us"] = diff_us(
                "a_ghost_leading_end", "a_ghost_leading_readback_end")
        last_a_label = "a_ghost_leading_end"
    if "a_ghost_trailing_end" in ticks:
        out["ghost_send_trailing_us"] = diff_us("a_ghost_trailing_end", last_a_label)
        if "a_ghost_trailing_dispatch_end" in ticks:
            out["ghost_send_trailing_dispatch_us"] = diff_us(
                "a_ghost_trailing_dispatch_end", last_a_label)
            out["ghost_send_trailing_readback_us"] = diff_us(
                "a_ghost_trailing_readback_end", "a_ghost_trailing_dispatch_end")
            out["ghost_send_trailing_host_barrier_us"] = diff_us(
                "a_ghost_trailing_end", "a_ghost_trailing_readback_end")
        last_a_label = "a_ghost_trailing_end"
    if (v := diff_us(last_a_label, "a_start")) is not None and last_a_label != "a_start":
        out["phase_a_us"] = v

    # --- Phase B + A→B gap ---
    if (v := diff_us("b_start", last_a_label)) is not None:
        out["a_to_b_gap_us"] = v
    if (v := diff_us("b_correction_interior_end", "b_start")) is not None:
        out["correction_interior_us"] = v
    # Path A+ P5: density_deep_interior added to Phase B. phase_b_us is the
    # total Phase B GPU time (= last Phase B tick - b_start).
    last_b_label = "b_start"
    if "b_correction_interior_end" in ticks:
        last_b_label = "b_correction_interior_end"
    if (v := diff_us("b_density_deep_interior_end", "b_correction_interior_end")) is not None:
        out["density_deep_interior_us"] = v
        last_b_label = "b_density_deep_interior_end"
    # V3.3 cascading force: force_deep_interior_scratch appended to Phase B.
    if (v := diff_us("b_force_deep_interior_end", "b_density_deep_interior_end")) is not None:
        out["force_deep_interior_us"] = v
        last_b_label = "b_force_deep_interior_end"
    if (v := diff_us(last_b_label, "b_start")) is not None and last_b_label != "b_start":
        out["phase_b_us"] = v

    # --- Phase C + B→C gap (the sync-hiding KPI) ---
    # b_to_c_gap = time GPU idled between Phase B end and Phase C start.
    # = c_start - (last Phase B tick). Last Phase B tick is density_deep_
    # interior_end if P5 dispatch is present, else correction_interior_end.
    if (v := diff_us("c_start", last_b_label)) is not None and last_b_label != "b_start":
        out["b_to_c_gap_us"] = v

    last_c_label = "c_start"
    if "c_install_leading_end" in ticks:
        out["install_leading_us"] = diff_us("c_install_leading_end", last_c_label)
        # Two-way split: upload DMA / (dispatch + barriers).
        if "c_install_leading_upload_end" in ticks:
            out["install_leading_upload_us"] = diff_us(
                "c_install_leading_upload_end", last_c_label)
            out["install_leading_dispatch_us"] = diff_us(
                "c_install_leading_end", "c_install_leading_upload_end")
        last_c_label = "c_install_leading_end"
    if "c_install_trailing_end" in ticks:
        out["install_trailing_us"] = diff_us("c_install_trailing_end", last_c_label)
        if "c_install_trailing_upload_end" in ticks:
            out["install_trailing_upload_us"] = diff_us(
                "c_install_trailing_upload_end", last_c_label)
            out["install_trailing_dispatch_us"] = diff_us(
                "c_install_trailing_end", "c_install_trailing_upload_end")
        last_c_label = "c_install_trailing_end"
    if (v := diff_us("c_correction_boundary_end", last_c_label)) is not None:
        out["correction_boundary_us"] = v
    if (v := diff_us("c_density_end", "c_correction_boundary_end")) is not None:
        out["density_us"] = v
    if (v := diff_us("c_force_end", "c_density_end")) is not None:
        out["force_us"] = v
    if (v := diff_us("c_force_end", "c_start")) is not None:
        out["phase_c_us"] = v

    # --- M5a: transfer-queue DMA segments (labels live in the transfer
    # pool; the runner merges both pools' tick dicts before calling).
    # Same-pool diffs (the *_dma_us / *_barrier_us) are exact. CROSS-pool
    # diffs (*_sched_gap_us / *_to_c_gap_us) assume both pools share one
    # device clock — empirically consistent on the NV driver but not
    # spec-guaranteed without calibrated timestamps (2026-07-22 audit). ---
    for direction in ("leading", "trailing"):
        if (v := diff_us(f"t_rb_{direction}_copy_end",
                         f"t_rb_{direction}_start")) is not None:
            out[f"readback_{direction}_dma_us"] = v
        if (v := diff_us(f"t_rb_{direction}_end",
                         f"t_rb_{direction}_copy_end")) is not None:
            out[f"readback_{direction}_barrier_us"] = v
        if (v := diff_us(f"t_up_{direction}_end",
                         f"t_up_{direction}_start")) is not None:
            out[f"upload_{direction}_dma_us"] = v
        # Cross-queue scheduling offsets (compute tick ↔ transfer tick):
        # how long after ghost_send's data was ready did the DMA actually
        # start, and how long before phase C did the upload land.
        if (v := diff_us(f"t_rb_{direction}_start",
                         f"a_ghost_{direction}_end")) is not None:
            out[f"readback_{direction}_sched_gap_us"] = v
        if (v := diff_us("c_start", f"t_up_{direction}_end")) is not None:
            out[f"upload_{direction}_to_c_gap_us"] = v

    # --- Defrag (only present on defrag-cycle frames) ---
    if (v := diff_us("defrag_end", "defrag_start")) is not None:
        out["defrag_us"] = v

    return out
