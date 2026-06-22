"""
bench_v4.py — Per-GPU GPU-side timestamp collector for V4 benchmarking.

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
  1. Construct BenchTimer(ctx, label) AFTER VulkanContextV4 is built.
  2. Attach to simulator: ``sim.bench = timer`` BEFORE prepare_step_cmd_buffers
     (so phase cmd recording sees a live bench and inserts ticks).
  3. After ``orchestrator.step()`` returns (i.e. wait_frame_done resolved
     for both sims; defrag, if any, has also waited), call ``read_frame()``
     to pull the durations.
  4. ``destroy()`` before VulkanContextV4.destroy().
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
_MAX_TICKS = 64


class BenchTimer:
    """GPU-timestamp collector for one sim's command stream.

    Slots are allocated lazily as labels are first seen (during cmd
    recording). Once recording is done the slot map is frozen; subsequent
    read_frame() calls reuse the same slots every frame.
    """

    def __init__(self, ctx, label: str):
        self.ctx = ctx
        self.label = label

        # Timestamp validity + ns conversion.
        properties = vkGetPhysicalDeviceProperties(ctx.physical_device)
        self.ns_per_tick: float = float(properties.limits.timestampPeriod)
        if self.ns_per_tick <= 0.0:
            raise RuntimeError(
                f"BenchTimer({label}): physical device reports "
                f"timestampPeriod={self.ns_per_tick}; timestamps unsupported.")

        queue_family_properties_list = vkGetPhysicalDeviceQueueFamilyProperties(
            ctx.physical_device)
        valid_bits = queue_family_properties_list[
            ctx.compute_queue_family_index].timestampValidBits
        if valid_bits == 0:
            raise RuntimeError(
                f"BenchTimer({label}): compute queue family "
                f"{ctx.compute_queue_family_index} has timestampValidBits=0; "
                f"GPU timestamps not supported on this queue.")
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

    # ----------------------------------------------------------------- recording

    def record_step_reset_and_start(self, cmd, start_label: str = "a_start") -> None:
        """First action of phase_a_cmd: reset all step slots and write the
        Phase A start tick. Defrag slots are NOT reset here (they have
        their own reset embedded in defrag_cmd)."""
        # Step slots cover everything except the defrag range, which is
        # appended *after* all step labels were seen on first recording.
        # On the very first call the step range may equal the entire pool;
        # subsequent defrag recording shrinks the step range.
        if self._defrag_recording:
            # defrag has already been recorded; step range was fixed at that
            # point. Reset only step slots.
            vkCmdResetQueryPool(cmd, self.pool, 0, self._step_slot_hi)
        else:
            # No defrag yet; reset the whole pool so we never have stale
            # data in unallocated slots.
            vkCmdResetQueryPool(cmd, self.pool, 0, _MAX_TICKS)
        self.tick(cmd, start_label)

    def record_defrag_reset_and_start(self, cmd, start_label: str = "defrag_start") -> None:
        """First action of defrag_cmd: reset defrag slots and write the
        defrag start tick. Splits the slot space so that step recording
        from this point on is locked to [0, defrag_lo)."""
        if not self._defrag_recording:
            self._defrag_recording = True
            # Lock the step range at whatever was seen by now.
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
        if label not in self.label_to_slot:
            slot = len(self.label_to_slot)
            if slot >= _MAX_TICKS:
                raise RuntimeError(
                    f"BenchTimer({self.label}): out of slots ({_MAX_TICKS}); "
                    f"labels so far: {list(self.label_to_slot)}")
            self.label_to_slot[label] = slot
            if not self._defrag_recording:
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
        # WITH_AVAILABILITY: each result is (uint64 value, uint64 available_flag).
        # Allocate as cffi array; python-vulkan needs a cdata pointer for pData.
        stride = 16
        data_size = stride * last_slot
        data = ffi.new(f"uint64_t[{2 * last_slot}]")
        vkGetQueryPoolResults(
            self.ctx.device, self.pool,
            0, last_slot, data_size, data, stride,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
        )
        result: dict[str, float] = {}
        for label, slot in self.label_to_slot.items():
            if slot >= last_slot:
                continue
            value = int(data[2 * slot])
            availability = int(data[2 * slot + 1])
            if availability == 0:
                continue
            result[label] = float(value) * self.ns_per_tick
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

    # --- Defrag (only present on defrag-cycle frames) ---
    if (v := diff_us("defrag_end", "defrag_start")) is not None:
        out["defrag_us"] = v

    return out
