"""
transport_cpu_staging.py — V1.0a cross-GPU sync 1 transport via CPU host
staging. Cross-vendor (NV+AMD) pair has no shared-memory path (Phase 2 probe
result), so each step's ghost data flows:

    sender device-local  →  sender host-visible staging (PCIe readback)
                            ↓ Python memcpy (numpy uint8 view)
    receiver host-visible staging  →  receiver device-local (PCIe upload)

Per direction-pair we pre-allocate one host-visible buffer on EACH device
(can't share VkDeviceMemory across VkDevice / vendor) and pre-record the
readback / upload cmd buffers (12 vkCmdCopyBuffer regions: 9 set-0 SoA + 2
set-1 voxel + 1 set-3 count). transfer() submits readback, memcpy, upload —
each waits a fence (3 fences per direction-pair per step).

Future optimizations (V2/V3):
  * Same-vendor (NV+NV 2×5090): switch to P2P vkCmdCopyBuffer between two
    devices' buffers via VK_KHR_external_memory + opaque handle import.
  * Pipeline overlap: use semaphores instead of fences, run forward + reverse
    direction-pair concurrently (different queues).
  * Async transfer queue: separate VkQueue family for staging copies, runs
    concurrently with compute on graphics queue.
"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

from experiment.v1.utils.simulator_v1 import SphSimulatorV1


class GhostTransportPair:
    """One-direction cross-GPU transport.

    Owns: 1 host-visible staging buffer on sender, 1 host-visible staging on
    receiver, and 2 pre-recorded cmd buffers (readback + upload).

    transfer() executes the 3-step pump (readback → host memcpy → upload).
    """

    def __init__(
        self,
        sim_send: SphSimulatorV1,
        send_direction: str,         # 'leading' or 'trailing'
        sim_recv: SphSimulatorV1,
        recv_direction: str,
    ):
        if send_direction not in ("leading", "trailing"):
            raise ValueError(f"bad send_direction: {send_direction}")
        if recv_direction not in ("leading", "trailing"):
            raise ValueError(f"bad recv_direction: {recv_direction}")
        self.sim_send = sim_send
        self.sim_recv = sim_recv
        self.label = f"{send_direction[0]}→{recv_direction[0]}"   # 't→l' or 'l→t'

        send_handles = sim_send.get_ghost_transport_handles().get(send_direction)
        recv_handles = sim_recv.get_ghost_transport_handles().get(recv_direction)
        if send_handles is None:
            raise ValueError(
                f"sim_send has no '{send_direction}' direction handles "
                f"(ghost pool size 0?)")
        if recv_handles is None:
            raise ValueError(
                f"sim_recv has no '{recv_direction}' direction handles "
                f"(ghost pool size 0?)")

        # Build the 12-segment pump list:
        #   (sender_buf, sender_off, recv_buf, recv_off, size, label)
        # Order matches simulator_v1's _SET0_GHOST_BUFFERS_AND_STRIDES + 2
        # set-1 entries + send_count→recv_count.
        segments: list[tuple] = []
        for i, ((sb, so, ssz), (rb, ro, rsz)) in enumerate(
                zip(send_handles["ghost_pid"], recv_handles["ghost_pid"])):
            if ssz != rsz:
                raise ValueError(
                    f"ghost_pid[{i}] size mismatch: send {ssz}, recv {rsz}")
            segments.append((sb, so, rb, ro, ssz, f"ghost_pid[{i}]"))
        for key in ("ghost_vid_count", "ghost_vid_index"):
            sb, so, ssz = send_handles[key]
            rb, ro, rsz = recv_handles[key]
            if ssz != rsz:
                raise ValueError(f"{key} size mismatch: send {ssz}, recv {rsz}")
            segments.append((sb, so, rb, ro, ssz, key))
        # send_count (sender's ghost_send_<dir>_count) → recv_count (receiver's
        # ghost_recv_<dir>_count). Both are 4-byte uints.
        sb, so, ssz = send_handles["send_count"]
        rb, ro, rsz = recv_handles["recv_count"]
        if ssz != 4 or rsz != 4:
            raise ValueError(f"count fields must be 4 B; got send {ssz} recv {rsz}")
        segments.append((sb, so, rb, ro, 4, "count"))
        self.segments = segments

        # Lay segments out in a contiguous staging blob.
        self.total_bytes = 0
        self.staging_offsets: list[int] = []
        for seg in segments:
            self.staging_offsets.append(self.total_bytes)
            self.total_bytes += seg[4]

        # Allocate host-visible staging on each device.
        usage = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        host_props = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        self.send_staging = sim_send._allocate_buffer(self.total_bytes, usage, host_props)
        self.recv_staging = sim_recv._allocate_buffer(self.total_bytes, usage, host_props)

        # Map both stagings for the pair's lifetime (we re-use them every step).
        self._send_mapped = vkMapMemory(
            sim_send.ctx.device, self.send_staging.memory, 0, self.total_bytes, 0)
        self._recv_mapped = vkMapMemory(
            sim_recv.ctx.device, self.recv_staging.memory, 0, self.total_bytes, 0)

        # numpy uint8 views over the mapped buffers for fast memcpy.
        # vkMapMemory already returns a buffer-protocol-compatible cffi buffer
        # (NOT a raw cdata pointer), so we feed it directly to np.frombuffer.
        self._send_view = np.frombuffer(
            self._send_mapped, dtype=np.uint8, count=self.total_bytes)
        self._recv_view = np.frombuffer(
            self._recv_mapped, dtype=np.uint8, count=self.total_bytes)

        # Pre-record readback + upload cmd buffers (SIMULTANEOUS_USE so they
        # can be re-submitted every step without re-recording).
        self._readback_cmd = self._record_readback_cmd()
        self._upload_cmd = self._record_upload_cmd()

    # ---- cmd buffer recording -------------------------------------------

    def _record_readback_cmd(self):
        """Sender device → sender staging. 12 vkCmdCopyBuffer.

        The pre_sync_cmd ends with a SHADER_WRITE → TRANSFER_READ barrier on
        the source buffers, so this cmd buffer can issue copies without an
        additional barrier (queue ordering covers it: pre_sync's fence
        signals before this cmd is submitted).
        """
        cmd = self.sim_send._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        for (sbuf, soff, _, _, size, _), st_off in zip(
                self.segments, self.staging_offsets):
            region = VkBufferCopy(srcOffset=soff, dstOffset=st_off, size=size)
            vkCmdCopyBuffer(cmd, sbuf, self.send_staging.handle, 1, [region])
        vkEndCommandBuffer(cmd)
        return cmd

    def _record_upload_cmd(self):
        """Receiver staging → receiver device. 12 vkCmdCopyBuffer.

        post_sync_cmd starts with a TRANSFER_WRITE → SHADER_READ barrier on
        the dest buffers, so this cmd doesn't need a trailing barrier.
        """
        cmd = self.sim_recv._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        for (_, _, rbuf, roff, size, _), st_off in zip(
                self.segments, self.staging_offsets):
            region = VkBufferCopy(srcOffset=st_off, dstOffset=roff, size=size)
            vkCmdCopyBuffer(cmd, self.recv_staging.handle, rbuf, 1, [region])
        vkEndCommandBuffer(cmd)
        return cmd

    # ---- main pump ------------------------------------------------------

    def transfer(self) -> None:
        """Combined 3-step pump (readback → memcpy → upload). Used by
        single-pair callers; for two-pair bidirectional transport, prefer
        the explicit readback/host_copy/upload phases via the
        CpuStagingMultiGpuTransport orchestrator (which interleaves both
        pairs' phases to avoid send/recv buffer aliasing corruption)."""
        self.readback()
        self.host_copy()
        self.upload()

    def readback(self) -> None:
        """Phase 1: sender device-local → sender host-visible staging."""
        self.sim_send.ctx.submit_and_wait(self._readback_cmd)

    def host_copy(self) -> None:
        """Phase 2: CPU memcpy sender staging → receiver staging."""
        self._recv_view[:] = self._send_view

    def upload(self) -> None:
        """Phase 3: receiver host-visible staging → receiver device-local."""
        self.sim_recv.ctx.submit_and_wait(self._upload_cmd)

    # ---- diagnostics ----------------------------------------------------

    def peek_send_staging(self) -> np.ndarray:
        """Return a copy of the current sender-side staging contents.
        For debug / verification; not used in the per-step hot path."""
        return self._send_view.copy()

    def peek_recv_staging(self) -> np.ndarray:
        return self._recv_view.copy()

    # ---- cleanup --------------------------------------------------------

    def destroy(self) -> None:
        # Ensure no pending GPU work touches send/recv staging before unmap.
        # Safe to call even if devices are idle.
        try:
            vkDeviceWaitIdle(self.sim_send.ctx.device)
            vkDeviceWaitIdle(self.sim_recv.ctx.device)
        except Exception:
            pass
        if hasattr(self, "_send_mapped") and self._send_mapped is not None:
            vkUnmapMemory(self.sim_send.ctx.device, self.send_staging.memory)
            self._send_mapped = None
        if hasattr(self, "_recv_mapped") and self._recv_mapped is not None:
            vkUnmapMemory(self.sim_recv.ctx.device, self.recv_staging.memory)
            self._recv_mapped = None
        if hasattr(self, "send_staging"):
            vkDestroyBuffer(self.sim_send.ctx.device, self.send_staging.handle, None)
            vkFreeMemory(self.sim_send.ctx.device, self.send_staging.memory, None)
        if hasattr(self, "recv_staging"):
            vkDestroyBuffer(self.sim_recv.ctx.device, self.recv_staging.handle, None)
            vkFreeMemory(self.sim_recv.ctx.device, self.recv_staging.memory, None)


class CpuStagingMultiGpuTransport:
    """V1.0a 2-GPU CPU-staged ghost transport.

    Constructs 2 GhostTransportPair instances for the bidirectional traffic:
        sim_a trailing → sim_b leading   (data flows sim_a → sim_b)
        sim_b leading  → sim_a trailing  (data flows sim_b → sim_a)

    transfer() runs both pumps sequentially. For 2-GPU on one host the
    forward and reverse pumps don't compete for distinct PCIe links per se
    (single PCIe root) but they DO serialize fence waits — V2 will overlap
    them via async transfer queues.
    """

    def __init__(self, sims: list[SphSimulatorV1]):
        if len(sims) != 2:
            raise NotImplementedError(
                f"V1.0a CPU-staged transport supports exactly 2 simulators; "
                f"got {len(sims)}")
        self.sims = sims
        sim_a, sim_b = sims

        # 2-GPU layout: sim_a (slot 0) has trailing peer; sim_b (slot 1) has
        # leading peer. Validate.
        a_dirs = set(sim_a.get_ghost_transport_handles().keys())
        b_dirs = set(sim_b.get_ghost_transport_handles().keys())
        if a_dirs != {"trailing"} or b_dirs != {"leading"}:
            raise ValueError(
                f"Expected sim_a active dirs={{'trailing'}}, got {a_dirs}; "
                f"sim_b active dirs={{'leading'}}, got {b_dirs}. "
                f"Pass sims in [leftmost, rightmost] order.")

        self.pair_a_to_b = GhostTransportPair(sim_a, "trailing", sim_b, "leading")
        self.pair_b_to_a = GhostTransportPair(sim_b, "leading",  sim_a, "trailing")
        bytes_per_step_per_dir = self.pair_a_to_b.total_bytes
        print(f"[CpuStaging] 2 pairs ready; "
              f"{bytes_per_step_per_dir / 1024:.1f} KB / direction / step "
              f"({bytes_per_step_per_dir * 2 / 1024 / 1024:.2f} MB total / step)")

    def transfer(self) -> None:
        """Run sync 1 transport.

        Buffer aliasing requires phasing: each GPU's ghost-pid + ghost-vid
        ranges are BOTH the SEND source (just-written by ghost_send) AND
        the RECV destination (about to be filled by peer's data). If we ran
        each direction's pump independently (read+upload), pair_a_to_b's
        upload would overwrite the source bytes that pair_b_to_a still needs
        to read for the reverse direction.

        Phasing:
          1. ALL readbacks first (snapshot both senders' data into their
             host stagings). Now device buffers can be safely overwritten.
          2. Host-to-host memcpy in BOTH directions.
          3. ALL uploads (write peer's data into receiver ghost ranges).
        """
        # Phase 1: snapshot both senders to their host stagings.
        self.pair_a_to_b.readback()
        self.pair_b_to_a.readback()
        # Phase 2: host-side memcpy.
        self.pair_a_to_b.host_copy()
        self.pair_b_to_a.host_copy()
        # Phase 3: write to both receivers' device buffers.
        self.pair_a_to_b.upload()
        self.pair_b_to_a.upload()

    def destroy(self) -> None:
        if hasattr(self, "pair_a_to_b"):
            self.pair_a_to_b.destroy()
        if hasattr(self, "pair_b_to_a"):
            self.pair_b_to_a.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()
