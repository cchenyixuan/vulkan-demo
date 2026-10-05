"""
shared_host_v5.py — two-hop cross-GPU ghost transport on top of the V5 solver.

V5 moves ghost bytes in three hops (experiment/v5/utils/transport_v5.py):

    sender VRAM -[transfer Q DMA]-> sender_staging -[CPU worker memcpy]->
        receiver_staging -[transfer Q DMA]-> receiver VRAM

The middle hop exists only because each device owns its own host-visible
allocation. Here both devices import the SAME page-aligned host allocation
through VK_EXT_external_memory_host, so the sender's readback DMA writes the
bytes the receiver's upload DMA reads:

    sender VRAM -[transfer Q DMA]-> shared host block -[transfer Q DMA]->
        receiver VRAM

Everything else is V5 as deployed: transfer queues, timeline semaphores, the
per-direction sync scheme, the phase A/B/C command recording and every shader.
Nothing under experiment/v5/ is modified — this module subclasses and
overrides.

Section 1 is resource management only: the context wrapper that enables the
extension, the host allocation, and the per-device import (also used by
probe_shared_host.py). Section 2 is the transport itself: the link pool, the
simulator / worker / orchestrator subclasses.
"""

from __future__ import annotations

import ctypes
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from vulkan import *  # noqa: F401, F403

from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
from experiment.v5.utils.simulator_v5 import SphSimulatorV5, _Buffer
from experiment.v5.utils.transport_v5 import _STOP_SENTINEL, GhostMigrationWorker
from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5


# ============================================================================
# Constants
# ============================================================================

HOST_POINTER_HANDLE_TYPE = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT

# VK_KHR_external_memory is core since Vulkan 1.1 (V5 targets 1.3), but it is
# the declared dependency of VK_EXT_external_memory_host and both 5090
# drivers still advertise it, so it is enabled explicitly.
TWO_HOP_DEVICE_EXTENSIONS: tuple[str, ...] = (
    "VK_KHR_external_memory",
    "VK_EXT_external_memory_host",
)

# VirtualAlloc hands out address space in 64 KiB units. Rounding every shared
# block up to this granularity costs nothing and keeps the allocation a
# multiple of any minImportedHostPointerAlignment a driver reports (4 KiB on
# both vendors measured so far).
HOST_BLOCK_GRANULARITY_BYTES = 65536

_MEMORY_PROPERTY_NAMES = (
    (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,  "DEVICE_LOCAL"),
    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,  "HOST_VISIBLE"),
    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"),
    (VK_MEMORY_PROPERTY_HOST_CACHED_BIT,   "HOST_CACHED"),
)


def describe_memory_property_flags(flags: int) -> str:
    names = [name for bit, name in _MEMORY_PROPERTY_NAMES if flags & bit]
    return "|".join(names) if names else "(none)"


# ============================================================================
# Context: V5 context + the host-pointer import extension
# ============================================================================

class VulkanContextTwoHop(VulkanContextV5):
    """VulkanContextV5 with VK_EXT_external_memory_host enabled on the
    device. Queue selection, features and command pools are V5's own
    (VulkanContextV5.create builds the instance of whatever class it is
    called on)."""

    @classmethod
    def create(
        cls,
        application_name: str = "sph_two_hop",
        enable_validation: bool = True,
        extra_instance_extensions=None,
        extra_device_extensions=None,
        device_index=None,
    ) -> "VulkanContextTwoHop":
        device_extensions = list(extra_device_extensions or [])
        for extension_name in TWO_HOP_DEVICE_EXTENSIONS:
            if extension_name not in device_extensions:
                device_extensions.append(extension_name)
        return super().create(
            application_name=application_name,
            enable_validation=enable_validation,
            extra_instance_extensions=extra_instance_extensions,
            extra_device_extensions=device_extensions,
            device_index=device_index,
        )


def query_min_imported_host_pointer_alignment(context: VulkanContextV5) -> int:
    """VkPhysicalDeviceExternalMemoryHostPropertiesEXT
    .minImportedHostPointerAlignment of the context's physical device."""
    host_properties = VkPhysicalDeviceExternalMemoryHostPropertiesEXT()
    properties_2 = VkPhysicalDeviceProperties2(pNext=host_properties)
    # Core since Vulkan 1.1 — python-vulkan exposes it directly (its
    # vkGetInstanceProcAddr only resolves extension entry points).
    vkGetPhysicalDeviceProperties2(context.physical_device, properties_2)
    return int(host_properties.minImportedHostPointerAlignment)


def query_host_pointer_memory_type_bits(context: VulkanContextV5,
                                        host_address: int) -> int:
    """vkGetMemoryHostPointerPropertiesEXT: bitmask of the memory types this
    device accepts for importing ``host_address`` as a HOST_ALLOCATION."""
    query_function = vkGetDeviceProcAddr(
        context.device, "vkGetMemoryHostPointerPropertiesEXT")
    if query_function is None:
        raise RuntimeError(
            f"vkGetMemoryHostPointerPropertiesEXT not found on "
            f"'{context.device_name}' — VK_EXT_external_memory_host is not "
            f"enabled (use VulkanContextTwoHop)")
    properties = VkMemoryHostPointerPropertiesEXT()
    # python-vulkan takes a plain int for the void* argument.
    query_function(context.device, HOST_POINTER_HANDLE_TYPE, host_address,
                   properties)
    return int(properties.memoryTypeBits)


# ============================================================================
# Host allocation
# ============================================================================

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_READWRITE = 0x04


def _kernel32():
    if sys.platform != "win32":
        raise NotImplementedError(
            "shared host blocks are allocated with VirtualAlloc; only the "
            "Windows rig is supported by this experiment")
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.VirtualAlloc.argtypes = [
        wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    kernel32.VirtualAlloc.restype = wintypes.LPVOID
    kernel32.VirtualFree.argtypes = [
        wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    kernel32.VirtualFree.restype = wintypes.BOOL
    return kernel32


@dataclass
class HostBlock:
    """One page-aligned host allocation, imported by one or more devices."""
    address: int
    aligned_size: int
    # numpy.uint8 view over the WHOLE aligned allocation. Callers that hand a
    # view to the V5 ghost worker must slice it to the logical size (the
    # worker reads the frame stamp at the view's last 4 bytes).
    view: np.ndarray = field(repr=False)


def round_up_to_alignment(size: int, alignment: int) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def allocate_host_block(logical_size: int, alignment: int) -> HostBlock:
    if logical_size <= 0:
        raise ValueError(f"logical_size must be positive; got {logical_size}")
    aligned_size = round_up_to_alignment(
        logical_size, max(alignment, HOST_BLOCK_GRANULARITY_BYTES))
    address = _kernel32().VirtualAlloc(
        None, aligned_size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
    if not address:
        raise OSError(f"VirtualAlloc({aligned_size}) returned NULL")
    address = int(address)
    if address % alignment != 0:
        _kernel32().VirtualFree(ctypes.c_void_p(address), 0, _MEM_RELEASE)
        raise OSError(
            f"VirtualAlloc returned 0x{address:x}, not aligned to {alignment}")
    view = np.frombuffer(
        (ctypes.c_uint8 * aligned_size).from_address(address),
        dtype=np.uint8, count=aligned_size)
    # Committed pages are not backed until first touch; fault them in now so
    # the drivers pin real pages at import time.
    view[:] = 0
    return HostBlock(address=address, aligned_size=aligned_size, view=view)


def free_host_block(block: HostBlock) -> None:
    """Release the host pages. Every device's import of the block must be
    freed (vkFreeMemory) before this."""
    if block.address:
        _kernel32().VirtualFree(ctypes.c_void_p(block.address), 0, _MEM_RELEASE)
        block.address = 0


# ============================================================================
# Per-device import
# ============================================================================

@dataclass
class ImportedHostBuffer:
    """One device's VkBuffer + VkDeviceMemory aliasing a HostBlock."""
    handle: object
    memory: object
    memory_type_index: int
    memory_type_flags: int
    host_pointer_type_bits: int     # what the driver accepts for this pointer
    buffer_type_bits: int           # what the VkBuffer accepts


def _select_import_memory_type(context: VulkanContextV5, type_bits: int) -> int:
    """Same preference V5 uses for sender_staging (HOST_VISIBLE +
    HOST_COHERENT required, HOST_CACHED preferred). The memory is never
    vkMapMemory'd — the host already owns the pointer — so a driver that
    offers no HOST_VISIBLE import type still gets whatever it does offer."""
    try:
        return context.find_memory_type(
            type_bits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
    except RuntimeError:
        return context.find_memory_type(type_bits, 0)


def import_host_block(context: VulkanContextV5, block: HostBlock,
                      usage: int) -> ImportedHostBuffer:
    """Create a VkBuffer on ``context``'s device backed by ``block``.

    SHARING_MODE_EXCLUSIVE, like V5's own staging buffers
    (SphSimulatorV5._allocate_staging_buffers)."""
    device = context.device
    buffer_handle = vkCreateBuffer(device, VkBufferCreateInfo(
        pNext=VkExternalMemoryBufferCreateInfo(
            handleTypes=HOST_POINTER_HANDLE_TYPE),
        size=block.aligned_size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    ), None)
    try:
        requirements = vkGetBufferMemoryRequirements(device, buffer_handle)
        if requirements.size > block.aligned_size:
            raise RuntimeError(
                f"'{context.device_name}' needs {requirements.size} B for a "
                f"{block.aligned_size} B buffer — larger than the host block")
        host_pointer_type_bits = query_host_pointer_memory_type_bits(
            context, block.address)
        usable_type_bits = requirements.memoryTypeBits & host_pointer_type_bits
        if usable_type_bits == 0:
            raise RuntimeError(
                f"no memory type on '{context.device_name}' satisfies both "
                f"the buffer (0x{requirements.memoryTypeBits:x}) and the "
                f"host-pointer import (0x{host_pointer_type_bits:x})")
        memory_type_index = _select_import_memory_type(context, usable_type_bits)
        memory = vkAllocateMemory(device, VkMemoryAllocateInfo(
            pNext=VkImportMemoryHostPointerInfoEXT(
                handleType=HOST_POINTER_HANDLE_TYPE,
                pHostPointer=block.address),
            allocationSize=block.aligned_size,
            memoryTypeIndex=memory_type_index,
        ), None)
        try:
            vkBindBufferMemory(device, buffer_handle, memory, 0)
        except BaseException:
            vkFreeMemory(device, memory, None)
            raise
    except BaseException:
        vkDestroyBuffer(device, buffer_handle, None)
        raise
    return ImportedHostBuffer(
        handle=buffer_handle,
        memory=memory,
        memory_type_index=memory_type_index,
        memory_type_flags=context.memory_type_flags(memory_type_index),
        host_pointer_type_bits=host_pointer_type_bits,
        buffer_type_bits=int(requirements.memoryTypeBits),
    )


def destroy_imported_host_buffer(context: VulkanContextV5,
                                 imported: ImportedHostBuffer) -> None:
    vkDestroyBuffer(context.device, imported.handle, None)
    vkFreeMemory(context.device, imported.memory, None)


# ============================================================================
# Section 2: shared links, simulator, relay worker, orchestrator
# ============================================================================

REQUIRED_SYNC_SCHEME = "per-direction"

# Flow of a directed link between slab i and slab i+1.
FLOW_RIGHTWARD = "rightward"    # slab i   -> slab i+1
FLOW_LEFTWARD = "leftward"      # slab i+1 -> slab i


def shared_link_key(slab_index: int, role: str, direction: str) -> tuple[int, str]:
    """(left slab index, flow) of the shared block behind
    ``<role>_staging_<direction>`` of slab ``slab_index``.

    The sender's trailing staging of slab i and the receiver's leading
    staging of slab i+1 resolve to the same key — that is the whole point."""
    if role not in ("sender", "receiver"):
        raise ValueError(f"unknown role {role!r}")
    if direction == "trailing":
        return (slab_index,
                FLOW_RIGHTWARD if role == "sender" else FLOW_LEFTWARD)
    if direction == "leading":
        return (slab_index - 1,
                FLOW_LEFTWARD if role == "sender" else FLOW_RIGHTWARD)
    raise ValueError(f"unknown direction {direction!r}")


class SharedHostLinkPool:
    """Owns the shared host blocks of one chain: one block per directed
    link, allocated by whichever endpoint simulator asks first and handed
    unchanged to the other endpoint.

    Lifetime: create after the contexts, destroy after every simulator
    (each simulator's destroy() frees its own import of the pages)."""

    def __init__(self, contexts: Sequence[VulkanContextV5]) -> None:
        self.alignment = max(
            query_min_imported_host_pointer_alignment(context)
            for context in contexts)
        self._blocks: dict[tuple[int, str], HostBlock] = {}
        self._logical_sizes: dict[tuple[int, str], int] = {}
        self._import_counts: dict[tuple[int, str], int] = {}

    def acquire(self, key: tuple[int, str], logical_size: int) -> HostBlock:
        block = self._blocks.get(key)
        if block is None:
            block = allocate_host_block(logical_size, self.alignment)
            self._blocks[key] = block
            self._logical_sizes[key] = logical_size
            self._import_counts[key] = 0
        elif self._logical_sizes[key] != logical_size:
            raise ValueError(
                f"shared link {key}: endpoints disagree on the transport "
                f"size ({self._logical_sizes[key]} vs {logical_size} B) — "
                f"partition / ghost pool misconfigured")
        self._import_counts[key] += 1
        if self._import_counts[key] > 2:
            raise RuntimeError(
                f"shared link {key} acquired {self._import_counts[key]} "
                f"times; a directed link has exactly two endpoints")
        return block

    def total_bytes(self) -> int:
        return sum(block.aligned_size for block in self._blocks.values())

    def destroy(self) -> None:
        for block in self._blocks.values():
            free_host_block(block)
        self._blocks = {}


class SphSimulatorTwoHop(SphSimulatorV5):
    """SphSimulatorV5 whose host stagings are imports of shared host blocks.

    Only _allocate_staging_buffers is overridden. The transport segments,
    the readback / upload command recording, bootstrap, the sync scheme and
    every kernel are inherited: they address ``staging_buffers[...]`` by
    name and never learn that sender_staging_<direction> here and
    receiver_staging_<opposite direction> on the peer are the same bytes."""

    def __init__(self, ctx: VulkanContextV5, case, *,
                 link_pool: SharedHostLinkPool, slab_index: int,
                 sync_scheme: str = REQUIRED_SYNC_SCHEME) -> None:
        if sync_scheme != REQUIRED_SYNC_SCHEME:
            # The relay worker gates the sender's next readback on the
            # receiver's upload_done; only the per-direction scheme exposes
            # that wait (dest_upload_guard_ops is empty in the aggregated one).
            raise ValueError(
                f"two-hop transport requires sync_scheme="
                f"{REQUIRED_SYNC_SCHEME!r}; got {sync_scheme!r}")
        # Set BEFORE super().__init__: it calls _allocate_staging_buffers.
        self._link_pool = link_pool
        self._slab_index = slab_index
        self.shared_blocks: dict[str, HostBlock] = {}
        super().__init__(ctx, case, sync_scheme=sync_scheme)

    def _allocate_staging_buffers(self) -> dict[str, _Buffer]:
        case = self.case
        usage = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)

        self._transport_segments = {}
        self._transport_total_bytes = {}
        self._recv_count_offsets = {}
        stagings: dict[str, _Buffer] = {}
        import_descriptions = []
        for direction_name, peer_attr in (
            ("leading",  "has_leading_peer"),
            ("trailing", "has_trailing_peer"),
        ):
            if not getattr(case.transport, peer_attr):
                continue
            segments, total = self._compute_transport_segments(direction_name)
            if total == 0:
                continue
            self._transport_segments[direction_name] = segments
            self._transport_total_bytes[direction_name] = total

            for role in ("sender", "receiver"):
                name = f"{role}_staging_{direction_name}"
                key = shared_link_key(self._slab_index, role, direction_name)
                block = self._link_pool.acquire(key, total)
                imported = import_host_block(self.ctx, block, usage)
                self.shared_blocks[name] = block
                # mapped=None: destroy() then skips vkUnmapMemory (this
                # memory is never vkMapMemory'd — the host owns the pointer)
                # and only releases the import. The view is cut to exactly
                # ``total`` bytes: the worker reads the frame stamp at the
                # view's last 4 bytes.
                stagings[name] = _Buffer(
                    handle=imported.handle, memory=imported.memory,
                    size=total, mapped=None,
                    mapped_view=block.view[:total])
                import_descriptions.append(
                    f"  {name}: link {key[0]}<->{key[0] + 1} {key[1]}, "
                    f"{total / 1024:.1f} KB at 0x{block.address:x}, "
                    f"type[{imported.memory_type_index}]="
                    f"{describe_memory_property_flags(imported.memory_type_flags)}")

        if stagings:
            print(f"[SimTwoHop] slab {self._slab_index}: {len(stagings)} "
                  f"stagings imported from shared host blocks (no private "
                  f"host staging)")
            for line in import_descriptions:
                print(line)
        else:
            print("[SimTwoHop] host staging buffers: 0 (no peer)")
        return stagings


class GhostRelayWorker(GhostMigrationWorker):
    """One pathway's worker with the byte copy removed.

    Same three waits, host stamp check and worker_done signal as
    GhostMigrationWorker._run. What changes is WHEN the source may reuse the
    block:

        three-hop: consumed(n) right after the memcpy — the sender staging
                   is private and has been read out.
        two-hop:   consumed(n) only after the DEST finished upload(n) — the
                   block the source would overwrite is the one the dest's
                   upload DMA is reading.

    So readback(n+1) on the source is now coupled to upload(n) on the dest,
    through this host thread."""

    def __init__(self, source_sim, dest_sim, source_direction: str,
                 dest_direction: str, label: str, queue_depth: int = 1) -> None:
        for sim in (source_sim, dest_sim):
            if sim.sync.name != REQUIRED_SYNC_SCHEME:
                raise ValueError(
                    f"relay worker {label}: sync scheme {sim.sync.name!r} "
                    f"cannot gate on the dest's upload_done; use "
                    f"{REQUIRED_SYNC_SCHEME!r}")
        super().__init__(source_sim, dest_sim, source_direction,
                         dest_direction, label, queue_depth=queue_depth)
        if not np.shares_memory(self._source_view, self._dest_view):
            raise ValueError(
                f"relay worker {label}: source and dest stagings are "
                f"different memory — nothing would carry the bytes")
        # Stamp changed between the pre-upload check and upload_done: the
        # source overwrote the block while the dest was reading it.
        self.overwrite_error_count = 0

    def _run(self) -> None:
        self.last_activity = ("init", 0, time.perf_counter_ns())
        self.iteration_count = 0
        self.last_completed_frame = -1
        try:
            while True:
                self.last_activity = ("wait_queue", -1, time.perf_counter_ns())
                frame_n = self.work_queue.get()
                if frame_n == _STOP_SENTINEL:
                    return
                self.iteration_count += 1

                # 1a. Source readback_done(n): the shared block holds frame n.
                t_dequeue = time.perf_counter_ns()
                self.last_activity = ("wait_source_timeline", frame_n, t_dequeue)
                source_semaphore, source_value = self.source.sync.source_readback_op(
                    self.source_direction, frame_n)
                self.source.wait_semaphore(source_semaphore, source_value)
                t_source_wait = time.perf_counter_ns()
                # 1b. Dest readback_done(n): host-signal monotonicity guard,
                #     unchanged (see GhostMigrationWorker._run).
                self.last_activity = ("wait_dest_timeline", frame_n, time.perf_counter_ns())
                guard_semaphore, guard_value = self.dest.sync.dest_guard_op(
                    self.dest_direction, frame_n)
                self.dest.wait_semaphore(guard_semaphore, guard_value)
                t_dest_guard = time.perf_counter_ns()
                # 1c. Dest upload_done(n-1). Kept for a like-for-like segment
                #     breakdown, but it no longer protects anything: the
                #     block's writer is the source's readback DMA, which
                #     finished before 1a returned. The consumed gate below
                #     is what keeps readback(n) behind upload(n-1).
                for upload_guard_semaphore, upload_guard_value in (
                        self.dest.sync.dest_upload_guard_ops(
                            self.dest_direction, frame_n)):
                    self.dest.wait_semaphore(upload_guard_semaphore,
                                             upload_guard_value)
                t_wait = time.perf_counter_ns()

                # 1c-bis. Host-side frame-stamp check, unchanged.
                stamp = struct.unpack_from(
                    "<I", self._source_view, len(self._source_view) - 4)[0]
                if self._stamp_base is None:
                    self._stamp_base = stamp - frame_n
                elif stamp != self._stamp_base + frame_n:
                    self.stamp_error_count += 1
                    if self.stamp_error_count <= 5:
                        print(f"[worker {self.label}] *** STALE READBACK at "
                              f"frame {frame_n}: stamp={stamp} expected="
                              f"{self._stamp_base + frame_n} ***", flush=True)

                # 2. No byte copy: the dest's upload reads the block the
                #    source's readback wrote.
                self.last_copy_bytes = 0
                t_copy = time.perf_counter_ns()

                # 3. Host-signal dest's worker_done(n) -> releases upload(n).
                #    Must precede the upload_done wait below (upload(n)
                #    waits this signal).
                self.last_activity = ("signal_dest_timeline", frame_n, time.perf_counter_ns())
                signal_semaphore, signal_value = self.dest.sync.worker_signal_op(
                    self.dest_direction, frame_n)
                current_dest = self.dest.semaphore_value(signal_semaphore)
                assert current_dest >= guard_value, (
                    f"worker {self.label} about to host_signal({signal_value}) on "
                    f"dest, but dest semaphore={current_dest} < readback_done"
                    f"={guard_value} (Vulkan backwards-signal hazard).")
                self.dest.host_signal_semaphore(signal_semaphore, signal_value)
                t_signal = time.perf_counter_ns()

                # 4. Wait dest upload_done(n): the upload DMA finished
                #    reading the block. dest_upload_guard_ops(frame_n + 1) is
                #    exactly "upload of frame_n is done".
                self.last_activity = ("wait_dest_upload", frame_n, time.perf_counter_ns())
                for upload_semaphore, upload_value in (
                        self.dest.sync.dest_upload_guard_ops(
                            self.dest_direction, frame_n + 1)):
                    self.dest.wait_semaphore(upload_semaphore, upload_value)
                t_upload_done = time.perf_counter_ns()

                # 4-bis. The block must still hold frame n: nothing may have
                #    started readback(n+1) before the consumed signal below.
                stamp_after_upload = struct.unpack_from(
                    "<I", self._source_view, len(self._source_view) - 4)[0]
                if stamp_after_upload != stamp:
                    self.overwrite_error_count += 1
                    if self.overwrite_error_count <= 5:
                        print(f"[worker {self.label}] *** BLOCK OVERWRITTEN "
                              f"DURING UPLOAD at frame {frame_n}: stamp "
                              f"{stamp} -> {stamp_after_upload} ***", flush=True)

                # 5. Consumed-ack on the SOURCE: readback(n+1) may now
                #    overwrite the block.
                self.last_activity = ("signal_source_consumed", frame_n, time.perf_counter_ns())
                consumed_semaphore, consumed_value = (
                    self.source.sync.consumed_signal_op(
                        self.source_direction, frame_n))
                self.source.host_signal_semaphore(consumed_semaphore,
                                                  consumed_value)
                t_consumed = time.perf_counter_ns()
                self.last_activity = ("done_frame", frame_n, time.perf_counter_ns())
                self.last_completed_frame = frame_n

                self.timestamps[frame_n] = {
                    # Same keys as the three-hop worker (copy = the empty
                    # segment between the stamp check and the signal), plus
                    # the two-hop tail.
                    "dequeue_ns": t_dequeue,
                    "source_wait_ns": t_source_wait,
                    "dest_guard_ns": t_dest_guard,
                    "wait_ns": t_wait,
                    "copy_ns": t_copy,
                    "signal_ns": t_signal,
                    "upload_done_ns": t_upload_done,
                    "consumed_ns": t_consumed,
                }
        except BaseException as error:  # noqa: BLE001 — capture everything for diagnostics
            self.last_error = error
            import traceback
            print(f"[worker {self.label}] DIED at {self.last_activity}: {error!r}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)


class ChainOrchestratorTwoHop(ChainOrchestratorV5):
    """ChainOrchestratorV5 driving GhostRelayWorker threads.

    ChainOrchestratorV5 constructs GhostMigrationWorker by name, so the
    base constructor runs unchanged and its (idle, never notified) workers
    are stopped and replaced one for one. Frame loop, bootstrap, defrag and
    watchdogs are inherited. The bootstrap bridge
    (receiver_view[:] = sender_view) becomes a copy of a block onto itself."""

    def __init__(self, sims: list, *, defrag_cadence: int = 1000,
                 worker_queue_depth: int = 4) -> None:
        for index, sim in enumerate(sims):
            if not isinstance(sim, SphSimulatorTwoHop):
                raise TypeError(
                    f"sim {index} is {type(sim).__name__}; two-hop needs "
                    f"SphSimulatorTwoHop on every slab")
        super().__init__(sims, defrag_cadence=defrag_cadence,
                         worker_queue_depth=worker_queue_depth)
        copy_workers = self.workers
        for worker in copy_workers:
            worker.stop()
        self.workers = tuple(
            GhostRelayWorker(
                source_sim=worker.source, dest_sim=worker.dest,
                source_direction=worker.source_direction,
                dest_direction=worker.dest_direction,
                label=worker.label, queue_depth=worker_queue_depth)
            for worker in copy_workers)
        for worker in self.workers:
            worker.start()
        print(f"[ChainOrchTwoHop] {len(self.workers)} relay workers "
              f"(semaphore relay only, no byte copy)")
