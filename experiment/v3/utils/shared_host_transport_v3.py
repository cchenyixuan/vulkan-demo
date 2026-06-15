"""
shared_host_transport_v3.py — V3.3 cross-GPU ghost transport: shared host RAM
+ cross-device binary semaphore. No CPU worker thread.

Replaces transport_v3.py's per-pathway GhostMigrationWorker. The transport
architecture collapses from:

    src VRAM ─[transfer Q]─> sender_staging ─[CPU memcpy]─> receiver_staging
                                                              ─[transfer Q]─> dst VRAM
    (5N timeline, 2 cross-family handshakes per direction per frame,
     CPU worker thread with host_signal_timeline)

into:

    src VRAM ─[compute Q]─> shared_host_buffer ─[compute Q]─> dst VRAM
    (signaled by cross_device_binary_semaphore; no CPU involvement,
     no transfer queue, no cross-family handshake)

Both VkDevices import the SAME host RAM pointer via VK_EXT_external_memory_host.
Coherence across the PCIe boundary is guaranteed by HOST_COHERENT memory type
(validated 4/4 PASS by probe_external_memory_host.py on AMD 7900XTX + NV 4060Ti,
2026-05-27).

Cross-vendor binary semaphores via VK_KHR_external_semaphore_win32 OPAQUE_WIN32
handles (validated 4/4 PASS by probe_interop_full.py, same date).

This module is PURE RESOURCE MANAGEMENT — no command-buffer recording, no
per-frame logic, no threads. The simulator_v3 phase_a / phase_c command-buffer
recorders are responsible for appending vkCmdCopyBuffer between device-local
SoA buffers and these shared buffers, plus signaling / waiting the cross-device
binary semaphore.

See docs/sph_v3_design.md §1-2 for the full V3.3 design.
See memory/project_cross_vendor_shared_host_breakthrough.md for probe data.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi

if TYPE_CHECKING:
    from experiment.v3.utils.vulkan_context_v3 import VulkanContextV3


# ============================================================================
# Win32 page-aligned host allocation
# ============================================================================

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_READWRITE = 0x04

_kernel32 = ctypes.windll.kernel32
_kernel32.VirtualAlloc.argtypes = [
    wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
_kernel32.VirtualAlloc.restype = wintypes.LPVOID
_kernel32.VirtualFree.argtypes = [
    wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
_kernel32.VirtualFree.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
_kernel32.CloseHandle.restype = wintypes.BOOL


def _virtual_alloc(size: int) -> int:
    """Allocate page-aligned host memory; returns address as uint64."""
    address = _kernel32.VirtualAlloc(
        None, size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
    if not address:
        raise OSError("VirtualAlloc returned NULL")
    return int(address)


def _virtual_free(address: int) -> None:
    if address:
        _kernel32.VirtualFree(ctypes.c_void_p(address), 0, _MEM_RELEASE)


def _close_handle(handle) -> None:
    """Close a Win32 HANDLE returned by vkGetSemaphoreWin32HandleKHR.

    The handle arrives as a cffi `void *` cdata (not a Python int), so we
    extract the underlying address via ffi.cast before handing it to ctypes.
    Tolerates plain Python int input as well (e.g. for tests).
    """
    if not handle:
        return
    if isinstance(handle, int):
        address = handle
    else:
        # cffi 'void *' cdata: extract address as uintptr_t.
        address = int(ffi.cast("uintptr_t", handle))
    if address:
        _kernel32.CloseHandle(ctypes.c_void_p(address))


# ============================================================================
# Extension name constants
# ============================================================================

# Instance-level extensions required for the VK_KHR_external_memory_win32 and
# VK_KHR_external_semaphore_win32 device extensions to be enabled. Both are
# provided by Vulkan 1.1+ core (vkGetPhysicalDeviceProperties2 etc.) but the
# explicit instance extension names are stable and must be enabled.
REQUIRED_INSTANCE_EXTENSIONS: tuple[str, ...] = (
    "VK_KHR_get_physical_device_properties2",
    "VK_KHR_external_memory_capabilities",
    "VK_KHR_external_semaphore_capabilities",
)

# Device-level extensions required on EACH VkDevice that participates in
# shared-host-transport or cross-device sem use.
REQUIRED_DEVICE_EXTENSIONS: tuple[str, ...] = (
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_win32",
    "VK_EXT_external_memory_host",
    "VK_KHR_external_semaphore",
    "VK_KHR_external_semaphore_win32",
)

# Vulkan handle-type bit used for host-pointer import (single source of truth).
_HOST_POINTER_HANDLE_TYPE = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT

# Handle type for cross-device binary semaphores. OPAQUE_WIN32 (not KMT)
# matches what probe_interop_full.py validated cross-vendor.
_SEMAPHORE_HANDLE_TYPE = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT

# Standard usage flags on the imported buffer. TRANSFER_SRC / TRANSFER_DST
# cover the only operations simulator_v3 will perform (vkCmdCopyBuffer); we
# also request STORAGE_BUFFER so a future ablation could shader-write to
# shared host RAM directly without recreating the buffer.
_SHARED_BUFFER_USAGE = (
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
)


# ============================================================================
# Per-device imported view of one shared host region
# ============================================================================

@dataclass
class _ImportedView:
    """One VkDevice's view of a shared host pointer.

    Both the src device and the dst device hold one of these pointing at the
    SAME host_addr; reads/writes through their respective VkBuffer hit the
    same physical RAM bytes (HOST_COHERENT guarantees cross-PCIe coherence).
    """
    device: object             # VkDevice
    buffer: object             # VkBuffer
    memory: object             # VkDeviceMemory
    size: int                  # logical bytes (== region.size, NOT aligned_size)


@dataclass
class SharedHostRegion:
    """One direction of shared host transport.

    Owns:
      - host_addr: VirtualAlloc'd page-aligned RAM pointer
      - aligned_size: rounded-up allocation (>= size)
      - size: logical payload bytes (what simulator records vkCmdCopyBuffer for)
      - src_view / dst_view: per-device imported VkBuffer + VkDeviceMemory
      - mapped_view: numpy.uint8 view over the host bytes (diagnostic only;
        the hot path does NOT read/write through this — both GPUs use
        vkCmdCopyBuffer directly).
    """
    host_addr: int
    aligned_size: int
    size: int
    src_view: _ImportedView
    dst_view: _ImportedView
    mapped_view: np.ndarray = field(repr=False)

    def buffer_for(self, ctx: "VulkanContextV3") -> object:
        """Return the VkBuffer that ``ctx``'s device should use to access
        this shared region. Raises if ``ctx`` is neither src nor dst."""
        if ctx.device == self.src_view.device:
            return self.src_view.buffer
        if ctx.device == self.dst_view.device:
            return self.dst_view.buffer
        raise ValueError(
            "ctx.device is neither src nor dst of this SharedHostRegion")


# ============================================================================
# Cross-device binary semaphore
# ============================================================================

@dataclass
class CrossDeviceSemaphore:
    """A pair of binary VkSemaphores that share the same underlying Win32
    handle. ``src_semaphore`` lives on src_device and is the one to pass to
    vkQueueSubmit.pSignalSemaphores; ``dst_semaphore`` lives on dst_device
    and goes into vkQueueSubmit.pWaitSemaphores.

    A binary semaphore is single-use: src signals it once, dst waits on it
    once, then both are auto-reset. For per-frame use, allocate one
    CrossDeviceSemaphore per direction per ping-pong slot (frame parity).

    The underlying OPAQUE_WIN32 handle is closed after both import calls
    succeed (Vulkan spec: drivers dup the handle internally).
    """
    src_device: object
    dst_device: object
    src_semaphore: object
    dst_semaphore: object


# ============================================================================
# SharedHostTransport — top-level container
# ============================================================================

@dataclass
class SharedHostTransport:
    """Container for both directions of shared host transport between two
    VulkanContextV3 instances.

    Construction is symmetric: pass two contexts and per-direction byte sizes.
    Both directions are allocated up front. The simulator/orchestrator
    accesses regions and semaphores via the named ``a_to_b`` / ``b_to_a``
    direction keys.

    The naming convention is fixed at construction: ctx_a is the "a" side,
    ctx_b is the "b" side. Caller decides which physical device maps to
    which. Convention in V3 is: ctx_a = first sim (lower slab x-index),
    ctx_b = second sim. So a_to_b carries ctx_a's TRAILING boundary to
    ctx_b's LEADING boundary, and b_to_a carries ctx_b's LEADING boundary
    back to ctx_a's TRAILING boundary.
    """
    ctx_a: "VulkanContextV3"
    ctx_b: "VulkanContextV3"
    region_a_to_b: SharedHostRegion
    region_b_to_a: SharedHostRegion
    semaphore_a_to_b: CrossDeviceSemaphore
    semaphore_b_to_a: CrossDeviceSemaphore

    _destroyed: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        ctx_a: "VulkanContextV3",
        ctx_b: "VulkanContextV3",
        a_to_b_bytes: int,
        b_to_a_bytes: int,
    ) -> "SharedHostTransport":
        if a_to_b_bytes <= 0 or b_to_a_bytes <= 0:
            raise ValueError(
                f"per-direction sizes must be positive; got "
                f"a_to_b={a_to_b_bytes}, b_to_a={b_to_a_bytes}")

        alignment = max(
            _query_min_imported_host_pointer_alignment(ctx_a),
            _query_min_imported_host_pointer_alignment(ctx_b),
        )

        region_a_to_b = None
        region_b_to_a = None
        semaphore_a_to_b = None
        semaphore_b_to_a = None
        try:
            region_a_to_b = _allocate_shared_region(
                ctx_a, ctx_b, a_to_b_bytes, alignment, direction_label="a_to_b")
            region_b_to_a = _allocate_shared_region(
                ctx_b, ctx_a, b_to_a_bytes, alignment, direction_label="b_to_a")
            semaphore_a_to_b = _create_cross_device_semaphore(
                ctx_a, ctx_b, direction_label="a_to_b")
            semaphore_b_to_a = _create_cross_device_semaphore(
                ctx_b, ctx_a, direction_label="b_to_a")
        except BaseException:
            # Construction is atomic: any partial allocation is unwound
            # before re-raising so the caller never holds half-initialized
            # state. _destroy_region / _destroy_semaphore are no-ops on None.
            _destroy_region(region_a_to_b)
            _destroy_region(region_b_to_a)
            _destroy_semaphore(semaphore_a_to_b)
            _destroy_semaphore(semaphore_b_to_a)
            raise

        print(
            f"[v3] SharedHostTransport: a_to_b "
            f"{a_to_b_bytes} B (alloc {region_a_to_b.aligned_size}), "
            f"b_to_a {b_to_a_bytes} B (alloc {region_b_to_a.aligned_size}), "
            f"page alignment {alignment} B")
        return cls(
            ctx_a=ctx_a,
            ctx_b=ctx_b,
            region_a_to_b=region_a_to_b,
            region_b_to_a=region_b_to_a,
            semaphore_a_to_b=semaphore_a_to_b,
            semaphore_b_to_a=semaphore_b_to_a,
        )

    # ------------------------------------------------------------------
    # Lookups (used by simulator_v3 cmd recording + orchestrator submit)
    # ------------------------------------------------------------------

    def region(self, direction: str) -> SharedHostRegion:
        if direction == "a_to_b":
            return self.region_a_to_b
        if direction == "b_to_a":
            return self.region_b_to_a
        raise ValueError(
            f"direction must be 'a_to_b' or 'b_to_a'; got {direction!r}")

    def semaphore(self, direction: str) -> CrossDeviceSemaphore:
        if direction == "a_to_b":
            return self.semaphore_a_to_b
        if direction == "b_to_a":
            return self.semaphore_b_to_a
        raise ValueError(
            f"direction must be 'a_to_b' or 'b_to_a'; got {direction!r}")

    # ------------------------------------------------------------------
    # Destruction
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if self._destroyed:
            return
        # Wait both devices idle so we don't free a buffer the GPU is mid-DMA on.
        # Caller is expected to have already drained the timeline, but this is
        # a defensive belt-and-suspenders.
        if self.ctx_a.device is not None:
            vkDeviceWaitIdle(self.ctx_a.device)
        if self.ctx_b.device is not None:
            vkDeviceWaitIdle(self.ctx_b.device)
        _destroy_region(self.region_a_to_b)
        _destroy_region(self.region_b_to_a)
        _destroy_semaphore(self.semaphore_a_to_b)
        _destroy_semaphore(self.semaphore_b_to_a)
        self._destroyed = True

    def __enter__(self) -> "SharedHostTransport":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()


# ============================================================================
# Internal helpers — host-pointer import path
# ============================================================================

def _query_min_imported_host_pointer_alignment(ctx: "VulkanContextV3") -> int:
    """VkPhysicalDeviceExternalMemoryHostPropertiesEXT.minImportedHostPointerAlignment.

    Both AMD 7900 XTX and NV 4060 Ti report 4096 B (one OS page) per probe
    measurements. We query at runtime to stay portable across vendors that
    may demand a larger alignment.
    """
    external_host_properties = VkPhysicalDeviceExternalMemoryHostPropertiesEXT()
    properties_2 = VkPhysicalDeviceProperties2(pNext=external_host_properties)
    function = vkGetInstanceProcAddr(
        ctx.instance, "vkGetPhysicalDeviceProperties2KHR")
    if function is None:
        function = vkGetInstanceProcAddr(
            ctx.instance, "vkGetPhysicalDeviceProperties2")
    if function is None:
        raise RuntimeError(
            "vkGetPhysicalDeviceProperties2 entry point not found; "
            "VK_KHR_get_physical_device_properties2 instance ext likely missing")
    function(ctx.physical_device, properties_2)
    return int(external_host_properties.minImportedHostPointerAlignment)


def _query_memory_host_pointer_properties(ctx: "VulkanContextV3",
                                          host_addr: int) -> int:
    """vkGetMemoryHostPointerPropertiesEXT — returns the bitmask of memory
    types compatible with importing this host pointer as a HOST_ALLOCATION."""
    function = vkGetDeviceProcAddr(
        ctx.device, "vkGetMemoryHostPointerPropertiesEXT")
    if function is None:
        raise RuntimeError(
            "vkGetMemoryHostPointerPropertiesEXT entry point not found on "
            f"device '{ctx.device_name}'; VK_EXT_external_memory_host likely "
            "missing from VulkanContextV3 device extensions")
    properties = VkMemoryHostPointerPropertiesEXT()
    # python-vulkan accepts plain int for void* arguments. Wrapping host_addr
    # in ctypes.c_void_p here raises TypeError in cffi conversion.
    function(ctx.device, _HOST_POINTER_HANDLE_TYPE, host_addr, properties)
    return int(properties.memoryTypeBits)


def _import_buffer_for_device(
    ctx: "VulkanContextV3",
    host_addr: int,
    aligned_size: int,
    logical_size: int,
) -> _ImportedView:
    """Create a VkBuffer on ctx.device that aliases the host pointer."""
    external_buffer_info = VkExternalMemoryBufferCreateInfo(
        handleTypes=_HOST_POINTER_HANDLE_TYPE)
    buffer_create_info = VkBufferCreateInfo(
        pNext=external_buffer_info,
        size=aligned_size,
        usage=_SHARED_BUFFER_USAGE,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    buffer_handle = vkCreateBuffer(ctx.device, buffer_create_info, None)

    try:
        buffer_requirements = vkGetBufferMemoryRequirements(
            ctx.device, buffer_handle)
        host_pointer_filter = _query_memory_host_pointer_properties(
            ctx, host_addr)
        combined_filter = (
            buffer_requirements.memoryTypeBits & host_pointer_filter)
        if combined_filter == 0:
            raise RuntimeError(
                f"no memory type on '{ctx.device_name}' satisfies both buffer "
                f"requirements (0x{buffer_requirements.memoryTypeBits:x}) "
                f"and host-pointer import filter (0x{host_pointer_filter:x})")

        # Both AMD and NV expose HOST_VISIBLE | HOST_COHERENT memory types
        # compatible with HOST_ALLOCATION import (validated by probe). The
        # find_memory_type helper on VulkanContextV3 handles preferred-flag
        # fallback if HOST_COHERENT isn't matched (shouldn't happen on either
        # vendor we test).
        memory_type_index = ctx.find_memory_type(
            type_bits=combined_filter,
            required_properties=(
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            preferred_properties=0,
        )

        import_info = VkImportMemoryHostPointerInfoEXT(
            handleType=_HOST_POINTER_HANDLE_TYPE,
            pHostPointer=host_addr,
        )
        memory_handle = vkAllocateMemory(ctx.device, VkMemoryAllocateInfo(
            pNext=import_info,
            allocationSize=aligned_size,
            memoryTypeIndex=memory_type_index,
        ), None)
        vkBindBufferMemory(ctx.device, buffer_handle, memory_handle, 0)
    except BaseException:
        vkDestroyBuffer(ctx.device, buffer_handle, None)
        raise

    return _ImportedView(
        device=ctx.device,
        buffer=buffer_handle,
        memory=memory_handle,
        size=logical_size,
    )


def _allocate_shared_region(
    src_ctx: "VulkanContextV3",
    dst_ctx: "VulkanContextV3",
    logical_size: int,
    alignment: int,
    direction_label: str,
) -> SharedHostRegion:
    """Allocate one direction of shared transport: VirtualAlloc + import on
    BOTH devices."""
    aligned_size = ((logical_size + alignment - 1) // alignment) * alignment

    host_addr = _virtual_alloc(aligned_size)
    src_view = None
    dst_view = None
    try:
        src_view = _import_buffer_for_device(
            src_ctx, host_addr, aligned_size, logical_size)
        dst_view = _import_buffer_for_device(
            dst_ctx, host_addr, aligned_size, logical_size)
    except BaseException:
        # Free both VkBuffer / VkDeviceMemory we managed to allocate, then
        # release the host pages. Re-raise so create() can roll back the
        # whole SharedHostTransport.
        if src_view is not None:
            vkFreeMemory(src_ctx.device, src_view.memory, None)
            vkDestroyBuffer(src_ctx.device, src_view.buffer, None)
        if dst_view is not None:
            vkFreeMemory(dst_ctx.device, dst_view.memory, None)
            vkDestroyBuffer(dst_ctx.device, dst_view.buffer, None)
        _virtual_free(host_addr)
        raise

    # numpy view for diagnostics only. Wraps the FULL aligned region so any
    # asserts that touch the tail bytes won't IOOB. Hot path never reads
    # through this — both GPUs do vkCmdCopyBuffer to/from their VkBuffer
    # view and hardware coherency handles the rest.
    mapped_view = np.frombuffer(
        (ctypes.c_uint8 * aligned_size).from_address(host_addr),
        dtype=np.uint8,
        count=aligned_size,
    )

    return SharedHostRegion(
        host_addr=host_addr,
        aligned_size=aligned_size,
        size=logical_size,
        src_view=src_view,
        dst_view=dst_view,
        mapped_view=mapped_view,
    )


def _destroy_region(region: Optional[SharedHostRegion]) -> None:
    if region is None:
        return
    for view in (region.src_view, region.dst_view):
        if view is None:
            continue
        if view.memory is not None:
            vkFreeMemory(view.device, view.memory, None)
        if view.buffer is not None:
            vkDestroyBuffer(view.device, view.buffer, None)
    _virtual_free(region.host_addr)


# ============================================================================
# Internal helpers — cross-device binary semaphore
# ============================================================================

def _create_cross_device_semaphore(
    src_ctx: "VulkanContextV3",
    dst_ctx: "VulkanContextV3",
    direction_label: str,
) -> CrossDeviceSemaphore:
    """Create an exportable binary VkSemaphore on src, export its OPAQUE_WIN32
    handle, then create-and-import the matching VkSemaphore on dst."""

    # ---- src: exportable binary semaphore --------------------------------
    src_export_info = VkExportSemaphoreCreateInfo(
        handleTypes=_SEMAPHORE_HANDLE_TYPE)
    src_type_info = VkSemaphoreTypeCreateInfo(
        sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        semaphoreType=VK_SEMAPHORE_TYPE_BINARY,
        initialValue=0,
        pNext=src_export_info,
    )
    src_semaphore = vkCreateSemaphore(
        src_ctx.device, VkSemaphoreCreateInfo(pNext=src_type_info), None)

    win32_handle = 0
    dst_semaphore = None
    try:
        # ---- export ------------------------------------------------------
        get_handle_function = vkGetDeviceProcAddr(
            src_ctx.device, "vkGetSemaphoreWin32HandleKHR")
        if get_handle_function is None:
            raise RuntimeError(
                "vkGetSemaphoreWin32HandleKHR not loaded; "
                "VK_KHR_external_semaphore_win32 likely not enabled")
        win32_handle = get_handle_function(
            src_ctx.device,
            VkSemaphoreGetWin32HandleInfoKHR(
                semaphore=src_semaphore, handleType=_SEMAPHORE_HANDLE_TYPE),
        )

        # ---- dst: matching binary semaphore, then import handle ----------
        dst_export_info = VkExportSemaphoreCreateInfo(
            handleTypes=_SEMAPHORE_HANDLE_TYPE)
        dst_type_info = VkSemaphoreTypeCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            semaphoreType=VK_SEMAPHORE_TYPE_BINARY,
            initialValue=0,
            pNext=dst_export_info,
        )
        dst_semaphore = vkCreateSemaphore(
            dst_ctx.device, VkSemaphoreCreateInfo(pNext=dst_type_info), None)

        import_function = vkGetDeviceProcAddr(
            dst_ctx.device, "vkImportSemaphoreWin32HandleKHR")
        if import_function is None:
            raise RuntimeError(
                "vkImportSemaphoreWin32HandleKHR not loaded; "
                "VK_KHR_external_semaphore_win32 likely not enabled on dst")
        import_function(
            dst_ctx.device,
            VkImportSemaphoreWin32HandleInfoKHR(
                semaphore=dst_semaphore,
                handleType=_SEMAPHORE_HANDLE_TYPE,
                handle=win32_handle,
                name=None,
            ),
        )
    except BaseException:
        if dst_semaphore is not None:
            vkDestroySemaphore(dst_ctx.device, dst_semaphore, None)
        vkDestroySemaphore(src_ctx.device, src_semaphore, None)
        _close_handle(win32_handle)
        raise

    # Per Vulkan spec for OPAQUE_WIN32 import: the driver dup's the handle
    # internally, so the original handle returned by vkGetSemaphoreWin32HandleKHR
    # must be closed by the application to avoid leaking a Win32 handle on
    # every frame's worth of semaphore creation (we only create per-sim, so
    # leak would be tiny — close anyway for correctness).
    _close_handle(win32_handle)

    return CrossDeviceSemaphore(
        src_device=src_ctx.device,
        dst_device=dst_ctx.device,
        src_semaphore=src_semaphore,
        dst_semaphore=dst_semaphore,
    )


def _destroy_semaphore(semaphore: Optional[CrossDeviceSemaphore]) -> None:
    if semaphore is None:
        return
    if semaphore.dst_semaphore is not None:
        vkDestroySemaphore(semaphore.dst_device, semaphore.dst_semaphore, None)
    if semaphore.src_semaphore is not None:
        vkDestroySemaphore(semaphore.src_device, semaphore.src_semaphore, None)
