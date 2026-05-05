"""
multi_gpu.py — MultiGPUContext: holds N VulkanContext instances simultaneously,
one per partition. V1 uses N=2 (cross-vendor: 4060 Ti + 7900 XTX); V3+ may
generalize.

This is V1.0 scaffolding only. There is no inter-GPU communication here —
ghost SoA, migration channel, and the double-sync run loop layer on later
(see docs/sph_v1_design.md). What this file proves: two VulkanContext
instances can coexist in one Python process, each pinning a different
physical device.

Imports unchanged V0 utilities from `utils.sph.*`. Does NOT modify V0.
"""

from typing import Optional

from utils.sph.vulkan_context import VulkanContext


class MultiGPUContext:
    """N coexisting VulkanContexts. Construct once; destroy in reverse order.

    Usage:
        with MultiGPUContext.create(device_indices=[0, 1]) as multi_ctx:
            for context_index, context in enumerate(multi_ctx.contexts):
                ...                                # per-GPU work
    """

    def __init__(self, contexts: list[VulkanContext]):
        self.contexts: list[VulkanContext] = contexts
        self._destroyed = False

    @classmethod
    def create(
        cls,
        device_indices: list[int],
        application_name_prefix: str = "sph_v1",
        enable_validation: bool = True,
        extra_device_extensions: Optional[list] = None,
    ) -> "MultiGPUContext":
        """Build one VulkanContext per requested physical-device index.

        Order matters: contexts[i] uses device_indices[i]. Downstream code
        (partitioner, sync loop) addresses GPUs by this list index, not by
        physical device index.
        """
        if len(device_indices) < 1:
            raise ValueError("device_indices must contain at least one index")

        contexts: list[VulkanContext] = []
        try:
            for slot_index, device_index in enumerate(device_indices):
                context = VulkanContext.create(
                    application_name=f"{application_name_prefix}_gpu{slot_index}",
                    enable_validation=enable_validation,
                    extra_device_extensions=extra_device_extensions,
                    device_index=device_index,
                )
                contexts.append(context)
        except Exception:
            for context in reversed(contexts):
                context.destroy()
            raise

        return cls(contexts=contexts)

    def __enter__(self) -> "MultiGPUContext":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()

    def destroy(self) -> None:
        if self._destroyed:
            return
        for context in reversed(self.contexts):
            context.destroy()
        self.contexts.clear()
        self._destroyed = True

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, slot_index: int) -> VulkanContext:
        return self.contexts[slot_index]
