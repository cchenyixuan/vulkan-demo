from vulkan import *


def create_command_pool(device, queue_family_index):
    pool_info = VkCommandPoolCreateInfo(
        flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queueFamilyIndex=queue_family_index,
    )
    return vkCreateCommandPool(device, pool_info, None)


def create_framebuffers(device, render_pass, image_views, extent):
    framebuffers = []
    for view in image_views:
        fb_info = VkFramebufferCreateInfo(
            renderPass=render_pass,
            attachmentCount=1,
            pAttachments=[view],
            width=extent.width,
            height=extent.height,
            layers=1,
        )
        framebuffers.append(vkCreateFramebuffer(device, fb_info, None))
    return framebuffers


def allocate_command_buffers(device, command_pool, count):
    alloc_info = VkCommandBufferAllocateInfo(
        commandPool=command_pool,
        level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=count,
    )
    return vkAllocateCommandBuffers(device, alloc_info)


def record_command_buffer(cmd_buf, render_pass, framebuffer, extent, pipeline):
    begin_info = VkCommandBufferBeginInfo()
    vkBeginCommandBuffer(cmd_buf, begin_info)

    clear_value = VkClearValue(color=VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0]))
    render_pass_info = VkRenderPassBeginInfo(
        renderPass=render_pass,
        framebuffer=framebuffer,
        renderArea=VkRect2D(
            offset=VkOffset2D(x=0, y=0),
            extent=extent,
        ),
        clearValueCount=1,
        pClearValues=[clear_value],
    )

    vkCmdBeginRenderPass(cmd_buf, render_pass_info, VK_SUBPASS_CONTENTS_INLINE)
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)

    viewport = VkViewport(
        x=0.0, y=0.0,
        width=float(extent.width),
        height=float(extent.height),
        minDepth=0.0, maxDepth=1.0,
    )
    vkCmdSetViewport(cmd_buf, 0, 1, [viewport])

    scissor = VkRect2D(offset=VkOffset2D(x=0, y=0), extent=extent)
    vkCmdSetScissor(cmd_buf, 0, 1, [scissor])

    vkCmdDraw(cmd_buf, 3, 1, 0, 0)

    vkCmdEndRenderPass(cmd_buf)
    vkEndCommandBuffer(cmd_buf)
