from vulkan import *
from utils.phase1.shaders import load_spirv, create_shader_module


def create_render_pass(device, image_format):
    color_attachment = VkAttachmentDescription(
        format=image_format,
        samples=VK_SAMPLE_COUNT_1_BIT,
        loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
        storeOp=VK_ATTACHMENT_STORE_OP_STORE,
        stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
        initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
        finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    )
    color_attachment_ref = VkAttachmentReference(
        attachment=0,
        layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    )
    subpass = VkSubpassDescription(
        pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
        colorAttachmentCount=1,
        pColorAttachments=[color_attachment_ref],
    )
    dependency = VkSubpassDependency(
        srcSubpass=VK_SUBPASS_EXTERNAL,
        dstSubpass=0,
        srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        srcAccessMask=0,
        dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    )
    render_pass_info = VkRenderPassCreateInfo(
        attachmentCount=1,
        pAttachments=[color_attachment],
        subpassCount=1,
        pSubpasses=[subpass],
        dependencyCount=1,
        pDependencies=[dependency],
    )
    return vkCreateRenderPass(device, render_pass_info, None)


PARTICLE_PUSH_CONSTANT_SIZE = 48  # vec4 view + vec4 params + vec4 color


def create_particle_graphics_pipeline(device, render_pass, descriptor_set_layout):
    vert_code = load_spirv("particle.vert.spv")
    frag_code = load_spirv("particle.frag.spv")
    vert_module = create_shader_module(device, vert_code)
    frag_module = create_shader_module(device, frag_code)

    stages = [
        VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_VERTEX_BIT, module=vert_module, pName="main",
        ),
        VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_FRAGMENT_BIT, module=frag_module, pName="main",
        ),
    ]

    vertex_input = VkPipelineVertexInputStateCreateInfo(
        vertexBindingDescriptionCount=0,
        vertexAttributeDescriptionCount=0,
    )
    input_assembly = VkPipelineInputAssemblyStateCreateInfo(
        topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        primitiveRestartEnable=VK_FALSE,
    )
    dynamic_state = VkPipelineDynamicStateCreateInfo(
        dynamicStateCount=2,
        pDynamicStates=[VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR],
    )
    viewport_state = VkPipelineViewportStateCreateInfo(viewportCount=1, scissorCount=1)
    rasterizer = VkPipelineRasterizationStateCreateInfo(
        depthClampEnable=VK_FALSE,
        rasterizerDiscardEnable=VK_FALSE,
        polygonMode=VK_POLYGON_MODE_FILL,
        lineWidth=1.0,
        cullMode=VK_CULL_MODE_NONE,
        frontFace=VK_FRONT_FACE_CLOCKWISE,
        depthBiasEnable=VK_FALSE,
    )
    multisampling = VkPipelineMultisampleStateCreateInfo(
        sampleShadingEnable=VK_FALSE,
        rasterizationSamples=VK_SAMPLE_COUNT_1_BIT,
    )
    color_blend_attachment = VkPipelineColorBlendAttachmentState(
        colorWriteMask=(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT),
        blendEnable=VK_FALSE,
    )
    color_blending = VkPipelineColorBlendStateCreateInfo(
        logicOpEnable=VK_FALSE,
        attachmentCount=1,
        pAttachments=[color_blend_attachment],
    )

    push_range = VkPushConstantRange(
        stageFlags=VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        offset=0,
        size=PARTICLE_PUSH_CONSTANT_SIZE,
    )
    layout_info = VkPipelineLayoutCreateInfo(
        setLayoutCount=1,
        pSetLayouts=[descriptor_set_layout],
        pushConstantRangeCount=1,
        pPushConstantRanges=[push_range],
    )
    pipeline_layout = vkCreatePipelineLayout(device, layout_info, None)

    pipeline_info = VkGraphicsPipelineCreateInfo(
        stageCount=len(stages),
        pStages=stages,
        pVertexInputState=vertex_input,
        pInputAssemblyState=input_assembly,
        pViewportState=viewport_state,
        pRasterizationState=rasterizer,
        pMultisampleState=multisampling,
        pColorBlendState=color_blending,
        pDynamicState=dynamic_state,
        layout=pipeline_layout,
        renderPass=render_pass,
        subpass=0,
    )
    pipelines = vkCreateGraphicsPipelines(device, None, 1, [pipeline_info], None)

    vkDestroyShaderModule(device, vert_module, None)
    vkDestroyShaderModule(device, frag_module, None)
    return pipeline_layout, pipelines[0]
