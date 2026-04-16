from vulkan import *
from utils.shaders import load_spirv, create_shader_module


COMPUTE_PUSH_CONSTANT_SIZE = 24  # 5 floats + 1 uint


def create_compute_descriptor_set_layout(device):
    """6-binding layout for Phase 1:
    0: particles_in  (SSBO, compute read)
    1: particles_out (SSBO, compute write; also vertex shader read)
    2: outgoing      (SSBO, compute write, host-visible)
    3: incoming      (SSBO, compute read,  host-visible)
    4: counters      (SSBO, compute read+write)
    5: indirect      (SSBO, compute write — shares storage with draw-indirect buffer)
    """
    stages = [
        VK_SHADER_STAGE_COMPUTE_BIT,                                   # 0: also vertex below
        VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT,      # 1: read by vertex for rendering
        VK_SHADER_STAGE_COMPUTE_BIT,                                   # 2
        VK_SHADER_STAGE_COMPUTE_BIT,                                   # 3
        VK_SHADER_STAGE_COMPUTE_BIT,                                   # 4
        VK_SHADER_STAGE_COMPUTE_BIT,                                   # 5
    ]
    bindings = [
        VkDescriptorSetLayoutBinding(
            binding=i, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1, stageFlags=stage,
        ) for i, stage in enumerate(stages)
    ]
    info = VkDescriptorSetLayoutCreateInfo(
        bindingCount=len(bindings),
        pBindings=bindings,
    )
    return vkCreateDescriptorSetLayout(device, info, None)


def create_graphics_descriptor_set_layout(device):
    """Single binding at 1 matching the compute output slot (particles_out).
    The pipeline layout uses a different set — graphics has its own binding 0."""
    binding = VkDescriptorSetLayoutBinding(
        binding=0,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=1,
        stageFlags=VK_SHADER_STAGE_VERTEX_BIT,
    )
    info = VkDescriptorSetLayoutCreateInfo(bindingCount=1, pBindings=[binding])
    return vkCreateDescriptorSetLayout(device, info, None)


def create_compute_pipeline(device, descriptor_set_layout, shader_filename="particle_update.comp.spv"):
    push_range = VkPushConstantRange(
        stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
        offset=0,
        size=COMPUTE_PUSH_CONSTANT_SIZE,
    )
    layout_info = VkPipelineLayoutCreateInfo(
        setLayoutCount=1,
        pSetLayouts=[descriptor_set_layout],
        pushConstantRangeCount=1,
        pPushConstantRanges=[push_range],
    )
    pipeline_layout = vkCreatePipelineLayout(device, layout_info, None)

    shader_code = load_spirv(shader_filename)
    shader_module = create_shader_module(device, shader_code)
    stage = VkPipelineShaderStageCreateInfo(
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=shader_module,
        pName="main",
    )
    pipeline_info = VkComputePipelineCreateInfo(
        stage=stage,
        layout=pipeline_layout,
    )
    pipelines = vkCreateComputePipelines(device, None, 1, [pipeline_info], None)
    vkDestroyShaderModule(device, shader_module, None)
    return pipeline_layout, pipelines[0]


def create_descriptor_pool(device, max_sets, ssbo_descriptors):
    pool_size = VkDescriptorPoolSize(
        type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=ssbo_descriptors,
    )
    info = VkDescriptorPoolCreateInfo(
        maxSets=max_sets,
        poolSizeCount=1,
        pPoolSizes=[pool_size],
    )
    return vkCreateDescriptorPool(device, info, None)


def allocate_compute_descriptor_set(device, pool, layout,
                                    particles_in, particles_out,
                                    outgoing, incoming, counters, indirect,
                                    sizes):
    """sizes is a dict with keys: particles, migration, counters, indirect."""
    alloc_info = VkDescriptorSetAllocateInfo(
        descriptorPool=pool,
        descriptorSetCount=1,
        pSetLayouts=[layout],
    )
    desc_set = vkAllocateDescriptorSets(device, alloc_info)[0]
    buf_infos = [
        VkDescriptorBufferInfo(buffer=particles_in,  offset=0, range=sizes["particles"]),
        VkDescriptorBufferInfo(buffer=particles_out, offset=0, range=sizes["particles"]),
        VkDescriptorBufferInfo(buffer=outgoing,      offset=0, range=sizes["migration"]),
        VkDescriptorBufferInfo(buffer=incoming,      offset=0, range=sizes["migration"]),
        VkDescriptorBufferInfo(buffer=counters,      offset=0, range=sizes["counters"]),
        VkDescriptorBufferInfo(buffer=indirect,      offset=0, range=sizes["indirect"]),
    ]
    writes = [
        VkWriteDescriptorSet(
            dstSet=desc_set, dstBinding=i, dstArrayElement=0,
            descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[buf_infos[i]],
        ) for i in range(6)
    ]
    vkUpdateDescriptorSets(device, len(writes), writes, 0, None)
    return desc_set


def allocate_graphics_descriptor_set(device, pool, layout, particles_out, size):
    alloc_info = VkDescriptorSetAllocateInfo(
        descriptorPool=pool,
        descriptorSetCount=1,
        pSetLayouts=[layout],
    )
    desc_set = vkAllocateDescriptorSets(device, alloc_info)[0]
    write = VkWriteDescriptorSet(
        dstSet=desc_set, dstBinding=0, dstArrayElement=0,
        descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        pBufferInfo=[VkDescriptorBufferInfo(buffer=particles_out, offset=0, range=size)],
    )
    vkUpdateDescriptorSets(device, 1, [write], 0, None)
    return desc_set
