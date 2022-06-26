#include "BasePass.h"

#include "../Scene.h"
#include "../Structs.h"

BasePass::BasePass(vk::Device            device,
                   vk::Extent2D          extent,
                   ResourceManager&      allocator,
                   vk::DescriptorPool    staticDescriptorPool,
                   vk::DescriptorPool    textureDescriptorPool,
                   const nvh::GltfScene& gltfScene,
                   const Scene&          scene)
    : _screenSize(extent)
{
    _vert = Shader(device, "shaders/base.vert.spv", "main", vk::ShaderStageFlagBits::eVertex);
    _frag = Shader(device, "shaders/base.frag.spv", "main", vk::ShaderStageFlagBits::eFragment);

    std::array<vk::DescriptorSetLayoutBinding, 3> bindings {
        {{.binding         = 0,
          .descriptorType  = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eVertex},

         {.binding         = 1,
          .descriptorType  = vk::DescriptorType::eUniformBufferDynamic,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eVertex},

         {.binding         = 2,
          .descriptorType  = vk::DescriptorType::eUniformBufferDynamic,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment}}
    };

    _setLayout = device.createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data(),
    });

    std::array<vk::DescriptorSetLayoutBinding, 4> textureDescriptorBindings {
        {{.binding         = 0,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},
         {.binding         = 1,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},
         {.binding         = 2,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},
         {.binding         = 3,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment}}
    };

    _textureDescriptorSetLayout = device.createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<uint32_t>(textureDescriptorBindings.size()),
        .pBindings    = textureDescriptorBindings.data(),
    });

    std::array<vk::DescriptorSetLayout, 2> descriptorSetLayouts {
        *_setLayout,
        *_textureDescriptorSetLayout,
    };

    _pipelineLayout = device.createPipelineLayoutUnique({
        .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
        .pSetLayouts    = descriptorSetLayouts.data(),
    });

    createPass(device);
    createGraphicsPipeline(device);

    UniformBuffer =
        allocator.createTypedBuffer<BasePass::Uniforms>(1,
                                                        vk::BufferUsageFlagBits::eUniformBuffer,
                                                        VMA_MEMORY_USAGE_CPU_TO_GPU);

    _descriptorSet = std::move(device.allocateDescriptorSetsUnique({
        .descriptorPool     = staticDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &*_setLayout,
    })[0]);

    std::vector<vk::DescriptorSetLayout> texturesLayout(gltfScene.m_materials.size());
    std::fill(texturesLayout.begin(), texturesLayout.end(), *_textureDescriptorSetLayout);

    _texturesDescriptors = device.allocateDescriptorSetsUnique({
        .descriptorPool     = textureDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(texturesLayout.size()),
        .pSetLayouts        = texturesLayout.data(),
    });

    _gltfScene = &gltfScene;
    _scene     = &scene;

    initializeResourcesFor(gltfScene, scene, device);
};

void BasePass::onResized(vk::Device device, vk::Extent2D screenSizes)
{
    _screenSize = screenSizes;
    createGraphicsPipeline(device);
}

void BasePass::issueCommands(vk::CommandBuffer commandBuffer, vk::Framebuffer framebuffer) const
{
    std::array<vk::ClearValue, 5> clearValues {
        {{.color = {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}}},
         {.color = {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}}},
         {.color = {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}}},
         {.color = {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}}},
         {.depthStencil = {1.0f}}}
    };

    commandBuffer.beginRenderPass(
        {
            .renderPass      = *RenderPass,
            .framebuffer     = framebuffer,
            .renderArea      = {.offset = {.x = 0, .y = 0}, .extent = _screenSize},
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues    = clearValues.data()
    },
        vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);
    commandBuffer.bindVertexBuffers(0, {*_scene->Vertices}, {0});
    commandBuffer.bindIndexBuffer(*_scene->Indices, 0, vk::IndexType::eUint32);

    for (std::size_t i = 0; i < _gltfScene->m_nodes.size(); ++i)
    {
        const nvh::GltfNode&     node = _gltfScene->m_nodes[i];
        const nvh::GltfPrimMesh& mesh = _gltfScene->m_primMeshes[node.primMesh];

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *_pipelineLayout,
            0,
            {
                *_descriptorSet,
                *_texturesDescriptors[mesh.materialIndex],
            },
            {
                static_cast<uint32_t>(i * sizeof(shader::ModelMatrices)),
                static_cast<uint32_t>(mesh.materialIndex * sizeof(shader::MaterialUniforms)),
            });

        commandBuffer.drawIndexed(mesh.indexCount,
                                  1,
                                  mesh.firstIndex,
                                  static_cast<uint32_t>(mesh.vertexOffset),
                                  0);
    }

    commandBuffer.endRenderPass();
}

void BasePass::initializeResourcesFor(const nvh::GltfScene& targetScene,
                                      const Scene&          buffers,
                                      vk::Device            device)
{
    _gltfScene = &targetScene;
    _scene     = &buffers;

    std::vector<vk::WriteDescriptorSet> bufferWrite;

    vk::DescriptorBufferInfo uniformBufferInfo {
        .buffer = *UniformBuffer,
        .offset = 0,
        .range  = sizeof(Uniforms),
    };

    bufferWrite.push_back({
        .dstSet          = *_descriptorSet,
        .dstBinding      = 0,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo     = &uniformBufferInfo,
    });

    vk::DescriptorBufferInfo matricesBufferInfo {
        .buffer = *buffers.Matrices,
        .offset = 0,
        .range  = sizeof(shader::ModelMatrices),
    };

    bufferWrite.push_back({
        .dstSet          = *_descriptorSet,
        .dstBinding      = 1,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eUniformBufferDynamic,
        .pBufferInfo     = &matricesBufferInfo,
    });

    vk::DescriptorBufferInfo materialsBufferInfo {
        .buffer = *buffers.Materials,
        .offset = 0,
        .range  = sizeof(shader::MaterialUniforms),
    };

    bufferWrite.push_back({
        .dstSet          = *_descriptorSet,
        .dstBinding      = 2,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eUniformBufferDynamic,
        .pBufferInfo     = &materialsBufferInfo,
    });

    std::vector<vk::DescriptorImageInfo> materialTextureInfo(buffers.Textures.size());
    vk::DescriptorImageInfo defaultNormalInfo = buffers.DefaultNormalTexture.getDescriptorInfo();
    vk::DescriptorImageInfo defaultWhiteInfo  = buffers.DefaultWhiteTexture.getDescriptorInfo();
    for (std::size_t i = 0; i < buffers.Textures.size(); ++i)
    {
        materialTextureInfo[i] = buffers.Textures[i].getDescriptorInfo();
    }
    for (std::size_t i = 0; i < targetScene.m_materials.size(); ++i)
    {
        vk::DescriptorSet        set = *_texturesDescriptors[i];
        const nvh::GltfMaterial& mat = targetScene.m_materials[i];

        switch (mat.shadingModel)
        {
            case METALLIC_ROUGHNESS: {
                bufferWrite.push_back({
                    .dstSet          = set,
                    .dstBinding      = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = mat.pbrBaseColorTexture >= 0
                                           ? &materialTextureInfo[mat.pbrBaseColorTexture]
                                           : &defaultWhiteInfo,
                });
                break;
            }
            case SPECULAR_GLOSSINESS: {
                bufferWrite.push_back({
                    .dstSet          = set,
                    .dstBinding      = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = mat.khrDiffuseTexture >= 0
                                           ? &materialTextureInfo[mat.khrDiffuseTexture]
                                           : &defaultWhiteInfo,
                });
                break;
            }
        }

        bufferWrite.push_back({
            .dstSet          = set,
            .dstBinding      = 1,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo      = mat.normalTexture >= 0 ? &materialTextureInfo[mat.normalTexture]
                                                      : &defaultNormalInfo,
        });

        switch (mat.shadingModel)
        {
            case METALLIC_ROUGHNESS: {
                bufferWrite.push_back({
                    .dstSet          = set,
                    .dstBinding      = 2,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = mat.pbrMetallicRoughnessTexture >= 0
                                           ? &materialTextureInfo[mat.pbrMetallicRoughnessTexture]
                                           : &defaultWhiteInfo,
                });
                break;
            }
            case SPECULAR_GLOSSINESS: {
                bufferWrite.push_back({
                    .dstSet          = set,
                    .dstBinding      = 2,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = mat.khrSpecularGlossinessTexture >= 0
                                           ? &materialTextureInfo[mat.khrSpecularGlossinessTexture]
                                           : &defaultWhiteInfo,
                });
                break;
            }
        }

        bufferWrite.push_back({
            .dstSet          = set,
            .dstBinding      = 3,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo      = mat.emissiveTexture >= 0 ? &materialTextureInfo[mat.emissiveTexture]
                                                        : &defaultWhiteInfo,
        });
    }

    device.updateDescriptorSets(bufferWrite, {});
}

void BasePass::createGraphicsPipeline(vk::Device device)
{
    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages {
        {*_frag, *_vert}
    };

    std::array<vk::VertexInputBindingDescription, 1> vertexInputBindingStorage {{{
        .binding   = 0,
        .stride    = static_cast<uint32_t>(sizeof(Vertex)),
        .inputRate = vk::VertexInputRate::eVertex,
    }}};

    std::array<vk::VertexInputAttributeDescription, 5> vertexInputAttributeStorage {
        {{.location = 0,
          .binding  = 0,
          .format   = vk::Format::eR32G32B32Sfloat,
          .offset   = static_cast<uint32_t>(offsetof(Vertex, position))},

         {.location = 1,
          .binding  = 0,
          .format   = vk::Format::eR32G32B32Sfloat,
          .offset   = static_cast<uint32_t>(offsetof(Vertex, normal))},

         {.location = 2,
          .binding  = 0,
          .format   = vk::Format::eR32G32B32A32Sfloat,
          .offset   = static_cast<uint32_t>(offsetof(Vertex, tangent))},

         {.location = 3,
          .binding  = 0,
          .format   = vk::Format::eR32G32B32A32Sfloat,
          .offset   = static_cast<uint32_t>(offsetof(Vertex, color))},

         {.location = 4,
          .binding  = 0,
          .format   = vk::Format::eR32G32Sfloat,
          .offset   = static_cast<uint32_t>(offsetof(Vertex, uv))}}
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputState {
        .vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindingStorage.size()),
        .pVertexBindingDescriptions    = vertexInputBindingStorage.data(),
        .vertexAttributeDescriptionCount =
            static_cast<uint32_t>(vertexInputAttributeStorage.size()),
        .pVertexAttributeDescriptions = vertexInputAttributeStorage.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState {
        .topology = vk::PrimitiveTopology::eTriangleList};

    std::vector<vk::Viewport> viewportStorage {{{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(_screenSize.width),
        .height   = static_cast<float>(_screenSize.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    }}};

    std::vector<vk::Rect2D> scissorStorage {{{
        .offset = {.x = 0, .y = 0},
        .extent = _screenSize,
    }}};

    vk::PipelineViewportStateCreateInfo viewportState {
        .viewportCount = static_cast<uint32_t>(viewportStorage.size()),
        .pViewports    = viewportStorage.data(),
        .scissorCount  = static_cast<uint32_t>(scissorStorage.size()),
        .pScissors     = scissorStorage.data(),
    };

    vk::PipelineRasterizationStateCreateInfo rasterizationState {
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eCounterClockwise,
        .lineWidth   = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampleState {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencilState {
        .depthTestEnable  = true,
        .depthWriteEnable = true,
        .depthCompareOp   = vk::CompareOp::eLess,
    };

    std::vector<vk::PipelineColorBlendAttachmentState> attachmentColorBlendStorage {
        {{.blendEnable    = false,
          .colorWriteMask = vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
                            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB},
         {.blendEnable    = false,
          .colorWriteMask = vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
                            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB},
         {.blendEnable    = false,
          .colorWriteMask = vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
                            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB},
         {.blendEnable    = false,
          .colorWriteMask = vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
                            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB}}
    };

    vk::PipelineColorBlendStateCreateInfo colorBlendState {
        .attachmentCount = static_cast<uint32_t>(attachmentColorBlendStorage.size()),
        .pAttachments    = attachmentColorBlendStorage.data(),
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateInfo {};

    vk::Result result;
    std::tie(result, _pipeline) =
        device
            .createGraphicsPipelineUnique(nullptr,
                                          {.stageCount = static_cast<uint32_t>(shaderStages.size()),
                                           .pStages    = shaderStages.data(),
                                           .pVertexInputState   = &vertexInputState,
                                           .pInputAssemblyState = &inputAssemblyState,
                                           .pViewportState      = &viewportState,
                                           .pRasterizationState = &rasterizationState,
                                           .pMultisampleState   = &multisampleState,
                                           .pDepthStencilState  = &depthStencilState,
                                           .pColorBlendState    = &colorBlendState,
                                           .pDynamicState       = &dynamicStateInfo,
                                           .layout              = *_pipelineLayout,
                                           .renderPass          = *RenderPass,
                                           .subpass             = 0})
            .asTuple();

    if (result != vk::Result::eSuccess)
    {
        std::cout << "Failed to create graphics pipeline!" << std::endl;
        std::abort();
    }
}

void BasePass::createPass(vk::Device device)
{
    const Formats& formats = Formats::get();

    std::array<vk::AttachmentDescription, 5> attachments {
        {
         {.format         = formats.Albedo,
             .samples        = vk::SampleCountFlagBits::e1,
             .loadOp         = vk::AttachmentLoadOp::eClear,
             .storeOp        = vk::AttachmentStoreOp::eStore,
             .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
             .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
             .initialLayout  = vk::ImageLayout::eUndefined,
             .finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal},

         {.format         = formats.Normal,
             .samples        = vk::SampleCountFlagBits::e1,
             .loadOp         = vk::AttachmentLoadOp::eClear,
             .storeOp        = vk::AttachmentStoreOp::eStore,
             .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
             .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
             .initialLayout  = vk::ImageLayout::eUndefined,
             .finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal},

         {.format         = formats.MaterialProperties,
             .samples        = vk::SampleCountFlagBits::e1,
             .loadOp         = vk::AttachmentLoadOp::eClear,
             .storeOp        = vk::AttachmentStoreOp::eStore,
             .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
             .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
             .initialLayout  = vk::ImageLayout::eUndefined,
             .finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal},

         {.format         = formats.WorldPosition,
             .samples        = vk::SampleCountFlagBits::e1,
             .loadOp         = vk::AttachmentLoadOp::eClear,
             .storeOp        = vk::AttachmentStoreOp::eStore,
             .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
             .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
             .initialLayout  = vk::ImageLayout::eUndefined,
             .finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal},

         {.format         = formats.Depth,
             .samples        = vk::SampleCountFlagBits::e1,
             .loadOp         = vk::AttachmentLoadOp::eClear,
             .storeOp        = vk::AttachmentStoreOp::eStore,
             .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
             .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
             .initialLayout  = vk::ImageLayout::eUndefined,
             .finalLayout    = vk::ImageLayout::eShaderReadOnlyOptimal},
         }
    };

    std::array<vk::AttachmentReference, 4> colorAttachmentReferences {
        {{.attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal},
         {.attachment = 1, .layout = vk::ImageLayout::eColorAttachmentOptimal},
         {.attachment = 2, .layout = vk::ImageLayout::eColorAttachmentOptimal},
         {.attachment = 3, .layout = vk::ImageLayout::eColorAttachmentOptimal}}
    };

    const vk::AttachmentReference depthAttachmentReference {
        .attachment = 4,
        .layout     = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    const std::array<vk::SubpassDescription, 1> subpasses {{{
        .pipelineBindPoint       = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount    = static_cast<uint32_t>(colorAttachmentReferences.size()),
        .pColorAttachments       = colorAttachmentReferences.data(),
        .pDepthStencilAttachment = &depthAttachmentReference,
    }}};

    std::array<vk::SubpassDependency, 1> dependencies {{{
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = {},
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
    }}};

    RenderPass = device.createRenderPassUnique({
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments    = attachments.data(),
        .subpassCount    = static_cast<uint32_t>(subpasses.size()),
        .pSubpasses      = subpasses.data(),
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies   = dependencies.data(),
    });
}
