#include "LightingPass.h"

#include "../Scene.h"
#include "../Structs.h"
#include "BasePass.h"

LightingPass::LightingPass(vk::Device                                      device,
                           vk::Format                                      format,
                           vk::DescriptorPool                              staticDescriptorPool,
                           ResourceManager&                                allocator,
                           std::array<FramebufferData, FRAMEBUFFER_COUNT>& framebufferData)
    : _swapchainFormat(format)
{
    _vert = Shader(device, "shaders/lighting.vert.spv", "main", vk::ShaderStageFlagBits::eVertex);
    _frag = Shader(device, "shaders/lighting.frag.spv", "main", vk::ShaderStageFlagBits::eFragment);

    _sampler = allocator.createSampler(device,
                                       vk::Filter::eNearest,
                                       vk::Filter::eNearest,
                                       vk::SamplerMipmapMode::eNearest);

    std::array<vk::DescriptorSetLayoutBinding, 8> descriptorBindings {
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
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},

         {.binding         = 4,
          .descriptorType  = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},

         {.binding         = 5,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},

         {.binding         = 6,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment},

         {.binding         = 7,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eFragment}}
    };

    _descriptorSetLayout = device.createDescriptorSetLayoutUnique(
        {.bindingCount = static_cast<uint32_t>(descriptorBindings.size()),
         .pBindings    = descriptorBindings.data()});

    vk::DescriptorSetLayout descriptorLayout {*_descriptorSetLayout};
    _pipelineLayout = device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts    = &descriptorLayout,
    });

    createPass(device);
    createGraphicsPipeline(device);

    UniformBuffer = allocator.createTypedBuffer<shader::LightingPassUniforms>(
        1,
        vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    std::array<vk::DescriptorSetLayout, FRAMEBUFFER_COUNT> setLayouts;
    for (vk::DescriptorSetLayout& setLayout : setLayouts)
    {
        setLayout = *_descriptorSetLayout;
    }

    std::vector<vk::UniqueDescriptorSet> descriptorSets = device.allocateDescriptorSetsUnique({
        .descriptorPool     = staticDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts        = setLayouts.data(),
    });

    for (size_t i = 0; i < framebufferData.size(); ++i)
    {
        framebufferData[i].LightingPassDescriptorSet = std::move(descriptorSets[i]);
    }
}

void LightingPass::issueCommands(vk::CommandBuffer commandBuffer, vk::Framebuffer framebuffer,
                                 vk::DescriptorSet lightingFrameDescriptorSet, vk::Extent2D
                                     screenSize) const
{
    std::array<vk::ClearValue, 1> clearValues {
        {{.color = {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}}}}};

    commandBuffer.beginRenderPass(
        {
            .renderPass      = *RenderPass,
            .framebuffer     = framebuffer,
            .renderArea      = {.offset = {.x = 0, .y = 0}, .extent = screenSize},
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues    = clearValues.data()
    },
        vk::SubpassContents::eInline);

    commandBuffer.setViewport(0,
                              {
                                  {.x        = 0.0f,
                                   .y        = 0.0f,
                                   .width    = static_cast<float>(screenSize.width),
                                   .height   = static_cast<float>(screenSize.height),
                                   .minDepth = 0.0f,
                                   .maxDepth = 1.0f}
    });
    commandBuffer.setScissor(0,
                             {
                                 {.offset = {.x = 0, .y = 0}, .extent = screenSize}
    });

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     *_pipelineLayout,
                                     0,
                                     lightingFrameDescriptorSet,
                                     {});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *_pipeline);
    commandBuffer.draw(4, 1, 0, 0);

    commandBuffer.endRenderPass();
}

void LightingPass::initializeDescriptorSetFor(const Framebuffer& framebuffer,
                                              const Scene&       scene,
                                              vk::Buffer         uniformBuffer,
                                              vk::Buffer         reservoirBuffer,
                                              vk::DeviceSize     reservoirBufferSize,
                                              vk::Device         device,
                                              vk::DescriptorSet  set)
{
    vk::DescriptorImageInfo albedoInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.AlbedoView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo normalInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.NormalView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo materialPropertiesInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.MaterialPropertiesView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo worldPositionInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.WorldPositionView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorBufferInfo uniformInfo {
        .buffer = uniformBuffer,
        .offset = 0,
        .range  = sizeof(shader::LightingPassUniforms),
    };

    vk::DescriptorBufferInfo reservoirsInfo {
        .buffer = reservoirBuffer,
        .offset = 0,
        .range  = reservoirBufferSize,
    };

    vk::DescriptorBufferInfo pointLightsInfo {
        .buffer = *scene.PointLights,
        .offset = 0,
        .range  = scene.PointLightsSize,
    };

    vk::DescriptorBufferInfo triangleLightsInfo {
        .buffer = *scene.TriangleLights,
        .offset = 0,
        .range  = scene.TriangleLightsSize,
    };

    device.updateDescriptorSets(
        {
            {{.dstSet          = set,
              .dstBinding      = 0,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &albedoInfo},

             {.dstSet          = set,
              .dstBinding      = 1,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &normalInfo},

             {.dstSet          = set,
              .dstBinding      = 2,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &materialPropertiesInfo},

             {.dstSet          = set,
              .dstBinding      = 3,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &worldPositionInfo},

             {.dstSet          = set,
              .dstBinding      = 4,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo     = &uniformInfo},

             {.dstSet          = set,
              .dstBinding      = 5,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &reservoirsInfo},

             {.dstSet          = set,
              .dstBinding      = 6,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &pointLightsInfo},

             {.dstSet          = set,
              .dstBinding      = 7,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &triangleLightsInfo}}
    },
        {});
}

void LightingPass::createPass(vk::Device device)
{
    vk::AttachmentDescription colorAttachment {
        .format         = _swapchainFormat,
        .samples        = vk::SampleCountFlagBits::e1,
        .loadOp         = vk::AttachmentLoadOp::eClear,
        .storeOp        = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout  = vk::ImageLayout::eColorAttachmentOptimal,
        .finalLayout    = vk::ImageLayout::ePresentSrcKHR,
    };

    vk::AttachmentReference colorAttachmentReference {
        .attachment = 0,
        .layout     = vk::ImageLayout::eColorAttachmentOptimal,
    };

    vk::SubpassDescription subpass {
        .pipelineBindPoint    = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachmentReference,
    };

    vk::SubpassDependency dependency {
        .srcSubpass   = VK_SUBPASS_EXTERNAL,
        .dstSubpass   = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eLateFragmentTests,
        .dstStageMask  = vk::PipelineStageFlagBits::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    RenderPass = device.createRenderPassUnique({
        .attachmentCount = 1,
        .pAttachments    = &colorAttachment,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &dependency,
    });
}

void LightingPass::createGraphicsPipeline(vk::Device device)
{
    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages {
        {*_frag, *_vert}
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputState;

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState {
        .topology = vk::PrimitiveTopology::eTriangleStrip};

    vk::PipelineViewportStateCreateInfo viewportState {
        .viewportCount = 1,
        .scissorCount  = 1,
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

    vk::PipelineDepthStencilStateCreateInfo depthStencilState;

    std::vector<vk::PipelineColorBlendAttachmentState> attachmentColorBlendStorage {{{
        .blendEnable    = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eA | vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB,
    }}};

    vk::PipelineColorBlendStateCreateInfo colorBlendState {
        .attachmentCount = static_cast<uint32_t>(attachmentColorBlendStorage.size()),
        .pAttachments    = attachmentColorBlendStorage.data(),
    };

    std::array<vk::DynamicState, 2> dynamicStates {
        {
         vk::DynamicState::eViewport,
         vk::DynamicState::eScissor,
         }
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateInfo {
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data()};

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
