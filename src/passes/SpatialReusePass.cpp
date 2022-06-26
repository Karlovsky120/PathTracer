#include "SpatialReusePass.h"

#include "../ShaderInclude.h"
#include "../Structs.h"
#include "BasePass.h"

SpatialReusePass::SpatialReusePass(vk::Device         device,
                                   vk::DescriptorPool staticDescriptorPool,
                                   ResourceManager&   allocator,
                                   std::array<FramebufferData, FRAMEBUFFER_COUNT>& framebufferData)
{
    _shader =
        Shader(device, "shaders/spatialReuse.comp.spv", "main", vk::ShaderStageFlagBits::eCompute);

    _sampler = allocator.createSampler(device,
                                       vk::Filter::eNearest,
                                       vk::Filter::eNearest,
                                       vk::SamplerMipmapMode::eNearest);

    std::array<vk::DescriptorSetLayoutBinding, 8> bindings {
        {
         {
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 1,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 2,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 3,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 4,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 5,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 6,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, {
                .binding         = 7,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute,
            }, }
    };

    _descriptorLayout = device.createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data(),
    });

    vk::PushConstantRange range {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset     = 0,
        .size       = sizeof(uint32_t),
    };

    _pipelineLayout = device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                         .pSetLayouts    = &*_descriptorLayout,
                                                         .pushConstantRangeCount = 1,
                                                         .pPushConstantRanges    = &range});

    auto [result, pipeline] = device.createComputePipelineUnique(nullptr,
                                                                 {
                                                                     .stage  = *_shader,
                                                                     .layout = *_pipelineLayout,
                                                                 });

    if (result != vk::Result::eSuccess)
    {
        std::cout << "Failed to create compute pipeline!" << std::endl;
        std::abort();
    }

    _pipeline = std::move(pipeline);

    std::array<vk::DescriptorSetLayout, FRAMEBUFFER_COUNT> setLayouts;
    for (vk::DescriptorSetLayout& setLayout : setLayouts)
    {
        setLayout = *_descriptorLayout;
    }

    std::vector<vk::UniqueDescriptorSet> spatialReuseSets = device.allocateDescriptorSetsUnique({
        .descriptorPool     = staticDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts        = setLayouts.data(),
    });

    std::vector<vk::UniqueDescriptorSet> spatialReuseSecondSets =
        device.allocateDescriptorSetsUnique({
            .descriptorPool     = staticDescriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts        = setLayouts.data(),
        });

    for (size_t i = 0; i < framebufferData.size(); ++i)
    {
        framebufferData[i].SpatialReuseDescriptor       = std::move(spatialReuseSets[i]);
        framebufferData[i].SpatialReuseSecondDescriptor = std::move(spatialReuseSecondSets[i]);
    }
}

void SpatialReusePass::issueCommands(vk::CommandBuffer buffer,
                                     vk::DescriptorSet spatialReuseFrameDescriptor,
                                     vk::Extent2D      screenSize)
{
    buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eComputeShader,
                           {},
                           {},
                           {},
                           {});
    buffer.bindPipeline(vk::PipelineBindPoint::eCompute, *_pipeline);
    buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              *_pipelineLayout,
                              0,
                              {spatialReuseFrameDescriptor},
                              {});

    uint32_t newRandom = _previousRandom + _random;
    _previousRandom    = _random;
    _random            = newRandom;

    buffer.pushConstants(*_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t),
                         &_random);
    buffer.dispatch(ceilDiv(screenSize.width, 8), ceilDiv(screenSize.height, 8), 1);
}

void SpatialReusePass::initializeDescriptorSetFor(const Framebuffer& framebuffer,
                                                  vk::Buffer         uniformBuffer,
                                                  vk::Buffer         reservoirBuffer,
                                                  vk::DeviceSize     reservoirBufferSize,
                                                  vk::Buffer         resultReservoirBuffer,
                                                  vk::Device         device,
                                                  vk::DescriptorSet  set)
{
    vk::DescriptorBufferInfo uniformInfo {
        .buffer = uniformBuffer,
        .offset = 0,
        .range  = sizeof(shader::RestirUniforms),
    };

    vk::DescriptorImageInfo worldPosImageInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.WorldPositionView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo albedoImageInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.AlbedoView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo normalImageInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.NormalView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    vk::DescriptorImageInfo materialImageInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.MaterialPropertiesView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo depthImageInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.DepthView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorBufferInfo reservoirInfo {
        .buffer = reservoirBuffer,
        .offset = 0,
        .range  = reservoirBufferSize,
    };

    vk::DescriptorBufferInfo resultReservoirInfo {
        .buffer = resultReservoirBuffer,
        .offset = 0,
        .range  = reservoirBufferSize,
    };

    device.updateDescriptorSets(
        {
            {{.dstSet          = set,
              .dstBinding      = 0,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eUniformBuffer,
              .pBufferInfo     = &uniformInfo},

             {.dstSet          = set,
              .dstBinding      = 1,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &worldPosImageInfo},

             {.dstSet          = set,
              .dstBinding      = 2,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &albedoImageInfo},

             {.dstSet          = set,
              .dstBinding      = 3,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &normalImageInfo},

             {.dstSet          = set,
              .dstBinding      = 4,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &materialImageInfo},

             {.dstSet          = set,
              .dstBinding      = 5,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &depthImageInfo},

             {.dstSet          = set,
              .dstBinding      = 6,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &reservoirInfo},

             {.dstSet          = set,
              .dstBinding      = 7,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &resultReservoirInfo}}
    },
        {});
}

constexpr uint32_t SpatialReusePass::ceilDiv(uint32_t a, uint32_t b) const
{
    return (a + b - 1) / b;
}