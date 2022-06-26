#include "RestirPass.h"

#include "../Scene.h"
#include "../Structs.h"
#include "BasePass.h"

RestirPass::RestirPass(vk::Device                                      device,
                       vk::PhysicalDevice                              physicalDevice,
                       vk::DescriptorPool                              staticDescriptorPool,
                       ResourceManager&                                allocator,
                       std::array<FramebufferData, FRAMEBUFFER_COUNT>& framebufferData)
{
    _sampler = allocator.createSampler(device,
                                       vk::Filter::eNearest,
                                       vk::Filter::eNearest,
                                       vk::SamplerMipmapMode::eNearest);

    _rayGen =
        Shader(device, "shaders/restir.rgen.spv", "main", vk::ShaderStageFlagBits::eRaygenKHR);
    _rayChit = Shader(device,
                      "shaders/visibility.rchit.spv",
                      "main",
                      vk::ShaderStageFlagBits::eClosestHitKHR);
    _rayMiss = Shader(device,
                      "shaders/visibility.rmiss.spv",
                      "main",
                      vk::ShaderStageFlagBits::eMissKHR);

    std::array<vk::DescriptorSetLayoutBinding, 5> staticBindings {
        {{.binding         = 0,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 1,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 2,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 3,
          .descriptorType  = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 4,
          .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}}
    };

    vk::DescriptorSetLayoutCreateInfo staticLayoutInfo;
    staticLayoutInfo.setBindings(staticBindings);
    _staticDescriptorSetLayout = device.createDescriptorSetLayoutUnique(staticLayoutInfo);

    std::array<vk::DescriptorSetLayoutBinding, 10> frameBindings {
        {{.binding         = 0,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 1,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 2,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 3,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 4,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 5,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 6,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 7,
          .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 8,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR},

         {.binding         = 9,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1,
          .stageFlags      = vk::ShaderStageFlagBits::eRaygenKHR}}
    };

    _frameDescriptorSetLayout = device.createDescriptorSetLayoutUnique({
        .bindingCount = static_cast<uint32_t>(frameBindings.size()),
        .pBindings    = frameBindings.data(),
    });

    std::array<vk::DescriptorSetLayout, 2> descriptorLayouts {*_staticDescriptorSetLayout,
                                                              *_frameDescriptorSetLayout};

    _pipelineLayout = device.createPipelineLayoutUnique({
        .setLayoutCount = static_cast<uint32_t>(descriptorLayouts.size()),
        .pSetLayouts    = descriptorLayouts.data(),
    });

    std::array<vk::RayTracingShaderGroupCreateInfoKHR, 3> shaderGroups {
        {{.type                            = vk::RayTracingShaderGroupTypeKHR::eGeneral,
          .generalShader                   = 0,
          .closestHitShader                = VK_SHADER_UNUSED_KHR,
          .anyHitShader                    = VK_SHADER_UNUSED_KHR,
          .intersectionShader              = VK_SHADER_UNUSED_KHR,
          .pShaderGroupCaptureReplayHandle = nullptr},
         {.type                            = vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
          .generalShader                   = VK_SHADER_UNUSED_KHR,
          .closestHitShader                = 1,
          .anyHitShader                    = VK_SHADER_UNUSED_KHR,
          .intersectionShader              = VK_SHADER_UNUSED_KHR,
          .pShaderGroupCaptureReplayHandle = nullptr},
         {.type                            = vk::RayTracingShaderGroupTypeKHR::eGeneral,
          .generalShader                   = 2,
          .closestHitShader                = VK_SHADER_UNUSED_KHR,
          .anyHitShader                    = VK_SHADER_UNUSED_KHR,
          .intersectionShader              = VK_SHADER_UNUSED_KHR,
          .pShaderGroupCaptureReplayHandle = nullptr}}
    };

    std::array<vk::PipelineShaderStageCreateInfo, 3> shaderStages {
        *_rayGen,
        *_rayChit,
        *_rayMiss,
    };

    vk::Result result;
    std::tie(result, _rayTracingPipeline) =
        device
            .createRayTracingPipelineKHRUnique(
                nullptr,
                nullptr,
                {
                    .stageCount                   = static_cast<uint32_t>(shaderStages.size()),
                    .pStages                      = shaderStages.data(),
                    .groupCount                   = static_cast<uint32_t>(shaderGroups.size()),
                    .pGroups                      = shaderGroups.data(),
                    .maxPipelineRayRecursionDepth = 1,
                    .layout                       = *_pipelineLayout,
                },
                nullptr)
            .asTuple();

    if (result != vk::Result::eSuccess)
    {
        std::cout << "Failed to create ray tracing pipeline!" << std::endl;
        std::abort();
    }

    RestirStaticDescriptor = std::move(device.allocateDescriptorSetsUnique({
        .descriptorPool     = staticDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &*_staticDescriptorSetLayout,
    })[0]);

    createShaderBindingTable(device, physicalDevice, allocator);

    std::array<vk::DescriptorSetLayout, FRAMEBUFFER_COUNT> setLayouts;
    for (vk::DescriptorSetLayout& setLayout : setLayouts)
    {
        setLayout = *_frameDescriptorSetLayout;
    }

    std::vector<vk::UniqueDescriptorSet> frameDescriptorSets = device.allocateDescriptorSetsUnique({
        .descriptorPool     = staticDescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts        = setLayouts.data(),
    });

    for (size_t i = 0; i < framebufferData.size(); ++i)
    {
        framebufferData[i].RestirFrameDescriptor = std::move(frameDescriptorSets[i]);
    }
}

void RestirPass::issueCommands(vk::CommandBuffer commandBuffer,
                               vk::DescriptorSet restirFrameDescriptor,
                               vk::Extent2D      screenSize) const
{
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                  vk::PipelineStageFlagBits::eAllCommands,
                                  {},
                                  {},
                                  {},
                                  {});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *_rayTracingPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
                                     *_pipelineLayout,
                                     0,
                                     {*RestirStaticDescriptor, restirFrameDescriptor},
                                     {});
    commandBuffer.traceRaysKHR(_rayGenSBT,
                               _rayMissSBT,
                               _rayHitSBT,
                               {},
                               screenSize.width,
                               screenSize.height,
                               1);
}

void RestirPass::initializeStaticDescriptorSetFor(const Scene&      scene,
                                                  vk::Buffer        uniformBuffer,
                                                  vk::Device        device,
                                                  vk::DescriptorSet set)
{
    vk::DescriptorBufferInfo pointLightBuffer {
        .buffer = *scene.PointLights,
        .offset = 0,
        .range  = scene.PointLightsSize,
    };

    vk::DescriptorBufferInfo triangleLightBuffer {
        .buffer = *scene.TriangleLights,
        .offset = 0,
        .range  = scene.TriangleLightsSize,
    };

    vk::DescriptorBufferInfo aliasTableBufferInfo {
        .buffer = *scene.AliasTable,
        .offset = 0,
        .range  = scene.AliasTableSize,
    };

    vk::DescriptorBufferInfo uniformBufferInfo {
        .buffer = uniformBuffer,
        .offset = 0,
        .range  = sizeof(shader::RestirUniforms),
    };

    std::array<vk::WriteDescriptorSet, 5> writeDescriptorSet {
        {{.dstSet          = set,
          .dstBinding      = 0,
          .descriptorCount = 1,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .pBufferInfo     = &pointLightBuffer},

         {.dstSet          = set,
          .dstBinding      = 1,
          .descriptorCount = 1,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .pBufferInfo     = &triangleLightBuffer},

         {.dstSet          = set,
          .dstBinding      = 2,
          .descriptorCount = 1,
          .descriptorType  = vk::DescriptorType::eStorageBuffer,
          .pBufferInfo     = &aliasTableBufferInfo},

         {.dstSet          = set,
          .dstBinding      = 3,
          .descriptorCount = 1,
          .descriptorType  = vk::DescriptorType::eUniformBuffer,
          .pBufferInfo     = &uniformBufferInfo}}
    };

    vk::StructureChain<vk::WriteDescriptorSet, vk::WriteDescriptorSetAccelerationStructureKHR>
        accelerationStructureWriteDescriptor {
            {
             .dstSet          = set,
             .dstBinding      = 4,
             .dstArrayElement = 0,
             .descriptorCount = 1,
             .descriptorType  = vk::DescriptorType::eAccelerationStructureKHR,
             },
            {
             .accelerationStructureCount = 1,
             .pAccelerationStructures    = &*scene.TLAS,
             }
    };

    writeDescriptorSet[4] = accelerationStructureWriteDescriptor.get<vk::WriteDescriptorSet>();

    device.updateDescriptorSets(writeDescriptorSet, {});
}

void RestirPass::initializeFrameDescriptorSetFor(const Framebuffer& framebuffer,
                                                 const Framebuffer& prevFrameFramebuffer,
                                                 vk::Buffer         reservoirBuffer,
                                                 vk::Buffer         prevFrameReservoirBuffer,
                                                 vk::DeviceSize     reservoirBufferSize,
                                                 vk::Device         device,
                                                 vk::DescriptorSet  set)
{
    vk::DescriptorImageInfo worldPositionInfo {
        .sampler     = *_sampler,
        .imageView   = *framebuffer.WorldPositionView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

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

    vk::DescriptorImageInfo previousWorldPositionInfo {
        .sampler     = *_sampler,
        .imageView   = *prevFrameFramebuffer.WorldPositionView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo previousAlbedoInfo {
        .sampler     = *_sampler,
        .imageView   = *prevFrameFramebuffer.AlbedoView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo previousNormalInfo {
        .sampler     = *_sampler,
        .imageView   = *prevFrameFramebuffer.NormalView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorImageInfo previousMaterialPropertiesInfo {
        .sampler     = *_sampler,
        .imageView   = *prevFrameFramebuffer.DepthView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::DescriptorBufferInfo reservoirInfo {
        .buffer = reservoirBuffer,
        .offset = 0,
        .range  = reservoirBufferSize,
    };

    vk::DescriptorBufferInfo prevReservoirInfo {
        .buffer = prevFrameReservoirBuffer,
        .offset = 0,
        .range  = reservoirBufferSize,
    };

    device.updateDescriptorSets(
        {
            {{.dstSet          = set,
              .dstBinding      = 0,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &worldPositionInfo},

             {.dstSet          = set,
              .dstBinding      = 1,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &albedoInfo},

             {.dstSet          = set,
              .dstBinding      = 2,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &normalInfo},

             {.dstSet          = set,
              .dstBinding      = 3,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &materialPropertiesInfo},

             {.dstSet          = set,
              .dstBinding      = 4,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &previousWorldPositionInfo},

             {.dstSet          = set,
              .dstBinding      = 5,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &previousAlbedoInfo},

             {.dstSet          = set,
              .dstBinding      = 6,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &previousNormalInfo},

             {.dstSet          = set,
              .dstBinding      = 7,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
              .pImageInfo      = &previousMaterialPropertiesInfo},

             {.dstSet          = set,
              .dstBinding      = 8,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &reservoirInfo},

             {.dstSet          = set,
              .dstBinding      = 9,
              .descriptorCount = 1,
              .descriptorType  = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo     = &prevReservoirInfo}}
    },
        {});
}

void RestirPass::createShaderBindingTable(vk::Device&         device,
                                          vk::PhysicalDevice& physicalDevice,
                                          ResourceManager&    allocator)
{
    vk::StructureChain<vk::PhysicalDeviceProperties2,
                       vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>
        physicalDeviceProperties {{}, {}};

    physicalDevice.getProperties2(&physicalDeviceProperties.get<vk::PhysicalDeviceProperties2>());

    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties =
        physicalDeviceProperties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    uint32_t shaderGroupSize        = 3;
    uint32_t shaderBindingTableSize = shaderGroupSize * rtProperties.shaderGroupBaseAlignment;

    _shaderBindingTable = allocator.createBuffer(
        {
            .size  = shaderBindingTableSize,
            .usage = vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        {.usage = VMA_MEMORY_USAGE_CPU_TO_GPU});

    uint8_t*             dstData = _shaderBindingTable.mapAs<uint8_t>();
    std::vector<uint8_t> shaderHandleStorage(shaderBindingTableSize);
    vk::Result           result = device.getRayTracingShaderGroupHandlesKHR(*_rayTracingPipeline,
                                                                  0,
                                                                  shaderGroupSize,
                                                                  shaderBindingTableSize,
                                                                  shaderHandleStorage.data());
    if (result != vk::Result::eSuccess)
    {
        std::cout << "Failed to create ray tracing pipeline!" << std::endl;
        std::abort();
    }

    for (uint32_t g = 0; g < shaderGroupSize; g++)
    {
        memcpy(dstData,
               shaderHandleStorage.data() + g * rtProperties.shaderGroupHandleSize,
               rtProperties.shaderGroupHandleSize);
        dstData += rtProperties.shaderGroupBaseAlignment;
    }

    _shaderBindingTable.unmap();

    vk::DeviceAddress sbtAddr = device.getBufferAddress({.buffer = *_shaderBindingTable});
    _rayGenSBT                = vk::StridedDeviceAddressRegionKHR({
                       .deviceAddress = sbtAddr,
                       .stride        = rtProperties.shaderGroupBaseAlignment,
                       .size          = rtProperties.shaderGroupBaseAlignment,
    });

    _rayHitSBT = vk::StridedDeviceAddressRegionKHR({
        .deviceAddress = sbtAddr + rtProperties.shaderGroupBaseAlignment,
        .stride        = rtProperties.shaderGroupBaseAlignment,
        .size          = rtProperties.shaderGroupBaseAlignment,
    });

    _rayMissSBT = vk::StridedDeviceAddressRegionKHR({
        .deviceAddress = sbtAddr + 2 * rtProperties.shaderGroupBaseAlignment,
        .stride        = rtProperties.shaderGroupBaseAlignment,
        .size          = rtProperties.shaderGroupBaseAlignment,
    });
}
