#pragma once

#include "../ResourceManager.h"
#include "../Shader.h"

class Framebuffer;
struct FramebufferData;
class Scene;

class RestirPass
{
public:
    RestirPass() = default;
    RestirPass(vk::Device                                      device,
               vk::PhysicalDevice                              physicalDevice,
               vk::DescriptorPool                              staticDescriptorPool,
               ResourceManager&                                allocator,
               std::array<FramebufferData, FRAMEBUFFER_COUNT>& framebufferData);

    void issueCommands(vk::CommandBuffer commandBuffer,
                       vk::DescriptorSet restirFrameDescriptor,
                       vk::Extent2D      screenSize) const;

    void initializeStaticDescriptorSetFor(const Scene&      scene,
                                          vk::Buffer        uniformBuffer,
                                          vk::Device        device,
                                          vk::DescriptorSet set);

    void initializeFrameDescriptorSetFor(const Framebuffer& framebuffer,
                                         const Framebuffer& prevFrameFramebuffer,
                                         vk::Buffer         reservoirBuffer,
                                         vk::Buffer         prevFrameReservoirBuffer,
                                         vk::DeviceSize     reservoirBufferSize,
                                         vk::Device         device,
                                         vk::DescriptorSet  set);

public:
    vk::UniqueDescriptorSet RestirStaticDescriptor;

private:
    UniqueBuffer                      _shaderBindingTable;
    vk::StridedDeviceAddressRegionKHR _rayGenSBT;
    vk::StridedDeviceAddressRegionKHR _rayMissSBT;
    vk::StridedDeviceAddressRegionKHR _rayHitSBT;

    Shader _rayGen;
    Shader _rayChit;
    Shader _rayMiss;

    vk::UniqueSampler             _sampler;
    vk::UniquePipelineLayout      _pipelineLayout;
    vk::UniqueDescriptorSetLayout _staticDescriptorSetLayout;
    vk::UniqueDescriptorSetLayout _frameDescriptorSetLayout;
    vk::UniquePipeline            _rayTracingPipeline;

    void createShaderBindingTable(vk::Device&         device,
                                  vk::PhysicalDevice& physicalDevice,
                                  ResourceManager&    allocator);
};
