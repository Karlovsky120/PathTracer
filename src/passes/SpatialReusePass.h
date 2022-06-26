#pragma once

#include "../ResourceManager.h"
#include "../Shader.h"

class Framebuffer;
struct FramebufferData;

class SpatialReusePass
{
public:
    SpatialReusePass() = default;
    SpatialReusePass(vk::Device                                      device,
                     vk::DescriptorPool                              staticDescriptorPool,
                     ResourceManager&                                allocator,
                     std::array<FramebufferData, FRAMEBUFFER_COUNT>& framebufferData);

    void issueCommands(vk::CommandBuffer buffer,
                       vk::DescriptorSet spatialReuseFrameDescriptor,
                       vk::Extent2D      screenSize);

    void initializeDescriptorSetFor(const Framebuffer& framebuffer,
                                    vk::Buffer         uniformBuffer,
                                    vk::Buffer         reservoirBuffer,
                                    vk::DeviceSize     reservoirBufferSize,
                                    vk::Buffer         resultReservoirBuffer,
                                    vk::Device         device,
                                    vk::DescriptorSet  set);

private:
    Shader                        _shader;
    vk::UniqueDescriptorSetLayout _descriptorLayout;
    vk::UniqueSampler             _sampler;

    uint32_t _random         = 1;
    uint32_t _previousRandom = 1;

    vk::UniquePipelineLayout _pipelineLayout;
    vk::UniquePipeline       _pipeline;

    constexpr uint32_t ceilDiv(uint32_t a, uint32_t b) const;
};
