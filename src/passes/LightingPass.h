#pragma once

#include "../ResourceManager.h"
#include "../Shader.h"

class Framebuffer;
struct FramebufferData;
class Scene;

class LightingPass
{
public:
    LightingPass() = default;
    LightingPass(vk::Device                                      device,
                 vk::Format                                      format,
                 vk::DescriptorPool                              staticDescriptorPool,
                 ResourceManager&                                allocator,
                 std::array<FramebufferData, FRAMEBUFFER_COUNT>& framebufferData);

    void issueCommands(vk::CommandBuffer commandBuffer,
                       vk::Framebuffer   framebuffer,
                       vk::DescriptorSet lightingFrameDescriptorSet,
                       vk::Extent2D      screenSize) const;

    void initializeDescriptorSetFor(const Framebuffer& framebuffer,
                                    const Scene&       scene,
                                    vk::Buffer         uniformBuffer,
                                    vk::Buffer         reservoirBuffer,
                                    vk::DeviceSize     reservoirBufferSize,
                                    vk::Device         device,
                                    vk::DescriptorSet  set);

    UniqueBuffer      UniformBuffer;

    vk::UniqueRenderPass RenderPass;

private:
    Shader _vert;
    Shader _frag;

    vk::Format _swapchainFormat;

    vk::UniqueSampler             _sampler;
    vk::UniquePipelineLayout      _pipelineLayout;
    vk::UniqueDescriptorSetLayout _descriptorSetLayout;

    vk::UniquePipeline _pipeline;

    void createPass(vk::Device device);
    void createGraphicsPipeline(vk::Device device);
};
