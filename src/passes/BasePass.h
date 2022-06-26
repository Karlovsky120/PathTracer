#pragma once

#include "../ResourceManager.h"
#include "../Shader.h"

#include <gltfscene.h>

class Scene;

class BasePass
{
public:
    struct Uniforms
    {
        nvmath::mat4 projectionViewMatrix;
    };

    BasePass() = default;
    BasePass(vk::Device            device,
             vk::Extent2D          extent,
             ResourceManager&      allocator,
             vk::DescriptorPool    staticDescriptorPool,
             vk::DescriptorPool    textureDescriptorPool,
             const nvh::GltfScene& gltfScene,
             const Scene&          scene);

    void issueCommands(vk::CommandBuffer  commandBuffer,
                       vk::Framebuffer    framebuffer) const;

    void onResized(vk::Device device, vk::Extent2D screenSizes);

    void initializeResourcesFor(const nvh::GltfScene&, const Scene&, vk::Device);

    vk::UniqueRenderPass RenderPass;

    UniqueBuffer UniformBuffer;

private:
    const nvh::GltfScene* _gltfScene = nullptr;
    const Scene*          _scene     = nullptr;

    vk::UniqueDescriptorSet              _descriptorSet;
    std::vector<vk::UniqueDescriptorSet> _texturesDescriptors;

    vk::UniqueDescriptorSetLayout _setLayout;
    vk::UniqueDescriptorSetLayout _textureDescriptorSetLayout;

    vk::Extent2D _screenSize;
    Shader       _vert;
    Shader       _frag;

    vk::UniquePipelineLayout _pipelineLayout;
    vk::UniquePipeline       _pipeline;

    void createPass(vk::Device device);
    void createGraphicsPipeline(vk::Device device);
};
