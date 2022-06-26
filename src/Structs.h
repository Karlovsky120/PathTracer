#pragma once

#include "passes/BasePass.h"

#include <nvmath_glsltypes.h>

class Framebuffer
{
public:
    Framebuffer() = default;
    Framebuffer(ResourceManager&        allocator,
                vk::Device              device,
                vk::Extent2D            screenSize,
                BasePass&               pass,
                TransientCommandBuffer& transientCommandBuffer);

    void resize(ResourceManager&        allocator,
                vk::Device              device,
                vk::Extent2D            screenSize,
                BasePass&               pass,
                TransientCommandBuffer& transientCommandBuffer);

public:
    vk::UniqueFramebuffer UniqueFramebuffer;

    UniqueImage AlbedoImage;
    UniqueImage NormalImage;
    UniqueImage MaterialPropertiesImage;
    UniqueImage WorldPositionImage;
    UniqueImage DepthImage;

    vk::UniqueImageView AlbedoView;
    vk::UniqueImageView NormalView;
    vk::UniqueImageView MaterialPropertiesView;
    vk::UniqueImageView WorldPositionView;
    vk::UniqueImageView DepthView;

private:
    void transitionFramebufferLayouts(TransientCommandBuffer& transientCommandBuffer);
};

struct FramebufferData
{
    Framebuffer framebuffer;

    vk::UniqueCommandBuffer MainCommandBuffer;

    UniqueBuffer            ReservoirBuffer;

    vk::UniqueDescriptorSet SpatialReuseDescriptor;
    vk::UniqueDescriptorSet SpatialReuseSecondDescriptor;
    vk::UniqueDescriptorSet LightingPassDescriptorSet;
    vk::UniqueDescriptorSet RestirFrameDescriptor;
    vk::UniqueDescriptorSet UnbiasedReusePassFrameDescriptor;
};

struct Formats
{
    vk::Format           Albedo;
    vk::Format           Normal;
    vk::Format           Depth;
    vk::Format           MaterialProperties;
    vk::Format           WorldPosition;
    vk::ImageAspectFlags DepthAspectFlags;

    static void           initialize(vk::PhysicalDevice);
    static vk::Format     findSupportedFormat(const std::vector<vk::Format>& candidates,
                                              vk::PhysicalDevice,
                                              vk::ImageTiling,
                                              vk::FormatFeatureFlags);
    static const Formats& get();

private:
    static bool    _initialized;
    static Formats _framebufferFormats;
};

struct Vertex
{
    nvmath::vec4 position;
    nvmath::vec4 normal;
    nvmath::vec4 tangent;
    nvmath::vec4 color;
    nvmath::vec2 uv;
};
