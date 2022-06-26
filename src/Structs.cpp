#include "Structs.h"

#include "TransientCommandBuffer.h"

bool    Formats::_initialized = false;
Formats Formats::_framebufferFormats;

void Formats::initialize(vk::PhysicalDevice physicalDevice)
{
    assert(!_initialized);
    _framebufferFormats.Albedo = findSupportedFormat({vk::Format::eR8G8B8A8Srgb},
                                                     physicalDevice,
                                                     vk::ImageTiling::eOptimal,
                                                     vk::FormatFeatureFlagBits::eColorAttachment);
    _framebufferFormats.Normal = findSupportedFormat({vk::Format::eR16G16B16Snorm,
                                                      vk::Format::eR16G16B16Sfloat,
                                                      vk::Format::eR16G16B16A16Snorm,
                                                      vk::Format::eR16G16B16A16Sfloat,
                                                      vk::Format::eR32G32B32Sfloat},
                                                     physicalDevice,
                                                     vk::ImageTiling::eOptimal,
                                                     vk::FormatFeatureFlagBits::eColorAttachment);
    _framebufferFormats.Depth  = findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        physicalDevice,
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    _framebufferFormats.MaterialProperties =
        findSupportedFormat({vk::Format::eR16G16Unorm},
                            physicalDevice,
                            vk::ImageTiling::eOptimal,
                            vk::FormatFeatureFlagBits::eColorAttachment);
    _framebufferFormats.WorldPosition =
        findSupportedFormat({vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32A32Sfloat},
                            physicalDevice,
                            vk::ImageTiling::eOptimal,
                            vk::FormatFeatureFlagBits::eColorAttachment);
    _framebufferFormats.DepthAspectFlags = vk::ImageAspectFlagBits::eDepth;
    if (_framebufferFormats.Depth != vk::Format::eD32Sfloat)
    {
        _framebufferFormats.DepthAspectFlags |= vk::ImageAspectFlagBits::eStencil;
    }
    _initialized = true;
}

vk::Format Formats::findSupportedFormat(const std::vector<vk::Format>& candidates,
                                        vk::PhysicalDevice             physicalDevice,
                                        vk::ImageTiling                requiredTiling,
                                        vk::FormatFeatureFlags         requiredFeatures)
{
    for (vk::Format fmt : candidates)
    {
        vk::FormatProperties   properties = physicalDevice.getFormatProperties(fmt);
        vk::FormatFeatureFlags features   = requiredTiling == vk::ImageTiling::eLinear
                                                ? properties.linearTilingFeatures
                                                : properties.optimalTilingFeatures;
        if ((features & requiredFeatures) == requiredFeatures)
        {
            return fmt;
        }
    }

    std::cout << "Failed to find format with tiling " << vk::to_string(requiredTiling)
              << " and features " << vk::to_string(requiredFeatures) << std::endl;
    std::abort();
}

const Formats& Formats::get()
{
    assert(_initialized);
    return _framebufferFormats;
}

Framebuffer::Framebuffer(ResourceManager&        allocator,
                         vk::Device              device,
                         vk::Extent2D            screenSize,
                         BasePass&               pass,
                         TransientCommandBuffer& transientCommandBuffer)
{
    resize(allocator, device, screenSize, pass, transientCommandBuffer);
}

void Framebuffer::resize(ResourceManager&        allocator,
                         vk::Device              device,
                         vk::Extent2D            screenSize,
                         BasePass&               pass,
                         TransientCommandBuffer& transientCommandBuffer)
{
    UniqueFramebuffer.reset();

    AlbedoView.reset();
    NormalView.reset();
    MaterialPropertiesView.reset();
    WorldPositionView.reset();
    DepthView.reset();

    AlbedoImage.reset();
    NormalImage.reset();
    MaterialPropertiesImage.reset();
    WorldPositionImage.reset();
    DepthImage.reset();

    const Formats& formats = Formats::get();

    AlbedoImage             = allocator.createImage2D(screenSize,
                                          formats.Albedo,
                                          vk::ImageUsageFlagBits::eColorAttachment |
                                              vk::ImageUsageFlagBits::eSampled);
    NormalImage             = allocator.createImage2D(screenSize,
                                          formats.Normal,
                                          vk::ImageUsageFlagBits::eColorAttachment |
                                              vk::ImageUsageFlagBits::eSampled);
    MaterialPropertiesImage = allocator.createImage2D(screenSize,
                                                      formats.MaterialProperties,
                                                      vk::ImageUsageFlagBits::eColorAttachment |
                                                          vk::ImageUsageFlagBits::eSampled);
    WorldPositionImage      = allocator.createImage2D(screenSize,
                                                 formats.WorldPosition,
                                                 vk::ImageUsageFlagBits::eColorAttachment |
                                                     vk::ImageUsageFlagBits::eSampled);
    DepthImage              = allocator.createImage2D(screenSize,
                                         formats.Depth,
                                         vk::ImageUsageFlagBits::eDepthStencilAttachment |
                                             vk::ImageUsageFlagBits::eSampled);

    AlbedoView             = allocator.createImageView2D(device,
                                             *AlbedoImage,
                                             formats.Albedo,
                                             vk::ImageAspectFlagBits::eColor);
    NormalView             = allocator.createImageView2D(device,
                                             *NormalImage,
                                             formats.Normal,
                                             vk::ImageAspectFlagBits::eColor);
    MaterialPropertiesView = allocator.createImageView2D(device,
                                                         *MaterialPropertiesImage,
                                                         formats.MaterialProperties,
                                                         vk::ImageAspectFlagBits::eColor);
    WorldPositionView      = allocator.createImageView2D(device,
                                                    *WorldPositionImage,
                                                    formats.WorldPosition,
                                                    vk::ImageAspectFlagBits::eColor);
    DepthView =
        allocator.createImageView2D(device, *DepthImage, formats.Depth, formats.DepthAspectFlags);

    std::array<vk::ImageView, 5> attachments {
        *AlbedoView,
        *NormalView,
        *MaterialPropertiesView,
        *WorldPositionView,
        *DepthView,
    };

    UniqueFramebuffer = device.createFramebufferUnique({
        .renderPass      = *pass.RenderPass,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments    = attachments.data(),
        .width           = screenSize.width,
        .height          = screenSize.height,
        .layers          = 1,
    });

    transitionFramebufferLayouts(transientCommandBuffer);
}

void Framebuffer::transitionFramebufferLayouts(TransientCommandBuffer& transientCommandBuffer)
{
    transientCommandBuffer.begin();

    ResourceManager::transitionImageLayout(*transientCommandBuffer,
                                           *AlbedoImage,
                                           Formats::get().Albedo,
                                           vk::ImageLayout::eUndefined,
                                           vk::ImageLayout::eShaderReadOnlyOptimal);
    ResourceManager::transitionImageLayout(*transientCommandBuffer,
                                           *NormalImage,
                                           Formats::get().Normal,
                                           vk::ImageLayout::eUndefined,
                                           vk::ImageLayout::eShaderReadOnlyOptimal);
    ResourceManager::transitionImageLayout(*transientCommandBuffer,
                                           *WorldPositionImage,
                                           Formats::get().WorldPosition,
                                           vk::ImageLayout::eUndefined,
                                           vk::ImageLayout::eShaderReadOnlyOptimal);
    ResourceManager::transitionImageLayout(*transientCommandBuffer,
                                           *DepthImage,
                                           Formats::get().Depth,
                                           vk::ImageLayout::eUndefined,
                                           vk::ImageLayout::eShaderReadOnlyOptimal);

    transientCommandBuffer.submitAndWait();
}