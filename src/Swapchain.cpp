#include "Swapchain.h"

Swapchain::Swapchain(vk::Device         device,
                     vk::PhysicalDevice physicalDevice,
                     vk::SurfaceKHR     surface,
                     GLFWwindow*        window)
{
    vk::SurfaceCapabilitiesKHR capabilities  = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    vk::SurfaceFormatKHR       surfaceFormat = chooseSurfaceFormat(physicalDevice, surface);

    ScreenFormat = surfaceFormat.format;
    ScreenSize   = chooseSwapExtent(capabilities, window);

    UniqueSwapchain = device.createSwapchainKHRUnique({
        .surface          = surface,
        .minImageCount    = chooseImageCount(capabilities),
        .imageFormat      = surfaceFormat.format,
        .imageColorSpace  = surfaceFormat.colorSpace,
        .imageExtent      = ScreenSize,
        .imageArrayLayers = 1,
        .imageUsage =
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform     = capabilities.currentTransform,
        .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode      = choosePresentMode(physicalDevice, surface),
        .clipped          = true,
    });
    Images          = device.getSwapchainImagesKHR(*UniqueSwapchain);
}

std::vector<Swapchain::BufferSet> Swapchain::getBuffers(vk::Device       device,
                                                        vk::RenderPass   renderPass,
                                                        vk::CommandPool  commandPool,
                                                        ResourceManager& allocator) const
{
    std::vector<BufferSet> result(Images.size());

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i].imageView = allocator.createImageView2D(device,
                                                          Images[i],
                                                          ScreenFormat,
                                                          vk::ImageAspectFlagBits::eColor);

        std::vector<vk::ImageView> attachments {*result[i].imageView};
        result[i].framebuffer = device.createFramebufferUnique({
            .renderPass      = renderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .width           = ScreenSize.width,
            .height          = ScreenSize.height,
            .layers          = 1,
        });
    }

    std::vector<vk::UniqueCommandBuffer> commandBuffers = device.allocateCommandBuffersUnique({
        .commandPool        = commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(result.size()),
    });

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result[i].commandBuffer = std::move(commandBuffers[i]);
    }

    return result;
}

void Swapchain::reset()
{
    Images.clear();
    UniqueSwapchain.reset();
}

vk::SurfaceFormatKHR Swapchain::chooseSurfaceFormat(vk::PhysicalDevice device,
                                                    vk::SurfaceKHR     surface)
{
    std::vector<vk::SurfaceFormatKHR> available = device.getSurfaceFormatsKHR(surface);
    vk::SurfaceFormatKHR              desired {
                     .format     = vk::Format::eB8G8R8A8Srgb,
                     .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear,
    };
    if (std::find(available.begin(), available.end(), desired) != available.end())
    {
        return desired;
    }
    return available[0];
}

vk::PresentModeKHR Swapchain::choosePresentMode(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    std::vector<vk::PresentModeKHR> available = device.getSurfacePresentModesKHR(surface);
    if (std::find(available.begin(), available.end(), vk::PresentModeKHR::eMailbox) !=
        available.end())
    {
        return vk::PresentModeKHR::eMailbox;
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                                         GLFWwindow*                       window)
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D result {
        .width  = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
    };

    result.width  = std::clamp(result.width,
                              capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width);
    result.height = std::clamp(result.height,
                               capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return result;
}

uint32_t Swapchain::chooseImageCount(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount != 0)
    {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }
    return imageCount;
}