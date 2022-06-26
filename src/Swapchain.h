#pragma once

#include "ResourceManager.h"

#include <vector>

class Swapchain
{
public:
    struct BufferSet
    {
        vk::UniqueImageView     imageView;
        vk::UniqueFramebuffer   framebuffer;
        vk::UniqueCommandBuffer commandBuffer;
    };

    Swapchain() = default;
    Swapchain(vk::Device         device,
              vk::PhysicalDevice physicalDevice,
              vk::SurfaceKHR     surface,
              GLFWwindow*        window);

    std::vector<BufferSet> getBuffers(vk::Device       device,
                                      vk::RenderPass   renderPass,
                                      vk::CommandPool  commandPool,
                                      ResourceManager& allocator) const;

    void reset();

    std::vector<vk::Image> Images;
    vk::UniqueSwapchainKHR UniqueSwapchain;
    vk::Format             ScreenFormat;
    vk::Extent2D           ScreenSize;

private:
    vk::SurfaceFormatKHR chooseSurfaceFormat(vk::PhysicalDevice device,
                                             vk::SurfaceKHR     surface);
    vk::PresentModeKHR   choosePresentMode(vk::PhysicalDevice device,
                                           vk::SurfaceKHR     surface);
    vk::Extent2D         chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                                          GLFWwindow*                       window);
    uint32_t             chooseImageCount(const vk::SurfaceCapabilitiesKHR& capabilities);
};
