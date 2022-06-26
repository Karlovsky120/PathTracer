#pragma once

#if defined(ENABLE_VALIDATION) || defined(ENABLE_API_DUMP)
#define ENABLE_VALIDATION_LAYERS
#endif

#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#undef APIENTRY
#include "vulkan/vulkan.hpp"

#include <cassert>
#include <iostream>

#define FRAMEBUFFER_COUNT 2