#include "Program.h"

#include <chrono>
#include <sstream>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Program::Program(const std::string& scene, uint32_t pointLightCount)
{
    initGlfw();

    createInstance();
    createSurface();
    createDevice();

    _swapchain = Swapchain(*_device, _physicalDevice, *_surface, _window);

    {
        vk::CommandPoolCreateInfo poolInfo;
        poolInfo.setQueueFamilyIndex(_queueIndex)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        _commandPool = _device->createCommandPoolUnique(poolInfo);
    }

    _allocator = ResourceManager(vulkanApiVersion, *_instance, _physicalDevice, *_device);

    _transientCommandBuffer = TransientCommandBuffer(*_device, _queue, _queueIndex);
    _scene = Scene(scene, _allocator, _transientCommandBuffer, *_device, pointLightCount);

    createDescriptorSets();

    Formats::initialize(_physicalDevice);

    _basePass = BasePass(*_device,
                         _swapchain.ScreenSize,
                         _allocator,
                         *_staticDescriptorPool,
                         *_textureDescriptorPool,
                         _scene.GltfScene,
                         _scene);

    for (FramebufferData& framebufferData : _framebufferData)
    {
        framebufferData.framebuffer = Framebuffer(_allocator,
                                                  *_device,
                                                  _swapchain.ScreenSize,
                                                  _basePass,
                                                  _transientCommandBuffer);
    }

    createUniformBuffer();

    _restirPass =
        RestirPass(*_device, _physicalDevice, *_staticDescriptorPool, _allocator, _framebufferData);
    _spatialReusePass =
        SpatialReusePass(*_device, *_staticDescriptorPool, _allocator, _framebufferData);

    updateRestirBuffers();

    _lightingPass = LightingPass(*_device,
                                 _swapchain.ScreenFormat,
                                 *_staticDescriptorPool,
                                 _allocator,
                                 _framebufferData);

    initializeLightingPassResources();

    std::vector<vk::UniqueCommandBuffer> commandBuffers =
        std::move(_device->allocateCommandBuffersUnique({
            .commandPool        = *_commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(FRAMEBUFFER_COUNT),
        }));

    for (size_t i = 0; i < FRAMEBUFFER_COUNT; ++i)
    {
        _framebufferData[i].MainCommandBuffer = std::move(commandBuffers[i]);
    }

    recordMainCommandBuffers();
    createSwapchainBuffers();

    createSyncObjects();

    vk::Extent2D extent = getWindowSize();
    _camera.AspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    _camera.update();
}

void Program::initGlfw()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    _window = glfwCreateWindow(1920, 1080, "PathTracer", nullptr, nullptr);

    glfwSetWindowUserPointer(_window, this);

    glfwSetMouseButtonCallback(_window, &Program::_onMouseButtonEvent);
    glfwSetCursorPosCallback(_window, &Program::_onMouseMoveEvent);
    glfwSetKeyCallback(_window, &Program::_onKeyEvent);
}

void Program::createInstance()
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init(
        _loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

    uint32_t                 count          = 0;
    const char**             glfwExtensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> requestedInstanceExtensions =
        std::vector<const char*>(glfwExtensions, glfwExtensions + count);
    requestedInstanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    requestedInstanceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    requestedInstanceExtensions.emplace_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    const std::vector<vk::ExtensionProperties> supportedInstanceExtensions =
        vk::enumerateInstanceExtensionProperties();

    for (const char* requestedInstanceExtension : requestedInstanceExtensions)
    {
        bool extensionSupported = false;
        for (vk::ExtensionProperties extensionProperty : supportedInstanceExtensions)
        {
            if (strcmp(extensionProperty.extensionName, requestedInstanceExtension) == 0)
            {
                extensionSupported = true;
                break;
            }
        }

        if (!extensionSupported)
        {
            std::cout << "Instance extension " << requestedInstanceExtension << " is not supported!"
                      << std::endl;
            std::abort();
        }
    }

#ifdef ENABLE_VALIDATION_LAYERS
    std::vector<const char*> requestedValidationLayers {
# ifdef ENABLE_VALIDATION
        "VK_LAYER_KHRONOS_validation",
# endif
# ifdef ENABLE_API_DUMP
        "VK_LAYER_LUNARG_api_dump",
# endif
    };

    const std::vector<vk::LayerProperties> supportedInstanceLayers =
        vk::enumerateInstanceLayerProperties();

    for (const char* requestedValidationLayer : requestedValidationLayers)
    {
        bool layerSupported = false;
        for (vk::LayerProperties layerProperty : vk::enumerateInstanceLayerProperties())
        {
            if (strcmp(layerProperty.layerName, requestedValidationLayer) == 0)
            {
                layerSupported = true;
                break;
            }
        }

        if (!layerSupported)
        {
            std::cout << "Extension " << requestedValidationLayer << " is not supported!"
                      << std::endl;
            std::abort();
        }
    }

    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoExt {
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugUtilsCallback};
#endif

    vk::ApplicationInfo appInfo {
        .pApplicationName = "PathTracer",
        .apiVersion       = VK_API_VERSION_1_3,
    };

    _instance = vk::createInstanceUnique({
#ifdef ENABLE_VALIDATION_LAYERS
        .pNext = &debugUtilsMessengerCreateInfoExt,
#endif
        .pApplicationInfo = &appInfo,
#ifdef ENABLE_VALIDATION_LAYERS
        .enabledLayerCount   = static_cast<uint32_t>(requestedValidationLayers.size()),
        .ppEnabledLayerNames = requestedValidationLayers.data(),
#endif
        .enabledExtensionCount   = static_cast<uint32_t>(requestedInstanceExtensions.size()),
        .ppEnabledExtensionNames = requestedInstanceExtensions.data(),
    });

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*_instance);

#ifdef ENABLE_VALIDATION_LAYERS
    _debugUtilsMessenger =
        _instance->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfoExt);
#endif
}

void Program::createSurface()
{
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(*_instance, _window, nullptr, &surface) != VK_SUCCESS)
    {
        std::cout << "Failed to create GLFW surface!" << std::endl;
        std::abort();
    }

    _surface = vk::UniqueSurfaceKHR(surface, *_instance);
}

void Program::createDevice()
{
    const std::array<const char*, 6> requestedDeviceExtensions {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    };

    std::vector<void*> featureStructs;

    for (const vk::PhysicalDevice& physicalDevice : _instance->enumeratePhysicalDevices())
    {
        const vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        if (properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu)
        {
            continue;
        }

        const std::vector<vk::ExtensionProperties> supportedDeviceExtensions =
            physicalDevice.enumerateDeviceExtensionProperties();

        bool physicalDeviceAdequate = true;
        for (const char* requestedDeviceExtension : requestedDeviceExtensions)
        {
            bool extensionSupported = false;
            for (vk::ExtensionProperties extensionProperty : supportedDeviceExtensions)
            {
                if (strcmp(extensionProperty.extensionName, requestedDeviceExtension) == 0)
                {
                    extensionSupported = true;
                    break;
                }
            }

            if (!extensionSupported)
            {
                physicalDeviceAdequate = false;
            }
        }

        if (physicalDeviceAdequate)
        {
            _physicalDevice = physicalDevice;
        }
    }

    if (!_physicalDevice)
    {
        std::cout << "No suitable GPU present!" << std::endl;
        std::abort();
    }

    auto queueFamilyProperties = _physicalDevice.getQueueFamilyProperties();

    for (std::size_t i = 0; i < queueFamilyProperties.size(); ++i)
    {
        const vk::QueueFamilyProperties& properties = queueFamilyProperties[i];

        if ((properties.queueFlags &
             (vk::QueueFlagBits::eGraphics & vk::QueueFlagBits::eCompute)) &&
            _physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *_surface))
        {
            _queueIndex = static_cast<uint32_t>(i);
        }
    }

    float                                    queuePriority = 1.0f;
    std::array<vk::DeviceQueueCreateInfo, 1> queueCreateInfos {{{
        .queueFamilyIndex = _queueIndex,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    }}};

    vk::StructureChain<vk::DeviceCreateInfo,
                       vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan12Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR>
        deviceCreateInfo {
            {.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
             .pQueueCreateInfos       = queueCreateInfos.data(),
             .enabledExtensionCount   = static_cast<uint32_t>(requestedDeviceExtensions.size()),
             .ppEnabledExtensionNames = requestedDeviceExtensions.data()},
            {.features {.samplerAnisotropy = true, .shaderInt64 = true}},
            {
             .bufferDeviceAddress = true,
             },
            {
             .maintenance4 = true,
             },
            {
             .rayTracingPipeline = true,
             },
            {
             .accelerationStructure = true,
             }
    };

    _device = _physicalDevice.createDeviceUnique(deviceCreateInfo.get<vk::DeviceCreateInfo>());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*_device);

    _queue = _device->getQueue(_queueIndex, 0);
}

void Program::createDescriptorSets()
{
    std::array<vk::DescriptorPoolSize, 6> staticPoolSizes {
        {{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 128},
         {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 128},
         {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 128},
         {.type = vk::DescriptorType::eUniformBufferDynamic, .descriptorCount = 128},
         {.type = vk::DescriptorType::eAccelerationStructureKHR, .descriptorCount = 128},
         {.type = vk::DescriptorType::eStorageImage, .descriptorCount = 128}}
    };

    _staticDescriptorPool = _device->createDescriptorPoolUnique({
        .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets       = 128,
        .poolSizeCount = static_cast<uint32_t>(staticPoolSizes.size()),
        .pPoolSizes    = staticPoolSizes.data(),
    });

    std::array<vk::DescriptorPoolSize, 1> texturePoolSizes {{{
        .type            = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = static_cast<uint32_t>(3 * _scene.GltfScene.m_materials.size()),
    }}};

    _textureDescriptorPool = _device->createDescriptorPoolUnique({
        .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets       = static_cast<uint32_t>(_scene.GltfScene.m_materials.size()),
        .poolSizeCount = static_cast<uint32_t>(texturePoolSizes.size()),
        .pPoolSizes    = texturePoolSizes.data(),
    });
}

void Program::createUniformBuffer()
{
    _restirUniformBuffer = _allocator.createTypedBuffer<shader::RestirUniforms>(
        1,
        vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto* restirUniforms = _restirUniformBuffer.mapAs<shader::RestirUniforms>();
    restirUniforms->screenSize =
        nvmath::uvec2(_swapchain.ScreenSize.width, _swapchain.ScreenSize.height);
    restirUniforms->frame                  = 0;
    restirUniforms->spatialPosThreshold    = _positionThreshold;
    restirUniforms->spatialNormalThreshold = _normalThreshold;
    restirUniforms->flags                  = 0;

    if (_enableVisibilityReuse)
    {
        restirUniforms->flags |= RESTIR_VISIBILITY_REUSE_FLAG;
    }

    if (_enableTemporalReuse)
    {
        restirUniforms->flags |= RESTIR_TEMPORAL_REUSE_FLAG;
    }
    restirUniforms->spatialNeighbors = _spatialReuseNeighbourCount;
    restirUniforms->spatialRadius    = 30.0f;
    _restirUniformBuffer.unmap();
    _restirUniformBuffer.flush();
}

void Program::createSyncObjects()
{
    _imageAvailableSemaphore.resize(maxFramesInFlight);
    _renderFinishedSemaphore.resize(maxFramesInFlight);
    _inFlightFences.resize(maxFramesInFlight);
    _inFlightImageFences.resize(_swapchainBuffers.size());
    for (std::size_t i = 0; i < maxFramesInFlight; ++i)
    {
        vk::SemaphoreCreateInfo semaphoreInfo;
        _imageAvailableSemaphore[i] = _device->createSemaphoreUnique(semaphoreInfo);
        _renderFinishedSemaphore[i] = _device->createSemaphoreUnique(semaphoreInfo);

        vk::FenceCreateInfo fenceInfo;
        fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);
        _inFlightFences[i] = _device->createFenceUnique(fenceInfo);
    }

    _mainFence = _device->createFenceUnique({.flags = vk::FenceCreateFlagBits::eSignaled});
}

void Program::mainLoop()
{
    glfwShowWindow(_window);

    std::size_t  currentPresentFrame     = 0;
    std::size_t  currentFrame            = 0;
    bool         needsResize             = false;
    vk::Extent2D windowSize              = getWindowSize();
    nvmath::mat4 prevFrameProjectionView = _camera.ProjectionViewMatrix;

    while (!glfwWindowShouldClose(_window))
    {
        glfwPollEvents();
        handleMovement();

        vk::Extent2D newWindowSize = getWindowSize();
        if (needsResize || newWindowSize != windowSize)
        {
            //_device->waitIdle();

            while (newWindowSize.width == 0 && newWindowSize.height == 0)
            {
                glfwWaitEvents();
                newWindowSize = getWindowSize();
            }

            windowSize = newWindowSize;

            _swapchainBuffers.clear();
            _swapchain.reset();

            _swapchain = Swapchain(*_device, _physicalDevice, *_surface, _window);

            currentFrame = 0;

            for (FramebufferData& concurrentFrameData : _framebufferData)
            {
                concurrentFrameData.framebuffer.resize(_allocator,
                                                       *_device,
                                                       _swapchain.ScreenSize,
                                                       _basePass,
                                                       _transientCommandBuffer);
            }

            _basePass.onResized(*_device, _swapchain.ScreenSize);

            auto* restirUniforms       = _restirUniformBuffer.mapAs<shader::RestirUniforms>();
            restirUniforms->screenSize = nvmath::uvec2(windowSize.width, windowSize.height);
            restirUniforms->frame      = 0;
            _restirUniformBuffer.unmap();
            _restirUniformBuffer.flush();

            updateRestirBuffers();

            initializeLightingPassResources();

            recordMainCommandBuffers();
            createSwapchainBuffers();

            _camera.AspectRatio = static_cast<float>(_swapchain.ScreenSize.width) /
                                  static_cast<float>(_swapchain.ScreenSize.height);
            _camera.update();
            _cameraUpdated = true;
        }

        std::chrono::high_resolution_clock::time_point now =
            std::chrono::high_resolution_clock::now();

        float frameTime    = std::chrono::duration<float>(now - _previousFrameTime).count();
        _previousFrameTime = now;

        _smoothedFPS =
            _smoothedFPS * _FPSSmoothingFactor + (1.0f / frameTime) * (1.0f - _FPSSmoothingFactor);

        std::string title =
            std::string("PathTracer | FPS: " + std::to_string(_smoothedFPS) +
                        " | Frametime (ms): " + std::to_string(1000.0f / _smoothedFPS));
        glfwSetWindowTitle(_window, title.c_str());

        auto [result, imageIndex] =
            _device->acquireNextImageKHR(*_swapchain.UniqueSwapchain,
                                         std::numeric_limits<std::uint64_t>::max(),
                                         *_imageAvailableSemaphore[currentPresentFrame],
                                         nullptr);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
        {
            needsResize = true;
            continue;
        }

        while (_device->waitForFences(*_mainFence, true, std::numeric_limits<uint64_t>::max()) ==
               vk::Result::eTimeout)
        {}
        _device->resetFences(*_mainFence);

        auto* restirUniforms = _restirUniformBuffer.mapAs<shader::RestirUniforms>();
        ++restirUniforms->frame;
        restirUniforms->lightSampleCount              = _lightSampleCount;
        restirUniforms->prevFrameProjectionViewMatrix = prevFrameProjectionView;
        restirUniforms->temporalSampleCountMultiplier = _temporalReuseSampleMultiplier;

        if (_enableTemporalReuse)
        {
            restirUniforms->flags |= RESTIR_TEMPORAL_REUSE_FLAG;
        }
        else
        {
            restirUniforms->flags &= ~RESTIR_TEMPORAL_REUSE_FLAG;
        }

        if (_cameraUpdated || _viewParamChanged)
        {
            _queue.waitIdle();

            auto* uniforms                 = _basePass.UniformBuffer.mapAs<BasePass::Uniforms>();
            uniforms->projectionViewMatrix = _camera.ProjectionViewMatrix;
            _basePass.UniformBuffer.unmap();
            _basePass.UniformBuffer.flush();

            restirUniforms->cameraPos = _camera.Position;

            auto* lightingPassUniforms =
                _lightingPass.UniformBuffer.mapAs<shader::LightingPassUniforms>();
            lightingPassUniforms->cameraPos = _camera.Position;
            lightingPassUniforms->bufferSize =
                nvmath::uvec2(_swapchain.ScreenSize.width, _swapchain.ScreenSize.height);
            lightingPassUniforms->gamma = _gamma;
            _lightingPass.UniformBuffer.unmap();
            _lightingPass.UniformBuffer.flush();

            _cameraUpdated    = false;
            _viewParamChanged = false;
        }
        else if (_renderPathChanged)
        {
            _device->waitIdle();
            updateRestirBuffers();
            recordMainCommandBuffers();
            initializeLightingPassResources();

            restirUniforms->frame                  = 0;
            restirUniforms->spatialPosThreshold    = _positionThreshold;
            restirUniforms->spatialNormalThreshold = _normalThreshold;
            restirUniforms->flags |= RESTIR_VISIBILITY_REUSE_FLAG;

            _renderPathChanged = false;
        }

        _restirUniformBuffer.unmap();
        _restirUniformBuffer.flush();

        _queue.submit(
            {
                {.commandBufferCount = 1,
                 .pCommandBuffers    = &*_framebufferData[currentFrame].MainCommandBuffer}
        },
            *_mainFence);

        prevFrameProjectionView = _camera.ProjectionViewMatrix;

        while (_device->waitForFences({*_inFlightFences[currentPresentFrame]},
                                      true,
                                      std::numeric_limits<std::uint64_t>::max()) ==
               vk::Result::eTimeout)
            ;

        if (_inFlightImageFences[imageIndex])
        {
            while (_device->waitForFences({*_inFlightImageFences[imageIndex]},
                                          true,
                                          std::numeric_limits<std::uint64_t>::max()) ==
                   vk::Result::eTimeout)
                ;
        }
        _device->resetFences({*_inFlightFences[currentPresentFrame]});

        vk::CommandBuffer commandBuffer = *_swapchainBuffers[imageIndex].commandBuffer;
        vk::Framebuffer   frameBuffer   = *_swapchainBuffers[imageIndex].framebuffer;

        commandBuffer.begin(vk::CommandBufferBeginInfo());

        ResourceManager::transitionImageLayout(commandBuffer,
                                               _swapchain.Images[imageIndex],
                                               _swapchain.ScreenFormat,
                                               vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::eColorAttachmentOptimal);

        _lightingPass.issueCommands(commandBuffer,
                                    frameBuffer,
                                    *_framebufferData[currentFrame].LightingPassDescriptorSet,
                                    _swapchain.ScreenSize);

        commandBuffer.end();

        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        _queue.submit(
            {
                {.waitSemaphoreCount   = 1,
                 .pWaitSemaphores      = &*_imageAvailableSemaphore[currentPresentFrame],
                 .pWaitDstStageMask    = &waitStage,
                 .commandBufferCount   = 1,
                 .pCommandBuffers      = &*_swapchainBuffers[imageIndex].commandBuffer,
                 .signalSemaphoreCount = 1,
                 .pSignalSemaphores    = &*_renderFinishedSemaphore[currentPresentFrame]}
        },
            *_inFlightFences[currentPresentFrame]);

        try
        {
            if (_queue.presentKHR({
                    .waitSemaphoreCount = 1,
                    .pWaitSemaphores    = &*_renderFinishedSemaphore[currentPresentFrame],
                    .swapchainCount     = 1,
                    .pSwapchains        = &*_swapchain.UniqueSwapchain,
                    .pImageIndices      = &imageIndex,
                }) == vk::Result::eSuboptimalKHR)
            {
                needsResize = true;
            }
        }
        catch (const vk::OutOfDateKHRError&)
        {
            needsResize = true;
        }

        currentPresentFrame = (currentPresentFrame + 1) % maxFramesInFlight;
        currentFrame        = (currentFrame + 1) % FRAMEBUFFER_COUNT;
    }

    _device->waitIdle();
}

void Program::onMouseButtonEvent(int button, int action, int /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (_pressedMouseButton == -1)
        {
            _pressedMouseButton = button;
        }
    }
    else
    {
        if (button == _pressedMouseButton)
        {
            _pressedMouseButton = -1;
        }
    }
}

void Program::onMouseMoveEvent(double x, double y)
{
    if (_disableMouse)
    {
        return;
    }

    nvmath::vec2f newPos(static_cast<float>(x), static_cast<float>(y));
    nvmath::vec2f offset = newPos - _lastMouse;
    if (offset.norm() > 0.0001f)
    {
        nvmath::vec3f camOffset = _camera.Position - _camera.LookAt;

        nvmath::vec3f yAxis = nvmath::normalize(_camera.WorldUp);
        nvmath::vec3f xAxis = nvmath::normalize(nvmath::cross(yAxis, camOffset));
        nvmath::vec3f zAxis = nvmath::cross(xAxis, yAxis);

        nvmath::vec2f angles = offset * -1.0f * 0.005f;
        nvmath::vec2f vert(std::cos(angles.y), std::sin(angles.y));
        nvmath::vec2f hori(std::cos(angles.x), std::sin(angles.x));

        nvmath::vec3f angle(0.0f, nvmath::dot(camOffset, yAxis), nvmath::dot(camOffset, zAxis));
        float         newZ = angle.y * vert.y + angle.z * vert.x;
        if (newZ > 0.0f)
        {
            angle = nvmath::vec3f(0.0f, angle.y * vert.x - angle.z * vert.y, newZ);
        }
        angle = nvmath::vec3f(angle.z * hori.y, angle.y, angle.z * hori.x);

        _camera.Position = _camera.LookAt + xAxis * angle.x + yAxis * angle.y + zAxis * angle.z;

        _camera.update();
        _cameraUpdated = true;
    }

    _lastMouse = newPos;
}

void Program::onKeyEvent(int key, int, int action, int)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_Z: {
                _disableMouse = !_disableMouse;
                break;
            }

            case GLFW_KEY_O: {
                _lightSampleCount = std::clamp(_lightSampleCount >> 1, 1, 1024);
                std::cout << "Initial Light Samples set to: " << _lightSampleCount << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_P: {
                _lightSampleCount = std::clamp(_lightSampleCount << 1, 1, 1024);
                std::cout << "Initial Light Samples set to: " << _lightSampleCount << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_T: {
                _enableTemporalReuse = !_enableTemporalReuse;
                std::cout << "Temporal reuse set to: " << _enableTemporalReuse << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_V: {
                _enableVisibilityReuse = !_enableVisibilityReuse;
                std::cout << "Visibility reuse set to: " << _enableVisibilityReuse << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_SEMICOLON: {
                _spatialReuseNeighbourCount = std::clamp(_spatialReuseNeighbourCount - 1, 1, 100);
                std::cout << "Spatial reuse neighbour count set to: " << _spatialReuseNeighbourCount
                          << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_APOSTROPHE: {
                _spatialReuseNeighbourCount = std::clamp(_spatialReuseNeighbourCount + 1, 1, 100);
                std::cout << "Spatial reuse neighbour count set to: " << _spatialReuseNeighbourCount
                          << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_Y: {
                _temporalReuseSampleMultiplier =
                    std::clamp(_temporalReuseSampleMultiplier - 10, 0, 100);
                std::cout << "Temporal reuse sample multiplier set to: "
                          << _temporalReuseSampleMultiplier << std::endl;
                break;
            }

            case GLFW_KEY_U: {
                _temporalReuseSampleMultiplier =
                    std::clamp(_temporalReuseSampleMultiplier + 10, 0, 100);
                std::cout << "Temporal reuse sample multiplier set to: "
                          << _temporalReuseSampleMultiplier << std::endl;
                break;
            }

            case GLFW_KEY_H: {
                _spatialReuseIterations = std::clamp(_spatialReuseIterations - 1, 0, 10);
                std::cout << "Spatial reuse iterations set to: " << _spatialReuseIterations
                          << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_J: {
                _spatialReuseIterations = std::clamp(_spatialReuseIterations + 1, 0, 10);
                std::cout << "Spatial reuse iterations set to: " << _spatialReuseIterations
                          << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_N: {
                _positionThreshold = std::clamp(_positionThreshold - 0.1f, 0.0f, 1.0f);
                std::cout << "Depth threshold set to: " << _positionThreshold << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_M: {
                _positionThreshold = std::clamp(_positionThreshold + 0.1f, 0.0f, 1.0f);
                std::cout << "Depth threshold set to: " << _positionThreshold << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_COMMA: {
                _normalThreshold = std::clamp(_normalThreshold - 5.0f, 5.0f, 45.0f);
                std::cout << "Normal threshold set to: " << _normalThreshold << std::endl;
                _renderPathChanged = true;
                break;
            }

            case GLFW_KEY_PERIOD: {
                _normalThreshold = std::clamp(_normalThreshold + 5.0f, 5.0f, 45.0f);
                std::cout << "Normal threshold set to: " << _normalThreshold << std::endl;
                _renderPathChanged = true;
                break;
            }
        }
    }
}

void Program::_onMouseMoveEvent(GLFWwindow* window, double x, double y)
{
    static_cast<Program*>(glfwGetWindowUserPointer(window))->onMouseMoveEvent(x, y);
}

void Program::_onMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
{
    static_cast<Program*>(glfwGetWindowUserPointer(window))
        ->onMouseButtonEvent(button, action, mods);
}

void Program::_onKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    static_cast<Program*>(glfwGetWindowUserPointer(window))
        ->onKeyEvent(key, scancode, action, mods);
}

vk::Extent2D Program::getWindowSize()
{
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    return {.width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height)};
}

void Program::createSwapchainBuffers()
{
    _swapchainBuffers.clear();
    _swapchainBuffers =
        _swapchain.getBuffers(*_device, *_lightingPass.RenderPass, *_commandPool, _allocator);
}

void Program::recordMainCommandBuffers()
{
    for (const FramebufferData& concurrentFameData : _framebufferData)
    {
        vk::CommandBufferBeginInfo beginInfo;
        concurrentFameData.MainCommandBuffer->begin(beginInfo);

        _basePass.issueCommands(*concurrentFameData.MainCommandBuffer,
                                *concurrentFameData.framebuffer.UniqueFramebuffer);

        _restirPass.issueCommands(*concurrentFameData.MainCommandBuffer,
                                  *concurrentFameData.RestirFrameDescriptor,
                                  _swapchain.ScreenSize);

        for (int32_t i = 0; i < _spatialReuseIterations; ++i)
        {
            _spatialReusePass.issueCommands(*concurrentFameData.MainCommandBuffer,
                                            *concurrentFameData.SpatialReuseDescriptor,
                                            _swapchain.ScreenSize);
        }

        concurrentFameData.MainCommandBuffer->end();
    }
}

void Program::updateRestirBuffers()
{
    uint32_t       numPixels           = _swapchain.ScreenSize.width * _swapchain.ScreenSize.height;
    vk::DeviceSize reservoirBufferSize = numPixels * sizeof(shader::Reservoir);
    {
        _transientCommandBuffer.begin();

        for (FramebufferData& concurrentFameData : _framebufferData)
        {
            concurrentFameData.ReservoirBuffer = _allocator.createTypedBuffer<shader::Reservoir>(
                numPixels,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_GPU_ONLY);
            _transientCommandBuffer->fillBuffer(*concurrentFameData.ReservoirBuffer,
                                                0,
                                                VK_WHOLE_SIZE,
                                                0);
        }

        _reservoirTemporaryBuffer = _allocator.createTypedBuffer<shader::Reservoir>(
            numPixels,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY);
        _transientCommandBuffer->fillBuffer(*_reservoirTemporaryBuffer, 0, VK_WHOLE_SIZE, 0);

        _transientCommandBuffer.submitAndWait();
    }

    _restirPass.initializeStaticDescriptorSetFor(_scene,
                                                 *_restirUniformBuffer,
                                                 *_device,
                                                 *_restirPass.RestirStaticDescriptor);

    for (std::size_t i = 0; i < FRAMEBUFFER_COUNT; ++i)
    {
        _restirPass.initializeFrameDescriptorSetFor(
            _framebufferData[i].framebuffer,
            _framebufferData[(i - 1) % FRAMEBUFFER_COUNT].framebuffer,
            *_framebufferData[i].ReservoirBuffer,
            *_framebufferData[(i - 1) % FRAMEBUFFER_COUNT].ReservoirBuffer,
            reservoirBufferSize,
            *_device,
            *_framebufferData[i].RestirFrameDescriptor);

        _spatialReusePass.initializeDescriptorSetFor(
            _framebufferData[i].framebuffer,
            *_restirUniformBuffer,
            *_framebufferData[i].ReservoirBuffer,
            reservoirBufferSize,
            *_framebufferData[(i + FRAMEBUFFER_COUNT - 1) % FRAMEBUFFER_COUNT].ReservoirBuffer,
            *_device,
            *_framebufferData[i].SpatialReuseDescriptor);

        _spatialReusePass.initializeDescriptorSetFor(
            _framebufferData[i].framebuffer,
            *_restirUniformBuffer,
            *_framebufferData[(i + FRAMEBUFFER_COUNT - 1) % FRAMEBUFFER_COUNT].ReservoirBuffer,
            reservoirBufferSize,
            *_framebufferData[i].ReservoirBuffer,
            *_device,
            *_framebufferData[i].SpatialReuseSecondDescriptor);
    }
}

void Program::initializeLightingPassResources()
{
    uint32_t       pixelCount          = _swapchain.ScreenSize.width * _swapchain.ScreenSize.height;
    vk::DeviceSize reservoirBufferSize = pixelCount * sizeof(shader::Reservoir);
    for (FramebufferData& concurrentFameData : _framebufferData)
    {
        _lightingPass.initializeDescriptorSetFor(concurrentFameData.framebuffer,
                                                 _scene,
                                                 *_lightingPass.UniformBuffer,
                                                 *concurrentFameData.ReservoirBuffer,
                                                 reservoirBufferSize,
                                                 *_device,
                                                 *concurrentFameData.LightingPassDescriptorSet);
    }
}

void Program::handleMovement()
{
    bool  cameraChanged    = false;
    float movementVelocity = _movementVelocity;
    if (glfwGetKey(_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        movementVelocity *= 10.0f;
    }
    if (glfwGetKey(_window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
    {
        movementVelocity /= 10.0f;
    }
    if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
    {
        _camera.Position += _camera.ForwardVec * movementVelocity;
        _camera.LookAt += _camera.ForwardVec * movementVelocity;
        cameraChanged = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
    {
        _camera.Position -= _camera.RightVec * movementVelocity;
        _camera.LookAt -= _camera.RightVec * movementVelocity;
        cameraChanged = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
    {
        _camera.Position -= _camera.ForwardVec * movementVelocity;
        _camera.LookAt -= _camera.ForwardVec * movementVelocity;
        cameraChanged = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
    {
        _camera.Position += _camera.RightVec * movementVelocity;
        _camera.LookAt += _camera.RightVec * movementVelocity;
        cameraChanged = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        _camera.Position += _camera.UpVec * movementVelocity;
        _camera.LookAt += _camera.UpVec * movementVelocity;
        cameraChanged = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        _camera.Position -= _camera.UpVec * movementVelocity;
        _camera.LookAt -= _camera.UpVec * movementVelocity;
        cameraChanged = true;
    }

    if (cameraChanged)
    {
        _camera.update();
        _cameraUpdated = true;
    }
}

Program::~Program()
{
    glfwDestroyWindow(_window);
    glfwTerminate();
}

#ifdef ENABLE_VALIDATION_LAYERS
VKAPI_ATTR VkBool32 VKAPI_CALL
Program::debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT,
                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                            void*)
{
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        std::cerr << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}
#endif