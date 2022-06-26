#pragma once

#include "Camera.h"
#include "ResourceManager.h"
#include "Scene.h"
#include "Swapchain.h"
#include "TransientCommandBuffer.h"

#include "passes/BasePass.h"
#include "passes/LightingPass.h"
#include "passes/RestirPass.h"
#include "passes/SpatialReusePass.h"

#include "Structs.h"

class Program
{
public:
    Program(const std::string& scene, uint32_t pointLightCount);
    ~Program();

    void mainLoop();

private:
    void initGlfw();
    void createInstance();
    void createSurface();
    void createDevice();
    void createDescriptorSets();

    void createUniformBuffer();

    void createSyncObjects();

    constexpr static uint32_t    vulkanApiVersion  = VK_MAKE_VERSION(1, 3, 0);
    constexpr static std::size_t maxFramesInFlight = 2;

    GLFWwindow* _window;

    Camera _camera;

    uint32_t _queueIndex = 0;

    vk::Queue _queue;

    vk::DynamicLoader _loader;

    vk::UniqueInstance _instance;
#ifdef ENABLE_VALIDATION
    vk::UniqueDebugUtilsMessengerEXT _debugUtilsMessenger;
#endif
    vk::PhysicalDevice   _physicalDevice;
    vk::UniqueSurfaceKHR _surface;
    vk::UniqueDevice     _device;

    ResourceManager          _allocator;
    vk::UniqueCommandPool    _commandPool;
    vk::UniqueDescriptorPool _staticDescriptorPool;
    vk::UniqueDescriptorPool _textureDescriptorPool;
    TransientCommandBuffer   _transientCommandBuffer;

    Swapchain                         _swapchain;
    std::vector<Swapchain::BufferSet> _swapchainBuffers;

    std::array<FramebufferData, FRAMEBUFFER_COUNT> _framebufferData;

    UniqueBuffer _restirUniformBuffer;
    UniqueBuffer _reservoirTemporaryBuffer;

    BasePass          _basePass;
    RestirPass        _restirPass;
    SpatialReusePass  _spatialReusePass;
    LightingPass      _lightingPass;

    Scene _scene;

    std::vector<vk::UniqueSemaphore> _imageAvailableSemaphore;
    std::vector<vk::UniqueSemaphore> _renderFinishedSemaphore;
    std::vector<vk::UniqueFence>     _inFlightFences;
    std::vector<vk::UniqueFence>     _inFlightImageFences;

    vk::UniqueFence _mainFence;

    int32_t _lightSampleCount      = 32;
    bool    _enableVisibilityReuse = true;
    bool    _enableTemporalReuse   = true;

    int32_t _temporalReuseSampleMultiplier = 20;

    int32_t _spatialReuseIterations     = 1;
    int32_t _spatialReuseNeighbourCount = 4;
    float   _positionThreshold          = 0.1f;
    float   _normalThreshold            = 25.0f;

    float _gamma = 1.1f;

    bool _viewParamChanged  = false;
    bool _renderPathChanged = false;

    bool _disableMouse = false;

    std::chrono::high_resolution_clock::time_point _previousFrameTime;

    float       _smoothedFPS        = 100.0f;
    float       _movementVelocity   = 0.1f;
    const float _FPSSmoothingFactor = 0.98f;

    nvmath::vec2f _lastMouse;
    int32_t       _pressedMouseButton = -1;
    bool          _cameraUpdated      = true;

    static void _onMouseMoveEvent(GLFWwindow* window, double x, double y);
    static void _onMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
    static void _onKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);

    void onMouseMoveEvent(double x, double y);
    void onMouseButtonEvent(int button, int action, int mods);
    void onKeyEvent(int key, int scancode, int action, int mods);

    vk::Extent2D getWindowSize();

    void createSwapchainBuffers();
    void recordMainCommandBuffers();
    void updateRestirBuffers();
    void initializeLightingPassResources();

    void handleMovement();

#ifdef ENABLE_VALIDATION_LAYERS
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                       VkDebugUtilsMessageTypeFlagsEXT             messageType,
                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                       void*                                       pUserData);
#endif
};
