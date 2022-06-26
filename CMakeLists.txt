cmake_minimum_required(VERSION 3.21)

project(PathTracer)

add_executable(${PROJECT_NAME} src/Structs.h)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

target_compile_options(${PROJECT_NAME}
	PRIVATE /W4 /permissive- /experimental:external /external:anglebrackets /external:W3)

target_sources(${PROJECT_NAME}
	PRIVATE
		"src/passes/BasePass.cpp"
		"src/passes/BasePass.h"
		"src/passes/LightingPass.cpp"
		"src/passes/LightingPass.h"
		"src/passes/RestirPass.cpp"
		"src/passes/RestirPass.h"
		"src/passes/SpatialReusePass.cpp"
		"src/passes/SpatialReusePass.h"
		"src/Camera.cpp"
		"src/Camera.h"
		"src/main.cpp"
		"src/Program.cpp"
		"src/Program.h"
		"src/ResourceManager.cpp"
		"src/ResourceManager.h"
		"src/Scene.cpp"
		"src/Scene.h"
		"src/Shader.cpp"
		"src/Shader.h"
		"src/ShaderInclude.h"
		"src/Structs.cpp"
		"src/Structs.h"
		"src/Swapchain.cpp"
		"src/Swapchain.h"
		"src/TransientCommandBuffer.cpp"
		"src/TransientCommandBuffer.h"
		)

find_package(Vulkan REQUIRED)
add_subdirectory("external/gltf/")
add_subdirectory("external/glfw/")

add_library(MikkTSpace
		"external/MikkTSpace/mikktspace.c"
		"external/MikkTSpace/mikktspace.h"
		)

add_subdirectory("src/shaders")
add_dependencies(${PROJECT_NAME} Shaders)

target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.h)

find_package(CUDAToolkit)
if(${CUDAToolkit_FOUND})
	target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw MikkTSpace gltf CUDA::cudart CUDA::cuda_driver)
else(${CUDAToolkit_FOUND})
	target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw MikkTSpace gltf)
endif(${CUDAToolkit_FOUND})

target_include_directories(${PROJECT_NAME}
	PRIVATE
		$ENV{CUDA_PATH}/include
		$ENV{OptiX_INSTALL_DIR}
		"external/nvmath/"
		"external/VulkanMemoryAllocator/include/"
		"external/tinygltf/")

if (ENABLE_VALIDATION)
	add_definitions(-DENABLE_VALIDATION=1)
endif (ENABLE_VALIDATION)

if (ENABLE_API_DUMP)
	add_definitions(-DENABLE_API_DUMP=1)
endif (ENABLE_API_DUMP)
