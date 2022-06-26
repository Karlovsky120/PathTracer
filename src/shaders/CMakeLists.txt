cmake_minimum_required(VERSION 3.21)
project(Shaders)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE ALL_SHADER_FILES LIST_DIRECTORIES false *)
file(GLOB COMPILABLE_SHADER_FILES LIST_DIRECTORIES false *)

list(REMOVE_ITEM ALL_SHADER_FILES ${PROJECT_SOURCE_DIR}/CMakeLists.txt)
list(REMOVE_ITEM COMPILABLE_SHADER_FILES ${PROJECT_SOURCE_DIR}/CMakeLists.txt)

foreach (GLSL ${COMPILABLE_SHADER_FILES})
	get_filename_component(FILE_NAME ${GLSL} NAME)
	set(SPIRV "${PROJECT_BINARY_DIR}/../../shaders/${FILE_NAME}.spv")

	add_custom_command(
			OUTPUT ${SPIRV}
			COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 ${GLSL} -o ${SPIRV}
			DEPENDS ${ALL_SHADER_FILES}
			VERBATIM
	)

	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach (GLSL)

add_custom_target(
		Shaders
		DEPENDS ${SPIRV_BINARY_FILES}
)
