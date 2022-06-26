#pragma once

#include <filesystem>

class Shader
{
public:
    const vk::PipelineShaderStageCreateInfo operator*() const
    {
        return _shaderInfo;
    }

    Shader() = default;
    Shader(vk::Device                   device,
           const std::filesystem::path& path,
           const char*                  entry,
           vk::ShaderStageFlagBits      stage);

private:
    vk::UniqueShaderModule            _module;
    vk::PipelineShaderStageCreateInfo _shaderInfo;
};
