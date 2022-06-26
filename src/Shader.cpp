#include "Shader.h"

#include "ResourceManager.h"

Shader::Shader(vk::Device                   device,
               const std::filesystem::path& path,
               const char*                  entry,
               vk::ShaderStageFlagBits      stage)
{
    std::vector<char> binary = ResourceManager::readFile(path);

    _module = device.createShaderModuleUnique({
        .codeSize = static_cast<uint32_t>(binary.size()),
        .pCode    = reinterpret_cast<const uint32_t*>(binary.data()),
    });

    _shaderInfo = {
        .stage  = stage,
        .module = *_module,
        .pName  = entry,
    };
}
