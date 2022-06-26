#pragma once

#include <gltfscene.h>

#include "ResourceManager.h"
#include "ShaderInclude.h"

#include <random>

class TransientCommandBuffer;

class Scene
{
public:
    struct SceneTexture
    {
        vk::UniqueImageView ImageView;
        vk::UniqueSampler   Sampler;
        UniqueImage         Image;

        vk::DescriptorImageInfo
        getDescriptorInfo(vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal) const
        {
            return {.sampler = *Sampler, .imageView = *ImageView, .imageLayout = layout};
        }
    };

    Scene() = default;
    Scene(const std::string&      filename,
          ResourceManager&        allocator,
          TransientCommandBuffer& transientCommandBuffer,
          vk::Device              device,
          uint32_t                pointLightCount);

    nvh::GltfScene GltfScene;

    UniqueBuffer Vertices;
    UniqueBuffer Indices;
    UniqueBuffer Matrices;
    UniqueBuffer Materials;
    UniqueBuffer PointLights;
    UniqueBuffer TriangleLights;
    UniqueBuffer AliasTable;

    std::vector<SceneTexture> Textures;
    SceneTexture              DefaultNormalTexture;
    SceneTexture              DefaultWhiteTexture;

    vk::DeviceSize PointLightsSize;
    vk::DeviceSize TriangleLightsSize;
    vk::DeviceSize AliasTableSize;

    vk::UniqueAccelerationStructureKHR TLAS;

private:
    UniqueBuffer                                    _tlasInstanceBuffer;
    std::vector<vk::UniqueAccelerationStructureKHR> _blases;
    std::vector<UniqueBuffer>                       _asAllocations;

    static UniqueBuffer createAccelerationStructureBuffer(vk::DeviceSize size, ResourceManager&
                                                                             allocator);

    static UniqueBuffer createScratchBuffer(vk::DeviceSize size, ResourceManager& allocator);

    static std::vector<shader::PointLight> collectPointLights(const nvh::GltfScene& scene);
    static std::vector<shader::PointLight> generateRandomPointLights(
        std::size_t                           count,
        const nvmath::vec3&                   min,
        const nvmath::vec3&                   max,
        std::uniform_real_distribution<float> distR = std::uniform_real_distribution<float>(0.0f,
                                                                                            1.0f),
        std::uniform_real_distribution<float> distG = std::uniform_real_distribution<float>(0.0f,
                                                                                            1.0f),
        std::uniform_real_distribution<float> distB = std::uniform_real_distribution<float>(0.0f,
                                                                                            1.0f));

    static std::vector<shader::TriangleLight> collectTriangleLights(const nvh::GltfScene& scene);

    static std::vector<shader::Bucket>
    createAliasTable(std::vector<shader::PointLight>&    pointLights,
                     std::vector<shader::TriangleLight>& triangleLights);

    template<typename T>
    constexpr T ceilDiv(T a, T b)
    {
        return (a + b - static_cast<T>(1)) / b;
    }

    template<typename Struct, typename PreArray>
    constexpr std::size_t alignPreArrayBlock()
    {
        return ceilDiv(sizeof(PreArray), alignof(Struct)) * alignof(Struct);
    }
};