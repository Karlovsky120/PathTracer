#include "Scene.h"

#include "ResourceManager.h"
#include "ShaderInclude.h"
#include "Structs.h"
#include "TransientCommandBuffer.h"

#include <gltfscene.h>
#include <nvmath_glsltypes.h>

#include <queue>

Scene::Scene(const std::string&      filename,
             ResourceManager&        allocator,
             TransientCommandBuffer& transientCommandBuffer,
             vk::Device              device,
             uint32_t                pointLightCount)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF context;
    std::string        warn;
    std::string        error;
    if (!context.LoadASCIIFromFile(&model, &error, &warn, filename))
    {
        std::cout << "Error while loading scene" << std::endl;
        std::abort();
    }

    GltfScene.importDrawableNodes(model,
                                  nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0 |
                                      nvh::GltfAttributes::Color_0 | nvh::GltfAttributes::Tangent);
    GltfScene.importMaterials(model);
    GltfScene.importTexutureImages(model);

    std::vector<shader::PointLight>    pointLights    = collectPointLights(GltfScene);
    std::vector<shader::TriangleLight> triangleLights = collectTriangleLights(GltfScene);

    if (pointLights.empty() && triangleLights.empty())
    {
        nvmath::vec3 min = nvmath::nv_min(GltfScene.m_dimensions.min, GltfScene.m_dimensions.max);
        nvmath::vec3 max = nvmath::nv_max(GltfScene.m_dimensions.min, GltfScene.m_dimensions.max);

        pointLights = generateRandomPointLights(pointLightCount, min, max);
    }

    std::vector<shader::Bucket> aliasTable = createAliasTable(pointLights, triangleLights);

    Vertices = allocator.createTypedBuffer<Vertex>(
        GltfScene.m_positions.size(),
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    Vertex* vertices = Vertices.mapAs<Vertex>();
    for (std::size_t i = 0; i < GltfScene.m_positions.size(); ++i)
    {
        Vertex& v  = vertices[i];
        v.position = GltfScene.m_positions[i];

        if (i < GltfScene.m_normals.size())
        {
            v.normal = GltfScene.m_normals[i];
        }

        if (i < GltfScene.m_colors0.size())
        {
            v.color = GltfScene.m_colors0[i];
        }
        else
        {
            v.color = nvmath::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        }

        if (i < GltfScene.m_texcoords0.size())
        {
            v.uv = GltfScene.m_texcoords0[i];
        }

        if (i < GltfScene.m_tangents.size())
        {
            v.tangent = GltfScene.m_tangents[i];
        }
    }
    Vertices.unmap();
    Vertices.flush();

    Indices = allocator.createTypedBuffer<int32_t>(
        GltfScene.m_indices.size(),
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    uint32_t* indices = Indices.mapAs<uint32_t>();
    for (std::size_t i = 0; i < GltfScene.m_indices.size(); ++i)
    {
        indices[i] = GltfScene.m_indices[i];
    }
    Indices.unmap();
    Indices.flush();

    Matrices =
        allocator.createTypedBuffer<shader::ModelMatrices>(GltfScene.m_nodes.size(),
                                                           vk::BufferUsageFlagBits::eUniformBuffer,
                                                           VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto* matrices = Matrices.mapAs<shader::ModelMatrices>();
    for (std::size_t i = 0; i < GltfScene.m_nodes.size(); ++i)
    {
        matrices[i].Transform = GltfScene.m_nodes[i].worldMatrix;
        matrices[i].TransformInverseTransposed =
            nvmath::transpose(nvmath::invert(matrices[i].Transform));
    }
    Matrices.unmap();
    Matrices.flush();

    Materials = allocator.createTypedBuffer<shader::MaterialUniforms>(
        GltfScene.m_materials.size(),
        vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto* mat_device = Materials.mapAs<shader::MaterialUniforms>();
    for (std::size_t i = 0; i < GltfScene.m_materials.size(); ++i)
    {
        const nvh::GltfMaterial&  mat    = GltfScene.m_materials[i];
        shader::MaterialUniforms& outMat = mat_device[i];

        outMat.emissiveFactor     = mat.emissiveFactor;
        outMat.shadingModel       = mat.shadingModel;
        outMat.alphaMode          = mat.alphaMode;
        outMat.alphaCutoff        = mat.alphaCutoff;
        outMat.normalTextureScale = mat.normalTextureScale;

        switch (outMat.shadingModel)
        {
            case METALLIC_ROUGHNESS:
                outMat.colorParam      = mat.pbrBaseColorFactor;
                outMat.materialParam.y = mat.pbrRoughnessFactor;
                outMat.materialParam.z = mat.pbrMetallicFactor;
                break;
            case SPECULAR_GLOSSINESS:
                outMat.colorParam      = mat.khrDiffuseFactor;
                outMat.materialParam   = mat.khrSpecularFactor;
                outMat.materialParam.w = mat.khrGlossinessFactor;
                break;
        }
    }
    Materials.unmap();
    Materials.flush();

    PointLightsSize = alignPreArrayBlock<shader::PointLight, int32_t>() +
                      sizeof(shader::PointLight) * pointLights.size();

    PointLights = allocator.createBuffer(static_cast<uint32_t>(PointLightsSize),
                                         vk::BufferUsageFlagBits::eStorageBuffer,
                                         VMA_MEMORY_USAGE_CPU_TO_GPU);

    int32_t* pointLightPtr = PointLights.mapAs<int32_t>();
    *pointLightPtr         = static_cast<int32_t>(pointLights.size());
    auto* ptLights =
        reinterpret_cast<shader::PointLight*>(reinterpret_cast<uintptr_t>(pointLightPtr) +
                                              alignPreArrayBlock<shader::PointLight, int32_t>());
    std::memcpy(ptLights, pointLights.data(), sizeof(shader::PointLight) * pointLights.size());
    PointLights.unmap();
    PointLights.flush();

    TriangleLightsSize = alignPreArrayBlock<shader::TriangleLight, int32_t>() +
                         sizeof(shader::TriangleLight) * triangleLights.size();

    TriangleLights = allocator.createBuffer(static_cast<uint32_t>(TriangleLightsSize),
                                            vk::BufferUsageFlagBits::eStorageBuffer,
                                            VMA_MEMORY_USAGE_CPU_TO_GPU);

    int32_t* triangleLightsPtr = TriangleLights.mapAs<int32_t>();
    *triangleLightsPtr         = static_cast<int32_t>(triangleLights.size());
    auto* triLights            = reinterpret_cast<shader::TriangleLight*>(
        reinterpret_cast<uintptr_t>(triangleLightsPtr) +
        alignPreArrayBlock<shader::TriangleLight, int32_t>());
    std::memcpy(triLights,
                triangleLights.data(),
                sizeof(shader::TriangleLight) * triangleLights.size());
    TriangleLights.unmap();
    TriangleLights.flush();

    AliasTableSize = alignPreArrayBlock<shader::Bucket, int32_t[4]>() +
                     sizeof(shader::Bucket) * aliasTable.size();

    AliasTable = allocator.createBuffer(static_cast<uint32_t>(AliasTableSize),
                                        vk::BufferUsageFlagBits::eStorageBuffer,
                                        VMA_MEMORY_USAGE_CPU_TO_GPU);

    int32_t* aliasTablePtr = AliasTable.mapAs<int32_t>();
    *aliasTablePtr         = static_cast<int32_t>(aliasTable.size());
    auto* aliasTableContentPtr =
        reinterpret_cast<shader::Bucket*>(reinterpret_cast<uintptr_t>(aliasTablePtr) +
                                          alignPreArrayBlock<shader::Bucket, int32_t[4]>());
    std::memcpy(aliasTableContentPtr,
                aliasTable.data(),
                sizeof(shader::Bucket) * aliasTable.size());
    AliasTable.unmap();
    AliasTable.flush();

    Textures.resize(GltfScene.m_textures.size());
    for (uint32_t i = 0; i < GltfScene.m_textures.size(); ++i)
    {
        tinygltf::Image& gltfImage = GltfScene.m_textures[i];

        uint32_t numMipLevels = 1 + static_cast<uint32_t>(std::ceil(
                                        std::log2(std::max(gltfImage.width, gltfImage.height))));

        Textures[i].Image = ResourceManager::loadTexture(gltfImage,
                                                         vk::Format::eR8G8B8A8Unorm,
                                                         numMipLevels,
                                                         allocator,
                                                         transientCommandBuffer);

        Textures[i].Sampler = allocator.createSampler(device,
                                                      vk::Filter::eLinear,
                                                      vk::Filter::eLinear,
                                                      vk::SamplerMipmapMode::eLinear,
                                                      16.0f);

        Textures[i].ImageView = allocator.createImageView2D(device,
                                                            *Textures[i].Image,
                                                            vk::Format::eR8G8B8A8Unorm,
                                                            vk::ImageAspectFlagBits::eColor,
                                                            0,
                                                            numMipLevels);
    }

    unsigned char defaultNormal[4] {255, 255, 255, 255};
    DefaultNormalTexture.Image     = ResourceManager::loadTexture(defaultNormal,
                                                              1,
                                                              1,
                                                              vk::Format::eR8G8B8A8Unorm,
                                                              1,
                                                              allocator,
                                                              transientCommandBuffer);
    DefaultNormalTexture.Sampler   = allocator.createSampler(device);
    DefaultNormalTexture.ImageView = allocator.createImageView2D(device,
                                                                 *DefaultNormalTexture.Image,
                                                                 vk::Format::eR8G8B8A8Unorm,
                                                                 vk::ImageAspectFlagBits::eColor);

    unsigned char defaultWhite[4] {255, 255, 255, 255};
    DefaultWhiteTexture.Image     = ResourceManager::loadTexture(defaultWhite,
                                                             1,
                                                             1,
                                                             vk::Format::eR8G8B8A8Unorm,
                                                             1,
                                                             allocator,
                                                             transientCommandBuffer);
    DefaultWhiteTexture.Sampler   = allocator.createSampler(device);
    DefaultWhiteTexture.ImageView = allocator.createImageView2D(device,
                                                                *DefaultWhiteTexture.Image,
                                                                vk::Format::eR8G8B8A8Unorm,
                                                                vk::ImageAspectFlagBits::eColor);

    _blases.resize(GltfScene.m_primMeshes.size());
    for (uint32_t i = 0; i < GltfScene.m_primMeshes.size(); ++i)
    {
        const nvh::GltfPrimMesh& primMesh = GltfScene.m_primMeshes[i];

        vk::AccelerationStructureGeometryTrianglesDataKHR triangles {
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData   = {.deviceAddress = device.getBufferAddress({.buffer = *Vertices})},
            .vertexStride = sizeof(Vertex),
            .maxVertex    = primMesh.vertexCount,
            .indexType    = vk::IndexType::eUint32,
            .indexData    = {.deviceAddress = device.getBufferAddress({.buffer = *Indices})},
        };

        vk::AccelerationStructureGeometryKHR blasAccelerationGeometry {
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry     = {.triangles = {.vertexFormat = vk::Format::eR32G32B32Sfloat,
                                           .vertexData   = {.deviceAddress = device.getBufferAddress(
                                                          {.buffer = *Vertices})},
                                           .vertexStride = sizeof(Vertex),
                                           .maxVertex    = primMesh.vertexCount,
                                           .indexType    = vk::IndexType::eUint32,
                                           .indexData    = {.deviceAddress = device.getBufferAddress(
                                                         {.buffer = *Indices})}}},
            .flags        = vk::GeometryFlagBitsKHR::eOpaque,
        };

        vk::AccelerationStructureBuildGeometryInfoKHR blasAccelerationBuildGeometryInfo {
            .type                     = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags                    = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .dstAccelerationStructure = *_blases.at(i),
            .geometryCount            = 1,
            .pGeometries              = &blasAccelerationGeometry,
        };

        vk::AccelerationStructureBuildSizesInfoKHR sizeInfo =
            device.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice,
                blasAccelerationBuildGeometryInfo,
                primMesh.indexCount / 3);

        UniqueBuffer blasScratchBuffer = createScratchBuffer(sizeInfo.buildScratchSize, allocator);

        blasAccelerationBuildGeometryInfo.scratchData.setDeviceAddress(
            device.getBufferAddress({.buffer = *blasScratchBuffer}));

        _asAllocations.emplace_back(
            createAccelerationStructureBuffer(sizeInfo.accelerationStructureSize, allocator));

        _blases.at(i) = device.createAccelerationStructureKHRUnique(
            {.buffer = *_asAllocations.back(),
             .offset = 0,
             .size   = sizeInfo.accelerationStructureSize,
             .type   = vk::AccelerationStructureTypeKHR::eBottomLevel},
            nullptr);

        blasAccelerationBuildGeometryInfo.setDstAccelerationStructure(*_blases[i]);

        vk::AccelerationStructureBuildRangeInfoKHR blasAccelerationBuildOffsetInfo {
            .primitiveCount  = primMesh.indexCount / 3,
            .primitiveOffset = static_cast<uint32_t>(primMesh.firstIndex * sizeof(uint32_t)),
            .firstVertex     = primMesh.vertexOffset,
            .transformOffset = 0};

        transientCommandBuffer.begin();
        transientCommandBuffer->buildAccelerationStructuresKHR(blasAccelerationBuildGeometryInfo,
                                                               {&blasAccelerationBuildOffsetInfo});
        transientCommandBuffer.submitAndWait();
    }

    std::vector<vk::AccelerationStructureInstanceKHR> tlasInstance;
    tlasInstance.reserve(GltfScene.m_nodes.size());
    for (auto& node : GltfScene.m_nodes)
    {
        vk::TransformMatrixKHR transformMatrix;
        for (std::size_t y = 0; y < 3; ++y)
        {
            for (std::size_t x = 0; x < 4; ++x)
            {
                transformMatrix.matrix[y][x] = node.worldMatrix.mat_array[x * 4 + y];
            }
        }

        tlasInstance.push_back(
            {.transform                              = transformMatrix,
             .instanceCustomIndex                    = static_cast<uint32_t>(node.primMesh),
             .mask                                   = 0xFF,
             .instanceShaderBindingTableRecordOffset = 0,
             .flags                                  = static_cast<VkGeometryInstanceFlagsKHR>(
                 vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable),
             .accelerationStructureReference = device.getAccelerationStructureAddressKHR(
                 {.accelerationStructure = *_blases[node.primMesh]})});
    }

    uint32_t instanceBufferByteLength =
        static_cast<uint32_t>(sizeof(vk::AccelerationStructureInstanceKHR) * tlasInstance.size());

    _tlasInstanceBuffer = allocator.createBuffer(
        {
            .size  = instanceBufferByteLength,
            .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress |
                     vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        {.usage = VMA_MEMORY_USAGE_CPU_TO_GPU});

    void* dstData = _tlasInstanceBuffer.map();
    memcpy(dstData, tlasInstance.data(), instanceBufferByteLength);
    _tlasInstanceBuffer.unmap();

    vk::AccelerationStructureGeometryKHR tlasAccelerationGeometry {
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry     = {.instances = {.arrayOfPointers = VK_FALSE,
                                       .data            = {.deviceAddress = device.getBufferAddress(
                                                {.buffer = *_tlasInstanceBuffer})}}},
        .flags        = vk::GeometryFlagBitsKHR::eOpaque,
    };

    vk::AccelerationStructureBuildGeometryInfoKHR tlasAccelerationBuildGeometryInfo {
        .type          = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .geometryCount = 1,
        .pGeometries   = &tlasAccelerationGeometry,
    };

    auto buildSize = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        tlasAccelerationBuildGeometryInfo,
        {static_cast<std::uint32_t>(tlasInstance.size())});

    _asAllocations.emplace_back(
        createAccelerationStructureBuffer(buildSize.accelerationStructureSize, allocator));

    TLAS = device.createAccelerationStructureKHRUnique(
        {.buffer = *_asAllocations.back(),
         .size   = buildSize.accelerationStructureSize,
         .type   = vk::AccelerationStructureTypeKHR::eTopLevel},
        nullptr);

    tlasAccelerationBuildGeometryInfo.setDstAccelerationStructure(*TLAS);

    UniqueBuffer tlasScratchBuffer = createScratchBuffer(buildSize.buildScratchSize, allocator);
    tlasAccelerationBuildGeometryInfo.scratchData.deviceAddress =
        device.getBufferAddress({.buffer = *tlasScratchBuffer});

    vk::AccelerationStructureBuildRangeInfoKHR tlasAccelerationBuildOffsetInfo {
        .primitiveCount  = static_cast<uint32_t>(tlasInstance.size()),
        .primitiveOffset = 0,
        .firstVertex     = 0,
        .transformOffset = 0,
    };

    transientCommandBuffer.begin();
    transientCommandBuffer->buildAccelerationStructuresKHR(tlasAccelerationBuildGeometryInfo,
                                                           &tlasAccelerationBuildOffsetInfo);
    transientCommandBuffer.submitAndWait();
}

UniqueBuffer Scene::createAccelerationStructureBuffer(vk::DeviceSize   size,
                                                      ResourceManager& allocator)
{
    return allocator.createBuffer(
        {
            .size  = size,
            .usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        {.usage = VMA_MEMORY_USAGE_GPU_ONLY});
}

UniqueBuffer Scene::createScratchBuffer(vk::DeviceSize size, ResourceManager& allocator)
{
    return allocator.createBuffer(
        {
            .size  = size,
            .usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                     vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        {.usage = VMA_MEMORY_USAGE_GPU_ONLY});
}

std::vector<shader::PointLight> Scene::collectPointLights(const nvh::GltfScene& scene)
{
    std::vector<shader::PointLight> pointLights;
    pointLights.reserve(scene.m_lights.size());
    for (const nvh::GltfLight& light : scene.m_lights)
    {
        pointLights.emplace_back(shader::PointLight {
            .pos = light.worldMatrix.col(3),
            .color_luminance =
                nvmath::vec4(light.light.color[0],
                             light.light.color[1],
                             light.light.color[2],
                             0.2126f * light.light.color[0] + 0.7152f * light.light.color[1] +
                                 0.0722f * light.light.color[2]),
        });
    }
    return pointLights;
}

std::vector<shader::PointLight>
Scene::generateRandomPointLights(std::size_t                           count,
                                 const nvmath::vec3&                   min,
                                 const nvmath::vec3&                   max,
                                 std::uniform_real_distribution<float> distR,
                                 std::uniform_real_distribution<float> distG,
                                 std::uniform_real_distribution<float> distB)
{
    std::uniform_real_distribution<float> distX(min.x, max.x);
    std::uniform_real_distribution<float> distY(min.y, max.y);
    std::uniform_real_distribution<float> distZ(min.z, max.z);
    std::default_random_engine            rand;

    std::vector<shader::PointLight> pointLights;
    pointLights.reserve(count);

    for (uint32_t i = 0; i < count; ++i)
    {
        nvmath::vec3 color = nvmath::vec3(distR(rand), distG(rand), distB(rand));
        pointLights.emplace_back(shader::PointLight {
            .pos = nvmath::vec4(distX(rand), distY(rand), distZ(rand), 1.0f),
            .color_luminance =
                nvmath::vec4(color[0],
                             color[1],
                             color[2],
                             0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2]),
        });
    }
    return pointLights;
}

std::vector<shader::TriangleLight> Scene::collectTriangleLights(const nvh::GltfScene& scene)
{
    std::vector<shader::TriangleLight> triangleLights;
    for (const nvh::GltfNode& node : scene.m_nodes)
    {
        const nvh::GltfPrimMesh& mesh     = scene.m_primMeshes[node.primMesh];
        const nvh::GltfMaterial& material = scene.m_materials[mesh.materialIndex];
        if (material.emissiveFactor.sq_norm() > 1e-6)
        {
            const uint32_t*     indices = scene.m_indices.data() + mesh.firstIndex;
            const nvmath::vec3* pos     = scene.m_positions.data() + mesh.vertexOffset;
            for (uint32_t i = 0; i < mesh.indexCount; i += 3, indices += 3)
            {
                nvmath::vec4 p1 = node.worldMatrix * nvmath::vec4(pos[indices[0]], 1.0f);
                nvmath::vec4 p2 = node.worldMatrix * nvmath::vec4(pos[indices[1]], 1.0f);
                nvmath::vec4 p3 = node.worldMatrix * nvmath::vec4(pos[indices[2]], 1.0f);

                nvmath::vec3 p1_vec3(p1.x, p1.y, p1.z);
                nvmath::vec3 p2_vec3(p2.x, p2.y, p2.z);
                nvmath::vec3 p3_vec3(p3.x, p3.y, p3.z);

                vec3  normal = nvmath::cross(p2_vec3 - p1_vec3, p3_vec3 - p1_vec3);
                float area   = normal.norm();
                normal /= area;
                area *= 0.5f;

                float emissionLuminance = 0.2126f * material.emissiveFactor.x +
                                          0.7152f * material.emissiveFactor.y +
                                          0.0722f * material.emissiveFactor.z;

                triangleLights.push_back(shader::TriangleLight {
                    .p1                 = p1,
                    .p2                 = p2,
                    .p3                 = p3,
                    .emission_luminance = nvmath::vec4(material.emissiveFactor, emissionLuminance),
                    .normalArea         = nvmath::vec4(normal, area),
                });
            }
        }
    }
    return triangleLights;
}

std::vector<shader::Bucket>
Scene::createAliasTable(std::vector<shader::PointLight>&    pointLights,
                        std::vector<shader::TriangleLight>& triangleLights)
{
    std::queue<uint32_t> bigger;
    std::queue<uint32_t> smaller;
    std::vector<float>   lightProbabilityVec;
    float                luminanceSum = 0.0f;
    uint32_t             lightNum     = 0;

    if (!pointLights.empty())
    {
        lightNum = static_cast<uint32_t>(pointLights.size());

        for (auto& pointLight : pointLights)
        {
            luminanceSum += pointLight.color_luminance.w;
            lightProbabilityVec.push_back(pointLight.color_luminance.w);
        }
    }
    else
    {
        lightNum = static_cast<uint32_t>(triangleLights.size());

        for (auto& triangleLight : triangleLights)
        {
            float triangleLightLuminance =
                triangleLight.emission_luminance.w * triangleLight.normalArea.w;
            luminanceSum += triangleLightLuminance;
            lightProbabilityVec.push_back(triangleLightLuminance);
        }
    }

    std::vector<shader::Bucket> result(lightNum,
                                       shader::Bucket {
                                           .probability              = 0.0f,
                                           .alias                    = -1,
                                           .originalProbability      = 0.0f,
                                           .aliasOriginalProbability = 0.0f,
                                       });

    for (uint32_t i = 0; i < lightProbabilityVec.size(); ++i)
    {
        result[i].originalProbability = lightProbabilityVec[i] / luminanceSum;
        lightProbabilityVec[i] =
            float(lightProbabilityVec.size()) * lightProbabilityVec[i] / luminanceSum;
        if (lightProbabilityVec[i] >= 1.0f)
        {
            bigger.push(i);
        }
        else
        {
            smaller.push(i);
        }
    }

    while (!bigger.empty() && !smaller.empty())
    {
        uint32_t bigSample = bigger.front();
        bigger.pop();
        uint32_t smallSample = smaller.front();
        smaller.pop();

        result[smallSample].probability = lightProbabilityVec[smallSample];
        result[smallSample].alias       = bigSample;

        lightProbabilityVec[bigSample] = (lightProbabilityVec[bigSample] + lightProbabilityVec[smallSample]) - 1.0f;

        if (lightProbabilityVec[bigSample] < 1.0f)
        {
            smaller.push(bigSample);
        }
        else
        {
            bigger.push(bigSample);
        }
    }

    while (!bigger.empty())
    {
        uint32_t bigSample = bigger.front();
        bigger.pop();
        result[bigSample].probability = 1.0f;
        result[bigSample].alias       = bigSample;
    }

    while (!smaller.empty())
    {
        uint32_t smallSample = smaller.front();
        smaller.pop();
        result[smallSample].probability = 1.0f;
        result[smallSample].alias       = smallSample;
    }

    for (shader::Bucket& col : result)
    {
        col.aliasOriginalProbability = result[col.alias].originalProbability;
    }

    return result;
}
