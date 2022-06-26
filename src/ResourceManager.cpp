#define VMA_IMPLEMENTATION
#define VMA_EXTERNAL_MEMORY 1
#include <vk_mem_alloc.h>

#include "ResourceManager.h"
#include "TransientCommandBuffer.h"

#include <fstream>

vk::DeviceMemory UniqueBuffer::getMemory()
{
    return _allocation->GetMemory();
}

vk::DeviceSize UniqueBuffer::getOffset()
{
    return _allocation->GetOffset();
}

ResourceManager::ResourceManager(uint32_t           version,
                                 vk::Instance       instance,
                                 vk::PhysicalDevice physicalDevice,
                                 vk::Device         device)
{
    VmaAllocatorCreateInfo allocatorInfo {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                 VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
                 VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT,
        .physicalDevice   = physicalDevice,
        .device           = device,
        .instance         = instance,
        .vulkanApiVersion = version,
    };

    std::vector<VkExternalMemoryHandleTypeFlags> handleTypes(
        physicalDevice.getMemoryProperties().memoryTypes.size(),
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);
    allocatorInfo.pTypeExternalMemoryHandleTypes = handleTypes.data();

    /*for (handleType)

        for (uint32_t i = 0; i < physicalDevice.getMemoryProperties().memoryTypes.size(); ++i) {}*/

    if (static_cast<vk::Result>(vmaCreateAllocator(&allocatorInfo, &_allocator)) !=
        vk::Result::eSuccess)
    {
        std::cout << "Failed to create allocator!" << std::endl;
        std::abort();
    }
}

UniqueBuffer ResourceManager::createBuffer(const vk::BufferCreateInfo&    createBufferInfoIn,
                                           const VmaAllocationCreateInfo& allocationInfo)
{
    vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> createBufferInfo {
        createBufferInfoIn,
        {.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32}};

    auto bufferInfo = static_cast<VkBufferCreateInfo>(createBufferInfo.get<vk::BufferCreateInfo>());
    UniqueBuffer result;
    VkBuffer     buffer;
    if (static_cast<vk::Result>(vmaCreateBuffer(_allocator,
                                                &bufferInfo,
                                                &allocationInfo,
                                                &buffer,
                                                &result._allocation,
                                                nullptr)) != vk::Result::eSuccess)
    {
        std::cout << "Failed to create VMA buffer!" << std::endl;
        std::abort();
    }

    result._object    = buffer;
    result._allocator = this;
    return result;
}

UniqueBuffer
ResourceManager::createBuffer(uint32_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    return createBuffer(
        {
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        {
            .usage = memoryUsage,
        });
}

UniqueImage ResourceManager::createImage(const vk::ImageCreateInfo&     createImageInfoIn,
                                         const VmaAllocationCreateInfo& allocationInfo)
{
    vk::StructureChain<vk::ImageCreateInfo, vk::ExternalMemoryImageCreateInfo> createImageInfo {
        createImageInfoIn,
        {.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32}};

    auto imageInfo = static_cast<VkImageCreateInfo>(createImageInfo.get<vk::ImageCreateInfo>());
    UniqueImage result;
    VkImage     image;

    if (static_cast<vk::Result>(vmaCreateImage(_allocator,
                                               &imageInfo,
                                               &allocationInfo,
                                               &image,
                                               &result._allocation,
                                               nullptr)) != vk::Result::eSuccess)
    {
        std::cout << "Failed to create VMA image!" << std::endl;
        std::abort();
    }

    result._object    = image;
    result._allocator = this;
    return result;
}

UniqueImage ResourceManager::createImage2D(vk::Extent2D            size,
                                           vk::Format              format,
                                           vk::ImageUsageFlags     usage,
                                           VmaMemoryUsage          memoryUsage,
                                           vk::ImageTiling         tiling,
                                           vk::ImageLayout         initialLayout,
                                           uint32_t                mipLevels,
                                           vk::SampleCountFlagBits sampleCount,
                                           uint32_t                arrayLayers)
{
    return createImage(
        {
            .imageType     = vk::ImageType::e2D,
            .format        = format,
            .extent        = {.width = size.width, .height = size.height, .depth = 1},
            .mipLevels     = mipLevels,
            .arrayLayers   = arrayLayers,
            .samples       = sampleCount,
            .tiling        = tiling,
            .usage         = usage,
            .sharingMode   = vk::SharingMode::eExclusive,
            .initialLayout = initialLayout,
    },
        {.usage = memoryUsage});
}
vk::UniqueImageView ResourceManager::createImageView2D(vk::Device           device,
                                                       vk::Image            image,
                                                       vk::Format           format,
                                                       vk::ImageAspectFlags aspect,
                                                       uint32_t             baseMipLevel,
                                                       uint32_t             mipLevelCount,
                                                       uint32_t             baseArrayLayer,
                                                       uint32_t             arrayLayerCount)
{
    return device.createImageViewUnique({
        .image    = image,
        .viewType = vk::ImageViewType::e2D,
        .format   = format,
        .subresourceRange =
            {
                               .aspectMask     = aspect,
                               .baseMipLevel   = baseMipLevel,
                               .levelCount     = mipLevelCount,
                               .baseArrayLayer = baseArrayLayer,
                               .layerCount     = arrayLayerCount,
                               },
    });
}

vk::UniqueSampler ResourceManager::createSampler(vk::Device                   device,
                                                 vk::Filter                   magFilter,
                                                 vk::Filter                   minFilter,
                                                 vk::SamplerMipmapMode        mipmapMode,
                                                 std::optional<float>         anisotropy,
                                                 float                        minMipLod,
                                                 float                        maxMipLod,
                                                 float                        mipLodBias,
                                                 vk::SamplerAddressMode       addressMode,
                                                 std::optional<vk::CompareOp> compare,
                                                 bool unnormalizedCoordinates)
{
    vk::SamplerCreateInfo samplerInfo {
        .magFilter               = magFilter,
        .minFilter               = minFilter,
        .mipmapMode              = mipmapMode,
        .addressModeU            = addressMode,
        .addressModeV            = addressMode,
        .addressModeW            = addressMode,
        .mipLodBias              = mipLodBias,
        .anisotropyEnable        = anisotropy.has_value(),
        .compareEnable           = compare.has_value(),
        .minLod                  = minMipLod,
        .maxLod                  = maxMipLod,
        .unnormalizedCoordinates = unnormalizedCoordinates,
    };

    if (anisotropy)
    {
        samplerInfo.setMaxAnisotropy(anisotropy.value());
    }
    if (compare)
    {
        samplerInfo.setCompareOp(compare.value());
    }

    return device.createSamplerUnique(samplerInfo);
}

void ResourceManager::transitionImageLayout(vk::CommandBuffer commandBuffer,
                                            vk::Image         image,
                                            vk::Format        format,
                                            vk::ImageLayout   oldImageLayout,
                                            vk::ImageLayout   newImageLayout,
                                            uint32_t          baseMipLevel,
                                            uint32_t          numMipLevels)
{
    vk::AccessFlags sourceAccessMask;
    switch (oldImageLayout)
    {
        case vk::ImageLayout::eTransferSrcOptimal:
            sourceAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            sourceAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::ePreinitialized:
            sourceAccessMask = vk::AccessFlagBits::eHostWrite;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            sourceAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eGeneral:
            [[fallthrough]];
        case vk::ImageLayout::eUndefined:
            break;
        default:
            assert(false);
            break;
    }

    vk::PipelineStageFlags sourceStage;
    switch (oldImageLayout)
    {
        case vk::ImageLayout::eGeneral:
            [[fallthrough]];
        case vk::ImageLayout::ePreinitialized:
            sourceStage = vk::PipelineStageFlagBits::eHost;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            [[fallthrough]];
        case vk::ImageLayout::eTransferDstOptimal:
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            break;
        case vk::ImageLayout::eUndefined:
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            break;
        default:
            assert(false);
            break;
    }

    vk::AccessFlags destinationAccessMask;
    switch (newImageLayout)
    {
        case vk::ImageLayout::eColorAttachmentOptimal:
            destinationAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            destinationAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                    vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            destinationAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            destinationAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            destinationAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eGeneral:
            [[fallthrough]];
        case vk::ImageLayout::ePresentSrcKHR:
            break;
        default:
            assert(false);
            break;
    }

    vk::PipelineStageFlags destinationStage;
    switch (newImageLayout)
    {
        case vk::ImageLayout::eColorAttachmentOptimal:
            destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
            break;
        case vk::ImageLayout::eGeneral:
            destinationStage = vk::PipelineStageFlagBits::eHost;
            break;
        case vk::ImageLayout::ePresentSrcKHR:
            destinationStage = vk::PipelineStageFlagBits::eBottomOfPipe;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            [[fallthrough]];
        case vk::ImageLayout::eTransferSrcOptimal:
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
            break;
        default:
            assert(false);
            break;
    }

    vk::ImageAspectFlags aspectMask;
    if (format == vk::Format::eD32Sfloat || format == vk::Format::eD16Unorm)
    {
        aspectMask = vk::ImageAspectFlagBits::eDepth;
    }
    else if (format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint)
    {
        aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    }
    else
    {
        aspectMask = vk::ImageAspectFlagBits::eColor;
    }

    vk::ImageMemoryBarrier imageMemoryBarrier {
        .srcAccessMask       = sourceAccessMask,
        .dstAccessMask       = destinationAccessMask,
        .oldLayout           = oldImageLayout,
        .newLayout           = newImageLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange =
            {
                               .aspectMask     = aspectMask,
                               .baseMipLevel   = baseMipLevel,
                               .levelCount     = numMipLevels,
                               .baseArrayLayer = 0,
                               .layerCount     = 1,
                               },
    };

    commandBuffer
        .pipelineBarrier(sourceStage, destinationStage, {}, nullptr, nullptr, imageMemoryBarrier);
}

UniqueImage ResourceManager::loadTexture(const unsigned char*    data,
                                         uint32_t                width,
                                         uint32_t                height,
                                         vk::Format              format,
                                         uint32_t                mipLevels,
                                         ResourceManager&        allocator,
                                         TransientCommandBuffer& transientCommandBuffer)
{
    UniqueBuffer buffer =
        allocator.createTypedBuffer<unsigned char>(width * height * 4,
                                                   vk::BufferUsageFlagBits::eTransferSrc,
                                                   VMA_MEMORY_USAGE_CPU_TO_GPU);

    void* bufferData = buffer.map();
    std::memcpy(bufferData, data, sizeof(unsigned char) * width * height * 4);
    buffer.unmap();
    buffer.flush();

    vk::ImageUsageFlags usageFlags =
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    if (mipLevels > 1)
    {
        usageFlags |= vk::ImageUsageFlagBits::eTransferSrc;
    }
    UniqueImage image = allocator.createImage2D({.width = width, .height = height},
                                                format,
                                                usageFlags,
                                                VMA_MEMORY_USAGE_GPU_ONLY,
                                                vk::ImageTiling::eOptimal,
                                                vk::ImageLayout::eUndefined,
                                                mipLevels);

    transientCommandBuffer.begin();

    transitionImageLayout(*transientCommandBuffer,
                          *image,
                          format,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal,
                          0,
                          mipLevels);

    vk::BufferImageCopy bufferImageCopy {
        .imageSubresource = {.aspectMask     = vk::ImageAspectFlagBits::eColor,
                             .mipLevel       = 0,
                             .baseArrayLayer = 0,
                             .layerCount     = 1},
        .imageExtent      = {
                             .width  = width,
                             .height = height,
                             .depth  = 1,
                             }
    };

    transientCommandBuffer->copyBufferToImage(*buffer,
                                              *image,
                                              vk::ImageLayout::eTransferDstOptimal,
                                              bufferImageCopy);

    if (mipLevels > 1)
    {
        int32_t mipWidth  = width;
        int32_t mipHeight = height;
        for (uint32_t i = 1; i < mipLevels; ++i)
        {
            int32_t nextWidth  = std::max<uint32_t>(mipWidth / 2, 1);
            int32_t nextHeight = std::max<uint32_t>(mipHeight / 2, 1);

            transitionImageLayout(*transientCommandBuffer,
                                  *image,
                                  format,
                                  vk::ImageLayout::eTransferDstOptimal,
                                  vk::ImageLayout::eTransferSrcOptimal,
                                  i - 1,
                                  1);

            vk::ImageSubresourceLayers a {
                .aspectMask     = vk::ImageAspectFlagBits::eColor,
                .mipLevel       = i - 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            };

            vk::ImageBlit blit {
                .srcSubresource = {.aspectMask     = vk::ImageAspectFlagBits::eColor,
                                   .mipLevel       = i - 1,
                                   .baseArrayLayer = 0,
                                   .layerCount     = 1},
                .srcOffsets     = vk::ArrayWrapper1D<vk::Offset3D, 2>({{
                    {.x = 0, .y = 0, .z = 0},
                    {.x = mipWidth, .y = mipHeight, .z = 1},
                }                              }
                  ),
                .dstSubresource = {.aspectMask     = vk::ImageAspectFlagBits::eColor,
                                   .mipLevel       = i,
                                   .baseArrayLayer = 0,
                                   .layerCount     = 1},
                .dstOffsets     = vk::ArrayWrapper1D<vk::Offset3D, 2>({{
                    {.x = 0, .y = 0, .z = 0},
                    {.x = nextWidth, .y = nextHeight, .z = 1},
                }                  }
                  )
            };

            transientCommandBuffer->blitImage(*image,
                                              vk::ImageLayout::eTransferSrcOptimal,
                                              *image,
                                              vk::ImageLayout::eTransferDstOptimal,
                                              blit,
                                              vk::Filter::eLinear);

            mipWidth  = nextWidth;
            mipHeight = nextHeight;
        }

        transitionImageLayout(*transientCommandBuffer,
                              *image,
                              format,
                              vk::ImageLayout::eTransferSrcOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal,
                              0,
                              mipLevels - 1);
        transitionImageLayout(*transientCommandBuffer,
                              *image,
                              format,
                              vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal,
                              mipLevels - 1,
                              1);
    }
    else
    {
        transitionImageLayout(*transientCommandBuffer,
                              *image,
                              format,
                              vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal,
                              0,
                              mipLevels);
    }

    transientCommandBuffer.submitAndWait();

    return image;
}

UniqueImage ResourceManager::loadTexture(const tinygltf::Image&  gltfImage,
                                         vk::Format              format,
                                         uint32_t                mipLevels,
                                         ResourceManager&        allocator,
                                         TransientCommandBuffer& transientCommandBuffer)
{
    return loadTexture(gltfImage.image.data(),
                       gltfImage.width,
                       gltfImage.height,
                       format,
                       mipLevels,
                       allocator,
                       transientCommandBuffer);
}

std::vector<char> ResourceManager::readFile(const std::filesystem::path& path)
{
    std::ifstream     fs(path, std::ios::ate | std::ios::binary);
    std::streampos    size = fs.tellg();
    std::vector<char> result(static_cast<std::size_t>(size));
    fs.seekg(0);
    fs.read(result.data(), size);
    return result;
}

void ResourceManager::reset()
{
    if (_allocator)
    {
        vmaDestroyAllocator(_allocator);
        _allocator = nullptr;
    }
}
