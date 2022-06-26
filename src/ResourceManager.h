#pragma once

#include <gltfscene.h>
#include "vk_mem_alloc.h"

#include <filesystem>
#include <optional>

class ResourceManager;
class TransientCommandBuffer;

template<typename T, typename Derived>
struct UniqueHandle
{
public:
    UniqueHandle() = default;
    UniqueHandle(Derived&& src)
        : _object(src._object)
        , _allocation(src._allocation)
        , _allocator(src._allocator)
    {

        assert(&src != this);
        src._object    = T();
        src._allocation = nullptr;
        src._allocator = nullptr;
    }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator            =(UniqueHandle&& src)
    {
        assert(&src != this);
        reset();

        _object    = src._object;
        _allocation = src._allocation;
        _allocator = src._allocator;

        src._object    = T();
        src._allocation = nullptr;
        src._allocator = nullptr;

        return *this;
    }

    const T operator*() const
    {
        return _object;
    };

    UniqueHandle& operator=(const UniqueHandle&) = delete;
    ~UniqueHandle()
    {
        reset();
    }

    void* map()
    {
        void* mapped = nullptr;

        if (static_cast<vk::Result>(vmaMapMemory(_getAllocator(), _allocation, &mapped)) !=
            vk::Result::eSuccess)
        {
            std::cout << "Failed to map VMA memory!" << std::endl;
            std::abort();
        }

        return mapped;
    }
    template<typename T>
    T* mapAs()
    {
        return static_cast<T*>(map());
    }

    void unmap()
    {
        vmaUnmapMemory(_getAllocator(), _allocation);
    }

    void flush()
    {
        vmaFlushAllocation(_getAllocator(), _allocation, 0, VK_WHOLE_SIZE);
    }

    void invalidate()
    {
        vmaInvalidateAllocation(_getAllocator(), _allocation, 0, VK_WHOLE_SIZE);
    }

    void reset()
    {
        if (_object)
        {
            static_cast<Derived*>(this)->_release();
            _object    = T();
            _allocation = nullptr;
            _allocator = nullptr;
        }
    }

    VmaAllocation _allocation = nullptr;

protected:
    T                _object;
    ResourceManager* _allocator = nullptr;

    VmaAllocator _getAllocator() const
    {
        return _allocator->_allocator;
    }
};

struct UniqueBuffer : public UniqueHandle<vk::Buffer, UniqueBuffer>
{
    friend ResourceManager;

private:
    using Base = UniqueHandle<vk::Buffer, UniqueBuffer>;
    friend Base;

public:
    UniqueBuffer() = default;
    UniqueBuffer(UniqueBuffer&& src)
        : Base(std::move(src))
    {}
    UniqueBuffer& operator=(UniqueBuffer&& src)
    {
        Base::operator=(std::move(src));
        return *this;
    }

    vk::DeviceMemory getMemory();
    vk::DeviceSize getOffset();

private:
    void _release()
    {
        vmaDestroyBuffer(_getAllocator(), _object, _allocation);
    }
};

struct UniqueImage : public UniqueHandle<vk::Image, UniqueImage>
{
    friend ResourceManager;

private:
    using Base = UniqueHandle<vk::Image, UniqueImage>;
    friend Base;

public:
    UniqueImage() = default;
    UniqueImage(UniqueImage&& src)
        : Base(std::move(src))
    {}
    UniqueImage& operator=(UniqueImage&& src)
    {
        Base::operator=(std::move(src));
        return *this;
    }

private:
    void _release()
    {
        vmaDestroyImage(_getAllocator(), _object, _allocation);
    }
};

class ResourceManager
{
    template<typename, typename>
    friend struct UniqueHandle;

public:
    ResourceManager() = default;
    ResourceManager(uint32_t vulkanApiVersion, vk::Instance, vk::PhysicalDevice, vk::Device);
    ResourceManager(ResourceManager&& src)
        : _allocator(src._allocator)
    {
        assert(&src != this);
        src._allocator = nullptr;
    }
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator               =(ResourceManager&& src)
    {
        assert(&src != this);
        reset();
        _allocator     = src._allocator;
        src._allocator = nullptr;
        return *this;
    }
    ResourceManager& operator=(const ResourceManager&) = delete;
    ~ResourceManager()
    {
        reset();
    }

    UniqueBuffer createBuffer(const vk::BufferCreateInfo&, const VmaAllocationCreateInfo&);

    UniqueBuffer
    createBuffer(uint32_t size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    UniqueImage createImage(const vk::ImageCreateInfo&     createImageInfoIn,
                            const VmaAllocationCreateInfo& allocationInfo);

    UniqueImage createImage2D(vk::Extent2D            size,
                              vk::Format              format,
                              vk::ImageUsageFlags     usage,
                              VmaMemoryUsage          memoryUsage   = VMA_MEMORY_USAGE_GPU_ONLY,
                              vk::ImageTiling         tiling        = vk::ImageTiling::eOptimal,
                              vk::ImageLayout         initialLayout = vk::ImageLayout::eUndefined,
                              uint32_t                mipLevels     = 1,
                              vk::SampleCountFlagBits sampleCount   = vk::SampleCountFlagBits::e1,
                              uint32_t                arrayLayers   = 1);

    vk::UniqueImageView createImageView2D(vk::Device           device,
                                          vk::Image            image,
                                          vk::Format           format,
                                          vk::ImageAspectFlags aspect,
                                          uint32_t             baseMipLevel    = 0,
                                          uint32_t             mipLevelCount   = 1,
                                          uint32_t             baseArrayLayer  = 0,
                                          uint32_t             arrayLayerCount = 1);

    vk::UniqueSampler
    createSampler(vk::Device                   device,
                  vk::Filter                   magFilter   = vk::Filter::eLinear,
                  vk::Filter                   minFilter   = vk::Filter::eLinear,
                  vk::SamplerMipmapMode        mipmapMode  = vk::SamplerMipmapMode::eLinear,
                  std::optional<float>         anisotropy  = std::nullopt,
                  float                        minMipLod   = 0.0f,
                  float                        maxMipLod   = std::numeric_limits<float>::max(),
                  float                        mipLodBias  = 0.0f,
                  vk::SamplerAddressMode       addressMode = vk::SamplerAddressMode::eRepeat,
                  std::optional<vk::CompareOp> compare     = std::nullopt,
                  bool                         unnormalizedCoordinates = false);

    template<typename T>
    UniqueBuffer createTypedBuffer(std::size_t                  numElements,
                                   vk::BufferUsageFlags         usage,
                                   VmaMemoryUsage               memoryUsage,
                                   const std::vector<uint32_t>* sharedQueues = nullptr)
    {
        vk::BufferCreateInfo bufferInfo;
        bufferInfo.setSize(static_cast<uint32_t>(sizeof(T) * numElements)).setUsage(usage);
        if (sharedQueues)
        {
            bufferInfo.setSharingMode(vk::SharingMode::eConcurrent)
                .setQueueFamilyIndices(*sharedQueues);
        }
        else
        {
            bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
        }

        VmaAllocationCreateInfo allocationInfo {};
        allocationInfo.usage = memoryUsage;

        return createBuffer(bufferInfo, allocationInfo);
    }

    static void transitionImageLayout(vk::CommandBuffer commandBuffer,
                                      vk::Image         image,
                                      vk::Format        format,
                                      vk::ImageLayout   oldImageLayout,
                                      vk::ImageLayout   newImageLayout,
                                      uint32_t          baseMipLevel = 0,
                                      uint32_t          numMipLevels = 1);

    static UniqueImage loadTexture(const unsigned char*    data,
                                   uint32_t                width,
                                   uint32_t                height,
                                   vk::Format              format,
                                   uint32_t                mipLevels,
                                   ResourceManager&        allocator,
                                   TransientCommandBuffer& transientCommandBuffer);

    static UniqueImage loadTexture(const tinygltf::Image&  gltfImage,
                                   vk::Format              format,
                                   uint32_t                mipLevels,
                                   ResourceManager&        allocator,
                                   TransientCommandBuffer& transientCommandBuffer);

    static std::vector<char> readFile(const std::filesystem::path& path);

private:
    void reset();

    VmaAllocator _allocator = nullptr;
};
