#include "TransientCommandBuffer.h"

TransientCommandBuffer::TransientCommandBuffer(vk::Device device,
                                               vk::Queue  queue,
                                               uint32_t   queueIndex)
    : _device(device)
    , _queue(queue)
{
    _pool = _device.createCommandPoolUnique(
        {.flags = vk::CommandPoolCreateFlagBits::eTransient, .queueFamilyIndex = queueIndex});
}

void TransientCommandBuffer::begin()
{
    _buffer =
        std::move(_device.allocateCommandBuffersUnique({.commandPool = *_pool,
                                                        .level = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = 1})[0]);

    _fence = _device.createFenceUnique({});
    _buffer->begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}

void TransientCommandBuffer::submitAndWait()
{
    _buffer->end();

    std::array<vk::CommandBuffer, 1> buffers {*_buffer};
    vk::SubmitInfo                   submitInfo;
    submitInfo.setCommandBuffers(buffers);
    _queue.submit(submitInfo, *_fence);
    while (_device.waitForFences(*_fence, true, std::numeric_limits<uint64_t>::max()) ==
           vk::Result::eTimeout)
    {}

    _buffer.reset();
    _fence.reset();
}

vk::CommandBuffer TransientCommandBuffer::operator*() const
{
    return *_buffer;
}

const vk::CommandBuffer* TransientCommandBuffer::operator->() const
{
    return &*_buffer;
}