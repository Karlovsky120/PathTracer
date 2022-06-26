#pragma once

class TransientCommandBuffer
{
public:
    TransientCommandBuffer() = default;
    TransientCommandBuffer(vk::Device device, vk::Queue queue, uint32_t queueIndex);

    void begin();
    void submitAndWait();

    vk::CommandBuffer operator*() const;
    const vk::CommandBuffer* operator->() const;

private:
    vk::UniqueCommandBuffer _buffer;
    vk::Device             _device;
    vk::UniqueFence        _fence;
    vk::Queue              _queue;
    vk::UniqueCommandPool  _pool;
};
