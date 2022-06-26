#pragma once

#include <nvmath.h>

class Camera
{
public:
    void update();

    nvmath::vec3f Position {1.0f, 1.0f, 1.0f};
    nvmath::vec3f LookAt {0.0f, 0.0f, 0.0f};
    nvmath::vec3f WorldUp {0.0f, 1.0f, 0.0f};

    float FovY        = 0.5f * nv_pi;
    float AspectRatio = 1.0f;

    nvmath::vec3f ForwardVec;
    nvmath::vec3f RightVec;
    nvmath::vec3f UpVec;

    nvmath::mat4f ProjectionViewMatrix;

private:
    const float ZNear = 0.01f;
    const float ZFar  = 1000.0f;

    nvmath::mat4f ViewMatrix;
    nvmath::mat4f ProjectionMatrix;
};
