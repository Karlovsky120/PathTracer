#include "Camera.h"

void Camera::update()
{
    ForwardVec = nvmath::normalize(LookAt - Position);
    RightVec   = nvmath::normalize(nvmath::cross(ForwardVec, WorldUp));
    UpVec      = nvmath::cross(RightVec, ForwardVec);

    nvmath::mat3f rotation;
    rotation.set_row(0, RightVec);
    rotation.set_row(1, -UpVec);
    rotation.set_row(2, ForwardVec);
    nvmath::vec3f offset = -(rotation * Position);

    ViewMatrix.set_row(0, nvmath::vec4f(rotation.row(0), offset.x));
    ViewMatrix.set_row(1, nvmath::vec4f(rotation.row(1), offset.y));
    ViewMatrix.set_row(2, nvmath::vec4f(rotation.row(2), offset.z));
    ViewMatrix.set_row(3, nvmath::vec4f_w);

    float f = 1.0f / std::tan(0.5f * FovY);

    ProjectionMatrix = nvmath::mat4f_zero;
    ProjectionMatrix.set_row(0, nvmath::vec4f(f / AspectRatio, 0.0f, 0.0f, 0.0f));
    ProjectionMatrix.set_row(1, nvmath::vec4f(0.0f, f, 0.0f, 0.0f));
    ProjectionMatrix.set_row(
        2,
        nvmath::vec4f(0.0f, 0.0f, -ZFar / (ZNear - ZFar), ZNear * ZFar / (ZNear - ZFar)));
    ProjectionMatrix.set_row(3, nvmath::vec4f(0.0f, 0.0f, 1.0f, 0.0f));

    ProjectionViewMatrix = ProjectionMatrix * ViewMatrix;
}
