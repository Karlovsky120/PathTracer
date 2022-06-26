#version 460 core
#extension GL_GOOGLE_include_directive : require

#include "include/structs.glsl"

layout (set = 0, binding = 0) uniform Uniforms
{
	mat4 projectionViewMatrix;
} uniforms;

layout (set = 0, binding = 1) uniform Matrices
{
	ModelMatrices matrices;
};

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in vec4 inColor;
layout (location = 4) in vec2 inUv;

layout (location = 0) out vec3 outPosition;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec4 outTangent;
layout (location = 3) out vec4 outColor;
layout (location = 4) out vec2 outUv;

void main()
{
	vec4 worldPos = matrices.Transform * vec4(inPosition, 1.0f);
	gl_Position = uniforms.projectionViewMatrix * worldPos;

	outPosition = worldPos.xyz;
	outNormal = normalize((matrices.TransformInverseTransposed * vec4(inNormal, 0.0f)).xyz);
	outTangent.xyz = normalize((matrices.Transform * vec4(inTangent.xyz, 0.0f)).xyz);
	outTangent.w = inTangent.w;
	outColor = inColor;
	outUv = inUv;
}
