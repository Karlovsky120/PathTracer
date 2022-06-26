#version 460 core
#extension GL_GOOGLE_include_directive : require

#include "include/structs.glsl"
#include "include/random.glsl"
#include "include/brdf.glsl"

layout (binding = 0) uniform sampler2D uniAlbedo;
layout (binding = 1) uniform sampler2D uniNormal;
layout (binding = 2) uniform sampler2D uniMaterialProperties;
layout (binding = 3) uniform sampler2D uniWorldPosition;

layout (binding = 4) uniform Uniforms
{
	LightingPassUniforms uniforms;
};

layout (binding = 5) buffer Reservoirs
{
	Reservoir reservoirs[];
};

layout (binding = 6) buffer PointLights
{
	int count;
	PointLight lights[];
} pointLights;

layout (binding = 7) buffer TriangleLights
{
	int count;
	TriangleLight lights[];
} triangleLights;

layout (location = 0) in vec2 inUv;

layout (location = 0) out vec3 outColor;

#define PI 3.1415926

void main()
{
	vec4 albedo = texture(uniAlbedo, inUv);
	vec3 normal = texture(uniNormal, inUv).xyz;
	vec2 materialProps = texture(uniMaterialProperties, inUv).xy;
	vec3 worldPos = texture(uniWorldPosition, inUv).xyz;

	uvec2 pixelCoord = uvec2(gl_FragCoord.xy);
	Reservoir reservoir = reservoirs[pixelCoord.y * uniforms.bufferSize.x + pixelCoord.x];
	outColor = vec3(0.0f);
	for (int i = 0; i < RESERVOIR_SIZE; ++i)
	{
		vec3 emission;
		int lightIndex = reservoir.samples[i].lightIndex;
		if (lightIndex < 0)
		{
			emission = triangleLights.lights[-1 - lightIndex].emission_luminance.rgb;
		}
		else
		{
			emission = pointLights.lights[lightIndex].color_luminance.rgb;
		}

		vec3 pHat = evaluatePHatFull(
			worldPos, reservoir.samples[i].position_emissionLum.xyz, uniforms.cameraPos.xyz,
			normal, reservoir.samples[i].normal.xyz, reservoir.samples[i].normal.w > 0.5f,
			albedo.rgb, emission, materialProps.x, materialProps.y
		);
		outColor += pHat * reservoir.samples[i].w;
	}

	outColor /= RESERVOIR_SIZE;
	if (albedo.w > 0.5f)
	{
		outColor = albedo.xyz;
	}

	outColor = pow(outColor, vec3(1.0f / uniforms.gamma));
}
