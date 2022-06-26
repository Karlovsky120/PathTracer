#version 460 core
#extension GL_GOOGLE_include_directive : require
#extension GL_ARB_separate_shader_objects: enable
#extension GL_EXT_scalar_block_layout : enable

#include "include/structs.glsl"

layout (set = 0, binding = 2) uniform Material
{
	MaterialUniforms material;
};

layout (set = 1, binding = 0) uniform sampler2D albedoTexture;
layout (set = 1, binding = 1) uniform sampler2D normalTexture;
layout (set = 1, binding = 2) uniform sampler2D materialTexture;
layout (set = 1, binding = 3) uniform sampler2D emissiveTexture;

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inTangent;
layout (location = 3) in vec4 inColor;
layout (location = 4) in vec2 inUv;

layout (location = 0) out vec4 outAlbedo;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outMaterialProperties;
layout (location = 3) out vec3 outWorldPosition;

void main()
{
	vec4 albedo = texture(albedoTexture, inUv) * material.colorParam;
	if (material.alphaMode == 1)
	{
		if (albedo.a < material.alphaCutoff)
		{
			discard;
		}
	}

	outAlbedo.rgb = albedo.rgb;

	vec3 bitangent = cross(inNormal, inTangent.xyz) * inTangent.w;
	vec3 normalTex = texture(normalTexture, inUv * material.normalTextureScale).xyz * 2.0f - 1.0f;
	outNormal = normalize(normalTex.x * inTangent.xyz + normalTex.y * bitangent + normalTex.z * inNormal);

	vec4 materialProp = texture(materialTexture, inUv) * material.materialParam;
	float roughness = 0.0f;
	float metallic = 0.0f;
	if (material.shadingModel == METALLIC_ROUGHNESS)
	{
		roughness = materialProp.y;
		metallic = materialProp.z;
	}
	else if (material.shadingModel == SPECULAR_GLOSSINESS)
	{
		roughness = 1.0f - materialProp.a;

		vec3 average = 0.5f * (albedo.rgb + materialProp.rgb);
		vec3 sqrtTerm = sqrt(average * average - 0.04f * albedo.rgb);
		vec3 metallicRgb = 25.0f * average - sqrtTerm;

		metallic = (metallicRgb.r + metallicRgb.g + metallicRgb.b) / 3.0f;
		outAlbedo.rgb = average + sqrtTerm;
	}

	outMaterialProperties = vec2(roughness, metallic);
	outWorldPosition = inPosition;

	outAlbedo.w = 0.0;
	if (length(material.emissiveFactor.xyz) > 0.0)
	{
		outAlbedo.xyz = material.colorParam.rgb * material.emissiveFactor.xyz * texture(emissiveTexture, inUv).rgb;
		outAlbedo.w = 1.0;
	}
}
